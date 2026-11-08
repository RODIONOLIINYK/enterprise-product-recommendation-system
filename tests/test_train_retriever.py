"""Unit tests for Stage 1 Modular Two-Tower Retriever Training Pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from recommender.sequences import ENRICHED_FEATURE_DIM, HISTORY_AGE_INDEX, build_customer_sequence
from recommender.train_retriever import (
    RetrieverDataset,
    build_retriever_product_inputs,
    build_retriever_transitions,
    compute_discovery_loss,
    compute_product_demand_features,
    compute_retrieval_metrics,
    train_retriever,
)
from recommender.text import CATALOGUE_CACHE_WITHOUT_SUBCATEGORY, generate_company_embeddings, load_cached_embeddings


class MockSentenceTransformer:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.call_count = 0

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.call_count += 1
        num_texts = len(texts)
        vectors = np.zeros((num_texts, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            val = float(len(t) % 100) / 100.0
            vectors[i, :] = val
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class TestTrainRetriever(unittest.TestCase):
    def test_unseen_candidates_use_the_full_history_before_the_target_basket(self):
        purchases = pd.DataFrame(
            {
                "customer_id": ["C1"] * 5,
                "purchase_date": pd.to_datetime(
                    ["2024-12-31", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"]
                ),
                "product_id": ["D", "A", "A", "B", "C"],
                "quantity": [1.0] * 5,
            }
        )
        transition = {
            "customer_id": "C1",
            "as_of_date": pd.Timestamp("2025-01-02"),
            "positive_product_ids": ["A", "B"],
        }
        dataset = RetrieverDataset(
            [transition], purchases, {"A": 0, "B": 1, "C": 2, "D": 3}, {}, {},
            vocabularies={"product_vocab": {"A": 1, "B": 2, "C": 3, "D": 4}},
            include_unseen_candidates=True,
            max_seq_len=1,
        )
        example = dataset[0]
        self.assertEqual(example["target"].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(example["unseen_candidates"].tolist(), [False, True, True, False])
        self.assertEqual(example["seq_product_ids"][example["attention_mask"].bool()].tolist(), [1])

    def test_discovery_loss_ignores_seen_skus_and_repeat_only_baskets(self):
        logits = torch.zeros((2, 3), requires_grad=True)
        targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        unseen = torch.tensor([[False, True, True], [False, True, True]])
        loss = compute_discovery_loss(logits, targets, unseen, torch.tensor([20.0]))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0.0, -5.0, 0.25], [0.0, 0.0, 0.0]]))

    def test_unspecified_container_has_its_own_one_hot_slot_and_embedding_id(self):
        catalogue = pd.DataFrame(
            {"product_id": ["A", "B"], "product_category": ["X", "X"], "container_type": ["БОЧКА", None]},
        )
        sales = pd.DataFrame(
            {"customer_id": ["C1", "C2"], "product_id": ["A", "B"],
             "purchase_date": pd.to_datetime(["2025-01-01", "2025-01-01"]), "quantity": [1, 2]},
        )
        args = (sales, catalogue, {}, ["A", "B"], {"container_vocab": {"БОЧКА": 1}})
        one_hot = build_retriever_product_inputs(*args, include_subcategory=False, container_encoding="one_hot_none")
        embedded = build_retriever_product_inputs(*args, include_subcategory=False, container_encoding="embedding_none")
        self.assertEqual(one_hot["demand_features"][:, -2:].tolist(), [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(embedded["container_id"].tolist(), [1, 0])
        self.assertEqual(embedded["demand_features"].shape, (2, 6))

    def test_build_retriever_transitions_temporal_bounds(self):
        """Transitions must strictly observe chronological boundaries."""
        records = [
            # Cust 1: 4 distinct dates
            {"customer_id": "C1", "purchase_date": "2024-05-01", "product_id": "P1", "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2025-06-01", "product_id": "P2", "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2025-11-01", "product_id": "P1", "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2026-03-01", "product_id": "P3", "product_category": "CAT2"},
            # Cust 2: 2 distinct dates (both train)
            {"customer_id": "C2", "purchase_date": "2024-02-01", "product_id": "P1", "product_category": "CAT1"},
            {"customer_id": "C2", "purchase_date": "2024-08-01", "product_id": "P2", "product_category": "CAT1"},
            # Cust 3: 1 date only (must be skipped because no next basket possible)
            {"customer_id": "C3", "purchase_date": "2026-03-01", "product_id": "P1", "product_category": "CAT1"},
        ]
        df = pd.DataFrame(records)

        train_t, val_t, test_t = build_retriever_transitions(
            df,
            train_cutoff=pd.Timestamp("2025-10-01"),
            val_cutoff=pd.Timestamp("2026-02-19"),
            min_baskets=2,
        )

        # C1 has target dates:
        # - 2025-06-01 (train)
        # - 2025-11-01 (val)
        # - 2026-03-01 (test)
        # C2 has target date:
        # - 2024-08-01 (train)
        # C3 is skipped
        self.assertEqual(len(train_t), 2)
        self.assertEqual(len(val_t), 1)
        self.assertEqual(len(test_t), 1)

        # Validate date constraints
        for t in train_t:
            self.assertTrue(pd.Timestamp(t["as_of_date"]) < pd.Timestamp("2025-10-01"))
        for t in val_t:
            as_of = pd.Timestamp(t["as_of_date"])
            self.assertTrue(pd.Timestamp("2025-10-01") <= as_of < pd.Timestamp("2026-02-19"))
        for t in test_t:
            self.assertTrue(pd.Timestamp(t["as_of_date"]) >= pd.Timestamp("2026-02-19"))

    def test_eval_transitions_use_full_prior_history(self):
        """Validation transitions must include full prior history, including train-period events."""
        records = [
            {"customer_id": "C1", "purchase_date": "2024-06-01", "product_id": "P1", "quantity": 5.0},
            {"customer_id": "C1", "purchase_date": "2025-02-01", "product_id": "P2", "quantity": 10.0},
            {"customer_id": "C1", "purchase_date": "2025-11-15", "product_id": "P3", "quantity": 2.0},  # Val basket
        ]
        df = pd.DataFrame(records)

        # Sequence as of validation date 2025-11-15
        seq_res = build_customer_sequence(
            df,
            customer_id="C1",
            as_of_date=pd.Timestamp("2025-11-15"),
            max_len=32,
        )

        mask = seq_res["attention_mask"]
        pids = seq_res["product_ids"]

        # Both train-period purchases (2024-06-01 P1 and 2025-02-01 P2) must be present
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual(pids[0], "P1")
        self.assertEqual(pids[1], "P2")

    def test_retrieval_metrics_exact_values(self):
        """Validate exact Recall@K, NDCG@10, and MRR calculations against hand-derived values."""
        # 2 samples, 5 products
        # Sample 0:
        # Scores: [0.1, 0.9, 0.4, 0.8, 0.2] -> Order: [1, 3, 2, 4, 0]
        # Targets: {1, 2} (num_pos = 2)
        # Hits:
        # Top 1: {1} (1 hit)
        # Top 2: {1, 3} (1 hit)
        # Top 3: {1, 3, 2} (2 hits)
        # DCG@10: Rank 1 (item 1) -> 1/log2(2)=1.0; Rank 3 (item 2) -> 1/log2(4)=0.5 -> DCG = 1.5
        # IDCG@10: 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        # NDCG@10 = 1.5 / 1.63092975 = 0.919720788
        # MRR: First hit at rank 1 -> 1.0

        # Sample 1:
        # Scores: [0.8, 0.1, 0.2, 0.3, 0.7] -> Order: [0, 4, 3, 2, 1]
        # Targets: {3} (num_pos = 1)
        # Hits:
        # Top 1: {0} (0 hits)
        # Top 2: {0, 4} (0 hits)
        # Top 3: {0, 4, 3} (1 hit)
        # DCG@10: Rank 3 (item 3) -> 1/log2(4)=0.5
        # IDCG@10: 1/log2(2)=1.0
        # NDCG@10 = 0.5 / 1.0 = 0.5
        # MRR: First hit at rank 3 -> 1/3 = 0.333333333

        preds = np.array([
            [0.1, 0.9, 0.4, 0.8, 0.2],
            [0.8, 0.1, 0.2, 0.3, 0.7],
        ], dtype=np.float32)

        targets = np.array([
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        metrics = compute_retrieval_metrics(preds, targets, k_list=(1, 2, 3, 10))

        # Recall@1: (1/2 + 0/1)/2 = 0.25
        self.assertAlmostEqual(metrics["recall@1"], 0.25, places=5)
        # Recall@2: (1/2 + 0/1)/2 = 0.25
        self.assertAlmostEqual(metrics["recall@2"], 0.25, places=5)
        # Recall@3: (2/2 + 1/1)/2 = 1.0
        self.assertAlmostEqual(metrics["recall@3"], 1.0, places=5)
        # Recall@10: (2/2 + 1/1)/2 = 1.0
        self.assertAlmostEqual(metrics["recall@10"], 1.0, places=5)

        # NDCG@10: (0.919720788 + 0.5) / 2 = 0.709860394
        expected_ndcg = (1.5 / (1.0 + 1.0 / math.log2(3)) + 0.5) / 2.0
        self.assertAlmostEqual(metrics["ndcg@10"], expected_ndcg, places=5)

        # MRR: (1.0 + 1.0/3.0)/2 = 0.666666667
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 1.0 / 3.0) / 2.0, places=5)

    def test_product_demand_features_point_in_time(self):
        """Demand features must be strictly point-in-time and normalized to [0, 1]."""
        records = [
            # P1: 2 events before cutoff 2025-10-01, 1 event after
            {"customer_id": "C1", "purchase_date": "2024-05-01", "product_id": "P1", "quantity": 10.0, "product_category": "C"},
            {"customer_id": "C2", "purchase_date": "2025-01-01", "product_id": "P1", "quantity": 20.0, "product_category": "C"},
            {"customer_id": "C1", "purchase_date": "2025-10-15", "product_id": "P1", "quantity": 100.0, "product_category": "C"},
            # P2: only after cutoff 2025-10-01
            {"customer_id": "C1", "purchase_date": "2025-11-01", "product_id": "P2", "quantity": 50.0, "product_category": "C"},
        ]
        df = pd.DataFrame(records)
        eligible_pids = ["P1", "P2"]

        features = compute_product_demand_features(
            purchases_df=df,
            as_of_date=pd.Timestamp("2025-10-01"),
            eligible_product_ids=eligible_pids,
            product_categories={"P1": "C", "P2": "C"},
        )

        self.assertEqual(features.shape, (2, 6))
        # Values must be strictly in [0.0, 1.0]
        self.assertTrue((features >= 0.0).all())
        self.assertTrue((features <= 1.0).all())

        # P1 had sales prior to 2025-10-01 (volume > 0)
        self.assertGreater(features[0, 0].item(), 0.0)

        # P2 had ZERO sales prior to 2025-10-01 (volume == 0, buyers == 0, recency == 1.0)
        self.assertEqual(features[1, 0].item(), 0.0)
        self.assertEqual(features[1, 1].item(), 0.0)
        self.assertEqual(features[1, 2].item(), 0.0)
        self.assertEqual(features[1, 3].item(), 0.0)
        self.assertEqual(features[1, 5].item(), 1.0)  # Recency default for cold product

    def test_company_embedding_generation_and_caching(self):
        """Company text embeddings must generate correctly and persist to disk cache."""
        companies = pd.DataFrame([
            {
                "customer_id": "C1",
                "company_name": "Agro Trading LLC",
                "city": "Полтава",
                "description": "Grain export and agricultural logistics",
                "industry_branch": "АГРО-001",
            },
            {
                "customer_id": "C2",
                "company_name": "Steel Fabrika",
                "city": "Дніпро",
                "description": "Heavy industrial steel fabrication",
                "industry_branch": "МЕТ-002",
            },
        ])
        mock_encoder = MockSentenceTransformer(dim=1024)

        with TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "bge_m3_company.parquet"

            # Pass 1: Encodes and writes to cache
            res1 = generate_company_embeddings(
                companies_df=companies,
                cache_path=cache_file,
                encoder=mock_encoder,
            )
            self.assertEqual(mock_encoder.call_count, 1)
            self.assertIn("C1", res1)
            self.assertIn("C2", res1)
            self.assertEqual(res1["C1"].shape, (1024,))
            self.assertTrue(cache_file.is_file())

            # Verify parquet contains 'customer_id' column
            df_parquet = pd.read_parquet(cache_file)
            self.assertIn("customer_id", df_parquet.columns)
            self.assertEqual(len(df_parquet), 2)

            # Pass 2: Re-use cache without encoding
            res2 = generate_company_embeddings(
                companies_df=companies,
                cache_path=cache_file,
                encoder=mock_encoder,
            )
            self.assertEqual(mock_encoder.call_count, 1)
            np.testing.assert_allclose(res1["C1"], res2["C1"])
            np.testing.assert_allclose(res1["C2"], res2["C2"])

    def test_train_retriever_synthetic_one_epoch(self):
        """End-to-end synthetic 1-epoch retriever training must produce valid checkpoint and history."""
        # Create minimal synthetic purchases, catalogue, and companies
        purchases = pd.DataFrame([
            # C1
            {"customer_id": "C1", "purchase_date": "2024-03-01", "product_id": "P1", "quantity": 2.0, "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2024-07-01", "product_id": "P2", "quantity": 1.0, "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2025-01-01", "product_id": "P1", "quantity": 3.0, "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2025-11-01", "product_id": "P1", "quantity": 3.0, "product_category": "CAT1"},
            {"customer_id": "C1", "purchase_date": "2025-11-01", "product_id": "P_NEW", "quantity": 50000.0, "product_category": "CAT1"},
            # C2
            {"customer_id": "C2", "purchase_date": "2024-04-01", "product_id": "P2", "quantity": 5.0, "product_category": "CAT2"},
            {"customer_id": "C2", "purchase_date": "2024-09-01", "product_id": "P3", "quantity": 4.0, "product_category": "CAT2"},
            {"customer_id": "C2", "purchase_date": "2025-12-01", "product_id": "P2", "quantity": 1.0, "product_category": "CAT2"},
        ])

        catalogue = pd.DataFrame([
            {"product_id": "P1", "brand": "B1", "product_category": "CAT1", "product_subcategory": "S1", "business_line": "L1", "product_name": "Product 1"},
            {"product_id": "P2", "brand": "B1", "product_category": "CAT1", "product_subcategory": "S2", "business_line": "L1", "product_name": "Product 2"},
            {"product_id": "P3", "brand": "B2", "product_category": "CAT2", "product_subcategory": "S3", "business_line": "L2", "product_name": "Product 3"},
        ])

        companies = pd.DataFrame([
            {"customer_id": "C1", "company_name": "Company One", "city": "Kyiv", "region_code": "R1", "industry_branch": "IND1", "description": "Desc 1"},
            {"customer_id": "C2", "company_name": "Company Two", "city": "Lviv", "region_code": "R2", "industry_branch": "IND2", "description": "Desc 2"},
        ])

        mock_encoder = MockSentenceTransformer(dim=1024)

        with TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Pre-populate catalogue & company embeddings in tmp cache
            cat_cache = cache_dir / CATALOGUE_CACHE_WITHOUT_SUBCATEGORY
            comp_cache = cache_dir / "bge_m3_company.parquet"
            from recommender.text import generate_catalogue_embeddings
            generate_company_embeddings(companies, cache_path=comp_cache, encoder=mock_encoder)

            import recommender.train_retriever as tr_module
            orig_cache_dir = tr_module.DEFAULT_CACHE_DIR
            tr_module.DEFAULT_CACHE_DIR = cache_dir

            try:
                with patch.object(
                    tr_module, "generate_catalogue_embeddings",
                    side_effect=lambda items, cache_path: generate_catalogue_embeddings(
                        items, cache_path=cache_path, encoder=mock_encoder,
                    ),
                ) as generate:
                    res = train_retriever(
                        purchases_df=purchases,
                        companies_df=companies,
                        catalogue_df=catalogue,
                        epochs=1,
                        batch_size=2,
                        lr=1e-3,
                        device="cpu",
                        checkpoint_dir=ckpt_dir,
                        seed=42,
                        tower_hidden_dim=64,
                        history_layers=3,
                        extra_tower_mlp_layer=True,
                        output_dim=96,
                        train_eval_size=4,
                    )
                self.assertNotIn("product_subcategory", generate.call_args.args[0].columns)
                self.assertTrue(cat_cache.is_file())

                best_ckpt = Path(res["best_checkpoint_path"])
                hist_path = Path(res["history_path"])

                self.assertTrue(best_ckpt.is_file(), "best_retriever.pt checkpoint must be saved.")
                self.assertTrue(hist_path.is_file(), "training_history.json must be saved.")

                # Validate checkpoint contents
                ckpt = torch.load(best_ckpt, map_location="cpu")
                self.assertIn("model_state_dict", ckpt)
                self.assertIn("optimizer_state_dict", ckpt)
                self.assertIn("product_ids", ckpt)
                self.assertIn("val_metrics", ckpt)
                self.assertEqual(ckpt["config"]["training_epochs"], 1)
                self.assertEqual(len(ckpt["product_ids"]), 3)
                self.assertEqual(ckpt["val_metrics"]["recall@100"], 0.75)

                from recommender.candidates import CandidateRetriever
                from recommender.features import MEDIAN_PURCHASE_QUANTITY, RETRIEVER_PROFILE_FEATURES

                retriever = CandidateRetriever.from_checkpoint(best_ckpt, purchases)
                self.assertIsNone(retriever.model.product_tower.subcat_embedding)
                self.assertEqual(retriever.quantity_reference, 3.0)
                self.assertEqual(retriever.profile_features, (*RETRIEVER_PROFILE_FEATURES, MEDIAN_PURCHASE_QUANTITY))
                self.assertEqual(ckpt["config"]["container_encoding"], "one_hot_none")
                self.assertIsNone(ckpt["config"]["sku_prices_uah"])
                tower = retriever.model.customer_tower
                self.assertEqual(tower.num_tabular_features, 6)
                self.assertEqual(tower.hidden_dim, 64)
                self.assertEqual(tower.output_dim, 96)
                self.assertEqual(len(tower.transformer.layers), 3)
                self.assertEqual(sum(isinstance(layer, torch.nn.Linear) for layer in tower.late_fusion), 3)
                self.assertIsNone(tower.product_embedding)
                self.assertIsNotNone(retriever.profile_normalizer)
                self.assertEqual(tower.catalog_history_encoder, "attention")
                self.assertEqual(tower.history_gap_reference_days, 122.0)
                self.assertEqual(tower.token_proj[0].in_features, tower.output_dim + 2)
                history = torch.zeros(1, 2, ENRICHED_FEATURE_DIM)
                catalog_vectors = torch.randn(3, tower.output_dim)
                purchase_dates = torch.tensor([[-0.25, 0.0]])
                customer_inputs = {
                    "text_embedding": torch.zeros(1, 1024),
                    "city_id": torch.zeros(1, dtype=torch.long),
                    "region_id": torch.zeros(1, dtype=torch.long),
                    "industry_id": torch.zeros(1, dtype=torch.long),
                    "tabular_features": torch.zeros(1, tower.num_tabular_features),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "seq_product_ids": torch.tensor([[1, 2]]),
                    "catalog_product_vectors": catalog_vectors,
                    "seq_purchase_dates": purchase_dates,
                }
                with torch.inference_mode():
                    before = tower(sequence=history, **customer_inputs)
                    history[..., HISTORY_AGE_INDEX] = 1
                    after = tower(sequence=history, **customer_inputs)
                    purchase_dates.add_(1)
                    shifted = tower(sequence=history, **customer_inputs)
                    purchase_dates[0, 0] = 0.25
                    different_purchase_date = tower(sequence=history, **customer_inputs)
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                torch.testing.assert_close(before, shifted, rtol=0, atol=0)
                self.assertFalse(torch.allclose(after, different_purchase_date))

                from recommender.ranker import UPSCANRRanker
                from recommender.train_ranker import load_ranker_retriever

                with patch("recommender.train_ranker.DEFAULT_CACHE_DIR", cache_dir):
                    slate_retriever, _ = load_ranker_retriever(
                        best_ckpt, purchases, catalogue, companies, torch.device("cpu"),
                    )
                self.assertNotIn("subcategory_id", slate_retriever.product_inputs)
                torch.testing.assert_close(slate_retriever.product_inputs["demand_features"][:, -1], torch.ones(3))
                self.assertEqual(slate_retriever.get_product_vectors().shape, (3, tower.output_dim))
                ranker = UPSCANRRanker.from_retriever_checkpoint(best_ckpt)
                self.assertIsNone(ranker.product_tower.subcat_embedding)
                self.assertEqual(ranker.product_tower.text_proj[0].out_features, 64)
                self.assertEqual(sum(isinstance(layer, torch.nn.Linear) for layer in ranker.product_tower.mlp), 3)
                self.assertGreater(ranker.subcat_embedding.num_embeddings, len(ckpt["vocabs"]["subcategory_vocab"]))

                # Validate training history JSON
                with open(hist_path, "r", encoding="utf-8") as f:
                    history = json.load(f)

                self.assertEqual(len(history), 1)
                entry = history[0]
                self.assertIn("train_eval_loss", entry)
                self.assertIn("train_eval_recall@10", entry)
                self.assertIn("val_new_to_customer_recall@50", entry)
                for required_key in [
                    "epoch", "train_loss", "val_loss",
                    "val_recall@10", "val_recall@50", "val_recall@100",
                    "val_ndcg@10", "val_mrr",
                ]:
                    self.assertIn(required_key, entry)
                    self.assertFalse(math.isnan(entry[required_key]))

            finally:
                tr_module.DEFAULT_CACHE_DIR = orig_cache_dir


if __name__ == "__main__":
    unittest.main()
