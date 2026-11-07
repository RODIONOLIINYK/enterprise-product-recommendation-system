"""Comprehensive Opaque-Box E2E Test Suite for Enterprise Product Recommendation System.

Structured according to the 4-Tier Test Design Methodology:
- Tier 1: Feature Coverage (Primary behaviors for R1–R5)
- Tier 2: Boundary & Corner Cases (Extreme inputs, limits, zero-cases, outliers)
- Tier 3: Cross-Feature Combinations (Pairwise interactions across pipeline modules)
- Tier 4: Real-World Application Scenarios (Full end-to-end multi-step pipelines)
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List
import unittest

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from recommender.catalogue import (
    ITEM_COLUMN_MAPPING,
    load_product_catalogue,
    select_eligible_products,
    select_sellable_products,
)
from recommender.company import (
    COMPANY_COLUMN_MAPPING,
    EXCLUDED_COMPANY_COLUMNS,
    load_companies,
)
from recommender.features import (
    Log1pScaler,
    PointInTimeNormalizer,
    compute_point_in_time_features,
    compute_urgency,
    normalize_feature,
)
from recommender.sequences import (
    DEFAULT_FEATURE_DIM,
    ENRICHED_FEATURE_DIM,
    DEFAULT_SEQUENCE_LENGTH,
    PAD_TOKEN,
    build_batch_sequences,
    build_customer_sequence,
)
from recommender.split import (
    TEST_START,
    TRAIN_CUTOFF,
    VAL_END,
    VAL_START,
    chronological_split,
    get_split_proportions,
    validate_no_leakage,
)
from recommender.train_retriever import RetrieverDataset, build_retriever_transitions
from recommender.text import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    generate_catalogue_embeddings,
    get_company_text,
    get_product_text,
    load_cached_embeddings,
    save_cached_embeddings,
)


class MockBgeM3Encoder:
    """Mock SentenceTransformer encoder producing 1024-dimensional normalized float32 embeddings."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim
        self.call_count = 0

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.call_count += 1
        num_texts = len(texts)
        embeddings = np.zeros((num_texts, self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic pseudo-embedding from string hash
            rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings[i] = vec
        return embeddings


def create_synthetic_customer_csv(tmp_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write synthetic customer records matching raw CSV structure (118 columns, sep=';')."""
    csv_path = tmp_dir / "customers.csv"
    lines = []
    for r in rows:
        cols = [""] * 118
        cols[1] = str(r.get("customer_id", "C001"))
        cols[2] = str(r.get("company_name", "Test Company"))
        cols[82] = str(r.get("legal_name", "Test Legal LLC"))
        cols[85] = str(r.get("region_code", "UA_02_02"))
        cols[88] = str(r.get("address", "м. Київ, вул. Хрещатик, 1"))
        cols[90] = str(r.get("description", "Distributor of automotive parts"))
        cols[94] = str(r.get("industry_branch", "ТБД-014"))
        # Unwanted columns in export
        cols[44] = str(r.get("sales_branch", "ДНЕПРГЛАВ"))
        cols[59] = str(r.get("vat_status", "К_ПЛАТ"))
        cols[98] = str(r.get("city_code", "ГОР-000062"))
        lines.append(";".join(cols))
    csv_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return csv_path


def create_synthetic_item_csv(tmp_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write synthetic item records matching raw CSV structure (169 columns, sep=';')."""
    csv_path = tmp_dir / "items.csv"
    lines = []
    for r in rows:
        cols = [""] * 169
        cols[1] = str(r.get("product_id", "PROD-001"))
        cols[3] = str(r.get("short_name", "Short Name"))
        cols[5] = str(r.get("product_type", "Type A"))
        cols[6] = str(r.get("unit", "шт"))
        cols[50] = str(r.get("item_type", "ТОВАР"))
        cols[59] = str(r.get("business_line", "PCMO"))
        cols[83] = str(r.get("brand", "SHELL"))
        cols[84] = str(r.get("product_category", "АВТО-СМ"))
        cols[86] = str(r.get("product_subcategory", "МОТОР-МАСЛА"))
        cols[121] = str(r.get("ukrainian_name", "Олива моторна 5W-40"))
        cols[132] = str(r.get("product_name", "Helix Ultra 5W-40 4л"))
        cols[133] = str(r.get("base_name", "Helix Ultra"))
        lines.append(";".join(cols))
    csv_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return csv_path


# ==============================================================================
# TIER 1: FEATURE COVERAGE (R1 - R5 PRIMARY SPECIFICATIONS)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage for requirements R1 through R5."""

    # --- R1: Phase 3 Metadata Preparation ---

    def test_r1_customer_metadata_retains_required_columns(self):
        """Verify load_companies retains customer_id, city, region_code, and industry_branch."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C101", "region_code": "UA_02_02", "industry_branch": "ТБД-014", "address": "м. Львів, вул. Городоцька, 5"},
                {"customer_id": "C102", "region_code": "UA_11_02", "industry_branch": "ТБД-080", "address": "м. Одеса, вул. Дерибасівська, 10"},
            ])
            purchases = pd.DataFrame([
                {"customer_id": "C101", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
                {"customer_id": "C102", "purchase_date": pd.Timestamp("2025-01-02"), "product_id": "P2", "quantity": 2.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])

            df = load_companies(cust_file, purchases=purchases, minimum_purchases=1)

            for col in ["customer_id", "city", "region_code", "industry_branch"]:
                assert col in df.columns, f"Expected required customer column '{col}' missing from schema."
            assert df.loc[df["customer_id"] == "C101", "region_code"].iloc[0] == "UA_02_02"
            assert df.loc[df["customer_id"] == "C101", "industry_branch"].iloc[0] == "ТБД-014"
            assert df.loc[df["customer_id"] == "C101", "city"].iloc[0] == "Львів"

    def test_r1_customer_metadata_excludes_dropped_columns(self):
        """Verify load_companies strictly excludes sales_branch, vat_status, and city_code."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [{"customer_id": "C101"}])
            purchases = pd.DataFrame([
                {"customer_id": "C101", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])

            df = load_companies(cust_file, purchases=purchases, minimum_purchases=1)

            dropped_fields = ["sales_branch", "vat_status", "city_code"]
            for field in dropped_fields:
                assert field not in df.columns, f"Forbidden dropped column '{field}' found in company schema."
            for col_idx, col_name in EXCLUDED_COMPANY_COLUMNS.items():
                assert col_name not in df.columns, f"Excluded column '{col_name}' ({col_idx}) present in output."

    def test_r1_product_catalogue_retains_required_columns(self):
        """Verify load_product_catalogue preserves brand, product_category, product_subcategory, business_line."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            item_file = create_synthetic_item_csv(tmp_path, [
                {"product_id": "P-10", "brand": "SHELL", "product_category": "АВТО-СМ", "product_subcategory": "МОТОР-МАСЛА", "business_line": "PCMO"},
                {"product_id": "P-20", "brand": "MOBIL", "product_category": "ИНД-СМ", "product_subcategory": "ГИДРАВЛИКА", "business_line": "IND"},
            ])

            df = load_product_catalogue(item_file)

            for col in ["product_id", "brand", "product_category", "product_subcategory", "business_line"]:
                assert col in df.columns, f"Required product feature '{col}' missing from catalogue schema."
            row = df.loc[df["product_id"] == "P-10"].iloc[0]
            assert row["brand"] == "SHELL"
            assert row["product_category"] == "АВТО-СМ"
            assert row["product_subcategory"] == "МОТОР-МАСЛА"
            assert row["business_line"] == "PCMO"

    def test_r1_plan_documents_container_type_note(self):
        """Verify plan.md contains explicit note on container_type marked to implement after user discussion."""
        plan_path = Path("plan.md")
        assert plan_path.is_file(), "plan.md must exist at project root."
        content = plan_path.read_text(encoding="utf-8")
        assert "container_type" in content, "plan.md must explicitly reference container_type."
        assert "discussion with the user" in content, "plan.md note must mark container_type for future user discussion."

    def test_r1_metadata_whitespace_stripping_and_null_handling(self):
        """Verify whitespace is trimmed and empty / NULL strings become pd.NA in customer & product schemas."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            item_file = create_synthetic_item_csv(tmp_path, [
                {"product_id": " P-30 ", "brand": "  TOTAL  ", "product_subcategory": "NULL", "business_line": ""},
            ])
            df = load_product_catalogue(item_file)
            row = df.iloc[0]
            assert row["product_id"] == "P-30"
            assert row["brand"] == "TOTAL"
            assert pd.isna(row["product_subcategory"])
            assert pd.isna(row["business_line"])

    def test_r1_product_catalogue_uppercase_normalization(self):
        """Verify categorical columns like product_category and business_line are normalized to uppercase."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            item_file = create_synthetic_item_csv(tmp_path, [
                {"product_id": "P-40", "product_category": "авто-см", "business_line": "pcmo"},
            ])
            df = load_product_catalogue(item_file)
            row = df.iloc[0]
            assert row["product_category"] == "АВТО-СМ"
            assert row["business_line"] == "PCMO"

    # --- R2: Phase 4 Chronological Data Split ---

    def test_r2_chronological_split_date_boundaries(self):
        """Verify dataset partitions into exact 70/15/15 schedule: Train < 2025-10-01, Val 2025-10-01..2026-02-18, Test >= 2026-02-19."""
        purchases = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-05-10")},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2025-09-30 23:59:59")},
            {"customer_id": "C3", "purchase_date": pd.Timestamp("2025-10-01 00:00:00")},
            {"customer_id": "C4", "purchase_date": pd.Timestamp("2026-01-15 12:00:00")},
            {"customer_id": "C5", "purchase_date": pd.Timestamp("2026-02-18 23:59:59")},
            {"customer_id": "C6", "purchase_date": pd.Timestamp("2026-02-19 00:00:00")},
            {"customer_id": "C7", "purchase_date": pd.Timestamp("2026-06-30 18:00:00")},
        ])

        train, val, test = chronological_split(purchases)

        assert len(train) == 2
        assert len(val) == 3
        assert len(test) == 2
        assert train["customer_id"].tolist() == ["C1", "C2"]
        assert val["customer_id"].tolist() == ["C3", "C4", "C5"]
        assert test["customer_id"].tolist() == ["C6", "C7"]
        assert (pd.to_datetime(train["purchase_date"]) < TRAIN_CUTOFF).all()
        assert (pd.to_datetime(val["purchase_date"]) >= VAL_START).all()
        assert (pd.to_datetime(val["purchase_date"]) <= VAL_END).all()
        assert (pd.to_datetime(test["purchase_date"]) >= TEST_START).all()

    def test_r2_chronological_split_mutually_exclusive_and_exhaustive(self):
        """Verify partition indices are completely disjoint and preserve all rows."""
        purchases = pd.DataFrame([
            {"customer_id": f"C_{i}", "purchase_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 15)}
            for i in range(60)
        ])

        train, val, test = chronological_split(purchases)

        total_rows = len(train) + len(val) + len(test)
        assert total_rows == len(purchases)
        assert set(train.index).isdisjoint(set(val.index))
        assert set(train.index).isdisjoint(set(test.index))
        assert set(val.index).isdisjoint(set(test.index))

    def test_r2_validate_no_leakage_on_clean_splits(self):
        """Verify validate_no_leakage returns True on valid chronological splits."""
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-01-01")}], index=[0])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-11-01")}], index=[1])
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[2])

        assert validate_no_leakage(train, val, test) is True

    def test_r2_validate_no_leakage_detects_future_event_in_train(self):
        """Verify validate_no_leakage raises ValueError if train contains an event >= 2025-10-01."""
        train = pd.DataFrame([
            {"purchase_date": pd.Timestamp("2025-01-01")},
            {"purchase_date": pd.Timestamp("2025-10-05")},  # Leaked validation date
        ], index=[0, 1])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-11-01")}], index=[2])
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[3])

        with pytest.raises(ValueError, match="Temporal leakage in train split"):
            validate_no_leakage(train, val, test)

    def test_r2_validate_no_leakage_detects_test_in_val(self):
        """Verify validate_no_leakage raises ValueError if val contains an event >= 2026-02-19."""
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-05-01")}], index=[0])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-02-20")}], index=[1])  # Test date in val!
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[2])

        with pytest.raises(ValueError, match="Temporal leakage in val split"):
            validate_no_leakage(train, val, test)

    def test_r2_validate_no_leakage_detects_index_overlap(self):
        """Verify validate_no_leakage catches duplicate dataframe indices across partitions."""
        train = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-05-01")}], index=[0])
        val = pd.DataFrame([{"purchase_date": pd.Timestamp("2025-11-01")}], index=[0])  # Duplicate index 0!
        test = pd.DataFrame([{"purchase_date": pd.Timestamp("2026-03-01")}], index=[1])

        with pytest.raises(ValueError, match="indices"):
            validate_no_leakage(train, val, test)

    # --- R3: Phase 5 Sequences, Point-in-Time Features, and Normalization ---

    def test_r3_customer_sequence_tensor_shape_and_mask(self):
        """Verify build_customer_sequence produces (32, feature_dim) sequence and (32,) mask."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "product_id": f"P-{i}", "quantity": float(i + 1)}
            for i in range(10)
        ])
        as_of_date = pd.Timestamp("2025-02-01")

        result = build_customer_sequence(history, "C1", as_of_date, max_len=32)

        seq = result["sequence"]
        mask = result["attention_mask"]
        pids = result["product_ids"]

        assert isinstance(seq, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert seq.shape == (32, DEFAULT_FEATURE_DIM)
        assert mask.shape == (32,)
        assert int(mask.sum()) == 10
        assert mask[:10].tolist() == [1] * 10
        assert mask[10:].tolist() == [0] * 22
        assert pids[:10] == [f"P-{i}" for i in range(10)]
        assert pids[10:] == [PAD_TOKEN] * 22

    def test_r3_customer_sequence_chronological_ordering(self):
        """Verify sequence events are sorted ascending chronologically by purchase_date."""
        events = [
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-03-15"), "product_id": "P3", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-10"), "product_id": "P1", "quantity": 2.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-02-20"), "product_id": "P2", "quantity": 3.0},
        ]
        history = pd.DataFrame(events)

        result = build_customer_sequence(history, "C1", pd.Timestamp("2025-04-01"), max_len=32)

        assert result["product_ids"][:3] == ["P1", "P2", "P3"]

    def test_r3_point_in_time_feature_calculation_isolation(self):
        """Verify point-in-time calculation strictly accesses events prior to as_of_date."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 5.0, "product_id": "P1"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-10"), "quantity": 10.0, "product_id": "P2"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-20"), "quantity": 100.0, "product_id": "P3"},  # Future relative to as_of
        ])
        as_of_date = pd.Timestamp("2025-01-15")

        feats = compute_point_in_time_features(history, "C1", as_of_date)

        assert feats["total_purchases"] == 2.0
        assert feats["total_quantity"] == 15.0
        assert feats["distinct_products"] == 2.0
        assert feats["days_since_last_order"] == 5.0  # Jan 15 - Jan 10

    def test_r3_purchase_cadence_urgency_formula(self):
        """Verify compute_urgency returns (days_since_last_order - median_order_gap)."""
        # Overdue
        assert compute_urgency(14.0, 7.0) == 7.0
        # Not yet due
        assert compute_urgency(3.0, 14.0) == -11.0
        # Exactly on cadence
        assert compute_urgency(10.0, 10.0) == 0.0

    def test_r3_log1p_normalization_bounds(self):
        """Verify normalize_feature strictly compresses non-negative values into [0.0, 1.0]."""
        arr = np.array([0.0, 1.0, 10.0, 50.0, 100.0, 500.0, 10000.0])
        max_val = 100.0

        scaled = normalize_feature(arr, max_val=max_val)

        assert (scaled >= 0.0).all()
        assert (scaled <= 1.0).all()
        assert scaled[0] == 0.0
        assert scaled[4] == 1.0  # x == max_val -> 1.0
        assert scaled[5] == 1.0  # x > max_val -> clipped to 1.0
        assert scaled[6] == 1.0  # extreme outlier clipped to 1.0

    def test_r3_log1p_normalization_monotonicity(self):
        """Verify normalize_feature is strictly monotonic for inputs within [0, max_val]."""
        x = np.linspace(0.0, 200.0, 50)
        y = normalize_feature(x, max_val=200.0)
        assert np.all(np.diff(y) >= 0.0)

    # --- R4: Phase 6 Text Embeddings and Offline Caching ---

    def test_r4_bge_m3_embedding_dimension_1024(self):
        """Verify catalogue embeddings generated with BAAI/bge-m3 produce 1024-dimensional float32 vectors."""
        mock_encoder = MockBgeM3Encoder(dim=DEFAULT_EMBEDDING_DIM)
        catalogue = pd.DataFrame([
            {"product_id": "P1", "brand": "SHELL", "product_name": "Helix 5W-40", "product_category": "АВТО-СМ"},
            {"product_id": "P2", "brand": "MOBIL", "product_name": "Delvac 15W-40", "product_category": "ИНД-СМ"},
        ])

        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)

        assert len(embeddings) == 2
        for pid in ["P1", "P2"]:
            assert pid in embeddings
            emb = embeddings[pid]
            assert isinstance(emb, np.ndarray)
            assert emb.dtype == np.float32
            assert emb.shape == (1024,)

    def test_r4_offline_caching_serialization_and_deserialization(self):
        """Verify save_cached_embeddings writes to disk and load_cached_embeddings reads accurately."""
        with TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "test_cache.parquet"
            mock_encoder = MockBgeM3Encoder()
            catalogue = pd.DataFrame([
                {"product_id": "P1", "brand": "SHELL", "product_name": "Helix", "product_category": "АВТО-СМ"},
            ])

            embeddings = generate_catalogue_embeddings(catalogue, cache_path=cache_file, encoder=mock_encoder)
            assert cache_file.is_file(), "Offline embedding cache file must be written to disk."

            loaded = load_cached_embeddings(cache_file)
            assert "P1" in loaded
            assert loaded["P1"].shape == (1024,)
            np.testing.assert_allclose(embeddings["P1"], loaded["P1"], rtol=1e-5)

    def test_r4_offline_caching_avoids_recomputation(self):
        """Verify existing cache file on disk is loaded directly without calling encoder.encode."""
        with TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "test_cache.parquet"
            mock_encoder = MockBgeM3Encoder()
            catalogue = pd.DataFrame([
                {"product_id": "P1", "brand": "SHELL", "product_name": "Helix", "product_category": "АВТО-СМ"},
            ])

            # First run writes cache
            generate_catalogue_embeddings(catalogue, cache_path=cache_file, encoder=mock_encoder)
            assert mock_encoder.call_count == 1

            # Second run with existing cache
            loaded = generate_catalogue_embeddings(catalogue, cache_path=cache_file, encoder=mock_encoder)
            assert mock_encoder.call_count == 1  # Not called again!
            assert "P1" in loaded

    def test_r4_embeddings_normalized_l2(self):
        """Verify generated embeddings are L2 normalized (unit length)."""
        mock_encoder = MockBgeM3Encoder()
        catalogue = pd.DataFrame([
            {"product_id": "P1", "brand": "SHELL", "product_name": "Helix Ultra", "product_category": "АВТО-СМ"},
        ])
        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)
        norm = np.linalg.norm(embeddings["P1"])
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_r4_empty_text_deterministic_embedding(self):
        """Verify empty or missing text yields deterministic 1024-d float32 vector."""
        mock_encoder = MockBgeM3Encoder()
        catalogue = pd.DataFrame([
            {"product_id": "P_EMPTY", "brand": None, "product_name": None, "product_category": None},
        ])
        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)
        assert "P_EMPTY" in embeddings
        assert embeddings["P_EMPTY"].shape == (1024,)
        assert embeddings["P_EMPTY"].dtype == np.float32

    # --- R5: Next-basket transitions ---

    def test_r5_sequential_next_basket_sampling(self):
        """Verify build_retriever_transitions samples sequential (history < t, target = basket_t) pairs before cutoff."""
        purchases = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-06-01"), "product_id": "P1", "product_category": "CAT-A"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-08-01"), "product_id": "P2", "product_category": "CAT-A"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-10-01"), "product_id": "P3", "product_category": "CAT-B"},
            {"customer_id": "C2", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "P4", "product_category": "CAT-A"},  # Single basket
        ])

        examples, _, _ = build_retriever_transitions(purchases)

        # C1 has 3 baskets -> yields 2 transitions: (B1 -> B2) and (B1+B2 -> B3)
        # C2 has 1 basket -> yields 0 transitions
        assert len(examples) == 2
        assert examples[0]["customer_id"] == "C1"
        assert examples[0]["as_of_date"] == pd.Timestamp("2024-08-01")
        assert examples[0]["positive_product_ids"] == ["P2"]
        assert examples[1]["as_of_date"] == pd.Timestamp("2024-10-01")
        assert examples[1]["positive_product_ids"] == ["P3"]


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 PER FEATURE ACROSS R1 - R5)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary, corner, extreme limit, and edge case conditions."""

    # --- R1 Boundaries ---

    def test_r1_boundary_empty_customer_export(self):
        """Verify empty customer export raises appropriate error or returns empty DataFrame."""
        with TemporaryDirectory() as tmp_dir:
            cust_file = Path(tmp_dir) / "empty_customers.csv"
            cust_file.write_text("", encoding="utf-8-sig")
            purchases = pd.DataFrame(columns=["customer_id", "purchase_date", "quantity"])
            with pytest.raises(Exception):
                load_companies(cust_file, purchases=purchases)

    def test_r1_boundary_unparseable_address_yields_na_city(self):
        """Verify addresses with no recognizable locality deterministically produce pd.NA city."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C_UNPARSE", "address": "123456789 ??? @@@"},
            ])
            purchases = pd.DataFrame([
                {"customer_id": "C_UNPARSE", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])
            df = load_companies(cust_file, purchases=purchases, minimum_purchases=1)
            assert pd.isna(df.loc[df["customer_id"] == "C_UNPARSE", "city"].iloc[0])

    def test_r1_boundary_missing_optional_region_branch_retained(self):
        """Verify customer with missing optional region_code or industry_branch is retained with pd.NA."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C_MISSING", "region_code": "NULL", "industry_branch": ""},
            ])
            purchases = pd.DataFrame([
                {"customer_id": "C_MISSING", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])
            df = load_companies(cust_file, purchases=purchases, minimum_purchases=1)
            assert len(df) == 1
            assert pd.isna(df["region_code"].iloc[0])
            assert pd.isna(df["industry_branch"].iloc[0])

    def test_r1_boundary_duplicate_customer_ids_merged(self):
        """Verify multiple lines for identical customer_id are merged into a single distinct entity."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C_DUP", "company_name": "Name 1", "region_code": "UA_02_02"},
                {"customer_id": "C_DUP", "company_name": "Name 1", "region_code": "UA_02_02"},
            ])
            purchases = pd.DataFrame([
                {"customer_id": "C_DUP", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])
            df = load_companies(cust_file, purchases=purchases, minimum_purchases=1)
            assert len(df) == 1

    def test_r1_boundary_catalogue_missing_optional_subcategory(self):
        """Verify product catalogue handles missing subcategory and brand without crashing."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            item_file = create_synthetic_item_csv(tmp_path, [
                {"product_id": "P_NO_SUBCAT", "brand": "", "product_category": "АВТО-СМ", "product_subcategory": "NULL"},
            ])
            df = load_product_catalogue(item_file)
            assert len(df) == 1
            assert pd.isna(df["product_subcategory"].iloc[0])
            assert pd.isna(df["brand"].iloc[0])

    # --- R2 Boundaries ---

    def test_r2_boundary_exact_train_cutoff_timestamp(self):
        """Verify boundary event at exactly 2025-09-30 23:59:59 is Train and 2025-10-01 00:00:00 is Val."""
        purchases = pd.DataFrame([
            {"customer_id": "LAST_TRAIN", "purchase_date": pd.Timestamp("2025-09-30 23:59:59")},
            {"customer_id": "FIRST_VAL", "purchase_date": pd.Timestamp("2025-10-01 00:00:00")},
        ])
        train, val, test = chronological_split(purchases)
        assert train["customer_id"].tolist() == ["LAST_TRAIN"]
        assert val["customer_id"].tolist() == ["FIRST_VAL"]
        assert test.empty

    def test_r2_boundary_exact_val_cutoff_timestamp(self):
        """Verify boundary event at exactly 2026-02-18 23:59:59 is Val and 2026-02-19 00:00:00 is Test."""
        purchases = pd.DataFrame([
            {"customer_id": "LAST_VAL", "purchase_date": pd.Timestamp("2026-02-18 23:59:59")},
            {"customer_id": "FIRST_TEST", "purchase_date": pd.Timestamp("2026-02-19 00:00:00")},
        ])
        train, val, test = chronological_split(purchases)
        assert train.empty
        assert val["customer_id"].tolist() == ["LAST_VAL"]
        assert test["customer_id"].tolist() == ["FIRST_TEST"]

    def test_r2_boundary_single_event_per_split(self):
        """Verify minimal 3-event dataset (1 train, 1 val, 1 test) partitions cleanly."""
        purchases = pd.DataFrame([
            {"customer_id": "T", "purchase_date": pd.Timestamp("2024-01-01")},
            {"customer_id": "V", "purchase_date": pd.Timestamp("2025-11-01")},
            {"customer_id": "TE", "purchase_date": pd.Timestamp("2026-03-01")},
        ])
        train, val, test = chronological_split(purchases)
        assert len(train) == 1
        assert len(val) == 1
        assert len(test) == 1
        assert validate_no_leakage(train, val, test) is True

    def test_r2_boundary_empty_dataset_split(self):
        """Verify empty purchases dataframe splits into three empty dataframes without error."""
        empty_df = pd.DataFrame(columns=["customer_id", "purchase_date"])
        train, val, test = chronological_split(empty_df)
        assert train.empty and val.empty and test.empty

    def test_r2_boundary_all_events_in_single_split(self):
        """Verify dataset where all events fall into train produces valid non-empty train and empty val/test."""
        purchases = pd.DataFrame([
            {"customer_id": f"T{i}", "purchase_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)}
            for i in range(10)
        ])
        train, val, test = chronological_split(purchases)
        assert len(train) == 10
        assert val.empty
        assert test.empty

    # --- R3 Boundaries ---

    def test_r3_boundary_empty_history_customer(self):
        """Verify brand-new customer with zero events yields all-zero mask, zero sequence tensor, and no NaNs."""
        history = pd.DataFrame(columns=["customer_id", "purchase_date", "product_id", "quantity"])
        result = build_customer_sequence(history, "C_NEW", pd.Timestamp("2025-01-01"), max_len=32)

        seq = result["sequence"]
        mask = result["attention_mask"]

        assert seq.shape == (32, DEFAULT_FEATURE_DIM)
        assert mask.shape == (32,)
        assert int(mask.sum()) == 0
        assert not torch.isnan(seq).any()
        assert result["product_ids"] == [PAD_TOKEN] * 32

    def test_r3_boundary_short_history_padding(self):
        """Verify history with only 2 events pads 30 tokens with attention_mask = 0."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-05"), "product_id": "P2", "quantity": 2.0},
        ])
        result = build_customer_sequence(history, "C1", pd.Timestamp("2025-01-10"), max_len=32)

        mask = result["attention_mask"]
        assert int(mask.sum()) == 2
        assert mask[:2].tolist() == [1, 1]
        assert mask[2:].tolist() == [0] * 30

    def test_r3_boundary_long_history_truncation(self):
        """Verify history with 80 events truncates to exactly the most recent 32 events."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "product_id": f"P-{i:03d}", "quantity": 1.0}
            for i in range(80)
        ])
        result = build_customer_sequence(history, "C1", pd.Timestamp("2025-01-01"), max_len=32)

        pids = result["product_ids"]
        assert len(pids) == 32
        assert int(result["attention_mask"].sum()) == 32
        # Most recent 32 events correspond to i=48 to i=79
        assert pids[0] == "P-048"
        assert pids[-1] == "P-079"

    def test_r3_boundary_single_order_urgency_fallback(self):
        """Verify customer with only 1 prior order imputes median gap with population median without NaNs."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 1.0},
        ])
        as_of = pd.Timestamp("2025-01-21")  # 20 days since order

        feats = compute_point_in_time_features(history, "C1", as_of, population_median_gap=14.0)

        assert feats["days_since_last_order"] == 20.0
        assert feats["median_order_gap"] == 14.0
        assert feats["urgency"] == 6.0  # 20 - 14
        assert feats["has_cadence_history"] == 0.0

    def test_r3_boundary_zero_quantity_normalization(self):
        """Verify quantity of 0.0 normalizes to exactly 0.0."""
        val = normalize_feature(0.0, max_val=100.0)
        assert val == 0.0

    def test_r3_boundary_extreme_outlier_quantity_clipped_to_one(self):
        """Verify massive outlier quantity (e.g. 100,000,000) clips strictly to 1.0."""
        val = normalize_feature(100_000_000.0, max_val=500.0)
        assert val == 1.0

    def test_r3_boundary_identical_order_dates_zero_gap(self):
        """Verify multiple orders on the same date do not produce negative gaps."""
        history = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 1.0},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-01-01"), "quantity": 2.0},
        ])
        feats = compute_point_in_time_features(history, "C1", pd.Timestamp("2025-01-10"))
        # Only 1 distinct order date -> fallback imputation
        assert feats["distinct_orders"] == 1.0
        assert feats["has_cadence_history"] == 0.0

    # --- R4 Boundaries ---

    def test_r4_boundary_all_empty_text_fields(self):
        """Verify product with None for brand, name, and category produces valid 1024-d embedding."""
        mock_encoder = MockBgeM3Encoder()
        catalogue = pd.DataFrame([{"product_id": "P_BLANK", "brand": None, "product_name": None, "product_category": None}])
        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)
        assert "P_BLANK" in embeddings
        assert embeddings["P_BLANK"].shape == (1024,)

    def test_r4_boundary_extremely_long_text(self):
        """Verify product with very long concatenated text (>1000 characters) encodes without error."""
        mock_encoder = MockBgeM3Encoder()
        long_name = "Synthetic Lubricant Oil Extreme " * 50
        catalogue = pd.DataFrame([{"product_id": "P_LONG", "brand": "SHELL", "product_name": long_name, "product_category": "АВТО-СМ"}])
        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)
        assert "P_LONG" in embeddings

    def test_r4_boundary_special_characters_and_cyrillic(self):
        """Verify Ukrainian text with quotes, slashes, and numbers encodes faithfully."""
        mock_encoder = MockBgeM3Encoder()
        special_name = 'Олива моторна "Преміум" №1 / 5W-40 (4л) — 100% синтетика'
        catalogue = pd.DataFrame([{"product_id": "P_UKR", "brand": "ЛУКОЙЛ", "product_name": special_name, "product_category": "АВТО-СМ"}])
        embeddings = generate_catalogue_embeddings(catalogue, encoder=mock_encoder)
        assert "P_UKR" in embeddings

    def test_r4_boundary_corrupted_cache_recovery(self):
        """Verify invalid cache file raises error on load and missing cache generates cleanly."""
        with TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "corrupt_cache.parquet"
            cache_file.write_text("NOT_A_VALID_PARQUET_FILE")
            # Loading corrupted cache must raise an error
            with pytest.raises(Exception):
                load_cached_embeddings(cache_file)

            # Fresh generation with non-existent cache creates valid file
            fresh_cache = Path(tmp_dir) / "fresh_cache.parquet"
            mock_encoder = MockBgeM3Encoder()
            catalogue = pd.DataFrame([{"product_id": "P1", "brand": "SHELL", "product_name": "Helix", "product_category": "АВТО-СМ"}])
            embeddings = generate_catalogue_embeddings(catalogue, cache_path=fresh_cache, encoder=mock_encoder)
            assert "P1" in embeddings
            loaded = load_cached_embeddings(fresh_cache)
            assert "P1" in loaded

    def test_r4_boundary_single_item_catalogue_cache(self):
        """Verify catalogue with exactly 1 product serializes and loads cleanly."""
        with TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "single_item.pt"
            embeddings = {"SINGLE_P": np.ones(1024, dtype=np.float32)}
            save_cached_embeddings(embeddings, cache_file)
            loaded = load_cached_embeddings(cache_file)
            assert "SINGLE_P" in loaded
            assert loaded["SINGLE_P"].shape == (1024,)

    # --- R5 Boundaries ---

    def test_r5_boundary_single_basket_customer_lifecycle(self):
        """Verify customers with only a single historical basket cannot produce transitions and are skipped."""
        purchases = pd.DataFrame([
            {"customer_id": "C_SOLO", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "P1", "product_category": "A"},
        ])
        examples, _, _ = build_retriever_transitions(purchases)
        assert len(examples) == 0


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (PAIRWISE INTERACTIONS)
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions between pipeline modules."""

    def test_tier3_customer_metadata_with_sequences_and_normalized_urgency(self):
        """Combine R1 customer metadata with R3 32-event sequence tensor and R3 normalized urgency."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C10", "region_code": "UA_02_02", "industry_branch": "ТБД-014", "address": "м. Київ, вул. Польова, 1"},
            ])
            purchases = pd.DataFrame([
                {"customer_id": "C10", "purchase_date": pd.Timestamp("2025-01-01"), "product_id": "P1", "quantity": 10.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
                {"customer_id": "C10", "purchase_date": pd.Timestamp("2025-01-11"), "product_id": "P2", "quantity": 20.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])

            companies = load_companies(cust_file, purchases=purchases, minimum_purchases=1)
            as_of = pd.Timestamp("2025-01-25")

            # 1. R1 Metadata
            comp_row = companies.loc[companies["customer_id"] == "C10"].iloc[0]
            assert comp_row["city"] == "Київ"
            assert comp_row["region_code"] == "UA_02_02"

            # 2. R3 Sequences
            seq_res = build_customer_sequence(purchases, "C10", as_of, max_len=32)
            assert int(seq_res["attention_mask"].sum()) == 2

            # 3. R3 Urgency & Normalization
            feats = compute_point_in_time_features(purchases, "C10", as_of)
            assert feats["urgency"] == 4.0  # 14 days since last order - 10 median gap

            normalizer = PointInTimeNormalizer()
            normalizer.fit(pd.DataFrame([feats]))
            norm_feats = normalizer.transform_dict(feats)
            assert 0.0 <= norm_feats["total_quantity"] <= 1.0

    def test_tier3_split_boundaries_with_next_basket_sampling(self):
        """Combine R2 chronological splits with R5 next-basket sampling to guarantee zero future leakage."""
        purchases = pd.DataFrame([
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-06-01"), "product_id": "P1", "product_category": "CAT"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-08-01"), "product_id": "P2", "product_category": "CAT"},
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2025-11-01"), "product_id": "P3", "product_category": "CAT"},  # Validation split
            {"customer_id": "C1", "purchase_date": pd.Timestamp("2026-03-01"), "product_id": "P4", "product_category": "CAT"},  # Test split
        ])

        train_df, val_df, test_df = chronological_split(purchases)
        validate_no_leakage(train_df, val_df, test_df)

        # Lifecycle sampling on training events ONLY
        train_examples, _, _ = build_retriever_transitions(train_df)

        assert len(train_examples) == 1
        # Target basket must be strictly prior to 2025-10-01
        assert train_examples[0]["as_of_date"] == pd.Timestamp("2025-08-01")
        assert train_examples[0]["positive_product_ids"] == ["P2"]

    def test_tier3_candidate_eligibility_with_text_embeddings_and_split(self):
        """Combine R2 temporal split, R1 catalogue filtering, and R4 text embeddings."""
        catalogue = pd.DataFrame([
            {"product_id": "P_SOLD_TRAIN", "item_type": "ТОВАР", "brand": "SHELL", "product_category": "LUB"},
            {"product_id": "P_SOLD_TEST", "item_type": "ТОВАР", "brand": "MOBIL", "product_category": "LUB"},
        ])
        purchases = pd.DataFrame([
            {"customer_id": "C1", "product_id": "P_SOLD_TRAIN", "purchase_date": pd.Timestamp("2025-05-01"), "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            {"customer_id": "C2", "product_id": "P_SOLD_TEST", "purchase_date": pd.Timestamp("2026-03-01"), "quantity": 1.0, "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
        ])

        # Candidate eligibility evaluated as of train cutoff
        eligible = select_eligible_products(catalogue, purchases, scoring_time=TRAIN_CUTOFF)
        eligible_pids = eligible["product_id"].tolist()

        assert "P_SOLD_TRAIN" in eligible_pids
        assert "P_SOLD_TEST" not in eligible_pids, "Product with only future test sales must not be eligible at train cutoff."


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (>=5 COMPREHENSIVE WORKFLOWS)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic enterprise product recommendation pipelines from raw data to batches."""

    def test_tier4_scenario_cold_start_customer_recommendation(self):
        """Scenario 1: Brand-new customer with minimal metadata and zero prior transactions."""
        # 1. New customer record
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_file = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "NEW_CORP", "company_name": "New Logistics LLC", "city": "Київ", "region_code": "UA_11_02", "industry_branch": "ТБД-014"},
            ])
            item_file = create_synthetic_item_csv(tmp_path, [
                {"product_id": "P10", "item_type": "ТОВАР", "brand": "SHELL", "product_category": "AUTO"},
                {"product_id": "P20", "item_type": "ТОВАР", "brand": "MOBIL", "product_category": "AUTO"},
            ])
            empty_purchases = pd.DataFrame(columns=["customer_id", "purchase_date", "product_id", "quantity", "transaction_type", "item_type"])

            # Load company with minimum_purchases=0 to support cold-start evaluation
            companies = load_companies(cust_file, purchases=empty_purchases, minimum_purchases=0)
            assert len(companies) == 1
            assert companies.iloc[0]["city"] == "Київ"

            # Build sequence for cold-start customer
            seq_data = build_customer_sequence(empty_purchases, "NEW_CORP", pd.Timestamp("2025-06-01"), max_len=32)
            assert int(seq_data["attention_mask"].sum()) == 0
            assert not torch.isnan(seq_data["sequence"]).any()

            # Point-in-time features default cleanly
            pit_feats = compute_point_in_time_features(empty_purchases, "NEW_CORP", pd.Timestamp("2025-06-01"), population_median_gap=14.0)
            assert pit_feats["total_purchases"] == 0.0
            assert pit_feats["urgency"] == -14.0  # (0 - 14)

    def test_tier4_scenario_mature_active_customer_pipeline(self):
        """Scenario 2: Mature active enterprise customer with 50+ purchases across 2 years."""
        # 50 weekly purchases from 2024 to 2025
        events = [
            {
                "customer_id": "MATURE_CORP",
                "purchase_date": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w),
                "product_id": f"PROD_{w % 5}",
                "quantity": float((w % 4 + 1) * 5),
                "product_category": "LUBRICANTS",
                "transaction_type": "ПРОДАЖА",
                "item_type": "ТОВАР",
            }
            for w in range(50)
        ]
        purchases = pd.DataFrame(events)

        # Chronological split
        train, val, test = chronological_split(purchases)
        validate_no_leakage(train, val, test)

        scoring_date = pd.Timestamp("2024-12-01")
        # Build sequence (truncates to most recent 32 events before scoring date)
        seq_res = build_customer_sequence(purchases, "MATURE_CORP", scoring_date, max_len=32)

        assert int(seq_res["attention_mask"].sum()) == 32
        assert len(seq_res["product_ids"]) == 32

        # Point-in-time cadence and lifetime features
        pit_feats = compute_point_in_time_features(purchases, "MATURE_CORP", scoring_date)
        assert pit_feats["distinct_orders"] > 40
        assert pit_feats["median_order_gap"] == 7.0  # 1-week intervals
        assert pit_feats["has_cadence_history"] == 1.0

    def test_tier4_scenario_multi_item_basket_reorder_prediction(self):
        """Scenario 3: Multi-product baskets retain every target SKU."""
        purchases = pd.DataFrame([
            # Basket 1
            {"customer_id": "C_MULTI", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "OIL_10W40", "quantity": 4.0, "product_category": "OILS"},
            {"customer_id": "C_MULTI", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "FILTER_A", "quantity": 1.0, "product_category": "FILTERS"},
            # Basket 2 (target basket)
            {"customer_id": "C_MULTI", "purchase_date": pd.Timestamp("2024-06-01"), "product_id": "OIL_10W40", "quantity": 4.0, "product_category": "OILS"},
            {"customer_id": "C_MULTI", "purchase_date": pd.Timestamp("2024-06-01"), "product_id": "FILTER_B", "quantity": 2.0, "product_category": "FILTERS"},
        ])

        examples, _, _ = build_retriever_transitions(purchases)
        assert len(examples) == 1
        positives = examples[0]["positive_product_ids"]
        assert sorted(positives) == ["FILTER_B", "OIL_10W40"]


    def test_tier4_scenario_seasonal_boundary_transition(self):
        """Scenario 4: Customer history crossing temporal splits; verifying validation scoring isolation."""
        purchases = pd.DataFrame([
            {"customer_id": "C_SEASONAL", "purchase_date": pd.Timestamp("2025-05-10"), "product_id": "P_SUMMER", "quantity": 5.0},
            {"customer_id": "C_SEASONAL", "purchase_date": pd.Timestamp("2025-11-15"), "product_id": "P_WINTER", "quantity": 5.0},  # Val period
            {"customer_id": "C_SEASONAL", "purchase_date": pd.Timestamp("2026-03-01"), "product_id": "P_SPRING", "quantity": 5.0},  # Test period
        ])

        # Scoring time in Validation period: 2025-12-01
        scoring_time = pd.Timestamp("2025-12-01")

        # Sequences must include Train + prior Val events, but ZERO Test events
        seq_res = build_customer_sequence(purchases, "C_SEASONAL", scoring_time, max_len=32)

        valid_pids = [p for p in seq_res["product_ids"] if p != PAD_TOKEN]
        assert valid_pids == ["P_SUMMER", "P_WINTER"]
        assert "P_SPRING" not in valid_pids, "Future test purchase must not leak into validation scoring."

    def test_tier4_scenario_full_end_to_end_data_to_training_batch(self):
        """Scenario 5: Complete pipeline from synthetic raw data through PyTorch DataLoader batch."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cust_csv = create_synthetic_customer_csv(tmp_path, [
                {"customer_id": "C1", "region_code": "UA_02_02", "industry_branch": "ТБД-014", "address": "м. Київ, вул. Медова, 1"},
                {"customer_id": "C2", "region_code": "UA_11_02", "industry_branch": "ТБД-080", "address": "м. Одеса, вул. Канатна, 2"},
            ])
            item_csv = create_synthetic_item_csv(tmp_path, [
                {"product_id": "P1", "item_type": "ТОВАР", "brand": "SHELL", "product_category": "LUB"},
                {"product_id": "P2", "item_type": "ТОВАР", "brand": "SHELL", "product_category": "LUB"},
                {"product_id": "P3", "item_type": "ТОВАР", "brand": "MOBIL", "product_category": "LUB"},
                {"product_id": "P4", "item_type": "ТОВАР", "brand": "MOBIL", "product_category": "LUB"},
            ])

            raw_purchases = pd.DataFrame([
                {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-02-01"), "product_id": "P1", "quantity": 5.0, "product_category": "LUB", "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
                {"customer_id": "C1", "purchase_date": pd.Timestamp("2024-04-01"), "product_id": "P2", "quantity": 10.0, "product_category": "LUB", "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
                {"customer_id": "C2", "purchase_date": pd.Timestamp("2024-03-01"), "product_id": "P3", "quantity": 2.0, "product_category": "LUB", "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
                {"customer_id": "C2", "purchase_date": pd.Timestamp("2024-05-01"), "product_id": "P4", "quantity": 8.0, "product_category": "LUB", "transaction_type": "ПРОДАЖА", "item_type": "ТОВАР"},
            ])

            # 1. R1: Load metadata
            companies = load_companies(cust_csv, purchases=raw_purchases, minimum_purchases=1)
            catalogue = load_product_catalogue(item_csv)
            assert len(companies) == 2
            assert len(catalogue) == 4

            # 2. R2: Split
            train, val, test = chronological_split(raw_purchases)
            validate_no_leakage(train, val, test)

            # 3. R4: Generate embeddings & offline cache
            mock_encoder = MockBgeM3Encoder(dim=1024)
            cache_file = tmp_path / "catalogue_emb.parquet"
            generate_catalogue_embeddings(catalogue, cache_path=cache_file, encoder=mock_encoder)
            assert cache_file.is_file()

            # 4. R3 & R5: Sample next-basket training dataset
            train_examples, _, _ = build_retriever_transitions(train)
            assert len(train_examples) == 2

            product_to_idx = {pid: i for i, pid in enumerate(catalogue["product_id"])}
            dataset = RetrieverDataset(
                transitions=train_examples,
                purchases_df=train,
                product_to_idx=product_to_idx,
                customer_metadata={},
                customer_embeddings={},
                vocabularies={"product_vocab": {pid: i + 1 for pid, i in product_to_idx.items()}},
            )

            # 5. PyTorch DataLoader Batch Iteration
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))

            assert "sequence" in batch
            assert "attention_mask" in batch
            assert batch["sequence"].shape == (2, 32, ENRICHED_FEATURE_DIM)
            assert batch["attention_mask"].shape == (2, 32)
            torch.testing.assert_close(batch["target"].sum(dim=1), torch.ones(2))
