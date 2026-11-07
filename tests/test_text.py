"""Unit tests for Phase 6: Text Embedding Generation and Offline Caching (R4)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from recommender.text import (
    DEFAULT_EMBEDDING_DIM,
    generate_catalogue_embeddings,
    get_company_text,
    get_product_text,
    load_cached_embeddings,
    save_cached_embeddings,
)


class MockSentenceTransformer:
    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim
        self.call_count = 0

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.call_count += 1
        num_texts = len(texts)
        # Deterministic synthetic embedding based on text length and character hash
        vectors = np.zeros((num_texts, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            val = float(len(t) % 100) / 100.0
            vectors[i, :] = val
            # Normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class TextEmbeddingTests(unittest.TestCase):
    def test_get_product_text_composition(self):
        row = {
            "brand": "SHELL",
            "product_name": "Helix Ultra 5W-40 4л",
            "product_category": "АВТО-СМ",
            "product_subcategory": "МОТОР-МАСЛА",
            "business_line": "PCMO",
        }
        text = get_product_text(row)
        expected = "SHELL Helix Ultra 5W-40 4л АВТО-СМ МОТОР-МАСЛА PCMO"
        self.assertEqual(text, expected)

    def test_get_product_text_with_missing_and_null_tokens(self):
        row = {
            "brand": " NULL ",
            "product_name": "Hydraulic Oil 46",
            "product_category": "",
            "product_subcategory": None,
            "business_line": "IND",
        }
        text = get_product_text(row)
        self.assertEqual(text, "Hydraulic Oil 46 IND")

    def test_get_company_text_composition(self):
        row = {
            "company_name": "Autotransport LLC",
            "city": "Київ",
            "description": "Logistics and truck fleet",
            "industry_branch": "ТБД-014",
        }
        text = get_company_text(row)
        self.assertEqual(text, "Autotransport LLC Київ Logistics and truck fleet ТБД-014")

    def test_generate_catalogue_embeddings_dimensions_and_zero_vector(self):
        catalogue = pd.DataFrame([
            {
                "product_id": "P1",
                "brand": "MOBIL",
                "product_name": "Delvac 15W-40",
                "product_category": "АВТО-СМ",
                "product_subcategory": "МОТОР-МАСЛА",
                "business_line": "HDMO",
            },
            {
                "product_id": "P2_EMPTY",
                "brand": "",
                "product_name": "",
                "product_category": None,
                "product_subcategory": "NULL",
                "business_line": "",
            },
        ])
        mock_encoder = MockSentenceTransformer(dim=1024)
        embeddings = generate_catalogue_embeddings(
            catalogue_df=catalogue,
            encoder=mock_encoder,
            embedding_dim=1024,
        )

        self.assertIn("P1", embeddings)
        self.assertIn("P2_EMPTY", embeddings)
        self.assertEqual(embeddings["P1"].shape, (1024,))
        self.assertEqual(embeddings["P1"].dtype, np.float32)
        # Empty text maps to 1024-d zero vector
        self.assertEqual(embeddings["P2_EMPTY"].shape, (1024,))
        self.assertTrue((embeddings["P2_EMPTY"] == 0.0).all())

    def test_offline_disk_caching_parquet_and_pt(self):
        vec1 = np.ones(1024, dtype=np.float32) * 0.5
        vec2 = np.ones(1024, dtype=np.float32) * 0.8
        orig = {"P1": vec1, "P2": vec2}

        with TemporaryDirectory() as tmpdir:
            # Test Parquet cache
            parquet_path = Path(tmpdir) / "cache.parquet"
            save_cached_embeddings(orig, parquet_path)
            self.assertTrue(parquet_path.is_file())
            loaded_parquet = load_cached_embeddings(parquet_path)
            np.testing.assert_allclose(loaded_parquet["P1"], orig["P1"])
            np.testing.assert_allclose(loaded_parquet["P2"], orig["P2"])

            # Test PT cache
            pt_path = Path(tmpdir) / "cache.pt"
            save_cached_embeddings(orig, pt_path)
            self.assertTrue(pt_path.is_file())
            loaded_pt = load_cached_embeddings(pt_path)
            np.testing.assert_allclose(loaded_pt["P1"], orig["P1"])
            np.testing.assert_allclose(loaded_pt["P2"], orig["P2"])

    def test_cache_reuse_skips_encoding(self):
        catalogue = pd.DataFrame([
            {"product_id": "P1", "product_name": "Oil", "brand": "SHELL"},
        ])
        mock_encoder = MockSentenceTransformer(dim=1024)

        with TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "embeddings.parquet"
            # First pass: generates and caches
            res1 = generate_catalogue_embeddings(
                catalogue, cache_path=cache_file, encoder=mock_encoder,
            )
            self.assertEqual(mock_encoder.call_count, 1)

            # Second pass: loads from cache, encoder not called!
            res2 = generate_catalogue_embeddings(
                catalogue, cache_path=cache_file, encoder=mock_encoder,
            )
            self.assertEqual(mock_encoder.call_count, 1)
            np.testing.assert_allclose(res1["P1"], res2["P1"])


if __name__ == "__main__":
    unittest.main()
