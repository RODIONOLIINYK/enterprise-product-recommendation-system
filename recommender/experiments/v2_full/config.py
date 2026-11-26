"""Configuration & strict chronological split boundaries for the 4-year `v2_full` dataset."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Dict, Optional

import pandas as pd

from recommender.artifacts import file_digest
from recommender.development import load_development_purchases

DOWNLOADS_ROOT = Path("/Users/kalyma123/Downloads")
DATA_ROOT = Path("data/v2_full")
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PREPARED_ROOT = INTERIM_ROOT / "prepared_v2_full"

ARTIFACT_ROOT = Path("checkpoints/v2_full")
EVIDENCE_ROOT = Path("experiments/v2_full")
V2_FULL_REFERENCE_PATH = EVIDENCE_ROOT / "v2_full_reference.json"
ACTIVE_V2_FULL_BASE_MODEL = "arm4_e2e_unseen_attn_aux"
ACTIVE_V2_FULL_REFERENCE_ID = "v2-full-arm4-w2a-2mem-recency-hl365-cal-disc-ep10"

RAW_WORKBOOK_PATH = RAW_ROOT / "dataset.xlsx"
RAW_CUSTOMERS_PATH = RAW_ROOT / "customers.csv"
RAW_ITEMS_PATH = RAW_ROOT / "items.csv"

CLEANED_PURCHASES_PATH = INTERIM_ROOT / "cleaned_purchases.csv"
DEV_PURCHASES_PATH = INTERIM_ROOT / "cleaned_dev_purchases.csv"
COMMERCIAL_DEV_PARQUET_PATH = INTERIM_ROOT / "commercial_dev_events.parquet"
COMPANY_TEXT_PARQUET_PATH = INTERIM_ROOT / "company_text_bge_m3.parquet"
BN_STATIC_CACHE_PATH = INTERIM_ROOT / "bn_static_v2_full.pt"

# Strict chronological split over 2023-01-03 .. 2026-11-20 (168,112 paid baskets):
#   Train: < 2026-02-01  (37 months, ~138,297 baskets = 82.3%)
#   Val:   2026-02-01 .. 2026-06-30  (5 months, ~16,625 baskets = 9.9%)
#   Test:  >= 2026-07-01 .. 2026-11-20  (4.7 months, ~13,190 baskets = 7.8% — STRICTLY UNTOUCHED!)
TRAIN_CUTOFF_V2 = pd.Timestamp("2026-02-01")
TEST_START_V2 = pd.Timestamp("2026-07-01")

SEED = 42
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 128
STAGE2_WEIGHT_DECAY = 1e-2
STAGE2_ETA_MIN = 1e-5
STAGE2_GRAD_CLIP_NORM = 1.0


def load_v2_development_purchases():
    # Existing local datasets and Colab bundles use the original CSV filename.
    path = DEV_PURCHASES_PATH if DEV_PURCHASES_PATH.is_file() else CLEANED_PURCHASES_PATH
    return load_development_purchases(path, test_start=TEST_START_V2)


def resolve_v2_full_reference_predictions(
    sync_dir: Optional[Path] = None,
    *,
    reference_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Resolve and SHA-256-verify the canonical `v2_full` reference prediction archives.

    Returns a mapping containing available verified keys from `v2_full_reference.json`:
      - `v2_full_ref_primary`: Primary Ep 6 `optionA_oof_discovery` (`== optionA_oof_unseen`) calibrated predictions (`predictions_primary`)
      - `v2_full_ref_ndcg`: Companion Ep 10 main-head NDCG predictions (`predictions_ndcg`)
      - `v2_full_ref_cal_ndcg`: Companion Ep 10 `optionA_oof_ndcg` calibrated predictions (`predictions_cal_ndcg`)
    Copies missing files from `sync_dir` when running on Colab or a clean checkout.
    """
    ref_p = V2_FULL_REFERENCE_PATH if reference_path is None else Path(reference_path)
    if not ref_p.is_file() and sync_dir is not None and not ref_p.is_absolute():
        stored_ref = sync_dir / ref_p
        if stored_ref.is_file():
            ref_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stored_ref, ref_p)
    if not ref_p.is_file():
        return {}

    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    ckpts = ref.get("checkpoints", {})
    resolved: Dict[str, Path] = {}
    for label, key in (
        ("v2_full_ref_primary", "predictions_primary"),
        ("v2_full_ref_ndcg", "predictions_ndcg"),
        ("v2_full_ref_cal_ndcg", "predictions_cal_ndcg"),
    ):
        entry = ckpts.get(key)
        if not entry or not entry.get("path"):
            continue
        p = Path(entry["path"])
        if not p.is_file() and sync_dir is not None and not p.is_absolute():
            stored = sync_dir / p
            if stored.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stored, p)
        if not p.is_file():
            continue
        expected_sha = entry.get("sha256")
        if expected_sha and file_digest(p) != expected_sha:
            raise ValueError(
                f"Checksum mismatch for v2_full reference {label} ({p}): "
                f"expected {expected_sha}, got {file_digest(p)}"
            )
        resolved[label] = p
    return resolved
