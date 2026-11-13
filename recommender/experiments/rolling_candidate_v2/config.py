"""Isolated paths, frozen reference expectations, and hyperparameters for rolling_candidate_v2."""

import os
from pathlib import Path

from recommender.candidates import (
    STATISTICS_POLICY_FROZEN,
    STATISTICS_POLICY_ROLLING,
)


LEGACY_V2_CACHE_ROOT = Path("data/interim/rolling_candidate_v2")
CACHE_ROOT = Path(
    os.environ.get("RECSYS_ROLLING_CANDIDATE_V2_CACHE", "data/interim/rolling_candidate_v2/v3")
)
ARTIFACT_ROOT = Path(
    os.environ.get("RECSYS_ROLLING_CANDIDATE_V2_ROOT", "checkpoints/rolling-candidate-v2")
)
EVIDENCE_ROOT = Path(
    os.environ.get("RECSYS_ROLLING_CANDIDATE_V2_EVIDENCE", "experiments/rolling_candidate_v2")
)

STATISTICS_POLICY = STATISTICS_POLICY_ROLLING
LEGACY_FROZEN_POLICY = STATISTICS_POLICY_FROZEN

CANDIDATE_SOURCE_ROLLING_SELECTOR = "rolling_selector_scores"
CANDIDATE_SOURCE_REFERENCE_SELECTOR = "reference_selector_scores"
CANDIDATE_SOURCE_REFERENCE_DOT = "reference_dot_scores"
DEFAULT_SLATE_CANDIDATE_SOURCE = CANDIDATE_SOURCE_ROLLING_SELECTOR

STAGE1_MLP_REF_POINTER = Path("experiments/stage1_mlp_reference/reference.json")
STAGE1_MLP_REF_DIR = Path("checkpoints/stage1-mlp-reference/repeat-no-month-v1")
STAGE1_MLP_REF_MANIFEST = STAGE1_MLP_REF_DIR / "manifest.json"
STAGE1_MLP_REF_RETRIEVER = STAGE1_MLP_REF_DIR / "best_retriever.pt"
STAGE1_MLP_REF_SELECTOR = STAGE1_MLP_REF_DIR / "best_selector.pt"

SELECTOR_SOURCE_RUN_DIR = Path(
    "checkpoints/mlp-repeat-restored-v1/runs/selector-E_no_brand-repeat_no_month-uv_45-seed42"
)
UPSTREAM_STAGE1_RUN_DIR = Path(
    "checkpoints/embedding-ablation-v1/runs/stage1-E_no_brand-seed42"
)
ABLATION_BUNDLE_PATH = Path("data/interim/embedding_ablation_v1/ablation_bundle.pt")
E1_E5_RETRIEVAL_CACHE_DIR = Path("data/interim/e1_e5_retrieval_v1")
DEFAULT_RETRIEVER = Path("checkpoints/retriever-discovery-reference/best_retriever.pt")
DEFAULT_DISCOVERY_RANKER = Path("checkpoints/ranker/best_discovery_ranker.pt")

EXPECTED_REFERENCE_ID = "repeat-no-month-v1"
EXPECTED_UPSTREAM_VARIANT = "E_no_brand"
EXPECTED_SELECTOR_VARIANT = "repeat_no_month"
EXPECTED_SELECTOR_RECIPE = "uv_45"
EXPECTED_SELECTOR_DIM = 301

EXPECTED_MANIFEST_SHA256 = (
    "7f8d8d1e01fc210b0247851cea0364dc95dc45556d0dadea57d5ceec0389ab53"
)
EXPECTED_RETRIEVER_SHA256 = (
    "2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981"
)
EXPECTED_SELECTOR_SHA256 = (
    "977241cf9fe8a36ae5c0b5afede6c8018e55eb3202e39d88fa7e3693bc0bc22a"
)
EXPECTED_TRAIN_SCORES_SHA256 = (
    "0cb99fa57b1d46ff295ba4d410ce95e2b45db34ff8344b2006e74a2e79d9e5cc"
)
EXPECTED_VAL_SCORES_SHA256 = (
    "bf7bb5ea75ba22260048f6894f54eba447d56ef99e409763e6e105b7a285e31d"
)
EXPECTED_TRAIN_PURCHASE_CACHE_SHA256 = (
    "e7308f97608a0363027a930ee1ff958f83c326cdac1dcf156b52475c554056a6"
)
EXPECTED_VAL_PURCHASE_CACHE_SHA256 = (
    "22329d61cb9a2ed58ce21e6dff72f1bd1914706bb7b85ef0bec59fada0f957c1"
)
EXPECTED_CONTRACT_DIGEST = (
    "8761650f542b7bdd7605065e686b3e61048c3656a313d3223c8ba4a0027535c2"
)

SLATE_POLICY = "quota100_50"
SLATE_SIZE = 100
MIN_ELIGIBLE_SKUS = 100
EXPECTED_PRODUCT_UNIVERSE = 841
EXPECTED_TRAIN_ROWS = 26253
EXPECTED_VAL_ROWS = 6191
EXPECTED_VAL_TOTAL_TARGETS = 10276
EXPECTED_VAL_NEW_TARGETS = 1857

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
ETA_MIN = 1e-5
GRAD_CLIP_NORM = 1.0
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 123
