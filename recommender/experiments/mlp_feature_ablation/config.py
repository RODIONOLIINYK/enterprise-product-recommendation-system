"""Configuration and verified reference constants for Stage 1 MLP feature ablation."""

import os
from pathlib import Path
from typing import Dict, Tuple


ARTIFACT_ROOT = Path(
    os.environ.get("RECSYS_MLP_FEAT_ABLATION_ROOT", "checkpoints/mlp-feature-ablation-v1")
)
EVIDENCE_ROOT = Path(
    os.environ.get("RECSYS_MLP_FEAT_ABLATION_EVIDENCE", "experiments/mlp_feature_ablation_v1")
)

REFERENCE_POINTER = Path("experiments/embedding_ablation_v1/reference.json")
REFERENCE_DIR = Path("checkpoints/stage1-mlp-reference/no-brand-v1")
REFERENCE_MANIFEST = REFERENCE_DIR / "manifest.json"
REFERENCE_RETRIEVER = REFERENCE_DIR / "best_retriever.pt"
REFERENCE_SELECTOR = REFERENCE_DIR / "best_selector.pt"

UPSTREAM_STAGE1_RUN_DIR = Path(
    "checkpoints/embedding-ablation-v1/runs/stage1-E_no_brand-seed42"
)
BASELINE_SELECTOR_RUN_DIR = Path(
    "checkpoints/embedding-ablation-v1/runs/selector-E_no_brand-uv_47-seed42"
)
BASELINE_EVAL_DIR = Path("experiments/mlp_baseline_v1/evaluation-20261113T070730Z")
DYNAMIC_CACHE_DIR = Path("data/interim/dynamic_campaign_v1")
E1_E5_RETRIEVAL_CACHE_DIR = Path("data/interim/e1_e5_retrieval_v1")
AUDIT_ARTIFACTS_PATH = Path("audit/artifacts.json")

EXPECTED_REFERENCE_ID = "no-brand-v1"
EXPECTED_RETRIEVER_SHA256 = (
    "2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981"
)
EXPECTED_SELECTOR_SHA256 = (
    "d083a5562d968e8e606f69d86e142e370567fa4f506f650e9e2234c53d66dad5"
)
EXPECTED_MANIFEST_SHA256 = (
    "4e223d3de158a79a1dfd08e65b372d699652239edb0d9462250a987e92e0be86"
)

OMITTED_CONTEXT_FEATURES: Tuple[str, ...] = (
    "is_repeat_sku",
    "month_sin",
    "month_cos",
)
EXPECTED_OMITTED_SOURCE_INDICES: Dict[str, int] = {
    "is_repeat_sku": 0,
    "month_sin": 31,
    "month_cos": 32,
}

VARIANT_NAME = "no_repeat_no_month"
UPSTREAM_VARIANT_NAME = "E_no_brand"
RECIPE_NAME = "uv_44"

REPEAT_RESTORED_ARTIFACT_ROOT = Path(
    os.environ.get("RECSYS_MLP_REPEAT_RESTORED_ROOT", "checkpoints/mlp-repeat-restored-v1")
)
REPEAT_RESTORED_EVIDENCE_ROOT = Path(
    os.environ.get("RECSYS_MLP_REPEAT_RESTORED_EVIDENCE", "experiments/mlp_repeat_restored_v1")
)
REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES: Tuple[str, ...] = (
    "month_sin",
    "month_cos",
)
REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES: Dict[str, int] = {
    "month_sin": 31,
    "month_cos": 32,
}
REPEAT_RESTORED_VARIANT_NAME = "repeat_no_month"
REPEAT_RESTORED_RECIPE_NAME = "uv_45"

PRIOR_300_RUN_DIR = Path(
    "checkpoints/mlp-feature-ablation-v1/runs/selector-E_no_brand-no_repeat_no_month-uv_44-seed42"
)
PRIOR_300_EVIDENCE_DIR = Path("experiments/mlp_feature_ablation_v1")
EXPECTED_PRIOR_300_SELECTOR_SHA256 = (
    "44904b172da3d2a7dc504fd3832374509753fe6181deeb8a4ddda60cf9f731f3"
)

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
POS_WEIGHT = 20.0
MIN_ELIGIBLE_SKUS = 100
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 123
