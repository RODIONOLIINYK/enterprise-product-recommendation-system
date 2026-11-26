"""Isolated paths, frozen reference expectations, and campaign hyperparameters for candidate_token_enrichment."""

import os
from pathlib import Path

from recommender.experiments.rolling_candidate_v2.config import (
    BATCH_SIZE,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    CACHE_ROOT as ROLLING_V3_CACHE_ROOT,
    DEFAULT_DISCOVERY_RANKER,
    DEFAULT_RETRIEVER,
    EPOCHS,
    ETA_MIN,
    EXPECTED_CONTRACT_DIGEST,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    EXPECTED_RETRIEVER_SHA256,
    EXPECTED_SELECTOR_DIM,
    EXPECTED_SELECTOR_RECIPE,
    EXPECTED_SELECTOR_SHA256,
    EXPECTED_SELECTOR_VARIANT,
    EXPECTED_TRAIN_PURCHASE_CACHE_SHA256,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SCORES_SHA256,
    EXPECTED_UPSTREAM_VARIANT,
    EXPECTED_VAL_NEW_TARGETS,
    EXPECTED_VAL_PURCHASE_CACHE_SHA256,
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SCORES_SHA256,
    EXPECTED_VAL_TOTAL_TARGETS,
    GRAD_CLIP_NORM,
    LR,
    MIN_ELIGIBLE_SKUS,
    SEED,
    SELECTOR_SOURCE_RUN_DIR,
    SLATE_POLICY,
    SLATE_SIZE,
    STAGE1_MLP_REF_MANIFEST,
    STAGE1_MLP_REF_POINTER,
    STAGE1_MLP_REF_RETRIEVER,
    STAGE1_MLP_REF_SELECTOR,
    STATISTICS_POLICY,
    UPSTREAM_STAGE1_RUN_DIR,
    WEIGHT_DECAY,
)

# Reuse sealed v3 rolling feature/score/slate cache read-only.
CACHE_ROOT = ROLLING_V3_CACHE_ROOT

ARTIFACT_ROOT = Path(
    os.environ.get(
        "RECSYS_CANDIDATE_TOKEN_ENRICHMENT_ROOT",
        "checkpoints/candidate-token-enrichment-v1",
    )
)
EVIDENCE_ROOT = Path(
    os.environ.get(
        "RECSYS_CANDIDATE_TOKEN_ENRICHMENT_EVIDENCE",
        "experiments/candidate_token_enrichment_v1",
    )
)

ARM_A_NAME = "arm_a_control_pointwise_425d"
ARM_B_NAME = "arm_b_cand_attn_425d"
ARM_C_NAME = "arm_c_cand_attn_enriched_128d"

ENRICHMENT_INIT_SEED_OFFSET = 1000
