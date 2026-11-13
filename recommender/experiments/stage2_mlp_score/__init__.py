"""Isolated Stage 2 UPS-CANR integration with frozen Stage 1 301D MLP score."""

from .config import (
    ARTIFACT_ROOT,
    CACHE_ROOT,
    EVIDENCE_ROOT,
    SCORE_FEATURE_NAME,
    SCORE_TRANSFORM_VERSION,
)
from .model import (
    Stage2MLPScoreInferenceAdapter,
    Stage2MLPScoreRanker,
    build_stage2_mlp_input_contract,
    gather_teacher_scores_by_canonical_indices,
    load_stage2_mlp_score_checkpoint,
    save_stage2_mlp_score_checkpoint,
    transform_teacher_logits_to_mlp_score,
)

__all__ = [
    "ARTIFACT_ROOT",
    "CACHE_ROOT",
    "EVIDENCE_ROOT",
    "SCORE_FEATURE_NAME",
    "SCORE_TRANSFORM_VERSION",
    "Stage2MLPScoreInferenceAdapter",
    "Stage2MLPScoreRanker",
    "build_stage2_mlp_input_contract",
    "gather_teacher_scores_by_canonical_indices",
    "load_stage2_mlp_score_checkpoint",
    "save_stage2_mlp_score_checkpoint",
    "transform_teacher_logits_to_mlp_score",
]
