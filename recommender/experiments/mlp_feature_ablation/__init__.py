"""Controlled Stage 1 CompactStage1Selector MLP feature-removal experiment."""

from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    OMITTED_CONTEXT_FEATURES,
    RECIPE_NAME,
    REPEAT_RESTORED_ARTIFACT_ROOT,
    REPEAT_RESTORED_EVIDENCE_ROOT,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
    VARIANT_NAME,
)
from .model import (
    FeatureAblatedStage1Selector,
    load_ablated_selector_checkpoint,
)
from .schema import (
    FeatureAblationSpec,
    build_feature_ablation_spec,
    build_input_contract,
    select_retained_context_features,
    validate_input_contract,
)
from .train import measure_repeat_flag_influence

__all__ = [
    "ARTIFACT_ROOT",
    "EVIDENCE_ROOT",
    "FeatureAblatedStage1Selector",
    "FeatureAblationSpec",
    "OMITTED_CONTEXT_FEATURES",
    "RECIPE_NAME",
    "REPEAT_RESTORED_ARTIFACT_ROOT",
    "REPEAT_RESTORED_EVIDENCE_ROOT",
    "REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES",
    "REPEAT_RESTORED_RECIPE_NAME",
    "REPEAT_RESTORED_VARIANT_NAME",
    "VARIANT_NAME",
    "build_feature_ablation_spec",
    "build_input_contract",
    "load_ablated_selector_checkpoint",
    "measure_repeat_flag_influence",
    "select_retained_context_features",
    "validate_input_contract",
]
