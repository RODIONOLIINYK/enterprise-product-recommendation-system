"""Name-based schema selection and strict input contract validation."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from recommender.artifacts import json_digest
from recommender.experiments.dynamic_selector.feature_schema import NAMES as SOURCE_CONTEXT_NAMES
from recommender.experiments.static_enrichment.config import TRANSFERRED_CUSTOMER_STAT_NAMES
from recommender.models import OUTPUT_DIM

from .config import EXPECTED_OMITTED_SOURCE_INDICES, OMITTED_CONTEXT_FEATURES, RECIPE_NAME


@dataclass(frozen=True)
class FeatureAblationSpec:
    """Ordered feature-selection contract mapping canonical 41-col context to retained columns."""

    name: str
    recipe: str
    source_names: Tuple[str, ...]
    omitted_names: Tuple[str, ...]
    omitted_indices: Tuple[int, ...]
    retained_names: Tuple[str, ...]
    retained_indices: Tuple[int, ...]
    customer_stats_names: Tuple[str, ...]
    embedding_dim: int = OUTPUT_DIM

    @property
    def source_num_features(self) -> int:
        return len(self.source_names)

    @property
    def num_retained_features(self) -> int:
        return len(self.retained_names)

    @property
    def num_customer_stats(self) -> int:
        return len(self.customer_stats_names)

    @property
    def scalar_feature_count(self) -> int:
        return self.num_retained_features + self.num_customer_stats

    @property
    def input_dim(self) -> int:
        return self.embedding_dim * 2 + self.scalar_feature_count

    @property
    def ordered_input_blocks(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("u", self.embedding_dim),
            ("v", self.embedding_dim),
            (f"features_{self.num_retained_features}", self.num_retained_features),
            (f"customer_stats_{self.num_customer_stats}", self.num_customer_stats),
        )


def build_feature_ablation_spec(
    omitted_names: Sequence[str] = OMITTED_CONTEXT_FEATURES,
    *,
    name: str = "no_repeat_no_month",
    recipe: str = RECIPE_NAME,
    source_names: Sequence[str] = SOURCE_CONTEXT_NAMES,
    customer_stats_names: Sequence[str] = TRANSFERRED_CUSTOMER_STAT_NAMES,
    embedding_dim: int = OUTPUT_DIM,
    expected_omitted_indices: Optional[Mapping[str, int]] = EXPECTED_OMITTED_SOURCE_INDICES,
) -> FeatureAblationSpec:
    """Resolve omitted/retained columns strictly by ordered schema name (never magic offsets)."""
    src_tuple = tuple(str(n) for n in source_names)
    if len(set(src_tuple)) != len(src_tuple):
        raise ValueError("source_names contains duplicate feature names.")
    stats_tuple = tuple(str(n) for n in customer_stats_names)
    if len(set(stats_tuple)) != len(stats_tuple):
        raise ValueError("customer_stats_names contains duplicate feature names.")

    omit_tuple = tuple(str(n) for n in omitted_names)
    if len(set(omit_tuple)) != len(omit_tuple):
        raise ValueError(f"Duplicate feature names in omitted_names: {omit_tuple}")

    name_to_idx = {feat_name: idx for idx, feat_name in enumerate(src_tuple)}
    missing = [feat_name for feat_name in omit_tuple if feat_name not in name_to_idx]
    if missing:
        raise ValueError(f"Omitted feature name(s) not found in canonical schema: {missing}")

    if expected_omitted_indices is not None and src_tuple == tuple(SOURCE_CONTEXT_NAMES):
        for feat_name in omit_tuple:
            if feat_name in expected_omitted_indices:
                actual_idx = name_to_idx[feat_name]
                exp_idx = int(expected_omitted_indices[feat_name])
                if actual_idx != exp_idx:
                    raise ValueError(
                        f"Schema offset mismatch for {feat_name!r}: expected index {exp_idx}, got {actual_idx}."
                    )

    omit_set = set(omit_tuple)
    omitted_indices = tuple(name_to_idx[feat_name] for feat_name in omit_tuple)
    retained_pairs = [
        (idx, feat_name)
        for idx, feat_name in enumerate(src_tuple)
        if feat_name not in omit_set
    ]
    if not retained_pairs:
        raise ValueError("Cannot omit all contextual features; at least one retained feature is required.")

    retained_indices = tuple(idx for idx, _ in retained_pairs)
    retained_names = tuple(feat_name for _, feat_name in retained_pairs)

    return FeatureAblationSpec(
        name=name,
        recipe=recipe,
        source_names=src_tuple,
        omitted_names=omit_tuple,
        omitted_indices=omitted_indices,
        retained_names=retained_names,
        retained_indices=retained_indices,
        customer_stats_names=stats_tuple,
        embedding_dim=int(embedding_dim),
    )


def _contract_body(spec: FeatureAblationSpec) -> Dict[str, Any]:
    return {
        "contract_version": 1,
        "variant_name": spec.name,
        "recipe": spec.recipe,
        "input_dim": spec.input_dim,
        "embedding_dim": spec.embedding_dim,
        "scalar_feature_count": spec.scalar_feature_count,
        "source_context_dim": spec.source_num_features,
        "source_context_names": list(spec.source_names),
        "omitted_context_names": list(spec.omitted_names),
        "omitted_context_indices": list(spec.omitted_indices),
        "retained_context_dim": spec.num_retained_features,
        "retained_context_names": list(spec.retained_names),
        "retained_context_indices": list(spec.retained_indices),
        "customer_stats_dim": spec.num_customer_stats,
        "customer_stats_names": list(spec.customer_stats_names),
        "ordered_input_blocks": [
            {"name": block_name, "dim": int(dim)}
            for block_name, dim in spec.ordered_input_blocks
        ],
    }


def build_input_contract(spec: FeatureAblationSpec) -> Dict[str, Any]:
    """Serialize the full ordered input contract with a deterministic digest."""
    body = _contract_body(spec)
    return {
        **body,
        "contract_digest": json_digest(body),
    }


def spec_from_input_contract(contract_dict: Mapping[str, Any]) -> FeatureAblationSpec:
    """Reconstruct and self-validate a FeatureAblationSpec from a persisted checkpoint contract."""
    required_keys = (
        "contract_version",
        "variant_name",
        "recipe",
        "input_dim",
        "embedding_dim",
        "source_context_dim",
        "source_context_names",
        "omitted_context_names",
        "omitted_context_indices",
        "retained_context_dim",
        "retained_context_names",
        "retained_context_indices",
        "customer_stats_dim",
        "customer_stats_names",
        "ordered_input_blocks",
        "contract_digest",
    )
    missing = [k for k in required_keys if k not in contract_dict]
    if missing:
        raise ValueError(f"Checkpoint input_contract is missing required fields: {missing}")

    spec = build_feature_ablation_spec(
        omitted_names=contract_dict["omitted_context_names"],
        name=str(contract_dict["variant_name"]),
        recipe=str(contract_dict["recipe"]),
        source_names=contract_dict["source_context_names"],
        customer_stats_names=contract_dict["customer_stats_names"],
        embedding_dim=int(contract_dict["embedding_dim"]),
        expected_omitted_indices=None,
    )
    expected = build_input_contract(spec)
    validate_input_contract(contract_dict, expected)
    return spec


def validate_input_contract(
    persisted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Strictly verify persisted checkpoint contract against expected contract."""
    if not isinstance(persisted, Mapping):
        raise ValueError("Checkpoint input_contract must be a mapping.")

    for key in (
        "contract_version",
        "variant_name",
        "recipe",
        "input_dim",
        "embedding_dim",
        "scalar_feature_count",
        "source_context_dim",
        "source_context_names",
        "omitted_context_names",
        "omitted_context_indices",
        "retained_context_dim",
        "retained_context_names",
        "retained_context_indices",
        "customer_stats_dim",
        "customer_stats_names",
        "ordered_input_blocks",
        "contract_digest",
    ):
        if key not in persisted:
            raise ValueError(f"Persisted input_contract missing required key {key!r}.")

    # Verify digest self-consistency first.
    body_without_digest = {k: v for k, v in persisted.items() if k != "contract_digest"}
    recomputed_digest = json_digest(body_without_digest)
    if persisted["contract_digest"] != recomputed_digest:
        raise ValueError(
            f"Persisted input_contract digest corrupted: stored={persisted['contract_digest']} != computed={recomputed_digest}."
        )

    for key in (
        "contract_version",
        "variant_name",
        "recipe",
        "input_dim",
        "embedding_dim",
        "scalar_feature_count",
        "source_context_dim",
        "retained_context_dim",
        "customer_stats_dim",
    ):
        if persisted[key] != expected[key]:
            raise ValueError(
                f"Input contract {key} mismatch: checkpoint={persisted[key]!r} vs expected={expected[key]!r}."
            )

    for key in (
        "source_context_names",
        "omitted_context_names",
        "omitted_context_indices",
        "retained_context_names",
        "retained_context_indices",
        "customer_stats_names",
        "ordered_input_blocks",
    ):
        if list(persisted[key]) != list(expected[key]):
            raise ValueError(
                f"Input contract {key} mismatch: checkpoint={persisted[key]!r} vs expected={expected[key]!r}."
            )

    if persisted["contract_digest"] != expected["contract_digest"]:
        raise ValueError(
            f"Input contract digest mismatch: checkpoint={persisted['contract_digest']} vs expected={expected['contract_digest']}."
        )


def select_retained_context_features(
    features_41: Union[torch.Tensor, np.ndarray],
    spec: FeatureAblationSpec,
) -> Union[torch.Tensor, np.ndarray]:
    """Slice canonical 41-column context tensor/array to the exact retained columns."""
    if features_41.shape[-1] != spec.source_num_features:
        raise ValueError(
            f"Expected canonical {spec.source_num_features} context columns, got shape {tuple(features_41.shape)}."
        )
    if isinstance(features_41, torch.Tensor):
        idx = torch.tensor(spec.retained_indices, dtype=torch.long, device=features_41.device)
        return torch.index_select(features_41, -1, idx)
    arr = np.asarray(features_41)
    return arr[..., list(spec.retained_indices)]


def describe_retained_features(spec: FeatureAblationSpec) -> List[Dict[str, Any]]:
    """Return ordered metadata table for all retained context features + customer stats."""
    rows: List[Dict[str, Any]] = []
    emb_offset = spec.embedding_dim * 2
    for new_idx, (src_idx, name) in enumerate(zip(spec.retained_indices, spec.retained_names)):
        rows.append(
            {
                "block": f"features_{spec.num_retained_features}",
                "source_context_index": int(src_idx),
                "ablated_context_index": int(new_idx),
                "mlp_column": int(emb_offset + new_idx),
                "name": name,
            }
        )
    stat_offset = emb_offset + spec.num_retained_features
    for s_idx, name in enumerate(spec.customer_stats_names):
        rows.append(
            {
                "block": f"customer_stats_{spec.num_customer_stats}",
                "source_context_index": None,
                "ablated_context_index": None,
                "mlp_column": int(stat_offset + s_idx),
                "name": name,
            }
        )
    return rows
