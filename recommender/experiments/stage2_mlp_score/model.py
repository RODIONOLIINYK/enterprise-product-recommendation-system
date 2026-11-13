"""Stage 2 UPS-CANR integration of frozen Stage 1 301D MLP score in attention and head."""

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.feature_schema import NAMES as DYNAMIC_FEATURE_NAMES
from recommender.experiments.mlp_feature_ablation.model import FeatureAblatedStage1Selector
from recommender.experiments.mlp_feature_ablation.schema import (
    FeatureAblationSpec,
    build_input_contract as build_mlp_input_contract,
    validate_input_contract as validate_mlp_input_contract,
)
from recommender.models import ProductTower
from recommender.ranker import RankerOutput, UPSCANRRanker
from recommender.shared_data import atomic_torch_save

from .config import SCORE_FEATURE_NAME, SCORE_TRANSFORM_VERSION


CHECKPOINT_FORMAT_VERSION = 1
EXPECTED_SOURCE_PAIRWISE_DIM = 41
EXPECTED_CORE_PAIRWISE_DIM = 13
EXPECTED_EXTRA_DYNAMIC_DIM = 28
EXPECTED_BASE_PAIRWISE_WITH_SCORE_DIM = 14
EXPECTED_BASE_ENCODED_DIM = 398
EXPECTED_HEAD_INPUT_DIM = 426
EXPECTED_ATTENTION_BIAS_SHAPE = (4, 15, 17)
EXPECTED_EXCLUDED_STATISTICS = ("history_repeat",)


def transform_teacher_logits_to_mlp_score(
    raw_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform teacher raw logits to [0, 1] float32 with pre-sigmoid masking of invalid positions."""
    if not isinstance(raw_logits, torch.Tensor):
        raise TypeError("raw_logits must be a torch.Tensor.")
    logits_f32 = raw_logits.to(dtype=torch.float32)
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean torch.Tensor.")
        if valid_mask.shape != logits_f32.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} != raw_logits shape {tuple(logits_f32.shape)}."
            )
        safe_logits = torch.where(valid_mask, logits_f32, logits_f32.new_zeros(()))
        if not bool(torch.isfinite(safe_logits).all()):
            raise ValueError("Teacher raw_logits contains NaN or Inf on valid candidates.")
        probs = torch.sigmoid(safe_logits).masked_fill(~valid_mask, 0.0)
    else:
        if not bool(torch.isfinite(logits_f32).all()):
            raise ValueError("Teacher raw_logits contains NaN or Inf.")
        probs = torch.sigmoid(logits_f32)
    if not bool(torch.isfinite(probs).all()) or not bool(((probs >= 0.0) & (probs <= 1.0)).all()):
        raise ValueError("Transformed mlp_score must be finite and within [0, 1].")
    return probs


def gather_teacher_scores_by_canonical_indices(
    full_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    canonical_product_ids: Sequence[str],
    score_product_ids: Optional[Sequence[str]] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather teacher scores strictly by canonical product indices (with SKU-order remap if needed)."""
    if full_scores.ndim != 2 or candidate_indices.ndim != 2:
        raise ValueError(
            f"Expected 2D full_scores and candidate_indices, got {tuple(full_scores.shape)} and {tuple(candidate_indices.shape)}."
        )
    if full_scores.shape[0] != candidate_indices.shape[0]:
        raise ValueError(
            f"Row count mismatch: full_scores={full_scores.shape[0]} vs candidate_indices={candidate_indices.shape[0]}."
        )
    if candidate_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"candidate_indices must be integer tensor, got {candidate_indices.dtype}.")

    canonical_list = list(canonical_product_ids)
    if len(set(canonical_list)) != len(canonical_list):
        raise ValueError("canonical_product_ids contains duplicates.")
    if full_scores.shape[1] != len(canonical_list):
        raise ValueError(
            f"full_scores column count {full_scores.shape[1]} != canonical_product_ids length {len(canonical_list)}."
        )

    aligned_scores = full_scores
    if score_product_ids is not None:
        score_list = list(score_product_ids)
        if len(set(score_list)) != len(score_list):
            raise ValueError("score_product_ids contains duplicates.")
        if len(score_list) != len(canonical_list) or set(score_list) != set(canonical_list):
            raise ValueError("score_product_ids does not match canonical_product_ids.")
        if score_list != canonical_list:
            pos = {pid: idx for idx, pid in enumerate(score_list)}
            remap = torch.tensor(
                [pos[pid] for pid in canonical_list],
                dtype=torch.long,
                device=full_scores.device,
            )
            aligned_scores = full_scores.index_select(1, remap)

    idx_long = candidate_indices.to(dtype=torch.long)
    if valid_mask is None:
        if bool((idx_long < 0).any()) or bool((idx_long >= len(canonical_list)).any()):
            raise ValueError("candidate_indices contains out-of-bounds product indices.")
        eff_mask = torch.ones_like(idx_long, dtype=torch.bool)
        safe_idx = idx_long
    else:
        if valid_mask.dtype != torch.bool or valid_mask.shape != idx_long.shape:
            raise ValueError("valid_mask must be boolean with same shape as candidate_indices.")
        eff_mask = valid_mask
        if bool(eff_mask.any()):
            valid_idx = idx_long[eff_mask]
            if bool((valid_idx < 0).any()) or bool((valid_idx >= len(canonical_list)).any()):
                raise ValueError("candidate_indices contains out-of-bounds indices on valid positions.")
        safe_idx = torch.where(eff_mask, idx_long, idx_long.new_zeros(()))

    raw_gathered = aligned_scores.gather(1, safe_idx)
    raw_masked = torch.where(eff_mask, raw_gathered.to(dtype=torch.float32), raw_gathered.new_zeros((), dtype=torch.float32))
    mlp_score = transform_teacher_logits_to_mlp_score(raw_masked, valid_mask=eff_mask)
    return mlp_score, raw_masked


def build_stage2_mlp_input_contract(
    *,
    pairwise_feature_names: Sequence[str] = DYNAMIC_FEATURE_NAMES,
    teacher_input_contract: Optional[Mapping[str, Any]] = None,
    score_feature_name: str = SCORE_FEATURE_NAME,
    score_transform_version: str = SCORE_TRANSFORM_VERSION,
) -> Dict[str, Any]:
    """Build deterministic ordered input contract for Stage 2 + MLP score ranker."""
    names_tuple = tuple(str(n) for n in pairwise_feature_names)
    if names_tuple != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError(
            "Ordered pairwise_feature_names do not match canonical 41-feature schema."
        )
    if score_transform_version != SCORE_TRANSFORM_VERSION:
        raise ValueError(
            f"Unsupported score_transform_version {score_transform_version!r}; expected {SCORE_TRANSFORM_VERSION!r}."
        )
    body = {
        "contract_version": 1,
        "model_class": "Stage2MLPScoreRanker",
        "pairwise_feature_count": EXPECTED_SOURCE_PAIRWISE_DIM,
        "ordered_pairwise_features": list(names_tuple),
        "core_pairwise_features": list(names_tuple[:EXPECTED_CORE_PAIRWISE_DIM]),
        "extra_dynamic_features": list(names_tuple[EXPECTED_CORE_PAIRWISE_DIM:]),
        "score_feature_name": str(score_feature_name),
        "score_transform_version": str(score_transform_version),
        "score_value_bounds": [0.0, 1.0],
        "attention_candidate_statistics": [
            "__intercept__",
            *names_tuple[:EXPECTED_CORE_PAIRWISE_DIM],
            str(score_feature_name),
        ],
        "attention_bias_shape": list(EXPECTED_ATTENTION_BIAS_SHAPE),
        "base_encoded_dim": EXPECTED_BASE_ENCODED_DIM,
        "head_input_dim": EXPECTED_HEAD_INPUT_DIM,
        "head_input_blocks": [
            {"name": "z_candidate", "dim": 128},
            {"name": "h_attended", "dim": 128},
            {"name": "interaction", "dim": 128},
            {"name": "core_pairwise_13", "dim": EXPECTED_CORE_PAIRWISE_DIM},
            {"name": str(score_feature_name), "dim": 1},
            {"name": "extra_dynamic_28", "dim": EXPECTED_EXTRA_DYNAMIC_DIM},
        ],
        "excluded_statistics": list(EXPECTED_EXCLUDED_STATISTICS),
        "teacher_input_contract_digest": (
            teacher_input_contract["contract_digest"]
            if teacher_input_contract is not None
            else None
        ),
    }
    return {
        **body,
        "contract_digest": json_digest(body),
    }


def validate_stage2_mlp_input_contract(
    persisted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Strictly verify persisted Stage 2 input contract against expected contract."""
    if not isinstance(persisted, Mapping):
        raise ValueError("Persisted input_contract must be a mapping.")
    if "contract_digest" not in persisted:
        raise ValueError("Persisted input_contract is missing 'contract_digest'.")
    body = {k: v for k, v in persisted.items() if k != "contract_digest"}
    recomputed = json_digest(body)
    if persisted["contract_digest"] != recomputed:
        raise ValueError(
            f"Stage 2 input_contract digest corrupted: {persisted['contract_digest']} != {recomputed}."
        )
    for key in expected:
        if key not in persisted:
            raise ValueError(f"Stage 2 input_contract missing key {key!r}.")
        if persisted[key] != expected[key]:
            raise ValueError(
                f"Stage 2 input_contract mismatch on {key!r}: {persisted[key]!r} != {expected[key]!r}."
            )


def _extract_upscanr_config(base: UPSCANRRanker) -> Dict[str, Any]:
    pt = base.product_tower
    pt_cfg: Optional[Dict[str, Any]] = None
    if pt is not None:
        # Extra MLP layer adds 4 modules beyond base 3 + final linear (= 8 total modules).
        extra_mlp = len(pt.mlp) > 4
        pt_cfg = {
            "text_dim": int(pt.text_proj[0].in_features),
            "hidden_dim": int(pt.text_proj[0].out_features),
            "output_dim": int(pt.output_dim),
            "num_brands": int(pt.brand_embedding.num_embeddings) if pt.brand_embedding is not None else 0,
            "num_categories": int(pt.cat_embedding.num_embeddings) if pt.cat_embedding is not None else 0,
            "num_subcategories": int(pt.subcat_embedding.num_embeddings) if pt.subcat_embedding is not None else 0,
            "num_business_lines": int(pt.line_embedding.num_embeddings) if pt.line_embedding is not None else 0,
            "num_demand_features": int(pt.num_demand_features),
            "num_containers": int(pt.container_embedding.num_embeddings) if pt.container_embedding is not None else 0,
            "one_hot_sizes": dict(pt.one_hot_sizes),
            "extra_mlp_layer": bool(extra_mlp),
        }
    return {
        "text_dim": int(base.text_dim),
        "memory_dim": int(base.memory_dim),
        "hidden_dim": int(base.hidden_dim),
        "num_heads": int(base.num_heads),
        "num_cities": int(base.city_embedding.num_embeddings),
        "num_regions": int(base.region_embedding.num_embeddings),
        "num_industries": int(base.industry_embedding.num_embeddings),
        "num_products": int(base.product_embedding.num_embeddings),
        "num_categories": int(base.cat_embedding.num_embeddings),
        "num_subcategories": int(base.subcat_embedding.num_embeddings),
        "num_business_lines": int(base.line_embedding.num_embeddings),
        "num_tabular_features": int(base.num_tabular_features),
        "seq_feature_dim": int(base.seq_feature_dim),
        "base_num_pairwise_features": int(base.num_pairwise_features),
        "excluded_statistics": list(base.excluded_statistics),
        "statistics_attention": base.statistics_attention_weights is not None,
        "cached_only": bool(base.cached_only),
        "has_catalog_text_embeddings": base.catalog_text_embeddings is not None,
        "catalog_text_num_rows": (
            int(base.catalog_text_embeddings.shape[0])
            if base.catalog_text_embeddings is not None
            else 0
        ),
        "product_tower": pt_cfg,
    }


class Stage2MLPScoreRanker(nn.Module):
    """UPS-CANR dynamic_features ranker conditioned on frozen Stage 1 MLP sigmoid score.

    The score channel is disabled by default following the user exclusion decision.
    Historical score-enabled checkpoints retain their explicit configuration.

    When explicitly enabled, integrates `mlp_score` (in [0, 1]) as:
    1. The 14th candidate pairwise statistic in cross-attention bilinear bias `W[4, 15, 17]`.
    2. The 398th encoded feature feeding the 426D dynamic scoring head (`384 + 14 + 28`).
    """

    def __init__(
        self,
        base: UPSCANRRanker,
        head: nn.Sequential,
        *,
        score_channel_enabled: bool = False,
        score_feature_name: str = SCORE_FEATURE_NAME,
        score_transform_version: str = SCORE_TRANSFORM_VERSION,
        ordered_pairwise_features: Sequence[str] = DYNAMIC_FEATURE_NAMES,
    ):
        super().__init__()
        if int(base.num_pairwise_features) != EXPECTED_BASE_PAIRWISE_WITH_SCORE_DIM:
            raise ValueError(
                f"Expected base.num_pairwise_features={EXPECTED_BASE_PAIRWISE_WITH_SCORE_DIM}, "
                f"got {base.num_pairwise_features}."
            )
        if base.statistics_attention_weights is None or tuple(base.statistics_attention_weights.shape) != EXPECTED_ATTENTION_BIAS_SHAPE:
            raise ValueError(
                f"Expected statistics_attention_weights shape {EXPECTED_ATTENTION_BIAS_SHAPE}, "
                f"got {tuple(base.statistics_attention_weights.shape) if base.statistics_attention_weights is not None else None}."
            )
        if tuple(base.excluded_statistics) != EXPECTED_EXCLUDED_STATISTICS:
            raise ValueError(
                f"Expected excluded_statistics={EXPECTED_EXCLUDED_STATISTICS}, got {base.excluded_statistics}."
            )
        if int(head[0].in_features) != EXPECTED_HEAD_INPUT_DIM:
            raise ValueError(
                f"Expected head[0].in_features={EXPECTED_HEAD_INPUT_DIM}, got {head[0].in_features}."
            )
        if tuple(ordered_pairwise_features) != tuple(DYNAMIC_FEATURE_NAMES):
            raise ValueError("ordered_pairwise_features must match canonical 41-feature schema.")

        self.base = base
        self.head = head
        self.base.configure_cached_training(True)
        # base.mlp is unused when dynamic head scores candidates; freeze to avoid dead trainable params.
        self.base.mlp.requires_grad_(False)
        self.score_channel_enabled = bool(score_channel_enabled)
        self.score_feature_name = str(score_feature_name)
        self.score_transform_version = str(score_transform_version)
        self.ordered_pairwise_features = tuple(ordered_pairwise_features)
        self.register_buffer(
            "extra_columns",
            torch.tensor(list(range(EXPECTED_CORE_PAIRWISE_DIM, EXPECTED_SOURCE_PAIRWISE_DIM)), dtype=torch.long),
            persistent=True,
        )

    @classmethod
    def from_dynamic_context_ranker(
        cls,
        dyn_ranker: CandidateContextRanker,
        *,
        score_channel_enabled: bool = False,
    ) -> "Stage2MLPScoreRanker":
        """Expand an existing 425D dynamic_features CandidateContextRanker with zero-init score weights."""
        if getattr(dyn_ranker, "mode", None) != "dynamic_features":
            raise ValueError(
                f"Expected CandidateContextRanker(mode='dynamic_features'), got mode={getattr(dyn_ranker, 'mode', None)!r}."
            )
        cloned = copy.deepcopy(dyn_ranker)
        base = cloned.base
        if int(base.num_pairwise_features) != EXPECTED_CORE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected base.num_pairwise_features={EXPECTED_CORE_PAIRWISE_DIM}, got {base.num_pairwise_features}."
            )
        old_w = base.statistics_attention_weights
        if old_w is None or tuple(old_w.shape) != (4, 14, 17):
            raise ValueError(
                f"Expected base statistics_attention_weights shape (4, 14, 17), got {tuple(old_w.shape) if old_w is not None else None}."
            )
        old_head = cloned.head
        old_head_first = old_head[0]
        old_fused = base.hidden_dim * 3 + EXPECTED_CORE_PAIRWISE_DIM  # 397
        if int(old_head_first.in_features) != old_fused + EXPECTED_EXTRA_DYNAMIC_DIM:  # 425
            raise ValueError(
                f"Expected dynamic_features head[0].in_features={old_fused + EXPECTED_EXTRA_DYNAMIC_DIM}, "
                f"got {old_head_first.in_features}."
            )

        # 1. Expand attention bias weights from (4, 14, 17) -> (4, 15, 17) with zero-init for score slot (idx 14).
        new_w = nn.Parameter(torch.zeros((4, 15, 17), dtype=old_w.dtype, device=old_w.device))
        with torch.no_grad():
            new_w[:, :14, :].copy_(old_w)
            new_w[:, 14, :].zero_()
        base.statistics_attention_weights = new_w
        base.num_pairwise_features = EXPECTED_BASE_PAIRWISE_WITH_SCORE_DIM

        # 2. Expand unused base.mlp[0] from 397 -> 398 for internal structural consistency.
        old_base_mlp_first = base.mlp[0]
        new_base_mlp_first = nn.Linear(
            EXPECTED_BASE_ENCODED_DIM,
            old_base_mlp_first.out_features,
            bias=old_base_mlp_first.bias is not None,
        ).to(device=old_base_mlp_first.weight.device, dtype=old_base_mlp_first.weight.dtype)
        with torch.no_grad():
            new_base_mlp_first.weight[:, :old_fused].copy_(old_base_mlp_first.weight)
            new_base_mlp_first.weight[:, old_fused].zero_()
            if old_base_mlp_first.bias is not None:
                new_base_mlp_first.bias.copy_(old_base_mlp_first.bias)
        base.mlp[0] = new_base_mlp_first

        # 3. Expand dynamic head[0] from 425 -> 426:
        #    [0..396] = old encoded (397D), [397] = mlp_score (zero-init), [398..425] = extra_28 (old 397..424).
        new_head = copy.deepcopy(old_head)
        new_head_first = nn.Linear(
            EXPECTED_HEAD_INPUT_DIM,
            old_head_first.out_features,
            bias=old_head_first.bias is not None,
        ).to(device=old_head_first.weight.device, dtype=old_head_first.weight.dtype)
        with torch.no_grad():
            new_head_first.weight[:, :old_fused].copy_(old_head_first.weight[:, :old_fused])
            new_head_first.weight[:, old_fused].zero_()
            new_head_first.weight[:, old_fused + 1 :].copy_(old_head_first.weight[:, old_fused:])
            if old_head_first.bias is not None:
                new_head_first.bias.copy_(old_head_first.bias)
        new_head[0] = new_head_first

        return cls(
            base,
            new_head,
            score_channel_enabled=score_channel_enabled,
        )

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "Stage2MLPScoreRanker":
        """Reconstruct architecture purely from persisted checkpoint model_config (no disk artifacts)."""
        ups_cfg = model_config["upscanr"]
        pt_cfg = ups_cfg.get("product_tower")
        product_tower = None
        if pt_cfg is not None:
            product_tower = ProductTower(
                text_dim=int(pt_cfg["text_dim"]),
                output_dim=int(pt_cfg["output_dim"]),
                num_brands=int(pt_cfg["num_brands"]),
                num_categories=int(pt_cfg["num_categories"]),
                num_subcategories=int(pt_cfg["num_subcategories"]),
                num_business_lines=int(pt_cfg["num_business_lines"]),
                num_demand_features=int(pt_cfg["num_demand_features"]),
                hidden_dim=int(pt_cfg["hidden_dim"]),
                num_containers=int(pt_cfg["num_containers"]),
                one_hot_sizes=pt_cfg.get("one_hot_sizes"),
                extra_mlp_layer=bool(pt_cfg["extra_mlp_layer"]),
            )
        cat_emb = (
            torch.zeros(int(ups_cfg["catalog_text_num_rows"]), int(ups_cfg["text_dim"]))
            if ups_cfg.get("has_catalog_text_embeddings")
            else None
        )
        base = UPSCANRRanker(
            text_dim=int(ups_cfg["text_dim"]),
            memory_dim=int(ups_cfg["memory_dim"]),
            num_cities=int(ups_cfg["num_cities"]),
            num_regions=int(ups_cfg["num_regions"]),
            num_industries=int(ups_cfg["num_industries"]),
            num_products=int(ups_cfg["num_products"]),
            num_categories=int(ups_cfg["num_categories"]),
            num_subcategories=int(ups_cfg["num_subcategories"]),
            num_business_lines=int(ups_cfg["num_business_lines"]),
            num_tabular_features=int(ups_cfg["num_tabular_features"]),
            num_pairwise_features=int(ups_cfg["base_num_pairwise_features"]),
            seq_feature_dim=int(ups_cfg["seq_feature_dim"]),
            hidden_dim=int(ups_cfg["hidden_dim"]),
            num_heads=int(ups_cfg["num_heads"]),
            catalog_text_embeddings=cat_emb,
            product_tower=product_tower,
            statistics_attention=bool(ups_cfg["statistics_attention"]),
            excluded_statistics=tuple(ups_cfg["excluded_statistics"]),
            cached_only=bool(ups_cfg["cached_only"]),
        )
        head = copy.deepcopy(base.mlp)
        head[0] = nn.Linear(int(model_config["head_input_dim"]), head[0].out_features)
        return cls(
            base,
            head,
            score_channel_enabled=bool(model_config["score_channel_enabled"]),
            score_feature_name=str(model_config["score_feature_name"]),
            score_transform_version=str(model_config["score_transform_version"]),
        )

    def clone_with_score_channel(self, score_channel_enabled: bool) -> "Stage2MLPScoreRanker":
        """Clone model weights identically for matched control/treatment initialization."""
        cloned = copy.deepcopy(self)
        cloned.score_channel_enabled = bool(score_channel_enabled)
        return cloned

    def model_config(self) -> Dict[str, Any]:
        return {
            "model_class": "Stage2MLPScoreRanker",
            "hidden_dim": int(self.base.hidden_dim),
            "num_heads": int(self.base.num_heads),
            "base_num_pairwise_features": int(self.base.num_pairwise_features),
            "base_encoded_dim": EXPECTED_BASE_ENCODED_DIM,
            "extra_dynamic_dim": EXPECTED_EXTRA_DYNAMIC_DIM,
            "head_input_dim": int(self.head[0].in_features),
            "attention_bias_shape": list(self.base.statistics_attention_weights.shape),
            "excluded_statistics": list(self.base.excluded_statistics),
            "score_channel_enabled": bool(self.score_channel_enabled),
            "score_feature_name": self.score_feature_name,
            "score_transform_version": self.score_transform_version,
            "ordered_pairwise_features": list(self.ordered_pairwise_features),
            "upscanr": _extract_upscanr_config(self.base),
        }

    def input_contract(self, *, teacher_input_contract: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return build_stage2_mlp_input_contract(
            pairwise_feature_names=self.ordered_pairwise_features,
            teacher_input_contract=teacher_input_contract,
            score_feature_name=self.score_feature_name,
            score_transform_version=self.score_transform_version,
        )

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        mlp_score: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        cached_customer_vectors: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RankerOutput:
        if pairwise_features.ndim != 3 or pairwise_features.shape[-1] != EXPECTED_SOURCE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected pairwise_features shape (B, C, {EXPECTED_SOURCE_PAIRWISE_DIM}), "
                f"got {tuple(pairwise_features.shape)}."
            )
        b_size, num_c, _ = pairwise_features.shape

        if candidate_mask is not None:
            if candidate_mask.dtype != torch.bool or candidate_mask.shape != (b_size, num_c):
                raise ValueError(
                    f"Expected boolean candidate_mask shape ({b_size}, {num_c}), got {tuple(candidate_mask.shape)} ({candidate_mask.dtype})."
                )
            pf = torch.where(candidate_mask.unsqueeze(-1), pairwise_features, pairwise_features.new_zeros(()))
            cand_vecs = kwargs.get("candidate_vectors")
            if cand_vecs is not None:
                kwargs["candidate_vectors"] = torch.where(
                    candidate_mask.unsqueeze(-1), cand_vecs, cand_vecs.new_zeros(())
                )
            cand_idx = kwargs.get("candidate_indices")
            if cand_idx is not None:
                kwargs["candidate_indices"] = torch.where(
                    candidate_mask, cand_idx, cand_idx.new_zeros(())
                )
        else:
            pf = pairwise_features

        if not bool(torch.isfinite(pf).all()):
            raise ValueError("pairwise_features contains NaN or Inf on valid candidates.")

        if self.score_channel_enabled and mlp_score is None:
            raise ValueError("mlp_score is required when score_channel_enabled=True.")

        if mlp_score is not None:
            if mlp_score.ndim != 2 or mlp_score.shape != (b_size, num_c):
                raise ValueError(
                    f"Expected mlp_score shape ({b_size}, {num_c}), got {tuple(mlp_score.shape)}."
                )
            score_f32 = mlp_score.to(dtype=pf.dtype)
            if candidate_mask is not None:
                score_f32 = torch.where(candidate_mask, score_f32, score_f32.new_zeros(()))
            if not bool(torch.isfinite(score_f32).all()):
                raise ValueError("mlp_score contains NaN or Inf on valid candidates.")
            if not bool(((score_f32 >= 0.0) & (score_f32 <= 1.0)).all()):
                raise ValueError("mlp_score must be bounded within [0, 1] on valid candidates.")
            effective_score = score_f32 if self.score_channel_enabled else pf.new_zeros((b_size, num_c))
        else:
            effective_score = pf.new_zeros((b_size, num_c))

        pf_14 = torch.cat([pf[..., :EXPECTED_CORE_PAIRWISE_DIM], effective_score.unsqueeze(-1)], dim=-1)
        encoded = self.base.encode_candidate_features(pairwise_features=pf_14, **kwargs)
        if encoded.features.shape[-1] != EXPECTED_BASE_ENCODED_DIM:
            raise AssertionError(
                f"Expected base encoded width {EXPECTED_BASE_ENCODED_DIM}, got {encoded.features.shape[-1]}."
            )
        extra_28 = pf[..., self.extra_columns]
        head_in = torch.cat([encoded.features, extra_28], dim=-1)
        if head_in.shape[-1] != EXPECTED_HEAD_INPUT_DIM:
            raise AssertionError(
                f"Expected head input width {EXPECTED_HEAD_INPUT_DIM}, got {head_in.shape[-1]}."
            )
        logits = self.head(head_in).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)
        return encoded.to_output(logits)


def save_stage2_mlp_score_checkpoint(
    path: Union[Path, str],
    model: Stage2MLPScoreRanker,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    selection_criterion: str,
    provenance: Mapping[str, Any],
    teacher_input_contract: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist Stage2MLPScoreRanker checkpoint with strict contract and provenance."""
    out_path = Path(path)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "Stage2MLPScoreRanker",
        "model_module": "recommender.experiments.stage2_mlp_score.model",
        "model_config": model.model_config(),
        "input_contract": model.input_contract(teacher_input_contract=teacher_input_contract),
        "provenance": dict(provenance),
        "epoch": int(epoch),
        "selection_criterion": str(selection_criterion),
        "metrics": dict(metrics),
        "settings": dict(settings) if settings is not None else {},
        "model_state_dict": model.state_dict(),
    }
    atomic_torch_save(payload, out_path)
    return out_path


def load_stage2_mlp_score_checkpoint(
    checkpoint_path: Union[Path, str],
    *,
    expected_provenance: Optional[Mapping[str, Any]] = None,
    expected_teacher_input_contract: Optional[Mapping[str, Any]] = None,
    expected_score_channel_enabled: Optional[bool] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[Stage2MLPScoreRanker, Dict[str, Any]]:
    """Strictly validate and load a Stage2MLPScoreRanker checkpoint."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}.")
    if ckpt.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_format_version={ckpt.get('checkpoint_format_version')!r}."
        )
    if ckpt.get("model_class") != "Stage2MLPScoreRanker":
        raise ValueError(f"Unexpected model_class={ckpt.get('model_class')!r}.")
    for req_key in ("model_config", "input_contract", "provenance", "model_state_dict"):
        if req_key not in ckpt:
            raise ValueError(f"Checkpoint {checkpoint_path} missing required field {req_key!r}.")

    model_cfg = ckpt["model_config"]
    if model_cfg.get("model_class") != "Stage2MLPScoreRanker":
        raise ValueError("model_config.model_class mismatch.")
    if model_cfg.get("score_transform_version") != SCORE_TRANSFORM_VERSION:
        raise ValueError(
            f"Score transform mismatch: {model_cfg.get('score_transform_version')!r} != {SCORE_TRANSFORM_VERSION!r}."
        )
    if tuple(model_cfg.get("ordered_pairwise_features", ())) != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError("Checkpoint ordered_pairwise_features mismatch.")
    if int(model_cfg.get("base_encoded_dim", -1)) != EXPECTED_BASE_ENCODED_DIM:
        raise ValueError(f"Checkpoint base_encoded_dim mismatch: {model_cfg.get('base_encoded_dim')}.")
    if int(model_cfg.get("head_input_dim", -1)) != EXPECTED_HEAD_INPUT_DIM:
        raise ValueError(f"Checkpoint head_input_dim mismatch: {model_cfg.get('head_input_dim')}.")
    if tuple(model_cfg.get("attention_bias_shape", ())) != EXPECTED_ATTENTION_BIAS_SHAPE:
        raise ValueError(
            f"Checkpoint attention_bias_shape mismatch: {model_cfg.get('attention_bias_shape')}."
        )
    if expected_score_channel_enabled is not None and bool(model_cfg.get("score_channel_enabled")) != bool(expected_score_channel_enabled):
        raise ValueError(
            f"score_channel_enabled mismatch: {model_cfg.get('score_channel_enabled')} != {expected_score_channel_enabled}."
        )

    teacher_contract_for_expected = expected_teacher_input_contract
    expected_contract = build_stage2_mlp_input_contract(
        teacher_input_contract=teacher_contract_for_expected,
    )
    persisted_contract = ckpt["input_contract"]
    if teacher_contract_for_expected is None:
        # Still verify contract digest self-consistency and matching fields ignoring teacher digest if omitted.
        expected_contract["teacher_input_contract_digest"] = persisted_contract.get("teacher_input_contract_digest")
        body = {k: v for k, v in expected_contract.items() if k != "contract_digest"}
        expected_contract["contract_digest"] = json_digest(body)
    validate_stage2_mlp_input_contract(persisted_contract, expected_contract)

    persisted_prov = ckpt["provenance"]
    if persisted_prov.get("score_feature_transform") != SCORE_TRANSFORM_VERSION:
        raise ValueError(
            f"Provenance score_feature_transform mismatch: {persisted_prov.get('score_feature_transform')!r}."
        )
    if expected_provenance is not None:
        for key, exp_val in expected_provenance.items():
            if persisted_prov.get(key) != exp_val:
                raise ValueError(
                    f"Checkpoint provenance mismatch on {key!r}: {persisted_prov.get(key)!r} != {exp_val!r}."
                )

    state = ckpt["model_state_dict"]
    if "base.statistics_attention_weights" not in state or tuple(state["base.statistics_attention_weights"].shape) != EXPECTED_ATTENTION_BIAS_SHAPE:
        raise ValueError("Checkpoint state_dict base.statistics_attention_weights shape mismatch.")
    if "head.0.weight" not in state or int(state["head.0.weight"].shape[1]) != EXPECTED_HEAD_INPUT_DIM:
        raise ValueError("Checkpoint state_dict head.0.weight width mismatch.")
    if "base.mlp.0.weight" not in state or int(state["base.mlp.0.weight"].shape[1]) != EXPECTED_BASE_ENCODED_DIM:
        raise ValueError("Checkpoint state_dict base.mlp.0.weight width mismatch.")

    model = Stage2MLPScoreRanker.from_model_config(model_cfg)
    model.load_state_dict(state, strict=True)
    return model, dict(ckpt)


class Stage2MLPScoreInferenceAdapter:
    """Frozen Stage 1 301D MLP + Stage 2 ranker inference adapter with cache/request parity."""

    def __init__(
        self,
        teacher: FeatureAblatedStage1Selector,
        ranker: Stage2MLPScoreRanker,
        *,
        expected_teacher_spec: Optional[FeatureAblationSpec] = None,
    ):
        if not isinstance(teacher, FeatureAblatedStage1Selector):
            raise TypeError("teacher must be a FeatureAblatedStage1Selector.")
        if not isinstance(ranker, Stage2MLPScoreRanker):
            raise TypeError("ranker must be a Stage2MLPScoreRanker.")
        if expected_teacher_spec is not None:
            validate_mlp_input_contract(
                teacher.input_contract(),
                build_mlp_input_contract(expected_teacher_spec),
            )
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.ranker = ranker
        self.ranker.eval()
        self.ranker.requires_grad_(False)

    @torch.no_grad()
    def compute_mlp_score(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        features_41: torch.Tensor,
        customer_stats_6: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-candidate float32 sigmoid(teacher_logit) strictly from frozen u/v/41-feat/6-stats."""
        self.teacher.eval()
        if u.ndim != 2 or u.shape[-1] != self.teacher.embedding_dim:
            raise ValueError(f"Expected u shape (B, {self.teacher.embedding_dim}), got {tuple(u.shape)}.")
        if customer_stats_6.ndim != 2 or customer_stats_6.shape[-1] != self.teacher.num_customer_stats:
            raise ValueError(
                f"Expected customer_stats_6 shape (B, {self.teacher.num_customer_stats}), got {tuple(customer_stats_6.shape)}."
            )
        if features_41.ndim != 3 or features_41.shape[-1] != EXPECTED_SOURCE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected features_41 shape (B, N, {EXPECTED_SOURCE_PAIRWISE_DIM}), got {tuple(features_41.shape)}."
            )

        b_size = u.shape[0]
        if candidate_indices is not None:
            if candidate_indices.ndim != 2 or candidate_indices.shape[0] != b_size:
                raise ValueError(
                    f"Expected candidate_indices shape ({b_size}, C), got {tuple(candidate_indices.shape)}."
                )
            num_c = candidate_indices.shape[1]
            if candidate_mask is not None:
                if candidate_mask.dtype != torch.bool or candidate_mask.shape != (b_size, num_c):
                    raise ValueError(
                        f"Expected boolean candidate_mask shape ({b_size}, {num_c}), got {tuple(candidate_mask.shape)}."
                    )
                safe_idx = torch.where(candidate_mask, candidate_indices.long(), candidate_indices.new_zeros((), dtype=torch.long))
            else:
                if bool((candidate_indices < 0).any()):
                    raise ValueError("candidate_indices contains negative index without candidate_mask.")
                safe_idx = candidate_indices.long()

            if v.ndim == 2:
                if bool((safe_idx >= v.shape[0]).any()):
                    raise ValueError("candidate_indices out of bounds for v.")
                cand_v = v[safe_idx]
            elif v.ndim == 3 and v.shape == (b_size, num_c, self.teacher.embedding_dim):
                cand_v = v
            else:
                raise ValueError(f"Incompatible v shape {tuple(v.shape)} with candidate_indices.")

            if features_41.shape[1] == num_c:
                cand_f41 = features_41
            else:
                if v.ndim != 2 or features_41.shape[1] != v.shape[0]:
                    raise ValueError(
                        f"features_41 shape {tuple(features_41.shape)} incompatible with slate size {num_c} or catalog {v.shape[0]}."
                    )
                row_idx = torch.arange(b_size, device=features_41.device)[:, None]
                cand_f41 = features_41[row_idx, safe_idx]
        else:
            cand_f41 = features_41
            cand_v = v
            num_c = cand_f41.shape[1]
            if candidate_mask is not None and (candidate_mask.dtype != torch.bool or candidate_mask.shape != (b_size, num_c)):
                raise ValueError(f"Expected boolean candidate_mask shape ({b_size}, {num_c}).")

        if candidate_mask is not None:
            cand_f41 = torch.where(candidate_mask.unsqueeze(-1), cand_f41, cand_f41.new_zeros(()))
            if cand_v.ndim == 3:
                cand_v = torch.where(candidate_mask.unsqueeze(-1), cand_v, cand_v.new_zeros(()))

        raw_logits = self.teacher(
            u=u,
            v=cand_v,
            features_41=cand_f41,
            customer_stats_6=customer_stats_6,
        )
        return transform_teacher_logits_to_mlp_score(raw_logits, valid_mask=candidate_mask)

    @torch.no_grad()
    def score_candidates(
        self,
        ranker_inputs: Mapping[str, Any],
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        customer_stats_6: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> RankerOutput:
        """Score a candidate slate end-to-end using frozen teacher MLP + Stage 2 ranker."""
        self.ranker.eval()
        features_41 = ranker_inputs["pairwise_features"]
        mlp_score = self.compute_mlp_score(
            u=u,
            v=v,
            features_41=features_41,
            customer_stats_6=customer_stats_6,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
        )
        call_kwargs = dict(ranker_inputs)
        call_kwargs.pop("mlp_score", None)
        call_kwargs.pop("candidate_mask", None)
        if "candidate_vectors" not in call_kwargs or call_kwargs["candidate_vectors"] is None:
            if candidate_indices is not None and v.ndim == 2:
                safe_idx = (
                    torch.where(candidate_mask, candidate_indices.long(), 0)
                    if candidate_mask is not None
                    else candidate_indices.long()
                )
                call_kwargs["candidate_vectors"] = v[safe_idx]
        return self.ranker(
            **call_kwargs,
            mlp_score=mlp_score,
            candidate_mask=candidate_mask,
        )
