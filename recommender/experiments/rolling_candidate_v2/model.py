"""Opt-in masked unordered candidate-set self-attention ranker over rolling 41D features."""

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.feature_schema import (
    NAMES as DYNAMIC_FEATURE_NAMES,
    STATISTICS_POLICY_ROLLING,
    rolling_schema_digest,
)
from recommender.models import ProductTower
from recommender.ranker import RankerOutput, UPSCANRRanker
from recommender.shared_data import atomic_torch_save


CHECKPOINT_FORMAT_VERSION = 2
EXPECTED_SOURCE_PAIRWISE_DIM = 41
EXPECTED_CORE_PAIRWISE_DIM = 13
EXPECTED_EXTRA_DYNAMIC_DIM = 28
EXPECTED_BASE_ENCODED_DIM = 397
EXPECTED_HEAD_INPUT_DIM = 425
EXPECTED_ATTENTION_BIAS_SHAPE = (4, 14, 17)
EXPECTED_EXCLUDED_STATISTICS = ("history_repeat",)


def build_rolling_ranker_input_contract(
    *,
    pairwise_feature_names: Sequence[str] = DYNAMIC_FEATURE_NAMES,
    statistics_policy: str = STATISTICS_POLICY_ROLLING,
    feature_contract_digest: Optional[str] = None,
    candidate_attention_enabled: bool = False,
    attention_num_heads: int = 4,
    attention_dim_feedforward: int = 256,
    attention_dropout: float = 0.1,
) -> Dict[str, Any]:
    """Build deterministic ordered input contract for RollingCandidateAttentionRanker."""
    names_tuple = tuple(str(n) for n in pairwise_feature_names)
    if names_tuple != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError("Ordered pairwise_feature_names do not match canonical 41-feature schema.")
    if statistics_policy != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"New candidate-attention ranker requires statistics_policy={STATISTICS_POLICY_ROLLING!r}, "
            f"got {statistics_policy!r}."
        )
    schema_dig = feature_contract_digest or rolling_schema_digest()
    body = {
        "contract_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "RollingCandidateAttentionRanker",
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_contract_digest": str(schema_dig),
        "teacher_score_included": False,
        "pairwise_feature_count": EXPECTED_SOURCE_PAIRWISE_DIM,
        "ordered_pairwise_features": list(names_tuple),
        "core_pairwise_features": list(names_tuple[:EXPECTED_CORE_PAIRWISE_DIM]),
        "extra_dynamic_features": list(names_tuple[EXPECTED_CORE_PAIRWISE_DIM:]),
        "base_num_pairwise_features": EXPECTED_CORE_PAIRWISE_DIM,
        "attention_bias_shape": list(EXPECTED_ATTENTION_BIAS_SHAPE),
        "base_encoded_dim": EXPECTED_BASE_ENCODED_DIM,
        "head_input_dim": EXPECTED_HEAD_INPUT_DIM,
        "excluded_statistics": list(EXPECTED_EXCLUDED_STATISTICS),
        "candidate_attention_enabled": bool(candidate_attention_enabled),
        "candidate_set_positional_encoding": False,
        "residual_zero_initialized": True,
        "attention_num_heads": int(attention_num_heads),
        "attention_dim_feedforward": int(attention_dim_feedforward),
        "attention_dropout": float(attention_dropout),
    }
    return {**body, "contract_digest": json_digest(body)}


def validate_rolling_ranker_input_contract(
    persisted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Strictly verify persisted input contract digest and fields."""
    if not isinstance(persisted, Mapping):
        raise ValueError("Persisted input_contract must be a mapping.")
    if "contract_digest" not in persisted:
        raise ValueError("Persisted input_contract missing 'contract_digest'.")
    body = {k: v for k, v in persisted.items() if k != "contract_digest"}
    recomputed = json_digest(body)
    if persisted["contract_digest"] != recomputed:
        raise ValueError(
            f"Rolling ranker input_contract digest corrupted: {persisted['contract_digest']} != {recomputed}."
        )
    for key, exp_val in expected.items():
        if key not in persisted:
            raise ValueError(f"Rolling ranker input_contract missing key {key!r}.")
        if persisted[key] != exp_val:
            raise ValueError(
                f"Rolling ranker input_contract mismatch on {key!r}: {persisted[key]!r} != {exp_val!r}."
            )


def _extract_upscanr_config(base: UPSCANRRanker) -> Dict[str, Any]:
    pt = base.product_tower
    pt_cfg: Optional[Dict[str, Any]] = None
    if pt is not None:
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


class RollingCandidateAttentionRanker(nn.Module):
    """425D dynamic UPS-CANR pointwise scorer + opt-in zero-init residual candidate self-attention.

    Architecture:
    1. Unchanged UPS-CANR base cross-attention over customer memory using 13 core pairwise stats
       (`W[4, 14, 17]`, `excluded_statistics=('history_repeat',)`), producing 397D encoded features.
    2. Unchanged 425D pointwise scorer `base_head` over `[encoded_397, extra_28]` -> `base_logits`
       (never overwritten or expanded; no Stage 1 MLP score channel).
    3. Opt-in zero-initialized masked unordered candidate-set residual branch:
       `425D -> Linear(425, width) -> TransformerEncoderLayer(batch_first=True, norm_first=True)
        -> residual_head(cat([proj, context]) -> 128 -> 1)` with zero-initialized final layer.
       At initialization, `logits == base_logits` with exact parity (`atol=0, rtol=0`),
       and candidate set order never affects outputs (permutation equivariant; no positional encodings).
    """

    def __init__(
        self,
        base: UPSCANRRanker,
        base_head: nn.Sequential,
        *,
        candidate_attention_enabled: bool = False,
        attention_num_heads: int = 4,
        attention_dim_feedforward: int = 256,
        attention_dropout: float = 0.1,
        ordered_pairwise_features: Sequence[str] = DYNAMIC_FEATURE_NAMES,
        statistics_policy: str = STATISTICS_POLICY_ROLLING,
    ):
        super().__init__()
        if int(base.num_pairwise_features) != EXPECTED_CORE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected base.num_pairwise_features={EXPECTED_CORE_PAIRWISE_DIM}, got {base.num_pairwise_features}."
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
        if int(base_head[0].in_features) != EXPECTED_HEAD_INPUT_DIM:
            raise ValueError(
                f"Expected base_head[0].in_features={EXPECTED_HEAD_INPUT_DIM}, got {base_head[0].in_features}."
            )
        if tuple(ordered_pairwise_features) != tuple(DYNAMIC_FEATURE_NAMES):
            raise ValueError("ordered_pairwise_features must match canonical 41-feature schema.")
        if statistics_policy != STATISTICS_POLICY_ROLLING:
            raise ValueError(
                f"RollingCandidateAttentionRanker requires statistics_policy={STATISTICS_POLICY_ROLLING!r}."
            )

        self.base = base
        self.base_head = base_head
        self.base.configure_cached_training(True)
        # base.mlp is unused when 425D base_head scores candidates; freeze to avoid dead trainable params.
        self.base.mlp.requires_grad_(False)

        width = int(base.hidden_dim)
        self.width = width
        self.attention_num_heads = int(attention_num_heads)
        self.attention_dim_feedforward = int(attention_dim_feedforward)
        self.attention_dropout = float(attention_dropout)
        self.ordered_pairwise_features = tuple(ordered_pairwise_features)
        self.statistics_policy = str(statistics_policy)
        self.candidate_attention_enabled = bool(candidate_attention_enabled)

        self.context_project = nn.Linear(EXPECTED_HEAD_INPUT_DIM, width)
        self.candidate_attention = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.attention_num_heads,
            dim_feedforward=self.attention_dim_feedforward,
            dropout=self.attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(width * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

        self._set_residual_branch_grad(self.candidate_attention_enabled)
        self.register_buffer(
            "extra_columns",
            torch.tensor(list(range(EXPECTED_CORE_PAIRWISE_DIM, EXPECTED_SOURCE_PAIRWISE_DIM)), dtype=torch.long),
            persistent=True,
        )

    def _set_residual_branch_grad(self, enabled: bool) -> None:
        for mod in (self.context_project, self.candidate_attention, self.residual_head):
            mod.requires_grad_(bool(enabled))

    @classmethod
    def from_dynamic_context_ranker(
        cls,
        dyn_ranker: CandidateContextRanker,
        *,
        candidate_attention_enabled: bool = False,
        attention_num_heads: int = 4,
        attention_dim_feedforward: int = 256,
        attention_dropout: float = 0.1,
    ) -> "RollingCandidateAttentionRanker":
        """Wrap an existing 425D dynamic_features CandidateContextRanker preserving base_head verbatim."""
        if getattr(dyn_ranker, "mode", None) != "dynamic_features":
            raise ValueError(
                f"Expected CandidateContextRanker(mode='dynamic_features'), got mode={getattr(dyn_ranker, 'mode', None)!r}."
            )
        cloned = copy.deepcopy(dyn_ranker)
        return cls(
            cloned.base,
            cloned.head,
            candidate_attention_enabled=candidate_attention_enabled,
            attention_num_heads=attention_num_heads,
            attention_dim_feedforward=attention_dim_feedforward,
            attention_dropout=attention_dropout,
        )

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "RollingCandidateAttentionRanker":
        """Reconstruct architecture purely from persisted checkpoint model_config."""
        if model_config.get("teacher_score_included") is not False:
            raise ValueError("RollingCandidateAttentionRanker strictly forbids teacher_score_included=True.")
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
        base_head = copy.deepcopy(base.mlp)
        base_head[0] = nn.Linear(int(model_config["head_input_dim"]), base_head[0].out_features)
        return cls(
            base,
            base_head,
            candidate_attention_enabled=bool(model_config["candidate_attention_enabled"]),
            attention_num_heads=int(model_config.get("attention_num_heads", 4)),
            attention_dim_feedforward=int(model_config.get("attention_dim_feedforward", 256)),
            attention_dropout=float(model_config.get("attention_dropout", 0.1)),
            ordered_pairwise_features=tuple(model_config["ordered_pairwise_features"]),
            statistics_policy=str(model_config["statistics_policy"]),
        )

    def clone_with_candidate_attention(self, candidate_attention_enabled: bool) -> "RollingCandidateAttentionRanker":
        """Clone weights identically for matched control (`False`) vs treatment (`True`) runs."""
        cloned = copy.deepcopy(self)
        cloned.candidate_attention_enabled = bool(candidate_attention_enabled)
        cloned._set_residual_branch_grad(bool(candidate_attention_enabled))
        return cloned

    def model_config(self) -> Dict[str, Any]:
        return {
            "model_class": "RollingCandidateAttentionRanker",
            "statistics_policy": self.statistics_policy,
            "teacher_score_included": False,
            "hidden_dim": int(self.base.hidden_dim),
            "num_heads": int(self.base.num_heads),
            "base_num_pairwise_features": int(self.base.num_pairwise_features),
            "base_encoded_dim": EXPECTED_BASE_ENCODED_DIM,
            "extra_dynamic_dim": EXPECTED_EXTRA_DYNAMIC_DIM,
            "head_input_dim": int(self.base_head[0].in_features),
            "attention_bias_shape": list(self.base.statistics_attention_weights.shape),
            "excluded_statistics": list(self.base.excluded_statistics),
            "candidate_attention_enabled": bool(self.candidate_attention_enabled),
            "candidate_set_positional_encoding": False,
            "residual_zero_initialized": True,
            "attention_num_heads": self.attention_num_heads,
            "attention_dim_feedforward": self.attention_dim_feedforward,
            "attention_dropout": self.attention_dropout,
            "ordered_pairwise_features": list(self.ordered_pairwise_features),
            "upscanr": _extract_upscanr_config(self.base),
        }

    def input_contract(self, *, feature_contract_digest: Optional[str] = None) -> Dict[str, Any]:
        return build_rolling_ranker_input_contract(
            pairwise_feature_names=self.ordered_pairwise_features,
            statistics_policy=self.statistics_policy,
            feature_contract_digest=feature_contract_digest,
            candidate_attention_enabled=self.candidate_attention_enabled,
            attention_num_heads=self.attention_num_heads,
            attention_dim_feedforward=self.attention_dim_feedforward,
            attention_dropout=self.attention_dropout,
        )

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        cached_customer_vectors: Optional[torch.Tensor] = None,
        disable_candidate_attention: bool = False,
        **kwargs,
    ) -> RankerOutput:
        if "mlp_score" in kwargs and kwargs["mlp_score"] is not None:
            raise ValueError("RollingCandidateAttentionRanker forbids Stage 1 mlp_score input.")
        kwargs.pop("mlp_score", None)

        if pairwise_features.ndim != 3 or pairwise_features.shape[-1] != EXPECTED_SOURCE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected pairwise_features shape (B, C, {EXPECTED_SOURCE_PAIRWISE_DIM}), "
                f"got {tuple(pairwise_features.shape)}."
            )
        b_size, num_c, _ = pairwise_features.shape

        if candidate_mask is not None:
            if candidate_mask.dtype != torch.bool or candidate_mask.shape != (b_size, num_c):
                raise ValueError(
                    f"Expected boolean candidate_mask shape ({b_size}, {num_c}), "
                    f"got {tuple(candidate_mask.shape)} ({candidate_mask.dtype})."
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

        encoded = self.base.encode_candidate_features(
            pairwise_features=pf[..., :EXPECTED_CORE_PAIRWISE_DIM],
            **kwargs,
        )
        if encoded.features.shape[-1] != EXPECTED_BASE_ENCODED_DIM:
            raise AssertionError(
                f"Expected base encoded width {EXPECTED_BASE_ENCODED_DIM}, got {encoded.features.shape[-1]}."
            )

        base_head_in = torch.cat([encoded.features, pf[..., self.extra_columns]], dim=-1)
        if base_head_in.shape[-1] != EXPECTED_HEAD_INPUT_DIM:
            raise AssertionError(
                f"Expected head input width {EXPECTED_HEAD_INPUT_DIM}, got {base_head_in.shape[-1]}."
            )
        if candidate_mask is not None:
            base_head_in = torch.where(candidate_mask.unsqueeze(-1), base_head_in, base_head_in.new_zeros(()))

        base_logits = self.base_head(base_head_in).squeeze(-1)

        if self.candidate_attention_enabled:
            projected = self.context_project(base_head_in)
            if candidate_mask is not None:
                projected = torch.where(candidate_mask.unsqueeze(-1), projected, projected.new_zeros(()))
            if disable_candidate_attention:
                layer = self.candidate_attention
                context = projected + layer.dropout2(
                    layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(projected)))))
                )
            else:
                if candidate_mask is not None:
                    has_valid = candidate_mask.any(dim=1)
                    pad_mask = torch.where(
                        has_valid.unsqueeze(1),
                        ~candidate_mask,
                        torch.zeros_like(candidate_mask),
                    )
                else:
                    pad_mask = None
                context = self.candidate_attention(projected, src_key_padding_mask=pad_mask)
                if candidate_mask is not None:
                    context = torch.where(candidate_mask.unsqueeze(-1), context, context.new_zeros(()))
            residual = self.residual_head(torch.cat([projected, context], dim=-1)).squeeze(-1)
            if candidate_mask is not None:
                residual = torch.where(candidate_mask, residual, residual.new_zeros(()))
            logits = base_logits + residual
        else:
            logits = base_logits

        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)
        return encoded.to_output(logits)

    def forward_scores(self, **kwargs) -> torch.Tensor:
        """Convenience inference helper returning ranker logits directly."""
        return self.forward(**kwargs).logits


def save_rolling_ranker_checkpoint(
    path: Union[Path, str],
    model: RollingCandidateAttentionRanker,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    selection_criterion: str,
    provenance: Mapping[str, Any],
    feature_contract_digest: str,
    settings: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist RollingCandidateAttentionRanker checkpoint with strict contract and provenance."""
    out_path = Path(path)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "RollingCandidateAttentionRanker",
        "model_module": "recommender.experiments.rolling_candidate_v2.model",
        "model_config": model.model_config(),
        "input_contract": model.input_contract(feature_contract_digest=feature_contract_digest),
        "provenance": dict(provenance),
        "epoch": int(epoch),
        "selection_criterion": str(selection_criterion),
        "metrics": dict(metrics),
        "settings": dict(settings) if settings is not None else {},
        "model_state_dict": model.state_dict(),
    }
    atomic_torch_save(payload, out_path)
    return out_path


def load_rolling_ranker_checkpoint(
    checkpoint_path: Union[Path, str],
    *,
    expected_provenance: Optional[Mapping[str, Any]] = None,
    expected_feature_contract_digest: Optional[str] = None,
    expected_candidate_attention_enabled: Optional[bool] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[RollingCandidateAttentionRanker, Dict[str, Any]]:
    """Strictly validate and load a RollingCandidateAttentionRanker checkpoint."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}.")
    if ckpt.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_format_version={ckpt.get('checkpoint_format_version')!r}."
        )
    if ckpt.get("model_class") != "RollingCandidateAttentionRanker":
        raise ValueError(f"Unexpected model_class={ckpt.get('model_class')!r}.")
    for req_key in ("model_config", "input_contract", "provenance", "model_state_dict"):
        if req_key not in ckpt:
            raise ValueError(f"Checkpoint {checkpoint_path} missing required field {req_key!r}.")

    model_cfg = ckpt["model_config"]
    if model_cfg.get("model_class") != "RollingCandidateAttentionRanker":
        raise ValueError("model_config.model_class mismatch.")
    if model_cfg.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"Checkpoint statistics_policy={model_cfg.get('statistics_policy')!r} != {STATISTICS_POLICY_ROLLING!r}."
        )
    if model_cfg.get("teacher_score_included") is not False:
        raise ValueError("Checkpoint violates teacher_score_included=False.")
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
    if (
        expected_candidate_attention_enabled is not None
        and bool(model_cfg.get("candidate_attention_enabled")) != bool(expected_candidate_attention_enabled)
    ):
        raise ValueError(
            f"candidate_attention_enabled mismatch: {model_cfg.get('candidate_attention_enabled')} != {expected_candidate_attention_enabled}."
        )

    persisted_contract = ckpt["input_contract"]
    expected_contract = build_rolling_ranker_input_contract(
        statistics_policy=STATISTICS_POLICY_ROLLING,
        feature_contract_digest=expected_feature_contract_digest or persisted_contract.get("feature_contract_digest"),
        candidate_attention_enabled=bool(model_cfg.get("candidate_attention_enabled")),
        attention_num_heads=int(model_cfg.get("attention_num_heads", 4)),
        attention_dim_feedforward=int(model_cfg.get("attention_dim_feedforward", 256)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
    )
    validate_rolling_ranker_input_contract(persisted_contract, expected_contract)

    persisted_prov = ckpt["provenance"]
    if persisted_prov.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"Provenance statistics_policy mismatch: {persisted_prov.get('statistics_policy')!r}."
        )
    if persisted_prov.get("teacher_score_included") is not False:
        raise ValueError("Provenance must record teacher_score_included=False.")
    if expected_provenance is not None:
        for key, exp_val in expected_provenance.items():
            if persisted_prov.get(key) != exp_val:
                raise ValueError(
                    f"Checkpoint provenance mismatch on {key!r}: {persisted_prov.get(key)!r} != {exp_val!r}."
                )

    state = ckpt["model_state_dict"]
    if (
        "base.statistics_attention_weights" not in state
        or tuple(state["base.statistics_attention_weights"].shape) != EXPECTED_ATTENTION_BIAS_SHAPE
    ):
        raise ValueError("Checkpoint state_dict base.statistics_attention_weights shape mismatch.")
    if "base_head.0.weight" not in state or int(state["base_head.0.weight"].shape[1]) != EXPECTED_HEAD_INPUT_DIM:
        raise ValueError("Checkpoint state_dict base_head.0.weight width mismatch.")
    if "context_project.weight" not in state or int(state["context_project.weight"].shape[1]) != EXPECTED_HEAD_INPUT_DIM:
        raise ValueError("Checkpoint state_dict context_project.weight width mismatch.")

    model = RollingCandidateAttentionRanker.from_model_config(model_cfg)
    model.load_state_dict(state, strict=True)
    return model, dict(ckpt)
