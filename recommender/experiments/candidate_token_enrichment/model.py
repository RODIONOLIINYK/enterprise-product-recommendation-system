"""Candidate-set self-attention + pre-attention statistics-enriched token ranker (C).

Compares on identical rolling_observed_v2 quota100_50 slates:
- Arm A (`candidate_attention_enabled=False, token_enrichment_enabled=False`):
  Rolling pointwise 425D control (identical to RollingCandidateAttentionRanker control).
- Arm B (`candidate_attention_enabled=True, token_enrichment_enabled=False`):
  Masked unordered candidate-set self-attention without pre-attention token enrichment.
- Arm C (`candidate_attention_enabled=True, token_enrichment_enabled=True`):
  Same candidate-set self-attention placement and 425D raw scalar / bilinear bias paths as B,
  PLUS three zero-initialized residual MLPs injecting [0, 1]-normalized statistics BEFORE
  memory self-attention (`memory_encoder`) and candidate-to-memory cross-attention:
  - Profile Token 0: `[z_prof (128), profile8] (136) -> 128 -> 128` (added before `type_embedding[0]`)
  - History Tokens 1..32: `[t_hist (128), event9] (137) -> 128 -> 128` (added before `pos_emb + type_embedding[1]`)
  - Candidate Queries: `[v_cand (128), all41] (169) -> 128 -> 128` (no candidate positional encoding)
"""

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from recommender.artifacts import json_digest
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.feature_schema import (
    NAMES as DYNAMIC_FEATURE_NAMES,
    STATISTICS_POLICY_ROLLING,
    rolling_schema_digest,
)
from recommender.experiments.rolling_candidate_v2.model import (
    EXPECTED_ATTENTION_BIAS_SHAPE,
    EXPECTED_BASE_ENCODED_DIM,
    EXPECTED_CORE_PAIRWISE_DIM,
    EXPECTED_EXCLUDED_STATISTICS,
    EXPECTED_EXTRA_DYNAMIC_DIM,
    EXPECTED_HEAD_INPUT_DIM,
    EXPECTED_SOURCE_PAIRWISE_DIM,
    _extract_upscanr_config,
)
from recommender.models import ProductTower
from recommender.ranker import RankerFeatures, RankerOutput, UPSCANRRanker, _fit_feature_width
from recommender.sequences import DAYS_PER_YEAR, HISTORY_AGE_INDEX
from recommender.shared_data import atomic_torch_save

from .normalization import (
    CANDIDATE_STAT_DIM,
    ENRICHMENT_SCALAR_TRANSFORM_VERSION,
    HISTORY_STAT_DIM,
    PROFILE_STAT_DIM,
    enrichment_normalization_spec,
    normalize_candidate_stats,
    normalize_history_stats,
    normalize_profile_stats,
    sanitize_candidate_inputs,
    sanitize_history_inputs,
)

CHECKPOINT_FORMAT_VERSION = 3


class ResidualTokenEnricher(nn.Module):
    """Zero-output-initialized residual MLP fusing semantic tokens with [0, 1] scalar statistics."""

    def __init__(self, semantic_dim: int, stat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.stat_dim = int(stat_dim)
        self.hidden_dim = int(hidden_dim)
        in_dim = self.semantic_dim + self.stat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.semantic_dim),
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, semantic_tokens: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([semantic_tokens, stats], dim=-1))


class EnrichableUPSCANRRanker(UPSCANRRanker):
    """Isolated UPSCANRRanker subclass adding pre-attention residual token enrichment."""

    def __init__(
        self,
        *args,
        token_enrichment_enabled: bool = False,
        enrichment_transform_version: str = ENRICHMENT_SCALAR_TRANSFORM_VERSION,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.token_enrichment_enabled = bool(token_enrichment_enabled)
        self.enrichment_transform_version = str(enrichment_transform_version)
        if self.enrichment_transform_version != ENRICHMENT_SCALAR_TRANSFORM_VERSION:
            raise ValueError(
                f"Unsupported enrichment_transform_version={self.enrichment_transform_version!r}."
            )
        self.candidate_enricher: Optional[ResidualTokenEnricher] = None
        self.profile_enricher: Optional[ResidualTokenEnricher] = None
        self.history_enricher: Optional[ResidualTokenEnricher] = None
        if self.token_enrichment_enabled:
            self._init_enrichment_modules()

    def _init_enrichment_modules(self) -> None:
        dim = int(self.hidden_dim)
        self.candidate_enricher = ResidualTokenEnricher(dim, CANDIDATE_STAT_DIM, hidden_dim=dim)
        self.profile_enricher = ResidualTokenEnricher(dim, PROFILE_STAT_DIM, hidden_dim=dim)
        self.history_enricher = ResidualTokenEnricher(dim, HISTORY_STAT_DIM, hidden_dim=dim)

    def enable_token_enrichment(
        self,
        enabled: bool,
        *,
        enrichment_seed: Optional[int] = None,
    ) -> None:
        self.token_enrichment_enabled = bool(enabled)
        if self.token_enrichment_enabled:
            if self.candidate_enricher is None:
                with torch.random.fork_rng(devices=[]):
                    if enrichment_seed is not None:
                        torch.manual_seed(int(enrichment_seed))
                    self._init_enrichment_modules()
            for mod in (self.candidate_enricher, self.profile_enricher, self.history_enricher):
                if mod is not None:
                    mod.requires_grad_(True)
        else:
            self.candidate_enricher = None
            self.profile_enricher = None
            self.history_enricher = None

    @classmethod
    def from_upscanr(
        cls,
        src: UPSCANRRanker,
        *,
        token_enrichment_enabled: bool = False,
        enrichment_seed: Optional[int] = None,
    ) -> "EnrichableUPSCANRRanker":
        """Clone an existing UPSCANRRanker state verbatim into an EnrichableUPSCANRRanker."""
        cfg = _extract_upscanr_config(src)
        pt_cfg = cfg.get("product_tower")
        pt = None
        if pt_cfg is not None:
            pt = ProductTower(
                text_dim=int(pt_cfg["text_dim"]),
                output_dim=int(pt_cfg["output_dim"]),
                num_brands=int(pt_cfg["num_brands"]),
                num_categories=int(pt_cfg["num_categories"]),
                num_subcategories=int(pt_cfg["num_subcategories"]),
                num_business_lines=int(pt_cfg["num_business_line" if "num_business_line" in pt_cfg else "num_business_lines"]),
                num_demand_features=int(pt_cfg["num_demand_features"]),
                hidden_dim=int(pt_cfg["hidden_dim"]),
                num_containers=int(pt_cfg["num_containers"]),
                one_hot_sizes=pt_cfg.get("one_hot_sizes"),
                extra_mlp_layer=bool(pt_cfg["extra_mlp_layer"]),
            )
        cat_emb = (
            torch.zeros(int(cfg["catalog_text_num_rows"]), int(cfg["text_dim"]))
            if cfg.get("has_catalog_text_embeddings")
            else None
        )
        dst = cls(
            text_dim=int(cfg["text_dim"]),
            memory_dim=int(cfg["memory_dim"]),
            num_cities=int(cfg["num_cities"]),
            num_regions=int(cfg["num_regions"]),
            num_industries=int(cfg["num_industries"]),
            num_products=int(cfg["num_products"]),
            num_categories=int(cfg["num_categories"]),
            num_subcategories=int(cfg["num_subcategories"]),
            num_business_lines=int(cfg["num_business_lines"]),
            num_tabular_features=int(cfg["num_tabular_features"]),
            num_pairwise_features=int(cfg["base_num_pairwise_features"]),
            seq_feature_dim=int(cfg["seq_feature_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_heads=int(cfg["num_heads"]),
            catalog_text_embeddings=cat_emb,
            product_tower=pt,
            statistics_attention=bool(cfg["statistics_attention"]),
            excluded_statistics=tuple(cfg["excluded_statistics"]),
            cached_only=bool(cfg["cached_only"]),
            token_enrichment_enabled=False,
        )
        dst.load_state_dict(copy.deepcopy(src.state_dict()), strict=True)
        if token_enrichment_enabled:
            dst.enable_token_enrichment(True, enrichment_seed=enrichment_seed)
        return dst

    def encode_customer_memory(
        self,
        *,
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        tabular_features: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: Optional[torch.Tensor] = None,
        seq_category_ids: Optional[torch.Tensor] = None,
        seq_subcategory_ids: Optional[torch.Tensor] = None,
        seq_business_line_ids: Optional[torch.Tensor] = None,
        seq_product_text: Optional[torch.Tensor] = None,
        profile_enrichment_stats: Optional[torch.Tensor] = None,
        history_enrichment_stats: Optional[torch.Tensor] = None,
        disable_profile_enrichment: bool = False,
        disable_history_enrichment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 33-token customer memory, injecting pre-attention residual enrichment when active."""
        cleaned_hist = sanitize_history_inputs(
            sequence=sequence,
            attention_mask=attention_mask,
            seq_product_ids=seq_product_ids,
            seq_category_ids=seq_category_ids,
            seq_subcategory_ids=seq_subcategory_ids,
            seq_business_line_ids=seq_business_line_ids,
            seq_product_text=seq_product_text,
        )
        valid_hist = cleaned_hist["valid_hist"]
        seq_clean = cleaned_hist["sequence"]
        p_ids = cleaned_hist["seq_product_ids"]
        c_ids = cleaned_hist["seq_category_ids"]
        sc_ids = cleaned_hist["seq_subcategory_ids"]
        bl_ids = cleaned_hist["seq_business_line_ids"]
        p_txt = cleaned_hist["seq_product_text"]

        if not self.token_enrichment_enabled:
            return super()._compute_customer_memory(
                text_embedding=text_embedding,
                city_id=city_id,
                region_id=region_id,
                industry_id=industry_id,
                tabular_features=tabular_features,
                sequence=seq_clean,
                attention_mask=attention_mask,
                seq_product_ids=p_ids,
                seq_category_ids=c_ids,
                seq_subcategory_ids=sc_ids,
                seq_business_line_ids=bl_ids,
                seq_product_text=p_txt,
            )

        b_size = text_embedding.size(0)
        seq_len = seq_clean.size(1)

        # 1. Base semantic Token 0 computed with ZEROS in tabular numeric branch (exact baseline parity)
        z_text = self.text_proj(text_embedding)
        c_city = self.city_embedding(city_id)
        c_reg = self.region_embedding(region_id)
        c_ind = self.industry_embedding(industry_id)
        tab_zeros = torch.zeros_like(tabular_features)
        z_tab = self.tabular_proj(torch.cat([c_city, c_reg, c_ind, tab_zeros], dim=-1))
        z_prof = self.profile_fusion(torch.cat([z_text, z_tab], dim=-1))

        if (
            self.profile_enricher is not None
            and profile_enrichment_stats is not None
            and not disable_profile_enrichment
        ):
            z_prof = z_prof + self.profile_enricher(z_prof, profile_enrichment_stats)

        t_0 = (z_prof + self.type_embedding.weight[0]).unsqueeze(1)

        # 2. Base semantic History Tokens 1..32 computed with ZEROS in habit numeric branch
        if p_ids is None:
            p_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=seq_clean.device)
        if c_ids is None:
            c_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=seq_clean.device)
        if sc_ids is None:
            sc_ids = torch.zeros_like(p_ids)
        if bl_ids is None:
            bl_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=seq_clean.device)

        p_emb = self.product_embedding(p_ids)
        cat_emb = self.cat_embedding(c_ids)
        subcat_emb = self.subcat_embedding(sc_ids)
        line_emb = self.line_embedding(bl_ids)
        h_proj = self.habit_proj(torch.zeros_like(seq_clean))
        if p_txt is not None:
            txt_emb = p_txt
        elif self.catalog_text_embeddings is not None:
            txt_emb = self.catalog_text_embeddings[p_ids]
            txt_emb = torch.where(valid_hist.unsqueeze(-1), txt_emb, txt_emb.new_zeros(()))
        else:
            txt_emb = seq_clean.new_zeros((b_size, seq_len, self.text_dim))
        extra = self.product_text_proj(txt_emb)
        hist_raw = torch.cat([p_emb, cat_emb, subcat_emb, line_emb, h_proj, extra], dim=-1)
        t_hist = self.token_proj(hist_raw)

        if (
            self.history_enricher is not None
            and history_enrichment_stats is not None
            and not disable_history_enrichment
        ):
            t_hist_in = torch.where(valid_hist.unsqueeze(-1), t_hist, t_hist.new_zeros(()))
            delta_hist = self.history_enricher(t_hist_in, history_enrichment_stats)
            t_hist = t_hist + torch.where(valid_hist.unsqueeze(-1), delta_hist, delta_hist.new_zeros(()))

        # Positional and token-type embeddings strictly AFTER enrichment
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        t_hist = t_hist + pos_emb + self.type_embedding.weight[1]

        # 3. Concatenate Token 0 + Tokens 1..32 -> (B, 33, 128) & run self-attention
        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = self.build_key_padding_mask(attention_mask)
        memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)
        return memory, k_mask

    def encode_candidate_features(
        self,
        # Customer Profile Inputs (Token 0)
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        tabular_features: torch.Tensor,
        # Customer History Inputs (Tokens 1..32)
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: Optional[torch.Tensor] = None,
        seq_category_ids: Optional[torch.Tensor] = None,
        seq_subcategory_ids: Optional[torch.Tensor] = None,
        seq_business_line_ids: Optional[torch.Tensor] = None,
        seq_product_text: Optional[torch.Tensor] = None,
        # Candidate Slate Inputs
        candidate_indices: Optional[torch.Tensor] = None,
        pairwise_features: Optional[torch.Tensor] = None,
        candidate_vectors: Optional[torch.Tensor] = None,
        catalog_product_vectors: Optional[torch.Tensor] = None,
        candidate_text_emb: Optional[torch.Tensor] = None,
        candidate_brand_ids: Optional[torch.Tensor] = None,
        candidate_cat_ids: Optional[torch.Tensor] = None,
        candidate_subcat_ids: Optional[torch.Tensor] = None,
        candidate_line_ids: Optional[torch.Tensor] = None,
        candidate_demand_feats: Optional[torch.Tensor] = None,
        candidate_container_ids: Optional[torch.Tensor] = None,
        seq_purchase_dates: Optional[torch.Tensor] = None,
        prediction_day: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
        memory_value_statistics: bool = False,
        candidate_mask: Optional[torch.Tensor] = None,
        pairwise_features_41: Optional[torch.Tensor] = None,
        disable_token_enrichment: bool = False,
        disable_candidate_enrichment: bool = False,
        disable_profile_enrichment: bool = False,
        disable_history_enrichment: bool = False,
    ) -> RankerFeatures:
        """Encode customer memory + candidate cross-attention with optional pre-attention enrichment."""
        if not bool(torch.isfinite(text_embedding).all()):
            raise ValueError("text_embedding contains NaN or Inf.")
        if not bool(torch.isfinite(tabular_features).all()):
            raise ValueError("tabular_features contains NaN or Inf.")

        cleaned_hist = sanitize_history_inputs(
            sequence=_fit_feature_width(sequence, self.seq_feature_dim),
            attention_mask=attention_mask,
            seq_purchase_dates=seq_purchase_dates,
            prediction_day=prediction_day,
            seq_product_ids=seq_product_ids,
            seq_category_ids=seq_category_ids,
            seq_subcategory_ids=seq_subcategory_ids,
            seq_business_line_ids=seq_business_line_ids,
            seq_product_text=seq_product_text,
        )

        if not self.token_enrichment_enabled:
            return super().encode_candidate_features(
                text_embedding=text_embedding,
                city_id=city_id,
                region_id=region_id,
                industry_id=industry_id,
                tabular_features=tabular_features,
                sequence=cleaned_hist["sequence"],
                attention_mask=attention_mask,
                seq_product_ids=cleaned_hist["seq_product_ids"],
                seq_category_ids=cleaned_hist["seq_category_ids"],
                seq_subcategory_ids=cleaned_hist["seq_subcategory_ids"],
                seq_business_line_ids=cleaned_hist["seq_business_line_ids"],
                seq_product_text=cleaned_hist["seq_product_text"],
                candidate_indices=candidate_indices,
                pairwise_features=pairwise_features,
                candidate_vectors=candidate_vectors,
                catalog_product_vectors=catalog_product_vectors,
                candidate_text_emb=candidate_text_emb,
                candidate_brand_ids=candidate_brand_ids,
                candidate_cat_ids=candidate_cat_ids,
                candidate_subcat_ids=candidate_subcat_ids,
                candidate_line_ids=candidate_line_ids,
                candidate_demand_feats=candidate_demand_feats,
                candidate_container_ids=candidate_container_ids,
                seq_purchase_dates=cleaned_hist["seq_purchase_dates"],
                prediction_day=prediction_day,
                return_diagnostics=return_diagnostics,
                memory_value_statistics=memory_value_statistics,
            )

        b_size = text_embedding.size(0)
        if pairwise_features_41 is None:
            raise ValueError("Enriched mode requires full 41D pairwise_features_41.")

        pf_41_clean, cand_vecs_clean, cand_idx_clean = sanitize_candidate_inputs(
            pairwise_features_41,
            candidate_mask=candidate_mask,
            candidate_vectors=candidate_vectors,
            candidate_indices=candidate_indices,
        )

        # Build [0, 1]-normalized stats for all 3 enrichment branches
        profile_stats_8 = normalize_profile_stats(
            tabular_features,
            excluded_statistics=self.excluded_statistics,
        )
        history_stats_9 = normalize_history_stats(
            cleaned_hist["sequence"],
            attention_mask,
            cleaned_hist["seq_purchase_dates"],
            prediction_day,
            excluded_statistics=self.excluded_statistics,
        )
        cand_stats_41 = normalize_candidate_stats(pf_41_clean, candidate_mask=candidate_mask)

        # Raw statistics path for existing bilinear statistics bias (identical to UPSCANRRanker)
        seq_raw = cleaned_hist["sequence"]
        if {
            "history_gap", "history_month", "history_repeat", "history_real_indicator",
        }.intersection(self.excluded_statistics):
            seq_raw = seq_raw.clone()
            if "history_gap" in self.excluded_statistics:
                seq_raw[..., 2] = 0
            if "history_month" in self.excluded_statistics:
                seq_raw[..., 3:5] = 0
            if "history_repeat" in self.excluded_statistics:
                seq_raw[..., 7] = 0
            if "history_real_indicator" in self.excluded_statistics:
                seq_raw[..., 8] = 0
        tab_raw = _fit_feature_width(tabular_features, self.num_tabular_features)
        if {"company_distinct_orders", "company_median_gap"}.intersection(self.excluded_statistics):
            tab_raw = tab_raw.clone()
            if "company_distinct_orders" in self.excluded_statistics:
                tab_raw[..., 3] = 0
            if "company_median_gap" in self.excluded_statistics:
                tab_raw[..., 7] = 0

        # 1. Enriched Customer Memory (memory_value_statistics=True semantics: base semantic zeroed,
        #    nonlinear residual enrichment applied BEFORE memory_encoder)
        memory, key_padding_mask = self.encode_customer_memory(
            text_embedding=text_embedding,
            city_id=city_id,
            region_id=region_id,
            industry_id=industry_id,
            tabular_features=tab_raw,
            sequence=seq_raw,
            attention_mask=attention_mask,
            seq_product_ids=cleaned_hist["seq_product_ids"],
            seq_category_ids=cleaned_hist["seq_category_ids"],
            seq_subcategory_ids=cleaned_hist["seq_subcategory_ids"],
            seq_business_line_ids=cleaned_hist["seq_business_line_ids"],
            seq_product_text=cleaned_hist["seq_product_text"],
            profile_enrichment_stats=profile_stats_8,
            history_enrichment_stats=history_stats_9,
            disable_profile_enrichment=bool(disable_token_enrichment or disable_profile_enrichment),
            disable_history_enrichment=bool(disable_token_enrichment or disable_history_enrichment),
        )

        # 2. Candidate Queries + Pre-Cross-Attention Residual Enrichment
        z_cand = self._resolve_candidate_vectors(
            b_size=b_size,
            candidate_vectors=cand_vecs_clean,
            candidate_indices=cand_idx_clean,
            catalog_product_vectors=catalog_product_vectors,
            candidate_text_emb=candidate_text_emb,
            candidate_brand_ids=candidate_brand_ids,
            candidate_cat_ids=candidate_cat_ids,
            candidate_subcat_ids=candidate_subcat_ids,
            candidate_line_ids=candidate_line_ids,
            candidate_demand_feats=candidate_demand_feats,
            candidate_container_ids=candidate_container_ids,
        )
        if candidate_mask is not None:
            z_cand = torch.where(candidate_mask.unsqueeze(-1), z_cand, z_cand.new_zeros(()))
        if (
            self.candidate_enricher is not None
            and not disable_token_enrichment
            and not disable_candidate_enrichment
        ):
            delta_cand = self.candidate_enricher(z_cand, cand_stats_41)
            if candidate_mask is not None:
                delta_cand = torch.where(candidate_mask.unsqueeze(-1), delta_cand, delta_cand.new_zeros(()))
            z_cand = z_cand + delta_cand

        c_slate = z_cand.size(1)
        if pairwise_features is None:
            pf_core = z_cand.new_zeros((b_size, c_slate, self.num_pairwise_features))
        else:
            pf_core = _fit_feature_width(pairwise_features, self.num_pairwise_features)
        if "candidate_repeat" in self.excluded_statistics:
            pf_core = pf_core.clone()
            pf_core[..., 0] = 0

        # 3. Unchanged Bilinear Statistics Bias + Cross-Attention
        statistics_in_scores = self.statistics_attention_weights is not None
        attention_bias = None
        attention_padding = key_padding_mask
        if statistics_in_scores:
            if prediction_day is None or cleaned_hist["seq_purchase_dates"] is None:
                raise ValueError("Date-aware attention requires prediction and purchase dates.")
            purchase_days = (cleaned_hist["seq_purchase_dates"] * DAYS_PER_YEAR).round()
            age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
            age = age_days / (age_days + 90.0)
            purchase_statistics = seq_raw.clone()
            purchase_statistics[..., HISTORY_AGE_INDEX] = age
            key_statistics = torch.cat((
                F.pad(tab_raw[:, None, :], (self.seq_feature_dim, 0)),
                F.pad(purchase_statistics, (0, self.num_tabular_features)),
            ), dim=1)
            candidate_statistics = F.pad(pf_core, (1, 0), value=1)
            coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
            bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
            attention_bias = bias.expand(-1, -1, c_slate, -1).reshape(
                b_size * self.num_heads, c_slate, memory.size(1),
            ).to(z_cand.dtype)
            attention_padding = None

        h_att, attn_w = self.cross_attention(
            query=z_cand,
            key=memory,
            value=memory,
            key_padding_mask=attention_padding,
            attn_mask=attention_bias,
            need_weights=return_diagnostics,
            average_attn_weights=True,
        )
        h_att = torch.nan_to_num(h_att, nan=0.0)
        token_0_weight = history_weights = None
        if return_diagnostics:
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            if key_padding_mask is not None:
                mask_exp = key_padding_mask.unsqueeze(1).expand(-1, c_slate, -1)
                attn_w = attn_w.masked_fill(mask_exp, 0.0)
                w_sum = attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn_w = attn_w / w_sum
            token_0_weight = attn_w[..., 0]
            history_weights = attn_w[..., 1:].sum(dim=-1)

        interaction = z_cand * h_att
        fused = torch.cat([z_cand, h_att, interaction, pf_core], dim=-1)
        return RankerFeatures(
            features=fused,
            h_attended=h_att,
            z_candidate=z_cand,
            memory=memory,
            attn_weights=attn_w,
            token_0_weight=token_0_weight,
            history_weights=history_weights,
        )


def build_enrichment_ranker_input_contract(
    *,
    pairwise_feature_names: Sequence[str] = DYNAMIC_FEATURE_NAMES,
    statistics_policy: str = STATISTICS_POLICY_ROLLING,
    feature_contract_digest: Optional[str] = None,
    candidate_attention_enabled: bool = False,
    token_enrichment_enabled: bool = False,
    enrichment_transform_version: str = ENRICHMENT_SCALAR_TRANSFORM_VERSION,
    attention_num_heads: int = 4,
    attention_dim_feedforward: int = 256,
    attention_dropout: float = 0.1,
) -> Dict[str, Any]:
    """Build deterministic ordered input contract for CandidateTokenEnrichmentRanker."""
    names_tuple = tuple(str(n) for n in pairwise_feature_names)
    if names_tuple != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError("Ordered pairwise_feature_names do not match canonical 41-feature schema.")
    if statistics_policy != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"CandidateTokenEnrichmentRanker requires statistics_policy={STATISTICS_POLICY_ROLLING!r}, "
            f"got {statistics_policy!r}."
        )
    schema_dig = feature_contract_digest or rolling_schema_digest()
    norm_spec = enrichment_normalization_spec(excluded_statistics=EXPECTED_EXCLUDED_STATISTICS)
    body = {
        "contract_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "CandidateTokenEnrichmentRanker",
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
        "token_enrichment_enabled": bool(token_enrichment_enabled),
        "enrichment_transform_version": str(enrichment_transform_version),
        "enrichment_normalization_digest": norm_spec["transform_digest"],
        "enrichment_dims": {
            "candidate": [128 + CANDIDATE_STAT_DIM, 128, 128],
            "profile": [128 + PROFILE_STAT_DIM, 128, 128],
            "history": [128 + HISTORY_STAT_DIM, 128, 128],
        },
        "candidate_set_positional_encoding": False,
        "residual_zero_initialized": True,
        "memory_value_statistics": bool(token_enrichment_enabled),
        "attention_num_heads": int(attention_num_heads),
        "attention_dim_feedforward": int(attention_dim_feedforward),
        "attention_dropout": float(attention_dropout),
    }
    return {**body, "contract_digest": json_digest(body)}


def validate_enrichment_ranker_input_contract(
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
            f"CandidateTokenEnrichmentRanker input_contract digest corrupted: "
            f"{persisted['contract_digest']} != {recomputed}."
        )
    for key, exp_val in expected.items():
        if key not in persisted:
            raise ValueError(f"CandidateTokenEnrichmentRanker input_contract missing key {key!r}.")
        if persisted[key] != exp_val:
            raise ValueError(
                f"CandidateTokenEnrichmentRanker input_contract mismatch on {key!r}: "
                f"{persisted[key]!r} != {exp_val!r}."
            )


class CandidateTokenEnrichmentRanker(nn.Module):
    """Unified Stage 2 ranker supporting Arms A (pointwise), B (cand-attn), and C (cand-attn + token enrichment)."""

    def __init__(
        self,
        base: EnrichableUPSCANRRanker,
        base_head: nn.Sequential,
        *,
        candidate_attention_enabled: bool = False,
        token_enrichment_enabled: bool = False,
        enrichment_seed: Optional[int] = None,
        attention_num_heads: int = 4,
        attention_dim_feedforward: int = 256,
        attention_dropout: float = 0.1,
        ordered_pairwise_features: Sequence[str] = DYNAMIC_FEATURE_NAMES,
        statistics_policy: str = STATISTICS_POLICY_ROLLING,
    ):
        super().__init__()
        if not isinstance(base, EnrichableUPSCANRRanker):
            raise TypeError("CandidateTokenEnrichmentRanker requires an EnrichableUPSCANRRanker base.")
        if int(base.num_pairwise_features) != EXPECTED_CORE_PAIRWISE_DIM:
            raise ValueError(
                f"Expected base.num_pairwise_features={EXPECTED_CORE_PAIRWISE_DIM}, got {base.num_pairwise_features}."
            )
        if base.statistics_attention_weights is None or tuple(base.statistics_attention_weights.shape) != EXPECTED_ATTENTION_BIAS_SHAPE:
            raise ValueError(
                f"Expected statistics_attention_weights shape {EXPECTED_ATTENTION_BIAS_SHAPE}."
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
                f"CandidateTokenEnrichmentRanker requires statistics_policy={STATISTICS_POLICY_ROLLING!r}."
            )

        self.base = base
        self.base_head = base_head
        self.base.configure_cached_training(True)
        self.base.mlp.requires_grad_(False)
        self.base.enable_token_enrichment(bool(token_enrichment_enabled), enrichment_seed=enrichment_seed)

        width = int(base.hidden_dim)
        self.width = width
        self.attention_num_heads = int(attention_num_heads)
        self.attention_dim_feedforward = int(attention_dim_feedforward)
        self.attention_dropout = float(attention_dropout)
        self.ordered_pairwise_features = tuple(ordered_pairwise_features)
        self.statistics_policy = str(statistics_policy)
        self.candidate_attention_enabled = bool(candidate_attention_enabled)
        self.token_enrichment_enabled = bool(token_enrichment_enabled)
        self.enrichment_transform_version = ENRICHMENT_SCALAR_TRANSFORM_VERSION

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

        self._set_candidate_attention_grad(self.candidate_attention_enabled)
        self.register_buffer(
            "extra_columns",
            torch.tensor(list(range(EXPECTED_CORE_PAIRWISE_DIM, EXPECTED_SOURCE_PAIRWISE_DIM)), dtype=torch.long),
            persistent=True,
        )

    def _set_candidate_attention_grad(self, enabled: bool) -> None:
        for mod in (self.context_project, self.candidate_attention, self.residual_head):
            mod.requires_grad_(bool(enabled))

    @classmethod
    def from_dynamic_context_ranker(
        cls,
        dyn_ranker: CandidateContextRanker,
        *,
        candidate_attention_enabled: bool = False,
        token_enrichment_enabled: bool = False,
        enrichment_seed: Optional[int] = None,
        attention_num_heads: int = 4,
        attention_dim_feedforward: int = 256,
        attention_dropout: float = 0.1,
    ) -> "CandidateTokenEnrichmentRanker":
        """Wrap an existing 425D dynamic_features CandidateContextRanker preserving base_head verbatim."""
        if getattr(dyn_ranker, "mode", None) != "dynamic_features":
            raise ValueError(
                f"Expected CandidateContextRanker(mode='dynamic_features'), got {getattr(dyn_ranker, 'mode', None)!r}."
            )
        enrichable_base = EnrichableUPSCANRRanker.from_upscanr(
            dyn_ranker.base,
            token_enrichment_enabled=False,
        )
        cloned_head = copy.deepcopy(dyn_ranker.head)
        return cls(
            enrichable_base,
            cloned_head,
            candidate_attention_enabled=candidate_attention_enabled,
            token_enrichment_enabled=token_enrichment_enabled,
            enrichment_seed=enrichment_seed,
            attention_num_heads=attention_num_heads,
            attention_dim_feedforward=attention_dim_feedforward,
            attention_dropout=attention_dropout,
        )

    def clone_arm(
        self,
        *,
        candidate_attention_enabled: bool,
        token_enrichment_enabled: bool,
        enrichment_seed: Optional[int] = None,
    ) -> "CandidateTokenEnrichmentRanker":
        """Clone base and candidate-attention initial weights identically for paired Arms A/B/C."""
        cloned = copy.deepcopy(self)
        cloned.candidate_attention_enabled = bool(candidate_attention_enabled)
        cloned.token_enrichment_enabled = bool(token_enrichment_enabled)
        cloned._set_candidate_attention_grad(bool(candidate_attention_enabled))
        cloned.base.enable_token_enrichment(bool(token_enrichment_enabled), enrichment_seed=enrichment_seed)
        return cloned

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "CandidateTokenEnrichmentRanker":
        """Reconstruct architecture purely from persisted checkpoint model_config."""
        if model_config.get("teacher_score_included") is not False:
            raise ValueError("CandidateTokenEnrichmentRanker forbids teacher_score_included=True.")
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
        token_enrichment = bool(model_config["token_enrichment_enabled"])
        base = EnrichableUPSCANRRanker(
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
            token_enrichment_enabled=token_enrichment,
            enrichment_transform_version=str(
                model_config.get("enrichment_transform_version", ENRICHMENT_SCALAR_TRANSFORM_VERSION)
            ),
        )
        base_head = copy.deepcopy(base.mlp)
        base_head[0] = nn.Linear(int(model_config["head_input_dim"]), base_head[0].out_features)
        return cls(
            base,
            base_head,
            candidate_attention_enabled=bool(model_config["candidate_attention_enabled"]),
            token_enrichment_enabled=token_enrichment,
            attention_num_heads=int(model_config.get("attention_num_heads", 4)),
            attention_dim_feedforward=int(model_config.get("attention_dim_feedforward", 256)),
            attention_dropout=float(model_config.get("attention_dropout", 0.1)),
            ordered_pairwise_features=tuple(model_config["ordered_pairwise_features"]),
            statistics_policy=str(model_config["statistics_policy"]),
        )

    def model_config(self) -> Dict[str, Any]:
        return {
            "model_class": "CandidateTokenEnrichmentRanker",
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
            "token_enrichment_enabled": bool(self.token_enrichment_enabled),
            "enrichment_transform_version": self.enrichment_transform_version,
            "candidate_set_positional_encoding": False,
            "residual_zero_initialized": True,
            "attention_num_heads": self.attention_num_heads,
            "attention_dim_feedforward": self.attention_dim_feedforward,
            "attention_dropout": self.attention_dropout,
            "ordered_pairwise_features": list(self.ordered_pairwise_features),
            "upscanr": _extract_upscanr_config(self.base),
        }

    def input_contract(self, *, feature_contract_digest: Optional[str] = None) -> Dict[str, Any]:
        return build_enrichment_ranker_input_contract(
            pairwise_feature_names=self.ordered_pairwise_features,
            statistics_policy=self.statistics_policy,
            feature_contract_digest=feature_contract_digest,
            candidate_attention_enabled=self.candidate_attention_enabled,
            token_enrichment_enabled=self.token_enrichment_enabled,
            enrichment_transform_version=self.enrichment_transform_version,
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
        disable_token_enrichment: bool = False,
        disable_candidate_enrichment: bool = False,
        disable_profile_enrichment: bool = False,
        disable_history_enrichment: bool = False,
        **kwargs,
    ) -> RankerOutput:
        if "mlp_score" in kwargs and kwargs["mlp_score"] is not None:
            raise ValueError("CandidateTokenEnrichmentRanker forbids Stage 1 mlp_score input.")
        kwargs.pop("mlp_score", None)

        pf, cand_vecs, cand_idx = sanitize_candidate_inputs(
            pairwise_features,
            candidate_mask=candidate_mask,
            candidate_vectors=kwargs.get("candidate_vectors"),
            candidate_indices=kwargs.get("candidate_indices"),
        )
        if cand_vecs is not None:
            kwargs["candidate_vectors"] = cand_vecs
        if cand_idx is not None:
            kwargs["candidate_indices"] = cand_idx

        encoded = self.base.encode_candidate_features(
            pairwise_features=pf[..., :EXPECTED_CORE_PAIRWISE_DIM],
            pairwise_features_41=pf,
            candidate_mask=candidate_mask,
            disable_token_enrichment=disable_token_enrichment,
            disable_candidate_enrichment=disable_candidate_enrichment,
            disable_profile_enrichment=disable_profile_enrichment,
            disable_history_enrichment=disable_history_enrichment,
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
        return self.forward(**kwargs).logits


def save_enrichment_ranker_checkpoint(
    path: Union[Path, str],
    model: CandidateTokenEnrichmentRanker,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    selection_criterion: str,
    provenance: Mapping[str, Any],
    feature_contract_digest: str,
    settings: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist CandidateTokenEnrichmentRanker checkpoint with strict contract and provenance."""
    out_path = Path(path)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "CandidateTokenEnrichmentRanker",
        "model_module": "recommender.experiments.candidate_token_enrichment.model",
        "model_config": model.model_config(),
        "input_contract": model.input_contract(feature_contract_digest=feature_contract_digest),
        "normalization_spec": enrichment_normalization_spec(
            excluded_statistics=model.base.excluded_statistics
        ),
        "provenance": dict(provenance),
        "epoch": int(epoch),
        "selection_criterion": str(selection_criterion),
        "metrics": dict(metrics),
        "settings": dict(settings) if settings is not None else {},
        "model_state_dict": model.state_dict(),
    }
    atomic_torch_save(payload, out_path)
    return out_path


def load_enrichment_ranker_checkpoint(
    checkpoint_path: Union[Path, str],
    *,
    expected_provenance: Optional[Mapping[str, Any]] = None,
    expected_feature_contract_digest: Optional[str] = None,
    expected_candidate_attention_enabled: Optional[bool] = None,
    expected_token_enrichment_enabled: Optional[bool] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[CandidateTokenEnrichmentRanker, Dict[str, Any]]:
    """Strictly validate and load a CandidateTokenEnrichmentRanker checkpoint."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}.")
    if ckpt.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_format_version={ckpt.get('checkpoint_format_version')!r}."
        )
    if ckpt.get("model_class") != "CandidateTokenEnrichmentRanker":
        raise ValueError(f"Unexpected model_class={ckpt.get('model_class')!r}.")
    for req_key in ("model_config", "input_contract", "normalization_spec", "provenance", "model_state_dict"):
        if req_key not in ckpt:
            raise ValueError(f"Checkpoint {checkpoint_path} missing required field {req_key!r}.")

    model_cfg = ckpt["model_config"]
    if model_cfg.get("model_class") != "CandidateTokenEnrichmentRanker":
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
        raise ValueError("Checkpoint base_encoded_dim mismatch.")
    if int(model_cfg.get("head_input_dim", -1)) != EXPECTED_HEAD_INPUT_DIM:
        raise ValueError("Checkpoint head_input_dim mismatch.")
    if tuple(model_cfg.get("attention_bias_shape", ())) != EXPECTED_ATTENTION_BIAS_SHAPE:
        raise ValueError("Checkpoint attention_bias_shape mismatch.")
    if model_cfg.get("enrichment_transform_version") != ENRICHMENT_SCALAR_TRANSFORM_VERSION:
        raise ValueError("Checkpoint enrichment_transform_version mismatch.")

    if (
        expected_candidate_attention_enabled is not None
        and bool(model_cfg.get("candidate_attention_enabled")) != bool(expected_candidate_attention_enabled)
    ):
        raise ValueError(
            f"candidate_attention_enabled mismatch: {model_cfg.get('candidate_attention_enabled')} != {expected_candidate_attention_enabled}."
        )
    if (
        expected_token_enrichment_enabled is not None
        and bool(model_cfg.get("token_enrichment_enabled")) != bool(expected_token_enrichment_enabled)
    ):
        raise ValueError(
            f"token_enrichment_enabled mismatch: {model_cfg.get('token_enrichment_enabled')} != {expected_token_enrichment_enabled}."
        )

    persisted_contract = ckpt["input_contract"]
    expected_contract = build_enrichment_ranker_input_contract(
        statistics_policy=STATISTICS_POLICY_ROLLING,
        feature_contract_digest=expected_feature_contract_digest or persisted_contract.get("feature_contract_digest"),
        candidate_attention_enabled=bool(model_cfg.get("candidate_attention_enabled")),
        token_enrichment_enabled=bool(model_cfg.get("token_enrichment_enabled")),
        enrichment_transform_version=str(model_cfg.get("enrichment_transform_version")),
        attention_num_heads=int(model_cfg.get("attention_num_heads", 4)),
        attention_dim_feedforward=int(model_cfg.get("attention_dim_feedforward", 256)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
    )
    validate_enrichment_ranker_input_contract(persisted_contract, expected_contract)

    expected_norm = enrichment_normalization_spec(
        excluded_statistics=tuple(model_cfg.get("excluded_statistics", ()))
    )
    if ckpt["normalization_spec"].get("transform_digest") != expected_norm["transform_digest"]:
        raise ValueError("Checkpoint normalization_spec transform_digest mismatch.")

    persisted_prov = ckpt["provenance"]
    if persisted_prov.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError("Provenance statistics_policy mismatch.")
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

    has_enrich_keys = any(k.startswith("base.candidate_enricher.") for k in state)
    if bool(model_cfg["token_enrichment_enabled"]) != has_enrich_keys:
        raise ValueError(
            f"State dict enrichment keys ({has_enrich_keys}) do not match "
            f"token_enrichment_enabled={model_cfg['token_enrichment_enabled']}."
        )
    if model_cfg["token_enrichment_enabled"]:
        for prefix, exp_in in (
            ("base.candidate_enricher.net.0.weight", 128 + CANDIDATE_STAT_DIM),
            ("base.profile_enricher.net.0.weight", 128 + PROFILE_STAT_DIM),
            ("base.history_enricher.net.0.weight", 128 + HISTORY_STAT_DIM),
        ):
            if prefix not in state or int(state[prefix].shape[1]) != exp_in:
                raise ValueError(f"Checkpoint missing or invalid enrichment weight {prefix!r}.")

    model = CandidateTokenEnrichmentRanker.from_model_config(model_cfg)
    model.load_state_dict(state, strict=True)
    return model, dict(ckpt)
