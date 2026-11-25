"""Single-Stage End-to-End Stage 2 Ranker for `v2_full` (zero separate Stage 1 pretraining).

Trains `ProductTower` (`v_bn`) and the Customer Profile Encoder (Token 0) directly inside Stage 2:
  - Arm 3 (`arm3_e2e_stage2_only`): Live `ProductTower` + live Token 0 customer encoder trained from
    scratch inside Stage 2 (`L_next + 0.3 * L_60d`). At inference, `v_bn` is precomputed once per
    catalogue refresh (0 ms product-tower overhead per request).
  - Arm 4 (`arm4_e2e_unseen_attn_aux`): Adds training-only `Head_attn([v_bn, h_att, v_bn * h_att])`
    with zero direct `cand_38` inputs to `Head_attn` (note: `cand_38` still influences cross-attention
    weights via query `z_cand` and statistics bias), masking all previously bought formulations out
    and supervising strictly on 60-day truly new adoptions (discarded at inference -> 0 extra params,
    0 ms latency).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.experiments.roadmap_v2.clean_ranker import (
    ACTIVE_SEQ_STAT_INDICES,
)
from recommender.experiments.static_enrichment.model import StaticEnrichedProductTower
from recommender.ranker import DEFAULT_SEQUENCE_LENGTH, RankerOutput
from recommender.sequences import DAYS_PER_YEAR, HISTORY_AGE_INDEX


class EndToEndStage2Ranker(nn.Module):
    """Single-stage full-pool UPS-CANR training product & customer embeddings directly in Stage 2."""

    def __init__(
        self,
        *,
        product_attr_vocab_sizes: Mapping[str, int],
        num_cities: int,
        num_regions: int,
        num_industries: int,
        num_demand_features: int = 65,
        num_candidate_features: int = 38,
        num_customer_features: int = 11,
        num_pairwise_bias_features: int = 13,
        active_seq_stat_indices: Sequence[int] = ACTIVE_SEQ_STAT_INDICES,
        pure_v_query: bool = False,
        text_dim: int = 1024,
        d_model: int = 128,
        num_heads: int = 8,
        memory_ffn_dim: int = 384,
        memory_layers: int = 1,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        mlp_widths: Sequence[int] = (384, 192, 96),
        dropout: float = 0.12,
        aux_60d_enabled: bool = True,
        unseen_attn_aux_enabled: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        if int(memory_layers) < 1:
            raise ValueError(f"memory_layers must be >= 1, got {memory_layers}")
        if int(max_seq_len) < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
        if not active_seq_stat_indices:
            raise ValueError("active_seq_stat_indices must be non-empty.")
        if int(num_pairwise_bias_features) > int(num_candidate_features):
            raise ValueError(
                f"num_pairwise_bias_features={num_pairwise_bias_features} exceeds "
                f"num_candidate_features={num_candidate_features}."
            )
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.memory_layers = int(memory_layers)
        self.max_seq_len = int(max_seq_len)
        self.num_candidate_features = int(num_candidate_features)
        self.num_customer_features = int(num_customer_features)
        self.num_pairwise_bias_features = int(num_pairwise_bias_features)
        self.active_seq_stat_indices: Tuple[int, ...] = tuple(int(i) for i in active_seq_stat_indices)
        self.num_hist_key_stats = len(self.active_seq_stat_indices)
        self.pure_v_query = bool(pure_v_query)
        self.aux_60d_enabled = bool(aux_60d_enabled)
        self.unseen_attn_aux_enabled = bool(unseen_attn_aux_enabled)

        # 1. Live ProductTower trained end-to-end inside Stage 2 (cached once at inference -> 0 ms!)
        self.product_tower = StaticEnrichedProductTower(
            attribute_vocab_sizes=product_attr_vocab_sizes,
            text_dim=int(text_dim),
            output_dim=self.d_model,
            hidden_dim=self.d_model,
            num_demand_features=int(num_demand_features),
            attr_emb_dim=16,
            dropout=0.1,
        )

        # 2. Live Token 0 Customer Profile Encoder (no separate Stage 1 CustomerTower needed!)
        self.cust_text_proj = nn.Sequential(
            nn.Linear(int(text_dim), self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.city_embedding = nn.Embedding(int(num_cities), 16, padding_idx=0)
        self.region_embedding = nn.Embedding(int(num_regions), 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(int(num_industries), 16, padding_idx=0)
        tab_in = 16 * 3 + self.num_customer_features
        self.cust_tab_proj = nn.Sequential(
            nn.Linear(tab_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.token0_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(2, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model) * 0.02)

        # 3. N-Layer 8-Head Memory Self-Attention over [Token 0, v_bn[seq_1..S]]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=int(memory_ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.memory_layers, enable_nested_tensor=False
        )

        # 4. Candidate Query Residual Adapter on live v_bn (omitted when pure_v_query=True -> Q = v_bn)
        if self.pure_v_query:
            self.candidate_mlp: Optional[nn.Sequential] = None
            self.candidate_norm: Optional[nn.LayerNorm] = None
        else:
            self.candidate_mlp = nn.Sequential(
                nn.Linear(self.d_model + self.num_candidate_features, self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.d_model),
            )
            self.candidate_norm = nn.LayerNorm(self.d_model)

        # 5. 8-Head Statistics-Biased Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.statistics_attention_weights = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.num_pairwise_bias_features + 1,
                self.num_hist_key_stats + self.num_customer_features,
            )
        )

        # 6. Main Scoring Trunk (384 -> 192 -> 96 -> 1) + Training-Only 60d Aux Head
        total_in = self.d_model * 3 + self.num_candidate_features
        self.mlp_blocks = nn.ModuleList()
        prev = total_in
        for idx, width in enumerate(mlp_widths):
            w = int(width)
            dp = 0.25 if idx == 0 else (0.1 if idx == len(mlp_widths) - 1 else 0.2)
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(prev, w),
                    nn.LayerNorm(w),
                    nn.GELU(),
                    nn.Dropout(dp),
                )
            )
            prev = w
        self.main_head = nn.Linear(prev, 1)
        self.aux_60d_head: Optional[nn.Linear] = nn.Linear(prev, 1) if self.aux_60d_enabled else None

        # 7. Training-Only Unseen-Discovery Attention Aux Head (ZERO candidate tabular shortcuts!)
        self.unseen_attn_head: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(self.d_model, 1),
            )
            if self.unseen_attn_aux_enabled
            else None
        )

    def optimizer_param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Return AdamW parameter groups for the end-to-end ranker."""
        decay_params = [param for param in self.parameters() if param.requires_grad]
        return [{"params": decay_params, "weight_decay": float(weight_decay)}]

    @torch.no_grad()
    def precompute_product_vectors(
        self,
        prod_inputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """Encode all formulations once for cached zero-overhead inference."""
        return self.product_tower(
            prod_inputs["text_embedding"],
            prod_inputs["demand_features"],
            prod_inputs["attr_ids"],
        )

    def _compute_product_vectors(
        self,
        prod_inputs: Optional[Mapping[str, Any]],
        precomputed_v_bn: Optional[torch.Tensor],
        *,
        train_bn_limit: Optional[int] = None,
        total_bn: Optional[int] = None,
    ) -> torch.Tensor:
        if precomputed_v_bn is not None:
            return precomputed_v_bn
        if prod_inputs is None:
            raise ValueError("Provide either prod_inputs or precomputed_v_bn.")
        if train_bn_limit is not None:
            v_tr = self.product_tower(
                prod_inputs["text_embedding"][:train_bn_limit],
                prod_inputs["demand_features"][:train_bn_limit],
                {k: v[:train_bn_limit] for k, v in prod_inputs["attr_ids"].items()},
            )
            n_tot = int(total_bn) if total_bn is not None else int(prod_inputs["text_embedding"].size(0))
            if n_tot > train_bn_limit:
                pad = torch.zeros((n_tot - train_bn_limit, v_tr.size(1)), dtype=v_tr.dtype, device=v_tr.device)
                return torch.cat([v_tr, pad], dim=0)
            return v_tr
        return self.product_tower(
            prod_inputs["text_embedding"],
            prod_inputs["demand_features"],
            prod_inputs["attr_ids"],
        )

    def forward(
        self,
        *,
        company_text_embedding: torch.Tensor,   # (B, 1024)
        city_id: torch.Tensor,                  # (B,)
        region_id: torch.Tensor,                # (B,)
        industry_id: torch.Tensor,              # (B,)
        customer_features: torch.Tensor,        # (B, C_cust)
        sequence: torch.Tensor,                 # (B, S, 9)
        attention_mask: torch.Tensor,           # (B, S)
        seq_product_ids: torch.Tensor,          # (B, S) 1-based into v_bn
        seq_purchase_dates: torch.Tensor,       # (B, S)
        prediction_day: torch.Tensor,           # (B,)
        candidate_features: torch.Tensor,       # (B, C, F_cand)
        candidate_mask: Optional[torch.Tensor] = None,
        prod_inputs: Optional[Mapping[str, Any]] = None,
        precomputed_v_bn: Optional[torch.Tensor] = None,
        train_bn_limit: Optional[int] = None,
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[RankerOutput, Dict[str, Any]]:
        b_size = company_text_embedding.size(0)
        seq_len = attention_mask.size(1)
        c_slate = candidate_features.size(1)

        v_bn = self._compute_product_vectors(
            prod_inputs, precomputed_v_bn, train_bn_limit=train_bn_limit, total_bn=c_slate
        )  # (C, d_model)

        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)

        # Token 0: Live Customer Profile (company BGE-M3 + city/region/industry + dynamic customer stats)
        z_text = self.cust_text_proj(company_text_embedding)
        z_tab = self.cust_tab_proj(
            torch.cat(
                [
                    self.city_embedding(city_id),
                    self.region_embedding(region_id),
                    self.industry_embedding(industry_id),
                    customer_features,
                ],
                dim=-1,
            )
        )
        t_0 = (self.token0_fusion(torch.cat([z_text, z_tab], dim=-1)) + self.type_embedding.weight[0]).unsqueeze(1)

        # Tokens 1..S: Live ProductTower v_bn lookup + pos & type embeddings
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.pos_embedding.size(1)}."
            )
        v_padded = F.pad(v_bn, (0, 0, 1, 0))  # index 0 = zeros
        t_hist = F.embedding(seq_product_ids, v_padded, padding_idx=0)
        pos = self.pos_embedding[:, :seq_len, :]
        t_hist = t_hist + pos + self.type_embedding.weight[1]

        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = torch.zeros((b_size, seq_len + 1), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 1:] = attention_mask == 0

        memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)

        # Candidate Queries: pure v_bn when pure_v_query=True, else residual MLP over [v_bn, cand_feats]
        v_exp = v_bn.unsqueeze(0).expand(b_size, -1, -1)
        if self.pure_v_query:
            z_cand = v_exp
        else:
            assert self.candidate_mlp is not None and self.candidate_norm is not None
            z_cand = self.candidate_norm(
                v_exp + self.candidate_mlp(torch.cat([v_exp, candidate_features], dim=-1))
            )

        # Cross-Attention key statistics
        seq_clean = sequence.clone()
        seq_clean[..., HISTORY_AGE_INDEX] = age
        active_hist_stats = seq_clean[..., self.active_seq_stat_indices]

        key_statistics = torch.cat(
            (
                F.pad(customer_features[:, None, :], (self.num_hist_key_stats, 0)),
                F.pad(active_hist_stats, (0, self.num_customer_features)),
            ),
            dim=1,
        )
        cand_bias_feats = candidate_features[..., : self.num_pairwise_bias_features]
        candidate_statistics = F.pad(cand_bias_feats, (1, 0), value=1.0)
        coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
        bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]  # (B, H, C, S+1)
        bias = bias.masked_fill(k_mask[:, None, None, :], -torch.inf)
        attention_bias = bias.reshape(b_size * self.num_heads, c_slate, memory.size(1)).to(z_cand.dtype)

        h_att, attn_w = self.cross_attention(
            query=z_cand,
            key=memory,
            value=memory,
            attn_mask=attention_bias,
            need_weights=return_diagnostics,
            average_attn_weights=True,
        )
        h_att = torch.nan_to_num(h_att, nan=0.0)

        token_0_weight = history_weights = None
        if return_diagnostics and attn_w is not None:
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            mask_exp = k_mask.unsqueeze(1).expand(-1, c_slate, -1)
            attn_w = attn_w.masked_fill(mask_exp, 0.0)
            attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            token_0_weight = attn_w[..., 0]
            history_weights = attn_w[..., 1:].sum(dim=-1)

        hidden = torch.cat([z_cand, h_att, z_cand * h_att, candidate_features], dim=-1)
        for block in self.mlp_blocks:
            hidden = block(hidden)

        logits = self.main_head(hidden).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)

        out = RankerOutput(
            scores=logits,
            raw_scores=logits,
            attn_weights=attn_w if return_diagnostics else None,
            h_attended=h_att,
            z_candidate=z_cand,
            token_0_weight=token_0_weight,
            history_weights=history_weights,
            memory=memory,
        )
        if not return_aux:
            return out

        aux_60d_logits = None
        if self.aux_60d_head is not None:
            aux_60d_logits = self.aux_60d_head(hidden).squeeze(-1)
            if candidate_mask is not None:
                aux_60d_logits = aux_60d_logits.masked_fill(~candidate_mask, -torch.inf)

        unseen_attn_logits = None
        if self.unseen_attn_head is not None:
            # Strictly ZERO candidate tabular inputs: uses raw live v_bn and Transformer h_att only!
            attn_feat = torch.cat([v_exp, h_att, v_exp * h_att], dim=-1)
            unseen_attn_logits = self.unseen_attn_head(attn_feat).squeeze(-1)

        return {
            "output": out,
            "aux_60d_logits": aux_60d_logits,
            "unseen_attn_logits": unseen_attn_logits,
        }


class TwoStageAdaptiveRanker(nn.Module):
    """Wraps canonical `SharedUVStage5Ranker` for pretrained-Stage-1 ablations:

      - `unseen_attn_aux_enabled`: Training-only `Head_attn([v_bn, h_att, v_bn * h_att])` with
        ZERO candidate tabular shortcuts, supervising Transformer discovery directly on 60d unseen items.
      - `product_tower` (optional): Pretrained Stage 1 `StaticEnrichedProductTower` unfrozen
        end-to-end inside Stage 2 (Ep 1 frozen warmup, Ep 2..10 low LR = 1e-4; cached once at
        inference for 0 ms latency overhead).
    """

    def __init__(
        self,
        *,
        ranker: nn.Module,
        product_tower: Optional[nn.Module] = None,
        d_model: int = 128,
        unseen_attn_aux_enabled: bool = True,
    ):
        super().__init__()
        self.ranker = ranker
        self.product_tower = product_tower
        self.d_model = int(d_model)
        self.unseen_attn_aux_enabled = bool(unseen_attn_aux_enabled)
        self.unseen_attn_head: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(self.d_model, 1),
            )
            if self.unseen_attn_aux_enabled
            else None
        )

    @torch.no_grad()
    def precompute_product_vectors(self, prod_inputs: Mapping[str, Any]) -> torch.Tensor:
        if self.product_tower is None:
            raise RuntimeError("No product_tower attached to TwoStageAdaptiveRanker.")
        return self.product_tower(
            prod_inputs["text_embedding"],
            prod_inputs["demand_features"],
            prod_inputs["attr_ids"],
        )

    def _resolve_v_bn(
        self,
        *,
        prod_inputs: Optional[Mapping[str, Any]],
        precomputed_v_bn: Optional[torch.Tensor],
        train_bn_limit: Optional[int],
        total_bn: int,
    ) -> torch.Tensor:
        if precomputed_v_bn is not None:
            return precomputed_v_bn
        if self.product_tower is None or prod_inputs is None:
            raise ValueError("Provide precomputed_v_bn or attach product_tower + prod_inputs.")
        if train_bn_limit is not None:
            v_tr = self.product_tower(
                prod_inputs["text_embedding"][:train_bn_limit],
                prod_inputs["demand_features"][:train_bn_limit],
                {k: v[:train_bn_limit] for k, v in prod_inputs["attr_ids"].items()},
            )
            if total_bn > train_bn_limit:
                pad = torch.zeros((total_bn - train_bn_limit, v_tr.size(1)), dtype=v_tr.dtype, device=v_tr.device)
                return torch.cat([v_tr, pad], dim=0)
            return v_tr
        return self.product_tower(
            prod_inputs["text_embedding"],
            prod_inputs["demand_features"],
            prod_inputs["attr_ids"],
        )

    def forward(
        self,
        *,
        customer_vector: torch.Tensor,
        customer_features: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: torch.Tensor,
        seq_purchase_dates: torch.Tensor,
        prediction_day: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        prod_inputs: Optional[Mapping[str, Any]] = None,
        precomputed_v_bn: Optional[torch.Tensor] = None,
        train_bn_limit: Optional[int] = None,
        return_aux: bool = False,
    ) -> Union[RankerOutput, Dict[str, Any]]:
        b_size, c_slate = candidate_features.size(0), candidate_features.size(1)
        v_bn = self._resolve_v_bn(
            prod_inputs=prod_inputs,
            precomputed_v_bn=precomputed_v_bn,
            train_bn_limit=train_bn_limit,
            total_bn=c_slate,
        )
        res = self.ranker(
            customer_vector=customer_vector,
            customer_features=customer_features,
            catalog_product_vectors=v_bn,
            sequence=sequence,
            attention_mask=attention_mask,
            seq_product_ids=seq_product_ids,
            seq_purchase_dates=seq_purchase_dates,
            prediction_day=prediction_day,
            candidate_features=candidate_features,
            candidate_mask=candidate_mask,
            return_aux=return_aux,
        )
        if not return_aux:
            return res
        out, aux_60d_logits = res
        unseen_attn_logits = None
        if self.unseen_attn_head is not None:
            v_exp = v_bn.unsqueeze(0).expand(b_size, -1, -1)
            h_att = out.h_attended
            unseen_attn_logits = self.unseen_attn_head(
                torch.cat([v_exp, h_att, v_exp * h_att], dim=-1)
            ).squeeze(-1)
        return {
            "output": out,
            "aux_60d_logits": aux_60d_logits,
            "unseen_attn_logits": unseen_attn_logits,
        }
