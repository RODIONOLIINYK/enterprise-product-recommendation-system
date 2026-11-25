"""Clean zero-dead-weight Stage 2 UPS-CANR ranker (modest_8h_ffn384_mlp384 topology) for Stage 5 v4.

Key architectural improvements over legacy BiggerStage5BaseNameRanker:
1. ZERO dead weights: permanently removes dead `product_text_proj` (32.8K zero-input params),
   dead `habit_proj`, and unused legacy `base.mlp` (135K frozen params).
2. Fresh Dynamic Customer Embedding (Token 0): injects all 19 point-in-time customer features
   (8 lifetime profile stats + 11 fresh cadence/overdue/30d/90d/financial/manager-change features)
   + `sales_branch` embedding DIRECTLY into Token 0's 128D representation AND into the 8-head
   statistics-attention bilinear bias — instead of copying customer-only constants 585x onto candidates!
3. Strictly Candidate / Customer-Candidate Features (38D): zero calendar month features, zero
   broadcast customer constants, zero exact binary/formula duplicates; keeps `base_name_orders_90d`
   and multi-value `user_viscosity_order_share`.
4. Strictly Non-Month, Non-Repeat History Statistics: selects only the 6 active sequence stat columns
   `(0, 1, 2, 5, 6, 8)` (permanently excluding `history_month` cols `3:5` and `history_repeat` col `7`).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.ranker import DEFAULT_SEQUENCE_LENGTH, RankerOutput
from recommender.sequences import DAYS_PER_YEAR, HISTORY_AGE_INDEX

from .config import SEED

# Exact 6 active history sequence stat indices (excluding month_sin=3, month_cos=4, is_repeat=7)
ACTIVE_SEQ_STAT_INDICES: Tuple[int, ...] = (0, 1, 2, 5, 6, 8)
NUM_ACTIVE_SEQ_STATS: int = len(ACTIVE_SEQ_STAT_INDICES)

# Clean 5-feature cadence-normalized self-attention bias over [Token 0, History 1..32]
SELF_ATTENTION_BIAS_NAMES: Tuple[str, ...] = (
    "earlier_hh",
    "later_hh",
    "t0_key_age",
    "key_rel_qty",
    "key_is_token0",
)
NUM_SELF_ATTENTION_BIAS_FEATURES: int = len(SELF_ATTENTION_BIAS_NAMES)
SELF_ATTENTION_BIAS_MODES: Tuple[str, ...] = ("none", "linear", "mlp")
CADENCE_FLOOR_DAYS: float = 7.0
CADENCE_DEFAULT_DAYS: float = 30.0
HH_CYCLE_HALF_SAT: float = 5.0
T0_CYCLE_HALF_SAT: float = 6.0


class CleanStage5BaseNameRanker(nn.Module):
    """Zero-dead-weight 8-head UPS-CANR (modest_8h_ffn384_mlp384) with current-stats Token 0 and 38D candidates."""

    def __init__(
        self,
        *,
        num_base_names: int,
        num_candidate_features: int = 38,
        num_customer_features: int = 11,
        num_pairwise_bias_features: int = 13,
        num_cities: int,
        num_regions: int,
        num_industries: int,
        num_sales_branches: int = 0,
        num_categories: int,
        num_subcategories: int,
        num_business_lines: int,
        text_dim: int = 1024,
        d_model: int = 128,
        num_heads: int = 8,
        memory_ffn_dim: int = 384,
        mlp_widths: Sequence[int] = (384, 192, 96),
        dropout: float = 0.12,
        aux_60d_enabled: bool = True,
        seed: int = SEED,
    ):
        super().__init__()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_candidate_features = int(num_candidate_features)
        self.num_customer_features = int(num_customer_features)
        self.num_pairwise_bias_features = int(num_pairwise_bias_features)
        self.num_sales_branches = int(num_sales_branches)
        self.aux_60d_enabled = bool(aux_60d_enabled)

        # ------------------------------------------------------------------
        # 1. Token 0: Dynamic Current-State Customer Profile Embedding (128D)
        # ------------------------------------------------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.city_embedding = nn.Embedding(int(num_cities), 16, padding_idx=0)
        self.region_embedding = nn.Embedding(int(num_regions), 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(int(num_industries), 16, padding_idx=0)
        self.sales_branch_embedding: Optional[nn.Embedding] = (
            nn.Embedding(self.num_sales_branches, 16) if self.num_sales_branches > 0 else None
        )

        tab_in = 16 * (4 if self.sales_branch_embedding is not None else 3) + self.num_customer_features
        self.customer_state_proj = nn.Sequential(
            nn.Linear(tab_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.profile_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(2, self.d_model)

        # ------------------------------------------------------------------
        # 2. Tokens 1..32: Clean Purchase History (Zero dead product_text_proj!)
        # ------------------------------------------------------------------
        self.product_embedding = nn.Embedding(int(num_base_names) + 1, 64, padding_idx=0)
        self.cat_embedding = nn.Embedding(int(num_categories), 32, padding_idx=0)
        self.subcat_embedding = nn.Embedding(int(num_subcategories), 16, padding_idx=0)
        self.line_embedding = nn.Embedding(int(num_business_lines), 16, padding_idx=0)
        self.token_proj = nn.Sequential(
            nn.Linear(64 + 32 + 16 + 16, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, DEFAULT_SEQUENCE_LENGTH, self.d_model) * 0.02)

        # ------------------------------------------------------------------
        # 3. 1-Layer 8-Head Memory Self-Attention (Token 0 NEVER masked)
        # ------------------------------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=int(memory_ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=1, enable_nested_tensor=False
        )

        # ------------------------------------------------------------------
        # 4. 8-Head Statistics-Biased Cross-Attention
        # ------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        # Bilinear bias: [1, cand_pairwise_13] @ W_h @ [active_seq_stats_6, customer_features_19]
        self.statistics_attention_weights = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.num_pairwise_bias_features + 1,
                NUM_ACTIVE_SEQ_STATS + self.num_customer_features,
            )
        )

        # ------------------------------------------------------------------
        # 5. Scoring Trunk (384 -> 192 -> 96 -> 1) + Training-Only 60d Aux Head
        # ------------------------------------------------------------------
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

    def encode_customer_memory(
        self,
        *,
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        customer_features: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: torch.Tensor,
        seq_category_ids: torch.Tensor,
        seq_subcategory_ids: torch.Tensor,
        seq_business_line_ids: torch.Tensor,
        sales_branch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_size = text_embedding.size(0)
        seq_len = attention_mask.size(1)

        # Token 0: Dynamic Current-State Customer Embedding (live customer_features ALWAYS active!)
        z_text = self.text_proj(text_embedding)
        tab_parts = [
            self.city_embedding(city_id),
            self.region_embedding(region_id),
            self.industry_embedding(industry_id),
        ]
        if self.sales_branch_embedding is not None:
            if sales_branch is None:
                raise ValueError("sales_branch required when num_sales_branches > 0.")
            tab_parts.append(self.sales_branch_embedding(sales_branch))
        tab_parts.append(customer_features)
        z_cust = self.customer_state_proj(torch.cat(tab_parts, dim=-1))
        t_0 = (self.profile_fusion(torch.cat([z_text, z_cust], dim=-1)) + self.type_embedding.weight[0]).unsqueeze(1)

        # Tokens 1..32: Clean Purchase History (zero dead product_text_proj / habit_proj)
        hist_raw = torch.cat(
            [
                self.product_embedding(seq_product_ids),
                self.cat_embedding(seq_category_ids),
                self.subcat_embedding(seq_subcategory_ids),
                self.line_embedding(seq_business_line_ids),
            ],
            dim=-1,
        )
        t_hist = self.token_proj(hist_raw)
        if seq_len <= self.pos_embedding.size(1):
            pos = self.pos_embedding[:, :seq_len, :]
        else:
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        t_hist = t_hist + pos + self.type_embedding.weight[1]

        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = torch.zeros((b_size, seq_len + 1), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 1:] = attention_mask == 0

        memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)
        return memory, k_mask

    def forward(
        self,
        *,
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        customer_features: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: torch.Tensor,
        seq_category_ids: torch.Tensor,
        seq_subcategory_ids: torch.Tensor,
        seq_business_line_ids: torch.Tensor,
        seq_purchase_dates: torch.Tensor,
        prediction_day: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_vectors: torch.Tensor,
        sales_branch: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[RankerOutput, Tuple[RankerOutput, torch.Tensor]]:
        b_size = text_embedding.size(0)
        c_slate = candidate_vectors.size(1)

        memory, key_padding_mask = self.encode_customer_memory(
            text_embedding=text_embedding,
            city_id=city_id,
            region_id=region_id,
            industry_id=industry_id,
            sales_branch=sales_branch,
            customer_features=customer_features,
            attention_mask=attention_mask,
            seq_product_ids=seq_product_ids,
            seq_category_ids=seq_category_ids,
            seq_subcategory_ids=seq_subcategory_ids,
            seq_business_line_ids=seq_business_line_ids,
        )

        # Build strictly non-month, non-repeat history statistics (6D) + live customer statistics (19D)
        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)
        seq_clean = sequence.clone()
        seq_clean[..., HISTORY_AGE_INDEX] = age
        active_hist_stats = seq_clean[..., ACTIVE_SEQ_STAT_INDICES]  # (B, 32, 6)

        key_statistics = torch.cat(
            (
                F.pad(customer_features[:, None, :], (NUM_ACTIVE_SEQ_STATS, 0)),
                F.pad(active_hist_stats, (0, self.num_customer_features)),
            ),
            dim=1,
        )  # (B, 33, 6 + 19)

        cand_bias_feats = candidate_features[..., : self.num_pairwise_bias_features]
        candidate_statistics = F.pad(cand_bias_feats, (1, 0), value=1.0)
        coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
        bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]
        bias = bias.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
        attention_bias = (
            bias.expand(-1, -1, c_slate, -1)
            .reshape(b_size * self.num_heads, c_slate, memory.size(1))
            .to(candidate_vectors.dtype)
        )

        h_att, attn_w = self.cross_attention(
            query=candidate_vectors,
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
            mask_exp = key_padding_mask.unsqueeze(1).expand(-1, c_slate, -1)
            attn_w = attn_w.masked_fill(mask_exp, 0.0)
            attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            token_0_weight = attn_w[..., 0]
            history_weights = attn_w[..., 1:].sum(dim=-1)

        hidden = torch.cat(
            [candidate_vectors, h_att, candidate_vectors * h_att, candidate_features],
            dim=-1,
        )
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
            z_candidate=candidate_vectors,
            token_0_weight=token_0_weight,
            history_weights=history_weights,
            memory=memory,
        )
        if return_aux and self.aux_60d_head is not None:
            aux_logits = self.aux_60d_head(hidden).squeeze(-1)
            if candidate_mask is not None:
                aux_logits = aux_logits.masked_fill(~candidate_mask, -torch.inf)
            return out, aux_logits
        return out


class SharedUVStage5Ranker(nn.Module):
    """Unified Stage-1-grounded Stage 2 UPS-CANR (v5_shared_uv):
      - Token 0 (Customer): residual MLP over [Stage-1 u_bn (128), current dynamic cust_11 (11)] -> 128D
      - Tokens 1..32 (History): direct precomputed Stage-1 v_bn[seq_product_ids] (128D) + pos_emb + type_emb[1] (NO redundant MLP!)
      - Candidate Queries Q: residual MLP over [Stage-1 v_bn (128), cand_38 (38)] -> 128D (incorporates dynamic candidate info before cross-attention!)
      - Zero scratch text_proj, city/region/industry, or product/cat/subcat/line embedding tables in Stage 2!
    """

    def __init__(
        self,
        *,
        num_candidate_features: int = 38,
        num_customer_features: int = 11,
        num_pairwise_bias_features: int = 13,
        d_model: int = 128,
        num_heads: int = 8,
        memory_ffn_dim: int = 384,
        memory_layers: int = 1,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        mlp_widths: Sequence[int] = (384, 192, 96),
        dropout: float = 0.12,
        aux_60d_enabled: bool = True,
        use_memory_encoder: bool = True,
        self_attention_bias_mode: str = "none",
        self_attention_bias_hidden: int = 16,
        seed: int = SEED,
    ):
        super().__init__()
        if int(memory_layers) < 1:
            raise ValueError(f"memory_layers must be >= 1, got {memory_layers}")
        if int(max_seq_len) < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
        if self_attention_bias_mode not in SELF_ATTENTION_BIAS_MODES:
            raise ValueError(
                f"Unsupported self_attention_bias_mode={self_attention_bias_mode!r}; "
                f"expected one of {SELF_ATTENTION_BIAS_MODES}"
            )
        if not use_memory_encoder and self_attention_bias_mode != "none":
            raise ValueError("self_attention_bias_mode requires use_memory_encoder=True.")
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.memory_layers = int(memory_layers)
        self.max_seq_len = int(max_seq_len)
        self.num_candidate_features = int(num_candidate_features)
        self.num_customer_features = int(num_customer_features)
        self.num_pairwise_bias_features = int(num_pairwise_bias_features)
        self.aux_60d_enabled = bool(aux_60d_enabled)
        self.use_memory_encoder = bool(use_memory_encoder)
        self.self_attention_bias_mode = str(self_attention_bias_mode)
        self.self_attention_bias_hidden = int(self_attention_bias_hidden)

        # 1. Token 0: Stage 1 customer vector u_bn (128) + fresh 11D current customer stats -> residual MLP
        self.customer_u_drop = nn.Dropout(dropout)
        self.customer_mlp = nn.Sequential(
            nn.Linear(self.d_model + self.num_customer_features, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.customer_norm = nn.LayerNorm(self.d_model)

        # 2. Tokens 1..S: Direct precomputed Stage 1 v_bn[seq_product_ids] (NO extra MLP!) + pos & type
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model) * 0.02)
        self.type_embedding = nn.Embedding(2, self.d_model)

        # 3. Candidate Query Modifier: Stage 1 v_bn (128) + dynamic 38D cand features -> residual MLP
        self.candidate_mlp = nn.Sequential(
            nn.Linear(self.d_model + self.num_candidate_features, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.candidate_norm = nn.LayerNorm(self.d_model)

        # 4. Optional N-Layer 8-Head Memory Self-Attention (Token 0 NEVER masked)
        if self.use_memory_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=int(memory_ffn_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
            )
            self.memory_encoder: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                enc_layer, num_layers=self.memory_layers, enable_nested_tensor=False
            )
        else:
            self.memory_encoder = None
        if self.self_attention_bias_mode == "linear":
            self.self_attention_bias_weights = nn.Parameter(
                torch.zeros(self.num_heads, NUM_SELF_ATTENTION_BIAS_FEATURES)
            )
            self.self_attention_bias_mlp = None
        elif self.self_attention_bias_mode == "mlp":
            self.self_attention_bias_weights = None
            hidden = self.self_attention_bias_hidden
            fc1 = nn.Linear(NUM_SELF_ATTENTION_BIAS_FEATURES, hidden)
            fc2 = nn.Linear(hidden, self.num_heads)
            nn.init.zeros_(fc2.weight)
            nn.init.zeros_(fc2.bias)
            self.self_attention_bias_mlp = nn.Sequential(fc1, nn.GELU(), fc2)
        else:
            self.self_attention_bias_weights = None
            self.self_attention_bias_mlp = None

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
                NUM_ACTIVE_SEQ_STATS + self.num_customer_features,
            )
        )

        # 6. Scoring Trunk (384 -> 192 -> 96 -> 1) + Training-Only 60d Aux Head
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

    def optimizer_param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Exempt compact self-attention bias parameters from AdamW weight decay."""
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("self_attention_bias_"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        groups = [{"params": decay_params, "weight_decay": float(weight_decay)}]
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    def _build_self_attention_pair_features(
        self,
        *,
        purchase_days: torch.Tensor,              # (B, S)
        prediction_day: torch.Tensor,             # (B,)
        attention_mask: torch.Tensor,             # (B, S)
        rel_qty: torch.Tensor,                    # (B, S)
        cust_median_gap_days: Optional[torch.Tensor],  # (B,)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build (B, S+1, S+1, 5) disjoint, customer-cadence-normalized pair features for self-attention bias.

        Let g_c = max(cust_median_gap_days, 7.0) be the customer's strictly-prior median basket gap.
        Features per (query i, key j) across [Token 0, History 1..S]:
          0. earlier_hh:    History -> History (i>=1, j>=1): r / (r + 5), r = max(0, d_i - d_j) / g_c
          1. later_hh:      History -> History (i>=1, j>=1): r / (r + 5), r = max(0, d_j - d_i) / g_c
          2. t0_key_age:    Token 0 -> History (i==0, j>=1): r / (r + 6), r = max(0, t - d_j) / g_c
          3. key_rel_qty:   Any     -> History (j>=1):       customer-relative event quantity of key j
          4. key_is_token0: History -> Token 0 (i>=1, j==0): 1.0 else 0.0
        """
        b_size, seq_len = attention_mask.shape
        mem_len = seq_len + 1
        device = attention_mask.device

        if cust_median_gap_days is None:
            cadence = torch.full((b_size,), CADENCE_DEFAULT_DAYS, dtype=purchase_days.dtype, device=device)
        else:
            cadence = cust_median_gap_days.to(device=device, dtype=purchase_days.dtype).clamp_min(CADENCE_FLOOR_DAYS)

        # 0 & 1: History -> History relative time in customer buying cycles (zero on Token 0 row/col)
        hh_delta = purchase_days[:, :, None] - purchase_days[:, None, :]  # (B, S, S)
        hh_earlier_r = hh_delta.clamp_min(0.0) / cadence[:, None, None]
        hh_later_r = (-hh_delta).clamp_min(0.0) / cadence[:, None, None]
        hh_earlier = hh_earlier_r / (hh_earlier_r + HH_CYCLE_HALF_SAT)
        hh_later = hh_later_r / (hh_later_r + HH_CYCLE_HALF_SAT)

        earlier_hh = torch.zeros((b_size, mem_len, mem_len), dtype=dtype, device=device)
        later_hh = torch.zeros((b_size, mem_len, mem_len), dtype=dtype, device=device)
        earlier_hh[:, 1:, 1:] = hh_earlier.to(dtype)
        later_hh[:, 1:, 1:] = hh_later.to(dtype)

        # 2: Token 0 -> History live age in customer buying cycles (zero for i>=1 and j==0)
        t0_age_days = (prediction_day[:, None].to(purchase_days.dtype) - purchase_days).clamp_min(0.0)
        t0_r = t0_age_days / cadence[:, None]
        t0_age_norm = t0_r / (t0_r + T0_CYCLE_HALF_SAT)
        t0_key_age = torch.zeros((b_size, mem_len, mem_len), dtype=dtype, device=device)
        t0_key_age[:, 0, 1:] = (t0_age_norm.to(dtype) * attention_mask.to(dtype))

        # 3: Customer-relative normalized event quantity of key j (zero for Token 0 / padding)
        valid_rel_qty = rel_qty.to(dtype) * attention_mask.to(dtype)
        key_rel_qty = F.pad(valid_rel_qty, (1, 0), value=0.0)[:, None, :].expand(b_size, mem_len, mem_len)

        # 4: History -> Token 0 indicator (1.0 only when i>=1 and j==0)
        key_is_token0 = torch.zeros((b_size, mem_len, mem_len), dtype=dtype, device=device)
        key_is_token0[:, 1:, 0] = 1.0

        return torch.stack(
            [earlier_hh, later_hh, t0_key_age, key_rel_qty, key_is_token0],
            dim=-1,
        )

    def _compute_self_attention_bias(self, pair_feats: torch.Tensor) -> torch.Tensor:
        """Map (B, L, L, 5) pair features to per-head bias (B, H, L, L)."""
        if self.self_attention_bias_weights is not None:
            return (pair_feats @ self.self_attention_bias_weights.transpose(0, 1)).permute(0, 3, 1, 2)
        if self.self_attention_bias_mlp is not None:
            return self.self_attention_bias_mlp(pair_feats).permute(0, 3, 1, 2)
        raise RuntimeError("Self-attention bias requested when self_attention_bias_mode='none'.")

    def forward(
        self,
        *,
        customer_vector: torch.Tensor,         # (B, 128) Stage 1 u_bn
        customer_features: torch.Tensor,       # (B, 11) current dynamic customer stats
        catalog_product_vectors: torch.Tensor, # (N_bn, 128) precomputed Stage 1 v_bn
        sequence: torch.Tensor,                # (B, 32, 9)
        attention_mask: torch.Tensor,          # (B, 32)
        seq_product_ids: torch.Tensor,         # (B, 32) 1-based into catalog_product_vectors (0=pad/OOV)
        seq_purchase_dates: torch.Tensor,      # (B, 32)
        prediction_day: torch.Tensor,          # (B,)
        candidate_features: torch.Tensor,      # (B, C, 38)
        candidate_mask: Optional[torch.Tensor] = None,
        seq_rel_qty: Optional[torch.Tensor] = None,            # (B, 32) customer-relative normalized quantity
        cust_median_gap_days: Optional[torch.Tensor] = None,   # (B,) strictly-prior customer median order gap
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[RankerOutput, Tuple[RankerOutput, torch.Tensor]]:
        b_size = customer_vector.size(0)
        seq_len = attention_mask.size(1)
        c_slate = catalog_product_vectors.size(0)

        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)
        rel_qty = seq_rel_qty if seq_rel_qty is not None else sequence[..., 0]

        # Token 0: Stage 1 u_bn modified by residual MLP with fresh 11D current customer stats
        u_in = self.customer_u_drop(customer_vector)
        t_0 = self.customer_norm(
            u_in + self.customer_mlp(torch.cat([u_in, customer_features], dim=-1))
        )
        t_0 = (t_0 + self.type_embedding.weight[0]).unsqueeze(1)  # (B, 1, 128)

        # Tokens 1..32: Direct precomputed Stage 1 v_bn lookup (NO extra MLP!) + pos & type embeddings
        v_padded = F.pad(catalog_product_vectors, (0, 0, 1, 0))  # index 0 = zeros
        t_hist = F.embedding(seq_product_ids, v_padded, padding_idx=0)  # (B, 32, 128)
        if seq_len <= self.pos_embedding.size(1):
            pos = self.pos_embedding[:, :seq_len, :]
        else:
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        t_hist = t_hist + pos + self.type_embedding.weight[1]

        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = torch.zeros((b_size, seq_len + 1), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 1:] = attention_mask == 0

        if self.memory_encoder is None:
            memory = m_raw
        elif self.self_attention_bias_mode != "none":
            pair_feats = self._build_self_attention_pair_features(
                purchase_days=purchase_days,
                prediction_day=prediction_day,
                attention_mask=attention_mask,
                rel_qty=rel_qty,
                cust_median_gap_days=cust_median_gap_days,
                dtype=m_raw.dtype,
            )
            self_bias = self._compute_self_attention_bias(pair_feats)
            self_bias = self_bias.masked_fill(k_mask[:, None, None, :], -torch.inf)
            self_attn_mask = self_bias.reshape(b_size * self.num_heads, seq_len + 1, seq_len + 1)
            memory = self.memory_encoder(m_raw, mask=self_attn_mask)
        else:
            memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)

        # Candidate Queries: Precomputed Stage 1 v_bn modified by residual MLP with dynamic 38D candidate info!
        v_exp = catalog_product_vectors.unsqueeze(0).expand(b_size, -1, -1)  # (B, 585, 128)
        z_cand = self.candidate_norm(
            v_exp + self.candidate_mlp(torch.cat([v_exp, candidate_features], dim=-1))
        )

        # Statistics-biased 8-head cross-attention
        seq_clean = sequence.clone()
        seq_clean[..., HISTORY_AGE_INDEX] = age
        active_hist_stats = seq_clean[..., ACTIVE_SEQ_STAT_INDICES]

        key_statistics = torch.cat(
            (
                F.pad(customer_features[:, None, :], (NUM_ACTIVE_SEQ_STATS, 0)),
                F.pad(active_hist_stats, (0, self.num_customer_features)),
            ),
            dim=1,
        )
        cand_bias_feats = candidate_features[..., : self.num_pairwise_bias_features]
        candidate_statistics = F.pad(cand_bias_feats, (1, 0), value=1.0)
        coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
        bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]
        bias = bias.masked_fill(k_mask[:, None, None, :], -torch.inf)
        attention_bias = (
            bias.expand(-1, -1, c_slate, -1)
            .reshape(b_size * self.num_heads, c_slate, memory.size(1))
            .to(z_cand.dtype)
        )

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
        if return_aux and self.aux_60d_head is not None:
            aux_logits = self.aux_60d_head(hidden).squeeze(-1)
            if candidate_mask is not None:
                aux_logits = aux_logits.masked_fill(~candidate_mask, -torch.inf)
            return out, aux_logits
        return out
