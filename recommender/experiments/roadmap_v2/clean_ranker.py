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

from typing import Optional, Sequence, Tuple, Union

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
        self.aux_60d_enabled = bool(aux_60d_enabled)

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

        # 2. Tokens 1..32: Direct precomputed Stage 1 v_bn[seq_product_ids] (no extra MLP!) + pos & type embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, DEFAULT_SEQUENCE_LENGTH, self.d_model) * 0.02)
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

        # 4. 1-Layer 8-Head Memory Self-Attention (Token 0 NEVER masked)
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
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[RankerOutput, Tuple[RankerOutput, torch.Tensor]]:
        b_size = customer_vector.size(0)
        seq_len = attention_mask.size(1)
        c_slate = catalog_product_vectors.size(0)

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

        memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)

        # Candidate Queries: Precomputed Stage 1 v_bn modified by residual MLP with dynamic 38D candidate info!
        v_exp = catalog_product_vectors.unsqueeze(0).expand(b_size, -1, -1)  # (B, 585, 128)
        z_cand = self.candidate_norm(
            v_exp + self.candidate_mlp(torch.cat([v_exp, candidate_features], dim=-1))
        )

        # Statistics-biased 8-head cross-attention
        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)
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
