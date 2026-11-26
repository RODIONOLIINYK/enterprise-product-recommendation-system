"""Wave 2B E2E neural architecture (`arm4_e2e_unseen_attn_aux`), Option A 70/30 Unseen Head, Pure Unseen metrics & OOF affine blend.

Built exclusively on the active single-stage E2E architecture (`Wave2BE2ERanker`) with the canonical
42-feature pruned schema (`cand_42 + CUST_7 + SEQ_5`):
  - Pruned schemas: `cand_42` (drops dead col 22 + weak `mgr_growth`; adds `extra_cand_5`), `CUST_7`
    (drops cols `1, 7, 9, 10`), `SEQ_5` (`(0, 1, 5, 6, 8)`, keeping `live_age` col `1` and dropping
    `gap_days`).
  - Pure Query `Q = v_bn` (`pure_v_query=True`): removes `candidate_mlp([v_bn, cand_42])` before
    cross-attention so candidate queries remain purely in the semantic product space matching
    history keys `Tokens 1..S = v_bn`, removing the duplicate tabular shortcut and reducing CPU latency.
  - Full-42 Cross-Attention Bias (`num_pairwise_bias_features=42`): expands `statistics_attention_weights`
    from 13 to all 42 candidate features (+1 constant intercept) paired with `SEQ_5 + CUST_7`.
  - Option A Dedicated Inference Unseen-Only Discovery Head (`OptionAUnseenHead`):
    - Takes `[v_bn, h_att, v_bn * h_att, cand_non_repeat_37]` (uses pure `v_bn`, NEVER `z_cand`,
      so masked customer-specific tabular features cannot leak past feature-group dropout!).
    - Applies per-basket 70/30 unscaled feature-group dropout during `model.training`:
        * 70% of baskets: all 37 non-repeat `cand_42` columns active.
        * 30% of baskets: only the 7 global product-level columns (`UNSEEN_37_GLOBAL_LOCAL_INDICES`)
          active; the 30 customer-specific non-repeat columns are zeroed, matching real cold-start
          distributions without inverted-dropout rescaling distortion.
        * 100% of 37 non-repeat columns active at inference (`eval()`).
    - Supervised strictly on `unseen_bn & allowed_bn` (`L_unseen_next + 0.3 * L_unseen_60d`).
  - 2-Parameter Customer-Grouped OOF Affine Calibration (`fit_oof_affine_unseen_blend`):
    - Fits `(a, b)` with `a > 0` for `s_blend = where(is_repeat, s_main, a * s_unseen + b)`
      using deterministic 2-fold customer-grouped CV and exact
      Top-K pre-filtering mathematically identical to full-pool ranking.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.experiments.static_enrichment.model import StaticEnrichedProductTower
from recommender.ranker import DEFAULT_SEQUENCE_LENGTH, RankerOutput
from recommender.sequences import DAYS_PER_YEAR, HISTORY_AGE_INDEX

from .extra_features import (
    CAND_42_NON_REPEAT_INDICES,
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    SEQ_5_ACTIVE_INDICES,
    UNSEEN_37_GLOBAL_LOCAL_INDICES,
)


class OptionAUnseenHead(nn.Module):
    """Inference-capable Unseen-Only Discovery Head with per-basket 70/30 feature-group dropout."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        non_repeat_indices: Sequence[int] = CAND_42_NON_REPEAT_INDICES,
        global_local_indices: Sequence[int] = UNSEEN_37_GLOBAL_LOCAL_INDICES,
        mlp_widths: Sequence[int] = (384, 192, 96),
        cold_dropout_prob: float = 0.30,
        aux_60d_enabled: bool = True,
    ):
        super().__init__()
        if not (0.0 <= float(cold_dropout_prob) <= 1.0):
            raise ValueError(f"cold_dropout_prob must be in [0, 1], got {cold_dropout_prob}")
        self.d_model = int(d_model)
        self.non_repeat_indices: Tuple[int, ...] = tuple(int(i) for i in non_repeat_indices)
        self.global_local_indices: Tuple[int, ...] = tuple(int(i) for i in global_local_indices)
        self.num_non_repeat = len(self.non_repeat_indices)
        self.cold_dropout_prob = float(cold_dropout_prob)
        self.aux_60d_enabled = bool(aux_60d_enabled)

        global_mask = torch.zeros(self.num_non_repeat, dtype=torch.float32)
        global_mask[list(self.global_local_indices)] = 1.0
        self.register_buffer("global_only_mask", global_mask, persistent=False)

        total_in = self.d_model * 3 + self.num_non_repeat
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

    def apply_feature_group_dropout(
        self,
        unseen_tab: torch.Tensor,
        *,
        force_cold_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply unscaled 70/30 per-basket feature-group masking during training.

        Returns `(masked_unseen_tab, is_cold_branch_bool)` where `is_cold_branch_bool` has shape `(B,)`.
        At inference (`not self.training` and `force_cold_mask is None`), returns `unseen_tab` unchanged.
        """
        b_size = unseen_tab.size(0)
        device = unseen_tab.device
        if force_cold_mask is not None:
            cold_b = force_cold_mask.to(device=device, dtype=torch.bool)
        elif self.training and self.cold_dropout_prob > 0.0:
            cold_b = torch.rand(b_size, device=device) < self.cold_dropout_prob
        else:
            return unseen_tab, torch.zeros(b_size, dtype=torch.bool, device=device)

        if not bool(cold_b.any()):
            return unseen_tab, cold_b
        keep_full = torch.ones((1, 1, self.num_non_repeat), dtype=unseen_tab.dtype, device=device)
        keep_cold = self.global_only_mask.to(dtype=unseen_tab.dtype, device=device).view(1, 1, -1)
        per_basket_mask = torch.where(cold_b.view(b_size, 1, 1), keep_cold, keep_full)
        return unseen_tab * per_basket_mask, cold_b

    def forward(
        self,
        *,
        v_exp: torch.Tensor,               # (B, C, d_model) pure product vectors (NO z_cand!)
        h_att: torch.Tensor,               # (B, C, d_model) attended memory context
        candidate_features: torch.Tensor,  # (B, C, F_cand)
        force_cold_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        unseen_tab = candidate_features[..., list(self.non_repeat_indices)]
        unseen_tab, _ = self.apply_feature_group_dropout(unseen_tab, force_cold_mask=force_cold_mask)
        logits, aux = self.score_masked(v_exp, h_att, unseen_tab, return_aux)
        return (logits, aux) if return_aux else logits

    def score_masked(self, v_exp, h_att, unseen_tab, return_aux):
        hidden = torch.cat([v_exp, h_att, v_exp * h_att, unseen_tab], dim=-1)
        for block in self.mlp_blocks:
            hidden = block(hidden)
        logits = self.main_head(hidden).squeeze(-1)
        aux_logits = self.aux_60d_head(hidden).squeeze(-1) if return_aux and self.aux_60d_head is not None else None
        return logits, aux_logits


class _Wave2BCandidateAttention(nn.Module):
    def _attend_candidates(self, v_exp, candidate_features, memory, k_mask, key_statistics, return_diagnostics):
        b_size, c_slate = candidate_features.shape[:2]
        if self.pure_v_query:
            z_cand = v_exp
        else:
            assert self.candidate_mlp is not None and self.candidate_norm is not None
            z_cand = self.candidate_norm(v_exp + self.candidate_mlp(torch.cat([v_exp, candidate_features], dim=-1)))

        cand_bias_feats = candidate_features[..., : self.num_pairwise_bias_features]
        candidate_statistics = F.pad(cand_bias_feats, (1, 0), value=1.0)
        coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
        bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]
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

        return z_cand, h_att, attn_w

    def _score_unseen(self, v_exp, features, h_att, memory, k_mask, key_statistics, force_cold_mask, return_aux):
        head = self.option_a_head
        unseen_tab, cold = head.apply_feature_group_dropout(
            features[..., list(head.non_repeat_indices)], force_cold_mask=force_cold_mask
        )
        # Reuse the main attention for warm baskets. Cold baskets must mask both
        # the candidate query and its statistics bias before attending to memory.
        # Skip the `.any()` device-to-host sync entirely when cold dropout is disabled (e.g. TPU/XLA).
        if ((self.training and head.cold_dropout_prob > 0.0) or force_cold_mask is not None) and cold.any():
            cold_features = torch.zeros_like(features[cold])
            cold_features[..., list(head.non_repeat_indices)] = unseen_tab[cold]
            _, cold_context, _ = self._attend_candidates(
                v_exp[cold], cold_features, memory[cold], k_mask[cold], key_statistics[cold], False
            )
            h_att = h_att.clone()
            h_att[cold] = cold_context
        return head.score_masked(v_exp, h_att, unseen_tab, return_aux)


class Wave2BE2ERanker(_Wave2BCandidateAttention):
    """Wave 2B Single-Stage E2E ranker (`arm4` base) with `cand_42`, Pure-Q, Full-42 bias & Option A."""

    def __init__(
        self,
        *,
        product_attr_vocab_sizes: Mapping[str, int],
        num_cities: int,
        num_regions: int,
        num_industries: int,
        num_demand_features: int = 65,
        num_candidate_features: int = NUM_CAND_42_FEATURES,
        num_customer_features: int = NUM_CUST_7_FEATURES,
        num_pairwise_bias_features: int = 13,
        active_seq_stat_indices: Sequence[int] = SEQ_5_ACTIVE_INDICES,
        pure_v_query: bool = False,
        option_a_unseen_head: bool = False,
        unseen_dropout_cold_prob: float = 0.30,
        unseen_attn_aux_enabled: bool = True,
        text_dim: int = 1024,
        d_model: int = 128,
        num_heads: int = 8,
        memory_ffn_dim: int = 384,
        memory_layers: int = 1,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        mlp_widths: Sequence[int] = (384, 192, 96),
        product_tower_hidden_dim: Optional[int] = None,
        cust_tower_hidden_dim: Optional[int] = None,
        attr_emb_dim: int = 16,
        dropout: float = 0.12,
        aux_60d_enabled: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        if int(memory_layers) < 1 or int(max_seq_len) < 1:
            raise ValueError(f"memory_layers and max_seq_len must be >= 1, got {memory_layers}, {max_seq_len}")
        if int(attr_emb_dim) < 1:
            raise ValueError(f"attr_emb_dim must be >= 1, got {attr_emb_dim}")
        if product_tower_hidden_dim is not None and int(product_tower_hidden_dim) < 1:
            raise ValueError(f"product_tower_hidden_dim must be >= 1, got {product_tower_hidden_dim}")
        if cust_tower_hidden_dim is not None and int(cust_tower_hidden_dim) < 1:
            raise ValueError(f"cust_tower_hidden_dim must be >= 1, got {cust_tower_hidden_dim}")
        if not active_seq_stat_indices:
            raise ValueError("active_seq_stat_indices must be non-empty.")
        if int(num_pairwise_bias_features) > int(num_candidate_features):
            raise ValueError(
                f"num_pairwise_bias_features={num_pairwise_bias_features} > num_candidate_features={num_candidate_features}"
            )
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.memory_layers = int(memory_layers)
        self.max_seq_len = int(max_seq_len)
        self.product_tower_hidden_dim = (
            int(product_tower_hidden_dim) if product_tower_hidden_dim is not None else self.d_model
        )
        self.cust_tower_hidden_dim = (
            int(cust_tower_hidden_dim) if cust_tower_hidden_dim is not None else self.d_model
        )
        self.attr_emb_dim = int(attr_emb_dim)
        self.num_candidate_features = int(num_candidate_features)
        self.num_customer_features = int(num_customer_features)
        self.num_pairwise_bias_features = int(num_pairwise_bias_features)
        self.active_seq_stat_indices: Tuple[int, ...] = tuple(int(i) for i in active_seq_stat_indices)
        self.num_hist_key_stats = len(self.active_seq_stat_indices)
        self.pure_v_query = bool(pure_v_query)
        self.option_a_unseen_head_enabled = bool(option_a_unseen_head)
        self.aux_60d_enabled = bool(aux_60d_enabled)
        # When Option A is active, it supersedes the discarded training-only unseen_attn_head
        self.unseen_attn_aux_enabled = bool(unseen_attn_aux_enabled) and not self.option_a_unseen_head_enabled

        self.product_tower = StaticEnrichedProductTower(
            attribute_vocab_sizes=product_attr_vocab_sizes,
            text_dim=int(text_dim),
            output_dim=self.d_model,
            hidden_dim=self.product_tower_hidden_dim,
            num_demand_features=int(num_demand_features),
            attr_emb_dim=self.attr_emb_dim,
            dropout=0.1,
        )

        self.cust_text_proj = nn.Sequential(
            nn.Linear(int(text_dim), self.cust_tower_hidden_dim),
            nn.LayerNorm(self.cust_tower_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.city_embedding = nn.Embedding(int(num_cities), self.attr_emb_dim, padding_idx=0)
        self.region_embedding = nn.Embedding(int(num_regions), self.attr_emb_dim, padding_idx=0)
        self.industry_embedding = nn.Embedding(int(num_industries), self.attr_emb_dim, padding_idx=0)
        tab_in = self.attr_emb_dim * 3 + self.num_customer_features
        self.cust_tab_proj = nn.Sequential(
            nn.Linear(tab_in, self.cust_tower_hidden_dim),
            nn.LayerNorm(self.cust_tower_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.cust_tower_hidden_dim, self.cust_tower_hidden_dim),
            nn.LayerNorm(self.cust_tower_hidden_dim),
        )
        self.token0_fusion = nn.Sequential(
            nn.Linear(self.cust_tower_hidden_dim * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(2, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model) * 0.02)

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
        self.option_a_head: Optional[OptionAUnseenHead] = (
            OptionAUnseenHead(
                d_model=self.d_model,
                non_repeat_indices=CAND_42_NON_REPEAT_INDICES,
                global_local_indices=UNSEEN_37_GLOBAL_LOCAL_INDICES,
                mlp_widths=mlp_widths,
                cold_dropout_prob=unseen_dropout_cold_prob,
                aux_60d_enabled=self.aux_60d_enabled,
            )
            if self.option_a_unseen_head_enabled
            else None
        )

    @torch.no_grad()
    def precompute_product_vectors(self, prod_inputs: Mapping[str, Any]) -> torch.Tensor:
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
        company_text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
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
        force_cold_mask: Optional[torch.Tensor] = None,
        skip_main_head: bool = False,
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        if skip_main_head and self.option_a_head is None:
            raise ValueError("skip_main_head=True requires option_a_unseen_head=True.")
        b_size = company_text_embedding.size(0)
        seq_len = attention_mask.size(1)
        c_slate = candidate_features.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len={self.pos_embedding.size(1)}.")

        v_bn = self._compute_product_vectors(
            prod_inputs, precomputed_v_bn, train_bn_limit=train_bn_limit, total_bn=c_slate
        )
        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)

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

        v_padded = F.pad(v_bn, (0, 0, 1, 0))
        t_hist = F.embedding(seq_product_ids, v_padded, padding_idx=0)
        t_hist = t_hist + self.pos_embedding[:, :seq_len, :] + self.type_embedding.weight[1]

        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = torch.zeros((b_size, seq_len + 1), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 1:] = attention_mask == 0

        memory = torch.nan_to_num(self.memory_encoder(m_raw, src_key_padding_mask=k_mask), nan=0.0)

        v_exp = v_bn.unsqueeze(0).expand(b_size, -1, -1)
        seq_clean = sequence.clone()
        seq_clean[..., HISTORY_AGE_INDEX] = age
        active_hist_stats = seq_clean[..., list(self.active_seq_stat_indices)]

        key_statistics = torch.cat(
            (
                F.pad(customer_features[:, None, :], (self.num_hist_key_stats, 0)),
                F.pad(active_hist_stats, (0, self.num_customer_features)),
            ),
            dim=1,
        )
        z_cand, h_att, attn_w = self._attend_candidates(
            v_exp, candidate_features, memory, k_mask, key_statistics, return_diagnostics
        )

        token_0_weight = history_weights = None
        if return_diagnostics and attn_w is not None:
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            mask_exp = k_mask.unsqueeze(1).expand(-1, c_slate, -1)
            attn_w = attn_w.masked_fill(mask_exp, 0.0)
            attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            token_0_weight = attn_w[..., 0]
            history_weights = attn_w[..., 1:].sum(dim=-1)

        hidden = None
        out = None
        if not skip_main_head:
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

        unseen_logits = unseen_aux_60d_logits = None
        if self.option_a_head is not None:
            unseen_logits, unseen_aux_60d_logits = self._score_unseen(
                v_exp, candidate_features, h_att, memory, k_mask, key_statistics, force_cold_mask, return_aux
            )
            if candidate_mask is not None:
                unseen_logits = unseen_logits.masked_fill(~candidate_mask, -torch.inf)
                if unseen_aux_60d_logits is not None:
                    unseen_aux_60d_logits = unseen_aux_60d_logits.masked_fill(~candidate_mask, -torch.inf)

        aux_60d_logits = None
        unseen_attn_logits = None
        if return_aux:
            if not skip_main_head and hidden is not None and self.aux_60d_head is not None:
                aux_60d_logits = self.aux_60d_head(hidden).squeeze(-1)
                if candidate_mask is not None:
                    aux_60d_logits = aux_60d_logits.masked_fill(~candidate_mask, -torch.inf)
            if self.unseen_attn_head is not None:
                attn_feat = torch.cat([v_exp, h_att, v_exp * h_att], dim=-1)
                unseen_attn_logits = self.unseen_attn_head(attn_feat).squeeze(-1)

        return {
            "output": out,
            "v_bn": v_bn,
            "aux_60d_logits": aux_60d_logits,
            "unseen_logits": unseen_logits,
            "unseen_aux_60d_logits": unseen_aux_60d_logits,
            "unseen_attn_logits": unseen_attn_logits,
        }


def strip_wave2b_training_aux(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip training-only aux heads while PRESERVING inference `option_a_head` (except its 60d aux)."""
    return {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("aux_60d_head.")
        and not k.startswith("unseen_attn_head.")
        and not k.startswith("option_a_head.aux_60d_head.")
    }


def evaluate_pure_unseen_predictions(
    scores: torch.Tensor,
    *,
    allowed_bn: torch.Tensor,
    unseen_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    first_observed: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate Pure Unseen Ranking (masking all previously bought formulations out).

    Reports exact overall, returning-customer, and first-observed-customer Pure Unseen@10 / @25
    hits, recall, and NDCG@10 (over baskets containing >= 1 new target).
    """
    elig_unseen = allowed_bn.bool() & unseen_bn.bool()
    if not torch.isfinite(scores[elig_unseen]).all():
        raise ValueError("Eligible unseen formulations require finite scores.")
    masked = scores.masked_fill(~elig_unseen, -torch.inf)
    k_max = min(25, scores.size(1))
    order = masked.argsort(dim=-1, descending=True, stable=True)[:, :k_max]
    valid_top = elig_unseen.gather(1, order)
    hits_mat = (new_target_bn.bool() & elig_unseen).gather(1, order) & valid_top

    new_counts = new_target_bn.bool().sum(dim=-1)
    fo_np = first_observed.bool().cpu().numpy()
    ret_np = ~fo_np
    hits_np = hits_mat.cpu().numpy().astype(np.float64)
    new_cnt_np = new_counts.cpu().numpy()
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))

    def _cohort(mask_b: np.ndarray) -> Dict[str, Any]:
        tot_new = int(new_cnt_np[mask_b].sum())
        has_new = mask_b & (new_cnt_np > 0)
        n_new_baskets = int(has_new.sum())
        h10 = int(hits_np[mask_b, :10].sum())
        h25 = int(hits_np[mask_b, :k_max].sum())
        if n_new_baskets > 0:
            dcg = (hits_np[has_new, :10] * discounts[:min(10, k_max)]).sum(axis=1)
            k_ideal = np.minimum(new_cnt_np[has_new], 10)
            cum_disc = np.concatenate([[0.0], np.cumsum(discounts)])
            idcg = cum_disc[k_ideal]
            ndcg10 = float(np.mean(dcg / np.maximum(idcg, 1e-12)))
        else:
            ndcg10 = 0.0
        return {
            "baskets_with_new": n_new_baskets,
            "total_new_targets": tot_new,
            "pure_unseen_hits@10": h10,
            "pure_unseen_recall@10": float(h10 / tot_new) if tot_new > 0 else 0.0,
            "pure_unseen_hits@25": h25,
            "pure_unseen_recall@25": float(h25 / tot_new) if tot_new > 0 else 0.0,
            "pure_unseen_ndcg@10_new_baskets": ndcg10,
        }

    all_mask = np.ones(len(fo_np), dtype=bool)
    return {
        "overall": _cohort(all_mask),
        "returning": _cohort(ret_np),
        "first_observed": _cohort(fo_np),
    }


def _customer_fold_assignment(customer_ids: Sequence[str], n_folds: int = 2) -> np.ndarray:
    """Deterministic cross-process SHA-256 customer-grouped fold assignment."""
    ids = [str(cid).strip() for cid in customer_ids]
    keys = sorted(set(ids), key=lambda cid: (hashlib.sha256(cid.encode()).digest(), cid))
    if not 2 <= n_folds <= len(keys):
        raise ValueError("Customer-grouped calibration requires at least one customer per fold and two folds.")
    assignments = {cid: i % n_folds for i, cid in enumerate(keys)}
    return np.asarray([assignments[cid] for cid in ids], dtype=np.int64)


def _prepare_exact_top_candidates(
    s_main: np.ndarray,
    s_unseen: np.ndarray,
    allowed: np.ndarray,
    unseen: np.ndarray,
    target: np.ndarray,
    new_target: np.ndarray,
    top_k_each: int = 25,
) -> Dict[str, np.ndarray]:
    """Pre-extract Top-K repeat items (by `s_main`) + Top-K unseen items (by `s_unseen`) per basket.

    Because `a > 0` preserves relative ordering among unseen items and repeats are untouched, the
    Top-10 (and Top-25) of `where(is_repeat, s_main, a * s_unseen + b)` is mathematically identical
    to ranking all `C` candidates whenever `top_k_each >= 10` (25 gives exact Top-25 parity too).
    """
    n_rows, n_c = s_main.shape
    k_rep = min(int(top_k_each), n_c)
    k_uns = min(int(top_k_each), n_c)

    rep_elig = allowed & (~unseen)
    uns_elig = allowed & unseen

    rep_scores_full = np.where(rep_elig, s_main, -np.inf)
    uns_scores_full = np.where(uns_elig, s_unseen, -np.inf)

    rep_idx = np.argsort(-rep_scores_full, axis=1, kind="mergesort")[:, :k_rep]
    uns_idx = np.argsort(-uns_scores_full, axis=1, kind="mergesort")[:, :k_uns]

    rows = np.arange(n_rows)[:, None]
    rep_s = rep_scores_full[rows, rep_idx]
    uns_s = uns_scores_full[rows, uns_idx]
    rep_y = (target[rows, rep_idx] & rep_elig[rows, rep_idx]).astype(np.float64)
    uns_y = (target[rows, uns_idx] & uns_elig[rows, uns_idx]).astype(np.float64)
    uns_new = (new_target[rows, uns_idx] & uns_elig[rows, uns_idx]).astype(np.float64)
    tot_pos = target.sum(axis=1)

    return {
        "candidate_indices": np.concatenate([rep_idx, uns_idx], axis=1),
        "rep_s": rep_s,
        "uns_s": uns_s,
        "rep_y": rep_y,
        "uns_y": uns_y,
        "uns_new": uns_new,
        "tot_pos": tot_pos,
    }


def _eval_affine_ab_fast(
    rep_s: np.ndarray,
    uns_s: np.ndarray,
    cand_y: np.ndarray,
    cand_new: np.ndarray,
    cand_idx: np.ndarray,
    tot_pos: np.ndarray,
    a: float,
    b: float,
) -> Tuple[float, int, float]:
    """Fast exact Top-10 evaluation over pre-sliced `(N_sub, <=10)` repeat + unseen arrays."""
    cand_s = np.concatenate([rep_s, a * uns_s + b], axis=1)
    order = np.lexsort((cand_idx, -cand_s), axis=1)[:, :10]
    rows = np.arange(len(cand_s))[:, None]
    return _mixed_top10_metrics(cand_y[rows, order], cand_new[rows, order], tot_pos)


def _eval_affine_ab_on_subset(
    pre: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    a: float,
    b: float,
) -> Tuple[float, int, float]:
    """Compute exact `(ndcg@10, new_hits@10, recall@10)` on `row_mask` for `(a, b)` with `a > 0`."""
    k_rep = min(10, pre["rep_s"].shape[1])
    k_uns = min(10, pre["uns_s"].shape[1])
    rep_s = pre["rep_s"][row_mask, :k_rep]
    uns_s = pre["uns_s"][row_mask, :k_uns]
    rep_y = pre["rep_y"][row_mask, :k_rep]
    uns_y = pre["uns_y"][row_mask, :k_uns]
    uns_new = pre["uns_new"][row_mask, :k_uns]
    tot_pos = pre["tot_pos"][row_mask]
    n_rep_full = pre["rep_s"].shape[1]
    cand_idx = np.concatenate(
        [pre["candidate_indices"][row_mask, :k_rep], pre["candidate_indices"][row_mask, n_rep_full : n_rep_full + k_uns]],
        axis=1,
    )
    cand_y = np.concatenate([rep_y, uns_y], axis=1)
    cand_new = np.concatenate([np.zeros_like(rep_y), uns_new], axis=1)
    return _eval_affine_ab_fast(rep_s, uns_s, cand_y, cand_new, cand_idx, tot_pos, a, b)


def _mixed_top10_metrics(hits, new_hits, target_counts):
    """Match evaluate_base_name_predictions, including ineligible targets and zero-hit rows."""
    if not len(target_counts):
        return 0.0, 0, 0.0
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    dcg = (hits * discounts[:hits.shape[1]]).sum(axis=1)
    ideal = np.r_[0.0, discounts.cumsum()][np.minimum(target_counts, 10)]
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    recall = hits.sum(axis=1) / np.maximum(target_counts, 1)
    return float(ndcg.mean()), int(new_hits.sum()), float(recall.mean())


def _fit_affine_ab_on_subset(
    pre: Mapping[str, np.ndarray],
    s_main: np.ndarray,
    s_unseen: np.ndarray,
    elig_unseen: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[float, float, Tuple[float, int, float]]:
    """Coarse-to-fine deterministic search over `(a > 0, b)` maximizing `(ndcg@10, new_hits@10, rec@10)`.

    Centers the `b` search grid using the active competition boundary between finite Top-10 repeat
    `s_main` scores (`pre["rep_s"]`) and finite Top-10 unseen `s_unseen` scores (`pre["uns_s"]`),
    never relying on `s_main[elig_unseen]` (which can collapse arbitrarily negative under separated
    repeat-only Main Head training while being unused by `where(unseen, a * s_unseen + b, s_main)`).
    """
    k_rep = min(10, pre["rep_s"].shape[1])
    k_uns = min(10, pre["uns_s"].shape[1])
    n_rep_full = pre["rep_s"].shape[1]
    rep_top = pre["rep_s"][fit_mask, :k_rep]
    uns_top = pre["uns_s"][fit_mask, :k_uns]
    rep_y_sub = pre["rep_y"][fit_mask, :k_rep]
    uns_y_sub = pre["uns_y"][fit_mask, :k_uns]
    uns_new_sub = pre["uns_new"][fit_mask, :k_uns]
    tot_pos_sub = pre["tot_pos"][fit_mask]
    cand_idx_sub = np.concatenate(
        [pre["candidate_indices"][fit_mask, :k_rep], pre["candidate_indices"][fit_mask, n_rep_full : n_rep_full + k_uns]],
        axis=1,
    )
    cand_y_sub = np.concatenate([rep_y_sub, uns_y_sub], axis=1)
    cand_new_sub = np.concatenate([np.zeros_like(rep_y_sub), uns_new_sub], axis=1)

    def _eval(a_v: float, b_v: float) -> Tuple[float, int, float]:
        return _eval_affine_ab_fast(rep_top, uns_top, cand_y_sub, cand_new_sub, cand_idx_sub, tot_pos_sub, a_v, b_v)

    rep_vals = rep_top[np.isfinite(rep_top)]
    uns_vals = uns_top[np.isfinite(uns_top)]

    if len(rep_vals) and len(uns_vals):
        rep_ref = float(np.median(rep_vals))
        uns_ref = float(np.median(uns_vals))
        rep_std = float(np.std(rep_vals)) if len(rep_vals) > 1 else 1.0
        uns_std = float(np.std(uns_vals)) if len(uns_vals) > 1 else 1.0
    else:
        u_m = elig_unseen[fit_mask]
        rep_ref = float(s_main[fit_mask][u_m].mean()) if bool(u_m.any()) else 0.0
        uns_ref = float(s_unseen[fit_mask][u_m].mean()) if bool(u_m.any()) else 0.0
        rep_std, uns_std = 1.0, 1.0

    # Also include legacy unseen-mean center as a candidate seed when finite
    u_m = elig_unseen[fit_mask]
    legacy_main_u = float(s_main[fit_mask][u_m].mean()) if bool(u_m.any()) else rep_ref
    legacy_uns_u = float(s_unseen[fit_mask][u_m].mean()) if bool(u_m.any()) else uns_ref

    best_a, best_b = 1.0, rep_ref - uns_ref
    best_key = _eval(best_a, best_b)
    leg_b = legacy_main_u - legacy_uns_u
    leg_key = _eval(1.0, leg_b)
    if leg_key > best_key:
        best_key = leg_key
        best_a, best_b = 1.0, leg_b

    # Adaptive span covers full repeat-vs-unseen transition regardless of logit scale
    b_span = float(np.clip(max(6.0, 3.0 * (rep_std + uns_std)), 6.0, 24.0))
    a_list = np.linspace(0.2, 2.0, 19).tolist()
    if uns_std > 1e-3 and rep_std > 1e-3:
        a_scale = float(np.clip(rep_std / uns_std, 0.05, 5.0))
        if a_scale < 0.2 or a_scale > 2.0:
            a_list.append(a_scale)
    b_offsets = np.linspace(-b_span, b_span, 25)
    for a in a_list:
        b_center = rep_ref - float(a) * uns_ref
        for db in b_offsets:
            b = b_center + float(db)
            key = _eval(float(a), b)
            if key > best_key:
                best_key = key
                best_a, best_b = float(a), b

    # Multi-stage fine refinement around coarse winner (handles wider b_span cleanly)
    coarse_b_step = (2.0 * b_span) / 24.0
    for step_a, step_b in (
        (0.05, max(0.25, coarse_b_step * 0.5)),
        (0.02, 0.08),
        (0.01, 0.03),
    ):
        cand_as = [max(0.05, best_a + da) for da in (-2 * step_a, -step_a, 0.0, step_a, 2 * step_a)]
        cand_bs = [best_b + db for db in (-2 * step_b, -step_b, 0.0, step_b, 2 * step_b)]
        for a in cand_as:
            for b in cand_bs:
                key = _eval(float(a), float(b))
                if key > best_key:
                    best_key = key
                    best_a, best_b = float(a), float(b)

    return best_a, best_b, best_key


def fit_oof_affine_unseen_blend(
    s_main: torch.Tensor,
    s_unseen: torch.Tensor,
    *,
    allowed_bn: torch.Tensor,
    unseen_bn: torch.Tensor,
    target_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    customer_ids: Sequence[str],
    n_folds: int = 2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Produce strictly out-of-fold 2-param affine blended scores `where(is_repeat, s_main, a*s_unseen + b)`.

    Splits customers into `n_folds` disjoint deterministic SHA-256 folds so every basket's `(a, b)`
    is fit strictly on OTHER customers. Also fits and reports the full-validation `(a, b)` pair for
    deployment inspection.
    """
    s_main_np = s_main.detach().cpu().numpy().astype(np.float64)
    s_unseen_np = s_unseen.detach().cpu().numpy().astype(np.float64)
    allowed_np = allowed_bn.detach().cpu().numpy().astype(bool)
    unseen_np = unseen_bn.detach().cpu().numpy().astype(bool)
    target_np = target_bn.detach().cpu().numpy().astype(bool)
    new_target_np = new_target_bn.detach().cpu().numpy().astype(bool)
    elig_unseen_np = allowed_np & unseen_np

    folds = _customer_fold_assignment(customer_ids, n_folds=n_folds)
    pre = _prepare_exact_top_candidates(
        s_main_np, s_unseen_np, allowed_np, unseen_np, target_np, new_target_np, top_k_each=25
    )

    oof_blended = np.empty_like(s_main_np)
    fold_records = []
    for f in range(int(n_folds)):
        val_m = folds == f
        fit_m = ~val_m
        a_f, b_f, (fit_ndcg, fit_new, fit_rec) = _fit_affine_ab_on_subset(
            pre, s_main_np, s_unseen_np, elig_unseen_np, fit_m
        )
        s_u_safe = np.where(elig_unseen_np[val_m], s_unseen_np[val_m], 0.0)
        s_m_safe = np.where(allowed_np[val_m], s_main_np[val_m], 0.0)
        oof_scores_f = np.where(unseen_np[val_m], a_f * s_u_safe + b_f, s_m_safe)
        oof_scores_f = np.where(allowed_np[val_m], oof_scores_f, -np.inf)
        oof_blended[val_m] = oof_scores_f
        fold_records.append(
            {
                "fold": f,
                "oof_baskets": int(val_m.sum()),
                "fit_baskets": int(fit_m.sum()),
                "a": float(a_f),
                "b": float(b_f),
                "fit_ndcg@10": float(fit_ndcg),
                "fit_new_hits@10": int(fit_new),
                "fit_recall@10": float(fit_rec),
            }
        )

    all_mask = np.ones(len(customer_ids), dtype=bool)
    a_full, b_full, (full_ndcg, full_new, full_rec) = _fit_affine_ab_on_subset(
        pre, s_main_np, s_unseen_np, elig_unseen_np, all_mask
    )
    meta = {
        "n_folds": int(n_folds),
        "customer_grouped_oof": True,
        "folds": fold_records,
        "deployment_full_fit": {
            "a": float(a_full),
            "b": float(b_full),
            "ndcg@10": float(full_ndcg),
            "new_hits@10": int(full_new),
            "recall@10": float(full_rec),
        },
    }
    return torch.from_numpy(oof_blended), meta
