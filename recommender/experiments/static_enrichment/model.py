"""Experimental static-enrichment Two-Tower models with strict demand-disabled guard."""

from typing import Mapping, Optional, Sequence
import torch
from torch import nn
from torch.nn import functional as F

from recommender.models import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEMPERATURE,
    OUTPUT_DIM,
    TEXT_DIM,
    CustomerTower,
    ProductTower,
    TwoTowerModel,
)
from recommender.ranker import RankerOutput
from recommender.sequences import normalize_purchase_history_gaps

from .config import VariantSpec


class StaticEnrichedProductTower(nn.Module):
    """Product tower supporting arbitrary training-fitted static categorical embeddings and zeroed demand."""

    def __init__(
        self,
        attribute_vocab_sizes: Mapping[str, int],
        *,
        text_dim: int = TEXT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 128,
        num_demand_features: int = 18,
        attr_emb_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_demand_features = num_demand_features
        self.attr_order = tuple(attribute_vocab_sizes.keys())

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # No padding_idx: bucket 0 (<NONE>) and bucket 1 (<OOV>) are explicit learned representations.
        self.attr_embeddings = nn.ModuleDict({
            name: nn.Embedding(int(size), attr_emb_dim)
            for name, size in attribute_vocab_sizes.items()
        })
        total_input_dim = hidden_dim + attr_emb_dim * len(self.attr_order) + num_demand_features
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        demand_features: torch.Tensor,
        attr_ids: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        t_proj = self.text_proj(text_embedding)
        parts = [t_proj]
        if self.attr_order:
            if attr_ids is None:
                raise ValueError("Attribute IDs are required when product attribute embeddings are enabled.")
            for name in self.attr_order:
                parts.append(self.attr_embeddings[name](attr_ids[name]))
        parts.append(demand_features)
        v_p = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(v_p, p=2, dim=-1)


class StaticEnrichedCustomerTower(nn.Module):
    """Customer tower matching S0 with optional static customer categorical embeddings."""

    def __init__(
        self,
        *,
        text_dim: int = TEXT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 128,
        num_cities: int,
        num_regions: int,
        num_industries: int,
        extra_attr_vocab_sizes: Optional[Mapping[str, int]] = None,
        extra_attr_emb_dim: int = 16,
        num_tabular_features: int = 6,
        history_gap_reference_days: float = 74.0,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.history_gap_reference_days = history_gap_reference_days
        self.num_tabular_features = num_tabular_features
        self.extra_attr_order = tuple((extra_attr_vocab_sizes or {}).keys())

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.city_embedding = nn.Embedding(num_cities, 16, padding_idx=0)
        self.region_embedding = nn.Embedding(num_regions, 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(num_industries, 16, padding_idx=0)
        self.extra_attr_embeddings = nn.ModuleDict({
            name: nn.Embedding(int(size), extra_attr_emb_dim)
            for name, size in (extra_attr_vocab_sizes or {}).items()
        })

        tab_input_dim = 16 * 3 + extra_attr_emb_dim * len(self.extra_attr_order) + num_tabular_features
        self.tabular_proj = nn.Sequential(
            nn.Linear(tab_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(output_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.history_pooling = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.late_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        tabular_features: Optional[torch.Tensor],
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: torch.Tensor,
        *,
        catalog_product_vectors: torch.Tensor,
        seq_purchase_dates: torch.Tensor,
        extra_attr_ids: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = text_embedding.size(0)
        h_text = self.text_proj(text_embedding)

        tab_parts = [
            self.city_embedding(city_id),
            self.region_embedding(region_id),
            self.industry_embedding(industry_id),
        ]
        if self.extra_attr_order:
            if extra_attr_ids is None:
                raise ValueError("Extra customer attribute IDs required for this variant.")
            for name in self.extra_attr_order:
                tab_parts.append(self.extra_attr_embeddings[name](extra_attr_ids[name]))
        if self.num_tabular_features > 0:
            if tabular_features is None:
                raise ValueError("Customer tabular features required when num_tabular_features > 0.")
            tab_parts.append(tabular_features[..., :self.num_tabular_features])
        h_tab = self.tabular_proj(torch.cat(tab_parts, dim=-1))

        catalog_with_padding = F.pad(catalog_product_vectors, (0, 0, 1, 0))
        history_products = F.embedding(seq_product_ids, catalog_with_padding, padding_idx=0)
        quantity = sequence[..., :1]
        time_feature = normalize_purchase_history_gaps(
            seq_purchase_dates, attention_mask, self.history_gap_reference_days
        )
        tokens = self.token_proj(torch.cat([history_products, quantity, time_feature.unsqueeze(-1)], dim=-1))
        seq_len = tokens.size(1)
        out_seq = tokens + self.pos_embedding[:, :seq_len, :]
        out_seq = self.transformer(out_seq, src_key_padding_mask=(attention_mask == 0))
        out_seq = torch.nan_to_num(out_seq, nan=0.0)

        mask_f = attention_mask.unsqueeze(-1).float()
        has_history = (attention_mask.sum(dim=1, keepdim=True) > 0).float()
        mean_vec = (out_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        last_idx_exp = last_idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
        last_vec = out_seq.gather(1, last_idx_exp).squeeze(1)
        h_hist = self.history_pooling(torch.cat([mean_vec, last_vec], dim=-1)) * has_history

        u_c = self.late_fusion(torch.cat([h_text, h_tab, h_hist], dim=-1))
        return F.normalize(u_c, p=2, dim=-1)


class StaticEnrichedRetriever(nn.Module):
    """Wrapper unifying C0 (exact S0 TwoTowerModel) and enriched static variants."""

    def __init__(
        self,
        spec: VariantSpec,
        *,
        num_cities: int,
        num_regions: int,
        num_industries: int,
        num_catalog_products: int,
        num_demand_features: int = 18,
        product_attr_vocab_sizes: Optional[Mapping[str, int]] = None,
        customer_attr_vocab_sizes: Optional[Mapping[str, int]] = None,
        text_dim: int = TEXT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 128,
        temperature: float = DEFAULT_TEMPERATURE,
        num_familiarity_features: int = 4,
    ):
        super().__init__()
        self.spec = spec
        self.temperature = temperature
        cust_tab_dim = int(spec.customer_tabular_features)
        prod_dem_dim = int(spec.product_demand_features)

        if spec.is_control:
            # Exact S0 parameter construction & initialization order (or clean 0-tab / 12-container variant).
            customer = CustomerTower(
                text_dim=text_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_cities=num_cities,
                num_regions=num_regions,
                num_industries=num_industries,
                num_tabular_features=cust_tab_dim,
                catalog_history_encoder="attention",
                history_gap_reference_days=74.0,
                max_seq_len=32,
                num_heads=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
                seq_feature_dim=2,
            )
            product = ProductTower(
                text_dim=text_dim,
                output_dim=output_dim,
                num_brands=0,
                num_categories=0,
                num_business_lines=0,
                num_subcategories=0,
                num_demand_features=prod_dem_dim,
                hidden_dim=hidden_dim,
                dropout=0.1,
            )
            self.towers = TwoTowerModel(
                customer,
                product,
                num_catalog_products=num_catalog_products,
                temperature=temperature,
                num_familiarity_features=num_familiarity_features,
            )
            self.customer_tower = None
            self.product_tower = None
            self.product_bias = None
            self.familiarity_weights = None
        else:
            p_sizes = {
                name: int((product_attr_vocab_sizes or {})[name])
                for name in spec.product_embeddings
            }
            c_sizes = {
                name: int((customer_attr_vocab_sizes or {})[name])
                for name in spec.customer_embeddings
            }
            self.towers = None
            self.customer_tower = StaticEnrichedCustomerTower(
                text_dim=text_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_cities=num_cities,
                num_regions=num_regions,
                num_industries=num_industries,
                extra_attr_vocab_sizes=c_sizes,
                num_tabular_features=cust_tab_dim,
                history_gap_reference_days=74.0,
                max_seq_len=32,
                num_heads=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
            )
            self.product_tower = StaticEnrichedProductTower(
                p_sizes,
                text_dim=text_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_demand_features=prod_dem_dim,
                dropout=0.1,
            )
            self.product_bias = nn.Parameter(torch.zeros(num_catalog_products))
            self.familiarity_weights = (
                nn.Parameter(torch.zeros(num_familiarity_features))
                if num_familiarity_features else None
            )

    def product_vectors(self, product_inputs: Mapping[str, object]) -> torch.Tensor:
        demand = product_inputs["demand_features"]
        if self.spec.product_demand_features == 12:
            # Clean ProductTower: strictly 12D static container_type one-hot only (no demand inputs).
            demand = demand[:, 6:] if demand.size(1) == 18 else demand
        else:
            # Enforce disabled_v1 policy: first 6 demand dimensions are always zero.
            demand = torch.cat([torch.zeros_like(demand[:, :6]), demand[:, 6:]], dim=-1)
        if self.towers is not None:
            return self.towers.product_tower(
                product_inputs["text_embedding"],
                product_inputs["brand_id"],
                product_inputs["category_id"],
                product_inputs.get("subcategory_id"),
                product_inputs["business_line_id"],
                demand,
                container_id=product_inputs.get("container_id"),
            )
        return self.product_tower(
            product_inputs["text_embedding"],
            demand,
            attr_ids=product_inputs.get("attr_ids"),
        )

    def encode_purchase(self, inputs: Mapping[str, object], vectors: torch.Tensor) -> torch.Tensor:
        tab = inputs["tabular_features"]
        if self.spec.customer_tabular_features == 0:
            tab = tab[..., :0]
        if self.towers is not None:
            return self.towers.customer_tower(
                inputs["text_embedding"],
                inputs["city_id"],
                inputs["region_id"],
                inputs["industry_id"],
                tab,
                inputs["sequence"],
                inputs["attention_mask"],
                inputs["seq_product_ids"],
                catalog_product_vectors=vectors,
                seq_purchase_dates=inputs["seq_purchase_dates"],
            ).unsqueeze(1)
        return self.customer_tower(
            inputs["text_embedding"],
            inputs["city_id"],
            inputs["region_id"],
            inputs["industry_id"],
            tab,
            inputs["sequence"],
            inputs["attention_mask"],
            inputs["seq_product_ids"],
            catalog_product_vectors=vectors,
            seq_purchase_dates=inputs["seq_purchase_dates"],
            extra_attr_ids=inputs.get("extra_attr_ids"),
        ).unsqueeze(1)

    def score_cached(
        self,
        customer_vectors: torch.Tensor,
        product_vectors: torch.Tensor,
        product_indices: Optional[torch.Tensor] = None,
        familiarity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.towers is not None:
            u = customer_vectors.squeeze(1) if customer_vectors.ndim == 3 else customer_vectors
            return self.towers.compute_logits(
                u,
                product_vectors,
                product_indices=product_indices,
                familiarity_features=familiarity,
            )
        u = customer_vectors.squeeze(1) if customer_vectors.ndim == 3 else customer_vectors
        scores = torch.matmul(u, product_vectors.t()) / self.temperature
        if product_indices is not None:
            scores = scores + self.product_bias[product_indices].unsqueeze(0)
        if self.familiarity_weights is not None:
            if familiarity is None:
                raise ValueError("Familiarity features required.")
            scores = scores + familiarity.to(scores) @ self.familiarity_weights
        return scores


class TransferredStatsDynamicRanker(nn.Module):
    """UPS-CANR dynamic_features ranker extended with the 6 transferred point-in-time customer stats."""

    def __init__(
        self,
        dynamic_ranker: nn.Module,
        *,
        num_customer_stats: int = 6,
    ):
        super().__init__()
        if getattr(dynamic_ranker, "mode", None) != "dynamic_features":
            raise ValueError("TransferredStatsDynamicRanker requires a dynamic_features CandidateContextRanker.")
        self.ranker = dynamic_ranker
        self.num_customer_stats = int(num_customer_stats)

        old = self.ranker.head[0]
        expanded = nn.Linear(old.in_features + self.num_customer_stats, old.out_features)
        with torch.no_grad():
            expanded.weight[:, : old.in_features].copy_(old.weight)
            expanded.weight[:, old.in_features :].zero_()
            expanded.bias.copy_(old.bias)
        self.ranker.head[0] = expanded

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        transferred_customer_stats: torch.Tensor,
        cached_customer_vectors: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RankerOutput:
        if transferred_customer_stats.shape[-1] != self.num_customer_stats:
            raise ValueError(
                f"Expected {self.num_customer_stats} transferred customer stats, got {transferred_customer_stats.shape[-1]}."
            )
        features = pairwise_features[..., :13].clone()
        encoded = self.ranker.base.encode_candidate_features(pairwise_features=features, **kwargs)
        extra_candidate = pairwise_features[..., self.ranker.extra_columns]
        stats_expanded = transferred_customer_stats.unsqueeze(1).expand(-1, pairwise_features.size(1), -1)
        combined = torch.cat([encoded.features, extra_candidate, stats_expanded], dim=-1)
        scores = self.ranker.head(combined).squeeze(-1)
        return RankerOutput(
            scores=scores,
            raw_scores=scores,
            attn_weights=encoded.attn_weights,
            h_attended=encoded.h_attended,
            z_candidate=encoded.z_candidate,
            token_0_weight=encoded.token_0_weight,
            history_weights=encoded.history_weights,
        )


class PrecomputedEmbeddingMLPRanker(nn.Module):
    """Lightweight scoring MLP over ONLY precomputed u_c, v_p, interactions, and 47 changeable features."""

    def __init__(
        self,
        embedding_dim: int = OUTPUT_DIM,
        num_pairwise_features: int = 41,
        num_customer_stats: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_pairwise_features = int(num_pairwise_features)
        self.num_customer_stats = int(num_customer_stats)
        in_dim = self.embedding_dim * 4 + 1 + self.num_pairwise_features + self.num_customer_stats
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        candidate_vectors: torch.Tensor,
        pairwise_features: torch.Tensor,
        cached_customer_vectors: torch.Tensor,
        transferred_customer_stats: torch.Tensor,
        **kwargs,
    ) -> RankerOutput:
        if cached_customer_vectors is None:
            raise ValueError("PrecomputedEmbeddingMLPRanker requires cached_customer_vectors.")
        if transferred_customer_stats.shape[-1] != self.num_customer_stats:
            raise ValueError(
                f"Expected {self.num_customer_stats} transferred customer stats, got {transferred_customer_stats.shape[-1]}."
            )
        v = candidate_vectors
        u = cached_customer_vectors.unsqueeze(1).expand_as(v)
        dot = (u * v).sum(dim=-1, keepdim=True)
        stats_expanded = transferred_customer_stats.unsqueeze(1).expand(-1, v.size(1), -1)
        x = torch.cat(
            [
                u,
                v,
                u * v,
                (u - v).abs(),
                dot,
                pairwise_features[..., : self.num_pairwise_features],
                stats_expanded,
            ],
            dim=-1,
        )
        scores = self.head(x).squeeze(-1)
        return RankerOutput(
            scores=scores,
            raw_scores=scores,
            attn_weights=None,
            h_attended=u,
            z_candidate=v,
            token_0_weight=None,
            history_weights=None,
        )


COMPACT_SELECTOR_RECIPES = ("uv_47", "uv_47_ref")


class CompactStage1Selector(nn.Module):
    """Pointwise Stage 1 full-pool selector over cached u_c, v_p, and 47 dynamic features (+ optional ref)."""

    def __init__(
        self,
        recipe: str = "uv_47",
        *,
        embedding_dim: int = OUTPUT_DIM,
        num_dynamic_features: int = 41,
        num_customer_stats: int = 6,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        if recipe not in COMPACT_SELECTOR_RECIPES:
            raise ValueError(f"Unsupported CompactStage1Selector recipe {recipe!r}; expected one of {COMPACT_SELECTOR_RECIPES}.")
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width.")
        self.recipe = recipe
        self.embedding_dim = int(embedding_dim)
        self.num_dynamic_features = int(num_dynamic_features)
        self.num_customer_stats = int(num_customer_stats)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)

        blocks = [
            ("u", self.embedding_dim),
            ("v", self.embedding_dim),
            ("features_41", self.num_dynamic_features),
            ("customer_stats_6", self.num_customer_stats),
        ]
        if recipe == "uv_47_ref":
            blocks.append(("reference_score_over_10", 1))
        self.ordered_input_blocks = tuple(blocks)
        self.input_dim = sum(dim for _, dim in self.ordered_input_blocks)

        layers = []
        prev_dim = self.input_dim
        for width in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def input_contract(self) -> dict:
        return {
            "recipe": self.recipe,
            "input_dim": self.input_dim,
            "ordered_input_blocks": [{"name": name, "dim": dim} for name, dim in self.ordered_input_blocks],
        }

    def _score_chunk(
        self,
        u_expanded: torch.Tensor,
        v_chunk: torch.Tensor,
        f_chunk: torch.Tensor,
        stats_expanded: torch.Tensor,
        ref_chunk: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [u_expanded, v_chunk, f_chunk, stats_expanded]
        if self.recipe == "uv_47_ref":
            if ref_chunk is None:
                raise ValueError("reference_scores must be provided for recipe='uv_47_ref'.")
            parts.append(ref_chunk)
        x = torch.cat(parts, dim=-1)
        return self.head(x).squeeze(-1)

    def forward(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        features_41: torch.Tensor,
        customer_stats_6: torch.Tensor,
        reference_scores: Optional[torch.Tensor] = None,
        product_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if u.ndim != 2 or u.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected u shape (B, {self.embedding_dim}), got {tuple(u.shape)}.")
        if customer_stats_6.ndim != 2 or customer_stats_6.shape[-1] != self.num_customer_stats:
            raise ValueError(
                f"Expected customer_stats_6 shape (B, {self.num_customer_stats}), got {tuple(customer_stats_6.shape)}."
            )
        if features_41.ndim != 3 or features_41.shape[-1] != self.num_dynamic_features:
            raise ValueError(
                f"Expected features_41 shape (B, P, {self.num_dynamic_features}), got {tuple(features_41.shape)}."
            )
        batch_size, num_products, _ = features_41.shape
        if v.ndim == 2:
            if v.shape != (num_products, self.embedding_dim):
                raise ValueError(f"Expected v shape ({num_products}, {self.embedding_dim}), got {tuple(v.shape)}.")
        elif v.ndim == 3:
            if v.shape != (batch_size, num_products, self.embedding_dim):
                raise ValueError(
                    f"Expected v shape ({batch_size}, {num_products}, {self.embedding_dim}), got {tuple(v.shape)}."
                )
        else:
            raise ValueError(f"Expected 2D or 3D v tensor, got shape {tuple(v.shape)}.")

        ref_scaled: Optional[torch.Tensor] = None
        if self.recipe == "uv_47_ref":
            if reference_scores is None:
                raise ValueError("reference_scores must be provided for recipe='uv_47_ref'.")
            if reference_scores.ndim == 2:
                ref_scaled = (reference_scores / 10.0).unsqueeze(-1)
            elif reference_scores.ndim == 3 and reference_scores.shape[-1] == 1:
                ref_scaled = reference_scores / 10.0
            else:
                raise ValueError(f"Unexpected reference_scores shape {tuple(reference_scores.shape)}.")
            if ref_scaled.shape[:2] != (batch_size, num_products):
                raise ValueError(
                    f"Expected reference_scores batch/product dims ({batch_size}, {num_products}), got {tuple(ref_scaled.shape[:2])}."
                )

        if product_chunk_size is not None and int(product_chunk_size) > 0 and int(product_chunk_size) < num_products:
            chunk = int(product_chunk_size)
            outputs = []
            for start in range(0, num_products, chunk):
                end = min(start + chunk, num_products)
                span = end - start
                u_chunk = u.unsqueeze(1).expand(batch_size, span, self.embedding_dim)
                stats_chunk = customer_stats_6.unsqueeze(1).expand(batch_size, span, self.num_customer_stats)
                v_chunk = (
                    v[start:end].unsqueeze(0).expand(batch_size, span, self.embedding_dim)
                    if v.ndim == 2
                    else v[:, start:end, :]
                )
                f_chunk = features_41[:, start:end, :]
                r_chunk = None if ref_scaled is None else ref_scaled[:, start:end, :]
                outputs.append(self._score_chunk(u_chunk, v_chunk, f_chunk, stats_chunk, r_chunk))
            return torch.cat(outputs, dim=1)

        u_exp = u.unsqueeze(1).expand(batch_size, num_products, self.embedding_dim)
        stats_exp = customer_stats_6.unsqueeze(1).expand(batch_size, num_products, self.num_customer_stats)
        v_exp = (
            v.unsqueeze(0).expand(batch_size, num_products, self.embedding_dim)
            if v.ndim == 2
            else v
        )
        return self._score_chunk(u_exp, v_exp, features_41, stats_exp, ref_scaled)

