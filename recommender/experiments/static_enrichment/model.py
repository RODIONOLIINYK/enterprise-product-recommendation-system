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
        tabular_features: torch.Tensor,
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

        if spec.is_control:
            # Exact S0 parameter construction & initialization order.
            customer = CustomerTower(
                text_dim=text_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_cities=num_cities,
                num_regions=num_regions,
                num_industries=num_industries,
                num_tabular_features=6,
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
                num_demand_features=num_demand_features,
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
                num_tabular_features=6,
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
                num_demand_features=num_demand_features,
                dropout=0.1,
            )
            self.product_bias = nn.Parameter(torch.zeros(num_catalog_products))
            self.familiarity_weights = (
                nn.Parameter(torch.zeros(num_familiarity_features))
                if num_familiarity_features else None
            )

    def product_vectors(self, product_inputs: Mapping[str, object]) -> torch.Tensor:
        demand = product_inputs["demand_features"]
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
        if self.towers is not None:
            return self.towers.customer_tower(
                inputs["text_embedding"],
                inputs["city_id"],
                inputs["region_id"],
                inputs["industry_id"],
                inputs["tabular_features"],
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
            inputs["tabular_features"],
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
