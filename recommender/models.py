"""Modular Two-Tower Retrieval Model for Next-Basket Recommendation (Phase 7).

Implements:
1. Modular Customer Tower (Late Fusion):
   - Module 1 (Static Company Text): 1024-d BAAI/bge-m3 projection -> 128-d
   - Module 2 (Tabular & Profile): City, region, industry embeddings + 8 point-in-time
     lifetime tabular features -> 128-d
   - Module 3 (Purchase History Transformer): TransformerEncoder over 32 purchase events
     with dual masked mean + last-event pooling -> 128-d
   - Late Fusion MLP: Concatenates [Text (128), Tabular (128), History (128)] -> 384 -> 256 -> 128
     producing L2-normalized 128-d customer vector u_c.
2. Product Tower:
   - Encodes product text (BAAI/bge-m3), catalog attributes (brand, category, subcat, line),
     and point-in-time demand velocity (lifetime volume, 30d/90d demand, market share).
   - Final projection producing L2-normalized 128-d product vector v_p.
3. TwoTowerModel:
   - Dot-product scoring with temperature scaling and product bias.
   - Multi-label binary cross-entropy (Sigmoid) loss against next-basket targets with pos_weight.
"""

import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender.sequences import normalize_purchase_history_gaps

OUTPUT_DIM = 128
TEXT_DIM = 1024
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_TEMPERATURE = 0.1


class CustomerTower(nn.Module):
    """Modular Customer Tower with Late Fusion across Text, Profile, and Purchase History."""

    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        output_dim: int = OUTPUT_DIM,
        num_cities: int = 1000,
        num_regions: int = 100,
        num_industries: int = 100,
        num_categories: int = 50,
        num_business_lines: int = 20,
        seq_feature_dim: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_products: Optional[int] = None,
        num_tabular_features: int = 8,
        history_feature_indices: Optional[tuple[int, ...]] = None,
        catalog_history_encoder: Optional[str] = None,
        history_gap_reference_days: Optional[float] = None,
        max_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
        *,
        extra_mlp_layer: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_feature_dim = seq_feature_dim
        self.num_tabular_features = num_tabular_features
        self.history_feature_indices = history_feature_indices
        if catalog_history_encoder not in (None, "attention", "pool", "none"):
            raise ValueError("History encoder must be 'attention', 'pool', or 'none'.")
        self.catalog_history_encoder = catalog_history_encoder
        self.max_seq_len = max_seq_len
        if history_gap_reference_days is not None and (
            not math.isfinite(history_gap_reference_days) or history_gap_reference_days <= 0
        ):
            raise ValueError("The history gap reference must be finite and positive.")
        self.history_gap_reference_days = history_gap_reference_days

        # Module 1: Static Company Text Projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Module 2: Tabular & Profile Categorical Metadata
        self.city_embedding = nn.Embedding(num_cities, 16, padding_idx=0)
        self.region_embedding = nn.Embedding(num_regions, 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(num_industries, 16, padding_idx=0)

        tab_input_dim = 16 * 3 + num_tabular_features
        self.tabular_proj = nn.Sequential(
            nn.Linear(tab_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if catalog_history_encoder != "none":
            # Module 3: Purchase History
            if catalog_history_encoder is not None:
                self.product_embedding = None
                self.cat_embedding = None
                self.event_cont_proj = None
                self.event_proj = None
                self.token_proj = nn.Sequential(
                    nn.Linear(output_dim + 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            elif num_products is not None and num_products > 0:
                self.product_embedding = nn.Embedding(num_products, 64, padding_idx=0)
                self.cat_embedding = nn.Embedding(num_categories, 32, padding_idx=0)
                self.event_cont_proj = nn.Linear(seq_feature_dim, 32)
                self.token_proj = nn.Sequential(
                    nn.Linear(64 + 32 + 32, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                self.event_proj = None
            else:
                self.product_embedding = None
                self.cat_embedding = None
                self.event_cont_proj = None
                self.token_proj = None
                self.event_proj = nn.Sequential(
                    nn.Linear(seq_feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )

            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim) * 0.02
            )

            self.transformer = None
            if catalog_history_encoder != "pool":
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

        # Late fusion of available branches.
        fusion_inputs = hidden_dim * (2 if catalog_history_encoder == "none" else 3)
        fusion_layers = [
            nn.Linear(fusion_inputs, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if extra_mlp_layer:
            fusion_layers.extend([
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.late_fusion = nn.Sequential(*fusion_layers, nn.Linear(hidden_dim * 2, output_dim))

    def forward(
        self,
        text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        tabular_features: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_product_ids: Optional[torch.Tensor] = None,
        seq_category_ids: Optional[torch.Tensor] = None,
        *,
        catalog_product_vectors: Optional[torch.Tensor] = None,
        seq_purchase_dates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass generating L2-normalized customer vector u_c.

        Args:
            text_embedding: FloatTensor of shape (B, 1024)
            city_id, region_id, industry_id: LongTensors of shape (B,)
            tabular_features: FloatTensor of shape (B, num_tabular_features)
            sequence: FloatTensor of shape (B, 32, input history feature count)
            attention_mask: LongTensor of shape (B, 32) (1=event, 0=pad)
            seq_product_ids: Optional LongTensor of shape (B, 32)
            seq_category_ids: Optional LongTensor of shape (B, 32)
            catalog_product_vectors: Product Tower outputs in catalogue order, without padding
            seq_purchase_dates: Purchase dates in years relative to a fixed date, shape (B, 32)

        Returns:
            u_c: FloatTensor of shape (B, 128) with unit L2 norm.
        """
        batch_size = text_embedding.size(0)

        # Module 1: Static Company Text
        h_text = self.text_proj(text_embedding)

        # Module 2: Tabular & Profile Metadata
        c_emb = self.city_embedding(city_id)
        r_emb = self.region_embedding(region_id)
        i_emb = self.industry_embedding(industry_id)

        if tabular_features.size(-1) < self.num_tabular_features:
            pad_tab = self.num_tabular_features - tabular_features.size(-1)
            tabular_features = F.pad(tabular_features, (0, pad_tab))
        elif tabular_features.size(-1) > self.num_tabular_features:
            tabular_features = tabular_features[..., :self.num_tabular_features]

        tab_cat = torch.cat([c_emb, r_emb, i_emb, tabular_features], dim=-1)
        h_tab = self.tabular_proj(tab_cat)

        if self.catalog_history_encoder == "none":
            return F.normalize(self.late_fusion(torch.cat([h_text, h_tab], dim=-1)), p=2, dim=-1)

        # Module 3: Purchase tokens
        if sequence is None or attention_mask is None:
            raise ValueError("This customer tower requires a purchase sequence and attention mask.")
        if self.catalog_history_encoder is not None:
            if catalog_product_vectors is None or seq_product_ids is None or seq_purchase_dates is None:
                raise ValueError("Catalogue history requires product vectors, SKU IDs, and purchase dates.")
            # Sequence IDs are one-based; zero represents padding and unknown products.
            catalog_with_padding = F.pad(catalog_product_vectors, (0, 0, 1, 0))
            history_products = F.embedding(seq_product_ids, catalog_with_padding, padding_idx=0)
            quantity = sequence[..., :1]
            time_feature = seq_purchase_dates
            if self.history_gap_reference_days is not None:
                time_feature = normalize_purchase_history_gaps(
                    seq_purchase_dates, attention_mask, self.history_gap_reference_days,
                )
            tokens = self.token_proj(torch.cat([history_products, quantity, time_feature.unsqueeze(-1)], dim=-1))
        else:
            tokens = self._legacy_history_tokens(sequence, seq_product_ids, seq_category_ids)

        seq_len = tokens.size(1)
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        out_seq = tokens + pos_emb

        if self.transformer is not None:
            out_seq = self.transformer(out_seq, src_key_padding_mask=(attention_mask == 0))
            out_seq = torch.nan_to_num(out_seq, nan=0.0)

        # Both encoders use masked mean and the last real event.
        mask_f = attention_mask.unsqueeze(-1).float()
        has_history = (attention_mask.sum(dim=1, keepdim=True) > 0).float()
        mean_vec = (out_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        last_idx_exp = last_idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
        last_vec = out_seq.gather(1, last_idx_exp).squeeze(1)

        pooled_hist = torch.cat([mean_vec, last_vec], dim=-1)
        h_hist = self.history_pooling(pooled_hist) * has_history

        h_combined = torch.cat([h_text, h_tab, h_hist], dim=-1)
        u_c = self.late_fusion(h_combined)
        return F.normalize(u_c, p=2, dim=-1)

    def _legacy_history_tokens(
        self,
        sequence: torch.Tensor,
        seq_product_ids: Optional[torch.Tensor],
        seq_category_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.history_feature_indices is not None:
            sequence = sequence[..., self.history_feature_indices]
        elif sequence.size(-1) < self.seq_feature_dim:
            sequence = F.pad(sequence, (0, self.seq_feature_dim - sequence.size(-1)))
        elif sequence.size(-1) > self.seq_feature_dim:
            sequence = sequence[..., :self.seq_feature_dim]

        if self.product_embedding is not None:
            if seq_product_ids is None:
                seq_product_ids = sequence.new_zeros(sequence.shape[:2], dtype=torch.long)
            if seq_category_ids is None:
                seq_category_ids = sequence.new_zeros(sequence.shape[:2], dtype=torch.long)
            p_emb = self.product_embedding(seq_product_ids)
            c_emb_seq = self.cat_embedding(seq_category_ids)
            cont_emb = self.event_cont_proj(sequence)
            return self.token_proj(torch.cat([p_emb, c_emb_seq, cont_emb], dim=-1))
        return self.event_proj(sequence)


class ProductTower(nn.Module):
    """Encodes product catalog attributes, text, and point-in-time popularity features."""

    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        output_dim: int = OUTPUT_DIM,
        num_brands: int = 50,
        num_categories: int = 50,
        num_subcategories: int = 150,
        num_business_lines: int = 30,
        num_demand_features: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        *,
        num_containers: int = 0,
        one_hot_sizes: Optional[dict[str, int]] = None,
        extra_mlp_layer: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_demand_features = num_demand_features
        self.one_hot_sizes = dict(one_hot_sizes or {})

        # An experimental field uses its vocabulary indicators in place of an embedding.
        self.brand_embedding = (
            nn.Embedding(num_brands, 16, padding_idx=0) if num_brands and "brand" not in self.one_hot_sizes else None
        )
        self.cat_embedding = (
            nn.Embedding(num_categories, 16, padding_idx=0)
            if num_categories and "category" not in self.one_hot_sizes else None
        )
        # Keep the optional subcategory embedding for older checkpoints.
        self.subcat_embedding = nn.Embedding(num_subcategories, 16, padding_idx=0) if num_subcategories else None
        self.line_embedding = (
            nn.Embedding(num_business_lines, 16, padding_idx=0)
            if num_business_lines and "business_line" not in self.one_hot_sizes else None
        )
        self.container_embedding = nn.Embedding(num_containers, 16) if num_containers else None

        # Text projection (1024 -> 128)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        categorical_dim = sum(
            self.one_hot_sizes.get(field, 16 if embedding is not None else 0)
            for field, embedding in (
                ("brand", self.brand_embedding), ("category", self.cat_embedding), ("business_line", self.line_embedding)
            )
        )
        if self.subcat_embedding is not None:
            categorical_dim += self.subcat_embedding.embedding_dim
        if self.container_embedding is not None:
            categorical_dim += self.container_embedding.embedding_dim
        total_input_dim = hidden_dim + categorical_dim + num_demand_features
        mlp_layers = [
            nn.Linear(total_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if extra_mlp_layer:
            mlp_layers.extend([
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*mlp_layers, nn.Linear(hidden_dim * 2, output_dim))

    def forward(
        self,
        text_embedding: torch.Tensor,
        brand_id: torch.Tensor,
        category_id: torch.Tensor,
        subcategory_id: Optional[torch.Tensor],
        business_line_id: torch.Tensor,
        demand_features: torch.Tensor,
        *,
        container_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass generating L2-normalized product vector v_p.

        Args:
            text_embedding: FloatTensor of shape (P, 1024)
            brand_id, category_id, business_line_id: LongTensors of shape (P,)
            subcategory_id: Legacy Product Tower input, required only for those checkpoints
            demand_features: FloatTensor of shape (P, num_demand_features)
            container_id: LongTensor of shape (P,), with 0 for unspecified packaging

        Returns:
            v_p: FloatTensor of shape (P, 128) with unit L2 norm.
        """
        t_proj = self.text_proj(text_embedding)
        b_emb = self._encode_attribute("brand", brand_id, self.brand_embedding, t_proj.dtype)
        c_emb = self._encode_attribute("category", category_id, self.cat_embedding, t_proj.dtype)
        l_emb = self._encode_attribute("business_line", business_line_id, self.line_embedding, t_proj.dtype)

        features = [t_proj, b_emb, c_emb]
        if self.subcat_embedding is not None:
            if subcategory_id is None:
                raise ValueError("This Product Tower checkpoint requires subcategory IDs.")
            features.append(self.subcat_embedding(subcategory_id))
        if self.container_embedding is not None:
            if container_id is None:
                container_id = torch.zeros_like(brand_id)
            features.append(self.container_embedding(container_id))
        concat = torch.cat([*features, l_emb, demand_features], dim=-1)
        v_p = self.mlp(concat)
        return F.normalize(v_p, p=2, dim=-1)

    def _encode_attribute(
        self, field: str, ids: torch.Tensor, embedding: Optional[nn.Embedding], dtype: torch.dtype,
    ) -> torch.Tensor:
        if embedding is not None:
            return embedding(ids)
        if field not in self.one_hot_sizes:
            return torch.empty((*ids.shape, 0), dtype=dtype, device=ids.device)
        # As with the padding embedding, ID 0 contributes no feature values.
        return F.one_hot(ids, num_classes=self.one_hot_sizes[field] + 1)[..., 1:].to(dtype=dtype)


class TwoTowerModel(nn.Module):
    """Complete Two-Tower retrieval model with multi-label binary cross-entropy loss."""

    def __init__(
        self,
        customer_tower: CustomerTower,
        product_tower: ProductTower,
        num_catalog_products: int = 1000,
        temperature: float = DEFAULT_TEMPERATURE,
        include_price_preference: bool = False,
        num_familiarity_features: int = 0,
    ):
        super().__init__()
        self.customer_tower = customer_tower
        self.product_tower = product_tower
        self.temperature = temperature

        # Learnable product bias term
        self.product_bias = nn.Parameter(torch.zeros(num_catalog_products))
        self.price_weights = nn.Parameter(torch.zeros(3)) if include_price_preference else None
        self.familiarity_weights = (
            nn.Parameter(torch.zeros(num_familiarity_features)) if num_familiarity_features else None
        )

    def compute_logits(
        self,
        customer_vectors: torch.Tensor,
        product_vectors: torch.Tensor,
        product_indices: Optional[torch.Tensor] = None,
        *,
        price_features: Optional[torch.Tensor] = None,
        familiarity_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product similarity scores with optional price preference.

        Args:
            customer_vectors: FloatTensor of shape (B, 128)
            product_vectors: FloatTensor of shape (P, 128)
            product_indices: Optional LongTensor of shape (P,) for product bias lookup.
            price_features: Point-in-time FloatTensor of shape (B, P, 3) when enabled.
            familiarity_features: Point-in-time attribute matches of shape (B, P, F) when enabled.

        Returns:
            logits: FloatTensor of shape (B, P)
        """
        scores = torch.matmul(customer_vectors, product_vectors.t()) / self.temperature
        if product_indices is not None:
            bias = self.product_bias[product_indices]  # (P,)
            scores = scores + bias.unsqueeze(0)
        if self.price_weights is not None:
            if price_features is None or price_features.shape != (*scores.shape, 3):
                raise ValueError("Price-aware retrieval requires price features for each customer and candidate.")
            scores = scores + price_features.to(scores) @ self.price_weights
        if self.familiarity_weights is not None:
            expected = (*scores.shape, self.familiarity_weights.numel())
            if familiarity_features is None or familiarity_features.shape != expected:
                raise ValueError("Familiarity-aware retrieval requires an attribute match for each customer and candidate.")
            scores = scores + familiarity_features.to(scores) @ self.familiarity_weights
        return scores

    def compute_multi_label_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute multi-label binary cross-entropy (Sigmoid) loss over eligible catalog products.

        Args:
            logits: FloatTensor of shape (B, P)
            targets: FloatTensor of shape (B, P) with binary labels {0.0, 1.0}
            pos_weight: Optional positive class weighting (e.g., 20.0 or tensor)

        Returns:
            scalar BCE loss
        """
        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                pw = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
            else:
                pw = pos_weight.to(device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        return F.binary_cross_entropy_with_logits(logits, targets)
