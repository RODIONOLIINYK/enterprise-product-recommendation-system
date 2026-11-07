"""Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker (UPS-CANR).

Implements the official Stage 2 neural ranking architecture for next-basket recommendation:
1. Unified 33-Token Customer Memory:
   - Token 0 (Customer Profile): Projected from 1024-d company text (BAAI/bge-m3) +
     customer categoricals (city, region, industry) + 8 continuous lifetime tabular features
     + learnable token-type embedding E_type[0].
   - Tokens 1..32 (Historical Purchases): Projected from product, category, subcategory,
     and business line embeddings + 9 continuous habit/temporal features + product text projection
     + learnable positional embedding + learnable token-type embedding E_type[1].
   - Contextual Self-Attention: 1-layer TransformerEncoder contextualizes Token 0 profile
     with historical purchase tokens. Token 0 is NEVER masked (False in key_padding_mask),
     guaranteeing zero NaNs for cold customers with empty purchase history.
2. Grouped Candidate Query Q (B, C, 128):
   - Pretrained/direct catalog representation of C candidate products.
3. Candidate-Aware Multi-Head Cross-Attention:
   - Query Q attends to Unified Memory (K, V) via 4-head cross-attention, producing attended
     context h_attended (B, C, 128) and attention weights alpha (B, C, 33).
4. Scoring & Residual Fusion Head:
   - Concatenates [z_candidate (128), h_attended (128), z_candidate * h_attended (128),
     pairwise_features (12)] -> 396-d MLP (396 -> 256 -> 128 -> 64 -> 1).
   - Learnable residual skip connection on Stage 1 retrieval score:
     s_final = s_raw + skip_weight * stage1_score.
5. RankerLoss:
   - Multi-label BCEWithLogitsLoss (pos_weight=10.0) + Listwise Softmax Cross-Entropy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.models import OUTPUT_DIM, TEXT_DIM, ProductTower

DEFAULT_MEMORY_DIM: int = OUTPUT_DIM
DEFAULT_NUM_HEADS: int = 4
DEFAULT_SEQUENCE_LENGTH: int = 32
DEFAULT_NUM_PAIRWISE_FEATURES: int = 12
DEFAULT_RESIDUAL_WEIGHT_INIT: float = 0.5


@dataclass
class RankerOutput:
    """Output dataclass from UPSCANRRanker forward pass.

    Attributes:
        scores: FloatTensor of shape (B, C) with final ranking logits.
        raw_scores: FloatTensor of shape (B, C) with MLP scores before residual skip.
        attn_weights: FloatTensor of shape (B, C, 33) with cross-attention weights.
        h_attended: FloatTensor of shape (B, C, 128) with attended context vectors.
        z_candidate: FloatTensor of shape (B, C, 128) with candidate query vectors.
        token_0_weight: FloatTensor of shape (B, C) with attention weight on Token 0 (Profile).
        history_weights: FloatTensor of shape (B, C) with total attention weight on Tokens 1..32.
        memory: Optional FloatTensor of shape (B, 33, 128) with unified memory representations.
    """

    scores: torch.Tensor
    raw_scores: torch.Tensor
    attn_weights: torch.Tensor
    h_attended: torch.Tensor
    z_candidate: torch.Tensor
    token_0_weight: torch.Tensor
    history_weights: torch.Tensor
    memory: Optional[torch.Tensor] = None

    @property
    def logits(self) -> torch.Tensor:
        """Alias for final ranking scores."""
        return self.scores

    @property
    def mlp_score(self) -> torch.Tensor:
        """Alias for raw MLP scores before residual skip."""
        return self.raw_scores

    @property
    def attention_weights(self) -> torch.Tensor:
        """Alias for cross-attention weights."""
        return self.attn_weights

    @property
    def attended_context(self) -> torch.Tensor:
        """Alias for attended customer context."""
        return self.h_attended

    @property
    def profile_attention(self) -> torch.Tensor:
        """Alias for attention weight allocated to Token 0 (Customer Profile)."""
        return self.token_0_weight

    @property
    def history_attention(self) -> torch.Tensor:
        """Alias for attention weight allocated to Tokens 1..32 (Historical Purchases)."""
        return self.history_weights

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"RankerOutput has no attribute '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class UPSCANRRanker(nn.Module):
    """Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker."""

    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        memory_dim: int = DEFAULT_MEMORY_DIM,
        num_cities: int = 1000,
        num_regions: int = 100,
        num_industries: int = 100,
        num_products: int = 1000,
        num_brands: int = 100,
        num_categories: int = 100,
        num_subcategories: int = 200,
        num_business_lines: int = 50,
        num_tabular_features: int = 8,
        num_pairwise_features: int = DEFAULT_NUM_PAIRWISE_FEATURES,
        seq_feature_dim: int = 9,
        hidden_dim: Optional[int] = None,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = 0.1,
        residual_weight_init: float = DEFAULT_RESIDUAL_WEIGHT_INIT,
        catalog_text_embeddings: Optional[torch.Tensor] = None,
        product_tower: Optional[ProductTower] = None,
    ):
        super().__init__()
        dim = hidden_dim if hidden_dim is not None else memory_dim
        self.text_dim = text_dim
        self.memory_dim = dim
        self.hidden_dim = dim
        self.num_heads = num_heads
        self.num_tabular_features = num_tabular_features
        self.num_pairwise_features = num_pairwise_features
        self.seq_feature_dim = seq_feature_dim

        # -------------------------------------------------------------
        # 1. Customer Profile (Token 0) Modules
        # -------------------------------------------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.city_embedding = nn.Embedding(num_cities, 16, padding_idx=0)
        self.region_embedding = nn.Embedding(num_regions, 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(num_industries, 16, padding_idx=0)

        tab_input_dim = 16 * 3 + num_tabular_features
        self.tabular_proj = nn.Sequential(
            nn.Linear(tab_input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.profile_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Token type embeddings: 0 -> Profile (Token 0), 1 -> History (Tokens 1..32)
        self.type_embedding = nn.Embedding(2, dim)

        # -------------------------------------------------------------
        # 2. Purchase History (Tokens 1..32) Modules
        # -------------------------------------------------------------
        self.product_embedding = nn.Embedding(num_products, 64, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_categories, 32, padding_idx=0)
        self.subcat_embedding = nn.Embedding(num_subcategories, 16, padding_idx=0)
        self.line_embedding = nn.Embedding(num_business_lines, 16, padding_idx=0)

        self.habit_proj = nn.Linear(seq_feature_dim, 32)
        self.product_text_proj = nn.Linear(text_dim, 32)

        # 64 (prod) + 32 (cat) + 16 (subcat) + 16 (line) + 32 (habit) + 32 (text) = 192
        history_token_dim = 64 + 32 + 16 + 16 + 32 + 32
        self.token_proj = nn.Sequential(
            nn.Linear(history_token_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, DEFAULT_SEQUENCE_LENGTH, dim) * 0.02)

        # -------------------------------------------------------------
        # 3. Contextual Self-Attention TransformerEncoder (33 Tokens)
        # -------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

        # -------------------------------------------------------------
        # 4. Candidate Tower & Product Representation
        # -------------------------------------------------------------
        self.product_tower = product_tower
        if catalog_text_embeddings is not None:
            self.register_buffer("catalog_text_embeddings", catalog_text_embeddings)
        else:
            self.catalog_text_embeddings = None

        # -------------------------------------------------------------
        # 5. Candidate-Aware Multi-Head Cross-Attention
        # -------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # -------------------------------------------------------------
        # 6. Scoring & Residual Fusion Head
        # -------------------------------------------------------------
        # Inputs: z_candidate (128) + h_attended (128) + interaction (128) + pairwise_features (12) = 396
        total_mlp_input_dim = dim * 3 + num_pairwise_features
        self.mlp = nn.Sequential(
            nn.Linear(total_mlp_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Learnable residual skip weight for Stage 1 score
        self.skip_weight = nn.Parameter(torch.tensor(float(residual_weight_init), dtype=torch.float32))

    @property
    def residual_weight(self) -> nn.Parameter:
        """Alias for skip_weight."""
        return self.skip_weight

    def build_key_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Construct key_padding_mask of shape (B, 33) where True indicates masked positions.

        Token 0 (Profile) is NEVER masked (strictly False), preventing all-padding NaNs.
        Tokens 1..32 follow sequence attention_mask (True where attention_mask == 0).
        """
        b_size = attention_mask.size(0)
        k_mask = torch.zeros((b_size, 33), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 0] = False
        k_mask[:, 1:] = (attention_mask == 0)
        return k_mask

    def _compute_customer_memory(
        self,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct and contextualize the Unified 33-Token Customer Memory."""
        b_size = text_embedding.size(0)
        seq_len = sequence.size(1)

        # Pad / slice tabular features to expected dimension
        if tabular_features.size(-1) < self.num_tabular_features:
            pad_tab = self.num_tabular_features - tabular_features.size(-1)
            tabular_features = F.pad(tabular_features, (0, pad_tab))
        elif tabular_features.size(-1) > self.num_tabular_features:
            tabular_features = tabular_features[..., : self.num_tabular_features]

        # 1. Profile Token 0
        z_text = self.text_proj(text_embedding)  # (B, 128)
        c_city = self.city_embedding(city_id)
        c_reg = self.region_embedding(region_id)
        c_ind = self.industry_embedding(industry_id)

        tab_concat = torch.cat([c_city, c_reg, c_ind, tabular_features], dim=-1)
        z_tab = self.tabular_proj(tab_concat)  # (B, 128)

        z_prof = self.profile_fusion(torch.cat([z_text, z_tab], dim=-1))  # (B, 128)
        t_0 = (z_prof + self.type_embedding.weight[0]).unsqueeze(1)  # (B, 1, 128)

        # 2. History Tokens 1..32
        if seq_product_ids is None:
            seq_product_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=sequence.device)
        if seq_category_ids is None:
            seq_category_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=sequence.device)
        if seq_subcategory_ids is None:
            seq_subcategory_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=sequence.device)
        if seq_business_line_ids is None:
            seq_business_line_ids = torch.zeros((b_size, seq_len), dtype=torch.long, device=sequence.device)

        # Ensure sequence continuous features match seq_feature_dim
        if sequence.size(-1) < self.seq_feature_dim:
            pad_seq = self.seq_feature_dim - sequence.size(-1)
            sequence = F.pad(sequence, (0, pad_seq))
        elif sequence.size(-1) > self.seq_feature_dim:
            sequence = sequence[..., : self.seq_feature_dim]

        # Product text representation for historical purchases
        if seq_product_text is not None:
            txt_emb = seq_product_text
        elif self.catalog_text_embeddings is not None:
            txt_emb = self.catalog_text_embeddings[seq_product_ids]
        else:
            txt_emb = torch.zeros((b_size, seq_len, self.text_dim), dtype=sequence.dtype, device=sequence.device)

        p_emb = self.product_embedding(seq_product_ids)       # (B, 32, 64)
        cat_emb = self.cat_embedding(seq_category_ids)        # (B, 32, 32)
        sub_emb = self.subcat_embedding(seq_subcategory_ids)  # (B, 32, 16)
        line_emb = self.line_embedding(seq_business_line_ids) # (B, 32, 16)
        h_proj = self.habit_proj(sequence)                    # (B, 32, 32)
        t_proj = self.product_text_proj(txt_emb)              # (B, 32, 32)

        hist_raw = torch.cat([p_emb, cat_emb, sub_emb, line_emb, h_proj, t_proj], dim=-1)  # (B, 32, 192)
        t_hist = self.token_proj(hist_raw)  # (B, 32, 128)

        # Positional and token-type embeddings
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)

        t_hist = t_hist + pos_emb + self.type_embedding.weight[1]  # (B, 32, 128)

        # 3. Concatenate Token 0 + Tokens 1..32 -> (B, 33, 128)
        m_raw = torch.cat([t_0, t_hist], dim=1)

        # 4. Contextual Self-Attention
        k_mask = self.build_key_padding_mask(attention_mask)
        memory = self.memory_encoder(m_raw, src_key_padding_mask=k_mask)
        memory = torch.nan_to_num(memory, nan=0.0)

        return memory, k_mask

    def _resolve_candidate_vectors(
        self,
        b_size: int,
        candidate_vectors: Optional[torch.Tensor] = None,
        candidate_indices: Optional[torch.Tensor] = None,
        catalog_product_vectors: Optional[torch.Tensor] = None,
        candidate_text_emb: Optional[torch.Tensor] = None,
        candidate_brand_ids: Optional[torch.Tensor] = None,
        candidate_cat_ids: Optional[torch.Tensor] = None,
        candidate_subcat_ids: Optional[torch.Tensor] = None,
        candidate_line_ids: Optional[torch.Tensor] = None,
        candidate_demand_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve or compute candidate query representations z_candidate (B, C, 128)."""
        if candidate_vectors is not None:
            return candidate_vectors

        if candidate_indices is not None:
            if catalog_product_vectors is not None:
                return catalog_product_vectors[candidate_indices]
            if self.product_tower is not None and hasattr(self, "_cached_catalog_vectors") and self._cached_catalog_vectors is not None:
                return self._cached_catalog_vectors[candidate_indices]

        if self.product_tower is not None and candidate_text_emb is not None:
            # Flatten B * C to compute through ProductTower
            c_dim = candidate_text_emb.size(1)
            f_text = candidate_text_emb.view(-1, self.text_dim)
            f_brand = candidate_brand_ids.view(-1) if candidate_brand_ids is not None else torch.zeros(b_size * c_dim, dtype=torch.long, device=candidate_text_emb.device)
            f_cat = candidate_cat_ids.view(-1) if candidate_cat_ids is not None else torch.zeros(b_size * c_dim, dtype=torch.long, device=candidate_text_emb.device)
            f_sub = candidate_subcat_ids.view(-1) if candidate_subcat_ids is not None else torch.zeros(b_size * c_dim, dtype=torch.long, device=candidate_text_emb.device)
            f_line = candidate_line_ids.view(-1) if candidate_line_ids is not None else torch.zeros(b_size * c_dim, dtype=torch.long, device=candidate_text_emb.device)
            f_dem = candidate_demand_feats.view(-1, 6) if candidate_demand_feats is not None else torch.zeros(b_size * c_dim, 6, dtype=candidate_text_emb.dtype, device=candidate_text_emb.device)

            v_p = self.product_tower(f_text, f_brand, f_cat, f_sub, f_line, f_dem)
            return v_p.view(b_size, c_dim, self.memory_dim)

        raise ValueError(
            "UPSCANRRanker requires candidate_vectors, candidate_indices with catalog vectors, "
            "or candidate attribute tensors."
        )

    def forward(
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
        candidate_scores: Optional[torch.Tensor] = None,
        pairwise_features: Optional[torch.Tensor] = None,
        candidate_vectors: Optional[torch.Tensor] = None,
        catalog_product_vectors: Optional[torch.Tensor] = None,
        candidate_text_emb: Optional[torch.Tensor] = None,
        candidate_brand_ids: Optional[torch.Tensor] = None,
        candidate_cat_ids: Optional[torch.Tensor] = None,
        candidate_subcat_ids: Optional[torch.Tensor] = None,
        candidate_line_ids: Optional[torch.Tensor] = None,
        candidate_demand_feats: Optional[torch.Tensor] = None,
    ) -> RankerOutput:
        """Forward pass of Stage 2 UPS-CANR ranker."""
        b_size = text_embedding.size(0)

        # 1. Customer Memory Contextualization (B, 33, 128)
        memory, key_padding_mask = self._compute_customer_memory(
            text_embedding=text_embedding,
            city_id=city_id,
            region_id=region_id,
            industry_id=industry_id,
            tabular_features=tabular_features,
            sequence=sequence,
            attention_mask=attention_mask,
            seq_product_ids=seq_product_ids,
            seq_category_ids=seq_category_ids,
            seq_subcategory_ids=seq_subcategory_ids,
            seq_business_line_ids=seq_business_line_ids,
            seq_product_text=seq_product_text,
        )

        # 2. Resolve Candidate Queries Q (B, C, 128)
        z_cand = self._resolve_candidate_vectors(
            b_size=b_size,
            candidate_vectors=candidate_vectors,
            candidate_indices=candidate_indices,
            catalog_product_vectors=catalog_product_vectors,
            candidate_text_emb=candidate_text_emb,
            candidate_brand_ids=candidate_brand_ids,
            candidate_cat_ids=candidate_cat_ids,
            candidate_subcat_ids=candidate_subcat_ids,
            candidate_line_ids=candidate_line_ids,
            candidate_demand_feats=candidate_demand_feats,
        )
        c_slate = z_cand.size(1)

        # 3. Candidate-Aware Multi-Head Cross-Attention
        # Query Q: (B, C, 128), Key K: (B, 33, 128), Value V: (B, 33, 128)
        h_att, attn_w = self.cross_attention(
            query=z_cand,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        h_att = torch.nan_to_num(h_att, nan=0.0)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        # Explicitly zero out attention weights for padding tokens to ensure strict 0.0 attention
        if key_padding_mask is not None:
            mask_exp = key_padding_mask.unsqueeze(1).expand(-1, c_slate, -1)
            attn_w = attn_w.masked_fill(mask_exp, 0.0)
            w_sum = attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            attn_w = attn_w / w_sum

        token_0_weight = attn_w[..., 0]
        history_weights = attn_w[..., 1:].sum(dim=-1)

        # 4. Feature Fusion
        interaction = z_cand * h_att

        if pairwise_features is None:
            pairwise_features = torch.zeros(
                (b_size, c_slate, self.num_pairwise_features),
                dtype=z_cand.dtype,
                device=z_cand.device,
            )
        elif pairwise_features.size(-1) < self.num_pairwise_features:
            pad_pw = self.num_pairwise_features - pairwise_features.size(-1)
            pairwise_features = F.pad(pairwise_features, (0, pad_pw))
        elif pairwise_features.size(-1) > self.num_pairwise_features:
            pairwise_features = pairwise_features[..., : self.num_pairwise_features]

        fused = torch.cat([z_cand, h_att, interaction, pairwise_features], dim=-1)  # (B, C, 396)

        # 5. Deep Ranking MLP
        mlp_raw = self.mlp(fused).squeeze(-1)  # (B, C)

        # 6. Residual Skip Connection
        if candidate_scores is not None:
            stage1_score = candidate_scores
        else:
            stage1_score = pairwise_features[..., 0]

        logits = mlp_raw + self.skip_weight * stage1_score

        return RankerOutput(
            scores=logits,
            raw_scores=mlp_raw,
            attn_weights=attn_w,
            h_attended=h_att,
            z_candidate=z_cand,
            token_0_weight=token_0_weight,
            history_weights=history_weights,
            memory=memory,
        )

    @classmethod
    def from_retriever_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "UPSCANRRanker":
        """Instantiate UPSCANRRanker and load pretrained CustomerTower and ProductTower weights."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=dev)
        state = ckpt.get("model_state_dict", {})
        config = ckpt.get("config", {})

        # Determine dimensions from checkpoint
        city_size = state.get("customer_tower.city_embedding.weight", torch.zeros(1000, 16)).shape[0]
        reg_size = state.get("customer_tower.region_embedding.weight", torch.zeros(100, 16)).shape[0]
        ind_size = state.get("customer_tower.industry_embedding.weight", torch.zeros(100, 16)).shape[0]
        cat_size = state.get("product_tower.cat_embedding.weight", torch.zeros(100, 16)).shape[0]
        product_subcategories = state.get("product_tower.subcat_embedding.weight")
        product_subcat_size = product_subcategories.shape[0] if product_subcategories is not None else 0
        subcat_size = product_subcat_size or max(len(ckpt.get("vocabs", {}).get("subcategory_vocab", {})) + 1, 150)
        line_size = state.get("product_tower.line_embedding.weight", torch.zeros(50, 16)).shape[0]
        brand_size = state.get("product_tower.brand_embedding.weight", torch.zeros(100, 16)).shape[0]

        num_products = len(ckpt.get("product_ids", []))
        if num_products == 0:
            num_products = config.get("num_catalog_products", 1000)

        # Build ProductTower
        product_tower = ProductTower(
            text_dim=config.get("text_dim", TEXT_DIM),
            output_dim=config.get("output_dim", OUTPUT_DIM),
            num_brands=brand_size,
            num_categories=cat_size,
            num_subcategories=product_subcat_size,
            num_business_lines=line_size,
            num_demand_features=6,
        )

        # Build ranker
        ranker_kwargs = dict(
            text_dim=config.get("text_dim", TEXT_DIM),
            memory_dim=config.get("output_dim", OUTPUT_DIM),
            num_cities=city_size,
            num_regions=reg_size,
            num_industries=ind_size,
            num_products=num_products + 1,
            num_brands=brand_size,
            num_categories=cat_size,
            num_subcategories=subcat_size,
            num_business_lines=line_size,
            product_tower=product_tower,
        )
        ranker_kwargs.update(kwargs)
        model = cls(**ranker_kwargs)

        # Transfer matching weights from CustomerTower
        cust_prefix = "customer_tower."
        prod_prefix = "product_tower."
        transferred = 0

        with torch.no_grad():
            # City / region / industry embeddings
            if f"{cust_prefix}city_embedding.weight" in state:
                model.city_embedding.weight.copy_(state[f"{cust_prefix}city_embedding.weight"])
                transferred += 1
            if f"{cust_prefix}region_embedding.weight" in state:
                model.region_embedding.weight.copy_(state[f"{cust_prefix}region_embedding.weight"])
                transferred += 1
            if f"{cust_prefix}industry_embedding.weight" in state:
                model.industry_embedding.weight.copy_(state[f"{cust_prefix}industry_embedding.weight"])
                transferred += 1

            # Text projection
            if f"{cust_prefix}text_proj.0.weight" in state:
                model.text_proj[0].weight.copy_(state[f"{cust_prefix}text_proj.0.weight"])
                model.text_proj[0].bias.copy_(state[f"{cust_prefix}text_proj.0.bias"])
                transferred += 1

            # Share corresponding columns while preserving the ranker's extra inputs.
            if f"{cust_prefix}tabular_proj.0.weight" in state:
                source = state[f"{cust_prefix}tabular_proj.0.weight"]
                target = model.tabular_proj[0].weight
                source_names = tuple(config.get("profile_features", LEGACY_PROFILE_FEATURES))
                cat_width = 16 * 3
                if (
                    source.shape[0] == target.shape[0]
                    and source.shape[1] == cat_width + len(source_names)
                    and target.shape[1] == cat_width + len(LEGACY_PROFILE_FEATURES)
                ):
                    target[:, :cat_width].copy_(source[:, :cat_width])
                    for i, name in enumerate(source_names):
                        if name in LEGACY_PROFILE_FEATURES:
                            j = LEGACY_PROFILE_FEATURES.index(name)
                            target[:, cat_width + j].copy_(source[:, cat_width + i])
                    model.tabular_proj[0].bias.copy_(state[f"{cust_prefix}tabular_proj.0.bias"])
                    transferred += 1

            # Product Tower weights
            prod_state = {
                k[len(prod_prefix) :]: v for k, v in state.items() if k.startswith(prod_prefix)
            }
            if prod_state:
                model.product_tower.load_state_dict(prod_state, strict=False)
                transferred += len(prod_state)

        model.to(dev)
        return model


class RankerLoss(nn.Module):
    """Combined Multi-Label BCE and Listwise Softmax Cross-Entropy Loss."""

    def __init__(self, pos_weight: float = 10.0, listwise_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.listwise_weight = float(listwise_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BCE loss, Listwise loss, and combined total loss.

        Args:
            logits: FloatTensor of shape (B, C) with Stage 2 ranking logits.
            targets: FloatTensor of shape (B, C) with binary ground-truth labels {0.0, 1.0}.

        Returns:
            Dict containing 'loss', 'bce_loss', and 'list_loss'.
        """
        pw = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

        # Listwise Softmax Cross-Entropy
        pos_mask = (targets > 0.5)
        pos_counts = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        target_probs = targets / pos_counts

        log_probs = F.log_softmax(logits, dim=-1)
        slate_has_pos = (pos_mask.sum(dim=-1) > 0)

        if slate_has_pos.any():
            loss_lw = -(target_probs * log_probs).sum(dim=-1)
            loss_lw = loss_lw[slate_has_pos].mean()
        else:
            loss_lw = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

        total_loss = loss_bce + self.listwise_weight * loss_lw

        return {
            "loss": total_loss,
            "bce_loss": loss_bce,
            "list_loss": loss_lw,
        }


def compute_ranker_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float = 10.0,
    listwise_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to compute ranker loss returning (total_loss, bce_loss, list_loss)."""
    loss_module = RankerLoss(pos_weight=pos_weight, listwise_weight=listwise_weight)
    res = loss_module(logits, labels)
    return res["loss"], res["bce_loss"], res["list_loss"]
