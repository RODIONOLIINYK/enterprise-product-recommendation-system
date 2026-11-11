"""Versioned experimental ranker; the reference UPS-CANR remains unchanged."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from recommender.candidates import PAIRWISE_FEATURE_NAMES
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.ranker import RankerOutput, UPSCANRRanker
from recommender.sequences import DAYS_PER_YEAR, ENRICHED_FEATURE_NAMES, PAD_TOKEN

ARCHITECTURE_VERSION = "shared_stage1_v1"
HISTORY_INPUT_INDICES = tuple(range(8))
HISTORY_INPUT_NAMES = ENRICHED_FEATURE_NAMES[:8]
TOKEN_REGIONS = {
    "stage1": (0, 128), "customer_statistics": (128, 136),
    "history_statistics": (136, 144), "candidate_statistics": (144, 157),
    "position": (157, 173), "type": (173, 176),
    "relationships": (176, 192), "categoricals": (192, 224),
}
REGIONS = {name: slice(*bounds) for name, bounds in TOKEN_REGIONS.items()}
SCALE_GROUP_NAMES = ("customer_statistics", "history_statistics", "candidate_statistics", "position", "type")
TYPE_CUSTOMER, TYPE_HISTORY, TYPE_CANDIDATE = range(*TOKEN_REGIONS["type"])


def feature_schema():
    return {
        "customer": list(LEGACY_PROFILE_FEATURES), "history": list(HISTORY_INPUT_NAMES),
        "history_source": list(ENRICHED_FEATURE_NAMES), "candidate": list(PAIRWISE_FEATURE_NAMES),
        "history_source_indices": list(HISTORY_INPUT_INDICES),
        "token_regions": {name: list(bounds) for name, bounds in TOKEN_REGIONS.items()},
        "token_types": ["customer", "history", "candidate"],
        "history_unknown_index": -1, "history_padding_index": -1,
        "padding_discriminator": "attention_mask", "candidate_index_base": 0,
        "bias_age": "age_days / (age_days + 90)",
    }


def tensor_digest(tensor):
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(repr((list(values.shape), str(values.dtype))).encode())
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def map_history_skus(rows, product_ids):
    """Canonical SKU lookup, independent of the original one-based training IDs."""
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Expanded catalogue IDs must be unique.")
    lookup = {sku: i for i, sku in enumerate(product_ids)}
    return torch.tensor([[lookup.get(sku, -1) if sku != PAD_TOKEN else -1 for sku in row]
                         for row in rows], dtype=torch.long)


@dataclass(frozen=True)
class SharedRankerConfig:
    representation: str = "shared"
    width: int = 176
    memory_statistics: bool = False
    query_statistics: bool = False
    bias_schema: str = "legacy17"
    exclude_history_repeat: bool = True
    learned_relationships: bool = False
    categorical_sizes: tuple[int, ...] = ()  # city, region, industry, category, subcategory, line
    statistics_scale: float = 0.1
    customer_statistics_scale: float | None = None
    history_statistics_scale: float | None = None
    candidate_statistics_scale: float | None = None
    position_scale: float = 1.0
    type_scale: float = 1.0
    dropout: float = 0.1
    profile_token: bool = True
    history_tokens: bool = True

    def __post_init__(self):
        if self.representation not in ("shared", "legacy_width"):
            raise ValueError("Unknown input representation.")
        if self.representation == "legacy_width" and (self.width != 176 or self.memory_statistics or self.query_statistics):
            raise ValueError("The width control uses 176D and the legacy bias-only numerical paths.")
        if self.width not in (176, 192, 224) or self.bias_schema not in ("legacy17", "reduced16", "none"):
            raise ValueError("Unsupported shared ranker width or bias schema.")
        if self.learned_relationships and self.width < 192:
            raise ValueError("Relationship adapters require width >= 192.")
        if self.width == 224 and (len(self.categorical_sizes) != 6 or min(self.categorical_sizes) < 1):
            raise ValueError("224D input requires six fitted categorical vocabulary sizes.")
        for value in (self.statistics_scale, self.customer_statistics_scale, self.history_statistics_scale,
                      self.candidate_statistics_scale, self.position_scale, self.type_scale):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Input group scales must be finite and positive.")


class SharedStage1TokenBuilder(nn.Module):
    def __init__(self, product_vectors, config, product_metadata_ids=None):
        super().__init__()
        if product_vectors.ndim != 2 or product_vectors.shape[1] != 128:
            raise ValueError("Expected a (products, 128) Stage 1 vector table.")
        self.config = config
        self.register_buffer("catalog_product_vectors", product_vectors.detach().clone())
        self.position = nn.Embedding(32, 16)
        nn.init.normal_(self.position.weight, std=0.02)
        self.log_scales = nn.Parameter(torch.tensor([
            *[math.log(value if value is not None else config.statistics_scale) for value in (
                config.customer_statistics_scale, config.history_statistics_scale, config.candidate_statistics_scale,
            )], math.log(config.position_scale), math.log(config.type_scale),
        ]))
        self.adapters = nn.ModuleDict()
        if config.learned_relationships:
            # Zero-output adapters must not change the width control's common
            # attention/scoring initialization by consuming extra random draws.
            with torch.random.fork_rng(devices=[]):
                for name, width in (("customer", 8), ("history", 8), ("candidate", 13)):
                    self.adapters[name] = nn.Linear(width, 16)
                    nn.init.zeros_(self.adapters[name].weight)
                    nn.init.zeros_(self.adapters[name].bias)
        if config.width == 224:
            if product_metadata_ids is None or product_metadata_ids.shape != (len(product_vectors), 3):
                raise ValueError("224D input requires category/subcategory/line IDs for every catalogue SKU.")
            self.register_buffer("product_metadata_ids", product_metadata_ids.clone())
            city, region, industry, category, subcategory, line = config.categorical_sizes
            self.customer_embeddings = nn.ModuleList([nn.Embedding(n, 8, padding_idx=0) for n in (city, region, industry)])
            self.item_embeddings = nn.ModuleList([
                nn.Embedding(len(product_vectors) + 1, 16, padding_idx=0),
                nn.Embedding(category, 8, padding_idx=0), nn.Embedding(subcategory, 4, padding_idx=0),
                nn.Embedding(line, 4, padding_idx=0),
            ])
            self.categorical_adapters = nn.ModuleDict({
                "customer": nn.Linear(24, 32), "history": nn.Linear(32, 32), "candidate": nn.Linear(32, 32),
            })

    def _item_categories(self, indices):
        metadata = F.pad(self.product_metadata_ids, (0, 0, 1, 0))[indices + 1]
        return torch.cat([self.item_embeddings[0](indices + 1), *[
            embedding(metadata[..., i]) for i, embedding in enumerate(self.item_embeddings[1:])
        ]], dim=-1)

    def forward(self, stage1_customer_vectors, seq_catalog_indices, candidate_indices,
                tabular_features, sequence, pairwise_features, attention_mask,
                city_id=None, region_id=None, industry_id=None, **_legacy_inputs):
        batch, length = attention_mask.shape
        if sequence.shape != (batch, length, 9) or tabular_features.shape != (batch, 8):
            raise ValueError("Shared input requires the unchanged 9D source sequence and 8D profile.")
        if stage1_customer_vectors.shape != (batch, 128) or pairwise_features.shape != (*candidate_indices.shape, 13):
            raise ValueError("Misaligned Stage 1 customer vectors or candidate statistics.")
        if seq_catalog_indices.shape != (batch, length) or length > 32:
            raise ValueError("Shared history mapping must have 0..32 positions aligned to its mask.")
        if seq_catalog_indices.numel() and (seq_catalog_indices.min() < -1 or seq_catalog_indices.max() >= len(self.catalog_product_vectors)):
            raise ValueError("History indices must be -1 or valid expanded catalogue indices.")
        if candidate_indices.numel() and (candidate_indices.min() < 0 or candidate_indices.max() >= len(self.catalog_product_vectors)):
            raise ValueError("Candidate indices must be valid expanded catalogue indices.")
        # Model inputs are frozen pretrained representations. Training is confined to Stage 2.
        table = F.pad(self.catalog_product_vectors, (0, 0, 1, 0))
        history_vectors = table[seq_catalog_indices + 1]
        candidate_vectors = self.catalog_product_vectors[candidate_indices]
        memory = stage1_customer_vectors.new_zeros((batch, length + 1, self.config.width))
        candidates = stage1_customer_vectors.new_zeros((*candidate_indices.shape, self.config.width))
        memory[:, 0, REGIONS["stage1"]] = stage1_customer_vectors.detach()
        memory[:, 1:, REGIONS["stage1"]] = history_vectors
        candidates[..., REGIONS["stage1"]] = candidate_vectors
        profile_stats = tabular_features
        history_stats = sequence[..., HISTORY_INPUT_INDICES].clone()
        if self.config.exclude_history_repeat:
            history_stats[..., 7] = 0
        scales = self.log_scales.exp()
        if self.config.memory_statistics:
            memory[:, 0, REGIONS["customer_statistics"]] = profile_stats * scales[0]
            memory[:, 1:, REGIONS["history_statistics"]] = history_stats * scales[1]
        if self.config.query_statistics:
            candidates[..., REGIONS["candidate_statistics"]] = pairwise_features * scales[2]
        memory[:, 1:, REGIONS["position"]] = self.position.weight[:length] * scales[3]
        memory[:, 0, TYPE_CUSTOMER] = scales[4]
        memory[:, 1:, TYPE_HISTORY] = scales[4]
        candidates[..., TYPE_CANDIDATE] = scales[4]
        if self.adapters:
            # Disabled statistics pathways also disable their derived adapters.
            if self.config.memory_statistics:
                memory[:, 0, REGIONS["relationships"]] = self.adapters["customer"](profile_stats)
                memory[:, 1:, REGIONS["relationships"]] = self.adapters["history"](history_stats)
            if self.config.query_statistics:
                candidates[..., REGIONS["relationships"]] = self.adapters["candidate"](pairwise_features)
        if self.config.width == 224:
            customer = torch.cat([emb(ids) for emb, ids in zip(
                self.customer_embeddings, (city_id, region_id, industry_id), strict=True,
            )], dim=-1)
            memory[:, 0, REGIONS["categoricals"]] = self.categorical_adapters["customer"](customer)
            memory[:, 1:, REGIONS["categoricals"]] = self.categorical_adapters["history"](self._item_categories(seq_catalog_indices))
            candidates[..., REGIONS["categoricals"]] = self.categorical_adapters["candidate"](self._item_categories(candidate_indices))
        memory[:, 1:] = memory[:, 1:].masked_fill(~attention_mask.bool()[..., None], 0)
        if not self.config.profile_token:
            memory[:, 0] = 0  # Finite null key also covers entirely cold customers.
        if not self.config.history_tokens:
            memory[:, 1:] = 0
        return memory, candidates


class StatisticalAttentionBias(nn.Module):
    def __init__(self, schema="legacy17", exclude_history_repeat=True):
        super().__init__()
        if schema not in ("legacy17", "reduced16"):
            raise ValueError("Unknown statistical attention schema.")
        self.schema = schema
        self.exclude_history_repeat = exclude_history_repeat
        self.history_width = 9 if schema == "legacy17" else 8
        self.weights = nn.Parameter(torch.zeros(4, 14, self.history_width + 8))

    def key_statistics(self, sequence, tabular_features, attention_mask, seq_purchase_dates, prediction_day):
        history = sequence[..., :self.history_width].clone()
        if self.exclude_history_repeat:
            history[..., 7] = 0
        if self.history_width == 9:
            history[..., 8] = attention_mask.to(history.dtype)
        age_days = (prediction_day[:, None] - (seq_purchase_dates * DAYS_PER_YEAR).round()).clamp_min(0)
        history[..., 1] = age_days / (age_days + 90.)
        return torch.cat([
            F.pad(tabular_features[:, None], (self.history_width, 0)), F.pad(history, (0, 8)),
        ], dim=1)

    def forward(self, pairwise_features, sequence, tabular_features, attention_mask,
                seq_purchase_dates, prediction_day, key_padding_mask):
        keys = self.key_statistics(sequence, tabular_features, attention_mask, seq_purchase_dates, prediction_day)
        coefficients = F.pad(pairwise_features, (1, 0), value=1)[:, None] @ self.weights
        bias = coefficients @ keys.transpose(-1, -2)[:, None]
        bias = bias.masked_fill(key_padding_mask[:, None, None], -torch.inf)
        return bias.reshape(-1, pairwise_features.size(1), keys.size(1))


class LegacyWidthTokenBuilder(nn.Module):
    """Old 128D raw memory values, padded to 176D before the sole encoder.

    This control retains the old trainable input branches, including their zero
    numerical/text inputs. It does not add a second memory encoder.
    """

    def __init__(self, vectors, config, reference):
        super().__init__()
        self.config = config
        self.register_buffer("catalog_product_vectors", vectors.detach().clone())
        names = ("text_proj", "city_embedding", "region_embedding", "industry_embedding", "tabular_proj",
                 "profile_fusion", "type_embedding", "product_embedding", "cat_embedding", "subcat_embedding",
                 "line_embedding", "habit_proj", "product_text_proj", "token_proj")
        self.parts = nn.ModuleDict({name: getattr(reference, name) for name in names})
        self.pos_embedding = nn.Parameter(reference.pos_embedding.detach().clone())

    def forward(self, stage1_customer_vectors, seq_catalog_indices, candidate_indices,
                tabular_features, sequence, pairwise_features, attention_mask,
                city_id=None, region_id=None, industry_id=None, text_embedding=None,
                seq_product_ids=None, seq_category_ids=None, seq_subcategory_ids=None, seq_business_line_ids=None,
                **_unused):
        del stage1_customer_vectors, seq_catalog_indices, pairwise_features
        parts = self.parts
        profile = parts["profile_fusion"](torch.cat([
            parts["text_proj"](text_embedding), parts["tabular_proj"](torch.cat([
                parts["city_embedding"](city_id), parts["region_embedding"](region_id),
                parts["industry_embedding"](industry_id), torch.zeros_like(tabular_features),
            ], -1)),
        ], -1)) + parts["type_embedding"].weight[0]
        batch, length = attention_mask.shape
        history = parts["token_proj"](torch.cat([
            parts["product_embedding"](seq_product_ids), parts["cat_embedding"](seq_category_ids),
            parts["subcat_embedding"](seq_subcategory_ids), parts["line_embedding"](seq_business_line_ids),
            parts["habit_proj"](torch.zeros_like(sequence)),
            parts["product_text_proj"](sequence.new_zeros(batch, length, 1024)),
        ], -1)) + self.pos_embedding[:, :length] + parts["type_embedding"].weight[1]
        raw = F.pad(torch.cat([profile[:, None], history], dim=1), (0, 48))
        if not self.config.profile_token:
            raw[:, 0] = 0
        raw[:, 1:] = raw[:, 1:].masked_fill(~attention_mask.bool()[..., None], 0)
        candidates = F.pad(self.catalog_product_vectors[candidate_indices], (0, 48))
        return raw, candidates


class SharedStage1Ranker(nn.Module):
    requires_shared_stage1_inputs = True

    def __init__(self, product_vectors, config=None, product_metadata_ids=None, legacy_input_model=None):
        super().__init__()
        self.config = config or SharedRankerConfig()
        width = self.config.width
        if self.config.representation == "legacy_width":
            if legacy_input_model is None:
                raise ValueError("The width control requires the legacy input modules.")
            self.token_builder = LegacyWidthTokenBuilder(product_vectors, self.config, legacy_input_model)
        else:
            self.token_builder = SharedStage1TokenBuilder(product_vectors, self.config, product_metadata_ids)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, self.config.dropout, "gelu", batch_first=True)
        self.memory_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.cross_attention = nn.MultiheadAttention(width, 4, self.config.dropout, batch_first=True)
        self.statistical_bias = (StatisticalAttentionBias(self.config.bias_schema, self.config.exclude_history_repeat)
                                 if self.config.bias_schema != "none" else None)
        self.context_readout = nn.Linear(width, 128)
        with torch.no_grad():
            self.context_readout.weight.zero_()
            self.context_readout.weight[:, :128].copy_(torch.eye(128))
            self.context_readout.bias.zero_()
        self.mlp = nn.Sequential(
            nn.Linear(397, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.2),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1),
        )

    def forward(self, stage1_customer_vectors, seq_catalog_indices, candidate_indices,
                tabular_features, sequence, pairwise_features, attention_mask,
                seq_purchase_dates=None, prediction_day=None, city_id=None, region_id=None, industry_id=None,
                **legacy_inputs):
        memory, candidates = self.token_builder(
            stage1_customer_vectors, seq_catalog_indices, candidate_indices, tabular_features,
            sequence, pairwise_features, attention_mask, city_id, region_id, industry_id, **legacy_inputs,
        )
        mask = F.pad(~attention_mask.bool(), (1, 0), value=False)
        if not self.config.history_tokens:
            mask[:, 1:] = True
        memory = self.memory_encoder(memory, src_key_padding_mask=mask)
        memory = memory.masked_fill(mask[..., None], 0)
        bias = None
        if self.statistical_bias is not None:
            if seq_purchase_dates is None or prediction_day is None:
                raise ValueError("Statistical bias requires purchase and prediction dates.")
            profile_stats = tabular_features if self.config.profile_token else torch.zeros_like(tabular_features)
            bias = self.statistical_bias(pairwise_features, sequence, profile_stats, attention_mask,
                                         seq_purchase_dates, prediction_day, mask)
        context, weights = self.cross_attention(
            candidates, memory, memory, key_padding_mask=mask if bias is None else None,
            attn_mask=bias, need_weights=True, average_attn_weights=True,
        )
        weights = weights.masked_fill(mask[:, None], 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        candidate_128 = candidates[..., :128]
        context_128 = self.context_readout(context)
        scores = self.mlp(torch.cat([
            candidate_128, context_128, candidate_128 * context_128, pairwise_features,
        ], dim=-1)).squeeze(-1)
        return RankerOutput(scores, scores, weights, context_128, candidate_128,
                            weights[..., 0], weights[..., 1:].sum(-1), memory)

    def metadata(self, stage1_digest, product_ids, cache_identity):
        return {
            "architecture_version": ARCHITECTURE_VERSION, "config": asdict(self.config),
            "feature_schema": feature_schema(), "stage1_digest": stage1_digest,
            "product_ids": list(product_ids), "cache_identity": cache_identity,
            "product_vector_digest": tensor_digest(self.token_builder.catalog_product_vectors),
            "stage1_temporal_policy": "fixed_train_cutoff_demand; inherited look-ahead in Stage1 training",
            "customer_vector_provenance": "per-transition; same frozen Stage1 customer forward used for slate retrieval",
            "scoring_head": "397D MLP; identity-initialized context readout to 128D",
            "input_layout": "legacy 128D token values + 48 zeros" if self.config.representation == "legacy_width" else "shared named regions with optional appended channels",
            "bias_indicator": "attention_mask" if self.config.bias_schema == "legacy17" else "absent",
            "bias_key_features": (list(ENRICHED_FEATURE_NAMES) if self.config.bias_schema == "legacy17" else list(HISTORY_INPUT_NAMES)) + list(LEGACY_PROFILE_FEATURES),
            "bias_candidate_features": ["intercept", *PAIRWISE_FEATURE_NAMES],
            "scale_group_order": list(SCALE_GROUP_NAMES),
            "input_scaling": "independent exp(log_scale) per group; first 128 coordinates unchanged",
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, product_vectors, product_ids, stage1_digest, cache_identity):
        metadata = checkpoint["model_metadata"]
        if (metadata["architecture_version"] != ARCHITECTURE_VERSION
                or metadata["feature_schema"] != feature_schema()
                or metadata["product_ids"] != list(product_ids)
                or metadata["stage1_digest"] != stage1_digest
                or metadata["cache_identity"] != cache_identity
                or metadata["product_vector_digest"] != tensor_digest(product_vectors)):
            raise ValueError("Experimental checkpoint schema, catalogue, retriever or cache is incompatible.")
        config = SharedRankerConfig(**metadata["config"])
        state = checkpoint["model_state_dict"]
        if tensor_digest(state["token_builder.catalog_product_vectors"]) != metadata["product_vector_digest"]:
            raise ValueError("Experimental checkpoint product-vector state is incompatible with its provenance.")
        legacy_model = None
        if config.representation == "legacy_width":
            names = {"num_cities": "city_embedding", "num_regions": "region_embedding", "num_industries": "industry_embedding",
                     "num_products": "product_embedding", "num_categories": "cat_embedding",
                     "num_subcategories": "subcat_embedding", "num_business_lines": "line_embedding"}
            legacy_model = UPSCANRRanker(**{name: state[f"token_builder.parts.{part}.weight"].size(0) for name, part in names.items()})
        model = cls(product_vectors, config, state.get("token_builder.product_metadata_ids"), legacy_model)
        model.load_state_dict(state, strict=True)
        return model


def schema_digest():
    return hashlib.sha256(json.dumps(feature_schema(), sort_keys=True).encode()).hexdigest()
