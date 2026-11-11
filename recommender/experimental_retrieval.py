"""Independent E3–E5 towers with purchase-triggered, versioned serving caches.

No prediction/request date is an input to these towers. Six demand values are
zeroed for both training and serving; packaging remains available.
"""
from dataclasses import dataclass
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from recommender.sequences import normalize_purchase_history_gaps


@dataclass(frozen=True)
class RetrievalConfig:
    long_memory: str = "none"
    interests: int = 0  # 0 = original mean + last pooling; 1 = query pooling control
    scoring: str = "hard"
    smooth_tau: float = .1

    def __post_init__(self):
        if self.long_memory not in ("none", "mean", "attention") or self.interests not in (0, 1, 2, 3):
            raise ValueError("Invalid experiment")
        if self.interests and self.long_memory != "none":
            raise ValueError("E3 and E4 must remain independent")
        if self.scoring not in ("hard", "smooth") or self.smooth_tau <= 0:
            raise ValueError("Invalid scoring")


def reset_reference(model):
    """Discard *all* learned reference weights, including attention projections."""
    def reset(module):
        if isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
    model.apply(reset)
    with torch.no_grad():
        model.customer_tower.pos_embedding.normal_(0, .02)
        model.product_bias.zero_()
        if model.familiarity_weights is not None:
            model.familiarity_weights.zero_()
    return model


def history_components(tower, inputs, vectors):
    """Same short-term encoder operations as CustomerTower.forward, unmodified."""
    h_text = tower.text_proj(inputs["text_embedding"])
    tab = torch.cat([tower.city_embedding(inputs["city_id"]), tower.region_embedding(inputs["region_id"]),
                     tower.industry_embedding(inputs["industry_id"]), inputs["tabular_features"]], -1)
    h_tab = tower.tabular_proj(tab)
    mask = inputs["attention_mask"]
    products = F.embedding(inputs["seq_product_ids"], F.pad(vectors, (0, 0, 1, 0)))
    gaps = normalize_purchase_history_gaps(inputs["seq_purchase_dates"], mask, tower.history_gap_reference_days)
    tokens = tower.token_proj(torch.cat([products, inputs["sequence"][..., :1], gaps.unsqueeze(-1)], -1))
    contextual = tower.transformer(tokens + tower.pos_embedding[:, :tokens.shape[1]], src_key_padding_mask=~mask.bool())
    contextual = torch.nan_to_num(contextual)
    has_history = mask.any(1, keepdim=True)
    mean = (contextual * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    last = contextual[torch.arange(len(mask), device=mask.device), (mask.sum(1).long() - 1).clamp_min(0)]
    history = tower.history_pooling(torch.cat([mean, last], -1)) * has_history
    return h_text, h_tab, history, contextual, last


def weighted_pool(values, mask, queries=None, counts=None):
    """Multiplicity-aware pooling is identical to expanding repeated event tokens."""
    if counts is None:
        counts = mask.to(values)
    counts = counts * mask
    if queries is None:
        return (values * counts.unsqueeze(-1)).sum(1) / counts.sum(1, keepdim=True).clamp_min(1)
    logits = torch.einsum("bld,kd->bkl", values, queries) / math.sqrt(values.shape[-1])
    logits = logits + counts.clamp_min(1).log().unsqueeze(1)
    logits = logits.masked_fill(~mask.bool().unsqueeze(1), -1e9)
    weights = logits.softmax(-1) * mask.unsqueeze(1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
    return weights @ values


class RetrievalExperiment(nn.Module):
    def __init__(self, reference, config=RetrievalConfig()):
        super().__init__()
        self.towers, self.config = reference, config
        dim = reference.customer_tower.output_dim
        hidden = reference.customer_tower.hidden_dim
        # Isolate additional parameter construction from common tower RNG states.
        if config.long_memory != "none":
            self.long_value = nn.Linear(dim, hidden, bias=False)
            if config.long_memory == "attention":
                self.long_key = nn.Linear(dim, hidden, bias=False)
                self.long_query = nn.Parameter(torch.randn(1, hidden) * .02)
            first = reference.customer_tower.late_fusion[0]
            expanded = nn.Linear(first.in_features + hidden, first.out_features)
            with torch.no_grad():
                expanded.weight[:, :-hidden].copy_(first.weight)
                expanded.weight[:, -hidden:].zero_()
                expanded.bias.copy_(first.bias)
            reference.customer_tower.late_fusion[0] = expanded
        if config.interests:
            # K-independent first query initialization and shared value/projection weights.
            self.interest_queries = nn.Parameter(torch.stack([torch.randn(hidden) * .02 for _ in range(config.interests)]))

    def product_vectors(self, inputs):
        demand = inputs["demand_features"]
        demand = torch.cat([torch.zeros_like(demand[:, :6]), demand[:, 6:]], -1)
        return self.towers.product_tower(
            inputs["text_embedding"], inputs["brand_id"], inputs["category_id"], inputs.get("subcategory_id"),
            inputs["business_line_id"], demand, container_id=inputs.get("container_id"))

    def encode_purchase(self, inputs, vectors, *, return_memory=False):
        tower = self.towers.customer_tower
        text, tab, history, contextual, last = history_components(tower, inputs, vectors)
        if self.config.interests:
            summaries = weighted_pool(contextual, inputs["attention_mask"], self.interest_queries)
            has = inputs["attention_mask"].any(1)[:, None, None]
            history = tower.history_pooling(torch.cat([summaries, last[:, None].expand_as(summaries)], -1)) * has
            k = self.config.interests
            fused = torch.cat([text[:, None].expand(-1, k, -1), tab[:, None].expand(-1, k, -1), history], -1)
            encoded = F.normalize(tower.late_fusion(fused), dim=-1)
            return (encoded, None) if return_memory else encoded
        parts = [text, tab, history]
        if self.config.long_memory != "none":
            indices, counts = inputs["older_ids"], inputs["older_counts"]
            old = F.embedding(indices, F.pad(vectors, (0, 0, 1, 0)))
            mask = counts > 0
            if self.config.long_memory == "attention":
                # Pool raw shared product vectors using learned keys; project values afterwards.
                keys = self.long_key(old)
                logits = (keys @ self.long_query.T).squeeze(-1) / math.sqrt(keys.shape[-1])
                logits += counts.clamp_min(1).log()
                weights = logits.masked_fill(~mask, -1e9).softmax(-1) * mask
                pooled = (old * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-9)
            else:
                pooled = weighted_pool(old, mask, counts=counts)
            parts.append(self.long_value(pooled))
        encoded = F.normalize(tower.late_fusion(torch.cat(parts, -1)), dim=-1).unsqueeze(1)
        memory = parts[-1] if self.config.long_memory != "none" else None
        return (encoded, memory) if return_memory else encoded

    def score_cached(self, customer_vectors, product_vectors, product_indices=None, familiarity=None):
        similarities = customer_vectors @ product_vectors.T
        if self.config.scoring == "hard":
            scores = similarities.max(1).values
        else:
            tau = self.config.smooth_tau
            # log-mean-exp removes K's constant offset; apply reference temperature after aggregation.
            scores = tau * (torch.logsumexp(similarities / tau, 1) - math.log(customer_vectors.shape[1]))
        scores = scores / self.towers.temperature
        if product_indices is not None:
            scores = scores + self.towers.product_bias[product_indices]
        if self.towers.familiarity_weights is not None:
            if familiarity is None:
                raise ValueError("Familiarity features required")
            scores = scores + familiarity.to(scores) @ self.towers.familiarity_weights
        return scores


class PurchaseEmbeddingCache:
    """Serving API: request path is a lookup + similarities, never a tower call.

    Models/catalogues/table versions form one cache namespace. Refresh creates a
    new instance and explicitly rebuilds on stored purchase histories.
    """
    def __init__(self, model, model_version, catalogue_version):
        self.model = model.eval()
        self.version = (str(model_version), str(catalogue_version))
        self.entries = {}

    @torch.no_grad()
    def on_purchase(self, customer_id, purchase_version, inputs, vectors):
        key = str(customer_id)
        previous = self.entries.get(key)
        if previous is not None and previous[0] == purchase_version:
            return previous[1]
        encoded, memory = self.model.encode_purchase(inputs, vectors, return_memory=True)
        encoded = encoded.detach()
        self.entries[key] = (purchase_version, encoded, memory.detach() if memory is not None else None)
        return encoded

    @torch.no_grad()
    def recommend(self, customer_id, vectors, *, model_version, catalogue_version, product_indices=None, familiarity=None):
        if (str(model_version), str(catalogue_version)) != self.version:
            raise ValueError("Versioned purchase cache rebuild required")
        if str(customer_id) not in self.entries:
            raise KeyError("Customer needs a purchase snapshot or explicit cold-start initialization")
        return self.model.score_cached(self.entries[str(customer_id)][1].to(vectors.device), vectors, product_indices, familiarity)

    def state_dict(self):
        return {"version": self.version, "entries": {key: (version, value.cpu(), memory.cpu() if memory is not None else None)
                for key, (version, value, memory) in self.entries.items()}}

    def load_state_dict(self, state):
        if tuple(state["version"]) != self.version:
            raise ValueError("Purchase cache model/catalogue version mismatch")
        self.entries = state["entries"]
