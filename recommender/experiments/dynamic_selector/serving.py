"""Purchase-triggered towers plus cheap clock-dependent pointwise scoring."""
import torch
from recommender.experimental_retrieval import PurchaseEmbeddingCache
from .model import select_candidates
from .feature_schema import quantize_features


class DynamicSelectorService:
    def __init__(self, selector, upstream_model, vectors, product_ids, model_version, catalogue_version):
        if len(set(product_ids)) != len(product_ids) or len(product_ids) != len(vectors):
            raise ValueError('Catalogue identity/vector alignment')
        self.selector = selector.eval() if selector is not None else None
        self.cache = PurchaseEmbeddingCache(upstream_model, model_version, catalogue_version)
        self.vectors, self.product_ids = vectors, tuple(product_ids)

    def on_purchase(self, customer_id, purchase_version, inputs):
        return self.cache.on_purchase(customer_id, purchase_version, inputs, self.vectors)

    @torch.no_grad()
    def score(self, customer_id, as_of, builder, familiarity, *, model_version, catalogue_version):
        if self.cache.version != (str(model_version), str(catalogue_version)):
            raise ValueError('Explicit versioned catalogue/model rebuild required')
        entry = self.cache.entries.get(str(customer_id))
        if entry is None: raise KeyError('Purchase snapshot or explicit cold-start rebuild required')
        u = entry[1].to(self.vectors.device)
        scores = self.cache.model.score_cached(u, self.vectors,
            torch.arange(len(self.vectors), device=self.vectors.device), familiarity)
        if self.selector is None: return scores
        features = torch.as_tensor(quantize_features(builder(str(customer_id), as_of)), device=self.vectors.device)
        return self.selector(u.squeeze(1), self.vectors, features.unsqueeze(0), scores)

    def recommend(self, customer_id, as_of, builder, familiarity, allowed, unseen, policy='quota50_25', **versions):
        scores = self.score(customer_id, as_of, builder, familiarity, **versions)
        return select_candidates(scores, allowed.to(scores.device), unseen.to(scores.device), policy)

    def rebuilt(self, selector, upstream_model, vectors, product_ids, model_version, catalogue_version):
        """New empty namespace: explicitly replay purchase snapshots before switching."""
        if (str(model_version), str(catalogue_version)) == self.cache.version:
            raise ValueError('A rebuild needs a new model/catalogue version')
        return type(self)(selector, upstream_model, vectors, product_ids, model_version, catalogue_version)
