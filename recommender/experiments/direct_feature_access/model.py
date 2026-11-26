"""Give the pointwise scorer direct profile8 and SKU/category/line inputs.

Keep every existing scalar, frozen vector, attention block and head width. Added
first-layer columns start at zero, preserving the control's initial function.
Catalogue categoricals follow the same snapshot-unverified policy as CatBoost.
"""

import copy
from pathlib import Path

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_token_enrichment.model import (
    CandidateTokenEnrichmentRanker, validate_enrichment_ranker_input_contract,
)
from recommender.experiments.candidate_token_enrichment.normalization import sanitize_candidate_inputs
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.shared_data import atomic_torch_save

VARIANTS = {
    'control': (False, False),
    'profile8': (True, False),
    'candidate_categories': (False, True),
    'profile8_and_categories': (True, True),
}
CATEGORY_FIELDS = ('product_id', 'product_category', 'business_line')
EMBEDDING_DIMS = (16, 8, 8)


class DirectFeatureRanker(CandidateTokenEnrichmentRanker):
    @classmethod
    def from_control(cls, source, *, variant, categorical_lookup, product_ids, init_seed=3042):
        if variant not in VARIANTS:
            raise ValueError('Unknown direct feature variant.')
        if source.candidate_attention_enabled or source.token_enrichment_enabled:
            raise ValueError('Requires the un-enriched pointwise control.')
        lookup = torch.as_tensor(categorical_lookup, dtype=torch.long).clone()
        if (lookup.ndim != 2 or lookup.shape != (len(product_ids), 3)
                or len(set(product_ids)) != len(product_ids) or (lookup < 0).any()
                or len(product_ids) == 0):
            raise ValueError('Unique aligned product IDs and non-negative categorical lookup required.')
        model = cls(copy.deepcopy(source.base), copy.deepcopy(source.base_head),
                    candidate_attention_enabled=False, token_enrichment_enabled=False,
                    attention_num_heads=source.attention_num_heads,
                    attention_dim_feedforward=source.attention_dim_feedforward,
                    attention_dropout=source.attention_dropout)
        model.load_state_dict(source.state_dict(), strict=True)
        model.variant = variant
        model.product_ids = list(product_ids)
        model.init_seed = int(init_seed)
        model.direct_profile, model.direct_categories = VARIANTS[variant]
        model.register_buffer('categorical_lookup', lookup)
        model.category_embeddings = nn.ModuleList()
        extra = 8 * model.direct_profile + sum(EMBEDDING_DIMS) * model.direct_categories
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model.init_seed)
            if model.direct_categories:
                model.category_embeddings.extend(
                    nn.Embedding(int(lookup[:, i].max()) + 1, dim)
                    for i, dim in enumerate(EMBEDDING_DIMS)
                )
            if extra:
                original = model.base_head[0]
                expanded = nn.Linear(original.in_features + extra, original.out_features)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :425].copy_(original.weight)
                    expanded.bias.copy_(original.bias)
                model.base_head[0] = expanded
        return model

    def model_config(self):
        cfg = super().model_config()
        cfg.update(model_class='DirectFeatureRanker', head_input_dim=425,
                   extended_head_input_dim=self.base_head[0].in_features,
                   direct_variant=self.variant, direct_init_seed=self.init_seed,
                   product_ids=self.product_ids, categorical_lookup=self.categorical_lookup.cpu().tolist(),
                   category_fields=list(CATEGORY_FIELDS), embedding_dims=list(EMBEDDING_DIMS))
        return cfg

    def input_contract(self, *, feature_contract_digest=None):
        contract = super().input_contract(feature_contract_digest=feature_contract_digest)
        contract.pop('contract_digest')
        contract.update(direct_variant=self.variant,
                        direct_profile_features=list(LEGACY_PROFILE_FEATURES) if self.direct_profile else [],
                        direct_categorical_fields=list(CATEGORY_FIELDS) if self.direct_categories else [],
                        extended_head_input_dim=self.base_head[0].in_features,
                        product_lookup_digest=json_digest({'product_ids': self.product_ids,
                                                          'lookup': self.categorical_lookup.cpu().tolist()}))
        return {**contract, 'contract_digest': json_digest(contract)}

    def forward(self, *, pairwise_features, candidate_mask=None, candidate_indices=None,
                disable_direct_profile=False, disable_direct_categories=False, **kwargs):
        if kwargs.pop('mlp_score', None) is not None:
            raise ValueError('Stage 1 MLP scores are excluded.')
        # These flags are accepted by the common evaluator; both branches are disabled.
        for flag in ('disable_candidate_attention', 'disable_token_enrichment',
                     'disable_candidate_enrichment', 'disable_profile_enrichment', 'disable_history_enrichment'):
            kwargs.pop(flag, None)
        kwargs.pop('cached_customer_vectors', None)
        pf, vectors, indices = sanitize_candidate_inputs(
            pairwise_features, candidate_mask=candidate_mask,
            candidate_vectors=kwargs.get('candidate_vectors'), candidate_indices=candidate_indices)
        if vectors is not None:
            kwargs['candidate_vectors'] = vectors
        encoded = self.base.encode_candidate_features(
            pairwise_features=pf[..., :13], pairwise_features_41=pf,
            candidate_mask=candidate_mask, candidate_indices=indices, **kwargs)
        parts = [encoded.features, pf[..., self.extra_columns]]
        if self.direct_profile:
            profile = kwargs['tabular_features']
            if profile.shape != (pf.shape[0], 8) or not torch.isfinite(profile).all():
                raise ValueError('Finite profile8 required.')
            if disable_direct_profile:
                profile = torch.zeros_like(profile)
            parts.append(profile.unsqueeze(1).expand(-1, pf.shape[1], -1))
        if self.direct_categories:
            if indices is None or indices.shape != pf.shape[:2] or indices.dtype != torch.long:
                raise ValueError('Aligned long candidate indices required.')
            if ((indices < 0) | (indices >= len(self.product_ids))).any():
                raise ValueError('Candidate index outside the ordered product universe.')
            categorical = self.categorical_lookup[indices]
            embeds = torch.cat([emb(categorical[..., i]) for i, emb in enumerate(self.category_embeddings)], -1)
            parts.append(torch.zeros_like(embeds) if disable_direct_categories else embeds)
        x = torch.cat(parts, -1)
        if candidate_mask is not None:
            x = torch.where(candidate_mask.unsqueeze(-1), x, x.new_zeros(()))
        logits = self.base_head(x).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)
        return encoded.to_output(logits)


def save_checkpoint(path, model, *, epoch, metrics, selection_criterion, provenance,
                    feature_contract_digest, settings=None):
    atomic_torch_save({
        'checkpoint_format_version': 1, 'model_class': 'DirectFeatureRanker',
        'model_config': model.model_config(),
        'input_contract': model.input_contract(feature_contract_digest=feature_contract_digest),
        'provenance': dict(provenance), 'epoch': epoch, 'metrics': dict(metrics),
        'selection_criterion': selection_criterion, 'settings': dict(settings or {}),
        'model_state_dict': model.state_dict(),
    }, Path(path))
    return Path(path)


def load_checkpoint(path, *, expected_provenance=None, expected_feature_contract_digest=None,
                    expected_candidate_attention_enabled=None, expected_token_enrichment_enabled=None,
                    map_location='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_class') != 'DirectFeatureRanker' or payload.get('checkpoint_format_version') != 1:
        raise ValueError('Unexpected direct feature checkpoint.')
    cfg = payload['model_config']
    if (cfg['candidate_attention_enabled'] or cfg['token_enrichment_enabled']
            or expected_candidate_attention_enabled or expected_token_enrichment_enabled
            or cfg.get('category_fields') != list(CATEGORY_FIELDS)
            or cfg.get('embedding_dims') != list(EMBEDDING_DIMS)):
        raise ValueError('Unexpected direct feature architecture.')
    source = CandidateTokenEnrichmentRanker.from_model_config(cfg)
    model = DirectFeatureRanker.from_control(
        source, variant=cfg['direct_variant'], categorical_lookup=cfg['categorical_lookup'],
        product_ids=cfg['product_ids'], init_seed=cfg['direct_init_seed'])
    if cfg != model.model_config():
        raise ValueError('Direct feature configuration mismatch.')
    digest = expected_feature_contract_digest or payload['input_contract'].get('feature_contract_digest')
    validate_enrichment_ranker_input_contract(payload['input_contract'], model.input_contract(feature_contract_digest=digest))
    provenance = payload['provenance']
    if (provenance.get('statistics_policy') != 'rolling_observed_v2'
            or provenance.get('teacher_score_included') is not False
            or (expected_provenance is not None and dict(expected_provenance) != provenance)):
        raise ValueError('Direct feature provenance mismatch.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if not torch.equal(model.categorical_lookup, torch.tensor(cfg['categorical_lookup'])):
        raise ValueError('Categorical lookup/config mismatch.')
    return model.to(map_location), payload
