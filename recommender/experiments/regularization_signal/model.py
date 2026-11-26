"""Change active ranker regularization without changing its weights or input signals."""

from pathlib import Path

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_token_enrichment.model import (
    CandidateTokenEnrichmentRanker, validate_enrichment_ranker_input_contract,
)
from recommender.experiments.direct_feature_access.model import DirectFeatureRanker
from recommender.shared_data import atomic_torch_save

VARIANTS = {
    'control': {'head_scale': 1., 'profile_scale': 1., 'attention_scale': 1., 'weight_decay': 1e-4, 'direct52': False},
    'head_half': {'head_scale': .5, 'profile_scale': 1., 'attention_scale': 1., 'weight_decay': 1e-4, 'direct52': False},
    'profile_half': {'head_scale': 1., 'profile_scale': .5, 'attention_scale': 1., 'weight_decay': 1e-4, 'direct52': False},
    'attention_half': {'head_scale': 1., 'profile_scale': 1., 'attention_scale': .5, 'weight_decay': 1e-4, 'direct52': False},
    'weight_decay_zero': {'head_scale': 1., 'profile_scale': 1., 'attention_scale': 1., 'weight_decay': 0., 'direct52': False},
    'all_zero': {'head_scale': 0., 'profile_scale': 0., 'attention_scale': 0., 'weight_decay': 0., 'direct52': False},
    'direct52_all_zero': {'head_scale': 0., 'profile_scale': 0., 'attention_scale': 0., 'weight_decay': 0., 'direct52': True},
}


def active_dropout_inventory(model):
    """Both ordinary activation dropout and float-valued attention dropout matter."""
    inventory = {}
    groups = {'head': model.base_head, 'profile_text': model.base.text_proj,
              'profile_tabular': model.base.tabular_proj, 'memory': model.base.memory_encoder,
              'cross_attention': model.base.cross_attention}
    for group, root in groups.items():
        for name, module in root.named_modules():
            if isinstance(module, nn.Dropout):
                inventory[f'{group}.{name}'] = float(module.p)
            elif isinstance(module, nn.MultiheadAttention):
                inventory[f'{group}.{name}.attention_weights'] = float(module.dropout)
    return inventory


class RegularizedRanker(DirectFeatureRanker):
    @classmethod
    def from_control(cls, source, *, reg_variant, categorical_lookup, product_ids):
        if reg_variant not in VARIANTS:
            raise ValueError('Unknown regularization variant.')
        cfg = VARIANTS[reg_variant]
        model = super().from_control(source, variant='profile8_and_categories' if cfg['direct52'] else 'control',
                                     categorical_lookup=categorical_lookup, product_ids=product_ids)
        model.reg_variant = reg_variant
        groups = [(model.base_head, cfg['head_scale']), (model.base.text_proj, cfg['profile_scale']),
                  (model.base.tabular_proj, cfg['profile_scale']),
                  (model.base.memory_encoder, cfg['attention_scale']),
                  (model.base.cross_attention, cfg['attention_scale'])]
        for root, scale in groups:
            for module in root.modules():
                if isinstance(module, nn.Dropout):
                    module.p *= scale
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout *= scale
        return model

    def model_config(self):
        return {**super().model_config(), 'model_class': 'RegularizedRanker',
                'regularization_variant': self.reg_variant, 'regularization': VARIANTS[self.reg_variant],
                'active_dropout_inventory': active_dropout_inventory(self)}

    def input_contract(self, *, feature_contract_digest=None):
        contract = super().input_contract(feature_contract_digest=feature_contract_digest)
        contract.pop('contract_digest')
        contract.update(regularization_variant=self.reg_variant, regularization=VARIANTS[self.reg_variant],
                        active_dropout_inventory=active_dropout_inventory(self))
        return {**contract, 'contract_digest': json_digest(contract)}


def save_checkpoint(path, model, *, epoch, metrics, selection_criterion, provenance,
                    feature_contract_digest, settings=None):
    atomic_torch_save({
        'checkpoint_format_version': 1, 'model_class': 'RegularizedRanker',
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
    if payload.get('model_class') != 'RegularizedRanker' or payload.get('checkpoint_format_version') != 1:
        raise ValueError('Unexpected regularization checkpoint.')
    cfg = payload['model_config']
    if (cfg.get('regularization') != VARIANTS.get(cfg.get('regularization_variant'))
            or cfg.get('candidate_attention_enabled') or cfg.get('token_enrichment_enabled')
            or expected_candidate_attention_enabled or expected_token_enrichment_enabled):
        raise ValueError('Regularization configuration mismatch.')
    source = CandidateTokenEnrichmentRanker.from_model_config(cfg)
    model = RegularizedRanker.from_control(source, reg_variant=cfg['regularization_variant'],
                                           categorical_lookup=cfg['categorical_lookup'], product_ids=cfg['product_ids'])
    if model.model_config() != cfg:
        raise ValueError('Regularization architecture/dropout mismatch.')
    digest = expected_feature_contract_digest or payload['input_contract'].get('feature_contract_digest')
    validate_enrichment_ranker_input_contract(payload['input_contract'], model.input_contract(feature_contract_digest=digest))
    provenance = payload['provenance']
    if (provenance.get('statistics_policy') != 'rolling_observed_v2'
            or provenance.get('teacher_score_included') is not False
            or (expected_provenance is not None and dict(expected_provenance) != provenance)):
        raise ValueError('Regularization provenance mismatch.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if not torch.equal(model.categorical_lookup, torch.tensor(cfg['categorical_lookup'])):
        raise ValueError('Categorical lookup/config mismatch.')
    return model.to(map_location), payload
