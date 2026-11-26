"""Change one capacity axis while retaining the existing input and masking paths."""

import copy
from pathlib import Path

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_token_enrichment.model import (
    CandidateTokenEnrichmentRanker,
    validate_enrichment_ranker_input_contract,
)
from recommender.shared_data import atomic_torch_save

VARIANTS = {
    'control32': {'head_widths': [256, 128, 64], 'memory_ffn': 256, 'memory_layers': 1},
    'wider_head': {'head_widths': [512, 256, 128], 'memory_ffn': 256, 'memory_layers': 1},
    'wider_ffn': {'head_widths': [256, 128, 64], 'memory_ffn': 512, 'memory_layers': 1},
    'deeper_memory': {'head_widths': [256, 128, 64], 'memory_ffn': 256, 'memory_layers': 2},
}


class CapacityRanker(CandidateTokenEnrichmentRanker):
    @classmethod
    def from_control(cls, source, *, variant, init_seed=2042):
        if variant not in VARIANTS:
            raise ValueError('Unknown capacity variant.')
        if source.candidate_attention_enabled or source.token_enrichment_enabled:
            raise ValueError('Capacity source must be the un-enriched pointwise control.')
        model = cls(copy.deepcopy(source.base), copy.deepcopy(source.base_head),
                    candidate_attention_enabled=False, token_enrichment_enabled=False,
                    attention_num_heads=source.attention_num_heads,
                    attention_dim_feedforward=source.attention_dim_feedforward,
                    attention_dropout=source.attention_dropout)
        model.load_state_dict(source.state_dict(), strict=True)
        model.variant = variant
        model.init_seed = int(init_seed)
        # Forking preserves the training RNG; all untouched parameters are exact copies.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model.init_seed)
            if variant == 'wider_head':
                layers = []
                previous = model.base_head[0].in_features
                for width, dropout in zip(VARIANTS[variant]['head_widths'], (0.2, 0.2, 0.1)):
                    layers.extend([nn.Linear(previous, width), nn.LayerNorm(width),
                                   nn.GELU(), nn.Dropout(dropout)])
                    previous = width
                model.base_head = nn.Sequential(*layers, nn.Linear(previous, 1))
            elif variant == 'wider_ffn':
                layer = model.base.memory_encoder.layers[0]
                layer.linear1 = nn.Linear(model.base.hidden_dim, 512)
                layer.linear2 = nn.Linear(512, model.base.hidden_dim)
            elif variant == 'deeper_memory':
                original = model.base.memory_encoder.layers[0]
                layer = nn.TransformerEncoderLayer(
                    d_model=model.base.hidden_dim, nhead=model.base.num_heads,
                    dim_feedforward=256, dropout=original.dropout.p,
                    activation='gelu', batch_first=True, norm_first=original.norm_first,
                    layer_norm_eps=original.norm1.eps,
                )
                model.base.memory_encoder.layers.append(layer)
                model.base.memory_encoder.num_layers = 2
        return model

    def model_config(self):
        return {**super().model_config(), 'model_class': 'CapacityRanker',
                'capacity_variant': self.variant, 'capacity': VARIANTS[self.variant],
                'capacity_init_seed': self.init_seed}

    def input_contract(self, *, feature_contract_digest=None):
        contract = super().input_contract(feature_contract_digest=feature_contract_digest)
        contract.pop('contract_digest')
        contract.update(capacity_variant=self.variant, capacity=VARIANTS[self.variant],
                        history_length=32, profile_layout='original_fused_profile')
        return {**contract, 'contract_digest': json_digest(contract)}


def save_checkpoint(path, model, *, epoch, metrics, selection_criterion, provenance,
                    feature_contract_digest, settings=None):
    payload = {
        'checkpoint_format_version': 1, 'model_class': 'CapacityRanker',
        'model_module': __name__, 'model_config': model.model_config(),
        'input_contract': model.input_contract(feature_contract_digest=feature_contract_digest),
        'provenance': dict(provenance), 'epoch': epoch, 'metrics': dict(metrics),
        'selection_criterion': selection_criterion, 'settings': dict(settings or {}),
        'model_state_dict': model.state_dict(),
    }
    atomic_torch_save(payload, Path(path))
    return Path(path)


def load_checkpoint(path, *, expected_provenance=None, expected_feature_contract_digest=None,
                    expected_candidate_attention_enabled=None, expected_token_enrichment_enabled=None,
                    map_location='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_class') != 'CapacityRanker' or payload.get('checkpoint_format_version') != 1:
        raise ValueError('Unexpected capacity checkpoint type/version.')
    config = payload['model_config']
    if config.get('candidate_attention_enabled') is not False or config.get('token_enrichment_enabled') is not False:
        raise ValueError('Capacity experiments require both experimental branches disabled.')
    if expected_candidate_attention_enabled or expected_token_enrichment_enabled:
        raise ValueError('Unexpected enabled branch request.')
    variant = config['capacity_variant']
    if variant not in VARIANTS or config.get('capacity') != VARIANTS[variant]:
        raise ValueError('Capacity configuration mismatch.')
    source = CandidateTokenEnrichmentRanker.from_model_config(config)
    model = CapacityRanker.from_control(source, variant=variant, init_seed=config['capacity_init_seed'])
    digest = expected_feature_contract_digest or payload['input_contract'].get('feature_contract_digest')
    validate_enrichment_ranker_input_contract(payload['input_contract'], model.input_contract(feature_contract_digest=digest))
    provenance = payload['provenance']
    if provenance.get('statistics_policy') != 'rolling_observed_v2' or provenance.get('teacher_score_included') is not False:
        raise ValueError('Unexpected input provenance.')
    if expected_provenance is not None and dict(expected_provenance) != provenance:
        raise ValueError('Capacity provenance mismatch.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(map_location)
    return model, payload
