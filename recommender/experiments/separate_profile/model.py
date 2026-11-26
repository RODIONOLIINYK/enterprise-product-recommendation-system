"""Two profile tokens, with isolated 32/64-event self-attention memory variants."""

import copy
from pathlib import Path

import torch
from torch import nn

from recommender.artifacts import json_digest
from recommender.experiments.candidate_token_enrichment.model import (
    CandidateTokenEnrichmentRanker, EnrichableUPSCANRRanker,
    validate_enrichment_ranker_input_contract,
)
from recommender.shared_data import atomic_torch_save


class SeparateProfileBase(EnrichableUPSCANRRanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_layout = 'fused1'
        self.register_parameter('attribute_type_offset', None)

    def set_profile_layout(self, layout):
        if layout not in ('fused1', 'fused2', 'split2'):
            raise ValueError('Unknown profile layout.')
        self.profile_layout = layout
        if layout != 'fused1' and self.attribute_type_offset is None:
            self.attribute_type_offset = nn.Parameter(self.type_embedding.weight.new_zeros(self.hidden_dim))
        self.profile_fusion.requires_grad_(layout != 'split2')

    def build_profile_tokens(self, z_text, z_tab):
        if self.profile_layout == 'fused1':
            return super().build_profile_tokens(z_text, z_tab)
        if self.profile_layout == 'fused2':
            fused = self.profile_fusion(torch.cat([z_text, z_tab], dim=-1))
            text, attributes = fused, fused
        else:
            text, attributes = z_text, z_tab
        profile_type = self.type_embedding.weight[0]
        return torch.stack([text + profile_type, attributes + profile_type + self.attribute_type_offset], dim=1)

    def build_key_padding_mask(self, attention_mask):
        profiles = 1 if self.profile_layout == 'fused1' else 2
        prefix = torch.zeros(attention_mask.shape[0], profiles, dtype=torch.bool, device=attention_mask.device)
        return torch.cat([prefix, attention_mask == 0], dim=1)

    def build_memory_statistics(self, tabular_features, purchase_statistics):
        default = super().build_memory_statistics(tabular_features, purchase_statistics)
        if self.profile_layout == 'fused1':
            return default
        # Both two-token arms use this identical mapping: profile stats on the attribute token,
        # zeros on the text token, and original event statistics on every history token.
        return torch.cat([torch.zeros_like(default[:, :1]), default], dim=1)

    def summarize_memory_attention(self, attn_weights):
        profiles = 1 if self.profile_layout == 'fused1' else 2
        return attn_weights[..., :profiles].sum(dim=-1), attn_weights[..., profiles:].sum(dim=-1)


class SeparateProfileRanker(CandidateTokenEnrichmentRanker):
    @classmethod
    def from_control(cls, source, *, profile_layout, history_length):
        if source.candidate_attention_enabled or source.token_enrichment_enabled:
            raise ValueError('Source must be the pointwise control without enrichment.')
        if history_length not in (32, 64):
            raise ValueError('Only versioned 32/64-event variants are supported.')
        base = SeparateProfileBase.from_upscanr(source.base)
        base.set_profile_layout(profile_layout)
        result = cls(base, copy.deepcopy(source.base_head), candidate_attention_enabled=False,
                     token_enrichment_enabled=False, attention_num_heads=source.attention_num_heads,
                     attention_dim_feedforward=source.attention_dim_feedforward,
                     attention_dropout=source.attention_dropout)
        state = result.state_dict(); state.update(source.state_dict())
        result.load_state_dict(state, strict=True)
        result.history_length = history_length
        return result

    def model_config(self):
        return {**super().model_config(), 'model_class': 'SeparateProfileRanker',
                'profile_layout': self.base.profile_layout, 'history_length': self.history_length}

    def input_contract(self, *, feature_contract_digest=None):
        contract = super().input_contract(feature_contract_digest=feature_contract_digest)
        contract.pop('contract_digest')
        contract.update(profile_layout=self.base.profile_layout, history_length=self.history_length,
                        profile_token_count=1 if self.base.profile_layout == 'fused1' else 2,
                        history_pooling='none; self-attended event tokens retained for candidate cross-attention',
                        position_policy='legacy learned32 positions; linear interpolation for64',
                        statistics_layout='text_zero_attribute_profile8_then_event9' if self.base.profile_layout != 'fused1' else 'legacy_profile8_then_event9')
        return {**contract, 'contract_digest': json_digest(contract)}

    def forward(self, *, sequence, **kwargs):
        if sequence.shape[1] != self.history_length:
            raise ValueError(f'Expected {self.history_length} event positions, got {sequence.shape[1]}.')
        return super().forward(sequence=sequence, **kwargs)


def save_checkpoint(path, model, *, epoch, metrics, selection_criterion, provenance,
                    feature_contract_digest, settings=None):
    payload = {
        'checkpoint_format_version': 1, 'model_class': 'SeparateProfileRanker',
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
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get('model_class') != 'SeparateProfileRanker' or payload.get('checkpoint_format_version') != 1:
        raise ValueError('Unexpected separate-profile checkpoint type/version.')
    config = payload['model_config']
    if config.get('candidate_attention_enabled') is not False or config.get('token_enrichment_enabled') is not False:
        raise ValueError('Candidate mixing and statistical enrichment must be disabled.')
    if expected_candidate_attention_enabled or expected_token_enrichment_enabled:
        raise ValueError('Unexpected enabled branch request.')
    # Reconstruct common UPS-CANR dimensions solely from the persisted model configuration.
    control = CandidateTokenEnrichmentRanker.from_model_config(config)
    model = SeparateProfileRanker.from_control(control, profile_layout=config['profile_layout'], history_length=config['history_length'])
    digest = expected_feature_contract_digest or payload['input_contract'].get('feature_contract_digest')
    validate_enrichment_ranker_input_contract(payload['input_contract'], model.input_contract(feature_contract_digest=digest))
    provenance = payload['provenance']
    if provenance.get('statistics_policy') != 'rolling_observed_v2' or provenance.get('teacher_score_included') is not False:
        raise ValueError('Unexpected input provenance.')
    if expected_provenance is not None and dict(expected_provenance) != provenance:
        raise ValueError('Separate-profile provenance mismatch.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(map_location)
    return model, payload
