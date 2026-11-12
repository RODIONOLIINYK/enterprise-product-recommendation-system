"""Permutation-equivariant candidate context and independent signal ablations."""
import copy
import math
import torch
from torch import nn
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.ranker import RankerOutput


class CachedPointwiseRanker(nn.Module):
    """Cheap matched-slate control: cached vectors and the original 13 features."""
    def __init__(self, dimension=128):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(dimension*4+14,128),nn.GELU(),nn.Dropout(.1),
                                nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, *, candidate_vectors, pairwise_features, cached_customer_vectors, **kwargs):
        v=candidate_vectors;u=cached_customer_vectors[:,None].expand_as(v)
        x=torch.cat([u,v,u*v,(u-v).abs(),pairwise_features[...,:13],(u*v).sum(-1,keepdim=True)],-1)
        scores=self.head(x).squeeze(-1)
        return RankerOutput(scores=scores,raw_scores=scores,attn_weights=None,h_attended=u,z_candidate=v,
                            token_0_weight=None,history_weights=None)


class CandidateContextRanker(nn.Module):
    MODES = ('control', 'no_repeat_flag', 'no_repeat_group', 'experts', 'group_gate',
             'recent_sku', 'hierarchy', 'complements', 'dynamic_features',
             'context_direct', 'context_dot', 'context_residual', 'combined_direct',
             'values_only', 'bias_and_values')

    def __init__(self, base, mode='control'):
        super().__init__()
        if mode not in self.MODES: raise ValueError('Unknown ranker mode')
        self.base, self.mode = base, mode
        self.base.configure_cached_training(True)
        width = base.hidden_dim
        fused = width * 3 + 13
        self.head = None
        self.uses_context = mode.startswith('context_') or mode == 'combined_direct'
        if self.uses_context:
            self.project = nn.Linear(fused, width)
            if mode == 'combined_direct':
                self.dynamic_project = nn.Linear(len(NAMES) - 13, width, bias=False)
                nn.init.zeros_(self.dynamic_project.weight)
            self.context = nn.TransformerEncoderLayer(width, 4, 256, dropout=.1, activation='gelu',
                                                      batch_first=True, norm_first=True)
            if mode == 'context_dot':
                self.user_projection = nn.Linear(width, width)
                self.candidate_projection = nn.Linear(width, width)
            else:
                self.head = nn.Sequential(nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, 1))
                if mode == 'context_residual':
                    nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        elif mode == 'experts':
            self.repeat_head = copy.deepcopy(base.mlp)
            self.unseen_head = copy.deepcopy(base.mlp)
        elif mode in ('recent_sku', 'hierarchy', 'complements', 'dynamic_features'):
            columns = {'recent_sku': [33, 34, 38], 'hierarchy': list(range(33, 40)),
                       'complements': [40], 'dynamic_features': list(range(13, len(NAMES)))}[mode]
            self.register_buffer('extra_columns', torch.tensor(columns, dtype=torch.long))
            self.head = copy.deepcopy(base.mlp)
            old = self.head[0]
            self.head[0] = nn.Linear(fused + len(columns), old.out_features)
            with torch.no_grad():
                self.head[0].weight[:, :fused].copy_(old.weight)
                self.head[0].weight[:, fused:].zero_()
                self.head[0].bias.copy_(old.bias)
        if mode == 'group_gate': self.repeat_gate = nn.Parameter(torch.tensor(2.))
        if mode == 'values_only':
            base.register_parameter('statistics_attention_weights', None)
        self.requires_customer_vectors = mode == 'context_dot'

    def forward(self, *, pairwise_features, cached_customer_vectors=None, candidate_mask=None,
                disable_candidate_attention=False, **kwargs):
        repeat = pairwise_features[..., 0] > .5
        features = pairwise_features[..., :13].clone()
        if self.mode == 'no_repeat_flag': features[..., 0] = 0
        elif self.mode == 'no_repeat_group': features[..., :5] = 0
        elif self.mode == 'group_gate': features[..., :5] *= self.repeat_gate.sigmoid()
        encoded = self.base.encode_candidate_features(pairwise_features=features,
            memory_value_statistics=self.mode == 'bias_and_values', **kwargs)
        if self.uses_context:
            projected = self.project(encoded.features)
            if self.mode == 'combined_direct': projected = projected + self.dynamic_project(pairwise_features[..., 13:])
            if disable_candidate_attention:
                # Remove only cross-candidate mixing, retaining the trained
                # pre-norm feed-forward path and residual at each position.
                layer=self.context
                context=projected+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(projected))))))
            else:
                context = self.context(projected, src_key_padding_mask=~candidate_mask if candidate_mask is not None else None)
            if self.mode == 'context_dot':
                if cached_customer_vectors is None: raise ValueError('Cached customer vector required')
                logits = torch.einsum('bd,bcd->bc', self.user_projection(cached_customer_vectors),
                                      self.candidate_projection(context)) / math.sqrt(context.shape[-1])
            else:
                logits = self.head(torch.cat([projected, context], -1)).squeeze(-1)
                if self.mode == 'context_residual': logits = logits + self.base.score_candidates(encoded)
        elif self.mode == 'experts':
            logits = torch.where(repeat, self.repeat_head(encoded.features).squeeze(-1), self.unseen_head(encoded.features).squeeze(-1))
        elif self.head is not None:
            logits = self.head(torch.cat([encoded.features, pairwise_features[..., self.extra_columns]], -1)).squeeze(-1)
        else: logits = self.base.score_candidates(encoded)
        if candidate_mask is not None: logits = logits.masked_fill(~candidate_mask, -torch.inf)
        return encoded.to_output(logits)
