"""Capacity changes must preserve unrelated weights, masks, gradients and reloads."""

import copy

import pytest
import torch

from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.candidate_token_enrichment.model import CandidateTokenEnrichmentRanker
from recommender.experiments.ranker_capacity.model import (
    VARIANTS, CapacityRanker, load_checkpoint, save_checkpoint,
)


def fixture():
    torch.manual_seed(42)
    base = UPSCANRRanker(text_dim=32, num_products=8, num_categories=4,
                        num_subcategories=4, num_business_lines=4, num_cities=4,
                        num_regions=4, num_industries=4, statistics_attention=True,
                        excluded_statistics=('history_repeat',), cached_only=True)
    model = CandidateTokenEnrichmentRanker.from_dynamic_context_ranker(CandidateContextRanker(base, 'dynamic_features'))
    mask = torch.ones(2, 32, dtype=torch.long)
    mask[0, 10:] = 0
    x = dict(text_embedding=torch.randn(2, 32), city_id=torch.zeros(2, dtype=torch.long),
             region_id=torch.zeros(2, dtype=torch.long), industry_id=torch.zeros(2, dtype=torch.long),
             tabular_features=torch.rand(2, 8), sequence=torch.rand(2, 32, 9), attention_mask=mask,
             seq_product_ids=torch.ones(2, 32, dtype=torch.long), seq_category_ids=torch.ones(2, 32, dtype=torch.long),
             seq_subcategory_ids=torch.ones(2, 32, dtype=torch.long), seq_business_line_ids=torch.ones(2, 32, dtype=torch.long),
             seq_purchase_dates=torch.full((2, 32), 90. / 365.25), prediction_day=torch.full((2,), 100.),
             candidate_vectors=torch.randn(2, 4, 128), pairwise_features=torch.rand(2, 4, 41),
             candidate_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]))
    return model, x


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_capacity_scope_gradients_and_padding(variant):
    source, x = fixture()
    model = CapacityRanker.from_control(source, variant=variant)
    for key, value in source.state_dict().items():
        changed = ((variant == 'wider_head' and key.startswith('base_head.')) or
                   (variant == 'wider_ffn' and key.startswith('base.memory_encoder.layers.0.linear')))
        if not changed:
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert model.base.memory_encoder.num_layers == VARIANTS[variant]['memory_layers']
    assert model.base.memory_encoder.layers[0].linear1.out_features == VARIANTS[variant]['memory_ffn']
    assert model.base_head[0].out_features == VARIANTS[variant]['head_widths'][0]
    labels = torch.zeros(2, 4)
    labels[:, 0] = 1
    output = model(**x)
    loss = RankerLoss()(output.logits, labels, candidate_mask=x['candidate_mask'])['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert model.base_head[0].weight.grad.abs().sum() > 0
    assert model.base.memory_encoder.layers[-1].self_attn.in_proj_weight.grad.abs().sum() > 0
    model.eval()
    altered = copy.deepcopy(x)
    altered['sequence'][x['attention_mask'] == 0] = float('nan')
    altered['candidate_vectors'][~x['candidate_mask']] = float('nan')
    altered['pairwise_features'][~x['candidate_mask']] = float('nan')
    with torch.no_grad():
        expected = model(**x).logits
        torch.testing.assert_close(expected, model(**altered).logits, rtol=0, atol=0)
        if variant == 'control32':
            source.eval()
            torch.testing.assert_close(expected, source(**x).logits, rtol=0, atol=0)


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_capacity_checkpoint_replay_and_tamper(tmp_path, variant):
    source, x = fixture()
    model = CapacityRanker.from_control(source, variant=variant).eval()
    path = tmp_path / 'capacity.pt'
    provenance = {'statistics_policy': 'rolling_observed_v2', 'teacher_score_included': False}
    save_checkpoint(path, model, epoch=1, metrics={}, selection_criterion='test',
                    provenance=provenance, feature_contract_digest='test')
    loaded, _ = load_checkpoint(path, expected_provenance=provenance, expected_feature_contract_digest='test')
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**x).logits, loaded(**x).logits, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    payload['model_config']['capacity'] = {**payload['model_config']['capacity'], 'memory_layers': 9}
    torch.save(payload, path)
    with pytest.raises(ValueError, match='configuration mismatch'):
        load_checkpoint(path)
