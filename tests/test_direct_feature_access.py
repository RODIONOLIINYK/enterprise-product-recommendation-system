"""Direct inputs retain temporal scalar contracts, masking and checkpoint identity."""

import copy

import pytest
import torch

from recommender.experiments.direct_feature_access.model import (
    VARIANTS, DirectFeatureRanker, load_checkpoint, save_checkpoint,
)
from test_ranker_capacity import fixture


def make(variant):
    source, x = fixture()
    x['candidate_indices'] = torch.tensor([[0, 1, 2, -999], [2, 1, 3, 0]])
    lookup = torch.tensor([[0, 0, 0], [1, 1, 0], [2, 0, 1], [3, 1, 1]])
    return source, DirectFeatureRanker.from_control(
        source, variant=variant, categorical_lookup=lookup, product_ids=['a', 'b', 'c', 'd']), x


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_init_parity_padding_order_and_actual_input_gradients(variant):
    source, model, x = make(variant)
    source.eval(); model.eval()
    with torch.no_grad():
        expected = source(**x).logits
        actual = model(**x).logits
        torch.testing.assert_close(expected, actual, rtol=0, atol=2e-6)
    # Make new first-layer columns nonzero; test dependence and symmetry after fitting.
    with torch.no_grad():
        model.base_head[0].weight[:, 425:].normal_(0, .05)
    changed = copy.deepcopy(x)
    changed['pairwise_features'][~x['candidate_mask']] = float('nan')
    changed['candidate_vectors'][~x['candidate_mask']] = float('nan')
    with torch.no_grad():
        intact = model(**x).logits
        torch.testing.assert_close(intact, model(**changed).logits, rtol=0, atol=0)
        reverse = {k: v.flip(1) if k in ('pairwise_features', 'candidate_vectors', 'candidate_indices', 'candidate_mask') else v
                   for k, v in x.items()}
        torch.testing.assert_close(intact, model(**reverse).logits.flip(1), rtol=0, atol=2e-6)
    if model.direct_profile:
        profile = x['tabular_features'].clone().requires_grad_()
        model(**{**x, 'tabular_features': profile}).logits[x['candidate_mask']].sum().backward()
        assert profile.grad.abs().sum() > 0
    if model.direct_categories:
        model.zero_grad()
        model(**x).logits[x['candidate_mask']].sum().backward()
        assert all(e.weight.grad.abs().sum() > 0 for e in model.category_embeddings)
        with pytest.raises(ValueError, match='outside'):
            model(**{**x, 'candidate_indices': torch.full_like(x['candidate_indices'], 9999)})


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_checkpoint_replay_and_lookup_tampering(tmp_path, variant):
    _, model, x = make(variant)
    model.eval()
    path = tmp_path / 'model.pt'
    provenance = {'statistics_policy': 'rolling_observed_v2', 'teacher_score_included': False}
    save_checkpoint(path, model, epoch=0, metrics={}, selection_criterion='test',
                    provenance=provenance, feature_contract_digest='schema')
    loaded, _ = load_checkpoint(path, expected_provenance=provenance, expected_feature_contract_digest='schema')
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**x).logits, loaded(**x).logits, rtol=0, atol=0)
    saved = torch.load(path, weights_only=False)
    saved['model_state_dict']['categorical_lookup'][0, 0] = 99
    torch.save(saved, path)
    with pytest.raises(ValueError, match='lookup/config'):
        load_checkpoint(path)
