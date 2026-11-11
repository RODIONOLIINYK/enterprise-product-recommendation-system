import copy
import pandas as pd
import pytest
import torch
from recommender.models import CustomerTower, ProductTower, TwoTowerModel
from recommender.experimental_retrieval import RetrievalConfig, RetrievalExperiment, weighted_pool, PurchaseEmbeddingCache, history_components
from recommender.experiment_retrieval_data import older_history


def build(config=RetrievalConfig()):
    torch.manual_seed(13)
    customer = CustomerTower(text_dim=16, hidden_dim=16, output_dim=16, num_cities=2, num_regions=2, num_industries=2,
                             num_tabular_features=6, catalog_history_encoder="attention", history_gap_reference_days=74.,
                             max_seq_len=32, num_heads=4, dim_feedforward=32, dropout=0.)
    product = ProductTower(text_dim=16, output_dim=16, num_brands=0, num_categories=0, num_business_lines=0, hidden_dim=16)
    return RetrievalExperiment(TwoTowerModel(customer, product, num_catalog_products=3), config).eval()


def inputs():
    mask = torch.zeros(2, 32, dtype=torch.long)
    mask[0, :3] = 1
    return dict(text_embedding=torch.randn(2, 16), city_id=torch.zeros(2, dtype=torch.long), region_id=torch.zeros(2, dtype=torch.long),
                industry_id=torch.zeros(2, dtype=torch.long), tabular_features=torch.rand(2, 6), sequence=torch.rand(2, 32, 9),
                attention_mask=mask, seq_product_ids=mask.clone(), seq_purchase_dates=torch.zeros(2, 32),
                older_ids=torch.tensor([[1, 2, 0], [0, 0, 0]]), older_counts=torch.tensor([[2., 1., 0.], [0., 0., 0.]]))


def test_unchanged_short_term_and_zero_extra_branch_initialization():
    model = build()
    x, v = inputs(), torch.randn(3, 16)
    expected = model.towers.customer_tower(**{k: v for k, v in x.items() if not k.startswith('older')}, catalog_product_vectors=v)
    torch.testing.assert_close(model.encode_purchase(x, v)[:, 0], expected)
    for mode in ('mean', 'attention'):
        extended = RetrievalExperiment(copy.deepcopy(model.towers), RetrievalConfig(long_memory=mode)).eval()
        torch.testing.assert_close(extended.encode_purchase(x, v)[:, 0], expected)
        extended.encode_purchase(x, v).sum().backward()
        assert extended.towers.customer_tower.late_fusion[0].weight.grad[:, -16:].abs().sum() > 0


@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('scoring', ['hard', 'smooth'])
def test_multi_interest_shapes_normalization_and_gradients(k, scoring):
    model = build(RetrievalConfig(interests=k, scoring=scoring))
    u = model.encode_purchase(inputs(), torch.randn(3, 16))
    assert u.shape == (2, k, 16) and torch.isfinite(u).all()
    torch.testing.assert_close(u.norm(dim=-1), torch.ones(2, k))
    score = model.score_cached(u, torch.randn(3, 16))
    assert score.shape == (2, 3)
    score.sum().backward()
    assert model.interest_queries.grad is not None
    assert torch.isfinite(model.interest_queries.grad).all()
    torch.testing.assert_close(model.encode_purchase(inputs(), torch.randn(3, 16))[1].norm(dim=-1), torch.ones(k))


def test_smooth_max_offset_and_one_interest_equivalence():
    hard = build(RetrievalConfig(interests=1))
    smooth = build(RetrievalConfig(interests=1, scoring='smooth'))
    u, v = torch.randn(2, 1, 16), torch.randn(3, 16)
    torch.testing.assert_close(hard.score_cached(u, v), smooth.score_cached(u, v))
    identical = u.expand(-1, 3, -1)
    torch.testing.assert_close(smooth.score_cached(identical, v), hard.score_cached(u, v))
    with pytest.raises(ValueError, match='independent'):
        RetrievalConfig(long_memory='attention', interests=2)


def test_attention_pooling_multiplicity_and_empty_history():
    values = torch.randn(1, 2, 16)
    q = torch.randn(3, 16)
    compressed = weighted_pool(values, torch.ones(1, 2), q, torch.tensor([[2., 1.]]))
    expanded = weighted_pool(values[:, [0, 0, 1]], torch.ones(1, 3), q)
    torch.testing.assert_close(compressed, expanded)
    assert weighted_pool(values, torch.zeros(1, 2), q).count_nonzero() == 0


def test_older_events_exclude_recent_32_and_prediction_basket():
    dates = pd.date_range('2025-01-01', periods=35)
    frame = pd.DataFrame({'customer_id': ['c'] * 35, 'product_id': ['A'] * 35, 'purchase_date': dates})
    data = {'customer_ids': ['c'], 'prediction_days': torch.tensor([34])}
    ids, counts = older_history(frame, data, ['A'])
    assert ids.tolist() == [[1]] and counts.tolist() == [[2.]]  # 34 prior events, 32 recent
    frame.loc[34, 'product_id'] = 'future'
    torch.testing.assert_close(older_history(frame, data, ['A'])[1], counts)


def test_purchase_cache_never_calls_tower_during_recommendation(monkeypatch):
    model = build(RetrievalConfig(interests=2))
    cache = PurchaseEmbeddingCache(model, 'm1', 'c1')
    x, v = inputs(), torch.randn(3, 16)
    cache.on_purchase('customer', 'purchase1', x, v)
    def forbidden(*args, **kwargs):
        raise AssertionError('Tower recomputed during request')
    monkeypatch.setattr(model, 'encode_purchase', forbidden)
    cache.on_purchase('customer', 'purchase1', x, v)
    first = cache.recommend('customer', v, model_version='m1', catalogue_version='c1')
    torch.testing.assert_close(first, cache.recommend('customer', v, model_version='m1', catalogue_version='c1'))
    with pytest.raises(ValueError, match='rebuild'):
        cache.recommend('customer', v, model_version='m2', catalogue_version='c1')
    other = PurchaseEmbeddingCache(model, 'm1', 'c1')
    other.load_state_dict(cache.state_dict())
    torch.testing.assert_close(other.recommend('customer', v, model_version='m1', catalogue_version='c1'), first)


def test_collapse_diagnostics_identifies_identical_interests():
    from recommender.experiment_retrieval import collapse_diagnostics
    u = torch.nn.functional.normalize(torch.randn(5, 1, 16), dim=-1).expand(-1, 3, -1)
    result = collapse_diagnostics(u, torch.randn(30, 16))
    assert result['fraction_pairs_above_0.99'] == 1.
    assert result['mean_top25_jaccard_first1000'] == 1.
    assert result['fraction_winner_margin_lt1e-5'] == 1.


def test_scratch_product_factory_and_demand_invariance():
    from recommender.experiment_retrieval import make_model
    info = {"product_ids": ["a", "b", "c"], "product_inputs": {"demand_features": torch.zeros(3, 18)}}
    model = make_model(RetrievalConfig(), info).eval()
    x = {"text_embedding": torch.randn(3, 1024), "brand_id": torch.zeros(3, dtype=torch.long),
         "category_id": torch.zeros(3, dtype=torch.long), "business_line_id": torch.zeros(3, dtype=torch.long),
         "demand_features": torch.rand(3, 18)}
    expected = model.product_vectors(x)
    assert expected.shape == (3, 128)
    changed = dict(x)
    changed["demand_features"] = x["demand_features"].clone()
    changed["demand_features"][:, :6] = torch.randn(3, 6) * 1000
    torch.testing.assert_close(model.product_vectors(changed), expected)
    model2 = make_model(RetrievalConfig(), info)
    assert not torch.equal(model.towers.customer_tower.text_proj[0].weight, model2.towers.customer_tower.text_proj[0].weight)


@pytest.mark.parametrize('config', [RetrievalConfig(), RetrievalConfig(long_memory='mean'),
                                    RetrievalConfig(long_memory='attention'), RetrievalConfig(interests=3)])
def test_customer_vectors_do_not_depend_on_request_clock(config):
    model = build(config)
    x, vectors = inputs(), torch.randn(3, 16)
    expected = model.encode_purchase(x, vectors)
    later = {k:v.clone() for k,v in x.items()}
    # Absolute ages/month/habit channels are not Stage 1 tower inputs. Gap
    # normalization is relative to latest purchase, not the recommendation date.
    later['sequence'][..., 1:] += 100
    later['seq_purchase_dates'] += 10.
    torch.testing.assert_close(model.encode_purchase(later, vectors), expected)


def test_retrieval_metrics_keep_missing_and_ineligible_targets_in_denominator():
    from recommender.experiment_retrieval import metrics_from_scores
    target = torch.zeros(2, 60, dtype=torch.bool)
    target[0, 0] = target[1, 1] = True
    allowed = torch.ones(2, 60, dtype=torch.bool)
    allowed[1, 1] = False
    logits = torch.zeros(2, 60)
    logits[0, 0] = logits[1, 1] = 100
    data = {'target': target, 'new_target': target.clone(), 'allowed': allowed, 'unseen': allowed}
    source = {'positive_pids': [['x', 'outside'], ['blocked']], 'new_positive_pids': [['x', 'outside'], ['blocked']]}
    metrics = metrics_from_scores(logits, data, source)
    assert metrics['new_targets'] == 3
    assert metrics['new_hits@10'] == metrics['new_hits@25'] == metrics['new_hits@50'] == 1
    assert metrics['val_recall@10'] == .25
    assert metrics['stage1_new_recall@25'] == 1/3


def test_single_vector_score_retains_temperature_bias_and_familiarity():
    model = build()
    model.towers.familiarity_weights = torch.nn.Parameter(torch.tensor([.2, -.3]))
    with torch.no_grad():
        model.towers.product_bias.copy_(torch.tensor([.1, -.2, .3]))
    u = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    flags = torch.rand(2, 3, 2)
    expected = model.towers.compute_logits(u, v, torch.arange(3), familiarity_features=flags)
    torch.testing.assert_close(model.score_cached(u[:, None], v, torch.arange(3), flags), expected)


def test_long_memory_is_persisted_in_purchase_cache():
    model = build(RetrievalConfig(long_memory='attention'))
    cache = PurchaseEmbeddingCache(model, 'model', 'catalogue')
    cache.on_purchase('customer', 'purchase', inputs(), torch.randn(3, 16))
    state = cache.state_dict()
    assert state['entries']['customer'][2].shape == (2, 16)
    assert state['entries']['customer'][2][1].count_nonzero() == 0
    restored = PurchaseEmbeddingCache(model, 'model', 'catalogue')
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.entries['customer'][2], cache.entries['customer'][2])


def test_global_catalogue_refresh_remaps_bias_without_mutating_old_model():
    from recommender.experiment_serving import refresh_catalogue
    model=build()
    with torch.no_grad(): model.towers.product_bias.copy_(torch.tensor([1.,2.,3.]))
    refreshed,cache=refresh_catalogue(model,['a','b','c'],['c','new','a'],'model','catalogue2')
    torch.testing.assert_close(refreshed.towers.product_bias,torch.tensor([3.,0.,1.]))
    torch.testing.assert_close(model.towers.product_bias,torch.tensor([1.,2.,3.]))
    assert cache.version==('model','catalogue2') and not cache.entries


def test_long_attention_key_query_value_receive_gradients_when_route_is_active():
    model=build(RetrievalConfig(long_memory='attention'))
    with torch.no_grad():
        model.towers.customer_tower.late_fusion[0].weight[:, -16:].normal_(0, .1)
    encoded=model.encode_purchase(inputs(),torch.randn(3,16))
    loss=model.score_cached(encoded,torch.randn(3,16)).square().sum()
    loss.backward()
    for parameter in (model.long_key.weight,model.long_query,model.long_value.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm()>0
