"""Do not change input signals when testing stochastic/optimizer regularization."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.experiments.regularization_signal.model import (
    VARIANTS, RegularizedRanker, active_dropout_inventory, load_checkpoint, save_checkpoint,
)
from recommender.experiments.regularization_signal.tabular import (
    StatisticalMLP, categorical_lookup, numeric_pairs, weighted_binary_loss,
)
from recommender.experiments.catboost_comparison.standalone import statistical_frame, sample_training_pairs
from test_ranker_capacity import fixture


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_regularization_preserves_init_and_replays_actual_dropout(tmp_path, variant):
    source, x = fixture()
    x['candidate_indices'] = torch.tensor([[0, 1, 2, -999], [2, 1, 3, 0]])
    model = RegularizedRanker.from_control(source, reg_variant=variant,
        categorical_lookup=[[0, 0, 0], [1, 1, 0], [2, 0, 1], [3, 1, 1]], product_ids=['a', 'b', 'c', 'd'])
    source.eval(); model.eval()
    with torch.no_grad():
        torch.testing.assert_close(source(**x).logits, model(**x).logits, rtol=0, atol=2e-6)
    if variant in ('all_zero', 'direct52_all_zero'):
        assert all(rate == 0 for rate in active_dropout_inventory(model).values())
    path = tmp_path/'ranker.pt'
    prov = {'statistics_policy':'rolling_observed_v2', 'teacher_score_included':False}
    save_checkpoint(path, model, epoch=1, metrics={}, selection_criterion='test',
                    provenance=prov, feature_contract_digest='fixed')
    loaded, _ = load_checkpoint(path, expected_provenance=prov, expected_feature_contract_digest='fixed')
    loaded.eval()
    assert active_dropout_inventory(loaded) == active_dropout_inventory(model)
    with torch.no_grad():
        torch.testing.assert_close(model(**x).logits, loaded(**x).logits, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    first = next(iter(payload['model_config']['active_dropout_inventory']))
    payload['model_config']['active_dropout_inventory'][first] = .99
    torch.save(payload,path)
    with pytest.raises(ValueError, match='dropout'):
        load_checkpoint(path)


def test_numeric_fields_and_categories_match_frozen_tree_contract():
    catalogue = pd.DataFrame({'product_id':['a','b','c'], 'product_category':['C','D','C'],
                              'business_line':['L','M','L']})
    split = {'features':np.arange(2*3*41,dtype=np.float32).reshape(2,3,41),
             'profile':np.arange(16,dtype=np.float32).reshape(2,8)}
    rows, cols = np.array([1,0,1]), np.array([2,1,0])
    frame = statistical_frame(split['features'],split['profile'],['a','b','c'],catalogue,rows,cols)
    np.testing.assert_array_equal(numeric_pairs(split,rows,cols),frame.iloc[:,3:].to_numpy())
    lookup,vocab = categorical_lookup(['a','b','c'],catalogue)
    for i,name in enumerate(('product_id','product_category','business_line')):
        assert [vocab[name][int(code)] for code in lookup[cols,i]] == frame[name].tolist()


def test_importance_normalizer_is_full_objective_and_negatives_can_rotate():
    logits=torch.tensor([1.,-2.,0.,3.]); y=torch.tensor([1.,0.,0.,1.]); w=torch.tensor([1.,4.,4.,1.])
    whole=weighted_binary_loss(logits,y,w,global_mean_weight=float(w.mean()))
    halves=sum(weighted_binary_loss(logits[s],y[s],w[s],global_mean_weight=float(w.mean()))
               for s in (slice(0,2),slice(2,4)))/2
    torch.testing.assert_close(whole,halves)
    allowed=np.ones((3,40),dtype=bool);allowed[:,0]=False
    targets=np.zeros_like(allowed);targets[:,2]=True
    r1,c1,w1=sample_training_pairs(allowed,targets,negatives=4,seed=42)
    r2,c2,w2=sample_training_pairs(allowed,targets,negatives=4,seed=43)
    assert not np.array_equal(c1,c2)
    for rows,cols,weights in ((r1,c1,w1),(r2,c2,w2)):
        assert allowed[rows,cols].all() and targets[rows,cols].sum()==3
        np.testing.assert_allclose(weights.sum(),allowed.sum(),rtol=1e-6)


def test_statistical_mlp_uses_features_has_no_towers_and_dropout_eval_parity():
    torch.manual_seed(42)
    model=StatisticalMLP([4,2,2]); no_dropout=copy.deepcopy(model)
    for m in no_dropout.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0
    x=torch.rand(8,49,requires_grad=True);ids=torch.zeros(8,3,dtype=torch.long)
    model.eval();no_dropout.eval()
    torch.testing.assert_close(model(x,ids),no_dropout(x,ids),rtol=0,atol=0)
    model(x,ids).sum().backward();assert x.grad.abs().sum()>0
    assert len(model.contract()['raw_feature_names'])==52
    with pytest.raises(ValueError,match='numeric49'):
        model(torch.rand(8,48),ids)


def test_standalone_mlp_checkpoint_contract_and_replay(tmp_path):
    from recommender.experiments.regularization_signal.train_tabular import (
        array_digest, load_checkpoint as load_tabular, save_checkpoint as save_tabular,
    )
    from recommender.experiments.catboost_comparison.standalone import FEATURE_NAMES
    lookup=np.array([[0,0,0],[1,1,1]],dtype=np.int64)
    prov={'feature_names':list(FEATURE_NAMES),'statistics_policy':'rolling_observed_v2',
          'lookup_digest':array_digest(lookup),'heldout_access':False}
    model=StatisticalMLP([2,2,2]).eval();x=torch.rand(6,49);ids=torch.zeros(6,3,dtype=torch.long)
    path=tmp_path/'tabular.pt'
    save_tabular(path,model,epoch=1,provenance=prov,lookup=lookup,metrics={})
    loaded,_,_=load_tabular(path,expected_provenance=prov);loaded.eval()
    with torch.no_grad():torch.testing.assert_close(model(x,ids),loaded(x,ids),rtol=0,atol=0)
    payload=torch.load(path,weights_only=False);payload['lookup'][0,0]=1;torch.save(payload,path)
    with pytest.raises(ValueError,match='input contract'):load_tabular(path)
