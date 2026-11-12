"""Synthetic acceptance checks; never read private campaign artifacts."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from recommender.ranker import UPSCANRRanker
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import NAMES, contract
from recommender.experiments.dynamic_selector.model import PointwiseSelector, select_candidates
from recommender.experiments.candidate_context.model import CandidateContextRanker


def tables():
    purchases = pd.DataFrame({'customer_id': ['a','a','b','b','a','b'], 'product_id': ['x','y','x','y','x','z'],
        'quantity': [1.,2.,1.,3.,2.,1.], 'purchase_date': pd.to_datetime(['2025-01-01','2025-01-10','2025-01-02','2025-01-12','2025-10-05','2025-10-07'])})
    catalogue = pd.DataFrame({'product_id': ['x','y','z'], 'product_category': ['c','c','d'],
        'product_subcategory':['s','s','t'], 'product_family':['f','f','g'], 'brand':['b','b','q'], 'business_line':['l','l','m']})
    companies = pd.DataFrame({'customer_id':['a','b'], 'industry_branch':['i','i']})
    return purchases, catalogue, companies


def test_future_perturbation_and_frozen_global_statistics():
    purchases, catalogue, companies = tables()
    extra = purchases.iloc[[-1]].copy(); extra['purchase_date'] = pd.Timestamp('2025-09-15')
    changed = pd.concat([purchases, extra, extra], ignore_index=True)
    before = DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'])
    after = DynamicFeatureBuilder(changed,catalogue,companies,['x','y','z'])
    np.testing.assert_array_equal(before.build('a','2025-01-11'),after.build('a','2025-01-11'))
    a = before.build('a','2025-10-06'); b = before.build('a','2025-10-10')
    # Frozen global and relationship counts, customer purchase state unchanged.
    np.testing.assert_array_equal(a[:,24:31],b[:,24:31])
    np.testing.assert_array_equal(a[:,33:],b[:,33:])
    assert not np.array_equal(a[:,13:15],b[:,13:15])
    assert np.isfinite(a).all() and a.shape == (3,len(NAMES))
    with pytest.raises(ValueError): before.build('a','2026-02-19')
    with pytest.raises(ValueError): before.build('a','2025-01-11')


def test_feature_contract_missingness_and_strict_prior_boundary():
    purchases,catalogue,companies=tables()
    builder=DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'])
    x=builder.build('a','2025-01-10')
    assert x[1,0]==0 and x[0,0]==1  # Today's y purchase is still unknown.
    assert x[2,NAMES.index('family_never_seen')]==1
    assert builder.store.compute_pairwise_features('a','2025-01-10','y')[0]==0
    assert len(contract()['names'])==len(set(NAMES))


@pytest.mark.parametrize('policy',['quota50_25','quota50_10','top50','quota100_50'])
def test_candidate_selection_exact_eligibility_quota_and_ties(policy):
    scores=torch.ones(2,120);allowed=torch.ones(2,120,dtype=torch.bool);allowed[:,:10]=False
    unseen=allowed.clone();unseen[:,10:40]=False
    indices=select_candidates(scores,allowed,unseen,policy)
    assert all(len(set(row))==len(row) for row in indices.tolist())
    assert bool(allowed.gather(1,indices).all())
    assert torch.equal(indices,select_candidates(scores,allowed,unseen,policy))
    if policy=='quota50_25': assert int((~unseen.gather(1,indices))[0].sum())==25
    assert indices.shape[1]==(100 if policy=='quota100_50' else 50)
    with pytest.raises(ValueError): select_candidates(scores,torch.zeros_like(allowed),unseen,policy)


def inputs():
    return dict(text_embedding=torch.randn(2,16),city_id=torch.zeros(2,dtype=torch.long),region_id=torch.zeros(2,dtype=torch.long),
        industry_id=torch.zeros(2,dtype=torch.long),tabular_features=torch.rand(2,8),sequence=torch.rand(2,4,9),
        attention_mask=torch.ones(2,4,dtype=torch.long),candidate_vectors=torch.randn(2,5,16),
        seq_purchase_dates=torch.zeros(2,4),prediction_day=torch.full((2,),30.),
        pairwise_features=torch.rand(2,5,len(NAMES)),cached_customer_vectors=torch.randn(2,16))


@pytest.mark.parametrize('mode',CandidateContextRanker.MODES)
def test_context_heads_permutation_scalar_gradient_and_padding(mode):
    base=UPSCANRRanker(text_dim=16,hidden_dim=16,statistics_attention=True,dropout=0.)
    model=CandidateContextRanker(base,mode).eval();x=inputs()
    output=model(**x)
    assert output.logits.shape==(2,5) and torch.isfinite(output.logits).all()
    permutation=torch.tensor([3,1,4,0,2]);permuted=dict(x)
    for name in ('candidate_vectors','pairwise_features'): permuted[name]=x[name][:,permutation]
    torch.testing.assert_close(output.logits[:,permutation],model(**permuted).logits,atol=2e-6,rtol=1e-5)
    if model.uses_context:
        mask=torch.tensor([[True,True,True,False,False]]*2)
        assert torch.isneginf(model(candidate_mask=mask,**x).logits[:,3:]).all()
        output.logits.sum().backward()
        # Zero-initialized residual intentionally has zero initial attention gradient.
        grad=model.context.self_attn.in_proj_weight.grad
        assert grad is not None
        if mode!='context_residual': assert grad.abs().sum()>0


def test_control_and_residual_exact_initialization_and_repeat_group_invariance():
    base=UPSCANRRanker(text_dim=16,hidden_dim=16,statistics_attention=True,dropout=0.).eval()
    x=inputs();plain=dict(x);plain.pop('cached_customer_vectors');plain['pairwise_features']=x['pairwise_features'][...,:13]
    original=base(**plain).logits
    for mode in ('control','context_residual'):
        torch.testing.assert_close(CandidateContextRanker(copy.deepcopy(base),mode).eval()(**x).logits,original,atol=0,rtol=0)
    model=CandidateContextRanker(copy.deepcopy(base),'no_repeat_group').eval()
    changed=dict(x);changed['pairwise_features']=x['pairwise_features'].clone();changed['pairwise_features'][...,:5]=99
    torch.testing.assert_close(model(**x).logits,model(**changed).logits,atol=0,rtol=0)


def test_residual_attention_learns_after_head_update_and_value_routing_is_explicit():
    base=UPSCANRRanker(text_dim=16,hidden_dim=16,statistics_attention=True,dropout=0.).eval()
    x=inputs()
    residual=CandidateContextRanker(copy.deepcopy(base),'context_residual').eval()
    optimizer=torch.optim.SGD(residual.parameters(),lr=.01)
    residual(**x).logits.square().sum().backward();optimizer.step();optimizer.zero_grad()
    residual(**x).logits.square().sum().backward()
    assert residual.context.self_attn.in_proj_weight.grad.abs().sum()>0
    before=CandidateContextRanker(copy.deepcopy(base),'control').eval()(**x).memory
    both=CandidateContextRanker(copy.deepcopy(base),'bias_and_values').eval()(**x).memory
    assert not torch.equal(before,both)


def test_selector_requires_no_tower_and_masks_dynamic_features_in_static_control():
    u,v=torch.randn(2,16),torch.randn(7,16);features=torch.rand(2,7,len(NAMES));dot=u@v.T
    model=PointwiseSelector('static',16)
    altered=features.clone();altered[...,13:23]=100
    torch.testing.assert_close(model(u,v,features,dot),model(u,v,altered,dot),atol=0,rtol=0)
    assert not any('tower' in name for name,_ in model.named_modules())
    model(u,v,features,dot).sum().backward()
    assert model.head[0].weight.grad.abs().sum()>0


def test_metrics_keep_unreachable_targets_as_misses_and_pair_identical_rows(tmp_path):
    from recommender.experiments.dynamic_selector.evaluate import ranking_metrics, paired_intervals
    allowed=torch.ones(2,120,dtype=torch.bool);allowed[:,119]=False
    targets=torch.zeros_like(allowed);targets[0,0]=True;targets[0,119]=True;targets[1,119]=True
    data={'allowed':allowed,'unseen':allowed,'target':targets,'new_target':targets,
          'source':{'positive_pids':[['a','ineligible','absent'],['ineligible']],
                    'new_positive_pids':[['a','ineligible','absent'],['ineligible']],
                    'customer_ids':['c1','c2'],'attention_mask':torch.ones(2,32),'prediction_days':torch.tensor([1.,32.])}}
    data['features']=np.zeros((2,120,len(NAMES)),dtype=np.float16)
    data['family_known']=torch.ones(120,dtype=torch.bool);data['family_known'][0]=False
    data['features'][0,0,22]=1
    scores=torch.zeros(2,120);scores[:,0]=1
    result=ranking_metrics(scores,data,save=tmp_path/'a.npz')
    assert result['new_targets']==4 and result['new_hits@10']==1
    assert result['new_targets_in_slate']==1 and result['conditional_new_recall@10']==1
    assert result['mrr']==.5 and result['recall@10']==pytest.approx(1/6)
    assert result['new_target_cohorts']['unknown_snapshot_family_metadata']=={'targets':1,'hits@10':1}
    assert result['new_target_cohorts']['first_snapshot_family']['targets']==0
    assert result['validation_months']['2025-01']['new_targets']==3
    ci=paired_intervals(tmp_path/'a.npz',tmp_path/'a.npz',repeats=20)
    assert all(values==[0.,0.,0.] for values in ci.values())


def test_serving_towers_change_only_on_purchase_and_versioned_rebuild():
    from recommender.experiments.dynamic_selector.serving import DynamicSelectorService
    class Upstream(torch.nn.Module):
        def __init__(self): super().__init__();self.calls=0
        def encode_purchase(self,inputs,vectors,return_memory=False):
            self.calls+=1
            return inputs['u'],None
        def score_cached(self,u,v,product_indices,familiarity): return (u@v.T).squeeze(1)
    model=Upstream();selector=PointwiseSelector('dynamic',16);v=torch.randn(120,16)
    service=DynamicSelectorService(selector,model,v,[str(i) for i in range(120)],'m1','c1')
    u=torch.randn(1,1,16)
    service.on_purchase('a','purchase1',{'u':u});service.on_purchase('a','purchase1',{'u':u})
    def build(cid,date):
        x=np.zeros((120,len(NAMES)),dtype=np.float32);x[:,13]=float(date);return x
    versions={'model_version':'m1','catalogue_version':'c1'}
    first=service.score('a',1.,build,None,**versions)
    second=service.score('a',2.,build,None,**versions)
    assert not torch.equal(first,second) and model.calls==1
    torch.testing.assert_close(service.cache.entries['a'][1],u,atol=0,rtol=0)
    service.on_purchase('a','purchase2',{'u':u+1});assert model.calls==2
    with pytest.raises(ValueError): service.score('a',2.,build,None,model_version='m2',catalogue_version='c1')
    rebuilt=service.rebuilt(selector,model,v,[str(i) for i in range(120)],'m2','c2')
    assert not rebuilt.cache.entries and service.cache.entries
    with pytest.raises(KeyError): rebuilt.score('a',2.,build,None,model_version='m2',catalogue_version='c2')


def test_unqualified_experiment_has_diagnostic_weights_without_selected_alias(tmp_path):
    from recommender.artifacts import resolve_run_checkpoint
    from recommender.experiments.dynamic_selector.train import run_directory,finish_run,evaluated_checkpoint
    settings={'seed':42,'mode':'synthetic'}
    directory,_=run_directory(tmp_path,'failed-guard',settings)
    (directory/'best.pt').write_bytes(b'synthetic weights')
    result={'qualified':False,'selected_checkpoint':str(directory/'best.pt')}
    finish_run(directory,settings,result)
    assert result['selected_checkpoint'] is None and not (directory/'best.pt').exists()
    assert evaluated_checkpoint(result).name=='best_unqualified.pt'
    with pytest.raises(ValueError,match='no qualifying'):resolve_run_checkpoint(directory)
    _,reused=run_directory(tmp_path,'failed-guard',settings)
    assert reused['qualified'] is False


def test_request_quantization_matches_training_cache_boundary():
    from recommender.experiments.dynamic_selector.feature_schema import quantize_features
    values=np.random.default_rng(17).random((3,len(NAMES)),dtype=np.float32)
    cached=values.astype(np.float16)
    np.testing.assert_array_equal(cached.astype(np.float32),quantize_features(values))


def test_cached_pointwise_control_is_scalar_equivariant_and_trains():
    from recommender.experiments.candidate_context.model import CachedPointwiseRanker
    model=CachedPointwiseRanker(16).eval();x=inputs()
    scores=model(**x).logits
    assert scores.shape==(2,5)
    permutation=torch.tensor([4,0,3,1,2]);other=dict(x)
    for name in ('candidate_vectors','pairwise_features'):other[name]=x[name][:,permutation]
    torch.testing.assert_close(scores[:,permutation],model(**other).logits,atol=2e-6,rtol=1e-5)
    scores.sum().backward();assert model.head[0].weight.grad.abs().sum()>0


def test_attention_ablation_retains_pointwise_feedforward_without_slate_mixing():
    torch.manual_seed(21)
    base=UPSCANRRanker(text_dim=16,hidden_dim=16,statistics_attention=True,dropout=0.)
    model=CandidateContextRanker(base,'context_direct').eval();x=inputs();subset=dict(x)
    for name in ('candidate_vectors','pairwise_features'):subset[name]=x[name][:,:3]
    with torch.no_grad():
        full=model(**x,disable_candidate_attention=True).logits[:,:3]
        reduced=model(**subset,disable_candidate_attention=True).logits
        torch.testing.assert_close(full,reduced,atol=2e-6,rtol=1e-5)
        assert not torch.allclose(model(**x).logits[:,:3],model(**subset).logits,atol=1e-6,rtol=0)


def test_dynamic_selector_can_omit_independent_relationship_tables_exactly():
    purchases,catalogue,companies=tables()
    full=DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'])
    simple=DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'],relationships=False)
    a=full.build('a','2025-10-06');b=simple.build('a','2025-10-06')
    np.testing.assert_array_equal(a[:,:33],b[:,:33])
    assert not b[:,33:].any() and not simple.tables and simple.complements is None
    with pytest.raises(ValueError):simple.build('a','2025-10-06',include_relationships=True)


def test_feature_subset_preserves_all_history_and_candidate_order():
    purchases,catalogue,companies=tables()
    full=DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'])
    subset=DynamicFeatureBuilder(purchases,catalogue,companies,['x','y','z'])
    a=full.build('a','2025-10-06')
    b=subset.build('a','2025-10-06',candidate_ids=['z','x'])
    np.testing.assert_array_equal(a[[2,0]],b)
    with pytest.raises(KeyError):subset.build('a','2025-10-06',candidate_ids=['unknown'])
    with pytest.raises(ValueError):subset.build('a','2025-10-06',candidate_ids=['x','x'])
