"""Synthetic correctness regressions for the follow-up audit; no local artifacts."""
import copy
import json
from pathlib import Path
import types
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.artifacts import (create_run_directory, payload_digest,
    ranker_provenance, resolve_run_checkpoint, seal_slates, validate_ranker_provenance, validate_slates)
from recommender.candidates import PointInTimeFeatureStore
from recommender.catalogue import RollingCandidatePool
from recommender.development import load_development_purchases
from recommender.experimental_signals import SignalRanker, TransitionStatistics
from recommender.features import Log1pScaler
from recommender.models import ProductTower
from recommender.preparation import export_purchases
from recommender.ranker import RankerFeatures, UPSCANRRanker
from recommender.train_ranker import build_slate_cache, load_slate_cache, prediction_day_offsets


def slates(split='train', c=3):
    date = pd.Timestamp('2025-09-01' if split == 'train' else '2025-10-02')
    products = [f's{i}' for i in range(c)]
    transitions = [{'customer_id':f'c{i}', 'as_of_date':date + pd.Timedelta(days=i),
                    'positive_product_ids':[products[i % c]]} for i in range(2)]
    days = prediction_day_offsets(transitions)
    mask = torch.zeros(2, 32, dtype=torch.long); mask[:, 0] = 1
    sequence = torch.zeros(2, 32, 9); sequence[..., 8] = mask
    dates = torch.zeros(2, 32); dates[:, 0] = (days-1)/365.25
    data = {'text_embedding':torch.zeros(2, 1024), 'tabular_features':torch.zeros(2,8),
            'retriever_tabular_features':torch.zeros(2,6), 'sequence':sequence,'attention_mask':mask,
            'prediction_days':days, 'seq_purchase_dates':dates,
            'candidate_indices':torch.arange(c).expand(2,-1).clone(), 'pairwise_features':torch.zeros(2,c,13),
            'labels':torch.tensor([[float(j==i%c) for j in range(c)] for i in range(2)]),
            'customer_ids':['c0','c1'], 'candidate_pids':[list(products) for _ in range(2)],
            'positive_pids':[row['positive_product_ids'] for row in transitions], 'cache_identity':f'input:{split}'}
    for name in ('city_id','region_id','industry_id'): data[name]=torch.zeros(2,dtype=torch.long)
    for name in ('seq_product_ids','seq_category_ids','seq_subcategory_ids','seq_business_line_ids'): data[name]=mask.clone()
    seal_slates(data, transitions, products)
    return data, transitions, products


def test_cache_roundtrip_and_identity_only_payload_is_rejected(tmp_path):
    data, transitions, products = slates()
    path=tmp_path/'cache.pt';torch.save(data,path)
    loaded=load_slate_cache(path,'input:train',transitions=transitions,product_ids=products,split='train',candidates=3)
    assert loaded is not None and loaded['cache_payload_digest']==data['cache_payload_digest']
    torch.save({'cache_identity':'input:train'},path)
    assert load_slate_cache(path,'input:train') is None


@pytest.mark.parametrize('corruption', ['shape','nan','negative_index','upper_index','candidate_ids','duplicate_candidates',
    'label','future_history','invalid_mask','indicator','feature_order','date_only_permutation','checksum','split','dtype','float64','flag'])
def test_slate_semantic_validation_rejects_corruption(corruption):
    data, transitions, products = slates()
    if corruption=='shape': data['sequence']=data['sequence'][:,:31]
    elif corruption=='nan': data['tabular_features'][0,0]=float('nan')
    elif corruption=='negative_index': data['candidate_indices'][0,0]=-1
    elif corruption=='upper_index': data['candidate_indices'][0,0]=3
    elif corruption=='candidate_ids': data['candidate_pids'][0].reverse()
    elif corruption=='duplicate_candidates': data['candidate_pids'][0][1]=data['candidate_pids'][0][0]
    elif corruption=='label': data['labels'][0,0]=0
    elif corruption=='future_history': data['seq_purchase_dates'][0,0]=data['prediction_days'][0]/365.25
    elif corruption=='invalid_mask': data['attention_mask'][0,1]=2
    elif corruption=='indicator': data['sequence'][0,0,8]=0
    elif corruption=='feature_order': data['input_contract']['pairwise'].reverse()
    elif corruption=='date_only_permutation':
        data['prediction_days']=data['prediction_days'].flip(0)
        data['row_keys']=[(cid,int(day),tuple(sorted(p))) for cid,day,p in zip(data['customer_ids'],data['prediction_days'],data['positive_pids'])]
        data['seq_purchase_dates'][:,0]=(data['prediction_days']-1)/365.25
    elif corruption=='split': data['prediction_days']+=400
    elif corruption=='dtype': data['text_embedding']=data['text_embedding'].long()
    elif corruption=='float64': data['text_embedding']=data['text_embedding'].double()
    elif corruption=='flag': data['pairwise_features'][0,0,0]=.5
    elif corruption=='checksum': data['text_embedding'][0,0]=.1
    if corruption!='checksum': data['cache_payload_digest']=payload_digest(data)
    with pytest.raises(ValueError):
        validate_slates(data,transitions=transitions,product_ids=products,split='train',candidates=3)


def test_new_cache_never_overwrites_existing_payload(tmp_path,monkeypatch):
    import recommender.train_ranker as training
    data, transitions, products=slates()
    path=tmp_path/'cache.pt';old=b'old historical cache';path.write_bytes(old)
    monkeypatch.setattr(training,'precompute_candidate_slates_for_transitions',lambda **kw:copy.deepcopy(data))
    retriever=types.SimpleNamespace(eligible_product_ids=products,feature_store=types.SimpleNamespace(metadata_policy='snapshot_unverified'))
    built=build_slate_cache(path,'input:train',transitions,{'retriever':retriever,'top_c':3})
    assert path.read_bytes()==old and Path(built['cache_path'])!=path
    assert load_slate_cache(path,'input:train',transitions=transitions,product_ids=products,split='train') is not None


def test_checkpoint_contract_provenance_and_legacy_modes():
    train,_,_=slates();val,_,_=slates('val')
    config={'hidden_dim':128,'pairwise_metadata_policy':'snapshot_unverified'}
    checkpoint={'retriever_digest':'retriever','slate_schema_version':3,'config':config,'selection_metric':'new_recall@10'}
    assert validate_ranker_provenance(checkpoint,retriever_digest='retriever').startswith('legacy')
    with pytest.raises(ValueError): validate_ranker_provenance(checkpoint,retriever_digest='retriever',allow_legacy=False)
    checkpoint['checkpoint_format_version']=4
    with pytest.raises(ValueError,match='required provenance'):
        validate_ranker_provenance(checkpoint,retriever_digest='retriever')
    checkpoint['provenance']=ranker_provenance(retriever_digest='retriever',train_data=train,val_data=val,
                                             model_config=config,selection_metric='new_recall@10')
    assert validate_ranker_provenance(checkpoint,retriever_digest='retriever',slate_identity='input:val',
        payload_digests={'val':val['cache_payload_digest']},allow_legacy=False)=='verified_named_contract'
    for change in ('order','config','origin','retriever','payload'):
        broken=copy.deepcopy(checkpoint)
        if change=='order': broken['provenance']['input_contract']['history'].reverse()
        elif change=='config': broken['config']['hidden_dim']=256
        elif change=='origin': broken['provenance']['slates']['val']['identity']='different'
        elif change=='retriever': broken['retriever_digest']='different'
        else: broken['provenance']['slates']['val']['payload_digest']='0'*64
        with pytest.raises(ValueError): validate_ranker_provenance(broken,retriever_digest='retriever',slate_identity='input:val',payload_digests={'val':val['cache_payload_digest']})


def test_run_directories_are_exclusive_and_never_resolve_stale_alias(tmp_path):
    (tmp_path/'best_ranker.pt').write_bytes(b'old alias')
    run=create_run_directory(tmp_path,'run1')
    with pytest.raises(FileExistsError): create_run_directory(tmp_path,'run1')
    with pytest.raises(ValueError): create_run_directory(tmp_path,'../outside')
    with pytest.raises(ValueError,match='no qualifying'): resolve_run_checkpoint(run)
    assert (tmp_path/'best_ranker.pt').read_bytes()==b'old alias'


@pytest.mark.parametrize('qualifies',[True,False])
def test_ranker_run_guardrail_manifest_and_full_provenance(tmp_path,monkeypatch,qualifies):
    import recommender.train_ranker as training
    train,_,products=slates(c=50); val,_,_=slates('val',c=50)
    train['new_positive_pids']=train['positive_pids'];val['new_positive_pids']=val['positive_pids']
    retriever=tmp_path/'retriever.pt';torch.save({'config':{}},retriever)
    root=tmp_path/'ranker';root.mkdir();old=root/'best_ranker.pt';old.write_bytes(b'preserve me')
    monkeypatch.setattr(training,'load_development_purchases',lambda path:pd.DataFrame())
    monkeypatch.setattr(training,'load_product_catalogue',lambda *a:pd.DataFrame())
    monkeypatch.setattr(training,'load_companies',lambda **kw:pd.DataFrame())
    monkeypatch.setattr(training,'load_or_precompute_slates',lambda **kw:(train,val,torch.randn(50,16),products,{}))
    def factory(cls,*args,**kwargs):
        return cls(text_dim=1024,hidden_dim=16,num_cities=2,num_regions=2,num_industries=2,num_products=51,
                   num_categories=2,num_subcategories=2,num_business_lines=2,statistics_attention=True,
                   excluded_statistics=('history_repeat',),cached_only=kwargs['cached_only'])
    monkeypatch.setattr(training.UPSCANRRanker,'from_retriever_checkpoint',classmethod(factory))
    metrics={'val_loss':1.,'val_recall@5':.5,'val_recall@10':.8 if qualifies else .1,'val_recall@20':.9,
             'val_ndcg@10':.5,'val_mrr':.5,'new_recall@10':.5,'new_hits@10':1,'new_target_count':2,
             'avg_token0_weight':.25,'avg_history_weight':.75}
    monkeypatch.setattr(training,'evaluate_ranker',lambda **kwargs:metrics)
    args=dict(epochs=1,batch_size=2,device='cpu',checkpoint_dir=root,retriever_checkpoint_path=retriever,
              run_id='new',discovery_min_overall_recall=.75)
    if qualifies:
        result=training.train_ranker(**args)
        checkpoint=torch.load(result['best_checkpoint_path'],map_location='cpu')
        assert validate_ranker_provenance(checkpoint,retriever_digest=training.checkpoint_digest(retriever),slate_identity='input:val')=='verified_named_contract'
        assert resolve_run_checkpoint(root/'runs'/'new')==Path(result['best_checkpoint_path'])
    else:
        with pytest.raises(RuntimeError,match='No discovery'): training.train_ranker(**args)
        assert not (root/'runs'/'new'/'best_ranker.pt').exists()
        manifest=json.loads((root/'runs'/'new'/'run.json').read_text())
        assert manifest['status']=='no_qualifying_checkpoint' and manifest['selected_checkpoint'] is None
    assert old.read_bytes()==b'preserve me'


def model_inputs():
    return dict(text_embedding=torch.randn(2,16),city_id=torch.zeros(2,dtype=torch.long),
        region_id=torch.zeros(2,dtype=torch.long),industry_id=torch.zeros(2,dtype=torch.long),
        tabular_features=torch.rand(2,8),sequence=torch.rand(2,4,9),
        attention_mask=torch.tensor([[1,1,0,0],[0,0,0,0]]),candidate_vectors=torch.randn(2,3,16),
        pairwise_features=torch.zeros(2,3,13),seq_purchase_dates=torch.zeros(2,4),prediction_day=torch.full((2,),100.))


def test_batch_inputs_use_only_required_fields_and_preserve_indexed_scores():
    from recommender.train_ranker import ranker_batch_inputs
    model=UPSCANRRanker(text_dim=16,hidden_dim=16,dropout=0.).eval()
    batch=model_inputs()
    # These have no .to(): a redundant transfer fails immediately.
    batch['candidate_indices']=object()
    batch['retriever_tabular_features']=object()
    direct=ranker_batch_inputs(batch,None,model,'cpu')
    assert 'retriever_tabular_features' not in direct and 'candidate_indices' not in direct
    catalogue=batch['candidate_vectors'].reshape(6,16)
    batch['candidate_vectors']=None
    batch['candidate_indices']=torch.arange(6).reshape(2,3)
    indexed=ranker_batch_inputs(batch,catalogue,model,'cpu')
    with torch.no_grad():
        torch.testing.assert_close(model(**direct).logits,model(**indexed).logits,atol=0,rtol=0)
    model.requires_shared_stage1_inputs=True
    batch['stage1_customer_vectors']=torch.randn(2,16)
    batch['seq_catalog_indices']=torch.zeros(2,4,dtype=torch.long)
    shared=ranker_batch_inputs(batch,catalogue,model,'cpu')
    assert shared['candidate_indices'] is batch['candidate_indices']
    assert shared['stage1_customer_vectors'] is batch['stage1_customer_vectors']


def test_named_features_scores_only_and_historical_signal_state_layout():
    model=UPSCANRRanker(text_dim=16,hidden_dim=16,statistics_attention=True,dropout=0.).eval()
    x=model_inputs()
    with torch.no_grad():
        model.statistics_attention_weights.normal_(0,.01)
        features=model.encode_candidate_features(**x)
        assert isinstance(features,RankerFeatures) and features.features.shape==(2,3,61)
        torch.testing.assert_close(model.score_candidates(features),model(**x).logits,rtol=0,atol=0)
        with pytest.raises(ValueError,match="one logit"):
            model.score_candidates(features,scoring_head=torch.nn.Identity())
        scores=model.forward_scores(**x)
        torch.testing.assert_close(scores,model(**x).logits,atol=1e-6,rtol=1e-5)
        fast=model(**x,return_diagnostics=False)
        assert fast.attn_weights is None and fast.token_0_weight is None
    wrapped=SignalRanker(copy.deepcopy(model),'transition_mlp').eval()
    x['pairwise_features']=torch.cat([x['pairwise_features'],torch.ones(2,3,1)],-1)
    output=wrapped(**x)
    assert output.raw_scores.shape==(2,3) and wrapped.base.mlp is None
    assert not any(key.startswith('base.mlp.') for key in wrapped.state_dict())


def test_cached_training_keeps_state_keys_and_raw_inference_can_be_reenabled():
    tower=ProductTower(text_dim=16,output_dim=16,num_brands=0,num_categories=0,num_subcategories=0,num_business_lines=0,hidden_dim=16)
    model=UPSCANRRanker(text_dim=16,hidden_dim=16,product_tower=tower).eval()
    keys=set(model.state_dict());model.configure_cached_training()
    assert set(model.state_dict())==keys and not any(p.requires_grad for p in tower.parameters())
    with pytest.raises(ValueError,match='Cached-only'):
        model._resolve_candidate_vectors(b_size=1,candidate_text_emb=torch.randn(1,2,16))
    model.configure_cached_training(False)
    assert all(p.requires_grad for p in tower.parameters())
    assert model._resolve_candidate_vectors(b_size=1,candidate_text_emb=torch.randn(1,2,16)).shape==(1,2,16)


def test_prefix_statistics_are_exact_bounded_and_do_not_mutate_past_snapshots():
    frame=pd.DataFrame({'customer_id':['c']*5,'product_id':['a','b','a','a','a'],'quantity':[2.,5.,3.,7.,11.],
                        'purchase_date':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-04','2025-01-04','2025-01-10'])})
    store=PointInTimeFeatureStore(frame,max_snapshots=2,max_prefixes=1)
    before=store.get_customer_snapshot('c','2025-01-05')
    assert before.total_volume==17 and before.sku_stats['a']['median_gap']==1.5
    later=store.get_customer_snapshot('c','2025-01-11')
    assert later.total_volume==28 and later.sku_stats['a']['median_gap']==3
    back=store.get_customer_snapshot('c','2025-01-03')
    assert back.total_volume==7 and back.total_orders==2
    assert before.sku_stats['a']['volume']==12 and before.sku_stats['a']['median_gap']==1.5
    store.get_customer_snapshot('other','2025-01-03')
    assert len(store._snapshots)<=2 and len(store._prefix_states)<=1


def test_observed_metadata_excludes_current_master_and_future_fallbacks():
    frame=pd.DataFrame({'customer_id':['c','c'],'product_id':['a','a'],'quantity':[1.,1.],
                        'purchase_date':pd.to_datetime(['2025-01-01','2025-06-01']),
                        'product_category':['','future'],'brand':['','brand'],'business_line':['','line']})
    catalogue=pd.DataFrame({'product_id':['a'],'product_category':['master_future'],'brand':['master_brand']})
    company=pd.DataFrame({'customer_id':['c'],'industry_branch':['unverified']})
    store=PointInTimeFeatureStore(frame,catalogue,company,metadata_policy='observed_only_v1')
    assert store.get_product_category('a','2025-06-01')==''
    assert store.get_product_category('a','2025-06-02')=='future'
    earlier=store.compute_pairwise_features('c','2025-05-01','a')
    changed=frame.copy();changed.loc[1,'product_category']='different future'
    other=PointInTimeFeatureStore(changed,catalogue,company,metadata_policy='observed_only_v1')
    np.testing.assert_array_equal(earlier,other.compute_pairwise_features('c','2025-05-01','a'))
    assert earlier[5]==earlier[9]==earlier[10]==0


def test_rolling_eligibility_cache_is_bounded_and_returns_unshared_sets():
    frame=pd.DataFrame({'customer_id':['c'],'product_id':['a'],'quantity':[1.],'purchase_date':pd.to_datetime(['2025-01-01']),
                        'item_type':['ТОВАР'],'transaction_type':['ПРОДАЖА']})
    pool=RollingCandidatePool(pd.DataFrame({'product_id':['a'],'item_type':['ТОВАР']}),frame,max_cached_dates=2)
    assert pool.eligible_on('2025-01-01')==set()
    got=pool.eligible_on('2025-01-02');got.clear()
    assert pool.eligible_on('2025-01-02')=={'a'}
    pool.eligible_on('2025-01-03')
    assert len(pool._eligible_dates)==2


def test_configured_log_clipping_and_default_parity():
    scalar=Log1pScaler(clip_min=.2,clip_max=2.).fit([1,9])
    np.testing.assert_allclose(scalar.transform([0,9,99,999]),[.2,1,2,2])
    assert scalar.transform(0)==.2
    assert Log1pScaler().fit([9]).transform(99)==1
    with pytest.raises(ValueError): Log1pScaler(1,0)
    with pytest.raises(ValueError): Log1pScaler(float('nan'),1)


def test_transition_unknown_score_does_not_depend_on_other_candidates():
    frame=pd.DataFrame({'customer_id':['c','c'],'product_id':['a','b'],'purchase_date':pd.to_datetime(['2025-01-01','2025-02-01'])})
    stats=TransitionStatistics(frame)
    single=stats.score('c','2025-03-01',['unknown'])[0]
    mixed=stats.score('c','2025-03-01',['b','unknown'])[1]
    assert single==mixed and single<0
    with pytest.raises(ValueError): TransitionStatistics(frame,float('nan'))


def test_explicit_export_is_atomic_exclusive_and_roundtrips(tmp_path):
    table=pd.DataFrame({'customer_id':['synthetic'],'purchase_date':pd.to_datetime(['2025-01-01'])})
    path=export_purchases(table,tmp_path/'out.csv')
    content=path.read_bytes()
    with pytest.raises(FileExistsError): export_purchases(table,path)
    assert path.read_bytes()==content
    export_purchases(table.iloc[:0],path,overwrite=True)
    assert pd.read_csv(path).empty


def test_development_gate_ignores_excluded_malformed_fields(tmp_path):
    path=tmp_path/'p.csv';path.write_text('customer_id,product_id,purchase_date,quantity\na,b,2025-01-01,1\nexcluded,ignored,2026-02-19,NOT_A_NUMBER\n')
    data=load_development_purchases(path)
    assert len(data)==1 and data.quantity.iloc[0]==1


def test_disabled_demand_protocol_preserves_packaging_and_legacy_replay():
    from recommender.train_retriever import build_retriever_product_inputs
    products=['a','b']
    catalogue=pd.DataFrame({'product_id':products,'brand':['B','B'],'product_category':['C','C'],
        'product_subcategory':['S','S'],'business_line':['L','L'],'container_type':['can','can']})
    vocab={'brand_vocab':{'B':1},'category_vocab':{'C':1},'subcategory_vocab':{'S':1},
           'business_line_vocab':{'L':1},'container_vocab':{'can':1}}
    frame=pd.DataFrame({'customer_id':['c','c','c'],'product_id':['a','b','a'],'quantity':[1.,10.,100.],
                        'purchase_date':pd.to_datetime(['2025-01-01','2025-01-02','2025-09-01'])})
    embeddings={sku:np.zeros(1024,dtype=np.float32) for sku in products}
    def build(rows,policy):
        return build_retriever_product_inputs(rows,catalogue,embeddings,products,vocab,include_subcategory=False,
            include_sku_median_quantity=True,container_encoding='one_hot_none',demand_policy=policy)['demand_features']
    changed=frame.copy();changed.loc[2,'quantity']=10000
    first,second=build(frame,'disabled_v1'),build(changed,'disabled_v1')
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert not first[:,:7].any()
    torch.testing.assert_close(first[:,7:],torch.tensor([[0.,1.],[0.,1.]]))
    assert not torch.equal(build(frame,'legacy_train_cutoff_v1'),build(changed,'legacy_train_cutoff_v1'))


def test_disabled_demand_checkpoint_sanitizes_external_inputs(monkeypatch):
    from recommender.candidates import CandidateRetriever
    from recommender.models import TwoTowerModel
    model=TwoTowerModel(torch.nn.Identity(),torch.nn.Identity(),num_catalog_products=1)
    checkpoint={'product_ids':['a'],'config':{'demand_policy':'disabled_v1','include_sku_median_quantity':True}}
    monkeypatch.setattr('recommender.candidates.load_retriever_from_checkpoint',lambda *a,**kw:(model,checkpoint))
    frame=pd.DataFrame({'customer_id':['c'],'product_id':['a'],'quantity':[1.],'purchase_date':pd.to_datetime(['2025-01-01'])})
    original={'demand_features':torch.ones(1,9)}
    loaded=CandidateRetriever.from_checkpoint(Path('synthetic.pt'),frame,product_inputs=original)
    assert not loaded.product_inputs['demand_features'][:,:7].any()
    assert loaded.product_inputs['demand_features'][:,7:].eq(1).all()
    assert original['demand_features'].eq(1).all()


def test_preparation_cli_explicit_export(tmp_path,monkeypatch):
    import recommender.__main__ as cli
    table=pd.DataFrame({'customer_id':['synthetic'],'purchase_date':pd.to_datetime(['2025-01-01'])})
    monkeypatch.setattr(cli,'run_preparation',lambda **kw:{'purchases':table})
    target=tmp_path/'cleaned.csv'
    monkeypatch.setattr('sys.argv',['recommender','--export-purchases',str(target)])
    cli.main()
    assert pd.read_csv(target).customer_id.tolist()==['synthetic']


def test_completed_run_detects_changed_weights(tmp_path):
    from recommender.artifacts import atomic_json,file_digest
    run=create_run_directory(tmp_path,'r')
    selected=run/'best_ranker.pt';selected.write_bytes(b'selected')
    atomic_json({'version':1,'status':'complete','selected_checkpoint':selected.name,'selected_sha256':file_digest(selected)},run/'run.json')
    assert resolve_run_checkpoint(run)==selected
    selected.write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'): resolve_run_checkpoint(run)
