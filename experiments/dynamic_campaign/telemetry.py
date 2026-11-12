"""Reconstruct selected models; CPU/MPS complete warm request and cache costs."""
import json
from pathlib import Path
import time
import pickle
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.scope import get_datasets_locs
from recommender.experiment_signals import sync
from recommender.train_shared_ranker import make_loader
from recommender.ranker import RankerLoss
from recommender.experiments.dynamic_selector.data import load_campaign_data, make_slates, UPSTREAM
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.model import PointwiseSelector, select_candidates
from recommender.experiments.dynamic_selector.train import make_ranker, context_inputs, predict_ranker, evaluated_checkpoint, predict_selector
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.dynamic_selector.feature_schema import NAMES, quantize_features


def timed(operation,device,repeats=50):
    for _ in range(5):operation()
    values=[]
    for _ in range(repeats):
        sync(device);start=time.perf_counter();operation();sync(device)
        values.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'samples':repeats}


def main():
    torch.set_num_threads(2)
    root=ARTIFACT_ROOT;evidence=EVIDENCE_ROOT
    results=json.loads((evidence/'results.json').read_text());decisions=json.loads((evidence/'decisions.json').read_text())
    train,val,info,manifest=load_campaign_data(build=False)
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue=load_product_catalogue(get_datasets_locs('items'));companies=load_companies(purchases=purchases,minimum_purchases=1)
    cid=val['source']['customer_ids'][0];date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(val['source']['prediction_days'][0]))
    telemetry={'warm_local':True,'network_storage_serialization_included':False,'devices':{},
               'eligibility_and_seen_masks':'resident versioned masks; sorting, quota construction and candidate feature building timed',
               'eligibility_mask_bytes_per_date':len(info['product_ids']),
               'unseen_mask_bytes_per_customer':len(info['product_ids']),
               'customer_vector_bytes':128*4,'product_vector_bytes':val['v'].numel()*val['v'].element_size(),
               'feature_cache_bytes':sum((Path('data/interim/dynamic_campaign_v1')/f'{s}_features.npy').stat().st_size for s in ('train','val')),
               'features_per_customer_product':len(NAMES),'dynamic_features_cached_per_request':False,
               'feature_float32_bytes_per_eligible_request':int(val['allowed'][0].sum())*len(NAMES)*4,
               'global_periodic_scalar_bytes_per_snapshot':len(info['product_ids'])*7*4,
               'upstream_digest':file_digest(UPSTREAM/'best.pt')}
    snapshot_names=('text_embedding','city_id','region_id','industry_id','tabular_features','sequence','attention_mask',
                    'seq_product_ids','seq_category_ids','seq_subcategory_ids','seq_business_line_ids','seq_purchase_dates')
    telemetry['ranker_purchase_input_bytes']=sum(val['source'][key][0].numel()*val['source'][key].element_size() for key in snapshot_names)
    telemetry['purchase_tensor_cache_bytes_including_stage1_vector']=telemetry['ranker_purchase_input_bytes']+telemetry['customer_vector_bytes']
    groups=(('customer_timing',range(13,17)),('category_family_timing',range(17,24)),
            ('frozen_global',range(24,31)),('calendar',range(31,33)),('relationships',range(33,41)))
    for key,result in results['selectors'].items():
        if key=='dot' or result['settings']['mode']=='static':continue
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        selector=PointwiseSelector(result['settings']['mode']).to(device).eval()
        selector.load_state_dict(torch.load(result['selected_checkpoint'],map_location='cpu')['model_state_dict'])
        original_mask=selector.feature_mask.clone();result['group_zero_ablations']={}
        for name,columns in groups:
            selector.feature_mask.copy_(original_mask);selector.feature_mask[list(columns)]=0
            result['group_zero_ablations'][name]=ranking_metrics(predict_selector(selector,val,device),val)
    # Enrich every selected checkpoint with identical final metric definitions
    # and target-level cohorts; no reselection after looking at these diagnostics.
    for key,result in results['rankers'].items():
        settings=result['settings'];name=key.rsplit('-'+settings['mode']+'-',1)[0]
        if name in ('personal-repeat','production50','finetune50'):
            from experiments.dynamic_campaign.production_control import slates as production_slates
            slate=production_slates(val,'val',info['product_ids'])
            selector,policy=None,None
        elif name=='fixed50':selector,policy='dot','quota50_25'
        elif name=='combined':selector,policy=decisions['combined_selector'],decisions['combined_policy']
        elif name in ('dot100','dynamic100'):selector,policy=('dot' if name=='dot100' else 'dynamic'),'quota100_50'
        else:selector,policy=name.split('-',1)
        seed=settings['seed']
        if selector is not None:
            score_file=root/'runs'/f'selector-{selector}-seed{seed}'/'val_scores.pt'
            scores=val['dot_scores'] if selector=='dot' else torch.load(score_file,map_location='cpu')['scores']
            slate=make_slates(val,scores,info['product_ids'],policy)
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model=make_ranker(settings['mode'],seed).to(device).eval()
        model.load_state_dict(torch.load(evaluated_checkpoint(result),map_location='cpu')['model_state_dict'],strict=True)
        result['metrics_final']=ranking_metrics(predict_ranker(model,slate,val['v'].to(device),device),val,slate['candidate_indices'])
        for metric in ('recall@10','ndcg@10','new_hits@10'):
            if result['metrics_final'][metric]!=result['metrics'][metric]:raise ValueError('Final replay differs')
        if getattr(model,'uses_context',False):
            result['ablation_no_candidate_attention']=ranking_metrics(
                predict_ranker(model,slate,val['v'].to(device),device,disable_candidate_attention=True),val,slate['candidate_indices'])
        if settings['mode'] in ('dynamic_features','combined_direct'):
            result['group_zero_ablations']={}
            for name,columns in groups:
                ablated=dict(slate);ablated['pairwise_features']=slate['pairwise_features'].clone()
                ablated['pairwise_features'][...,list(columns)]=0
                result['group_zero_ablations'][name]=ranking_metrics(predict_ranker(model,ablated,val['v'].to(device),device),val,slate['candidate_indices'])
        model.zero_grad(set_to_none=True)
        batch=next(iter(make_loader(slate)))
        output=model(**context_inputs(batch,val['v'].to(device),model,device))
        RankerLoss()(output.logits,batch['labels'].to(device))['loss'].backward()
        result['active_gradient_parameters']=sum(p.numel() for p in model.parameters() if p.grad is not None)
        result['unused_trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is None)
        model.zero_grad(set_to_none=True)
        one=next(iter(make_loader(slate,batch_size=1)))
        current=context_inputs(one,val['v'].to(device),model,device)
        with torch.no_grad():
            result['forward_latency_batch1']={str(device):timed(lambda:model(**current),device,repeats=30)}
        if device.type!='cpu':
            model.to('cpu')
            current=context_inputs(one,val['v'],model,torch.device('cpu'))
            with torch.no_grad():result['forward_latency_batch1']['cpu']=timed(lambda:model(**current),torch.device('cpu'),repeats=30)
    selected=decisions['combined_selector'];policy=decisions['combined_policy']
    selector_payload=None if selected=='dot' else torch.load(root/'runs'/f'selector-{selected}-seed42'/'best.pt',map_location='cpu')
    for device in (torch.device('cpu'),torch.device('mps')) if torch.backends.mps.is_available() else (torch.device('cpu'),):
        vectors=val['v'].to(device);u=val['u'][:1].to(device);reference=val['dot_scores'][:1].to(device)
        selector=None if selected=='dot' else PointwiseSelector(selected).to(device).eval()
        if selector is not None:selector.load_state_dict(selector_payload['model_state_dict'])
        allowed=val['allowed'][:1].to(device);unseen=val['unseen'][:1].to(device)
        builder=DynamicFeatureBuilder(purchases,catalogue,companies,info['product_ids'])
        simple_builder=DynamicFeatureBuilder(purchases,catalogue,companies,info['product_ids'],relationships=selected=='relations')
        pair=builder.build(cid,date)
        if 'global_relationship_array_bytes' not in telemetry:
            telemetry['global_relationship_array_bytes']=sum(
                table.pairs.nbytes+table.source.nbytes+table.dest.nbytes+table.mapping.nbytes
                for table in (*builder.tables.values(),builder.complements))
            simple_tables=[*simple_builder.tables.values()]+([simple_builder.complements] if simple_builder.complements is not None else [])
            telemetry['global_relationship_bytes_simple_pipeline']=sum(table.pairs.nbytes+table.source.nbytes+table.dest.nbytes+table.mapping.nbytes for table in simple_tables)
            telemetry['global_cache_max_dates']=512
            telemetry['global_cache_max_numeric_bytes']=512*len(info['product_ids'])*7*4
        if 'customer_facts_serialized_bytes' not in telemetry:
            sizes=[]
            for row in range(min(256,len(val['u']))):
                customer=val['source']['customer_ids'][row]
                at=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(val['source']['prediction_days'][row]))
                snapshot=builder.store.get_customer_snapshot(customer,at)
                history=[event for event in builder.store.customer_events.get(customer,[]) if event[0]<at]
                sizes.append(len(pickle.dumps((asdict(snapshot),history))))
            telemetry['customer_facts_serialized_bytes']={'median':float(np.median(sizes)),'p95':float(np.quantile(sizes,.95)),
                'sample_rows':len(sizes),'note':'serialized strictly prior events plus snapshot, excludes Python allocator overhead'}
        models={mode:make_ranker(mode,42).to(device).eval() for mode in ('control','combined_direct')}
        for mode,model in models.items():
            checkpoint=evaluated_checkpoint(results['rankers'][f'combined-{mode}-42'])
            model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model_state_dict'])
        scores=val['dot_scores'] if selected=='dot' else torch.load(root/'runs'/f'selector-{selected}-seed42'/'val_scores.pt',map_location='cpu')['scores']
        slate=make_slates(val,scores,info['product_ids'],policy)
        batch=next(iter(make_loader(slate,batch_size=1)))
        fixed=context_inputs(batch,vectors,models['control'],device)
        def selector_score(features):
            return reference if selector is None else selector(u,vectors,features,reference)
        feature_tensor=torch.from_numpy(quantize_features(pair)).unsqueeze(0).to(device)
        def full(model,provider):
            fresh=torch.from_numpy(quantize_features(provider.build(cid,date))).unsqueeze(0).to(device)
            score=selector_score(fresh)
            chosen=select_candidates(score,allowed,unseen,policy)
            current=dict(fixed);current['candidate_vectors']=vectors[chosen]
            current['pairwise_features']=fresh.gather(1,chosen[:,:,None].expand(-1,-1,len(NAMES)))
            return model(**current).logits.topk(10).indices
        with torch.no_grad():
            telemetry['devices'][str(device)]={
                'feature_generation':timed(lambda:builder.build(cid,date),device),
                'feature_generation_no_unused_relationships':timed(lambda:simple_builder.build(cid,date),device),
                'selector_scoring':timed(lambda:selector_score(feature_tensor),device),
                'ranker_forward':{mode:timed(lambda m=model:m(**fixed),device) for mode,model in models.items()},
                'complete_request':{mode:timed(lambda m=model,p=(simple_builder if mode=='control' else builder):full(m,p),device) for mode,model in models.items()}}
        # Essential depth-matched serving comparison: dot selection can build
        # pair features only for its shortlist, with no full-pool pointwise MLP.
        teacher=torch.load(UPSTREAM/'best.pt',map_location='cpu')['model_state_dict']
        bias=teacher['towers.product_bias'].to(device);weights=teacher['towers.familiarity_weights'].to(device)
        flags=val['familiarity'][:1].to(device).float()
        dot_slate=make_slates(val,val['dot_scores'],info['product_ids'],'quota100_50')
        one=next(iter(make_loader(dot_slate,batch_size=1)))
        dot_measurements={}
        for mode in ('control','dynamic_features'):
            model=make_ranker(mode,42).to(device).eval()
            result=results['rankers'][f'dot100-{mode}-42']
            model.load_state_dict(torch.load(evaluated_checkpoint(result),map_location='cpu')['model_state_dict'])
            fixed_dot=context_inputs(one,vectors,model,device)
            provider=DynamicFeatureBuilder(purchases,catalogue,companies,info['product_ids'],relationships=mode=='dynamic_features')
            def dot_request():
                score=u@vectors.T/.1+bias+flags@weights
                chosen=select_candidates(score,allowed,unseen,'quota100_50')
                pids=[info['product_ids'][i] for i in chosen[0].cpu().tolist()]
                if mode=='control':values=provider.store.compute_slate_pairwise_features(cid,date,pids).numpy()
                else:values=provider.build(cid,date,candidate_ids=pids)
                current=dict(fixed_dot);current['candidate_vectors']=vectors[chosen]
                current['pairwise_features']=torch.from_numpy(quantize_features(values)).unsqueeze(0).to(device)
                return model(**current).logits.topk(10).indices
            with torch.no_grad():dot_measurements[mode]=timed(dot_request,device)
        telemetry['devices'][str(device)]['depth_matched_dot_complete_request']=dot_measurements
    atomic_json(telemetry,evidence/'telemetry.json',overwrite=False)
    atomic_json(results,evidence/'results_enriched.json',overwrite=False)
    print('Selected checkpoints replayed; warm CPU/MPS complete-request telemetry recorded.',flush=True)


if __name__=='__main__':main()
