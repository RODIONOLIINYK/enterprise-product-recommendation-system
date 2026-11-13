import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from recommender.development import load_development_purchases
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.rolling_candidate_v2.data import RollingDynamicFeatureAdapter,verify_active_reference
from recommender.experiments.rolling_candidate_v2.benchmark import _time_callable
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.features import compute_point_in_time_features, PointInTimeNormalizer
from recommender.sequences import PURCHASE_DATE_ORIGIN,DAYS_PER_YEAR

torch.set_num_threads(4)
purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
cat=load_product_catalogue(get_datasets_locs('items'))
companies=load_companies(purchases=purchases,minimum_purchases=1)
tr,va,info,ts,vs=prepare_retrieval_data()
ref=verify_active_reference(load_selector=True)
adapter=RollingDynamicFeatureAdapter(purchases,cat,companies,info['product_ids'])
ck=torch.load('checkpoints/retriever-discovery-reference/best_retriever.pt',map_location='cpu',weights_only=False)
normalizer=PointInTimeNormalizer.from_state(ck['config']['profile_normalizer'])
profile=ck['config']['profile_features']
report={'training_executed':False,'heldout_access':False,'products':len(info['product_ids']),'checked_splits':{},'timings':{}}
query_dates=[]
for split,ret,src in [('train',tr,ts),('val',va,vs)]:
    kept=torch.where(ret['allowed'].sum(1)>=100)[0]
    cache=np.load(f'data/interim/rolling_candidate_v2/{split}_features.npy',mmap_mode='r')
    date_order=torch.argsort(src['prediction_days'][kept],stable=True).tolist()
    selected=[date_order[i] for i in np.linspace(0,len(kept)-1,24,dtype=int)][::-1]
    feat_diff=stats_diff=0.; max_history_delta=-1e9
    groups={str(cid):g for cid,g in purchases.groupby('customer_id',sort=False)}
    for row in selected:
        raw_i=int(kept[row]); cid=str(src['customer_ids'][raw_i]); date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(src['prediction_days'][raw_i]))
        f=adapter.build_quantized(cid,date)
        feat_diff=max(feat_diff,float(np.max(np.abs(f-cache[row].astype(np.float32)))))
        stats=compute_point_in_time_features(groups[cid],cid,date,normalizer=normalizer,include_median_purchase_quantity=True)
        stat_vec=torch.tensor([stats[n] for n in profile],dtype=torch.float32)
        stats_diff=max(stats_diff,float((stat_vec-src['retriever_tabular_features'][raw_i]).abs().max()))
        histdates=src['seq_purchase_dates'][raw_i][src['attention_mask'][raw_i].bool()]
        asof_years=(date-PURCHASE_DATE_ORIGIN).days/DAYS_PER_YEAR
        if len(histdates): max_history_delta=max(max_history_delta,float(histdates.max())-asof_years)
        if split=='val': query_dates.append((row,raw_i,cid,date))
    assert feat_diff==0.,(split,feat_diff)
    assert stats_diff < 1e-6,(split,stats_diff)
    assert max_history_delta<0.,(split,max_history_delta)
    report['checked_splits'][split]={'first_date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(src['prediction_days'][kept].min()))).date()),'last_date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(src['prediction_days'][kept].max()))).date()),'rows':len(selected),'feature_max_abs_diff':feat_diff,'stats_6_max_abs_diff':stats_diff,'latest_history_minus_prediction_years_max':max_history_delta,'reverse_query_order':True}

# Independent prefix-only reconstruction: future/same-day purchases unavailable to builder.
report['prefix_only_rebuilds']=[]
for row,raw_i,cid,date in [query_dates[-1],query_dates[len(query_dates)//2],query_dates[0]]:
    prefix=RollingDynamicFeatureAdapter(purchases.loc[purchases.purchase_date<date].copy(),cat,companies,info['product_ids'])
    diff=float(np.max(np.abs(prefix.build_quantized(cid,date)-adapter.build_quantized(cid,date))))
    assert diff==0.,(date,diff)
    report['prefix_only_rebuilds'].append({'prediction_date':str(date.date()),'all_41_features_max_abs_diff':diff})

row,raw_i,cid,date=query_dates[len(query_dates)//2]
cp=torch.load('checkpoints/embedding-ablation-v1/runs/stage1-E_no_brand-seed42/val_purchase_cache.pt',map_location='cpu',weights_only=False)
u=cp['customer_vectors'][raw_i].reshape(1,128); v=cp['product_vectors']; stats=quantize_transferred_stats(vs['retriever_tabular_features'][raw_i:raw_i+1]); allowed=va['allowed'][raw_i:raw_i+1]
selector=ref['selector'].eval()
for name in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
    device=torch.device(name); sel=copy.deepcopy(selector).to(device); ud=u.to(device);vd=v.to(device);sd=stats.to(device);mask=allowed.to(device)
    fd=adapter.build_tensor(cid,date,device=device)
    with torch.inference_mode():
        def forward(): return sel(u=ud,v=vd,features_41=fd,customer_stats_6=sd).masked_fill(~mask,-torch.inf).topk(100,dim=1)
        def request():
            feats=adapter.build_tensor(cid,date,device=device)
            return sel(u=ud,v=vd,features_41=feats,customer_stats_6=sd).masked_fill(~mask,-torch.inf).topk(100,dim=1)
        report['timings'][name]={'fullpool_forward_mask_top100':_time_callable(forward,device,warmup=3,repeats=30),'rolling_features_transfer_forward_top100_warm_query':_time_callable(request,device,warmup=3,repeats=30)}
report['timing_scope']='One customer, 841 canonical products; eligible mask before top100; real cached vectors/stats; includes request feature extraction and transfer where named. Excludes purchase ingestion, lifetime-stat/history preparation, vector recomputation, I/O, networking.'
report['normalization']='Legacy signed 8 retained context features and signed latent vectors preserved; not all inputs in [0,1].'
Path('experiments/rolling_pipeline_audit_v1/independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
