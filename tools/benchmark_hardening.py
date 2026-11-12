"""Synthetic pre/post audit parity and latency. Requires Git revision c7788ec.

No company data, checkpoint files or held-out purchases are read. Historical
reports remain unchanged; results go to audit/hardening_benchmarks.json.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np
import pandas as pd
import torch
from recommender.candidates import PointInTimeFeatureStore
from recommender.ranker import UPSCANRRanker
from recommender.experimental_signals import SignalRanker

REVISION='c7788ec'


def old_module(path, name):
    source=subprocess.run(['git','show',f'{REVISION}:{path}'],check=True,capture_output=True,text=True).stdout
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(source,f'{REVISION}:{path}','exec'),module.__dict__)
    return module


def synchronize(device):
    if device.type=='mps': torch.mps.synchronize()
    elif device.type=='cuda': torch.cuda.synchronize()


def latency(operation, device, repeats=100):
    for _ in range(10): operation()
    samples=[]
    for _ in range(repeats):
        synchronize(device);start=time.perf_counter();operation();synchronize(device)
        samples.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),'repeats':repeats}


def main():
    torch.set_num_threads(2);torch.manual_seed(23)
    reference=old_module('recommender/candidates.py','audit_old_candidates')
    old_ranker=old_module('recommender/ranker.py','audit_old_ranker')
    old_signals=old_module('recommender/experimental_signals.py','audit_old_signals')
    dates=pd.Timestamp('2025-01-01')+pd.to_timedelta(np.cumsum(np.arange(1000)%3+1),unit='h')
    purchases=pd.DataFrame({'customer_id':['synthetic']*1000,'product_id':[f's{i%11}' for i in range(1000)],
                            'quantity':(np.arange(1000)%5+1).astype(float),'purchase_date':dates})
    catalogue=pd.DataFrame({'product_id':[f's{i}' for i in range(11)],'brand':['b']*11,
                            'product_category':['c']*11,'product_subcategory':['s']*11,'business_line':['l']*11})
    queries=list(dates[::10]+pd.Timedelta(seconds=1))
    timings={};snapshots={};features={}
    for label, cls in (('before',reference.PointInTimeFeatureStore),('after',PointInTimeFeatureStore)):
        samples=[]
        for _ in range(5):
            store=cls(purchases,catalogue)
            start=time.perf_counter();rows=[asdict(store.get_customer_snapshot('synthetic',day)) for day in queries]
            samples.append((time.perf_counter()-start)*1000)
        snapshots[label]=rows
        features[label]=[store.compute_slate_pairwise_features('synthetic',day,['s0','s1']) for day in queries]
        timings[label]={'median_ms':float(np.median(samples)),'samples':samples}
    assert snapshots['before']==snapshots['after']
    for left,right in zip(features['before'],features['after']): torch.testing.assert_close(left,right,atol=0,rtol=0)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    config=dict(statistics_attention=True,excluded_statistics=('history_repeat',),dropout=.1)
    old=old_ranker.UPSCANRRanker(**config).eval().to(device)
    with torch.no_grad(): old.statistics_attention_weights.normal_(0,.01)
    new=UPSCANRRanker(**config).eval().to(device);new.load_state_dict(old.state_dict(),strict=True)
    x=dict(text_embedding=torch.randn(1,1024,device=device),city_id=torch.zeros(1,dtype=torch.long,device=device),
           region_id=torch.zeros(1,dtype=torch.long,device=device),industry_id=torch.zeros(1,dtype=torch.long,device=device),
           tabular_features=torch.rand(1,8,device=device),sequence=torch.rand(1,32,9,device=device),
           attention_mask=torch.ones(1,32,dtype=torch.long,device=device),
           seq_purchase_dates=torch.arange(32,device=device).float().unsqueeze(0)/365.25,
           prediction_day=torch.full((1,),100.,device=device),
           candidate_vectors=torch.nn.functional.normalize(torch.randn(1,50,128,device=device),dim=-1),
           pairwise_features=torch.rand(1,50,13,device=device))
    with torch.no_grad():
        original=old(**x).logits;default=new(**x).logits;fast=new.forward_scores(**x)
        torch.testing.assert_close(original,default,atol=0,rtol=0)
        torch.testing.assert_close(original,fast,atol=2e-6,rtol=1e-5)
        old_e2=old_signals.SignalRanker(copy.deepcopy(old),'transition_mlp').eval().to(device)
        old_e2.head[0].weight[:,-1].normal_(0,.1)
        new_e2=SignalRanker(copy.deepcopy(new),'transition_mlp').eval().to(device)
        new_e2.load_state_dict(old_e2.state_dict(),strict=True)
        y=dict(x);y['pairwise_features']=torch.cat([x['pairwise_features'],torch.randn(1,50,1,device=device)],-1)
        old_e2_scores=old_e2(**y).logits;new_e2_scores=new_e2(**y).logits
        torch.testing.assert_close(old_e2_scores,new_e2_scores,atol=0,rtol=0)
        scores={'diagnostics':latency(lambda:new(**x),device),'scores_only':latency(lambda:new.forward_scores(**x),device)}
    result={'reference_revision':REVISION,'synthetic_only':True,'heldout_access':False,
            'environment':{'torch':torch.__version__,'pandas':pd.__version__,'device':str(device),'cpu_threads':torch.get_num_threads()},
            'prefix':{'events':1000,'queries':100,'snapshots_exact':True,'pairwise_features_exact':True,'timings':timings},
            'ranker':{'default_max_abs_difference':float((original-default).abs().max()),
                      'scores_only_max_abs_difference':float((original-fast).abs().max()),
                      'historical_e2_max_abs_difference':float((old_e2_scores-new_e2_scores).abs().max()),
                      'timings':scores,'batch_size':1,'candidates':50,'history_events':32},
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('recommender/ranker.py'),Path('recommender/candidates.py')]},
            'limitation':'Synthetic warm local measurements, not an end-to-end deployment or validation-quality benchmark.'}
    target=Path('audit/hardening_benchmarks.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
