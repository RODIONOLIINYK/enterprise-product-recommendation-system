"""Warm B=1 cached Stage1 -> production slate -> Stage2 request latency."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from recommender.experiment_signals import make_model as ranker_model, load_inputs, sync
from recommender.experiment_retrieval import OUTPUT, CONFIGS, make_model, pd_day
from recommender.experiment_retrieval_data import CACHE
from recommender.shared_data import DEFAULT_RETRIEVER, load_development_purchases
from recommender.train_ranker import load_ranker_retriever
from recommender.train_shared_ranker import make_loader, batch_inputs
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs


def main():
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _,source,_=load_inputs()
    info=torch.load(CACHE/'manifest.pt',map_location='cpu')
    val=torch.load(CACHE/'val.pt',map_location='cpu')
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    companies=load_companies(purchases=purchases,minimum_purchases=1)
    retriever,_=load_ranker_retriever(DEFAULT_RETRIEVER,purchases,catalogue,companies,torch.device('cpu'))
    retriever.eligible_product_ids=info['product_ids']
    small={key:value[:32] if isinstance(value,(torch.Tensor,list)) else value for key,value in source.items()}
    batches=list(make_loader(small,batch_size=1))
    evidence={}
    with torch.no_grad():
        for path in sorted(OUTPUT.glob('*/downstream.json')):
            run_dir=path.parent; name=run_dir.name.split('-seed')[0]; seed=int(run_dir.name.split('-seed')[1])
            result=json.loads(path.read_text())
            ranker_path=Path(result['selected_checkpoint'])
            ranker=ranker_model('control').to(device).eval()
            ranker.load_state_dict(torch.load(ranker_path,map_location='cpu')['model_state_dict'],strict=True)
            retrieval=make_model(CONFIGS[name],info).to(device).eval()
            retrieval.load_state_dict(torch.load(run_dir/'best.pt',map_location='cpu')['model_state_dict'],strict=True)
            cache=torch.load(run_dir/'val_purchase_cache.pt',map_location='cpu')
            customer_vectors=cache['customer_vectors'][:32].to(device)
            vectors=cache['product_vectors'].to(device)
            indices=torch.arange(len(vectors),device=device)
            flags=val['familiarity'][:32].to(device)
            prepared=[]
            for batch in batches:
                placeholder=dict(batch)
                placeholder['candidate_indices']=torch.zeros_like(batch['candidate_indices'])
                prepared.append(batch_inputs(placeholder,vectors,ranker,device))
            def request(row):
                logits=retrieval.score_cached(customer_vectors[row:row+1],vectors,indices,flags[row:row+1])
                batch=batches[row]
                candidates=retriever.generate_candidate_batch(logits.cpu(),batch['customer_ids'],
                    [pd_day(float(batch['prediction_day'][0]))],heuristic_fraction=.5)
                arguments=dict(prepared[row])
                arguments['candidate_vectors']=vectors[candidates.candidate_indices.to(device)]
                arguments['pairwise_features']=candidates.pairwise_features.to(device)
                output=ranker(**arguments)
                return output.logits.topk(10,dim=-1).indices
            for row in range(5): request(row)
            times=[]
            for row in range(32):
                sync(device); started=time.perf_counter(); request(row); sync(device)
                times.append((time.perf_counter()-started)*1000)
            evidence[run_dir.name]={'device':str(device),'batch_size':1,'samples':len(times),
                'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                'includes':'cached customer lookup, similarities/bias/familiarity, rolling eligibility, production mixed slate/pairwise builder, transfer, Stage2 forward, Top10',
                'excludes':'network/serialization/storage I/O, global refresh and purchase encoding; all model and feature-store state warm',
                'purchase_tower_calls_per_request':0}
            print(run_dir.name,evidence[run_dir.name]['median_ms'],flush=True)
    Path('experiments/e1_e5/pipeline_latency.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__': main()
