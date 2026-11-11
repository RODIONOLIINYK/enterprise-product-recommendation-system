"""Independent E3–E5 training, cached retrieval, mixed slates and downstream fits.

PYTHONPATH=. ./venv/bin/python -m recommender.experiment_retrieval --seeds 42
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from recommender.experimental_retrieval import RetrievalConfig, RetrievalExperiment
from recommender.experiment_retrieval_data import prepare_retrieval_data, FIELDS
from recommender.experiment_signals import sync, fit, OUTPUT as SIGNAL_OUTPUT
from recommender.models import CustomerTower, ProductTower, TwoTowerModel
from recommender.shared_data import DEFAULT_RETRIEVER, atomic_torch_save, load_development_purchases
from recommender.train_retriever import get_device
from recommender.train_ranker import checkpoint_digest, load_ranker_retriever
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs

OUTPUT = Path('checkpoints/e1-e5/retrieval-v1')
CONFIGS = {
    'S0': RetrievalConfig(),
    'E3mean': RetrievalConfig(long_memory='mean'),
    'E3attention': RetrievalConfig(long_memory='attention'),
    'E4k1': RetrievalConfig(interests=1),
    'E4k2': RetrievalConfig(interests=2),
    'E4k3': RetrievalConfig(interests=3),
    'E5k2': RetrievalConfig(interests=2, scoring='smooth'),
    'E5k3': RetrievalConfig(interests=3, scoring='smooth'),
}


def make_model(config, info):
    original = torch.load(DEFAULT_RETRIEVER, map_location='cpu')
    c, s = original['config'], original['model_state_dict']
    # Construct from shapes/config only; do not load learned reference weights.
    customer = CustomerTower(text_dim=1024, output_dim=128, hidden_dim=128,
        num_cities=s['customer_tower.city_embedding.weight'].shape[0],
        num_regions=s['customer_tower.region_embedding.weight'].shape[0],
        num_industries=s['customer_tower.industry_embedding.weight'].shape[0],
        num_tabular_features=6, catalog_history_encoder='attention', history_gap_reference_days=74.,
        max_seq_len=32, num_heads=4, num_layers=2, dim_feedforward=256, dropout=.1, seq_feature_dim=2)
    product = ProductTower(text_dim=1024, output_dim=128, num_brands=0, num_categories=0,
        num_business_lines=0, num_subcategories=0, num_demand_features=info['product_inputs']['demand_features'].shape[1], hidden_dim=128, dropout=.1)
    towers = TwoTowerModel(customer, product, num_catalog_products=len(info['product_ids']), temperature=.1, num_familiarity_features=4)
    return RetrievalExperiment(towers, config)


def inputs_for(data, rows, device):
    return {key: data[key][rows].to(device) for key in (*FIELDS, 'tabular_features', 'seq_product_ids', 'older_ids', 'older_counts')}


def metrics_from_scores(logits, data, source):
    eligible_scores = logits.masked_fill(~data['allowed'], -torch.inf)
    unseen_scores = logits.masked_fill(~data['unseen'], -torch.inf)
    orders = eligible_scores.argsort(-1, descending=True)
    novel_orders = unseen_scores.argsort(-1, descending=True)
    positive_counts = np.array([len(row) for row in source['positive_pids']])
    new_count = sum(map(len, source['new_positive_pids']))
    top10 = orders[:, :10]
    hits = data['target'].gather(1, top10).numpy()
    recall = hits.sum(1) / positive_counts
    discounts = 1 / np.log2(np.arange(10) + 2)
    ideal = (discounts[None, :] * (np.arange(10)[None, :] < positive_counts[:, None])).sum(1)
    ndcg = (hits * discounts[None, :]).sum(1) / ideal
    all_new10 = int(data['new_target'].gather(1, top10).sum())
    top50 = novel_orders[:, :50]
    novel_hits = data['new_target'].gather(1, top50) & data['unseen'].gather(1, top50)
    new25, new50 = int(novel_hits[:, :25].sum()), int(novel_hits.sum())
    return {'val_recall@10': float(np.mean(recall)), 'val_ndcg@10': float(np.mean(ndcg)),
            'new_hits@10': all_new10, 'new_recall@10': all_new10/max(new_count,1),
            'new_hits@25': new25, 'new_hits@50': new50, 'new_targets': new_count,
            'stage1_new_recall@25': new25/max(new_count,1), 'stage1_new_recall@50': new50/max(new_count,1)}



@torch.no_grad()
def encode_and_score(model, data, info, device, return_memory=False):
    model.eval()
    product_inputs = {key: value.to(device) for key, value in info['product_inputs'].items()}
    vectors = model.product_vectors(product_inputs)
    encoded, scores, memories = [], [], []
    indices = torch.arange(len(vectors), device=device)
    for rows in torch.arange(len(data['target'])).split(256):
        customers, memory = model.encode_purchase(inputs_for(data, rows, device), vectors, return_memory=True)
        if memory is not None:
            memories.append(memory.cpu())
        encoded.append(customers.cpu())
        scores.append(model.score_cached(customers, vectors, indices, data['familiarity'][rows].to(device)).cpu())
    result = (vectors.cpu(), torch.cat(encoded), torch.cat(scores))
    return (*result, torch.cat(memories) if memories else None) if return_memory else result


def collapse_diagnostics(customers, vectors):
    if customers.shape[1] == 1:
        return {'interests': 1}
    cosine = customers @ customers.transpose(1,2)
    upper = torch.triu(torch.ones(customers.shape[1], customers.shape[1], dtype=torch.bool), diagonal=1)
    pairwise = cosine[:, upper]
    similarity = customers @ vectors.T
    top_two = similarity.topk(2, dim=1).values
    margins = top_two[:, 0] - top_two[:, 1]
    winners = similarity.argmax(1)
    frequencies = torch.bincount(winners.flatten(), minlength=customers.shape[1]).float()
    frequencies /= frequencies.sum()
    # Diversity of actual rankings, not just of query parameters.
    top = similarity.topk(min(25,len(vectors)), dim=-1).indices
    jaccard = []
    for row in top[:1000]:
        for i in range(len(row)):
            for j in range(i+1,len(row)):
                a,b=set(row[i].tolist()),set(row[j].tolist())
                jaccard.append(len(a&b)/len(a|b))
    return {'interests': customers.shape[1], 'mean_pairwise_cosine': float(pairwise.mean()),
            'cosine_quantiles': torch.quantile(pairwise.flatten(), torch.tensor([0.,.5,.9,1.])).tolist(),
            'fraction_pairs_above_0.99': float((pairwise>.99).float().mean()),
            'winner_fraction_all_products': frequencies.tolist(), 'mean_winner_margin': float(margins.mean()),
            'fraction_winner_margin_lt1e-5': float((margins < 1e-5).float().mean()), 'mean_top25_jaccard_first1000': float(np.mean(jaccard))}


def serving_latency(model, customers, vectors, data, info, device):
    model.eval()
    u,v=customers[:1].to(device),vectors.to(device)
    familiarity=data['familiarity'][:1].to(device)
    ids=torch.arange(len(v),device=device)
    x=inputs_for(data, torch.tensor([0]), device)
    result={}
    with torch.no_grad():
        for label, operation in (('cached_similarity', lambda:model.score_cached(u,v,ids,familiarity)),
                                 ('purchase_encoding', lambda:model.encode_purchase(x,v))):
            for _ in range(5): operation()
            times=[]
            for _ in range(30):
                sync(device); start=time.perf_counter(); operation(); sync(device)
                times.append((time.perf_counter()-start)*1000)
            result[label]={'median_ms':float(np.median(times)), 'p95_ms':float(np.quantile(times,.95))}
    return result


def train_retrieval(run, seed, train, val, info, train_source, val_source, output=OUTPUT, epochs=30, device=None):
    device=get_device(device)
    run_dir=Path(output)/f'{run}-seed{seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    settings={'run':run,'seed':seed,'epochs':epochs,'config':asdict(CONFIGS[run]),'identity':info['identity'],
              'batch_size':256,'lr':.001,'weight_decay':.0001,'pos_weight':20.,
              'selection':'new_hits@25, then new_hits@50; first epoch wins ties',
              'demand_policy':info['demand_policy'],'initialization':'scratch'}
    path=run_dir/'result.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['settings']!=settings: raise ValueError('Incompatible completed retrieval experiment')
        return result
    if (run_dir/'best.pt').exists(): raise ValueError('Incomplete run: choose a new output path')
    torch.manual_seed(seed); np.random.seed(seed)
    model=make_model(CONFIGS[run],info).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5)
    # Only prior-sale, unseen products enter BCE; globally first-sale targets cannot be retrieved.
    useful=torch.where((train['target'] & train['unseen']).any(-1))[0]
    generator=torch.Generator().manual_seed(seed)
    product_inputs={key:value.to(device) for key,value in info['product_inputs'].items()}
    n_train=info['trained_products']
    train_inputs={key:value[:n_train] for key,value in product_inputs.items()}
    history=[]; best=(-1,-1); started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for rows in useful[torch.randperm(len(useful),generator=generator)].split(256):
            optimizer.zero_grad(set_to_none=True)
            vectors=model.product_vectors(train_inputs)
            customers=model.encode_purchase(inputs_for(train,rows,device),vectors)
            logits=model.score_cached(customers,vectors,torch.arange(n_train,device=device),train['familiarity'][rows,:n_train].to(device))
            allowed=train['unseen'][rows,:n_train].to(device)
            target=train['target'][rows,:n_train].to(device).float()
            loss=F.binary_cross_entropy_with_logits(logits[allowed],target[allowed],pos_weight=torch.tensor(20.,device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        vectors, customers, logits=encode_and_score(model,val,info,device)
        metrics=metrics_from_scores(logits,val,val_source)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'metrics':metrics})
        selection=(metrics['new_hits@25'],metrics['new_hits@50'])
        if selection>best:
            best=selection
            atomic_torch_save({'settings':settings,'model_state_dict':model.state_dict(),'epoch':epoch,'metrics':metrics},run_dir/'best.pt')
        (run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(f"{run} seed={seed} epoch={epoch} new25={metrics['new_hits@25']} new50={metrics['new_hits@50']} rec10={metrics['val_recall@10']:.4f}",flush=True)
    sync(device); elapsed=time.perf_counter()-started
    checkpoint=torch.load(run_dir/'best.pt',map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    vectors,customers,logits,memory=encode_and_score(model,val,info,device,True)
    metrics=metrics_from_scores(logits,val,val_source)
    assert metrics==checkpoint['metrics']
    atomic_torch_save({'cache_version':1,'model_digest':checkpoint_digest(run_dir/'best.pt'),'data_identity':info['identity'],
                       'customer_vectors':customers,'long_memory':memory,'product_vectors':vectors,'scores':logits},run_dir/'val_purchase_cache.pt')
    result={'settings':settings,'selected_epoch':checkpoint['epoch'],'metrics':metrics,
            'training_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters()),
            'collapse':collapse_diagnostics(customers,vectors),'latency':serving_latency(model,customers,vectors,val,info,device),
            'cache_bytes_per_customer':(customers.shape[1]+int(memory is not None))*128*4,'product_cache_bytes':vectors.numel()*vectors.element_size(),
            'device':str(device),'training_discovery_rows':len(useful)}
    if CONFIGS[run].long_memory!='none':
        ablated=dict(val); ablated['older_counts']=torch.zeros_like(val['older_counts'])
        _,_,scores=encode_and_score(model,ablated,info,device)
        result['ablation_no_older_memory']=metrics_from_scores(scores,ablated,val_source)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def downstream(run,seed,train,val,info,train_source,val_source,output,device,ranker_epochs=10):
    run_dir=Path(output)/f'{run}-seed{seed}'
    checkpoint=torch.load(run_dir/'best.pt',map_location='cpu')
    model=make_model(CONFIGS[run],info).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    companies=load_companies(purchases=purchases,minimum_purchases=1)
    retriever,_=load_ranker_retriever(DEFAULT_RETRIEVER,purchases,catalogue,companies,torch.device('cpu'))
    retriever.eligible_product_ids=info['product_ids']
    slates=[]
    for split,data,source in (('train',train,train_source),('val',val,val_source)):
        cached=run_dir/f'{split}_purchase_cache.pt'
        if cached.exists():
            payload=torch.load(cached,map_location='cpu')
            if payload['model_digest']!=checkpoint_digest(run_dir/'best.pt') or payload['data_identity']!=info['identity']:
                raise ValueError('Purchase cache version mismatch')
            vectors,customers,logits=payload['product_vectors'],payload['customer_vectors'],payload['scores']
        else:
            vectors,customers,logits,memory=encode_and_score(model,data,info,device,True)
            atomic_torch_save({'cache_version':1,'model_digest':checkpoint_digest(run_dir/'best.pt'),'data_identity':info['identity'],
                               'customer_vectors':customers,'long_memory':memory,'product_vectors':vectors,'scores':logits},cached)
        path=run_dir/f'stage2_{split}.pt'
        if path.exists():
            slate=torch.load(path,map_location='cpu')
        else:
            slate=dict(source)
            dates=[pd_day(day) for day in source['prediction_days'].tolist()]
            # Use the production 25-repeat / remaining-unseen construction and exact pairwise feature builder.
            candidate_batches=[]
            for start in range(0,len(logits),256):
                end=start+256
                candidate_batches.append(retriever.generate_candidate_batch(logits[start:end],source['customer_ids'][start:end],dates[start:end],
                    source['positive_pids'][start:end],heuristic_fraction=.5))
            for key in ('candidate_indices','pairwise_features','labels'):
                slate[key]=torch.cat([getattr(batch,key) for batch in candidate_batches])
            slate['candidate_pids']=[row for batch in candidate_batches for row in batch.candidate_pids]
            slate['cache_identity']=f"{info['identity']}:{checkpoint_digest(run_dir/'best.pt')}:{split}"
            atomic_torch_save(slate,path)
        slates.append(slate)
    # The downstream ranker transfers matching projections from *this clean model*.
    manifest={'identity':f"{info['identity']}:{checkpoint_digest(run_dir/'best.pt')}",
              'stage1_digest':checkpoint_digest(run_dir/'best.pt'),'product_vectors':vectors,
              'clean_stage1_state':checkpoint['model_state_dict']}
    result=fit('control',seed,*slates,manifest,run_dir/'downstream',ranker_epochs,device)
    (run_dir/'downstream.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def pd_day(day):
    import pandas as pd
    return pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(day))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--ranker-epochs',type=int,default=10)
    parser.add_argument('--device',default=None)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--skip-downstream',action='store_true')
    args=parser.parse_args(); device=get_device(args.device)
    train,val,info,train_source,val_source=prepare_retrieval_data()
    for seed in args.seeds:
        for run in args.runs:
            train_retrieval(run,seed,train,val,info,train_source,val_source,args.output,args.epochs,device)
            if not args.skip_downstream:
                downstream(run,seed,train,val,info,train_source,val_source,args.output,device,args.ranker_epochs)


if __name__=='__main__': main()
