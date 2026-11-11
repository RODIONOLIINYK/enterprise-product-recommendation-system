"""Rebuild E1–E5 aggregate evidence from completed, local development runs."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from recommender.experimental_signals import TransitionStatistics, transition_features
from recommender.experiment_signals import load_inputs
from recommender.shared_data import load_development_purchases
from recommender.train_ranker import checkpoint_digest
from recommender.train_shared_ranker import paired_customer_bootstrap

ROOT = Path('experiments/e1_e5')
SIGNALS = Path('checkpoints/e1-e5/signals-v1')
RETRIEVAL = Path('checkpoints/e1-e5/retrieval-v1')


def describe(values):
    values = [float(value) for value in values]
    return {'n': len(values), 'mean': float(np.mean(values)),
            'sample_sd': float(np.std(values, ddof=1)) if len(values)>1 else None,
            'values': values}


def collect():
    groups, runs = {}, []
    for root, stage in ((SIGNALS, 'stage2_signals'), (RETRIEVAL, 'stage1_retrieval')):
        for path in sorted(root.glob('*/result.json')):
            result = json.loads(path.read_text())
            settings=result['settings']; name=settings.get('mode',settings.get('run'))
            row={'name':name,'seed':settings['seed'],'stage':stage,'path':str(path),'result':result}
            runs.append(row)
            group=groups.setdefault((stage,name),[])
            group.append(result)
    aggregates=[]
    for (stage,name), results in groups.items():
        entry={'stage':stage,'name':name,'seeds':[r['settings']['seed'] for r in results]}
        for key in ('val_recall@10','val_ndcg@10','new_hits@10','new_recall@10','new_hits@25','new_hits@50','stage1_new_recall@25','stage1_new_recall@50'):
            if all(key in r['metrics'] for r in results): entry[key]=describe([r['metrics'][key] for r in results])
        for key in ('training_seconds','parameters','cache_bytes_per_customer','product_cache_bytes'):
            if all(key in r for r in results): entry[key]=describe([r[key] for r in results])
        if stage=='stage2_signals':
            entry['conditional_new_recall@10']=describe([r['diagnostics']['conditional_new_recall@10'] for r in results])
        else:
            downstream=[]
            for r in results:
                p=RETRIEVAL/f"{name}-seed{r['settings']['seed']}"/'downstream.json'
                if p.exists(): downstream.append(json.loads(p.read_text()))
            entry['downstream']={key:describe([r['metrics'][key] for r in downstream])
                                 for key in ('val_recall@10','val_ndcg@10','new_hits@10','new_recall@10')} if downstream else {}
            if downstream:
                entry['downstream']['conditional_new_recall@10']=describe([r['diagnostics']['conditional_new_recall@10'] for r in downstream])
                entry['downstream']['new_targets_in_slate']=describe([r['diagnostics']['new_targets_in_slate'] for r in downstream])
                entry['downstream']['training_seconds']=describe([r['training_seconds'] for r in downstream])
                entry['downstream']['guardrail_passes']=sum(r.get('qualified_recall_guardrail',True) for r in downstream)
        aggregates.append(entry)
    paired={}
    for row in runs:
        seed=row['seed']; name=row['name']
        if row['stage']=='stage2_signals':
            control=SIGNALS/f'control-seed{seed}'/'predictions.npz'
            candidate=Path(row['path']).parent/'predictions.npz'
            if name!='control' and control.exists() and candidate.exists():
                paired[f'{name}-seed{seed}']=paired_customer_bootstrap(control,candidate)
        else:
            control=RETRIEVAL/f'S0-seed{seed}'/'downstream'/f'control-seed{seed}'/'predictions.npz'
            candidate=Path(row['path']).parent/'downstream'/f'control-seed{seed}'/'predictions.npz'
            if name!='S0' and control.exists() and candidate.exists():
                paired[f'downstream-{name}-seed{seed}']=paired_customer_bootstrap(control,candidate)
    return {'runs':runs,'aggregates':aggregates,'paired_customer_bootstrap':paired}


def provenance():
    audit=json.loads(Path('audit/artifacts.json').read_text())
    # Audit artifact inventory uses a nested or flat mapping depending on audit revision.
    def entries(value):
        for key,item in value.items():
            if isinstance(item,dict):
                if 'sha256' in item and Path(key).exists(): yield key,item
                else: yield from entries(item)
    preservation=[]
    for path,entry in entries(audit):
        if path.endswith('.pt'):
            actual=checkpoint_digest(Path(path))
            preservation.append({'path':path,'expected':entry['sha256'],'actual':actual,'unchanged':actual==entry['sha256']})
    if not preservation or not all(r['unchanged'] for r in preservation):
        raise ValueError('Original artifact preservation check failed')
    dev=load_development_purchases('data/interim/cleaned_purchases.csv')
    digest=hashlib.sha256(pd.util.hash_pandas_object(dev,index=True).values.tobytes()).hexdigest()
    return {'preservation':preservation,'development_source_digest':digest,
            'heldout_policy':'Date gate discards held-out records before pandas materialization; no held-out fields/statistics/targets used.',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},
            'implementation_sha256':{str(p):checkpoint_digest(p) for p in sorted(Path('recommender').glob('experiment*.py'))}}


def signal_resources():
    train,val,manifest=load_inputs()
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    table=TransitionStatistics(purchases)
    date=pd.Timestamp('2025-10-01'); table.advance(date)
    candidates=manifest['product_ids']
    cid=val['customer_ids'][0]
    timings=[]
    for _ in range(35):
        started=time.perf_counter(); table.score(cid,date,candidates); timings.append((time.perf_counter()-started)*1000)
    output={'transition_table_dense_count_bytes':table.ab.nbytes+table.a.nbytes+table.b.nbytes,
            'table_products':len(table.ids),'table_training_transitions':table.total,
            'transition_feature_cache_development_50_candidates_bytes':(len(train['customer_ids'])+len(val['customer_ids']))*50*4,
            'transition_purchase_refresh_all_catalogue_median_ms':float(np.median(timings[5:])),
            'transition_purchase_refresh_all_catalogue_p95_ms':float(np.quantile(timings[5:],.95)),
            'transition_full_catalogue_customer_cache_bytes':len(candidates)*4,
            'stage1_full_catalogue_product_vectors_bytes':len(candidates)*128*4,
            'active_candidate_order':len(candidates)}
    return output


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    summary=collect(); summary['provenance']=provenance(); summary['resources']=signal_resources()
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[]
    for entry in summary['aggregates']:
        metrics=entry.get('downstream',entry) if entry['stage']=='stage1_retrieval' else entry
        if not metrics: continue
        def cell(key, scale=1, digits=3):
            value=metrics[key]
            mean=f"{value['mean']*scale:.{digits}f}"
            return mean+' ± '+f"{value['sample_sd']*scale:.{digits}f}" if value['sample_sd'] is not None else mean+' (single seed)'
        rows.append(f"| {entry['name']} | {len(entry['seeds'])} | {cell('val_recall@10',100)}% | {cell('val_ndcg@10',1,6)} | {cell('new_hits@10',1,2)} | {cell('conditional_new_recall@10',100)}% |")
    (ROOT/'comparison_table.md').write_text('| Configuration | Seeds | Recall@10 | NDCG@10 | New hits@10 / 1,857 | Conditional new recall@10 |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n')
    print(f"Summarized {len(summary['runs'])} fitted retrieval/signal models",flush=True)


if __name__=='__main__': main()
