"""Full-pool MLP with exactly the frozen standalone CatBoost information inputs.

Same 49 numeric values, 3 categorical fields, seed42 sampled pairs and importance
weights, and binary Logloss. Categorical encoders are necessarily model-specific.
A separate arm rotates only negative samples while preserving every positive and
full-pool importance weighting; it does not fabricate purchase labels.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.catboost_comparison.standalone import (
    FEATURE_NAMES, load_statistical_data, sample_training_pairs, statistical_frame,
)
from recommender.experiments.rolling_candidate_v2.evaluate import (
    evaluate_rolling_slate_metrics, paired_rolling_intervals,
)
from recommender.shared_data import atomic_torch_save
from .tabular import StatisticalMLP, categorical_lookup, numeric_pairs, weighted_binary_loss
from .train_ranker import EVIDENCE, ROOT, frozen_inputs

VARIANTS = {
    'mlp_fixed_regularized': {'dropout':[.2,.1], 'weight_decay':1e-4, 'resample_negatives':False},
    'mlp_fixed_unregularized': {'dropout':[0.,0.], 'weight_decay':0., 'resample_negatives':False},
    'mlp_rotating_negatives': {'dropout':[0.,0.], 'weight_decay':0., 'resample_negatives':True},
}


def array_digest(*arrays):
    digest=hashlib.sha256()
    for arr in arrays:
        arr=np.ascontiguousarray(arr)
        digest.update(str(arr.dtype).encode());digest.update(str(arr.shape).encode());digest.update(memoryview(arr))
    return digest.hexdigest()


@torch.no_grad()
def predict(model, split, lookup, *, device, batch_size=65536):
    model.eval()
    rows,columns=np.where(split['allowed'].numpy())
    dense=torch.full(split['allowed'].shape,-torch.inf)
    total_loss=0.
    for start in range(0,len(rows),batch_size):
        rr,cc=rows[start:start+batch_size],columns[start:start+batch_size]
        numeric=torch.from_numpy(numeric_pairs(split,rr,cc)).to(device)
        codes=torch.from_numpy(lookup[cc]).to(device)
        logits=model(numeric,codes)
        labels=split['target'][rr,cc].to(device,dtype=torch.float32)
        total_loss+=float(F.binary_cross_entropy_with_logits(logits,labels,reduction='sum'))
        dense[rr,cc]=logits.cpu()
    return dense,total_loss/len(rows)


def save_checkpoint(path, model, *, epoch, provenance, lookup, metrics):
    atomic_torch_save({'checkpoint_format_version':1,'model_class':'StatisticalMLP',
                       'contract':model.contract(),'provenance':provenance,'epoch':epoch,
                       'lookup':lookup,'state_dict':model.state_dict(),'metrics':metrics},path)


def load_checkpoint(path, *, expected_provenance=None):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('checkpoint_format_version')!=1 or payload.get('model_class')!='StatisticalMLP':
        raise ValueError('Unexpected standalone MLP checkpoint.')
    cfg=payload['contract'];prov=payload['provenance'];lookup=np.asarray(payload['lookup'])
    if expected_provenance is not None and prov!=expected_provenance:
        raise ValueError('Statistical MLP source/provenance mismatch.')
    if (prov.get('feature_names')!=list(FEATURE_NAMES) or prov.get('statistics_policy')!='rolling_observed_v2'
            or prov.get('lookup_digest')!=array_digest(lookup)
            or prov.get('heldout_access') is not False):
        raise ValueError('Statistical MLP input contract mismatch.')
    model=StatisticalMLP(cfg['category_sizes'],dropout=cfg['dropout'])
    if cfg!=model.contract():raise ValueError('Statistical MLP architecture/feature contract mismatch.')
    model.load_state_dict(payload['state_dict'],strict=True)
    return model,lookup,payload


def run(device_name="cpu"):
    torch.set_num_threads(2)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    before=frozen_inputs()
    tree=json.loads(Path('experiments/catboost_standalone_v1/summary.json').read_text())
    data,pids,catalogue,manifest=load_statistical_data()
    if (tree['provenance']['feature_names']!=list(FEATURE_NAMES)
            or tree['provenance']['feature_bundle_identity']!=manifest['identity']
            or tree['provenance']['observed_source_digest']!=manifest['observed_source_digest']):
        raise ValueError('Frozen tree feature identity differs from MLP inputs.')
    for path,digest in tree['provenance']['source_hashes'].items():
        if file_digest(Path(path))!=digest:raise ValueError('CatBoost statistical/query source changed.')
    lookup,vocab=categorical_lookup(pids,catalogue)
    rows,columns,weights=sample_training_pairs(data['train']['allowed'],data['train']['target'],seed=42)
    base_draw=(rows,columns,weights)
    labels=data['train']['target'].numpy()[rows,columns].astype(np.float32)
    if (len(rows)!=tree['provenance']['training_pairs_sampled']
            or labels.sum()!=tree['provenance']['training_positive_pairs_sampled']
            or int(data['val']['allowed'].sum())!=tree['provenance']['validation_eligible_pairs']):
        raise ValueError('Tree/MLP training sample or validation population mismatch.')
    # Direct value parity against the actual CatBoost frame builder, not a reworded schema.
    check=np.random.default_rng(42).choice(len(rows),4096,replace=False)
    frame=statistical_frame(data['train']['features'],data['train']['profile'],pids,catalogue,rows[check],columns[check])
    np.testing.assert_array_equal(numeric_pairs(data['train'],rows[check],columns[check]),frame.iloc[:,3:].to_numpy())
    for i,name in enumerate(('product_id','product_category','business_line')):
        assert [vocab[name][int(code)] for code in lookup[columns[check],i]]==frame[name].tolist()
    train_prior=float((data['train']['allowed'] & data['train']['target']).sum()/data['train']['allowed'].sum())
    torch.manual_seed(42)
    initial=StatisticalMLP([len(v) for v in vocab.values()],initial_positive_fraction=train_prior)
    device=torch.device(device_name)
    indices=torch.arange(len(pids)).expand(len(data['val']['allowed']),-1)
    results={}
    initial_pairs_digest=array_digest(*base_draw)
    for variant,settings in VARIANTS.items():
        out=ROOT/'runs'/f'{variant}-seed42'
        if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
        out.mkdir(parents=True)
        model=copy.deepcopy(initial)
        model.dropout_rates=tuple(settings['dropout'])
        for module,rate in zip([m for m in model.head.modules() if isinstance(m,torch.nn.Dropout)],settings['dropout']):
            module.p=rate
        with torch.no_grad():
            initial.eval();model.eval()
            check_numeric=torch.from_numpy(numeric_pairs(data['train'],rows[:32],columns[:32]))
            check_codes=torch.from_numpy(lookup[columns[:32]])
            parity=float((initial(check_numeric,check_codes)-model(check_numeric,check_codes)).abs().max())
        if parity!=0:raise AssertionError('MLP arms have different initial functions.')
        provenance={'feature_names':list(FEATURE_NAMES),'product_ids':pids,'lookup_digest':array_digest(lookup),
                    'statistics_policy':'rolling_observed_v2','feature_bundle_identity':manifest['identity'],
                    'observed_source_digest':manifest['observed_source_digest'],'source_hashes':tree['provenance']['source_hashes'],
                    'initial_training_pairs_digest':initial_pairs_digest,'training_pairs':len(rows),
                    'validation_eligible_pairs':int(data['val']['allowed'].sum()),'seed':42,
                    'epochs':10,'batch_size':4096,'lr':.001,'eta_min':1e-5,'settings':settings,'device':str(device),
                    'objective':'same sampled full-pool binary Logloss as frozen CatBoost; global importance normalizer',
                    'initial_positive_fraction':train_prior,'initial_logit_parity':parity,
                    'heldout_access':False,'promoted':False,'metadata_policy':'snapshot_unverified'}
        atomic_json({**provenance,'status':'running'},out/'run.json',overwrite=False)
        torch.manual_seed(42)
        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=settings['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10,eta_min=1e-5)
        history=[];draws=[];best={'ndcg':None,'discovery':None,'logloss':None}
        started=time.perf_counter()
        for epoch in range(1,11):
            if settings['resample_negatives'] and epoch>1:
                rr,cc,ww=sample_training_pairs(data['train']['allowed'],data['train']['target'],seed=42+epoch-1)
            else:rr,cc,ww=base_draw
            yy=data['train']['target'].numpy()[rr,cc].astype(np.float32)
            if len(rr)!=len(rows) or yy.sum()!=labels.sum():raise AssertionError('Augmentation changed positive population.')
            draws.append({'epoch':epoch,'seed':42+epoch-1 if settings['resample_negatives'] else 42,
                          'pairs_digest':array_digest(rr,cc,ww),'positive_pairs':int(yy.sum()),
                          'sum_importance_weights':float(ww.astype(np.float64).sum())})
            # Materialize only the sampled training rows, not a broad dense all-product tensor.
            numeric=torch.from_numpy(numeric_pairs(data['train'],rr,cc)).to(device)
            categories=torch.from_numpy(lookup[cc]).to(device)
            target=torch.from_numpy(yy).to(device);weight=torch.from_numpy(ww).to(device)
            mean_weight=float(ww.astype(np.float64).mean())
            order=torch.from_numpy(np.random.default_rng(1042+epoch).permutation(len(rr))).to(device)
            model.train();loss_sum=0.
            for start in range(0,len(rr),4096):
                selected=order[start:start+4096]
                optimizer.zero_grad(set_to_none=True)
                loss=weighted_binary_loss(model(numeric[selected],categories[selected]),target[selected],weight[selected],
                                          global_mean_weight=mean_weight)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite tabular MLP loss.')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();loss_sum+=float(loss.detach())*len(selected)
            del numeric,categories,target,weight,order
            scheduler.step()
            scores,val_loss=predict(model,data['val'],lookup,device=device)
            metrics=evaluate_rolling_slate_metrics(scores,data['val'],indices);metrics['val_binary_logloss']=val_loss
            qualified=metrics['recall@10']>=.7558833119851873 and metrics['ndcg@10']>=.5918928743676547
            history.append({'epoch':epoch,'train_binary_logloss_online':loss_sum/len(rr),'metrics':metrics,'qualified':qualified})
            keys={'ndcg':(metrics['ndcg@10'],metrics['recall@10'],metrics['new_hits@10']),
                  'discovery':(int(qualified),metrics['new_hits@10'],metrics['ndcg@10']),
                  'logloss':(-val_loss,metrics['ndcg@10'])}
            for selection,key in keys.items():
                if best[selection] is None or key>best[selection]['key']:
                    path=out/f'best_{selection}.pt'
                    save_checkpoint(path,model,epoch=epoch,provenance=provenance,lookup=lookup,metrics=metrics)
                    best[selection]={'key':key,'epoch':epoch,'checkpoint':str(path),'metrics':metrics,'qualified':qualified}
            print(f'[statistical_mlp/{variant}] ep={epoch:02d} train={loss_sum/len(rr):.5f} val={val_loss:.5f} '
                  f'rec10={metrics["recall@10"]:.5f} ndcg10={metrics["ndcg@10"]:.5f} new10={metrics["new_hits@10"]}',flush=True)
        elapsed=time.perf_counter()-started
        selected_results={}
        for selection,record in best.items():
            loaded,reloaded_lookup,payload=load_checkpoint(record['checkpoint'],expected_provenance=provenance)
            loaded.to(device)
            replay,val_loss=predict(loaded,data['val'],reloaded_lookup,device=device)
            npz=out/f'{selection}_predictions.npz'
            metrics=evaluate_rolling_slate_metrics(replay,data['val'],indices,save_path=npz)
            metrics['val_binary_logloss']=val_loss
            if metrics!=record['metrics']:raise AssertionError('Standalone MLP replay mismatch.')
            logits_path=out/f'{selection}_logits.pt';atomic_torch_save({'logits':replay},logits_path)
            selected_results[selection]={**{k:v for k,v in record.items() if k!='key'},
                'checkpoint_sha256':file_digest(Path(record['checkpoint'])),'predictions_npz':str(npz),
                'logits_pt':str(logits_path),'strict_metric_replay':True,
                'tree_to_mlp_95ci':paired_rolling_intervals(tree['selected']['ndcg' if selection=='logloss' else selection]['predictions_npz'],npz)}
        results[variant]={'settings':settings,'provenance':provenance,'parameters':sum(p.numel() for p in model.parameters()),
                          'training_seconds':elapsed,'history':history,'training_draws':draws,'selected':selected_results}
        atomic_json(results[variant],out/'result.json',overwrite=False)
        atomic_json({**provenance,'status':'complete','selected':selected_results},out/'run.json')
        atomic_json(results,EVIDENCE/'completed_tabular_arms.json')
        del model,optimizer,loaded
    if frozen_inputs()!=before:raise AssertionError('Frozen tree/input/reference artifacts changed.')
    intervals={selection:{name:paired_rolling_intervals(results['mlp_fixed_regularized']['selected'][selection]['predictions_npz'],
                                                       result['selected'][selection]['predictions_npz'])
                         for name,result in results.items()} for selection in ('ndcg','discovery')}
    atomic_json({'training_executed':True,'heldout_access':False,'promoted':False,'variants':VARIANTS,'arms':results,
                 'catboost_baseline':tree['selected'],'paired_bootstrap_95ci':intervals,'frozen_inputs':before,
                 'numeric_and_categorical_value_parity_checked_rows':4096,'initial_training_pairs_digest':initial_pairs_digest,
                 'feature_names':list(FEATURE_NAMES),'categorical_vocabularies':vocab,
                 'limitations':['Three raw categorical fields are encoded with learned embeddings in the MLP, tree categorical encoding differs.',
                                'Same raw numeric values/fields/pair draw/objective; optimizer and learning schedule are model-specific.',
                                'Uniform negative resampling adds comparisons, not new positive preference evidence.',
                                'Single-seed ten-epoch screen; no claim of optimal MLP tuning.',
                                'Catalogue metadata is snapshot_unverified; no synthetic purchases or future labels are used.']},
                EVIDENCE/'tabular_summary.json',overwrite=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-training',action='store_true')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=p.parse_args()
    if not args.execute_training:p.error('Pass --execute-training.')
    run(args.device)
