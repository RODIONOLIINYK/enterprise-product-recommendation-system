"""Independent CPU replay, full-denominator metric replay and augmentation audit."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.catboost_comparison.standalone import load_statistical_data, sample_training_pairs
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle, prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache, make_rolling_slate_loader, rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import evaluate_rolling_slate_metrics, paired_rolling_intervals
from .model import load_checkpoint as load_ranker
from .tabular import numeric_pairs
from .train_tabular import array_digest, load_checkpoint as load_tabular
from .train_ranker import EVIDENCE, frozen_inputs


def verify_tabular():
    summary=json.loads((EVIDENCE/'tabular_summary.json').read_text())
    if frozen_inputs()!=summary['frozen_inputs']:raise AssertionError('Frozen source changed.')
    data,pids,_,_=load_statistical_data()
    indices=torch.arange(len(pids)).expand(len(data['val']['allowed']),-1)
    vr,vc=np.where(data['val']['allowed'].numpy())
    pick=np.random.default_rng(17).choice(len(vr),4096,replace=False)
    rr,cc=vr[pick],vc[pick]
    numeric=torch.from_numpy(numeric_pairs(data['val'],rr,cc))
    records=[]
    for name,arm in summary['arms'].items():
        for selection,record in arm['selected'].items():
            from pathlib import Path
            if file_digest(Path(record['checkpoint']))!=record['checkpoint_sha256']:
                raise AssertionError('MLP checkpoint digest changed.')
            model,lookup,_=load_tabular(record['checkpoint'],expected_provenance=arm['provenance'])
            model.eval()
            scores=torch.load(record['logits_pt'],map_location='cpu',weights_only=False)['logits']
            metrics=evaluate_rolling_slate_metrics(scores,data['val'],indices)
            if any(record['metrics'][k]!=v for k,v in metrics.items()):raise AssertionError('MLP metric replay mismatch.')
            with torch.no_grad():diff=float((model(numeric,torch.from_numpy(lookup[cc]))-scores[rr,cc]).abs().max())
            if diff>1e-4:raise AssertionError('Independent CPU MLP replay mismatch.')
            records.append({'arm':name,'selection':selection,'full_metrics_replayed':True,'cpu4096_max_abs_diff':diff})
    # Verify every rotating draw against its recorded digest and count distinct comparisons.
    covered=np.zeros(data['train']['allowed'].shape,dtype=bool)
    coverage=[]
    for draw in summary['arms']['mlp_rotating_negatives']['training_draws']:
        r,c,w=sample_training_pairs(data['train']['allowed'],data['train']['target'],seed=draw['seed'])
        if array_digest(r,c,w)!=draw['pairs_digest']:raise AssertionError('Augmentation draw mismatch.')
        if int(data['train']['target'].numpy()[r,c].sum())!=draw['positive_pairs']:
            raise AssertionError('Augmentation modified positives.')
        covered[r,c]=True
        coverage.append({'epoch':draw['epoch'],'distinct_training_pairs_so_far':int(covered.sum())})
    extra=paired_rolling_intervals(
        summary['arms']['mlp_fixed_unregularized']['selected']['ndcg']['predictions_npz'],
        summary['arms']['mlp_rotating_negatives']['selected']['ndcg']['predictions_npz'])
    tree_scores=torch.load('checkpoints/catboost-standalone-v1/runs/statistics52-seed42/ndcg_logits.pt',
                            map_location='cpu',weights_only=False)['logits']
    mask=data['val']['allowed']
    tree_logloss=float(F.binary_cross_entropy_with_logits(tree_scores[mask],data['val']['target'][mask].float()))
    if frozen_inputs()!=summary['frozen_inputs']:raise AssertionError('Frozen source changed.')
    atomic_json({'heldout_access':False,'selected_records':records,'rotating_negative_coverage':coverage,
                 'fixed_unregularized_to_rotating_95ci':extra,'catboost_val_binary_logloss':tree_logloss,
                 'all_frozen_inputs_unchanged':True},EVIDENCE/'tabular_verification.json',overwrite=False)


def verify_ranker():
    summary=json.loads((EVIDENCE/'ranker_summary.json').read_text())
    if frozen_inputs()!=summary['frozen_inputs']:raise AssertionError('Frozen source changed.')
    data,info,_,meta=prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data,info,meta)
    slates,manifest=prepare_or_load_rolling_slate_cache(data,info,meta)
    if manifest['split_cache_identity']!=summary['slate_cache_identities']:raise AssertionError('Slate identity changed.')
    x=rolling_ranker_batch_inputs(next(iter(make_rolling_slate_loader(slates['val'],batch_size=32))),
                                 data['train']['v'],torch.device('cpu'))
    records=[]
    for name,arm in summary['arms'].items():
        for selection in ('ndcg','discovery'):
            record=arm[selection]
            from pathlib import Path
            if file_digest(Path(record['checkpoint']))!=record['checkpoint_sha256']:
                raise AssertionError('Ranker checkpoint changed.')
            model,_=load_ranker(record['checkpoint'],expected_provenance=arm['provenance'],
                                expected_feature_contract_digest=meta['feature_schema_digest'])
            model.eval()
            scores=torch.load(record['logits_pt'],map_location='cpu',weights_only=False)['logits']
            metrics=evaluate_rolling_slate_metrics(scores,data['val'],slates['val']['candidate_indices'])
            if any(record['metrics'][k]!=v for k,v in metrics.items()):raise AssertionError('Ranker metric replay mismatch.')
            with torch.no_grad():diff=float((model(**x).logits-scores[:32]).abs().max())
            if diff>1e-4:raise AssertionError('Independent CPU ranker replay mismatch.')
            records.append({'arm':name,'selection':selection,'full_metrics_replayed':True,'cpu32_max_abs_diff':diff})
    if frozen_inputs()!=summary['frozen_inputs']:raise AssertionError('Frozen source changed.')
    atomic_json({'heldout_access':False,'selected_records':records,'all_frozen_inputs_unchanged':True},
                EVIDENCE/'ranker_verification.json',overwrite=False)


if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['tabular','ranker'],required=True)
    args=p.parse_args()
    verify_tabular() if args.stage=='tabular' else verify_ranker()
