"""Matched profile-token split followed by an isolated unpooled 64-event test."""

import argparse
import json
from pathlib import Path

import torch

from recommender.artifacts import atomic_json,file_digest
from recommender.experiments.candidate_token_enrichment.config import ARM_A_NAME
from recommender.experiments.candidate_token_enrichment.train import (
    build_initial_arms,fit_single_enrichment_arm,snapshot_preserved_hashes,
)
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle,prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,make_rolling_slate_loader,rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from .data import expand_history
from .model import SeparateProfileRanker,save_checkpoint,load_checkpoint

ROOT=Path('checkpoints/separate-profile-v1')
EVIDENCE=Path('experiments/separate_profile_v1')
ARMS=(('fused_two_profile_32','fused2',32),('split_text_attributes_32','split2',32),('split_text_attributes_64','split2',64))


def run():
    before=snapshot_preserved_hashes()
    previous_path=Path('experiments/pointwise_token_enrichment_v1/summary.json')
    previous=json.loads(previous_path.read_text())
    if before!=previous['preserved_hashes']:raise ValueError('Reference/rolling snapshot changed.')
    data,info,features,meta=prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data,info,meta)
    slates,manifest=prepare_or_load_rolling_slate_cache(data,info,meta)
    if previous['slate_cache_identities']!=manifest['split_cache_identity']:raise ValueError('Candidate slates changed.')
    vectors=data['train']['v'];_,source,_,_=build_initial_arms(seed=42)
    # New core extension hooks must preserve the saved baseline's default calculation.
    source.eval();batch=next(iter(make_rolling_slate_loader(slates['val'],batch_size=32,shuffle=False)))
    x=rolling_ranker_batch_inputs(batch,vectors,torch.device('cpu'))
    clone=SeparateProfileRanker.from_control(source,profile_layout='fused1',history_length=32).eval()
    with torch.no_grad():torch.testing.assert_close(source(**x).logits,clone(**x).logits,rtol=0,atol=0)
    results={};history_diagnostics=None;long_slates=None
    for name,layout,length in ARMS:
        arm_slates=slates;arm_meta=dict(meta)
        if length==64:
            long_slates,history_diagnostics,history_sha=expand_history(slates,meta['observed_source_digest'])
            arm_slates=long_slates
            arm_meta['data_identity']=meta['data_identity']+':history64:'+history_sha
        initial=SeparateProfileRanker.from_control(source,profile_layout=layout,history_length=length)
        result=fit_single_enrichment_arm(
            arm_name=name,init_model=initial,train_slate=arm_slates['train'],val_slate=arm_slates['val'],
            val_data=data['val'],product_vectors=vectors,meta=arm_meta,slate_manifest=manifest,
            artifact_root=ROOT,seed=42,epochs=10,batch_size=64,device='mps' if torch.backends.mps.is_available() else 'cpu',
            checkpoint_writer=save_checkpoint,checkpoint_loader=load_checkpoint,experiment_name='separate_profile_v1',
        )
        results[name]=result
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        atomic_json(results,EVIDENCE/'completed_arms.json',overwrite=True)
    intervals={}
    for selection in ('ndcg','discovery'):
        entries={**{ARM_A_NAME:previous['arms'][ARM_A_NAME]},**results}
        intervals[selection]={}
        for first,second in ((ARM_A_NAME,ARMS[1][0]),(ARMS[0][0],ARMS[1][0]),(ARMS[1][0],ARMS[2][0])):
            intervals[selection][first+'_to_'+second]=paired_rolling_intervals(entries[first][selection]['predictions_npz'],entries[second][selection]['predictions_npz'])
    if snapshot_preserved_hashes()!=before:raise AssertionError('Existing snapshots changed.')
    summary={'training_executed':True,'heldout_access':False,'promoted':False,'seed':42,'epochs':10,
             'statistics_policy':meta['statistics_policy'],'observed_source_digest':meta['observed_source_digest'],
             'slate_cache_identities':manifest['split_cache_identity'],'history_diagnostics':history_diagnostics,
             'baseline':previous['arms'][ARM_A_NAME],'arms':results,'paired_bootstrap_95ci':intervals,
             'preserved_hashes':before,'prior_summary_sha256':file_digest(previous_path),
             'default_extension_hook_init_parity':0.0,
             'limits':['Single training seed; bootstrap is conditional on fitted weights.',
                       'Fused2 controls token count/statistics mapping; split2 also removes the fused profile projection.',
                       'The64 variant extends the memory; it does not pool older events into one vector.',
                       'Events are chronological purchase rows, not32/64 distinct baskets.']}
    atomic_json(summary,EVIDENCE/'summary.json',overwrite=False)
    print(json.dumps({name:{s:r[s]['metrics'] for s in ('ndcg','discovery')} for name,r in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-training',action='store_true')
    if not parser.parse_args().execute_training:parser.error('Pass --execute-training to run the three controlled arms.')
    run()
