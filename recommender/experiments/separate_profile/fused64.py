"""Isolate64-event history on the original fused-profile architecture as well."""
import argparse
import json
import torch
from recommender.artifacts import atomic_json
from recommender.experiments.candidate_token_enrichment.train import build_initial_arms,fit_single_enrichment_arm,snapshot_preserved_hashes
from recommender.experiments.rolling_candidate_v2.data import prepare_or_load_rolling_feature_bundle,prepare_or_load_rolling_selector_scores,prepare_or_load_rolling_slate_cache
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from .data import expand_history
from .model import SeparateProfileRanker,save_checkpoint,load_checkpoint
from .train import ROOT,EVIDENCE


def run():
    before=snapshot_preserved_hashes()
    previous=json.loads((EVIDENCE/'summary.json').read_text())
    if previous['preserved_hashes']!=before:raise ValueError('Input snapshot changed.')
    data,info,_,meta=prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data,info,meta)
    slates,manifest=prepare_or_load_rolling_slate_cache(data,info,meta)
    extended,diagnostics,digest=expand_history(slates,meta['observed_source_digest'])
    _,source,_,_=build_initial_arms(seed=42)
    model=SeparateProfileRanker.from_control(source,profile_layout='fused1',history_length=64)
    long_meta={**meta,'data_identity':meta['data_identity']+':history64:'+digest}
    result=fit_single_enrichment_arm(
        arm_name='fused_one_profile_64',init_model=model,train_slate=extended['train'],val_slate=extended['val'],
        val_data=data['val'],product_vectors=data['train']['v'],meta=long_meta,slate_manifest=manifest,
        artifact_root=ROOT,seed=42,epochs=10,batch_size=64,device='mps' if torch.backends.mps.is_available() else 'cpu',
        checkpoint_writer=save_checkpoint,checkpoint_loader=load_checkpoint,experiment_name='separate_profile_v1',
    )
    intervals={s:{
        'A32_to_A64':paired_rolling_intervals(previous['baseline'][s]['predictions_npz'],result[s]['predictions_npz']),
        'A64_to_split64':paired_rolling_intervals(result[s]['predictions_npz'],previous['arms']['split_text_attributes_64'][s]['predictions_npz']),
        'A32_to_split64':paired_rolling_intervals(previous['baseline'][s]['predictions_npz'],previous['arms']['split_text_attributes_64'][s]['predictions_npz']),
    } for s in ('ndcg','discovery')}
    if snapshot_preserved_hashes()!=before:raise AssertionError('Protected artifacts changed.')
    atomic_json({'training_executed':True,'heldout_access':False,'promoted':False,
                 'result':result,'paired_bootstrap_95ci':intervals,'history_diagnostics':diagnostics,
                 'preserved_hashes':before},EVIDENCE/'fused64_control.json',overwrite=False)
    print(json.dumps({s:result[s]['metrics'] for s in ('ndcg','discovery')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-training',action='store_true')
    if not p.parse_args().execute_training:p.error('Pass --execute-training.')
    run()
