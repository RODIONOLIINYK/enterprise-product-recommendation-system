"""Single-seed stage-wise dropout/weight-decay screen with fixed feature contracts."""

import argparse
import json
from pathlib import Path

import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.candidate_token_enrichment.train import (
    build_initial_arms, fit_single_enrichment_arm, snapshot_preserved_hashes,
)
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle, prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache, make_rolling_slate_loader, rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.scope import get_datasets_locs
from .model import VARIANTS, RegularizedRanker, load_checkpoint, save_checkpoint
from .tabular import categorical_lookup

ROOT = Path('checkpoints/regularization-signal-v1')
EVIDENCE = Path('experiments/regularization_signal_v1')


def frozen_inputs():
    """Freeze the standalone tree's inputs/model/results too; never retrain/change it."""
    tree_path = Path('experiments/catboost_standalone_v1/summary.json')
    tree = json.loads(tree_path.read_text())
    paths = [tree_path, Path('recommender/experiments/catboost_comparison/standalone.py'),
             Path('recommender/experiments/catboost_comparison/features.py'),
             Path('experiments/catboost_current_v1/historical_validation.json')]
    paths += [Path(r['checkpoint']) for r in tree['selected'].values()]
    return {'references_and_caches': snapshot_preserved_hashes(),
            'catboost_inputs_and_artifacts': {str(p): file_digest(p) for p in paths}}


def run():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before = frozen_inputs()
    data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    catalogue = load_product_catalogue(get_datasets_locs('items'))
    pids = info['product_ids']; lookup, vocab = categorical_lookup(pids, catalogue)
    _, source, _, _ = build_initial_arms(seed=42)
    source.eval()
    inputs = rolling_ranker_batch_inputs(next(iter(make_rolling_slate_loader(slates['val'], batch_size=16))),
                                        data['train']['v'], torch.device('cpu'))
    results, parity = {}, {}
    for variant, cfg in VARIANTS.items():
        model = RegularizedRanker.from_control(source, reg_variant=variant,
                                               categorical_lookup=lookup, product_ids=pids).eval()
        with torch.no_grad():
            parity[variant] = float((source(**inputs).logits - model(**inputs).logits).abs().max())
        if parity[variant] > 1e-5:
            raise AssertionError('Regularization changed the initial eval-mode function.')
        arm_meta = {**meta, 'data_identity': meta['data_identity'] + ':reg:' + json_digest(model.model_config())}
        results[variant] = fit_single_enrichment_arm(
            arm_name=variant, init_model=model, train_slate=slates['train'], val_slate=slates['val'],
            val_data=data['val'], product_vectors=data['train']['v'], meta=arm_meta,
            slate_manifest=manifest, artifact_root=ROOT, seed=42, epochs=10, batch_size=64,
            weight_decay=cfg['weight_decay'], device='mps' if torch.backends.mps.is_available() else 'cpu',
            checkpoint_writer=save_checkpoint, checkpoint_loader=load_checkpoint,
            experiment_name='regularization_signal_v1')
        atomic_json(results, EVIDENCE/'completed_ranker_arms.json')
    baseline = json.loads(Path('experiments/direct_feature_access_v1/summary.json').read_text())['baseline']
    intervals = {sel: {name: {
        'fresh_control_to_treatment': paired_rolling_intervals(results['control'][sel]['predictions_npz'], result[sel]['predictions_npz']),
        'saved_A_to_treatment': paired_rolling_intervals(baseline[sel]['predictions_npz'], result[sel]['predictions_npz'])}
        for name,result in results.items()} for sel in ('ndcg','discovery')}
    if frozen_inputs() != before:
        raise AssertionError('Frozen CatBoost/reference/cache artifacts changed.')
    atomic_json({'training_executed':True,'heldout_access':False,'promoted':False,
                 'seed':42,'epochs':10,'batch_size':64,'variants':VARIANTS,'init_parity':parity,
                 'arms':results,'baseline':baseline,'paired_bootstrap_95ci':intervals,
                 'frozen_inputs':before,'slate_cache_identities':manifest['split_cache_identity'],
                 'observed_source_digest':meta['observed_source_digest'],'categorical_vocabularies':vocab,
                 'limitations':['Single-seed repeated validation screen; no final-test evidence.',
                                'Stage1 tower weights/vectors are frozen; changing their dropout at inference has no effect.',
                                'Head/profile/attention half-dropout and zero-weight-decay are one-axis changes.',
                                'All-zero and direct52-all-zero are explicitly combined changes.',
                                'Source coherence and effective-dated catalogue status remain unverified.']},
                EVIDENCE/'ranker_summary.json',overwrite=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-training',action='store_true')
    if not p.parse_args().execute_training:p.error('Pass --execute-training.')
    run()
