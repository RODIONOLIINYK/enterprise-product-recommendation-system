"""Validation-only factorial screen of direct profile and categorical inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recommender.artifacts import atomic_json, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.candidate_token_enrichment.train import (
    build_initial_arms, fit_single_enrichment_arm, snapshot_preserved_hashes,
)
from recommender.experiments.catboost_comparison.features import categorical_frame
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle, prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache, make_rolling_slate_loader, rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.scope import get_datasets_locs
from .model import CATEGORY_FIELDS, VARIANTS, DirectFeatureRanker, load_checkpoint, save_checkpoint

ROOT = Path('checkpoints/direct-feature-access-v1')
EVIDENCE = Path('experiments/direct_feature_access_v1')


def run():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before = snapshot_preserved_hashes()
    previous = json.loads(Path('experiments/ranker_capacity_v1/summary.json').read_text())
    if before != previous['preserved_hashes']:
        raise ValueError('Protected reference/cache snapshot changed.')
    data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    catalogue = load_product_catalogue(get_datasets_locs('items'))
    pids = info['product_ids']
    frame = categorical_frame(pids, catalogue, np.arange(len(pids)))
    vocabularies = {name: sorted(frame[name].astype(str).unique()) for name in CATEGORY_FIELDS}
    lookup = torch.tensor(np.stack([
        frame[name].astype(str).map({v: i for i, v in enumerate(vocabularies[name])}).to_numpy()
        for name in CATEGORY_FIELDS], -1), dtype=torch.long)
    _, source, _, _ = build_initial_arms(seed=42)
    source.eval()
    inputs = rolling_ranker_batch_inputs(next(iter(make_rolling_slate_loader(slates['val'], batch_size=16))),
                                        data['train']['v'], torch.device('cpu'))
    results, parity = {}, {}
    for variant in VARIANTS:
        model = DirectFeatureRanker.from_control(source, variant=variant,
                                                categorical_lookup=lookup, product_ids=pids).eval()
        with torch.no_grad():
            parity[variant] = float((source(**inputs).logits - model(**inputs).logits).abs().max())
        if parity[variant] > 1e-5:
            raise AssertionError('Initial function changed beyond float rounding.')
        arm_meta = {**meta, 'data_identity': meta['data_identity'] + ':direct:' + json_digest(model.model_config())}
        results[variant] = fit_single_enrichment_arm(
            arm_name=variant, init_model=model, train_slate=slates['train'], val_slate=slates['val'],
            val_data=data['val'], product_vectors=data['train']['v'], meta=arm_meta,
            slate_manifest=manifest, artifact_root=ROOT, seed=42, epochs=10, batch_size=64,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            checkpoint_writer=save_checkpoint, checkpoint_loader=load_checkpoint,
            experiment_name='direct_feature_access_v1')
        atomic_json(results, EVIDENCE / 'completed_arms.json')
    intervals = {selection: {name: {
        'fresh_control_to_treatment': paired_rolling_intervals(results['control'][selection]['predictions_npz'],
                                                             arm[selection]['predictions_npz']),
        'saved_A_to_treatment': paired_rolling_intervals(previous['baseline'][selection]['predictions_npz'],
                                                       arm[selection]['predictions_npz']),
    } for name, arm in results.items()} for selection in ('ndcg', 'discovery')}
    if snapshot_preserved_hashes() != before:
        raise AssertionError('Protected artifact changed.')
    atomic_json({
        'training_executed': True, 'heldout_access': False, 'promoted': False,
        'seed': 42, 'epochs': 10, 'batch_size': 64, 'init_max_abs_diff': parity,
        'arms': results, 'baseline': previous['baseline'], 'paired_bootstrap_95ci': intervals,
        'preserved_hashes': before, 'observed_source_digest': meta['observed_source_digest'],
        'slate_cache_identities': manifest['split_cache_identity'], 'categorical_vocabularies': vocabularies,
        'limitations': ['Single-seed exploratory validation; repeated validation use is not final-test evidence.',
                       'Catalogue metadata is snapshot_unverified; frozen upstream vector freshness remains unverified.',
                       'Same 100-candidate slates, optimizer/loss/ten-epoch schedule; only direct head input paths change.',
                       'New first-layer columns start at zero; categorical embedding dimensions are 16/8/8.'],
    }, EVIDENCE / 'summary.json', overwrite=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true')
    if not parser.parse_args().execute_training:
        parser.error('Pass --execute-training.')
    run()
