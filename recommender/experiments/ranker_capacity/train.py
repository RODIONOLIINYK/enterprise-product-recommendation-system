"""Run one-axis capacity comparisons on fixed rolling candidates, seed 42."""

import argparse
import json
from pathlib import Path

import torch

from recommender.artifacts import atomic_json, json_digest
from recommender.experiments.candidate_token_enrichment.benchmark import run_stage2_latency_benchmark
from recommender.experiments.candidate_token_enrichment.config import ARM_A_NAME
from recommender.experiments.candidate_token_enrichment.model import load_enrichment_ranker_checkpoint
from recommender.experiments.candidate_token_enrichment.train import (
    build_initial_arms, fit_single_enrichment_arm, snapshot_preserved_hashes,
)
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader, prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores, prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from .model import VARIANTS, CapacityRanker, load_checkpoint, save_checkpoint

ROOT = Path('checkpoints/ranker-capacity-v1')
EVIDENCE = Path('experiments/ranker_capacity_v1')


def run():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before = snapshot_preserved_hashes()
    previous = json.loads(Path('experiments/candidate_token_enrichment_v1/summary.json').read_text())
    if previous['preserved_hashes'] != before:
        raise ValueError('Reference/cache snapshot changed.')
    data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    _, source, _, _ = build_initial_arms(seed=42)
    source.eval()
    batch = next(iter(make_rolling_slate_loader(slates['val'], batch_size=16, shuffle=False)))
    inputs = rolling_ranker_batch_inputs(batch, data['train']['v'], torch.device('cpu'))
    control = CapacityRanker.from_control(source, variant='control32').eval()
    with torch.no_grad():
        init_diff = float((source(**inputs).logits - control(**inputs).logits).abs().max())
    if init_diff != 0:
        raise AssertionError('Capacity control does not preserve original initialization.')
    results = {}
    for variant in VARIANTS:
        model = CapacityRanker.from_control(source, variant=variant)
        config_digest = json_digest(model.model_config())
        arm_meta = {**meta, 'data_identity': meta['data_identity'] + ':capacity:' + config_digest}
        results[variant] = fit_single_enrichment_arm(
            arm_name=variant, init_model=model, train_slate=slates['train'], val_slate=slates['val'],
            val_data=data['val'], product_vectors=data['train']['v'], meta=arm_meta,
            slate_manifest=manifest, artifact_root=ROOT, seed=42, epochs=10, batch_size=64,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            checkpoint_writer=save_checkpoint, checkpoint_loader=load_checkpoint,
            experiment_name='ranker_capacity_v1',
        )
        atomic_json(results, EVIDENCE / 'completed_arms.json')
    baseline = previous['arms'][ARM_A_NAME]
    intervals = {selection: {
        variant: {
            'saved_A_to_treatment': paired_rolling_intervals(baseline[selection]['predictions_npz'], result[selection]['predictions_npz']),
            'fresh_control_to_treatment': paired_rolling_intervals(results['control32'][selection]['predictions_npz'], result[selection]['predictions_npz']),
        } for variant, result in results.items()
    } for selection in ('ndcg', 'discovery')}
    fitted = {name: load_checkpoint(result['ndcg']['checkpoint'])[0] for name, result in results.items()}
    fitted['saved_A'] = load_enrichment_ranker_checkpoint(baseline['ndcg']['checkpoint'])[0]
    benchmark = run_stage2_latency_benchmark(fitted, slates['val'], data['train']['v'],
                                           evidence_root=EVIDENCE, warmup=5, repeats=20, save=True)
    if snapshot_preserved_hashes() != before:
        raise AssertionError('Protected reference/cache files changed.')
    summary = {
        'training_executed': True, 'heldout_access': False, 'promoted': False,
        'variants': VARIANTS, 'seed': 42, 'epochs': 10, 'batch_size': 64,
        'control_init_max_abs_diff': init_diff, 'baseline': baseline, 'arms': results,
        'paired_bootstrap_95ci': intervals, 'preserved_hashes': before,
        'observed_source_digest': meta['observed_source_digest'],
        'slate_cache_identities': manifest['split_cache_identity'], 'benchmark': benchmark,
        'limitations': ['Single training seed; MPS training is not bitwise deterministic.',
                       'Changed-shape layers use independent fresh initialization; unchanged weights are copied exactly.',
                       'Same ten-epoch budget; this does not establish optimal tuning of larger models.',
                       'Snapshot metadata and frozen upstream vectors retain documented source-coherence limits.'],
    }
    atomic_json(summary, EVIDENCE / 'summary.json', overwrite=False)
    print(json.dumps({name: {s: result[s]['metrics'] for s in ('ndcg', 'discovery')}
                      for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true')
    if not parser.parse_args().execute_training:
        parser.error('Pass --execute-training.')
    run()
