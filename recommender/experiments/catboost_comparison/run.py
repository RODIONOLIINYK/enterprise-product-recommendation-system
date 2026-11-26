"""Rebuild the historical classifier on matched, strictly causal current slates."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.candidate_token_enrichment.train import snapshot_preserved_hashes
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores, prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.evaluate import evaluate_rolling_slate_metrics, paired_rolling_intervals
from recommender.scope import get_datasets_locs
from recommender.split import TRAIN_CUTOFF, TEST_START
from .features import CATEGORICAL, LEGACY_FEATURES, build_legacy_numeric, categorical_frame, legacy_frame

ROOT = Path('checkpoints/catboost-current-v1')
EVIDENCE = Path('experiments/catboost_current_v1')
HISTORICAL_COMMIT = '2cb7f75'
PRIOR_SUMMARY = Path('experiments/pointwise_token_enrichment_v1/summary.json')


def historical_file(path):
    return subprocess.check_output(['git', 'show', f'{HISTORICAL_COMMIT}:{path}'])


def recover_history():
    config = json.loads(historical_file('configs/catboost_training.json'))
    metrics = json.loads(historical_file('artifacts/catboost_classifier/metrics.json'))
    assert tuple(metrics['feature_columns']) == LEGACY_FEATURES
    return config, {
        'source_commit': HISTORICAL_COMMIT,
        'source_commit_full': subprocess.check_output(['git', 'rev-parse', HISTORICAL_COMMIT], text=True).strip(),
        'model_type': metrics['model_type'], 'split_strategy': metrics['split_strategy'],
        'old_train_date_min': metrics['split_summary']['train']['date_min'],
        'old_train_date_max': metrics['split_summary']['train']['date_max'],
        'old_train_queries': metrics['split_summary']['train']['groups'],
        'old_validation_queries': metrics['split_summary']['validation']['groups'],
        'old_candidates_per_query': metrics['split_summary']['validation']['average_candidates_per_group'],
        'validation_metrics': metrics['ranking_metrics']['validation'],
        'feature_columns': metrics['feature_columns'], 'model_config': config['model'],
        'last_model_change': '535e211 (model retrained without a corresponding metrics.json update)',
        'limitations': [
            'Historical customer-disjoint split overlaps current chronological validation/held-out dates in training.',
            'Historical 25-item groups insert all positive target products plus sampled negatives.',
            'Saved historical metrics are not comparable to current retrieved-slate complete-target metrics.',
            'Old model weights are not loaded or evaluated; only public git code/configuration and saved report metadata are read.',
        ],
    }


def modern_frame(slate, data, catalogue, pids):
    indices = slate['candidate_indices'].numpy()
    frame = categorical_frame(pids, catalogue, indices)
    values = slate['pairwise_features'].numpy().astype(np.float32)
    for i, name in enumerate(NAMES):
        frame[name] = values[..., i].reshape(-1)
    profile = slate['tabular_features'].numpy()
    for i in range(profile.shape[1]):
        frame[f'profile_{i}'] = np.repeat(profile[:, i], indices.shape[1])
    # A dot product is an input feature, not the rejected per-item Stage 1 MLP score.
    u = data['u'].numpy(); v = data['v'].numpy()
    similarities = np.einsum('nd,ncd->nc', u, v[indices])
    frame['tower_dot_similarity'] = similarities.reshape(-1)
    return frame


def validate_dates(slates):
    origin = np.datetime64('2025-01-01')
    for split, slate in slates.items():
        dates = origin + slate['prediction_days'].numpy().astype('timedelta64[D]')
        if (dates >= np.datetime64(TEST_START.date())).any():
            raise ValueError('Held-out queries are forbidden.')
        if split == 'train' and (dates >= np.datetime64(TRAIN_CUTOFF.date())).any():
            raise ValueError('Training dates cross the cutoff.')
        if split == 'val' and (dates < np.datetime64(TRAIN_CUTOFF.date())).any():
            raise ValueError('Validation dates precede its cutoff.')
        if not bool(slate['candidate_mask'].all()):
            raise ValueError('This comparison requires the identical all-valid A-D slates.')


def fit_variant(name, frames, slates, data, params, provenance):
    from catboost import CatBoostClassifier, Pool
    out = ROOT / 'runs' / f'{name}-seed42'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}.')
    out.mkdir(parents=True)
    pools = {split: Pool(frame, label=slates[split]['labels'].numpy().reshape(-1),
                         cat_features=list(CATEGORICAL), feature_names=list(frame.columns))
             for split, frame in frames.items()}
    model = CatBoostClassifier(**params)
    started = time.perf_counter()
    model.fit(pools['train'], eval_set=pools['val'], use_best_model=True)
    elapsed = time.perf_counter() - started
    model.save_model(str(out / 'classifier.cbm'))
    raw = model.predict(pools['val'], prediction_type='RawFormulaVal')
    scores = torch.tensor(raw.reshape(slates['val']['candidate_indices'].shape), dtype=torch.float32)
    torch.save({'logits': scores}, out / 'val_scores.pt')
    metrics = evaluate_rolling_slate_metrics(scores, data['val'], slates['val']['candidate_indices'], save_path=out / 'predictions.npz')
    reload = CatBoostClassifier()
    reload.load_model(str(out / 'classifier.cbm'))
    check = reload.predict(pools['val'], prediction_type='RawFormulaVal')
    if not np.array_equal(raw, check):
        raise AssertionError('Reloaded CatBoost scores changed.')
    # Model-only latency on the same first 100-candidate query.
    tiny = Pool(frames['val'].iloc[:100], cat_features=list(CATEGORICAL))
    for _ in range(3):
        reload.predict(tiny, prediction_type='RawFormulaVal', thread_count=4)
    timings = []
    for _ in range(15):
        t = time.perf_counter(); reload.predict(tiny, prediction_type='RawFormulaVal', thread_count=4)
        timings.append(1000 * (time.perf_counter() - t))
    result = {
        'name': name, 'model_type': 'CatBoostClassifier', 'params': params,
        'feature_names': list(frames['train'].columns), 'selection': 'validation Logloss (historical recipe)',
        'best_iteration_zero_based': model.get_best_iteration(), 'trees': model.tree_count_,
        'training_seconds': elapsed, 'metrics': metrics,
        'model_path': str(out / 'classifier.cbm'), 'model_sha256': file_digest(out / 'classifier.cbm'),
        'predictions_npz': str(out / 'predictions.npz'), 'predictions_sha256': file_digest(out / 'predictions.npz'),
        'logits_pt': str(out / 'val_scores.pt'), 'logits_sha256': file_digest(out / 'val_scores.pt'),
        'strict_replay_max_abs_diff': float(np.max(np.abs(raw - check))),
        'model_only_cpu_request_median_ms': float(np.median(timings)),
        'provenance': provenance, 'heldout_access': False, 'promoted': False,
        'feature_importance': dict(zip(frames['train'].columns, model.get_feature_importance().tolist())),
    }
    atomic_json(result, out / 'result.json')
    return result


def run():
    import catboost
    before = snapshot_preserved_hashes()
    previous = json.loads(PRIOR_SUMMARY.read_text())
    if previous['preserved_hashes'] != before:
        raise ValueError('Prior neural comparison snapshot changed.')
    config, historical = recover_history()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    atomic_json(historical, EVIDENCE / 'historical_validation.json', overwrite=False)
    data, info, features_manifest, meta = prepare_or_load_rolling_feature_bundle(build=False)
    selector_scores, score_manifest = prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    validate_dates(slates)
    for key, value in [('observed_source_digest', meta['observed_source_digest']),
                       ('feature_bundle_identity', features_manifest['identity']),
                       ('slate_cache_identities', slate_manifest['split_cache_identity'])]:
        if previous[key] != value:
            raise ValueError(f'Neural controls have different {key}.')
    pids = info['product_ids']
    catalogue = load_product_catalogue(get_datasets_locs('items'))
    purchases = load_development_purchases('data/interim/cleaned_purchases.csv')
    provenance = {
        'statistics_policy': meta['statistics_policy'], 'observed_source_digest': meta['observed_source_digest'],
        'feature_bundle_identity': features_manifest['identity'],
        'slate_cache_identities': slate_manifest['split_cache_identity'],
        'selector_scores_identity': score_manifest['identity'],
        'historical_recipe_commit': HISTORICAL_COMMIT,
        'metadata_policy': 'snapshot_unverified', 'heldout_access': False,
    }
    params = {**config['model'], 'thread_count': 4}
    results = {}
    for name in ('historical17', 'current53'):
        frames = {}
        for split, slate in slates.items():
            print(f'Preparing {name}/{split} on current fixed candidates.', flush=True)
            if name == 'historical17':
                dates = np.datetime64('2025-01-01') + slate['prediction_days'].numpy().astype('timedelta64[D]')
                idx = slate['candidate_indices'].numpy()
                numeric = build_legacy_numeric(purchases, catalogue, pids, slate['customer_ids'], dates, idx)
                frames[split] = legacy_frame(numeric, categorical_frame(pids, catalogue, idx))
            else:
                frames[split] = modern_frame(slate, data[split], catalogue, pids)
        print(f'Training {name}, {len(frames["train"]):,} candidate rows.', flush=True)
        results[name] = fit_variant(name, frames, slates, data, params, provenance)
        del frames
    baseline_npz = EVIDENCE / 'mlp_same_slate_predictions.npz'
    mlp_metrics = evaluate_rolling_slate_metrics(
        selector_scores['val'].gather(1, slates['val']['candidate_indices']), data['val'],
        slates['val']['candidate_indices'], save_path=baseline_npz,
    )
    comparisons = {}
    for name, result in results.items():
        comparisons[name] = {'MLP_to_tree': paired_rolling_intervals(baseline_npz, result['predictions_npz'])}
        for arm_name, arm in previous['arms'].items():
            # A and B have identical discovery/NDCG selections; C's selections differ.
            for selection in ('ndcg', 'discovery'):
                comparisons[name][arm_name + '_' + selection + '_to_tree'] = paired_rolling_intervals(
                    arm[selection]['predictions_npz'], result['predictions_npz'])
    if snapshot_preserved_hashes() != before:
        raise AssertionError('Existing reference/cache evidence changed.')
    summary = {
        'training_executed': True, 'heldout_access': False, 'promoted': False,
        'catboost_version': catboost.__version__, 'historical': historical,
        'provenance': provenance, 'models': results, 'same_slate_mlp_metrics': mlp_metrics,
        'neural_controls': {name: {s: arm[s] for s in ('ndcg', 'discovery')} for name, arm in previous['arms'].items()},
        'paired_bootstrap_95ci': comparisons, 'preserved_hashes': before,
        'previous_summary_sha256': file_digest(PRIOR_SUMMARY),
        'limitations': ['Historical numbers use different dates, candidate construction and denominators.',
                       'Tree features and objective differ from neural memory; this is a practical model comparison, not an architecture-only ablation.',
                       'No original trained CatBoost weights were used because their training includes future dates relative to current validation.'],
    }
    atomic_json(summary, EVIDENCE / 'summary.json', overwrite=False)
    print(json.dumps({k: v['metrics'] for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true')
    if not parser.parse_args().execute_training:
        parser.error('Pass --execute-training to fit the two matched classifiers.')
    run()
