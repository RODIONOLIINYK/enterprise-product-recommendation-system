"""CatBoost directly ranks every eligible SKU; no neural vectors or model scores.

Reuses only dated query definitions, statistical features and eligibility from the
sealed development caches. No Stage 1 checkpoint or score cache is loaded; neural
fields in shared dataset containers are discarded, and neural slates are not used.
Training uses all eligible positives and uniform sampled negatives with inverse-
probability weights; validation scores the entire eligible pool.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import RollingCandidatePool, load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.experiments.rolling_candidate_v2.config import CACHE_ROOT
from recommender.experiments.rolling_candidate_v2.data import (
    _cohort_slices, compute_rolling_source_digest, validate_rolling_feature_manifest,
)
from recommender.experiments.rolling_candidate_v2.evaluate import (
    evaluate_rolling_slate_metrics, paired_rolling_intervals,
)
from recommender.features import LEGACY_PROFILE_FEATURES
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from .features import CATEGORICAL, categorical_frame

ROOT = Path('checkpoints/catboost-standalone-v1')
EVIDENCE = Path('experiments/catboost_standalone_v1')
FEATURE_NAMES = (*CATEGORICAL, *NAMES, *(f'profile_{name}' for name in LEGACY_PROFILE_FEATURES))
SAMPLE_NEGATIVES = 100


def sample_training_pairs(allowed, target, *, negatives=SAMPLE_NEGATIVES, seed=42):
    """An unbiased sampled estimate of full eligible-pool binary Logloss."""
    allowed, target = np.asarray(allowed, dtype=bool), np.asarray(target, dtype=bool)
    if allowed.shape != target.shape or allowed.ndim != 2 or negatives < 1:
        raise ValueError('Aligned query/product matrices and positive negative budget required.')
    rng = np.random.default_rng(seed)
    rows, columns, weights = [], [], []
    for row in range(len(allowed)):
        positive = np.flatnonzero(allowed[row] & target[row])
        negative = np.flatnonzero(allowed[row] & ~target[row])
        k = min(negatives, len(negative))
        chosen = rng.choice(negative, size=k, replace=False)
        selected = np.concatenate([positive, chosen])
        rows.append(np.full(len(selected), row, dtype=np.int32))
        columns.append(selected.astype(np.int32))
        weights.append(np.concatenate([np.ones(len(positive)), np.full(k, len(negative) / max(k, 1))]).astype(np.float32))
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(weights)


def statistical_frame(features, profile, product_ids, catalogue, rows, columns):
    """Accept only source statistics; learned representations are not inputs."""
    if features.shape[-1] != len(NAMES) or profile.shape != (features.shape[0], len(LEGACY_PROFILE_FEATURES)):
        raise ValueError('Unexpected statistical feature dimensions.')
    frame = categorical_frame(product_ids, catalogue, columns)
    values = np.asarray(features[rows, columns], dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(profile).all():
        raise ValueError('Non-finite standalone statistics.')
    for i, name in enumerate(NAMES):
        frame[name] = values[:, i]
    for i, name in enumerate(LEGACY_PROFILE_FEATURES):
        frame[f'profile_{name}'] = np.asarray(profile[rows, i], dtype=np.float32)
    if tuple(frame.columns) != FEATURE_NAMES:
        raise AssertionError('Unexpected standalone feature order.')
    return frame


def load_statistical_data():
    """Read immutable development datasets directly, never instantiating neural models."""
    manifest = json.loads((CACHE_ROOT / 'features_manifest.json').read_text())
    purchases = load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue = load_product_catalogue(get_datasets_locs('items'))
    companies = load_companies(get_datasets_locs('companies'), purchases=purchases, minimum_purchases=1)
    digests = compute_rolling_source_digest(purchases, catalogue, companies)
    validate_rolling_feature_manifest(manifest, expected_identity=manifest['identity'],
                                     expected_observed_source_digest=digests['observed_source_digest'])
    if manifest['source_input_digests'] != digests:
        raise ValueError('Statistical feature source identity mismatch.')
    retrieval_root = Path('data/interim/e1_e5_retrieval_v1')
    shared_root = Path('data/interim/shared_stage1_v1')
    retrieval = [torch.load(retrieval_root / f'{s}.pt', map_location='cpu', weights_only=False)
                 for s in ('train', 'val')]
    sources = [torch.load(shared_root / f'{s}.pt', map_location='cpu', weights_only=False)
               for s in ('train', 'val')]
    info = torch.load(retrieval_root / 'manifest.pt', map_location='cpu', weights_only=False)
    if info['identity'] != manifest['source_identity'] or list(info['product_ids']) != manifest['product_ids']:
        raise ValueError('Statistical query/catalogue source identity mismatch.')
    sliced_source, sliced_retrieval, _ = _cohort_slices(*retrieval, *sources)
    expected_identity = json_digest({
        'cache_version': manifest['version'], 'namespace': manifest['namespace'],
        'statistics_policy': manifest['statistics_policy'], 'feature_schema': manifest['schema_digest'],
        'reference_id': manifest['active_reference_id'], 'upstream': manifest['upstream_digest'],
        'selector': manifest['selector_digest'], 'source': info['identity'],
        'observed_source_digest': digests['observed_source_digest'],
        'train_rows': sliced_source[0]['transition_keys'], 'val_rows': sliced_source[1]['transition_keys'],
        'minimum_eligible': manifest['minimum_eligible'],
    })
    if expected_identity != manifest['identity'] or any(v['identity'] != info['identity'] for v in retrieval):
        raise ValueError('Query order/statistical cache identity mismatch.')
    pids = info['product_ids']
    index = {str(p): i for i, p in enumerate(pids)}
    pool = RollingCandidatePool(catalogue, purchases)
    data = {}
    for split, source, values in zip(('train', 'val'), sliced_source, sliced_retrieval):
        dates = np.datetime64('2025-01-01') + source['prediction_days'].numpy().astype('timedelta64[D]')
        if np.isnat(dates).any() or (dates >= np.datetime64(TEST_START.date())).any():
            raise ValueError('Held-out/missing query dates forbidden.')
        if split == 'train' and (dates >= np.datetime64(TRAIN_CUTOFF.date())).any():
            raise ValueError('Training queries cross cutoff.')
        if split == 'val' and (dates < np.datetime64(TRAIN_CUTOFF.date())).any():
            raise ValueError('Validation query before cutoff.')
        for name, positive_key in (('target', 'positive_pids'), ('new_target', 'new_positive_pids')):
            expected_targets = torch.zeros_like(values[name])
            for row, positives in enumerate(source[positive_key]):
                expected_targets[row, [index[str(p)] for p in positives if str(p) in index]] = True
            if not torch.equal(values[name], expected_targets):
                raise ValueError('Query target order mismatch.')
        for date in np.unique(dates):
            eligible = pool.eligible_on(pd.Timestamp(date))
            if not eligible <= set(index):
                raise ValueError('Modelled catalogue omits an eligible development SKU.')
            expected = torch.zeros(len(pids), dtype=torch.bool)
            expected[[index[p] for p in eligible]] = True
            if not torch.equal(values['allowed'][dates == date], expected.expand(int((dates == date).sum()), -1)):
                raise ValueError('Cached eligibility differs from raw purchase/catalogue policy.')
        feature_path = CACHE_ROOT / f'{split}_features.npy'
        if file_digest(feature_path) != manifest['feature_sha256'][split]:
            raise ValueError('Statistical feature checksum mismatch.')
        features = np.load(feature_path, mmap_mode='r')
        if features.shape != (*values['target'].shape, len(NAMES)):
            raise ValueError('Statistical feature/query shape mismatch.')
        source_minimal = {name: source[name] for name in
                          ('customer_ids', 'positive_pids', 'new_positive_pids', 'attention_mask', 'prediction_days')}
        # No text, latent vectors, dot products, or Stage 1 scores survive this boundary.
        data[split] = {name: values[name] for name in ('allowed', 'target', 'new_target')}
        data[split].update(source=source_minimal, features=features,
                           profile=source['tabular_features'].numpy().astype(np.float32))
    del sources, retrieval, sliced_source, sliced_retrieval
    gc.collect()
    return data, pids, catalogue, manifest


def run():
    from catboost import CatBoostClassifier, Pool
    import catboost
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'runs' / 'statistics52-seed42'
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    data, pids, catalogue, manifest = load_statistical_data()
    out.mkdir(parents=True)
    source_paths = [CACHE_ROOT / 'features_manifest.json']
    source_paths += [Path('data/interim') / root / f'{s}.pt'
                     for root in ('e1_e5_retrieval_v1', 'shared_stage1_v1') for s in ('train', 'val')]
    source_hashes = {str(p): file_digest(p) for p in source_paths}
    tr_rows, tr_cols, weight = sample_training_pairs(data['train']['allowed'], data['train']['target'])
    va_rows, va_cols = np.where(data['val']['allowed'].numpy())
    train_frame = statistical_frame(data['train']['features'], data['train']['profile'], pids, catalogue, tr_rows, tr_cols)
    train_labels = data['train']['target'].numpy()[tr_rows, tr_cols].astype(np.float32)
    train_pool = Pool(train_frame, label=train_labels, weight=weight, cat_features=list(CATEGORICAL))
    del train_frame
    gc.collect()
    val_frame = statistical_frame(data['val']['features'], data['val']['profile'], pids, catalogue, va_rows, va_cols)
    val_labels = data['val']['target'].numpy()[va_rows, va_cols].astype(np.float32)
    val_pool = Pool(val_frame, label=val_labels, cat_features=list(CATEGORICAL))
    historical = json.loads(Path('experiments/catboost_current_v1/historical_validation.json').read_text())
    params = {**historical['model_config'], 'thread_count': 4}
    provenance = {
        'model_type': 'CatBoostClassifier', 'catboost_version': catboost.__version__,
        'feature_names': list(FEATURE_NAMES), 'neural_inputs': [], 'neural_retrieval': False,
        'candidate_policy': 'all eligible strictly prior-sale SKUs; no retrieved shortlist',
        'statistics_policy': 'rolling_observed_v2', 'feature_bundle_identity': manifest['identity'],
        'observed_source_digest': manifest['observed_source_digest'], 'product_ids': pids,
        'source_hashes': source_hashes, 'seed': 42, 'params': params,
        'training_negative_sampling': {'maximum_per_query': SAMPLE_NEGATIVES, 'distribution': 'uniform_without_replacement',
                                       'positive_weight': 1, 'negative_weight': 'eligible_negative_count / sampled_negative_count',
                                       'objective': 'unbiased sampled full-pool binary Logloss'},
        'training_pairs_sampled': len(tr_rows), 'training_eligible_pairs': int(data['train']['allowed'].sum()),
        'training_eligible_positives': int((data['train']['allowed'] & data['train']['target']).sum()),
        'training_positive_pairs_sampled': int(train_labels.sum()),
        'training_sum_weights': float(weight.astype(np.float64).sum()),
        'validation_eligible_pairs': len(va_rows), 'heldout_access': False, 'promoted': False,
        'metadata_policy': 'snapshot_unverified',
    }
    atomic_json({**provenance, 'status': 'running', 'identity': json_digest(provenance)}, out / 'run.json', overwrite=False)
    print(f'Standalone CatBoost: {len(tr_rows):,} sampled training pairs representing '
          f'{provenance["training_eligible_pairs"]:,} eligible pairs; {len(va_rows):,} full-pool validation pairs.', flush=True)
    started = time.perf_counter()
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    elapsed = time.perf_counter() - started
    model.save_model(str(out / 'logloss_selected.cbm'))
    del train_pool
    gc.collect()
    indices = torch.arange(len(pids)).expand(len(data['val']['source']['customer_ids']), -1)
    baseline = json.loads(Path('experiments/candidate_token_enrichment_v1/summary.json').read_text())['arms']['arm_a_control_pointwise_425d']
    best, history = {'ndcg': None, 'discovery': None}, []
    raw_logloss = model.predict(val_pool, prediction_type='RawFormulaVal', thread_count=4)
    def dense_scores(raw):
        scores = torch.full(data['val']['allowed'].shape, -torch.inf)
        scores[va_rows, va_cols] = torch.tensor(raw, dtype=torch.float32)
        return scores
    logloss_metrics = evaluate_rolling_slate_metrics(dense_scores(raw_logloss), data['val'], indices,
                                                   save_path=out / 'logloss_predictions.npz')
    for step, raw in enumerate(model.staged_predict(val_pool, prediction_type='RawFormulaVal', eval_period=50, thread_count=4), 1):
        trees = min(step * 50, model.tree_count_)
        scores = dense_scores(raw)
        metrics = evaluate_rolling_slate_metrics(scores, data['val'], indices)
        qualified = metrics['recall@10'] >= 0.7558833119851873 and metrics['ndcg@10'] >= 0.5918928743676547
        history.append({'trees': trees, 'metrics': metrics, 'qualified': qualified})
        keys = {'ndcg': (metrics['ndcg@10'], metrics['recall@10'], metrics['new_hits@10']),
                'discovery': (int(qualified), metrics['new_hits@10'], metrics['ndcg@10'])}
        for selection, key in keys.items():
            if best[selection] is None or key > best[selection]['key']:
                best[selection] = {'key': key, 'trees': trees, 'metrics': metrics, 'qualified': qualified, 'scores': scores}
    selected = {}
    for selection, result in best.items():
        path = out / f'best_{selection}.cbm'
        saved = model.copy()
        saved.shrink(ntree_end=result['trees'])
        saved.save_model(str(path))
        loaded = CatBoostClassifier()
        loaded.load_model(str(path))
        replay = dense_scores(loaded.predict(val_pool, prediction_type='RawFormulaVal', thread_count=4))
        torch.testing.assert_close(replay, result['scores'], rtol=0, atol=0)
        npz_path = out / f'{selection}_predictions.npz'
        metrics = evaluate_rolling_slate_metrics(replay, data['val'], indices, save_path=npz_path)
        if metrics != result['metrics']:
            raise AssertionError('Standalone metric replay failed.')
        atomic_torch_save({'logits': replay}, out / f'{selection}_logits.pt')
        tiny = Pool(val_frame.iloc[np.flatnonzero(va_rows == 0)], cat_features=list(CATEGORICAL))
        times = []
        for i in range(25):
            tick = time.perf_counter()
            loaded.predict(tiny, prediction_type='RawFormulaVal', thread_count=4)
            if i >= 5:
                times.append(1000 * (time.perf_counter() - tick))
        selected[selection] = {'trees': result['trees'], 'metrics': metrics, 'qualified': result['qualified'],
                               'checkpoint': str(path), 'checkpoint_sha256': file_digest(path),
                               'predictions_npz': str(npz_path), 'predictions_sha256': file_digest(npz_path),
                               'strict_replay_max_abs_diff': 0.0, 'model_only_cpu_full_pool_median_ms': float(np.median(times)),
                               'latency_query_eligible_skus': int((va_rows == 0).sum()),
                               'A_to_standalone_95ci': paired_rolling_intervals(baseline[selection]['predictions_npz'], npz_path)}
    for path, digest in source_hashes.items():
        if file_digest(Path(path)) != digest:
            raise AssertionError('Standalone statistical source changed.')
    summary = {'training_executed': True, 'heldout_access': False, 'promoted': False,
               'provenance': provenance, 'training_seconds': elapsed, 'selected': selected,
               'baseline': baseline, 'history': history, 'logloss_selected': {'trees': model.tree_count_, 'metrics': logloss_metrics},
               'limitations': ['Single seed and historical CatBoost hyperparameters; no claim of optimal CatBoost tuning.',
                              'Training samples negatives with inverse-probability weights; validation scores every eligible SKU.',
                              'Model has 52 categorical/statistical features, no text or learned representations.',
                              'Comparison changes candidate policy and objective as well as model family.',
                              'Query/statistical caches retain historical metadata and train-fitted normalization assumptions.']}
    atomic_json(summary, EVIDENCE / 'summary.json', overwrite=False)
    atomic_json({**provenance, 'status': 'complete', 'selected': selected}, out / 'run.json')
    print(json.dumps({s: r['metrics'] for s, r in selected.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true')
    if not parser.parse_args().execute_training:
        parser.error('Pass --execute-training.')
    run()
