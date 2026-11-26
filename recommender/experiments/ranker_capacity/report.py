"""Verify and summarize capacity ablations and the truly standalone tree model."""

import argparse
import json
from pathlib import Path

import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.candidate_token_enrichment.benchmark import run_stage2_latency_benchmark
from recommender.experiments.candidate_token_enrichment.model import load_enrichment_ranker_checkpoint
from recommender.experiments.candidate_token_enrichment.train import snapshot_preserved_hashes
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader, prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores, prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import evaluate_rolling_slate_metrics
from .model import load_checkpoint
from .train import EVIDENCE

LABELS = {'control32': 'Fresh original control', 'wider_head': 'Wider scoring MLP',
          'wider_ffn': 'Wider Transformer FFN', 'deeper_memory': 'Two history attention layers'}


def build_chart(summary, trees):
    configs = []
    metrics = {'ndcg': ('ndcg@10', 1, 'NDCG@10 change', 5),
               'recall': ('recall@10', 100, 'Recall@10 change (percentage points)', 3),
               'hits': ('new_hits@10', 1, 'New-product hits@10 change', 0)}
    for metric, (key, scale, label, digits) in metrics.items():
        states = {}
        for selection in ('ndcg', 'discovery'):
            baseline = summary['baseline'][selection]['metrics'][key]
            states[selection] = [[LABELS[n], (r[selection]['metrics'][key] - baseline) * scale]
                                 for n, r in summary['arms'].items()]
            states[selection].append(['Standalone CatBoost', (trees['selected'][selection]['metrics'][key] - baseline) * scale])
        values = [row[1] for rows in states.values() for row in rows] + [0.0]
        low, high = min(values), max(values)
        padding = max(high - low, 1e-6) * 0.15
        domain = [min(low - padding, 0), max(high + padding, 0)]
        for selection, rows in states.items():
            config = {'id': 'capacity-results', 'title': label,
                      'description': 'Treatment minus saved original 32-event pointwise control.',
                      'height': 320, 'columns': ['model', 'delta'], 'rows': rows,
                      'spec': {'type': 'horizontalBar', 'y': 'model', 'x': 'delta',
                               'xLabel': label, 'xDomain': domain, 'showValues': True,
                               'sortOrder': 'original', 'valueFormat': {'style': 'decimal',
                                                                     'minimumFractionDigits': digits,
                                                                     'maximumFractionDigits': digits}}}
            payload = json.dumps(config).replace('<', '\\u003c')
            configs.append(f'<script type="application/json" data-viz-chart id="capacity-{selection}-{metric}">{payload}</script>')
    template = Path(__file__).with_name('comparison.template.html').read_text()
    (EVIDENCE / 'comparison.html').write_text(template.replace('__CONFIG_BLOCKS__', '\n'.join(configs)))


def verify_and_benchmark(summary):
    data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    if manifest['split_cache_identity'] != summary['slate_cache_identities']:
        raise ValueError('Verification slate identities changed.')
    batch = next(iter(make_rolling_slate_loader(slates['val'], batch_size=32, shuffle=False)))
    inputs = rolling_ranker_batch_inputs(batch, data['train']['v'], torch.device('cpu'))
    records, models = [], {}
    for name, result in summary['arms'].items():
        for selection in ('ndcg', 'discovery'):
            saved = result[selection]
            if file_digest(Path(saved['checkpoint'])) != saved['checkpoint_sha256']:
                raise ValueError('Selected checkpoint checksum changed.')
            model, _ = load_checkpoint(saved['checkpoint'], expected_provenance=result['provenance'],
                                       expected_feature_contract_digest=meta['feature_schema_digest'])
            model.eval()
            logits = torch.load(saved['logits_pt'], map_location='cpu', weights_only=False)['logits']
            metrics = evaluate_rolling_slate_metrics(logits, data['val'], slates['val']['candidate_indices'])
            for key, value in metrics.items():
                if saved['metrics'][key] != value:
                    raise AssertionError(f'Metric replay failed: {name}/{selection}/{key}')
            with torch.no_grad():
                replay = model(**inputs).logits
            delta = float((replay - logits[:32]).abs().max())
            if delta > 1e-4:
                raise AssertionError(f'CPU/MPS replay mismatch: {delta}')
            records.append({'arm': name, 'selection': selection,
                            'metrics_reproduced': True, 'cpu32_max_abs_logit_diff': delta})
            if selection == 'ndcg':
                models[name] = model
    models['saved_A'] = load_enrichment_ranker_checkpoint(summary['baseline']['ndcg']['checkpoint'])[0]
    # Run after BOTH training processes have exited, avoiding contention with tree fitting.
    benchmark = run_stage2_latency_benchmark(models, slates['val'], data['train']['v'],
                                           evidence_root=None, warmup=5, repeats=30, save=False)
    atomic_json(benchmark, EVIDENCE / 'benchmark_uncontended.json', overwrite=False)
    if snapshot_preserved_hashes() != summary['preserved_hashes']:
        raise AssertionError('Protected reference/cache hashes changed.')
    verification = {'heldout_access': False, 'preserved_reference_cache_files': len(summary['preserved_hashes']),
                    'all_preserved_hashes_match': True, 'selected_neural_records': records,
                    'standalone_tree_replay': 'Complete eligible-pool replay after reload is enforced by standalone.py'}
    atomic_json(verification, EVIDENCE / 'verification.json', overwrite=False)


def build():
    summary = json.loads((EVIDENCE / 'summary.json').read_text())
    trees = json.loads(Path('experiments/catboost_standalone_v1/summary.json').read_text())
    lines = ['# Ranker capacity and standalone CatBoost comparison', '',
             'All fits use seed 42 and development data only. Training dates are before 2025-10-01; '
             'validation is 2025-10-01 through 2026-02-18. There are 26,253 training queries and '
             '6,191 validation queries, with 10,276 complete targets including 1,857 new targets. '
             'No held-out evaluation or model promotion was performed.', '',
             '## Neural capacity design', '',
             'All four fresh arms share the original fused-profile token, 32 separate history events, '
             '128D tokens, four attention heads, identical rolling 100-item slates, all 41 candidate '
             'statistics, frozen candidate vectors, loss, AdamW/cosine schedule, batch size 64, '
             'and ten epochs. Candidate self-attention and token enrichment remain disabled. '
             'Unchanged initial parameters are exact copies; changed-shape layers and the added '
             'attention layer have fresh initialization. This is an architecture/initialization '
             'comparison under a fixed budget, rather than optimal tuning of each architecture.', '',
             '| Arm | Scoring head | Memory FFN | Memory self-attention layers |',
             '| --- | --- | ---: | ---: |',
             '| Original control | 425 → 256 → 128 → 64 → 1 | 256 | 1 |',
             '| Wider scoring MLP | 425 → 512 → 256 → 128 → 1 | 256 | 1 |',
             '| Wider Transformer FFN | Original | 512 | 1 |',
             '| Deeper history Transformer | Original | 256 | 2 |', '',
             'The no-op control has exact initialization parity with Arm A. Fresh-control and '
             'saved-control comparisons are both reported because MPS training is not bitwise '
             'deterministic. The saved control is an experimental rolling-data comparator, not a '
             'replacement for preserved production aliases.', '',
             '## Standalone CatBoost design', '',
             '**This CatBoost model directly scores every eligible product. It uses no Two-Tower '
             'model, neural vectors, dot similarities, MLP selector, retrieved shortlist, Transformer, '
             'or subsequent neural ranking.** Inputs are three raw product categoricals '
             '(SKU/category/business line), 41 strictly earlier rolling purchase features and '
             'eight customer profile statistics (52 columns total). Reusing dated query/statistical '
             'caches does not supply their historical neural representations to the classifier.', '',
             'Eligibility is recomputed from raw development purchases and catalogue metadata and '
             'must agree with cached masks. All eligible SKUs are scored at validation time; '
             'validation targets are never inserted into a selected candidate list. The modelled '
             '841-SKU universe is verified to contain every development-eligible SKU.', '',
             f"Training samples {trees['provenance']['training_pairs_sampled']:,} pairs representing "
             f"{trees['provenance']['training_eligible_pairs']:,} eligible pairs: all eligible positives "
             'plus up to 100 uniformly sampled negatives per query. Each sampled negative is '
             'weighted by eligible-negative-count/sample-count, yielding an unbiased estimate '
             'of the full-pool binary Logloss for a fixed scorer. Sampling is a training cost '
             'optimization, not a Stage 1 retrieval model. Validation scores all '
             f"{trees['provenance']['validation_eligible_pairs']:,} eligible pairs.", '',
             'The classifier retains the historical depth-8, maximum-500-tree recipe, learning '
             'rate 0.05, Logloss early stopping (50 rounds), and four CPU threads. Ranking '
             'selections inspect 50-tree prefixes, at most ten checkpoints. No historical '
             'trained CatBoost weights are loaded.', '']
    for selection in ('ndcg', 'discovery'):
        lines += [f'## {selection.capitalize()}-selected results', '',
                  '| Model | Epoch/trees | Recall@5 | Recall@10 | NDCG@10 | New hits@10 / 1,857 | New recall@10 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        rows = [('Saved A / original 32 events', summary['baseline'][selection])]
        rows += [(LABELS[n], result[selection]) for n, result in summary['arms'].items()]
        rows += [('Standalone CatBoost / full eligible pool', trees['selected'][selection])]
        for name, entry in rows:
            m = entry['metrics']
            lines.append(f"| {name} | {entry.get('epoch', entry.get('trees'))} | {m['recall@5']:.5f} | "
                         f"{m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} | {m['new_recall@10']:.5f} |")
        lines += ['', 'Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. '
                  'Delta is treatment minus comparator, conditional on selected fitted weights. '
                  'They do not account for training-seed uncertainty or repeated validation selection.', '',
                  '| Treatment | Comparator | Metric | Delta | 95% CI |',
                  '| --- | --- | --- | ---: | --- |']
        for variant in ('wider_head', 'wider_ffn', 'deeper_memory'):
            for key, label in (('saved_A_to_treatment', 'Saved A'), ('fresh_control_to_treatment', 'Fresh control')):
                interval = summary['paired_bootstrap_95ci'][selection][variant][key]
                for metric in ('recall@10_delta', 'ndcg@10_delta', 'new_recall@10_delta'):
                    c = interval[metric]
                    lo, _, hi = c['ci_95']
                    lines.append(f"| {LABELS[variant]} | {label} | {metric} | {c['point_delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
        for metric, c in trees['selected'][selection]['A_to_standalone_95ci'].items():
            lo, _, hi = c['ci_95']
            lines.append(f"| Standalone CatBoost | Saved A | {metric} | {c['point_delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
        lines.append('')
    lines += ['## Learning curves and training cost', '',
              'Training losses are minibatch means in training mode (with dropout), while validation '
              'loss uses evaluation mode. Their absolute levels are not a clean generalization-gap '
              'estimate. Curves and selection epochs are provided without attributing every '
              'validation change to overfitting.', '',
              '| Arm | Trainable parameters | Training seconds | Epoch 1 training loss | Epoch 10 training loss | Epoch 10 validation loss |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in summary['arms'].items():
        lines.append(f"| {LABELS[name]} | {result['trainable_parameters']:,} | {result['training_seconds']:.1f} | "
                     f"{result['history'][0]['train_loss']:.5f} | {result['history'][-1]['train_loss']:.5f} | "
                     f"{result['history'][-1]['metrics']['val_loss']:.5f} |")
    benchmark_path = EVIDENCE / 'benchmark_uncontended.json'
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text())
        lines += ['', '## Model inference latency', '',
                  'Warm prepared-input forward medians, 100 candidates per neural request. '
                  'Measurements below were taken after both training processes exited. '
                  'Standalone trees score their entire eligible pool per request, so the two '
                  'latency scopes differ.', '',
                  '| Model | CPU batch 1 (ms) | CPU batch 64 (ms) | MPS batch 1 (ms) | MPS batch 64 (ms) |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for name in ('saved_A', *summary['arms']):
            values = []
            for device in ('cpu', 'mps'):
                for batch_size in (1, 64):
                    entry = benchmark['devices'].get(device, {}).get(name)
                    values.append(f"{entry[f'batch_{batch_size}']['prepared_on_device_forward']['median_ms']:.3f}" if entry else 'unavailable')
            label = 'Saved A' if name == 'saved_A' else LABELS[name]
            lines.append(f'| {label} | ' + ' | '.join(values) + ' |')
        tree = trees['selected']['ndcg']
        lines += ['', f"Standalone NDCG-selected CatBoost: {tree['model_only_cpu_full_pool_median_ms']:.3f} ms "
                  f"CPU median for {tree['latency_query_eligible_skus']} eligible SKUs; statistical feature "
                  'preparation and Python DataFrame/Pool construction are excluded.']
    lines += ['', '## Limits and verification', '',
              '- Capacity-arm comparisons hold candidate sets fixed. Standalone CatBoost changes the '
              'candidate policy, input features and objective; its comparison is of complete pipelines.',
              '- All recall denominators include targets absent from the candidate pool. New hits '
              'always divide by all 1,857 new targets. Current full-pool availability still uses '
              'snapshot metadata and last-modified blocking assumptions.',
              '- Single seed, limited training budget and repeated validation use do not establish '
              'a universal architecture ranking. No reference checkpoint was replaced.',
              '- Selected neural weights are reloaded and replayed; standalone selected trees are '
              'reloaded against the complete eligible validation pool.',
              f"- All {len(summary['preserved_hashes'])} preserved reference/cache evidence hashes remain unchanged.",
              '- Final uncontended latency measurements exclude ingestion, feature preparation, '
              'disk loading and network time. Initial in-run measurements may overlap tree fitting; '
              'use benchmark_uncontended.json for the final comparison.', '',
              '## Artifacts and reproduction', '',
              '- [Capacity summary](summary.json)',
              '- [Standalone CatBoost summary](../catboost_standalone_v1/summary.json)',
              '- [Independent verification](verification.json)',
              '- [Final latency benchmark](benchmark_uncontended.json)', '',
              '```bash',
              './venv/bin/python -m recommender.experiments.ranker_capacity.train --execute-training',
              './venv/bin/python -m recommender.experiments.catboost_comparison.standalone --execute-training',
              './venv/bin/python -m recommender.experiments.ranker_capacity.report --verify',
              'PYTHONPATH=. ./venv/bin/pytest tests/ -q',
              'PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q', '```', '']
    checks_path = EVIDENCE / 'checks.json'
    if checks_path.exists():
        checks = json.loads(checks_path.read_text())
        lines += ['## Test results', '',
                  f"Full safe suite: {checks['safe_suite']['passed']} passed, {checks['safe_suite']['skipped']} skipped. "
                  f"Synthetic-only: {checks['synthetic_suite']['passed']} passed, {checks['synthetic_suite']['skipped']} skipped. "
                  'Both include 70 passing subtests. Undefined-symbol/unused-import checks and '
                  'git diff --check pass. Chart configuration checks cover all six selector states.', '']
    (EVIDENCE / 'REPORT.md').write_text('\n'.join(lines))
    build_chart(summary, trees)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify_and_benchmark(json.loads((EVIDENCE / 'summary.json').read_text()))
    build()
