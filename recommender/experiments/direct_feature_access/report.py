"""Build the feature audit, measured diagnosis and standalone scientific figures."""

import base64
import json
from pathlib import Path

from recommender.artifacts import atomic_json
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.experiments.static_enrichment.config import TRANSFERRED_CUSTOMER_STAT_NAMES
from recommender.features import LEGACY_PROFILE_FEATURES
from .model import CATEGORY_FIELDS
from .train import EVIDENCE

LABELS = {'control': 'Fresh control', 'profile8': 'Direct profile8',
          'candidate_categories': 'Direct SKU/category/line',
          'profile8_and_categories': 'Both direct paths'}


def build():
    s = json.loads((EVIDENCE / 'summary.json').read_text())
    d = json.loads((EVIDENCE / 'diagnostics.json').read_text())
    trees = json.loads(Path('experiments/catboost_standalone_v1/summary.json').read_text())
    audit = []
    for name in NAMES:
        audit.append({'feature': name, 'catboost': 'direct scalar',
                      'stage1_mlp': 'deliberately omitted in repeat-no-month-v1' if name in ('month_sin', 'month_cos') else 'direct scalar',
                      'stage2_before': 'direct scoring scalar; first 13 also attention bias',
                      'stage2_treatment': 'unchanged direct scalar'})
    for name in LEGACY_PROFILE_FEATURES:
        audit.append({'feature': name, 'catboost': 'direct scalar',
                      'stage1_mlp': 'direct scalar from sealed upstream cache' if name in TRANSFERRED_CUSTOMER_STAT_NAMES else 'not a separate direct input',
                      'stage2_before': 'profile attention-bias key; zeroed in token value under bias-only mode',
                      'stage2_treatment': 'also directly concatenated to scoring MLP in profile8 variants'})
    for name in CATEGORY_FIELDS:
        audit.append({'feature': name, 'catboost': 'categorical input',
                      'stage1_mlp': 'indirect through frozen candidate product vector; no separate scoring category input',
                      'stage2_before': 'indirect through frozen candidate vector; history IDs are separate inputs',
                      'stage2_treatment': 'also independent learned candidate embedding in categorical variants'})
    atomic_json({'features': audit, 'source_policy': 'rolling_observed_v2', 'heldout_access': False,
                 'profile_and_context_tensor_parity_verified': d['profile_and_context_parity']},
                EVIDENCE / 'feature_audit.json')
    rows = [('Saved pointwise A', s['baseline']['ndcg'])]
    rows += [(LABELS[k], a['ndcg']) for k, a in s['arms'].items()]
    rows.append(('Standalone CatBoost', trees['selected']['ndcg']))
    lines = ['# Direct feature access and underfitting audit', '',
             'Development-only screen: train before 2025-10-01, validation 2025-10-01 through 2026-02-18. '
             '26,253 train queries, 6,191 validation queries, 10,276 complete validation targets, including '
             '1,857 new targets. No held-out data access, reference promotion, commit or push.', '',
             '## What was and was not missing', '',
             'All 41 CatBoost rolling context features already enter the 425D Stage 2 scoring MLP. '
             'Popularity, category affinity, cadence, demand, transition lifts and same-basket signals were not missing. '
             'The Stage 1 reference directly consumes 39 of these, deliberately excluding month sine/cosine, '
             'plus six upstream customer statistics. Five of those six overlap CatBoost profile8; the remaining '
             'one is median_purchase_quantity. The three CatBoost profile fields absent as separate Stage 1 '
             'scalars are total_purchases, days_since_first_order and days_since_last_order. Some overlap '
             'other context statistics, so absence of a named scalar does not imply total absence of its information.', '',
             'Stage 2 profile8 previously acted through profile attention bias; under statistics_attention the '
             'profile numeric token values are zero. Candidate IDs/category/line were indirect through frozen '
             'candidate vectors, with explicit identity embeddings only on historical items. See '
             '[feature_audit.json](feature_audit.json) for all 52 fields.', '',
             '## Controlled implementation', '',
             'Four fresh arms share identical rolling quota100_50 slates, frozen u/v sources, original 32-event '
             'memory and attention, head widths 256/128/64, RankerLoss, AdamW/cosine schedule, batch64, seed42, '
             'and ten epochs. Only direct scoring inputs change. Profile8 adds eight existing source scalars; '
             'categoricals add separately learned SKU16/category8/line8 embeddings. Head input widths are '
             '425/433/457/465. Added first-layer columns start at zero; all original weights are exact copies. '
             'Candidate attention, token enrichment and Stage 1 MLP score inputs remain disabled.', '',
             f"Initial maximum logit differences: `{s['init_max_abs_diff']}` (float rounding only). "
             'A fresh control quantifies MPS run variation; saved A remains the stronger comparator. '
             'Catalogue categoricals retain the same snapshot_unverified limitation as CatBoost. '
             'The Stage 1 selector/default aliases are not altered.', '',
             '## Validation results: NDCG-selected checkpoints', '',
             '| Model | Epoch/trees | Recall@10 | NDCG@10 | New hits@10 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, r in rows:
        m = r['metrics']
        lines.append(f"| {name} | {r.get('epoch', r.get('trees'))} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} |")
    lines += ['', 'None of the three direct-input treatments beats the fresh control or saved A on NDCG, Recall@10 or new hits. No variant is promoted. The CatBoost row uses a full eligible pool and a different binary objective; it is a practical pipeline comparison, not an architecture-only ablation.', '',
              'Discovery-selected checkpoints, without mixing their metrics with NDCG selections:', '',
              '| Arm | Epoch | Recall@10 | NDCG@10 | New hits@10 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, a in s['arms'].items():
        r = a['discovery']; m = r['metrics']
        lines.append(f"| {LABELS[name]} | {r['epoch']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} |")
    lines += ['', 'Paired customer-clustered bootstrap: delta in NDCG@10, 95% interval; '
              'conditional on these selected seed42 checkpoints and repeated validation use.', '',
              '| Arm | vs fresh control | vs saved A |', '| --- | --- | --- |']
    for name, comp in s['paired_bootstrap_95ci']['ndcg'].items():
        if name == 'control':
            continue
        values = []
        for reference in ('fresh_control_to_treatment', 'saved_A_to_treatment'):
            metric = comp[reference]['ndcg@10_delta']
            low, _, high = metric['ci_95']
            values.append(f"{metric['point_delta']:+.5f} [{low:+.5f}, {high:+.5f}]")
        lines.append(f"| {LABELS[name]} | {values[0]} | {values[1]} |")
    lines += ['', '## Is this underfitting?', '',
              'Capacity is not the same as useful generalization. Earlier wider-head training loss fell from '
              '3.13755 to 2.82055 while its validation loss reached 2.68227 at epoch6 and rose to 2.70246 at '
              'epoch10. Its selected NDCG remained below A. The fresh original control likewise improved training '
              'loss while validation loss stopped improving late in training. Training-mode loss includes dropout; '
              'it must not be interpreted as a directly comparable train-vs-validation generalization gap.', '',
              'Below, losses use evaluation mode and identical loss settings on fixed independently sampled '
              '4,096-row training/validation cohorts. The two periods have different difficulty/population mixes, '
              'so the gap alone cannot identify overfitting. Conditional recall excludes targets missing from '
              'the candidate slate and is not the full-denominator headline Recall@10.', '',
              '| Model | Eval train loss | Eval validation loss | Train conditional micro R@10 | Val conditional micro R@10 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, fit in d['eval_mode_fit_metrics'].items():
        tr, va = fit['train'], fit['val']
        lines.append(f"| {LABELS.get(name,name)} | {tr['loss']:.4f} | {va['loss']:.4f} | "
                     f"{tr['slate_conditional_recall@10']:.4f} | {va['slate_conditional_recall@10']:.4f} |")
    probe = d['tiny_train_memorization_probe']; start, end = probe['initial'], probe['curve'][-1]
    lines += ['', f"Train-only optimizer/capacity probe: original pointwise model, 128 fixed training queries, "
              f"600 optimizer updates, lr=0.001, dropout disabled, weight_decay=0. Loss "
              f"`{start['loss']:.4f} -> {end['loss']:.4f}`, conditional micro Recall@10 "
              f"`{start['slate_conditional_recall@10']:.4f} -> {end['slate_conditional_recall@10']:.4f}`. "
              f"The multi-positive listwise entropy floor is `{probe['listwise_entropy_lower_bound']:.4f}`, "
              'so zero combined loss is not the appropriate memorization target. This is a sanity check, '
              'not a validation candidate or proof of optimal full-data optimization.', '',
              '## How much independent signal is available?', '',
              'Millions of query/product pairs are mostly generated non-purchases, not millions of independent '
              'observed preference events. Training queries share customers and overlapping histories. Purchase-only '
              'labels lack exposure and inventory information; they cannot distinguish refusal from non-exposure. '
              'No universal sample threshold determines whether this architecture has enough data.', '',
              'Measured support:', '', '```json', json.dumps(d['support'], indent=2), '```', '',
              '## Why CatBoost is competitive', '',
              'Its 41 context scalars account for '
              f"{d['catboost_prediction_values_change']['groups']['context41']:.2f}% of PredictionValuesChange importance, "
              f"profile8 {d['catboost_prediction_values_change']['groups']['profile8']:.2f}%, categorical3 "
              f"{d['catboost_prediction_values_change']['groups']['categorical3']:.2f}%. These are fitted-model "
              'prediction-change importances, not causal contributions or percentages of accuracy. '
              'The dominant features are hierarchical_backoff and industry_sku_affinity; both already reach '
              'the neural scorer. CatBoost is a boosted tree ensemble using extensive engineered relational '
              'statistics, not a model learning everything from raw records.', '',
              'Primary external context: [Grinsztajn et al., NeurIPS 2022]'
              '(https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf) '
              'find trees strong on medium-sized tabular datasets, with robustness to irrelevant features and '
              'irregular target functions. This is supporting context, not evidence about our exact model. '
              '[CatBoost importance definition](https://catboost.ai/docs/en/concepts/fstr).', '',
              '## Practical next decision', '',
              'Keep CatBoost as a first-class practical baseline. Larger attention/MLP capacity has not earned '
              'its complexity in completed screens. Direct access to existing fields must win a matched comparison '
              'before integration. Prioritize genuinely new preference evidence (exposures, availability and '
              'purchase opportunity), coherent snapshots, and validation learning curves before another size increase. '
              'The next architecture-focused diagnostic would compare a statistics-only MLP with CatBoost using '
              'identical full-pool rows, inputs, sampling weights and binary objective; the present tree/neural '
              'comparison changes objective/candidate policy too. Objective repeat/new weighting was already tried '
              'by the user and is not proposed as a new experiment.', '',
              '## Verification and reproduction', '',
              'Full selected-model inference replay, permutation/padding checks, CPU replay on 32 validation rows, '
              'tensor parity with shared profile/context sources, strict checkpoint lookup/config/provenance checks, '
              f"and all {len(s['preserved_hashes'])} protected reference/cache hashes are checked. "
              'See [diagnostics.json](diagnostics.json), [summary.json](summary.json), and [checks.json](checks.json).', '',
              '```bash', './venv/bin/python -m recommender.experiments.direct_feature_access.train --execute-training',
              './venv/bin/python -m recommender.experiments.direct_feature_access.diagnostics',
              './venv/bin/python -m recommender.experiments.direct_feature_access.report',
              'PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q',
              'PYTHONPATH=. ./venv/bin/pytest tests/ -q', '```', '',
              'Training creates exclusive run artifacts and the summary/diagnostics refuse overwrites. '
              'Do not rerun these commands to overwrite evidence.']
    (EVIDENCE / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    figure(s, d)


def figure(s, d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    capacity = json.loads(Path('experiments/ranker_capacity_v1/summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    colors = ['#176b98', '#cf7632', '#218b6e', '#a34f97']
    for (name, arm), color in zip(capacity['arms'].items(), colors):
        label = {'control32':'Original control','wider_head':'Wider scoring MLP',
                 'wider_ffn':'Wider Transformer FFN','deeper_memory':'Two attention layers'}[name]
        history = arm['history']; xs = [r['epoch'] for r in history]
        axes[0].plot(xs, [r['train_loss'] for r in history], label=label, color=color)
        axes[1].plot(xs, [r['metrics']['ndcg@10'] for r in history], label=label, color=color)
    axes[0].set(title='Larger models can lower training loss', xlabel='Epoch', ylabel='Training loss (dropout active)')
    axes[1].set(title='Validation ranking does not improve with size', xlabel='Epoch', ylabel='Validation NDCG@10')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15); ax.set_xticks([1, 3, 5, 7, 10])
    axes[1].legend(fontsize=8, loc='lower right', frameon=False)
    fig.savefig(EVIDENCE / 'capacity-diagnosis.png', dpi=160, facecolor='white')
    plt.close(fig)
    encoded = base64.b64encode((EVIDENCE / 'capacity-diagnosis.png').read_bytes()).decode()
    (EVIDENCE / 'capacity-diagnosis.html').write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
        '<div style="font-size:13px;color:var(--foreground)">'
        '<img style="display:block;width:100%;border-radius:var(--radius)" '
        'alt="Four capacity runs: training loss decreases while larger models fail to improve validation NDCG" '
        f'src="data:image/png;base64,{encoded}">'
        '<p>Earlier matched capacity tests · seed 42 · 10 epochs · validation only. '
        'Training loss includes dropout; the panels show trends, not a directly comparable train/validation loss gap.</p></div></body></html>')


if __name__ == '__main__':
    build()
