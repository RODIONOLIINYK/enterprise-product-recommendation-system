"""Comparable eval-mode losses, data support, and a train-only memorization probe."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from recommender.artifacts import atomic_json
from recommender.experiments.candidate_token_enrichment.model import load_enrichment_ranker_checkpoint
from recommender.experiments.candidate_token_enrichment.train import build_initial_arms, snapshot_preserved_hashes
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader, rolling_ranker_batch_inputs,
    prepare_or_load_rolling_feature_bundle, prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
)
from recommender.ranker import RankerLoss
from .model import load_checkpoint
from .train import EVIDENCE


def sliced(slate, rows):
    n = len(slate['customer_ids'])
    return {k: v[rows] if isinstance(v, torch.Tensor) and v.ndim and len(v) == n
            else [v[int(i)] for i in rows] if isinstance(v, list) and len(v) == n
            else v for k, v in slate.items()}


@torch.no_grad()
def fit_metrics(model, slate, vectors, device):
    model.eval()
    totals = {'loss': 0., 'bce_loss': 0., 'list_loss': 0.}
    hits, positive, new_hits, new_positive, rows = 0, 0, 0, 0, 0
    for batch in make_rolling_slate_loader(slate, batch_size=64, shuffle=False):
        x = rolling_ranker_batch_inputs(batch, vectors, device)
        labels = batch['labels'].to(device)
        logits = model(**x).logits
        n = len(labels); rows += n
        loss = RankerLoss()(logits, labels, candidate_mask=x['candidate_mask'])
        for key in totals:
            totals[key] += float(loss[key]) * n
        top = logits.topk(10, dim=1).indices
        hits += int(labels.gather(1, top).sum()); positive += int(labels.sum())
        new = labels * (x['pairwise_features'][..., 0] < .5)
        new_hits += int(new.gather(1, top).sum()); new_positive += int(new.sum())
    return {**{k: v / rows for k, v in totals.items()}, 'rows': rows,
            'slate_conditional_recall@10': hits / max(positive, 1),
            'slate_conditional_new_recall@10': new_hits / max(new_positive, 1),
            'reachable_positive_instances': positive, 'reachable_new_instances': new_positive}


def run():
    torch.set_num_threads(2)
    summary = json.loads((EVIDENCE / 'summary.json').read_text())
    before = snapshot_preserved_hashes()
    if summary['preserved_hashes'] != before:
        raise ValueError('Protected snapshot changed.')
    data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data, info, meta)
    slates, manifest = prepare_or_load_rolling_slate_cache(data, info, meta)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    vectors = data['train']['v'].to(device)
    rng = np.random.default_rng(42)
    samples = {s: sliced(slates[s], np.sort(rng.choice(len(slates[s]['customer_ids']), 4096, replace=False)))
               for s in ('train', 'val')}
    support = {}
    for split, d in data.items():
        slate = slates[split]
        counts = d['target'].sum(0).numpy()
        support[split] = {
            'queries': len(slate['customer_ids']), 'distinct_customers': len(set(slate['customer_ids'])),
            'target_instances': int(d['target'].sum()), 'new_target_instances': int(d['new_target'].sum()),
            'reachable_target_instances': int(slate['labels'].sum()),
            'target_count_per_sku_median': float(np.median(counts)),
            'skus_with_at_most_10_target_instances': int((counts <= 10).sum()),
            'max_sku_target_instances': int(counts.max()),
        }
        if not torch.equal(slate['tabular_features'], d['source']['tabular_features']):
            raise AssertionError('Profile statistics differ from original query source.')
        features = d['features']
        for row in np.linspace(0, len(slate['customer_ids'])-1, 16).astype(int):
            expected = torch.tensor(np.asarray(features[row, slate['candidate_indices'][row].numpy()], dtype=np.float32))
            torch.testing.assert_close(slate['pairwise_features'][row], expected, rtol=0, atol=0)
    train_counts = data['train']['target'].sum(0)
    val_new = data['val']['new_target'].sum(0)
    support['validation_new_targets_by_training_support'] = {
        f'at_most_{n}_training_target_instances': int(val_new[train_counts <= n].sum())
        for n in (0, 5, 10, 20, 50)
    }
    support['validation_customers_unseen_in_training'] = len(set(slates['val']['customer_ids']) - set(slates['train']['customer_ids']))
    fits, replay = {}, []
    for name, arm in summary['arms'].items():
        selected = arm['ndcg']
        model, _ = load_checkpoint(selected['checkpoint'], expected_provenance=arm['provenance'],
                                   expected_feature_contract_digest=meta['feature_schema_digest'])
        model.to(device)
        fits[name] = {s: fit_metrics(model, samples[s], vectors, device) for s in ('train', 'val')}
        cpu_x = rolling_ranker_batch_inputs(next(iter(make_rolling_slate_loader(slates['val'], batch_size=32))),
                                           data['train']['v'], torch.device('cpu'))
        model.cpu().eval()
        saved = torch.load(selected['logits_pt'], map_location='cpu', weights_only=False)['logits']
        with torch.no_grad():
            diff = float((model(**cpu_x).logits - saved[:32]).abs().max())
        if diff > 1e-4:
            raise AssertionError('Independent CPU replay mismatch.')
        replay.append({'arm': name, 'cpu32_max_abs_diff': diff})
    baseline = load_enrichment_ranker_checkpoint(summary['baseline']['ndcg']['checkpoint'])[0].to(device)
    fits['saved_A'] = {s: fit_metrics(baseline, samples[s], vectors, device) for s in ('train', 'val')}
    # Deliberately remove stochastic dropout/weight decay for a train-only capacity test.
    # It is not a validation model or a proposed production training recipe.
    _, initial, _, _ = build_initial_arms(seed=42)
    probe = copy.deepcopy(initial).to(device).eval()
    tiny = sliced(slates['train'], np.sort(np.random.default_rng(17).choice(len(slates['train']['customer_ids']), 128, replace=False)))
    batches = list(make_rolling_slate_loader(tiny, batch_size=64, shuffle=False))
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    start = fit_metrics(probe, tiny, vectors, device)
    curve = []
    for step in range(1, 601):
        batch = batches[(step-1) % len(batches)]
        x = rolling_ranker_batch_inputs(batch, vectors, device)
        optimizer.zero_grad(set_to_none=True)
        loss = RankerLoss()(probe(**x).logits, batch['labels'].to(device), candidate_mask=x['candidate_mask'])['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.)
        optimizer.step()
        if step in (100, 300, 600):
            curve.append({'optimizer_steps': step, **fit_metrics(probe, tiny, vectors, device)})
    # Tree split importances describe the fitted model, not causal accuracy contributions.
    from catboost import CatBoostClassifier
    tree_summary = json.loads(Path('experiments/catboost_standalone_v1/summary.json').read_text())
    tree = CatBoostClassifier(); tree.load_model(tree_summary['selected']['ndcg']['checkpoint'])
    importance = dict(zip(tree.feature_names_, map(float, tree.get_feature_importance(thread_count=2))))
    groups = Counter()
    for name, value in importance.items():
        groups['profile8' if name.startswith('profile_') else 'categorical3'
               if name in ('product_id', 'product_category', 'business_line') else 'context41'] += value
    if snapshot_preserved_hashes() != before:
        raise AssertionError('Protected files changed.')
    atomic_json({
        'heldout_access': False, 'support': support, 'eval_mode_fit_metrics': fits,
        'sample_seed': 42, 'sample_rows_per_split': 4096, 'profile_and_context_parity': True,
        'cpu_replay': replay,
        'tiny_train_memorization_probe': {'rows': 128, 'lr': 1e-3, 'dropout_disabled': True,
                                         'weight_decay': 0, 'initial': start, 'curve': curve,
                                         'listwise_entropy_lower_bound': float(tiny['labels'].sum(1).clamp_min(1).log().mean()),
                                         'limitation': 'Capacity/optimizer sanity check only; not a full-data or generalization claim.'},
        'catboost_prediction_values_change': {'all_features': importance, 'groups': dict(groups),
                                              'caveat': 'Not causal, not percentage of predictive accuracy; correlated features can share credit.'},
        'all_preserved_hashes_match': True,
    }, EVIDENCE / 'diagnostics.json', overwrite=False)


if __name__ == '__main__':
    run()
