"""Complete-denominator metrics and customer-clustered paired uncertainty."""
import numpy as np
import torch


def ranking_metrics(scores, data, indices=None, save=None):
    scores = scores.detach().cpu()
    if indices is None:
        indices = torch.arange(scores.shape[1]).expand(len(scores), -1)
        scores = scores.masked_fill(~data['allowed'], -torch.inf)
    order = scores.argsort(dim=-1, descending=True, stable=True)
    ranked = indices.gather(1, order)
    eligible_ranked = data['allowed'].gather(1, ranked)
    hits = (data['target'].gather(1, ranked) & eligible_ranked).numpy()
    novel = (data['new_target'].gather(1, ranked) & eligible_ranked).numpy()
    source = data['source']
    count = np.array([len(x) for x in source['positive_pids']])
    new_count = np.array([len(x) for x in source['new_positive_pids']])
    repeat_count = count - new_count
    new_hits = novel[:, :10].sum(1)
    recall = hits[:, :10].sum(1) / count
    discount = 1 / np.log2(np.arange(10) + 2)
    ideal = (discount[None] * (np.arange(10)[None] < count[:, None])).sum(1)
    ndcg = (hits[:, :10] * discount).sum(1) / ideal
    any_hit = hits.any(1)
    mrr = np.where(any_hit, 1 / (hits.argmax(1) + 1), 0)
    eligible_indices = data['allowed'].gather(1, indices)
    present = (data['target'].gather(1, indices) & eligible_indices).sum(1).numpy()
    new_present = (data['new_target'].gather(1, indices) & eligible_indices).sum(1).numpy()
    result = {'recall@5': float((hits[:, :5].sum(1) / count).mean()), 'recall@10': float(recall.mean()),
              'ndcg@10': float(ndcg.mean()), 'mrr': float(mrr.mean()), 'new_hits@10': int(new_hits.sum()),
              'new_targets': int(new_count.sum()), 'new_recall@10': float(new_hits.sum() / max(new_count.sum(), 1)),
              'conditional_new_recall@10': float(new_hits.sum() / max(new_present.sum(), 1)),
              'repeat_recall@10_micro': float((hits[:, :10].sum() - new_hits.sum()) / max(repeat_count.sum(), 1)),
              'repeat_recall@10_macro': float(np.mean((hits[:, :10].sum(1)-new_hits)[repeat_count>0]/repeat_count[repeat_count>0])) if bool((repeat_count>0).any()) else None,
              'mixed_coverage_macro': float((present / count).mean()), 'mixed_coverage_micro': float(present.sum() / count.sum()),
              'new_targets_in_slate': int(new_present.sum()), 'new_slate_coverage': float(new_present.sum() / max(new_count.sum(), 1)),
              'new_reachable_lost_at10': int(new_present.sum() - new_hits.sum())}
    if scores.shape[1] == data['allowed'].shape[1]:
        unseen_scores = scores.masked_fill(~data['unseen'], -torch.inf)
        unseen_order = unseen_scores.argsort(dim=-1, descending=True, stable=True)
        for k in (25, 50, 100):
            top = unseen_order[:, :k]
            valid = data['unseen'].gather(1, top)
            n = int((data['new_target'].gather(1, top) & valid).sum())
            result[f'new_hits@{k}'] = n
            result[f'new_recall@{k}'] = n / max(int(new_count.sum()), 1)
    cohorts = {}
    observed = source['attention_mask'].sum(1).numpy()
    for name, mask in (('cold_0', observed == 0), ('light_1_8', (observed > 0) & (observed <= 8)), ('warm_9_31', (observed > 8) & (observed < 32)), ('truncated_32', observed == 32)):
        cohorts[name] = {'rows': int(mask.sum()), 'recall@10': float(recall[mask].mean()) if mask.any() else None,
                         'new_hits@10': int(new_hits[mask].sum()), 'new_targets': int(new_count[mask].sum())}
    result['history_cohorts'] = cohorts
    if 'prediction_days' in source:
        dates=np.datetime64('2025-01-01')+source['prediction_days'].numpy().astype(np.int64).astype('timedelta64[D]')
        months=dates.astype('datetime64[M]').astype(str)
        result['validation_months']={month:{'rows':int((months==month).sum()),
            'recall@10':float(recall[months==month].mean()),'ndcg@10':float(ndcg[months==month].mean()),
            'new_hits@10':int(new_hits[months==month].sum()),'new_targets':int(new_count[months==month].sum())}
            for month in sorted(set(months))}
    if 'features' in data:
        # Target-level cohorts use only earlier feature values, never outcome
        # counts. Unrepresented catalogue targets remain in overall denominators.
        target_rows, target_ids = torch.where(data['new_target'])
        values = np.array(data['features'][target_rows.numpy(), target_ids.numpy()])
        selected = ranked[:, :10]
        reached = (selected[target_rows] == target_ids[:, None]).any(1).numpy()
        rare = values[:, 26] <= np.log1p(5) / np.log1p(1000)
        first_family = values[:, 22] > .5
        family_known=data.get('family_known',torch.ones(data['target'].shape[1],dtype=torch.bool))[target_ids].numpy()
        sparse = values[:, 38] <= .1
        result['new_target_cohorts'] = {
            name: {'targets': int(mask.sum()), 'hits@10': int(reached[mask].sum())}
            for name, mask in (('first_snapshot_family', first_family & family_known), ('known_snapshot_family', ~first_family & family_known),
                               ('unknown_snapshot_family_metadata', ~family_known),
                               ('sku_rare_frozen90d', rare), ('sku_frequent_frozen90d', ~rare),
                               ('sparse_transition', sparse), ('supported_transition', ~sparse))}
        result['new_target_cohorts']['unrepresented_catalogue'] = {'targets': int(new_count.sum()-len(target_ids)), 'hits@10':0}
    if save is not None:
        np.savez_compressed(save, customers=np.array(source['customer_ids']), recall=recall, ndcg=ndcg,
                            new_hits=new_hits, new_targets=new_count, new_present=new_present)
    return result


def paired_intervals(control, treatment, repeats=2000):
    a, b = np.load(control), np.load(treatment)
    if not np.array_equal(a['customers'], b['customers']) or not np.array_equal(a['new_targets'], b['new_targets']):
        raise ValueError('Paired predictions require identical ordered validation rows')
    _, groups = np.unique(a['customers'], return_inverse=True)
    n = groups.max() + 1
    rows = np.bincount(groups)
    new_targets = np.bincount(groups, weights=a['new_targets'])
    deltas = [np.bincount(groups, weights=b[key] - a[key]) for key in ('recall', 'ndcg', 'new_hits')]
    rng = np.random.default_rng(123)
    draws = rng.integers(n, size=(repeats, n))
    denominators = [rows[draws].sum(1), rows[draws].sum(1), new_targets[draws].sum(1)]
    return {name: np.quantile(values[draws].sum(1) / np.maximum(denom, 1), [.025, .5, .975]).tolist()
            for name, values, denom in zip(('recall@10_delta', 'ndcg@10_delta', 'new_recall@10_delta'), deltas, denominators)}
