"""Write the final comparison/telemetry sections from completed campaign evidence."""
import json
from pathlib import Path
import numpy as np

BASE=Path('experiments/e1_e5')


def cell(value,scale=1,digits=2):
    result=f"{value['mean']*scale:.{digits}f}"
    return result+(f" ± {value['sample_sd']*scale:.{digits}f}" if value['sample_sd'] is not None else '')


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    pipeline=json.loads((BASE/'pipeline_latency.json').read_text())
    aggregates={entry['name']:entry for entry in summary['aggregates']}
    names=['S0','E3mean','E3attention','E4k1','E4k2','E4k3','E5k2','E5k3']
    sections=['\n## Completed retrieval comparisons\n\nValues are mean ± sample SD where N=3. N=1 entries are screening results, with no estimated seed SD. All recall values in this table concern unseen targets and retain the full denominator of 1,857.\n',
              '| Configuration | Seeds | Hits@25 | Recall@25 | Hits@50 | Recall@50 |\n|---|---:|---:|---:|---:|---:|']
    for name in names:
        a=aggregates[name]
        sections.append(f"| {name} | {len(a['seeds'])} | {cell(a['new_hits@25'])} | {cell(a['stage1_new_recall@25'],100)}% | {cell(a['new_hits@50'])} | {cell(a['stage1_new_recall@50'],100)}% |")
    sections.extend(['\nEach model\'s own mixed slate and separately trained ranker produce the following final Top-10 results. Conditional recall uses the unseen targets present in that model\'s slate. The table reports the selected discovery checkpoint, including its NDCG, rather than combining peaks from different epochs.\n',
                     '| Configuration | Seeds | Overall Recall@10 | NDCG@10 | New hits@10 | Conditional new Recall@10 | New targets in slate | Guard passes |\n|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name in names:
        a=aggregates[name]; d=a['downstream']
        sections.append(f"| {name} | {d['new_hits@10']['n']} | {cell(d['val_recall@10'],100,3)}% | {cell(d['val_ndcg@10'],1,6)} | {cell(d['new_hits@10'])} | {cell(d['conditional_new_recall@10'],100,3)}% | {cell(d['new_targets_in_slate'])} | {d['guardrail_passes']}/{d['new_hits@10']['n']} |")
    sections.append('''
E3's mean and attention models both retrieved 700/1,857 at 25 and 1,002/1,857 at 50 in the screen, versus S0's 707 and 1,005. Removing their older-memory representation at fixed weights reduced those counts only to 697 and 1,000. Attention's distribution stayed close to frequency-weighted mean pooling: mean total-variation distance 0.001311, maximum 0.004382, and mean KL divergence 0.0000066 nats on the 3,211 validation prefixes with older history. The key/query/value route receives gradients; the failure is not an accidentally disconnected branch. This particular attention formulation did not learn strongly selective older-event weighting.

The one-query control was also below the original pooling control downstream. Every K=2/K=3 model had **100% of inter-interest vector pairs above cosine 0.99**. Mean cosines were 0.999996–0.9999998, with Top-25 Jaccard overlap of 99.65–99.93%. Query parameters themselves differed (for example, query cosine about 0.35 in hard K=2), but their final customer representations converged to essentially the same vector. Balanced winner frequencies therefore do not demonstrate specialization.

At fixed selected weights, using only the first interest changed unseen hits@25/50 by at most one; switching hard/smooth aggregation also changed at most one. Smooth scoring gave both interests gradients, yet did not resolve collapse. The evidence supports rejecting this implementation of multiple interests. It does **not** prove that one customer vector is sufficient in every possible architecture: this formulation failed to learn distinct interests in the first place. Shared summaries/fusion and the absence of a specialization incentive are plausible causes, rather than established causal explanations.

No E3–E5 extension passed the preregistered repetition screen: at least 10 additional Stage 1 hits@25 or @50, or at least five additional downstream hits with the fixed recall guardrail, relative to S0. All were therefore retained as single-seed negative screens. S0 was nevertheless repeated at three seeds to quantify the corrected training protocol's variability. E1/E2 had already been repeated at all three seeds because they were inexpensive and showed initial gains. No screening threshold was used to promote a model.

## Training, serving and cache costs

Training times below include per-epoch validation and exclude one-time input/slate cache construction, final diagnostics and latency sampling. Stage 1 has 30 epochs; each downstream ranker has 10. Model timings are synchronized, warm measurements on MPS. They are workstation telemetry, not isolated hardware throughput claims.

| Configuration | Stage 1 allocated parameters | Stage 1 training (s) | Downstream training (s) | Representation bytes/customer |
|---|---:|---:|---:|---:|''')
    for name in names:
        a=aggregates[name]
        sections.append(f"| {name} | {a['parameters']['mean']:,.0f} | {cell(a['training_seconds'])} | {cell(a['downstream']['training_seconds'])} | {a['cache_bytes_per_customer']['mean']:,.0f} |")
    sections.extend(['\nSeed-42 serving timings (median / p95 milliseconds) distinguish a purchase update from a recommendation request. Requests score the 841 development SKUs and apply the same rolling pool of 735–771 eligible SKUs. Full requests use cached customer vectors and the production rolling-eligibility, mixed-slate and pairwise-feature builders, followed by Stage 2 and Top-10 selection. They make zero Stage 1 tower calls. Network, serialization and storage I/O are excluded; models and feature stores are warm.\n',
                     '| Configuration | Cached Stage 1 score | Purchase encoding | Complete cached request |\n|---|---:|---:|---:|'])
    for name in names:
        r=next(row['result'] for row in summary['runs'] if row['name']==name and row['seed']==42)
        def timing(v): return f"{v['median_ms']:.3f} / {v['p95_ms']:.3f}"
        sections.append(f"| {name} | {timing(r['latency']['cached_similarity'])} | {timing(r['latency']['purchase_encoding'])} | {timing(pipeline[f'{name}-seed42'])} |")
    sections.extend(['\nE1/E2 use the same cached slate, so their neural forward is measured separately (B=1, 50 candidates; seed 42):\n',
                     '| Configuration | Allocated parameters | 10-epoch training (s), three-seed mean ± SD | Ranker latency, median / p95 ms |\n|---|---:|---:|---:|'])
    for name in ('control','unseen','transition_residual','transition_mlp'):
        a=aggregates[name]; r=next(row['result'] for row in summary['runs'] if row['name']==name and row['seed']==42)
        l=r['latency']['1']
        sections.append(f"| {name} | {a['parameters']['mean']:,.0f} | {cell(a['training_seconds'])} | {l['median_ms']:.3f} / {l['p95_ms']:.3f} |")
    resources=summary['resources']
    disk=sum(p.stat().st_size for p in Path('checkpoints/e1-e5').rglob('*') if p.is_file())
    sections.append(f'''
Representation sizes are float32 payloads, excluding the common feature store, attribute sets, dictionary keys, purchase IDs and allocator overhead. S0 stores one 128D vector (512 bytes); E3 stores that vector plus a separate 128D older-memory summary (1,024 bytes); K=2/K=3 store 1,024/1,536 bytes. The sparse 841-product vector table costs 430,592 bytes. A full 13,729-product table would cost {resources['stage1_full_catalogue_product_vectors_bytes']:,} bytes. Catalogue refreshes remap biases by canonical SKU, assign zero bias to new SKUs and require a new purchase-cache namespace; the old model/cache remains intact during rebuilding.

E2's dense count arrays cost {resources['transition_table_dense_count_bytes']:,} bytes. Cached development transition features for all 50-item train/validation slates cost {resources['transition_feature_cache_development_50_candidates_bytes']:,} bytes. If transition scores are cached for the entire expanded catalogue, they add {resources['transition_full_catalogue_customer_cache_bytes']:,} float32 bytes per customer; restricting the payload to the 841 active development SKUs requires 3,364 bytes plus the shared mapping/default handling. The measured full-catalogue transition-score refresh costs {resources['transition_purchase_refresh_all_catalogue_median_ms']:.3f} ms median / {resources['transition_purchase_refresh_all_catalogue_p95_ms']:.3f} ms p95 on CPU. This is separate from the ranker-only timings above.

All downstream original-architecture rankers allocate 866,313 parameters. Counts include 201,984 attached ProductTower parameters bypassed by cached candidate vectors and other checkpoint-compatible branches. The frozen external BGE text encoder is preprocessing, not part of these trained tower/ranker counts. The campaign's saved checkpoints, per-transition purchase caches and slates currently occupy {disk/(1024**3):.2f} GiB under `checkpoints/e1-e5/`; these audit/training snapshots are deliberately larger than the production per-customer cache.

## Decision

**Keep the original architectural shape: one normalized 128D customer vector, the existing 32-event Transformer with original pooling, and the original 13-feature UPS-CANR ranker.** E1 adds no information and has no demonstrated direct calibration benefit; E3 adds no useful screen-level gain; E4/E5 converge to duplicate interests. None warrants additional production complexity from these results.

E2's MLP feature is the most consistent small positive signal against freshly trained frozen-retriever controls. Keep its implementation as an opt-in experiment, not a default feature: the paired customer-cluster intervals include zero, its mean remains below the preserved 209-hit discovery checkpoint, and its result is conditional on the old retriever. The residual is less stable and has a slightly worse overall-recall tradeoff.

The demand-disabled S0 protocol removes the identified demand look-ahead path while respecting purchase-only caching and improves Stage 1 coverage relative to the preserved frozen reference. Its downstream ranking remains seed-sensitive and does not establish a final-system improvement over the preserved discovery checkpoint. Treat it as a correctness-oriented experimental training baseline, not an automatically approved replacement checkpoint. Demand and eligibility corrections were introduced together, so their individual contributions are not identified here. Historical text/catalogue metadata availability remains outside this experiment.

The principal finding is that added candidates are not converting reliably into additional final Top-10 discovery hits, while the tested multi-interest optimization does not produce distinct interests. This provides no reproducible basis for widening the maintained system with these branches. No experimental model was promoted and no default checkpoint alias was changed.

## Verification and inspectable evidence

The safe regression suite ran before training; the final suite passes **420 tests and 70 subtests**, with six unsplit real-data readers skipped. Tests cover zero initialization, residual/MLP gradients, strict destination-date transition counts, training-only global fitting, future-outcome mutation invariance, unchanged short-term outputs, older/recent exclusion, count-compressed pooling equivalence, demand invariance, request-clock invariance, K=1 score equivalence, active attention gradients, collapse diagnostics, cache persistence/version rejection and global catalogue remapping.

Every selected fitted checkpoint was strictly reloaded and its validation metrics replayed at fit completion. Read-only reference replay reconciles the local epoch-4/epoch-7 checkpoints with the historical documentation. All seven audited checkpoint/cache SHA-256 hashes remain unchanged. No held-out targets, fields or statistics were inspected or evaluated; the existing date gate discards excluded records before pandas materialization.

- [Commands and implementation guide](experiments/e1_e5/README.md)
- [Aggregate metrics, seed SD, paired intervals, resources and provenance](experiments/e1_e5/summary.json)
- [Compact comparison table](experiments/e1_e5/comparison_table.md)
- [Reference replay](experiments/e1_e5/reference_replay.json)
- [Scoring/head ablations](experiments/e1_e5/scoring_ablations.json)
- [Older-attention diagnostics](experiments/e1_e5/attention_diagnostics.json)
- [Full cached-request latency](experiments/e1_e5/pipeline_latency.json)
- [Repetition selection rule and outcome](experiments/e1_e5/repetition_selection.json)

The customer-cluster bootstrap uses 2,000 paired resamples. Its intervals characterize selected fixed models and do not include model-selection or training-seed uncertainty; repeated-seed SD is reported separately. Single-seed negative screens are evidence against retaining these tested configurations, not universal proofs that longer history or multiple interests can never help.
''')
    path=Path('E1_E5_EXPERIMENTS.md'); text=path.read_text()
    start='Working report: E1/E2 are complete; the E3–E5 campaign is running. Final conclusions will replace this status after all selected repetitions finish.'
    fit_count=len(summary['runs'])+sum(1 for _ in Path('checkpoints/e1-e5/retrieval-v1').glob('*/downstream.json'))
    text=text.replace(start,f'Completed {fit_count} controlled fits: 12 E1/E2 rankers, 10 independent retrieval fits, and their 10 downstream rankers. No experimental neural extension established a reproducible improvement that warrants promotion.')
    text=text.replace('Final E3–E5 tables, telemetry, repetitions and architecture decision are pending completion of the running campaign.','')
    marker='\n## Completed retrieval comparisons'
    if marker in text: text=text.split(marker)[0]
    path.write_text(text+'\n'.join(sections)+'\n')
    print(path,fit_count)


if __name__=='__main__': main()
