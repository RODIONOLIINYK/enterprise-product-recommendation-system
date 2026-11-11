# E1–E5 validation experiments — 2026-11-11–12

Completed 32 controlled fits: 12 E1/E2 rankers, 10 independent retrieval fits, and their 10 downstream rankers. No experimental neural extension established a reproducible improvement that warrants promotion.

The implementation and reproduction commands are in [the campaign guide](experiments/e1_e5/README.md). Experimental checkpoints are under `checkpoints/e1-e5/`, with a separate namespace for signals and retrieval. The production architecture and default aliases have not changed. The held-out test split was not inspected or evaluated.

## Reference reconciliation

Read-only replay reproduced Stage 1 **652/1,857 at Top-25 unseen (35.1104%)** and **921/1,857 at Top-50 unseen (49.5961%)**. The local Stage 2 discovery checkpoint is epoch 4 with **209 hits**, **76.1505% Recall@10** and **0.591893 NDCG@10**. The local NDCG-selected alias is epoch 7 with **201 hits**, **76.0417% Recall@10** and **0.593815 NDCG@10**. The historical epoch-10 result is not substituted for either local artifact. Replay evidence is in [reference_replay.json](experiments/e1_e5/reference_replay.json).

All seven audited checkpoint/cache files retained their original SHA-256 hashes. Original shared-experiment working-tree changes and caches were also retained. Experimental customer/candidate representations are versioned separately.

## E1–E2: frozen-retriever comparisons

These experiments share all 26,263 training and 6,191 validation slates, the 50 candidates per basket, the 13 original pairwise features, the original loss/optimizer budget, and the fixed overall-recall selection guardrail. They remain conditional on the reference retriever's previously identified training-demand look-ahead. The new transition statistic itself is strictly point-in-time.

Each fit trains for 10 epochs. Seeds 42, 17 and 2026 receive identically seeded batch orders across variants. Each selected checkpoint was strictly reloaded and replayed. The independent shuffle generator makes batch order comparable across architectures; it does not reproduce the historical trainer's global-RNG draw order. A seed-42 fresh control is therefore distinct from replaying the preserved seed-42 checkpoint. Numbers below are mean ± sample standard deviation across three seeds.

| Configuration | Overall Recall@10 | NDCG@10 | New hits@10 / 1,857 | Conditional new Recall@10 |
|---|---:|---:|---:|---:|
| Fresh original ranker control | 75.979 ± 0.310% | 0.591956 ± 0.001618 | 196.33 ± 5.69 | 25.732 ± 0.745% |
| E1: trainable unseen scalar | 76.050 ± 0.089% | 0.593508 ± 0.001177 | 202.33 ± 0.58 | 26.518 ± 0.076% |
| E2: transition residual | 75.846 ± 0.202% | 0.591728 ± 0.001872 | 201.67 ± 6.11 | 26.431 ± 0.801% |
| E2: transition in scoring MLP | 75.973 ± 0.019% | 0.592487 ± 0.000821 | 203.67 ± 1.53 | 26.693 ± 0.200% |

The common mixed slate contains 763 unseen targets; conditional recall divides by 763, while end-to-end new recall always divides by all 1,857 targets. E1/E2 do not change Stage 1 or candidate coverage.

E1 gains 13, 4 and 1 hits over its matched controls. Its learned scalar is negative, approximately −0.2. At fixed selected weights, setting that scalar to zero does not remove the observed gains consistently. This points toward an optimization/parameterization effect rather than a beneficial positive unseen-product bonus. It does not justify a permanent calibration term by itself. A plausible explanation for its negative sign is that the BCE/listwise objective learns the lower base rate of unseen candidates rather than a positive discovery preference.

E2's MLP variant gains 15, 6 and 1 hits; the residual gains 5, 9 and 2. Removing the transition input at fixed weights reduces hits materially in the first two seeds, showing use of the signal. These perturbations are distinct from retraining without the feature. None of the per-seed paired customer-cluster 95% intervals excludes zero for new recall, and none of the three-seed mean hit counts exceeds the preserved discovery checkpoint's 209. The MLP is the more stable transition variant in this screen, with modest evidence rather than an established replacement. The table pools cross-customer information absent from the original pairwise features; the MLP can condition it on repeat/familiarity context, while one residual coefficient applies a global adjustment. Estimation noise and full-history aggregation may dilute immediate transitions. These are plausible explanations, not separately identified causal effects.

Transition counts use adjacent date-grouped baskets. The destination basket must occur strictly before the example's prediction date. Validation uses global counts frozen at the training cutoff, while customer history can include strictly earlier validation purchases. The background probability uses `(N_B+0.5)/(M+1)`, alpha is fixed at 20, and customer source weights are normalized past-event frequencies. Sources and destinations are basket sets; multiple lines for the same SKU cannot inflate a transition count. Tests mutate future outcomes and assert unchanged earlier scores.

## E3–E5: separate demand-disabled training protocol

S0 starts from fresh neural weights and retains the original 32-event Transformer and mean-plus-last pooling. All six demand inputs are forcibly zero in training and the product forward; packaging is retained. This removes the specific demand look-ahead path and supports frozen product vectors between global refreshes. Training masks also enforce prior-sale/blocked eligibility. Consequently, comparisons against the old frozen reference change both the demand policy and training eligibility; they cannot attribute an improvement to either correction alone. Snapshot text/catalogue metadata and training-fitted scales are inherited, so this is not a claim that historical metadata availability has been audited.

Every E3–E5 comparison uses the same training rows, validation baskets, candidate eligibility and complete target denominators. Products absent from the catalogue or eligible pool remain misses. A sparse 841-product development table includes every SKU that can pass development eligibility; its first 790 IDs are the original training catalogue. Additional validation-only SKU biases remain zero during training. Model inputs never contain a recommendation clock.

E3 compares no memory, older-event mean pooling and trainable attention pooling, preserving the recent 32-event Transformer. Older history is available for 8,510 training and 3,211 validation transitions. Repeated older product vectors are represented by SKU/count pairs; tests prove equivalence to explicitly repeated tokens. E4 independently compares one, two and three learned interest queries over recent contextual history. E4k1 isolates attention pooling from the number of vectors. E5 fits hard-max and smooth-max models separately at K=2 and K=3; smooth scoring uses tau=0.1 log-mean-exp before the shared temperature, removing the K-dependent constant offset. No E4/E5 model depends on E3 memory.

Each Stage 1 fit has 30 epochs and is selected by new hits@25, then hits@50. Its own purchase-vector caches generate new mixed training/validation slates through the existing production constructor. A separate original-architecture Stage 2 ranker is then trained for 10 epochs, with matching profile projections transferred from the selected clean retriever. Checkpoints failing the fixed overall-recall guardrail are labeled diagnostic fallbacks and cannot qualify for recommendation.

The first S0 screen selects epoch 13: 707 unseen hits@25 (38.0722%) and 1,005@50 (54.1195%). Its mixed slate covers 834 unseen targets; the downstream ranker selects 207 hits, 76.1258% overall Recall@10 and 0.591995 NDCG@10. Its conditional new recall is 24.8201%. This illustrates that additional candidate coverage does not automatically yield more final discovery hits.



## Completed retrieval comparisons

Values are mean ± sample SD where N=3. N=1 entries are screening results, with no estimated seed SD. All recall values in this table concern unseen targets and retain the full denominator of 1,857.

| Configuration | Seeds | Hits@25 | Recall@25 | Hits@50 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| S0 | 3 | 719.00 ± 14.42 | 38.72 ± 0.78% | 980.00 ± 23.26 | 52.77 ± 1.25% |
| E3mean | 1 | 700.00 | 37.70% | 1002.00 | 53.96% |
| E3attention | 1 | 700.00 | 37.70% | 1002.00 | 53.96% |
| E4k1 | 1 | 708.00 | 38.13% | 1002.00 | 53.96% |
| E4k2 | 1 | 699.00 | 37.64% | 983.00 | 52.93% |
| E4k3 | 1 | 694.00 | 37.37% | 993.00 | 53.47% |
| E5k2 | 1 | 704.00 | 37.91% | 972.00 | 52.34% |
| E5k3 | 1 | 703.00 | 37.86% | 974.00 | 52.45% |

Each model's own mixed slate and separately trained ranker produce the following final Top-10 results. Conditional recall uses the unseen targets present in that model's slate. The table reports the selected discovery checkpoint, including its NDCG, rather than combining peaks from different epochs.

| Configuration | Seeds | Overall Recall@10 | NDCG@10 | New hits@10 | Conditional new Recall@10 | New targets in slate | Guard passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 3 | 75.919 ± 0.259% | 0.591359 ± 0.002065 | 194.67 ± 11.24 | 23.132 ± 1.590% | 842.00 ± 10.58 | 3/3 |
| E3mean | 1 | 75.825% | 0.592677 | 188.00 | 22.843% | 823.00 | 1/1 |
| E3attention | 1 | 76.099% | 0.591963 | 202.00 | 24.544% | 823.00 | 1/1 |
| E4k1 | 1 | 75.819% | 0.592907 | 193.00 | 23.114% | 835.00 | 1/1 |
| E4k2 | 1 | 75.692% | 0.590724 | 195.00 | 23.636% | 825.00 | 1/1 |
| E4k3 | 1 | 75.639% | 0.591408 | 190.00 | 22.700% | 837.00 | 1/1 |
| E5k2 | 1 | 76.018% | 0.592051 | 205.00 | 24.818% | 826.00 | 1/1 |
| E5k3 | 1 | 75.680% | 0.592204 | 198.00 | 23.377% | 847.00 | 1/1 |

E3's mean and attention models both retrieved 700/1,857 at 25 and 1,002/1,857 at 50 in the screen, versus S0's 707 and 1,005. Removing their older-memory representation at fixed weights reduced those counts only to 697 and 1,000. Attention's distribution stayed close to frequency-weighted mean pooling: mean total-variation distance 0.001311, maximum 0.004382, and mean KL divergence 0.0000066 nats on the 3,211 validation prefixes with older history. The key/query/value route receives gradients; the failure is not an accidentally disconnected branch. This particular attention formulation did not learn strongly selective older-event weighting.

The one-query control was also below the original pooling control downstream. Every K=2/K=3 model had **100% of inter-interest vector pairs above cosine 0.99**. Mean cosines were 0.999996–0.9999998, with Top-25 Jaccard overlap of 99.65–99.93%. Query parameters themselves differed (for example, query cosine about 0.35 in hard K=2), but their final customer representations converged to essentially the same vector. Balanced winner frequencies therefore do not demonstrate specialization.

At fixed selected weights, using only the first interest changed unseen hits@25/50 by at most one; switching hard/smooth aggregation also changed at most one. Smooth scoring gave both interests gradients, yet did not resolve collapse. The evidence supports rejecting this implementation of multiple interests. It does **not** prove that one customer vector is sufficient in every possible architecture: this formulation failed to learn distinct interests in the first place. Shared summaries/fusion and the absence of a specialization incentive are plausible causes, rather than established causal explanations.

No E3–E5 extension passed the preregistered repetition screen: at least 10 additional Stage 1 hits@25 or @50, or at least five additional downstream hits with the fixed recall guardrail, relative to S0. All were therefore retained as single-seed negative screens. S0 was nevertheless repeated at three seeds to quantify the corrected training protocol's variability. E1/E2 had already been repeated at all three seeds because they were inexpensive and showed initial gains. No screening threshold was used to promote a model.

## Training, serving and cache costs

Training times below include per-epoch validation and exclude one-time input/slate cache construction, final diagnostics and latency sampling. Stage 1 has 30 epochs; each downstream ranker has 10. Model timings are synchronized, warm measurements on MPS. They are workstation telemetry, not isolated hardware throughput claims.

| Configuration | Stage 1 allocated parameters | Stage 1 training (s) | Downstream training (s) | Representation bytes/customer |
|---|---:|---:|---:|---:|
| S0 | 825,133 | 50.39 ± 5.13 | 99.87 ± 3.64 | 512 |
| E3mean | 874,285 | 47.42 | 103.05 | 1,024 |
| E3attention | 890,797 | 51.02 | 102.69 | 1,024 |
| E4k1 | 825,261 | 49.86 | 96.07 | 512 |
| E4k2 | 825,389 | 47.59 | 99.12 | 1,024 |
| E4k3 | 825,517 | 48.32 | 97.84 | 1,536 |
| E5k2 | 825,389 | 45.90 | 98.32 | 1,024 |
| E5k3 | 825,517 | 45.47 | 105.07 | 1,536 |

Seed-42 serving timings (median / p95 milliseconds) distinguish a purchase update from a recommendation request. Requests score the 841 development SKUs and apply the same rolling pool of 735–771 eligible SKUs. Full requests use cached customer vectors and the production rolling-eligibility, mixed-slate and pairwise-feature builders, followed by Stage 2 and Top-10 selection. They make zero Stage 1 tower calls. Network, serialization and storage I/O are excluded; models and feature stores are warm.

| Configuration | Cached Stage 1 score | Purchase encoding | Complete cached request |
|---|---:|---:|---:|
| S0 | 0.213 / 0.532 | 1.270 / 1.453 | 8.537 / 9.806 |
| E3mean | 0.207 / 0.501 | 1.304 / 1.446 | 7.968 / 8.736 |
| E3attention | 0.213 / 0.540 | 1.204 / 1.562 | 8.657 / 16.530 |
| E4k1 | 0.216 / 0.525 | 1.403 / 1.586 | 7.638 / 8.675 |
| E4k2 | 0.654 / 0.777 | 1.434 / 2.174 | 7.639 / 7.912 |
| E4k3 | 0.474 / 0.759 | 1.459 / 2.299 | 7.594 / 8.128 |
| E5k2 | 0.468 / 0.529 | 1.266 / 1.686 | 11.054 / 13.470 |
| E5k3 | 0.450 / 0.473 | 1.076 / 1.192 | 8.916 / 9.961 |

E1/E2 use the same cached slate, so their neural forward is measured separately (B=1, 50 candidates; seed 42):

| Configuration | Allocated parameters | 10-epoch training (s), three-seed mean ± SD | Ranker latency, median / p95 ms |
|---|---:|---:|---:|
| control | 866,313 | 97.35 ± 2.27 | 1.507 / 2.285 |
| unseen | 866,314 | 98.60 ± 3.86 | 1.525 / 1.603 |
| transition_residual | 866,314 | 97.00 ± 2.96 | 1.214 / 1.707 |
| transition_mlp | 866,569 | 98.50 ± 2.90 | 1.523 / 2.145 |

Representation sizes are float32 payloads, excluding the common feature store, attribute sets, dictionary keys, purchase IDs and allocator overhead. S0 stores one 128D vector (512 bytes); E3 stores that vector plus a separate 128D older-memory summary (1,024 bytes); K=2/K=3 store 1,024/1,536 bytes. The sparse 841-product vector table costs 430,592 bytes. A full 13,729-product table would cost 7,029,248 bytes. Catalogue refreshes remap biases by canonical SKU, assign zero bias to new SKUs and require a new purchase-cache namespace; the old model/cache remains intact during rebuilding.

E2's dense count arrays cost 5,671,704 bytes. Cached development transition features for all 50-item train/validation slates cost 6,490,800 bytes. If transition scores are cached for the entire expanded catalogue, they add 54,916 float32 bytes per customer; restricting the payload to the 841 active development SKUs requires 3,364 bytes plus the shared mapping/default handling. The measured full-catalogue transition-score refresh costs 1.009 ms median / 1.041 ms p95 on CPU. This is separate from the ranker-only timings above.

All downstream original-architecture rankers allocate 866,313 parameters. Counts include 201,984 attached ProductTower parameters bypassed by cached candidate vectors and other checkpoint-compatible branches. The frozen external BGE text encoder is preprocessing, not part of these trained tower/ranker counts. The campaign's saved checkpoints, per-transition purchase caches and slates currently occupy 5.04 GiB under `checkpoints/e1-e5/`; these audit/training snapshots are deliberately larger than the production per-customer cache.

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
