# Dynamic selection and candidate context: completed P0–P7 campaign

Completed 2026-11-12 against `3fbec83`: **68 full fits**, validation only. All model implementations, tests, selected-weight replays, paired-seed comparisons, ablations, latency measurements and cache checks are complete. No model or default alias was promoted. The later pasted review is addressed in [REVIEW_DISPOSITION.md](experiments/dynamic_campaign/REVIEW_DISPOSITION.md).

I recommend the **simpler demand-disabled, single-vector dot retriever → 100-item mixed slate → original UPS-CANR** as the conservative candidate architecture. Wider slates produced the clearest gain over matched fresh controls. Keep the added scalar scoring features as an optional experimental enhancement. They improve mean quality, but their increment over the depth-matched control is uncertain. The full-pool MLP selector and candidate self-attention do not earn their added complexity in these experiments.

The strongest mean validation configuration was the dynamic selector plus scalar-augmented 100-item ranker: 233.3 ± 6.7 discovery hits. Its advantage over simpler depth-matched alternatives has confidence intervals spanning zero and much higher complete-request latency. It is the accuracy-oriented research candidate, not an automatic replacement. The current 201-hit NDCG-selected default and 209-hit discovery checkpoint remain unchanged.

## Scope, controls and reproducibility

The common upstream is the saved, scratch-trained **S0-seed42 epoch-13** model with dynamic demand disabled and prior-sale eligibility masks. All new seeds vary selector/ranker fitting while holding that upstream fixed. P0 separately reproduced all three saved S0 selections and the legacy references. This avoids crediting demand removal and candidate-mask changes separately: their original S0 effects remain coupled.

Every new fit uses 26,253 training transitions and all 6,191 validation transitions. The existing development cache had 26,263 training rows; ten additional early rows were excluded from every new comparison so all experiments could use 100 eligible candidates. All 1,857 new-target instances remain in validation denominators, including unavailable targets. The original 26,264-transition population already had one row omitted by that source cache. Eligibility always uses strictly earlier genuine paid merchandise sales and the same blocking policy.

Global feature observations are strictly earlier during training and frozen at 2025-10-01 during validation; customer state incorporates strictly earlier purchases. Global sale age is measured at that aggregate refresh, not represented as a current-market observation. Customer and product vectors contain no request-clock inputs. Customer encoding occurs on purchase or an explicit versioned rebuild; the request path uses cached vectors and cheap as-of features.

Primary metadata is explicitly `snapshot_unverified`. P1 confirmed that purchase-observed mode gives zero brand-share, subcategory-share and industry-affinity features with these cleaned records; snapshot-mode nonzero rates on the sampled eligible pairs were 62.7%, 32.7% and 34.1%. Missing metadata is reported separately in family cohorts. Effective-date catalogue/company/text/block history is still unavailable. Results are conditional on those snapshots, not proof of historical metadata validity. Missing category/family age uses the declared never-seen sentinel; shares and unknown relationship lifts use zero.

New 41-feature inputs comprise the original 13 features, 20 timing/global/family features and eight relationship features. The full-pool MLP has 79,361 parameters and `554 → 128 → 64 → 1` geometry, using cached vectors, their product/absolute difference, features and the reference score. It trains on every eligible product, not on preselected 50-item slates. Its static control and dynamic/relationship variants share geometry. Frozen S0 is discovery-trained; the new selectors are all-target-trained. Their direct overall Top-10 scores are therefore not an isolated dot-versus-MLP objective comparison.

The new protocol uses a fixed `log1p(1000)` popularity scale and float16-to-float32 feature boundary consistently across its controls. Those changes, the clean upstream and common-row restriction mean its fresh control is not a claim to reproduce the historical 209-hit model. P0 separately replays the exact preserved references.

P3–P5 keep original S0 personal-repeat 50-item slates fixed. P2/P6 selector quotas rank repeat and unseen subsets by the selector score. The completed dot-repeat policy probe exposed lower overall quality; one incomplete component probe was archived before restarting independent component screens on original slates. Both the retained probe and the correction are documented. The 100-item policy reserves up to 50 repeat slots and fills the remainder with unseen candidates; changing width also changes allocation, so this is a wider-policy result rather than a pure tensor-padding effect.

Ten epochs per new selector/ranker, batch sizes 64, paired seeds 42/17/2026, identical losses and selection rules across comparable models. Both quality floors came directly from the preserved discovery artifact: Recall@10 ≥ 0.7558833 and NDCG@10 ≥ 0.5918929. Selected metrics always come from one checkpoint. Failed runs keep `best_unqualified.pt` diagnostics and publish no selected alias. One of three fresh original-slate controls failed the NDCG floor; it is retained and marked, not discarded.

Near-threshold repetitions and the later review controls are disclosed in [supplementary_protocol.json](experiments/dynamic_campaign/supplementary_protocol.json). They did not relax final acceptance. The pretrained residual comparison starts from identical clean control weights and uses a matched additional ten epochs at 0.0001; initial residual logits matched exactly. Its post-update checkpoints are compared with the preserved epoch-zero source separately. Neither fine-tuning branch improved that source.

## Selected-checkpoint results

Mean ± sample standard deviation. Recall is macro over baskets; new recall uses all 1,857 new targets; conditional recall divides by new targets in the slate. Historic rows are single preserved selections, not repeated-seed estimates. All new rows below have three seeds.

| Model | New hits@10 | Recall@10 | NDCG@10 | Conditional new recall | Repeat recall, micro | Qualifying seeds |
|---|---:|---:|---:|---:|---:|---:|
| Preserved default, epoch 7 | 201 | 0.760417 | 0.593815 | — | — | fixed |
| Preserved discovery, epoch 4 | 209 | 0.761505 | 0.591893 | — | — | fixed |
| Fresh S0 / original 50 slate | 188.0 ± 4.4 | 0.758502 ± 0.002493 | 0.591905 ± 0.001472 | 0.2254 ± 0.0052 | 0.8875 ± 0.0030 | 2/3 |
| Same 50 slate + scalar features | 216.7 ± 10.8 | 0.760419 ± 0.001414 | 0.594883 ± 0.000602 | 0.2598 ± 0.0129 | 0.8851 ± 0.0008 | 3/3 |
| Dot / 100 / original ranker | 214.0 ± 5.3 | 0.761997 ± 0.002197 | 0.592923 ± 0.000509 | 0.1692 ± 0.0042 | 0.8885 ± 0.0019 | 3/3 |
| Dot / 100 / scalar features | 227.7 ± 4.9 | 0.762542 ± 0.001153 | 0.595762 ± 0.001362 | 0.1800 ± 0.0039 | 0.8860 ± 0.0012 | 3/3 |
| Dynamic MLP / 100 / original ranker | 215.7 ± 9.2 | 0.762309 ± 0.001439 | 0.593869 ± 0.001028 | 0.1685 ± 0.0073 | 0.8879 ± 0.0019 | 3/3 |
| Dynamic MLP / 100 / scalar features | 233.3 ± 6.7 | 0.764186 ± 0.000939 | 0.596093 ± 0.000550 | 0.1823 ± 0.0052 | 0.8872 ± 0.0029 | 3/3 |
| Dynamic MLP / 100 / scalar + candidate attention | 239.0 ± 5.3 | 0.761306 ± 0.002229 | 0.587515 ± 0.002350 | 0.1868 ± 0.0043 | 0.8828 ± 0.0024 | 0/3 |

The complete [comparison table](experiments/dynamic_campaign/COMPARISON.md) includes every screened and repeated configuration, overall/new/conditional/repeat metrics and mixed-slate coverage. [Raw results](experiments/dynamic_campaign/results_enriched.json) also include Recall@5, MRR, macro repeat recall, month/history/family/frequency/support cohorts, all seeds, checkpoint paths and zero-feature/attention ablations.

| Full-pool selector | Unseen hits@25 | Recall@25 | Unseen hits@50 | Recall@50 |
|---|---:|---:|---:|---:|
| Frozen dot | 707.0 fixed | 38.07% | 1005.0 fixed | 54.12% |
| Static MLP | 708.7 ± 6.5 | 38.16% | 991.7 ± 7.1 | 53.40% |
| Dynamic MLP | 731.7 ± 4.5 | 39.40% | 1011.0 ± 7.0 | 54.44% |
| Dynamic + relationships MLP | 731.0 ± 5.3 | 39.36% | 1019.0 ± 7.5 | 54.87% |

The original clean 50-item slate covers 834 new targets; dot-selected 100 covers 1,265. Overall micro coverage rises from 88.67% to 93.95%. The dynamic selector at 100 covers 1,279.7 ± 1.5 new targets and 94.27% overall. Conditional new recall falls with the broader slate because many more hard candidates are available; full-denominator hits still increase. The 763-target historical mixed slate belongs to the legacy retriever and is not substituted for these S0 controls.

## What contributes, and what does not

The broader dot policy gains 26 mean discovery hits over the matched original-slate control. Its nested seed/customer 95% interval for new recall improvement is approximately **+0.46 to +2.38 percentage points**, with a positive overall recall interval. NDCG’s interval includes zero. This is the clearest conservative change; its mean NDCG remains below the preserved NDCG-selected alias, a tradeoff hidden by quoting only discovery hits.

At depth 100, the MLP selector adds only 1.7 mean hits over dot scoring. Its new-recall interval is **−0.83 to +1.00 pp**. Adding scalar features to the dot-selected ranker adds 13.7 mean hits, but its interval is **−0.32 to +1.75 pp**. Adding the MLP selector when those scalar features are already present adds 5.7 mean hits, with interval **−0.39 to +1.03 pp**. None establishes a reliable incremental advantage. The highest mean alone is insufficient to prefer the most complicated model.

On fixed original slates, scalar features improve 188.0 to 216.7 mean discovery hits and pass the exploratory common-control criteria. Hierarchical transitions (199.7), same-basket complements (198.0), and two experts (197.0) show smaller means with inconclusive intervals. Removing only the repeat flag does not help; removing all five SKU repeat signals increases discovery to 268 in a diagnostic but collapses Recall@10 to 0.7078 and NDCG to 0.5423. Those features encode valuable repeat evidence.

The global-demand feature group is heavily used: zeroing it in the selected seed-42 dot/100 augmented ranker reduces discovery from 230 to 173; removing relationship features gives 210, category/family timing 220, and calendar 230. The selected dynamic selector falls from 727 to 551 unseen hits@25 when its global group is zeroed. These are **post-training perturbations**, not causal feature-importance estimates or independently optimized replacements. They support the matched feature-family fits but do not justify claiming every new ratio is useful.

Candidate attention does not qualify. Direct and customer-dot heads fail the quality floor; a fresh residual gives only 191 versus 190 control hits. The combined model’s 239 mean hits are accompanied by NDCG 0.5875, failing in all three seeds. Its paired NDCG change versus the no-context combined control is negative throughout the 95% interval (about −0.0101 to −0.0035). It trades ranking quality for discovery rather than providing a usable improvement. Attention-disabled replays retain the trained pointwise feed-forward path, and permutation/padding/gradient tests confirm the block actually functions.

The cheaper fixed-slate pointwise model has 75,777 parameters and fast scoring, but only 0.5805 NDCG; UPS-CANR still adds useful detailed history reasoning. Bias-only versus values-only/both routing did not reveal a qualifying gain. The later review’s pretrained residual extension exactly preserved initial logits but finished at 187 hits, versus 187 for its matched fine-tuning control and 190 for the preserved clean source. Extra fine-tuning and that residual design did not earn retention.

Likely causes are sparse unseen labels, strong repeat base rates, redundant broad transitions, and a ranking task where additional set interaction supplies little new evidence. Wider slates recover targets lost before ranking. Explicit demand/family/relationship inputs help the scoring MLP use evidence that was absent or indirectly routed through attention. These are explanations consistent with the ablations, not proven causal mechanisms. The experiments do not establish that every possible candidate-attention or multi-interest method fails.

## Cost and cache lifecycle

The [resource table](experiments/dynamic_campaign/RESOURCES.md) reports every configuration’s fit time, parameter allocation, one-batch gradient-active count and CPU/MPS forward latency. Recorded fit times sum to 7,749 seconds (129.2 minutes), including per-epoch validation, excluding feature preparation, analysis and the interrupted setup probe. Dot/100 fits average 114.4 ± 1.0 seconds; dot/100 with scalar signals 119.0 ± 1.4 seconds. Frozen upstream training is prior work and is not silently counted as new fitting.

| Warm complete request, median / p95 | CPU | MPS |
|---|---:|---:|
| Dot / 100 / original ranker | 1.395 / 1.487 ms | 3.115 / 3.291 ms |
| Dot / 100 / scalar features | 1.845 / 1.944 ms | 3.483 / 3.723 ms |
| Dynamic MLP / 100 / original ranker | 8.508 / 8.791 ms | 10.634 / 10.943 ms |
| Dynamic MLP / 100 / scalar + candidate attention | 8.765 / 8.919 ms | 11.213 / 11.963 ms |

These use 50 warm local samples, resident versioned eligibility/seen masks and purchase inputs. Sorting, quota construction, candidate features and final scoring are timed. Storage, network, serialization, global-refresh construction and application event ingestion are excluded. Dot selection computes additional features only for the chosen 100; the MLP requires full-pool inputs. Subset tests establish exact values and complete-history semantics. These measurements are not deployment latency guarantees.

Dot/100 retains 866,313 allocated ranker parameters, with 664,329 receiving gradient tensors in a diagnostic batch. Scalar augmentation has 1,017,482 allocated but 671,497 gradient-active parameters: the checkpoint-compatible old scoring head remains allocated and unused. The MLP selector adds 79,361 parameters; combined candidate attention reaches 1,086,346 allocated and 740,361 gradient-active parameters. Gradient presence does not imply every weight entry is useful.

Customer-vector storage is 512 bytes; Stage 2 purchase-input tensor shapes require another 6,712 bytes (7,224 combined), with cheap clock fields refreshed from facts. Product vectors require 430,592 bytes. Serialized prior events plus snapshot facts in the 256-row sample have median 4,664 bytes and p95 52,331, excluding allocator overhead and overlapping information in tensors. A resident date eligibility mask and per-customer unseen mask each use 841 bytes. Global scalar snapshots use 23,548 bytes; the optional relationship arrays use 11,560,896 bytes. They are omitted for the simpler non-relationship serving path. The 2,237,403,384-byte float16 feature cache is offline experimental data, not a per-request or per-customer requirement.

The request adapter contains no customer tower call. Purchase updates and explicit model/catalogue rebuilds are versioned and tested; rebuilds create a new namespace and preserve old caches. A caller supplies current as-of facts/eligibility. The benchmark is an experimental resident-fact implementation, not a deployed ingestion/feature service.

## Validation and commands

Selected weights replay exactly on the key metrics. Full upstream train/validation caches were re-encoded and checked, eight real development request rows matched the quantized feature cache exactly, and all seven inventoried old checkpoint/cache hashes remain unchanged. [Verification](experiments/dynamic_campaign/verification.json) binds fit artifacts and input hashes. Seeded MPS retraining is not claimed bitwise deterministic: a synthetic repeated-index diagnostic diverged after updates despite identical initial weights and RNG state. Both same-seed control repetitions are reported separately rather than cherry-picked or pooled as independent seeds.

Final regression suite: **489 passed, 6 protected tests skipped, 70 subtests passed**. A temporary source checkout containing no private data/checkpoints passes **483 tests, 12 skipped, 70 subtests** using the installed environment. Linux CI for `3fbec83` was independently confirmed successful; the new commit's CI is recorded separately when pushed.

Customer-clustered intervals also resample fitted seeds. They are exploratory intervals on reused validation data, without a correction for this experiment family’s multiple comparisons. Only three seeds were used for promising configurations. Cold customers without any prior purchase are not represented by this next-basket task. The historical metadata limitation remains. No held-out purchases were materialized for model development or evaluated; the CSV partitioner reads dates to retain development rows.

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/run.py --phase all
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/inventory.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/coverage.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/pointwise_control.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/telemetry.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/summarize.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/verify.py
```

Prerequisites, exact input policies, screening supplements and resumption behavior are in the [campaign README](experiments/dynamic_campaign/README.md). The commands reproduce the pipeline with the preserved local upstream/cache inputs; completed matching fits reuse checked artifacts. Report-only artifacts refuse replacement where documented. Models, IDs and per-customer predictions stay in ignored local paths; committed evidence contains aggregate results and hashes.

Fresh retraining uses `RECSYS_CAMPAIGN_ROOT` and `RECSYS_CAMPAIGN_EVIDENCE` to select new namespaces; the README gives the full sequence. Existing campaign files and upstream inputs do not need to be moved or overwritten.

The recommendation is a conservative architecture candidate for approval, not a default-system change. Keep the augmented 100-item scorer available for an accuracy-oriented choice, with its incremental uncertainty stated. Do not replace dot retrieval or add candidate attention solely to obtain the largest validation discovery count.
