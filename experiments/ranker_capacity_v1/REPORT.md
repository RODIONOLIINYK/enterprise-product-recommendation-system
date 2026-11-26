# Ranker capacity and standalone CatBoost comparison

All fits use seed 42 and development data only. Training dates are before 2025-10-01; validation is 2025-10-01 through 2026-02-18. There are 26,253 training queries and 6,191 validation queries, with 10,276 complete targets including 1,857 new targets. No held-out evaluation or model promotion was performed.

## Neural capacity design

All four fresh arms share the original fused-profile token, 32 separate history events, 128D tokens, four attention heads, identical rolling 100-item slates, all 41 candidate statistics, frozen candidate vectors, loss, AdamW/cosine schedule, batch size 64, and ten epochs. Candidate self-attention and token enrichment remain disabled. Unchanged initial parameters are exact copies; changed-shape layers and the added attention layer have fresh initialization. This is an architecture/initialization comparison under a fixed budget, rather than optimal tuning of each architecture.

| Arm | Scoring head | Memory FFN | Memory self-attention layers |
| --- | --- | ---: | ---: |
| Original control | 425 → 256 → 128 → 64 → 1 | 256 | 1 |
| Wider scoring MLP | 425 → 512 → 256 → 128 → 1 | 256 | 1 |
| Wider Transformer FFN | Original | 512 | 1 |
| Deeper history Transformer | Original | 256 | 2 |

The no-op control has exact initialization parity with Arm A. Fresh-control and saved-control comparisons are both reported because MPS training is not bitwise deterministic. The saved control is an experimental rolling-data comparator, not a replacement for preserved production aliases.

## Standalone CatBoost design

**This CatBoost model directly scores every eligible product. It uses no Two-Tower model, neural vectors, dot similarities, MLP selector, retrieved shortlist, Transformer, or subsequent neural ranking.** Inputs are three raw product categoricals (SKU/category/business line), 41 strictly earlier rolling purchase features and eight customer profile statistics (52 columns total). Reusing dated query/statistical caches does not supply their historical neural representations to the classifier.

Eligibility is recomputed from raw development purchases and catalogue metadata and must agree with cached masks. All eligible SKUs are scored at validation time; validation targets are never inserted into a selected candidate list. The modelled 841-SKU universe is verified to contain every development-eligible SKU.

Training samples 2,670,525 pairs representing 15,999,865 eligible pairs: all eligible positives plus up to 100 uniformly sampled negatives per query. Each sampled negative is weighted by eligible-negative-count/sample-count, yielding an unbiased estimate of the full-pool binary Logloss for a fixed scorer. Sampling is a training cost optimization, not a Stage 1 retrieval model. Validation scores all 4,700,328 eligible pairs.

The classifier retains the historical depth-8, maximum-500-tree recipe, learning rate 0.05, Logloss early stopping (50 rounds), and four CPU threads. Ranking selections inspect 50-tree prefixes, at most ten checkpoints. No historical trained CatBoost weights are loaded.

## Ndcg-selected results

| Model | Epoch/trees | Recall@5 | Recall@10 | NDCG@10 | New hits@10 / 1,857 | New recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved A / original 32 events | 8 | 0.66012 | 0.77104 | 0.60223 | 255 | 0.13732 |
| Fresh original control | 8 | 0.65893 | 0.76941 | 0.60214 | 253 | 0.13624 |
| Wider scoring MLP | 9 | 0.65619 | 0.76632 | 0.60052 | 239 | 0.12870 |
| Wider Transformer FFN | 8 | 0.65901 | 0.76682 | 0.59975 | 245 | 0.13193 |
| Two history attention layers | 10 | 0.65861 | 0.76899 | 0.59955 | 245 | 0.13193 |
| Standalone CatBoost / full eligible pool | 318 | 0.64886 | 0.75815 | 0.59373 | 202 | 0.10878 |

Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. Delta is treatment minus comparator, conditional on selected fitted weights. They do not account for training-seed uncertainty or repeated validation selection.

| Treatment | Comparator | Metric | Delta | 95% CI |
| --- | --- | --- | ---: | --- |
| Wider scoring MLP | Saved A | recall@10_delta | -0.00472 | [-0.00791, -0.00163] |
| Wider scoring MLP | Saved A | ndcg@10_delta | -0.00171 | [-0.00415, +0.00055] |
| Wider scoring MLP | Saved A | new_recall@10_delta | -0.00862 | [-0.01749, +0.00104] |
| Wider scoring MLP | Fresh control | recall@10_delta | -0.00309 | [-0.00638, +0.00021] |
| Wider scoring MLP | Fresh control | ndcg@10_delta | -0.00162 | [-0.00411, +0.00079] |
| Wider scoring MLP | Fresh control | new_recall@10_delta | -0.00754 | [-0.01711, +0.00216] |
| Wider Transformer FFN | Saved A | recall@10_delta | -0.00422 | [-0.00723, -0.00123] |
| Wider Transformer FFN | Saved A | ndcg@10_delta | -0.00248 | [-0.00479, -0.00015] |
| Wider Transformer FFN | Saved A | new_recall@10_delta | -0.00539 | [-0.01460, +0.00327] |
| Wider Transformer FFN | Fresh control | recall@10_delta | -0.00259 | [-0.00562, +0.00039] |
| Wider Transformer FFN | Fresh control | ndcg@10_delta | -0.00238 | [-0.00475, -0.00009] |
| Wider Transformer FFN | Fresh control | new_recall@10_delta | -0.00431 | [-0.01255, +0.00325] |
| Two history attention layers | Saved A | recall@10_delta | -0.00205 | [-0.00510, +0.00104] |
| Two history attention layers | Saved A | ndcg@10_delta | -0.00268 | [-0.00493, -0.00035] |
| Two history attention layers | Saved A | new_recall@10_delta | -0.00539 | [-0.01375, +0.00267] |
| Two history attention layers | Fresh control | recall@10_delta | -0.00042 | [-0.00360, +0.00284] |
| Two history attention layers | Fresh control | ndcg@10_delta | -0.00259 | [-0.00508, -0.00016] |
| Two history attention layers | Fresh control | new_recall@10_delta | -0.00431 | [-0.01270, +0.00383] |
| Standalone CatBoost | Saved A | recall@5_delta | -0.01126 | [-0.01738, -0.00556] |
| Standalone CatBoost | Saved A | recall@10_delta | -0.01290 | [-0.01775, -0.00833] |
| Standalone CatBoost | Saved A | recall@25_delta | -0.01052 | [-0.01556, -0.00614] |
| Standalone CatBoost | Saved A | recall@50_delta | -0.01342 | [-0.01845, -0.00865] |
| Standalone CatBoost | Saved A | recall@100_delta | -0.00783 | [-0.01285, -0.00293] |
| Standalone CatBoost | Saved A | ndcg@10_delta | -0.00850 | [-0.01188, -0.00509] |
| Standalone CatBoost | Saved A | new_recall@10_delta | -0.02854 | [-0.04291, -0.01475] |

## Discovery-selected results

| Model | Epoch/trees | Recall@5 | Recall@10 | NDCG@10 | New hits@10 / 1,857 | New recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved A / original 32 events | 8 | 0.66012 | 0.77104 | 0.60223 | 255 | 0.13732 |
| Fresh original control | 8 | 0.65893 | 0.76941 | 0.60214 | 253 | 0.13624 |
| Wider scoring MLP | 5 | 0.65641 | 0.76966 | 0.59881 | 250 | 0.13463 |
| Wider Transformer FFN | 9 | 0.65620 | 0.76724 | 0.59935 | 249 | 0.13409 |
| Two history attention layers | 4 | 0.65700 | 0.76712 | 0.59749 | 247 | 0.13301 |
| Standalone CatBoost / full eligible pool | 318 | 0.64886 | 0.75815 | 0.59373 | 202 | 0.10878 |

Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. Delta is treatment minus comparator, conditional on selected fitted weights. They do not account for training-seed uncertainty or repeated validation selection.

| Treatment | Comparator | Metric | Delta | 95% CI |
| --- | --- | --- | ---: | --- |
| Wider scoring MLP | Saved A | recall@10_delta | -0.00138 | [-0.00458, +0.00192] |
| Wider scoring MLP | Saved A | ndcg@10_delta | -0.00342 | [-0.00581, -0.00104] |
| Wider scoring MLP | Saved A | new_recall@10_delta | -0.00269 | [-0.01114, +0.00565] |
| Wider scoring MLP | Fresh control | recall@10_delta | +0.00025 | [-0.00296, +0.00363] |
| Wider scoring MLP | Fresh control | ndcg@10_delta | -0.00332 | [-0.00563, -0.00102] |
| Wider scoring MLP | Fresh control | new_recall@10_delta | -0.00162 | [-0.01081, +0.00721] |
| Wider Transformer FFN | Saved A | recall@10_delta | -0.00381 | [-0.00709, -0.00042] |
| Wider Transformer FFN | Saved A | ndcg@10_delta | -0.00288 | [-0.00536, -0.00045] |
| Wider Transformer FFN | Saved A | new_recall@10_delta | -0.00323 | [-0.01298, +0.00656] |
| Wider Transformer FFN | Fresh control | recall@10_delta | -0.00217 | [-0.00529, +0.00082] |
| Wider Transformer FFN | Fresh control | ndcg@10_delta | -0.00278 | [-0.00530, -0.00048] |
| Wider Transformer FFN | Fresh control | new_recall@10_delta | -0.00215 | [-0.01072, +0.00669] |
| Two history attention layers | Saved A | recall@10_delta | -0.00392 | [-0.00742, -0.00051] |
| Two history attention layers | Saved A | ndcg@10_delta | -0.00474 | [-0.00735, -0.00194] |
| Two history attention layers | Saved A | new_recall@10_delta | -0.00431 | [-0.01378, +0.00523] |
| Two history attention layers | Fresh control | recall@10_delta | -0.00229 | [-0.00580, +0.00115] |
| Two history attention layers | Fresh control | ndcg@10_delta | -0.00465 | [-0.00748, -0.00200] |
| Two history attention layers | Fresh control | new_recall@10_delta | -0.00323 | [-0.01258, +0.00577] |
| Standalone CatBoost | Saved A | recall@5_delta | -0.01126 | [-0.01738, -0.00556] |
| Standalone CatBoost | Saved A | recall@10_delta | -0.01290 | [-0.01775, -0.00833] |
| Standalone CatBoost | Saved A | recall@25_delta | -0.01052 | [-0.01556, -0.00614] |
| Standalone CatBoost | Saved A | recall@50_delta | -0.01342 | [-0.01845, -0.00865] |
| Standalone CatBoost | Saved A | recall@100_delta | -0.00783 | [-0.01285, -0.00293] |
| Standalone CatBoost | Saved A | ndcg@10_delta | -0.00850 | [-0.01188, -0.00509] |
| Standalone CatBoost | Saved A | new_recall@10_delta | -0.02854 | [-0.04291, -0.01475] |

## Learning curves and training cost

Training losses are minibatch means in training mode (with dropout), while validation loss uses evaluation mode. Their absolute levels are not a clean generalization-gap estimate. Curves and selection epochs are provided without attributing every validation change to overfitting.

| Arm | Trainable parameters | Training seconds | Epoch 1 training loss | Epoch 10 training loss | Epoch 10 validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh original control | 671,497 | 144.9 | 3.17875 | 2.85642 | 2.70287 |
| Wider scoring MLP | 904,585 | 295.4 | 3.13755 | 2.82055 | 2.70246 |
| Wider Transformer FFN | 737,289 | 340.5 | 3.18180 | 2.85223 | 2.69947 |
| Two history attention layers | 803,977 | 263.3 | 3.18667 | 2.92116 | 2.69098 |

## Model inference latency

Warm prepared-input forward medians, 100 candidates per neural request. Measurements below were taken after both training processes exited. Standalone trees score their entire eligible pool per request, so the two latency scopes differ.

| Model | CPU batch 1 (ms) | CPU batch 64 (ms) | MPS batch 1 (ms) | MPS batch 64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Saved A | 0.712 | 10.687 | 2.875 | 6.707 |
| Fresh original control | 0.629 | 10.675 | 2.959 | 6.624 |
| Wider scoring MLP | 0.806 | 15.654 | 2.898 | 8.066 |
| Wider Transformer FFN | 0.667 | 12.402 | 2.867 | 6.798 |
| Two history attention layers | 0.775 | 12.470 | 3.029 | 7.225 |

Standalone NDCG-selected CatBoost: 0.977 ms CPU median for 752 eligible SKUs; statistical feature preparation and Python DataFrame/Pool construction are excluded.

## Limits and verification

- Capacity-arm comparisons hold candidate sets fixed. Standalone CatBoost changes the candidate policy, input features and objective; its comparison is of complete pipelines.
- All recall denominators include targets absent from the candidate pool. New hits always divide by all 1,857 new targets. Current full-pool availability still uses snapshot metadata and last-modified blocking assumptions.
- Single seed, limited training budget and repeated validation use do not establish a universal architecture ranking. No reference checkpoint was replaced.
- Selected neural weights are reloaded and replayed; standalone selected trees are reloaded against the complete eligible validation pool.
- All 31 preserved reference/cache evidence hashes remain unchanged.
- Final uncontended latency measurements exclude ingestion, feature preparation, disk loading and network time. Initial in-run measurements may overlap tree fitting; use benchmark_uncontended.json for the final comparison.

## Artifacts and reproduction

- [Capacity summary](summary.json)
- [Standalone CatBoost summary](../catboost_standalone_v1/summary.json)
- [Independent verification](verification.json)
- [Final latency benchmark](benchmark_uncontended.json)

```bash
./venv/bin/python -m recommender.experiments.ranker_capacity.train --execute-training
./venv/bin/python -m recommender.experiments.catboost_comparison.standalone --execute-training
./venv/bin/python -m recommender.experiments.ranker_capacity.report --verify
PYTHONPATH=. ./venv/bin/pytest tests/ -q
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
```

## Test results

Full safe suite: 579 passed, 6 skipped. Synthetic-only: 569 passed, 16 skipped. Both include 70 passing subtests. Undefined-symbol/unused-import checks and git diff --check pass. Chart configuration checks cover all six selector states.
