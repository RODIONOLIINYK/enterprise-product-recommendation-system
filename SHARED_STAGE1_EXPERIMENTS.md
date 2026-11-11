# Shared Stage 1 representations in Stage 2 — 2026-11-11

Implementation starts from `main` at `99bc1c5c17138c45ed7fe2ec128cf14c46dbe400`. The original Stage 1/Stage 2 checkpoints, original slate caches and default ranker are preserved. Experimental inputs, checkpoints and evaluation use separate paths.

## Representation and bias contract

| Dimensions (inclusive) | Count | Contents |
|---|---:|---|
| 0–127 | 128 | Unmodified, frozen Stage 1 tower output |
| 128–135 | 8 | Customer statistics in `LEGACY_PROFILE_FEATURES` order |
| 136–143 | 8 | History statistics from source columns 0–7 |
| 144–156 | 13 | Candidate statistics in `PAIRWISE_FEATURE_NAMES` order |
| 157–172 | 16 | Learned history positions |
| 173–175 | 3 | Customer/history/candidate one-hot token type |
| **Total** | **176** | |

The ordered history input is `quantity_norm`, `days_since_purchase_norm`, `inter_purchase_gap_norm`, `purchase_month_sin`, `purchase_month_cos`, `prior_sku_purchase_count_norm`, `cumulative_sku_quantity_share`, `is_repeat_purchase`. There is **no `real_event_indicator` in these tokens**. Source sequences remain nine-dimensional for legacy compatibility. The initial comparisons exclude `is_repeat_purchase` from explicit statistics, matching the reference; its reserved input coordinate is zero. With that exclusion disabled, all eight source values are recoverable, apart from the documented scaling.

Each numerical group has an independent positive learned scale, initialized to 0.1 (with independent configurable overrides). Position and type scales start at 1. The first 128 coordinates are never scaled, projected or normalized during input assembly. Padded input tokens are entirely zero, and their keys/values are masked. Unknown real purchases have a zero pretrained-vector region plus their real-event statistics/type/position; padding is distinguished by `attention_mask`.

`legacy17` preserves the four-head `(4, 14, 17)` statistical-bias interface and derives its history validity column from `attention_mask`. `reduced16` deletes that column, giving `(4, 14, 16)`. The original `UPSCANRRanker` continues to use its original bias and source schema. R3 and R4 have identical non-bias initialization and identical initial scores; the bias change saves exactly 56 coefficients. The sequence age input remains its original normalized value; the bias still recomputes `age_days / (age_days + 90)`.

The new encoder uses one 176D layer, four heads, GELU and a 352D feed-forward layer. Cross-attention uses the same width. A trainable context readout, initially `[I_128 | 0]`, feeds the unchanged conceptual 397D scoring MLP. Subsequent Transformer transformations can change the pretrained coordinates; the lossless guarantee applies to input assembly.

192D zero-channel and zero-initialized adapter variants, and a 224D categorical variant, are implemented and tested. The two 192D variants have identical common initial weights and initial scores. Their training is conditional on evidence from the simpler model; availability in code is not evidence of ranking improvement.

## Provenance and temporal limits

All comparisons use Stage 1 SHA-256 `2dcfbc7eab6795ded6f25a6cb0ffe1f9e606f13c9885d4ee020b724d0a2b5ab7`, 790 trained SKU IDs and 13,729 expanded catalogue IDs. A canonical-SKU lookup maps history into the expanded table; it never treats old one-based training IDs as expanded zero-based indices. This correctly represents **417 validation history tokens** whose old vocabulary ID was unknown. All real history tokens in the generated development caches mapped to the expanded catalogue.

For every transition, the exact Stage 1 customer forward used for candidate retrieval is cached alongside canonical history SKUs, expanded indices, prediction dates, row keys and source schemas. The customer history/profile inputs are strictly prior to that transition. **The frozen product vectors still inherit the fixed train-cutoff demand snapshot and the retriever's training-period look-ahead. These experiments are conditional comparisons under that retriever, not evidence from a temporally clean Stage 1 model.** No Stage 1 retraining was mixed into the Stage 2 experiments.

A separate frozen-weight sensitivity probe changes only product demand from the train-cutoff snapshot to a strictly prior-to-date snapshot. Across its three training dates, mean product cosine distance is approximately 0.0013–0.0034, maximum product distance reaches 0.1095, and customer vectors also change. This confirms that the temporal issue reaches the pretrained representations. It does not establish a clean retrained reference or measure ranking quality. Details and preprocessing timings are in [temporal_sensitivity.json](experiments/shared_stage1/temporal_sensitivity.json). A separately versioned rolling-demand Stage 1 training/control experiment remains necessary before a production promotion.

Development loading consults the partition date before materializing purchase fields in pandas and excludes all dates from 2026-02-19 onward. Excluded records never contribute targets, labels, statistics or model inputs. All unit tests use synthetic data; the ordinary suite retains its six protected real-data skips.

The new caches contain **26,263 training** and **6,191 validation** transitions. Candidate IDs/indices, labels, pairwise features, old sequence IDs/features and customer statistics are bitwise equal to the original train/validation slate caches. Every run uses the same 50 candidates. New-target coverage is **763 / 1,857 = 41.09%**; targets missing from the slate remain in the recall denominator. Unknown/padding behavior and schema/date/catalogue mismatches are explicitly tested and rejected where incompatible.

## Training and selection

Each training run has 10 epochs, batch size 64, AdamW (`lr=0.001`, `weight_decay=0.0001`), cosine scheduling to `1e-5`, gradient clipping at 1, BCE positive weight 10 plus listwise loss weight 1, and no special new-positive upweighting. Batch-order generators are seeded independently of parameter construction so the compared models receive the same shuffled batches for a given seed. The fixed overall Recall@10 selection guardrail is copied from the local discovery checkpoint: `0.7558833119851873`.

Selection uses strictly improving new-target Recall@10 under that guardrail. All metrics in a row come from the **same selected checkpoint**. Peak NDCG from another epoch is recorded separately, never combined with discovery-selected metrics. An unconstrained selection is reported explicitly if no checkpoint meets the guardrail.

R0 is a fresh training control, separate from replaying saved reference weights. R0W uses the old 128D raw token values padded to 176D before a single encoder, with common attention/readout/scoring initialization matched to R1. Its old input branches, positions and type embeddings remain part of that control; it is not a claim that every aspect except width is identical to R0. An earlier unmatched-initialization R0W screening run was archived and excluded from the final comparisons.

R3P0 zeros all customer-token input information and customer bias statistics while retaining an unmasked null token for finite cold starts. R3H0 masks separate history tokens while retaining the Stage 1 customer vector, which already encodes history. These are trained controls. Separately recorded inference perturbations hold weights fixed and must not be interpreted as retrained causal ablations.

## Results

| Run / seed | Selected epoch | Recall@10 | NDCG@10 | New hits / 1,857 | New Recall@10 | Conditional new Recall@10 | Val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preserved NDCG alias | 7 | 76.0417% | 0.593815 | 201 | 10.82% | 26.34% | 2.5266 |
| Preserved discovery reference | 4 | 76.1505% | 0.591893 | 209 | 11.25% | 27.39% | 2.5520 |
| R0 / 42 | 6 | 75.8045% | 0.591497 | 191 | 10.29% | 25.03% | 2.5364 |
| R0 / 17 | 7 | 76.0262% | 0.593256 | 201 | 10.82% | 26.34% | 2.5268 |
| R0 / 2026 | 8 | 76.0727% | 0.595224 | 198 | 10.66% | 25.95% | 2.5303 |
| R0W / 42 | 6 | 75.8427% | 0.590884 | 196 | 10.55% | 25.69% | 2.5388 |
| R1 / 42 | 9 | 76.0352% | 0.593411 | 197 | 10.61% | 25.82% | 2.5129 |
| R2 / 42 | 5 | 75.9212% | 0.591031 | 197 | 10.61% | 25.82% | 2.5420 |
| R3 / 42 | 7 | 75.9465% | 0.593058 | 198 | 10.66% | 25.95% | 2.5205 |
| R3 / 17 | 4 | 75.9737% | 0.589046 | 196 | 10.55% | 25.69% | 2.5485 |
| R3 / 2026 | 5 | 75.9674% | 0.591306 | 195 | 10.50% | 25.56% | 2.5304 |
| R4 / 42 | 10 | 75.9990% | 0.592362 | 200 | 10.77% | 26.21% | 2.5151 |
| R4 / 17 | 4 | 75.8970% | 0.589412 | 193 | 10.39% | 25.29% | 2.5501 |
| R4 / 2026 | 2 | 75.9520% | 0.586054 | 194 | 10.45% | 25.43% | 2.5868 |
| R3P0 / 42 | 7 | 76.1908% | 0.593438 | 206 | 11.09% | 27.00% | 2.5270 |
| R3P0 / 17 | 7 | 75.9018% | 0.592717 | 193 | 10.39% | 25.29% | 2.5242 |
| R3P0 / 2026 | 10 | 76.2351% | 0.594397 | 201 | 10.82% | 26.34% | 2.5173 |
| R3H0 / 42 | 7 | 75.8722% | 0.592557 | 192 | 10.34% | 25.16% | 2.5336 |

All rows passed the fixed Recall@10 guardrail. Seed-42 R1 uses only the shared vectors/position/type in memory and queries; R2 adds numerical memory, R3 adds numerical queries, and R4 removes the indicator from the bias. R3P0 removes customer-token information; R3H0 removes separate history tokens.

| Three-seed run | New hits, mean ± sample SD | Recall@10, mean ± SD (pp) | NDCG@10, mean ± SD |
|---|---:|---:|---:|
| R0 | 196.67 ± 5.13 | 75.9678% ± 0.1433 | 0.593325 ± 0.001865 |
| R3 | 196.33 ± 1.53 | 75.9625% ± 0.0142 | 0.591137 ± 0.002011 |
| R4 | 195.67 ± 3.79 | 75.9493% ± 0.0510 | 0.589276 ± 0.003156 |
| R3P0 | 200.00 ± 6.56 | 76.1092% ± 0.1810 | 0.593517 ± 0.000843 |

| Run | Parameters | 10-epoch runtime range (s) | Sampled MPS driver peak range (GiB) |
|---|---:|---:|---:|
| R0 | 866,313 | 95.8–112.2 | 1.10–1.10 |
| R0W | 862,809 | 125.4–125.4 | 1.12–1.12 |
| R1 | 542,478 | 107.1–107.1 | 1.13–1.13 |
| R2 | 542,478 | 118.3–118.3 | 1.13–1.13 |
| R3 | 542,478 | 114.1–126.0 | 1.12–1.13 |
| R4 | 542,422 | 110.0–120.8 | 1.12–1.13 |
| R3P0 | 542,478 | 108.8–122.0 | 1.10–1.10 |
| R3H0 | 542,478 | 109.8–109.8 | 1.10–1.10 |

Attention means/quantiles/histograms, entropy, performance by observed history length (capped at 32), cumulative RSS, per-epoch curves and same-checkpoint metadata are in the aggregate JSON files.

## Uncertainty and decision

**Recommendation: retain the existing ranker checkpoints and the original statistical bias. Do not promote the shared Stage 1 architecture.**

The full 176D model averages 196.33 hits versus fresh R0's 196.67; removing the indicator averages 195.67. The best no-customer-token variant averages 200.00 and reaches 206 in its best individual fit, still below the preserved discovery checkpoint's 209. Its mean overall Recall@10 is also slightly below that preserved checkpoint, although its NDCG is higher. No tested shared-vector variant improves the requested discovery objective while maintaining the reference's overall metrics.

The indicator ablation changes predictions and selected checkpoints; it is not a harmless refactor. Its new-hit differences versus the legacy bias are +2, −3 and −1 across seeds 42, 17 and 2026. The paired 95% intervals for new-target Recall@10 are:

- Seed 42: +0.108 pp [-0.646, +0.812]
- Seed 17: -0.162 pp [-0.600, +0.277]
- Seed 2026: -0.054 pp [-0.918, +0.844]

All include zero. R1 versus the matched width control changes only one selected discovery hit (+0.054 pp [-1.037, +1.152]); this provides no convincing evidence for sharing representations beyond width.

Removing all customer-token information gives +8, −3 and +6 hits versus R3 across the three seeds. All three paired new-recall intervals include zero. This does not establish useful incremental ranking information from the pretrained customer token, nor does it justify promoting its removal. Removing separate history tokens gives 192 versus R3's 198 hits at seed 42; it was a single-seed trained control. Both findings are conditional on the fixed candidate slates and remaining pairwise features.

R5–R9 training (192D expansion, 224D categorical information, new relationship edges and simpler scoring heads) was deferred because the simpler shared model showed no reproducible gain. The source supports the proposed 192D/224D layouts and tests their mechanics, but there are no ranking claims for those untrained modes. A clean rolling-demand Stage 1 control, and investigation of candidate coverage/objective signals, take priority over further width increases.


Paired intervals use 2,000 customer-cluster bootstrap samples on identical validation baskets. They describe fixed selected models, not uncertainty from model selection or training seeds. Three-seed means/sample standard deviations are reported separately. Small differences in new-product hits are not sufficient evidence for replacing the baseline.

## Parameter and feature utilization

R0 allocates 866,313 parameters; 201,984 attached ProductTower parameters receive no gradients with cached candidate vectors. It also retains the reference's zero-input habit/text weights and numerical tabular columns. The new shared model allocates 542,478 parameters (542,422 for the reduced bias) and attaches no unused Stage 1 tower. Its pretrained table is a frozen buffer, not a trainable parameter. In the representative shared-model probes every parameter tensor participates in the graph, although excluded statistics, disabled routes and unused positions can have zero-gradient entries. A graph-connected zero gradient is not proof of a globally dead parameter.

Both self-attention and cross-attention Q/K/V, bias coefficients, readout, scoring MLP, optional adapters and active scale groups receive finite gradients. For selected R3/seed 42 the learned group scales are approximately customer 0.122, history 0.152, candidate 0.112, position 1.031 and type 1.098. The auxiliary readout columns have nonzero norm (1.255), showing that the wider context can enter the scoring head.

Fixed-weight path perturbations in R3/seed 42 reduce 198 hits to 190 without customer-token information, 160 without history tokens and 182 without explicit numerical memory; disabling query statistics leaves 198. Retrained controls differ from these perturbations, so these numbers show model reliance rather than a guaranteed benefit from adding each pathway. Full named-parameter gradient norms/counts are retained in each local `result.json`; aggregate utilization and history-length/attention distributions are in `summary.json`.


Measured runtimes are 10-epoch wall times on MPS. MPS memory is sampled driver allocation, including caching; process RSS is cumulative across runs in that process. Verification jobs shared the machine. These are recorded telemetry, not isolated throughput or memory-efficiency claims.

## Reproduce and inspect

```bash
# Initial screening, including a matched width control; output paths are separate.
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R0 R0W R1 R2 R3 R4 --device mps

# Repeat the main model and bias comparison at seeds 17 and 2026.
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R0 R3 R4 --seed 17 --device mps
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R0 R3 R4 --seed 2026 --device mps

# Trained token controls; repeat R3P0 at seeds 17 and 2026 as recorded in the report.
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R3P0 R3H0 --device mps
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R3P0 --seed 17 --device mps
PYTHONPATH=. ./venv/bin/python -m recommender.train_shared_ranker --runs R3P0 --seed 2026 --device mps

# Rebuild aggregate evidence by replaying the selected experimental checkpoints.
PYTHONPATH=. ./venv/bin/python -m recommender.summarize_shared_ranker --device mps
PYTHONPATH=. ./venv/bin/python -m recommender.shared_temporal_probe
PYTHONPATH=. ./venv/bin/pytest tests/
```

Caches: `data/interim/shared_stage1_v1/`. Checkpoints, epoch histories, per-basket predictions and full parameter-gradient reports: `checkpoints/shared-stage1-experiments/`. Aggregate public evidence: [summary.json](experiments/shared_stage1/summary.json), [curves.json](experiments/shared_stage1/curves.json), [paired_comparisons.json](experiments/shared_stage1/paired_comparisons.json), [reference_integrity.json](experiments/shared_stage1/reference_integrity.json). Raw purchase records and customer-level prediction arrays are not included in the aggregate evidence files.

Changed implementation files: `recommender/shared_ranker.py` (token layout, models, bias and strict checkpoint loader), `recommender/shared_data.py` (development gate and validated caches), `recommender/train_ranker.py` (opt-in capture/Dataset/collator/evaluator integration), `recommender/train_shared_ranker.py` (training, replay, diagnostics and bootstrap), `recommender/shared_temporal_probe.py`, `recommender/summarize_shared_ranker.py`, and `tests/test_shared_ranker.py`.

## Verification

- Safe full suite: **392 passed, 6 skipped, 70 subtests passed**; skips are the protected unsplit-real-data tests.
- Tests cover exact pretrained coordinates, all eight raw history statistics with repeat exclusion configurable, negative month values, distinct token types, strict padding, unknown/expanded SKUs, date isolation, same-customer transition-specific vectors, legacy/reduced bias behavior, all history lengths 0–32, Q/K/V/adapters/readout/scoring gradients, common-initialization controls, checkpoint roundtrips and incompatible cache/checkpoint rejection.
- Every selected checkpoint was replayed using the current reconstruction path; Top-10 recall, NDCG and new-hit counts exactly match its recorded results.
- Candidate/input equivalence to original train/validation slate caches was checked exactly. All seven original checkpoint/cache SHA-256 hashes remain unchanged.
- Compile, static checks on the new files, production undefined/unused-import checks and `git diff --check` are recorded in `experiments/shared_stage1/validation.json`.
- Sixteen completed controlled model/seed fits used 10 epochs each; one earlier width-control screening fit is archived and excluded. No held-out targets were inspected or evaluated, no Stage 1 model was retrained, and no default ranker/checkpoint alias was changed.
