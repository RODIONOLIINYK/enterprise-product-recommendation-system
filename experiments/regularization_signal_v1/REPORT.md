# Fixed-input regularization and negative-resampling experiments

Training is before 2025-10-01, validation is 2025-10-01 through 2026-02-18. 26,253 training / 6,191 validation queries; 10,276 full validation targets, including 1,857 new targets. Seed 42; ten epochs per fit. No held-out access, promotion, commit or push.

Observed result: the 32,953-parameter full-pool MLP improved from NDCG@10=0.58915 with regularization to 0.59143 with zero dropout/weight decay, then 0.59372 with rotating negatives. The frozen CatBoost has 0.59373. This is a useful small-model result; uncertainty does not establish superiority over CatBoost. None of the Stage 2 NDCG-selected treatments beats the saved rolling reference 0.60223. The zero-regularization discovery checkpoint reaches 258 new hits versus 255 for the reference, with lower overall Recall@10 and NDCG. References remain unchanged.

## Transition features already reach both neural scorers

The active Stage 1 MLP keeps 39/41 context features, omitting only month sine/cosine, plus six customer statistics. The pointwise Stage 2 scorer receives all 41 context features directly. The following are physical input columns, not merely indirect information inside embeddings:

| Feature | Active Stage 1 MLP column | Pointwise Stage 2 MLP column | CatBoost input |
| --- | ---: | ---: | --- |
| sku_last_lift | 287 | 417 | direct scalar |
| sku_recent3_lift | 288 | 418 | direct scalar |
| category_last_lift | 289 | 419 | direct scalar |
| family_last_lift | 290 | 420 | direct scalar |
| line_last_lift | 291 | 421 | direct scalar |
| sku_support_confidence | 292 | 422 | direct scalar |
| hierarchical_backoff | 293 | 423 | direct scalar |
| same_basket_lift | 294 | 424 | direct scalar |

All columns are zero-indexed. Counts use strictly prior observed purchases, never prediction-day or held-out purchases. Same-basket statistics describe earlier observed baskets. The direct52 treatment adds all profile8 scalars and explicit candidate SKU/category/business-line embeddings to the Stage 2 scoring head; the existing transition inputs remain present.

## CatBoost identities: avoid conflating three experiments

1. **Historical 17-field classifier:** three categorical fields plus fourteen numerical features. The recipe is preserved in `experiments/catboost_current_v1/historical_validation.json`. Its historical customer-disjoint, approximately 25-candidate, target-inserted result is not comparable to this temporal full-pool benchmark. The later causal reconstruction of that recipe on the current 100 retrieved candidates reached R@10=0.75793, NDCG@10=0.59457, 143 new hits. It depended on that candidate policy.
2. **Current 53-field retrieved-slate classifier:** three categoricals +41 rolling scalars +8 profile scalars +one tower-dot score. This is a distinct earlier experiment, not a standalone raw-data model.
3. **Standalone 52-field classifier:** the same three categoricals, 41 rolling scalars and profile8, without the tower-dot score or a neural shortlist. It scores every eligible SKU. This was a new baseline added to answer the standalone question, not a replacement or reproduction of the original 17-field recipe.

This campaign freezes the third model, its 52-field schema, numeric values, source identities, saved model files and results. CatBoost is neither retrained nor changed. See [feature_lock.json](feature_lock.json).

## Full-pool MLP: use exactly those information inputs

The standalone MLP has 49 numeric inputs plus the same three categorical fields, encoded with learned SKU16/category8/line8 embeddings; head 81→128→64→1, LayerNorm and GELU, 32,953 parameters. No company/product text vectors, pretrained towers, Stage 1 scores, history Transformer or retrieved shortlist are inputs. Categorical encoding differs naturally between model families; raw fields and numeric values do not. Direct parity against CatBoost frame construction was checked on 4,096 sampled rows.

Fixed-pair arms use exactly the seed 42 CatBoost training sample: 2,670,525 pairs, including 45,225 eligible positives, representing 15,999,865 eligible pairs through importance weights. All positives have weight 1; uniformly sampled negatives have weight eligible_negative_count/sample_count. The objective is the same weighted binary Logloss; no pos_weight10, listwise term or repeat/new weighting. A fixed global importance normalizer makes minibatch gradients estimate the full sampled objective. All models rank all 4,700,328 eligible validation pairs without inserting targets.

MLP arms share initial weights and independent deterministic batch orders, AdamW lr 0.001, cosine decay to 1e-5, batch 4096, CPU 2 threads, ten epochs. Removing dropout and weight decay together is a combined regularization change. The third arm changes only the sampled negatives relative to the second. Per-epoch draw digests, positives and importance mass are recorded.

| NDCG-selected model | Epoch/trees | Recall@10 | NDCG@10 | New hits@10 | Qualified |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen standalone CatBoost 52 | 318 | 0.75815 | 0.59373 | 202 | True |
| MLP / fixed pairs / dropout .2,.1 | 8 | 0.76130 | 0.58915 | 221 | False |
| MLP / fixed pairs / zero dropout and weight decay | 8 | 0.76293 | 0.59143 | 223 | False |
| MLP / rotating negatives / zero dropout and weight decay | 9 | 0.76263 | 0.59372 | 224 | True |

Qualification uses the preserved discovery guardrails: Recall@10≥0.75588 and NDCG@10≥0.59189. The first two MLP arms have diagnostic discovery selections but do not pass the NDCG guardrail. Full discovery/logloss selections are retained in [tabular_summary.json](tabular_summary.json); do not mix their hits with NDCG-selected metrics.

Paired customer-clustered bootstrap 95% intervals, conditional on selected weights:

| Comparison | NDCG@10 delta [95% CI] | Recall@10 delta [95% CI] |
| --- | --- | --- |
| Regularized fixed → unregularized fixed | +0.00229 [+0.00006, +0.00460] | +0.00162 [-0.00105, +0.00420] |
| Unregularized fixed → rotating negatives | +0.00228 [-0.00031, +0.00461] | -0.00030 [-0.00263, +0.00198] |
| Frozen CatBoost → rotating-negative MLP | -0.00001 [-0.00348, +0.00336] | +0.00448 [-0.00004, +0.00886] |

The rotating-negative MLP numerically matches CatBoost NDCG and has more top-10 new hits (224 versus 202), but the paired CatBoost comparison does not establish superiority. It is a promising small standalone alternative, not an end-to-end neural pipeline promotion. The existing rolling Stage 1 frozen selector scores R@10=0.75845/NDCG@10=0.58822/206 new hits under the saved replay; this new MLP changes inputs, objective and training as well as representation, so that comparison is practical rather than an architecture-only ablation. New MLP scores were not used to rebuild Stage 2 candidate slates in this campaign.

## Stage 2 regularization: keep architecture and statistics fixed

Seven fresh fits share the same initial weights, original 32-event memory, head 256/128/64, rolling quota100_50 candidates, RankerLoss, lr 0.001 cosine schedule, batch 64, ten epochs and MPS. Candidate self-attention, token enrichment and Stage 1 MLP score inputs remain disabled. Head half-dropout, profile half-dropout, attention half-dropout and zero weight decay are separate single-axis changes. The final two runs deliberately combine changes. Both module Dropout and MultiheadAttention float dropout are versioned/reconstructed in saved checkpoints.

| Variant | Head dropout | Profile dropout | Memory/cross attention dropout | Weight decay | Direct 52 |
| --- | --- | --- | --- | ---: | --- |
| Fresh control | 0.20,0.20,0.10 | 0.10 | 0.10 | 0.0001 | False |
| Half scoring-head dropout | 0.10,0.10,0.05 | 0.10 | 0.10 | 0.0001 | False |
| Half profile dropout | 0.20,0.20,0.10 | 0.05 | 0.10 | 0.0001 | False |
| Half attention dropout | 0.20,0.20,0.10 | 0.10 | 0.05 | 0.0001 | False |
| Zero weight decay | 0.20,0.20,0.10 | 0.10 | 0.10 | 0.0 | False |
| All active dropout/weight decay zero | 0.00,0.00,0.00 | 0.00 | 0.00 | 0.0 | False |
| Direct 52 + all regularization zero | 0.00,0.00,0.00 | 0.00 | 0.00 | 0.0 | True |

Profile means company-text and tabular projections; attention includes memory attention-weight, memory feedforward/residual and candidate cross-attention dropout. Inactive attached ProductTower, unused old head and disabled candidate-attention branch are not changed. Frozen Stage 1 towers are not retrained: changing dropout at request time cannot change cached vectors.

| NDCG-selected model | Epoch | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Saved rolling pointwise A | 8 | 0.77104 | 0.60223 | 255 |
| Fresh control | 8 | 0.76938 | 0.60060 | 253 |
| Half scoring-head dropout | 8 | 0.76942 | 0.59964 | 244 |
| Half profile dropout | 6 | 0.76936 | 0.60032 | 246 |
| Half attention dropout | 8 | 0.76918 | 0.60095 | 244 |
| Zero weight decay | 8 | 0.76825 | 0.60109 | 249 |
| All active dropout/weight decay zero | 6 | 0.76830 | 0.60138 | 235 |
| Direct 52 + all regularization zero | 5 | 0.76403 | 0.59705 | 232 |

Paired NDCG-selected comparisons:

| Variant | NDCG delta vs fresh control [95% CI] | NDCG delta vs saved A [95% CI] |
| --- | --- | --- |
| Half scoring-head dropout | -0.00096 [-0.00324, +0.00139] | -0.00259 [-0.00522, -0.00033] |
| Half profile dropout | -0.00028 [-0.00238, +0.00186] | -0.00191 [-0.00390, +0.00015] |
| Half attention dropout | +0.00036 [-0.00203, +0.00261] | -0.00127 [-0.00352, +0.00084] |
| Zero weight decay | +0.00049 [-0.00124, +0.00215] | -0.00114 [-0.00274, +0.00036] |
| All active dropout/weight decay zero | +0.00078 [-0.00169, +0.00329] | -0.00085 [-0.00312, +0.00148] |
| Direct 52 + all regularization zero | -0.00355 [-0.00646, -0.00068] | -0.00518 [-0.00800, -0.00246] |

Discovery-selected records are separately shown to keep checkpoint selection explicit:

| Variant | Epoch | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Fresh control | 8 | 0.76938 | 0.60060 | 253 |
| Half scoring-head dropout | 5 | 0.77029 | 0.59931 | 252 |
| Half profile dropout | 6 | 0.76936 | 0.60032 | 246 |
| Half attention dropout | 4 | 0.76733 | 0.59761 | 250 |
| Zero weight decay | 8 | 0.76825 | 0.60109 | 249 |
| All active dropout/weight decay zero | 8 | 0.76785 | 0.60077 | 258 |
| Direct 52 + all regularization zero | 4 | 0.76426 | 0.59531 | 237 |

## What augmentation actually added

Ten rotating draws exposed 13,240,017 distinct training query/product pairs versus 2,670,525 in the fixed draw. Every epoch preserves all 45,225 eligible positive instances and the full-pool importance weighting. All products are eligible at their original query date. Validation candidates, inputs and targets remain unchanged. This broadens negative comparisons, not independent positive purchase evidence.

Arbitrary fake purchases, SMOTE/mixup over SKU IDs, or date jitter would alter history, eligibility, quantities and derived statistics inconsistently. Existing query transitions already supply historical prefix examples. History masking/cropping could be tested as an explicitly noisy training view with a coherent statistics policy, but it adds regularization and was not combined with this screen. No future validation/held-out target is used as an augmented training positive.

## Interpretation and limits

Regularization is model-dependent: reducing it helped the small statistics-only MLP; it does not follow that every ranker stage should use zero dropout. Weight decay was already modest at 1e-4. Compare each Stage 2 treatment with both its fresh control and saved A, including all ranking metrics and paired uncertainty, rather than choosing a single discovery maximum. A 128-query memorization probe from the preceding audit verified small-cohort capacity, not optimal full-data generalization.

This remains a single-seed screen with repeated validation use and fixed ten-epoch schedules. No model is promoted. Catalogue metadata are snapshot_unverified and upstream inference snapshot coherence remains a separate production prerequisite. Standalone models use common statistical containers but discard neural fields; they have no neural inference dependency.

## Verification and reproduction

Independent replay covers 14 selected ranker and 9 selected standalone MLP records. Full-denominator saved metrics, independent CPU logits, augmentation draw digests/positive counts and frozen artifact hashes are checked. Ranker training also verifies fitted permutation/padding stability. Full safe/synthetic suite results are in [checks.json](checks.json). No held-out data was accessed.

```bash
./venv/bin/python -m recommender.experiments.regularization_signal.train_ranker --execute-training
./venv/bin/python -m recommender.experiments.regularization_signal.train_tabular --execute-training --device cpu
./venv/bin/python -m recommender.experiments.regularization_signal.verify --stage tabular
./venv/bin/python -m recommender.experiments.regularization_signal.verify --stage ranker
./venv/bin/python -m recommender.experiments.regularization_signal.report
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
PYTHONPATH=. ./venv/bin/pytest tests/ -q
```

Training and verification refuse overwriting completed evidence. Reproduction requires an explicitly separate run namespace. Source code is isolated under recommender/experiments/regularization_signal/.
