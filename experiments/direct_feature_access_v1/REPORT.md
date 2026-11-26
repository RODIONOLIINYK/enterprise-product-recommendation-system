# Direct feature access and underfitting audit

Development-only screen: train before 2025-10-01, validation 2025-10-01 through 2026-02-18. 26,253 train queries, 6,191 validation queries, 10,276 complete validation targets, including 1,857 new targets. No held-out data access, reference promotion, commit or push.

## What was and was not missing

All 41 CatBoost rolling context features already enter the 425D Stage 2 scoring MLP. Popularity, category affinity, cadence, demand, transition lifts and same-basket signals were not missing. The Stage 1 reference directly consumes 39 of these, deliberately excluding month sine/cosine, plus six upstream customer statistics. Five of those six overlap CatBoost profile8; the remaining one is median_purchase_quantity. The three CatBoost profile fields absent as separate Stage 1 scalars are total_purchases, days_since_first_order and days_since_last_order. Some overlap other context statistics, so absence of a named scalar does not imply total absence of its information.

Stage 2 profile8 previously acted through profile attention bias; under statistics_attention the profile numeric token values are zero. Candidate IDs/category/line were indirect through frozen candidate vectors, with explicit identity embeddings only on historical items. See [feature_audit.json](feature_audit.json) for all 52 fields.

## Controlled implementation

Four fresh arms share identical rolling quota100_50 slates, frozen u/v sources, original 32-event memory and attention, head widths 256/128/64, RankerLoss, AdamW/cosine schedule, batch64, seed42, and ten epochs. Only direct scoring inputs change. Profile8 adds eight existing source scalars; categoricals add separately learned SKU16/category8/line8 embeddings. Head input widths are 425/433/457/465. Added first-layer columns start at zero; all original weights are exact copies. Candidate attention, token enrichment and Stage 1 MLP score inputs remain disabled.

Initial maximum logit differences: `{'control': 0.0, 'profile8': 0.0, 'candidate_categories': 0.0, 'profile8_and_categories': 0.0}` (float rounding only). A fresh control quantifies MPS run variation; saved A remains the stronger comparator. Catalogue categoricals retain the same snapshot_unverified limitation as CatBoost. The Stage 1 selector/default aliases are not altered.

## Validation results: NDCG-selected checkpoints

| Model | Epoch/trees | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Saved pointwise A | 8 | 0.77104 | 0.60223 | 255 |
| Fresh control | 8 | 0.76954 | 0.60105 | 251 |
| Direct profile8 | 6 | 0.76910 | 0.59877 | 238 |
| Direct SKU/category/line | 8 | 0.76604 | 0.59693 | 238 |
| Both direct paths | 8 | 0.76633 | 0.59774 | 242 |
| Standalone CatBoost | 318 | 0.75815 | 0.59373 | 202 |

None of the three direct-input treatments beats the fresh control or saved A on NDCG, Recall@10 or new hits. No variant is promoted. The CatBoost row uses a full eligible pool and a different binary objective; it is a practical pipeline comparison, not an architecture-only ablation.

Discovery-selected checkpoints, without mixing their metrics with NDCG selections:

| Arm | Epoch | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Fresh control | 6 | 0.77074 | 0.60011 | 254 |
| Direct profile8 | 7 | 0.76675 | 0.59648 | 243 |
| Direct SKU/category/line | 8 | 0.76604 | 0.59693 | 238 |
| Both direct paths | 8 | 0.76633 | 0.59774 | 242 |

Paired customer-clustered bootstrap: delta in NDCG@10, 95% interval; conditional on these selected seed42 checkpoints and repeated validation use.

| Arm | vs fresh control | vs saved A |
| --- | --- | --- |
| Direct profile8 | -0.00228 [-0.00440, -0.00011] | -0.00346 [-0.00583, -0.00113] |
| Direct SKU/category/line | -0.00412 [-0.00709, -0.00130] | -0.00530 [-0.00845, -0.00225] |
| Both direct paths | -0.00332 [-0.00620, -0.00050] | -0.00449 [-0.00762, -0.00146] |

## Is this underfitting?

Capacity is not the same as useful generalization. Earlier wider-head training loss fell from 3.13755 to 2.82055 while its validation loss reached 2.68227 at epoch6 and rose to 2.70246 at epoch10. Its selected NDCG remained below A. The fresh original control likewise improved training loss while validation loss stopped improving late in training. Training-mode loss includes dropout; it must not be interpreted as a directly comparable train-vs-validation generalization gap.

Below, losses use evaluation mode and identical loss settings on fixed independently sampled 4,096-row training/validation cohorts. The two periods have different difficulty/population mixes, so the gap alone cannot identify overfitting. Conditional recall excludes targets missing from the candidate slate and is not the full-denominator headline Recall@10.

| Model | Eval train loss | Eval validation loss | Train conditional micro R@10 | Val conditional micro R@10 |
| --- | ---: | ---: | ---: | ---: |
| Fresh control | 2.8640 | 2.7149 | 0.7504 | 0.7909 |
| Direct profile8 | 2.9259 | 2.7115 | 0.7482 | 0.7909 |
| Direct SKU/category/line | 2.8539 | 2.7498 | 0.7561 | 0.7888 |
| Both direct paths | 2.8558 | 2.7399 | 0.7541 | 0.7874 |
| saved_A | 2.8652 | 2.7112 | 0.7498 | 0.7917 |

Train-only optimizer/capacity probe: original pointwise model, 128 fixed training queries, 600 optimizer updates, lr=0.001, dropout disabled, weight_decay=0. Loss `5.3721 -> 0.4032`, conditional micro Recall@10 `0.4317 -> 1.0000`. The multi-positive listwise entropy floor is `0.3924`, so zero combined loss is not the appropriate memorization target. This is a sanity check, not a validation candidate or proof of optimal full-data optimization.

## How much independent signal is available?

Millions of query/product pairs are mostly generated non-purchases, not millions of independent observed preference events. Training queries share customers and overlapping histories. Purchase-only labels lack exposure and inventory information; they cannot distinguish refusal from non-exposure. No universal sample threshold determines whether this architecture has enough data.

Measured support:

```json
{
  "train": {
    "queries": 26253,
    "distinct_customers": 2319,
    "target_instances": 45721,
    "new_target_instances": 11530,
    "reachable_target_instances": 44227,
    "target_count_per_sku_median": 8.0,
    "skus_with_at_most_10_target_instances": 448,
    "max_sku_target_instances": 1082
  },
  "val": {
    "queries": 6191,
    "distinct_customers": 1676,
    "target_instances": 10276,
    "new_target_instances": 1857,
    "reachable_target_instances": 9703,
    "target_count_per_sku_median": 2.0,
    "skus_with_at_most_10_target_instances": 627,
    "max_sku_target_instances": 231
  },
  "validation_new_targets_by_training_support": {
    "at_most_0_training_target_instances": 97,
    "at_most_5_training_target_instances": 210,
    "at_most_10_training_target_instances": 335,
    "at_most_20_training_target_instances": 465,
    "at_most_50_training_target_instances": 734
  },
  "validation_customers_unseen_in_training": 252
}
```

## Why CatBoost is competitive

Its 41 context scalars account for 84.91% of PredictionValuesChange importance, profile8 13.61%, categorical3 1.48%. These are fitted-model prediction-change importances, not causal contributions or percentages of accuracy. The dominant features are hierarchical_backoff and industry_sku_affinity; both already reach the neural scorer. CatBoost is a boosted tree ensemble using extensive engineered relational statistics, not a model learning everything from raw records.

Primary external context: [Grinsztajn et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf) find trees strong on medium-sized tabular datasets, with robustness to irrelevant features and irregular target functions. This is supporting context, not evidence about our exact model. [CatBoost importance definition](https://catboost.ai/docs/en/concepts/fstr).

## Practical next decision

Keep CatBoost as a first-class practical baseline. Larger attention/MLP capacity has not earned its complexity in completed screens. Direct access to existing fields must win a matched comparison before integration. Prioritize genuinely new preference evidence (exposures, availability and purchase opportunity), coherent snapshots, and validation learning curves before another size increase. The next architecture-focused diagnostic would compare a statistics-only MLP with CatBoost using identical full-pool rows, inputs, sampling weights and binary objective; the present tree/neural comparison changes objective/candidate policy too. Objective repeat/new weighting was already tried by the user and is not proposed as a new experiment.

## Verification and reproduction

Full selected-model inference replay, permutation/padding checks, CPU replay on 32 validation rows, tensor parity with shared profile/context sources, strict checkpoint lookup/config/provenance checks, and all 31 protected reference/cache hashes are checked. See [diagnostics.json](diagnostics.json), [summary.json](summary.json), and [checks.json](checks.json).

```bash
./venv/bin/python -m recommender.experiments.direct_feature_access.train --execute-training
./venv/bin/python -m recommender.experiments.direct_feature_access.diagnostics
./venv/bin/python -m recommender.experiments.direct_feature_access.report
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
PYTHONPATH=. ./venv/bin/pytest tests/ -q
```

Training creates exclusive run artifacts and the summary/diagnostics refuse overwrites. Do not rerun these commands to overwrite evidence.
