# Stage 1 MLP Feature-Removal Experiment (`no_repeat_no_month` / `uv_44`)

## 1. Experiment Scope & Controlled Setup

- **Baseline Reference:** `no-brand-v1` (`CompactStage1Selector`, `uv_47`, `303D = 128 + 128 + 41 + 6`, `47,617` params, selected epoch `1/10`, checkpoint SHA-256 `d083a5562d968e8e606f69d86e142e370567fa4f506f650e9e2234c53d66dad5`)
- **Treatment Variant:** `no_repeat_no_month` (`FeatureAblatedStage1Selector`, `uv_44`, `300D = 128 + 128 + 38 + 6`, `47,233` params, selected epoch `5/10`, checkpoint SHA-256 `44904b172da3d2a7dc504fd3832374509753fe6181deeb8a4ddda60cf9f731f3`)
- **Frozen Upstream Towers:** `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt` (epoch 16, SHA-256 `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981`); fixed saved `u/v` purchase caches reused read-only with zero tower re-encoding.
- **Omitted Contextual Features (3 of 41, resolved by schema name):** `0: is_repeat_sku`, `31: month_sin`, `32: month_cos` (`38` retained context features + `6` customer stats = `44` scalar inputs; `head.0` physically `Linear(300, 128)`).
- **Training Protocol:** `seed=42`, `10` epochs, `AdamW(lr=0.001, weight_decay=0.0001)`, `CosineAnnealingLR(eta_min=1e-5)`, `batch_size=64`, `BCEWithLogitsLoss(pos_weight=20.0)` over full eligible pool, `clip_grad_norm_(1.0)`, selection `(new_hits@50, ndcg@10, recall@10)`.
- **Data Safety & Denominators:** Train `< 2025-10-01` (`26,253` queries with >=100 eligible SKUs); Val `2025-10-01..2026-02-18` (`6,191` queries, `1,857` new-to-customer targets, `10,276` total targets, `841` SKU universe, `735–771` eligible/query). Held-out split `>= 2026-02-19` untouched (`heldout_access=false`). No Stage 2 execution, no model promotion.

## 2. Validation Comparison (All 6,191 Queries / 1,857 New Targets)

> [!NOTE]
> Confidence intervals below are paired **customer-clustered 95% bootstrap intervals** (`2,000` draws, seed `123`) over the `6,191` validation queries at selected weights for a single seed (`seed=42`). They measure query/customer sampling uncertainty conditional on these selected weights, not multi-seed training variance.

### Overall Ranking Metrics (All Eligible Candidates, Repeat + New, No Unseen Filtering)

| Metric | Baseline `303D` (`uv_47`, Ep 1) | Variant `300D` (`uv_44`, Ep 5) | Delta | 95% Customer-Clustered Bootstrap CI |
|---|---:|---:|---:|---:|
| Overall Macro `Recall@5` | 64.025% (`0.640251`) | 63.885% (`0.638847`) | `-0.140 pp` | `-0.144 pp [-0.555, +0.288]` |
| Overall Macro `Recall@10` | 75.656% (`0.756559`) | 75.625% (`0.756250`) | `-0.031 pp` | `-0.027 pp [-0.426, +0.378]` |
| Overall Macro `Recall@25` | 85.195% (`0.851952`) | 84.990% (`0.849901`) | `-0.205 pp` | `-0.197 pp [-0.609, +0.202]` |
| Overall Macro `Recall@50` | 89.696% (`0.896959`) | 89.629% (`0.896294`) | `-0.067 pp` | `-0.068 pp [-0.471, +0.343]` |
| Overall Macro `Recall@100` | 93.685% (`0.936849`) | 93.703% (`0.937029`) | `+0.018 pp` | `+0.021 pp [-0.338, +0.364]` |
| Overall `NDCG@10` | `0.578286` | `0.583116` | `+0.004830` | `+0.00473 [+0.00144, +0.00905]` |
| Overall `MRR` | `0.553445` | `0.561641` | `+0.008195` | `—` |

### New-Target Discovery Metrics (`1,857` New-to-Customer Target Instances)

| Metric | Ranking Pool | Baseline `303D` (`uv_47`) | Variant `300D` (`uv_44`) | Delta | 95% Customer-Clustered Bootstrap CI |
|---|---|---:|---:|---:|---:|
| `new_hits@10` / `new_recall@10` | Ordinary Top-10 (all eligible) | `214 / 1,857` (11.524%) | `205 / 1,857` (11.039%) | `-9` (`-0.485 pp`) | `-0.453 pp [-1.578, +0.590]` |
| `new_hits@25` / `new_recall@25` | Unseen-only | `741 / 1,857` (39.903%) | `729 / 1,857` (39.257%) | `-12` (`-0.646 pp`) | `-0.615 pp [-2.345, +1.088]` |
| `new_hits@50` / `new_recall@50` (selection key) | Unseen-only | `1037 / 1,857` (55.843%) | `1038 / 1,857` (55.897%) | `+1` (`+0.054 pp`) | `+0.052 pp [-1.849, +1.975]` |
| `new_hits@100` / `new_recall@100` | Unseen-only | `1324 / 1,857` (71.298%) | `1335 / 1,857` (71.890%) | `+11` (`+0.592 pp`) | `+0.601 pp [-1.097, +2.222]` |

## 3. Per-Epoch History (`300D` Variant, `seed=42`)

| Epoch | Train Loss | Overall `Recall@5` | Overall `Recall@10` | Overall `Recall@25` | Overall `Recall@50` | Overall `Recall@100` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.09100 | 0.63909 | 0.75636 | 0.84877 | 0.89572 | 0.93654 | 0.58073 | 219 | 736 | 1026 | 1337 |  |
| 2 | 0.07300 | 0.63699 | 0.75522 | 0.84870 | 0.89330 | 0.93579 | 0.58137 | 206 | 721 | 1031 | 1314 |  |
| 3 | 0.07003 | 0.64031 | 0.75631 | 0.84766 | 0.89436 | 0.93460 | 0.58252 | 213 | 713 | 1003 | 1312 |  |
| 4 | 0.06905 | 0.64036 | 0.75564 | 0.84885 | 0.89650 | 0.93668 | 0.58275 | 209 | 747 | 1012 | 1342 |  |
| 5 | 0.06775 | 0.63885 | 0.75625 | 0.84990 | 0.89629 | 0.93703 | 0.58312 | 205 | 729 | 1038 | 1335 | YES |
| 6 | 0.06701 | 0.64014 | 0.75705 | 0.85138 | 0.89660 | 0.93564 | 0.58440 | 206 | 751 | 1015 | 1315 |  |
| 7 | 0.06616 | 0.63926 | 0.75663 | 0.85040 | 0.89520 | 0.93418 | 0.58514 | 200 | 746 | 1016 | 1315 |  |
| 8 | 0.06556 | 0.64110 | 0.75661 | 0.85058 | 0.89439 | 0.93437 | 0.58522 | 209 | 748 | 1012 | 1311 |  |
| 9 | 0.06496 | 0.64045 | 0.75715 | 0.85061 | 0.89386 | 0.93450 | 0.58544 | 204 | 732 | 1007 | 1327 |  |
| 10 | 0.06467 | 0.64096 | 0.75693 | 0.85019 | 0.89440 | 0.93442 | 0.58555 | 202 | 734 | 1005 | 1321 |  |

## 4. Training Cost, Invariance Checks & Preserved Hashes

- **Device:** `mps`
- **Training time (10 epochs, 26,253 rows):** `116.88s`
- **Validation replay inference time (6,191 rows x 841 SKUs):** `0.813s`
- **Parameters:** `47,233` (`first_layer_in_features=300`, reduced by `384 = 3 * 128` weights vs baseline `47,617`)
- **Omission invariance max abs diff (perturbing cols 0, 31, 32):** `0.0`
- **Retained feature sensitivity max abs diff (zeroing retained 38 cols):** `10.584016`
- **Preserved artifact/audit/reference hashes before & after:** `ALL MATCHED`

## 5. Retained Feature Contract (`38` Context + `6` Customer Stats = `44` Scalars, `300D` Total Input)

Remaining scales are unchanged from `no-brand-v1`: `u` (`128D`) and `v` (`128D`) are L2-normalized signed vectors; `customer_stats_6` are in `[0, 1]`; of the `38` retained contextual features, `30` are in `[0, 1]` and `8` remain signed in `[-1, 1]` (`sku_growth_30d`, `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`, `line_last_lift`, `hierarchical_backoff`, `same_basket_lift`).

| Retained Context Index | Source Cache Index | MLP Column | Feature Name |
|---:|---:|---:|---|
| 0 | 1 | 256 | `user_sku_order_count_norm` |
| 1 | 2 | 257 | `user_sku_qty_share` |
| 2 | 3 | 258 | `days_since_last_sku_buy_norm` |
| 3 | 4 | 259 | `sku_urgency` |
| 4 | 5 | 260 | `user_cat_order_share` |
| 5 | 6 | 261 | `user_subcat_order_share` |
| 6 | 7 | 262 | `user_brand_order_share` |
| 7 | 8 | 263 | `global_sku_popularity_norm` |
| 8 | 9 | 264 | `industry_sku_affinity` |
| 9 | 10 | 265 | `is_repeat_category` |
| 10 | 11 | 266 | `is_repeat_brand` |
| 11 | 12 | 267 | `is_repeat_business_line` |
| 12 | 13 | 268 | `customer_age` |
| 13 | 14 | 269 | `customer_overdue_ratio` |
| 14 | 15 | 270 | `customer_30d_activity` |
| 15 | 16 | 271 | `customer_90d_activity` |
| 16 | 17 | 272 | `category_age` |
| 17 | 18 | 273 | `category_overdue_ratio` |
| 18 | 19 | 274 | `category_never_seen` |
| 19 | 20 | 275 | `family_age` |
| 20 | 21 | 276 | `family_overdue_ratio` |
| 21 | 22 | 277 | `family_never_seen` |
| 22 | 23 | 278 | `family_history_share` |
| 23 | 24 | 279 | `sku_orders_7d` |
| 24 | 25 | 280 | `sku_orders_30d` |
| 25 | 26 | 281 | `sku_orders_90d` |
| 26 | 27 | 282 | `sku_buyers_30d` |
| 27 | 28 | 283 | `sku_growth_30d` |
| 28 | 29 | 284 | `category_relative_share` |
| 29 | 30 | 285 | `global_sale_age` |
| 30 | 33 | 286 | `sku_last_lift` |
| 31 | 34 | 287 | `sku_recent3_lift` |
| 32 | 35 | 288 | `category_last_lift` |
| 33 | 36 | 289 | `family_last_lift` |
| 34 | 37 | 290 | `line_last_lift` |
| 35 | 38 | 291 | `sku_support_confidence` |
| 36 | 39 | 292 | `hierarchical_backoff` |
| 37 | 40 | 293 | `same_basket_lift` |

Plus `customer_stats_6` at MLP columns `294..299`: `distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`.

## 6. Reproduction Commands

```bash
./venv/bin/python -m recommender.experiments.mlp_feature_ablation.train
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
```
