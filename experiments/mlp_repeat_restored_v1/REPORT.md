# Stage 1 MLP Repeat-Restored Experiment (`repeat_no_month` / `uv_45` / `301D`)

## 1. Experiment Scope & Controlled Setup

- **Baseline Reference (`303D`):** `no-brand-v1` (`CompactStage1Selector`, `uv_47`, `303D = 128 + 128 + 41 + 6`, `47,617` params, selected epoch `1/10`, SHA-256 `d083a5562d968e8e606f69d86e142e370567fa4f506f650e9e2234c53d66dad5`).
- **Prior Ablated Variant (`300D`):** `no_repeat_no_month` (`FeatureAblatedStage1Selector`, `uv_44`, `300D = 128 + 128 + 38 + 6`, `47,233` params, selected epoch `5/10`, SHA-256 `44904b172da3d2a7dc504fd3832374509753fe6181deeb8a4ddda60cf9f731f3`).
- **Treatment Variant (`301D`):** `repeat_no_month` (`FeatureAblatedStage1Selector`, `uv_45`, `301D = 128 + 128 + 39 + 6`, `47,361` params, selected epoch `5/10`, SHA-256 `977241cf9fe8a36ae5c0b5afede6c8018e55eb3202e39d88fa7e3693bc0bc22a`).
- **Frozen Upstream Towers:** `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt` (epoch 16, SHA-256 `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981`); saved train/val `u`/`v` purchase caches reused read-only with zero tower re-encoding.
- **Feature Change Only:** Restores ONLY `0: is_repeat_sku` while keeping `31: month_sin` and `32: month_cos` physically omitted (`39` retained context features + `6` customer stats = `45` scalar inputs; `head.0` physically `Linear(301, 128)`).
- **Training Protocol:** `seed=42`, `10` epochs from scratch, `AdamW(lr=0.001, weight_decay=0.0001)`, `CosineAnnealingLR(eta_min=1e-5)`, `batch_size=64`, `BCEWithLogitsLoss(pos_weight=20.0)` over full rolling eligible pool, `clip_grad_norm_(1.0)`, selection `(new_hits@50, ndcg@10, recall@10)`.
- **Data Safety & Explicit Denominators:** Train `< 2025-10-01` (`26,253` queries with >=100 eligible SKUs); Val `2025-10-01..2026-02-18` (`6,191` queries, `10,276` total targets, `1,857` new-to-customer targets, `841` SKU universe, `735–771` eligible/query). Held-out split `>= 2026-02-19` untouched (`heldout_access=false`). Metadata policy: `snapshot_unverified`. No Stage 2 execution, no active-reference mutation (`promoted=false`).
- **Pending Legacy Normalization:** Current numeric normalization remains legacy (`u`/`v` L2-normalized in `[-1, 1]`, `customer_stats_6` in `[0, 1]`, `31` retained context columns in `[0, 1]`, and `8` signed lift/growth columns in `[-1, 1]`). Numeric normalization and category-feature work remain pending.

## 2. Validation Comparison vs Both Baselines (`6,191` Queries / `10,276` Total Targets / `1,857` New Targets)

> [!NOTE]
> Confidence intervals are paired **customer-clustered 95% bootstrap intervals** (`2,000` draws, seed `123`) over the `6,191` validation queries at selected weights for a single training seed (`seed=42`). They reflect query/customer sampling uncertainty conditional on these selected weights, **not** multi-seed training variance; do not claim repeatable superiority from query bootstrap alone.

### Overall Ranking Metrics (All Eligible Candidates, Repeat + New, No Unseen Filtering)

| Metric | Baseline `303D` (`uv_47`, Ep 1) | Prior `300D` (`uv_44`, Ep 5) | Variant `301D` (`uv_45`, Ep 5) | Delta vs `303D` [95% CI] | Delta vs `300D` [95% CI] |
|---|---:|---:|---:|---:|---:|
| Overall Macro `Recall@5` | 64.025% (`0.640251`) | 63.885% (`0.638847`) | 64.335% (`0.643349`) | `+0.310 pp [-0.139, +0.731]` | `+0.450 pp [+0.102, +0.803]` |
| Overall Macro `Recall@10` | 75.656% (`0.756559`) | 75.625% (`0.756250`) | 75.411% (`0.754114`) | `-0.245 pp [-0.676, +0.152]` | `-0.214 pp [-0.526, +0.097]` |
| Overall Macro `Recall@25` | 85.195% (`0.851952`) | 84.990% (`0.849901`) | 84.850% (`0.848498`) | `-0.345 pp [-0.749, +0.074]` | `-0.140 pp [-0.420, +0.138]` |
| Overall Macro `Recall@50` | 89.696% (`0.896959`) | 89.629% (`0.896294`) | 89.542% (`0.895424`) | `-0.154 pp [-0.571, +0.242]` | `-0.087 pp [-0.375, +0.169]` |
| Overall Macro `Recall@100` | 93.685% (`0.936849`) | 93.703% (`0.937029`) | 93.733% (`0.937329`) | `+0.048 pp [-0.318, +0.394]` | `+0.030 pp [-0.181, +0.240]` |
| Overall `NDCG@10` | `0.578286` | `0.583116` | `0.586331` | `+0.00804 [+0.00418, +0.01227]` | `+0.00321 [+0.00070, +0.00547]` |
| Overall `MRR` | `0.553445` | `0.561641` | `0.566148` | `+0.012702` | `+0.004507` |
| Repeat `Recall@10` (Macro / Micro) | 90.958% / 88.241% | 91.050% / 88.324% | 91.041% / 88.407% | `+0.083` / `+0.166 pp` | `-0.009` / `+0.083 pp` |

### New-Target Discovery Metrics (`1,857` New-to-Customer Target Instances)

| Metric | Ranking Pool | Baseline `303D` (`uv_47`, Ep 1) | Prior `300D` (`uv_44`, Ep 5) | Variant `301D` (`uv_45`, Ep 5) | Delta vs `303D` [95% CI] | Delta vs `300D` [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| `new_hits@10` / `new_recall@10` | Ordinary Top-10 (all eligible) | `214 / 1,857` (11.524%) | `205 / 1,857` (11.039%) | `194 / 1,857` (10.447%) | `-20` (`-1.077 pp [-2.159, +0.000]`) | `-11` (`-0.592 pp [-1.412, +0.168]`) |
| `new_hits@25` / `new_recall@25` | Unseen-only | `741 / 1,857` (39.903%) | `729 / 1,857` (39.257%) | `726 / 1,857` (39.095%) | `-15` (`-0.808 pp [-2.612, +1.075]`) | `-3` (`-0.162 pp [-1.418, +1.101]`) |
| `new_hits@50` / `new_recall@50` (selection key) | Unseen-only | `1037 / 1,857` (55.843%) | `1038 / 1,857` (55.897%) | `1029 / 1,857` (55.412%) | `-8` (`-0.431 pp [-2.394, +1.532]`) | `-9` (`-0.485 pp [-1.529, +0.626]`) |
| `new_hits@100` / `new_recall@100` | Unseen-only | `1324 / 1,857` (71.298%) | `1335 / 1,857` (71.890%) | `1337 / 1,857` (71.998%) | `+13` (`+0.700 pp [-0.957, +2.305]`) | `+2` (`+0.108 pp [-0.909, +1.153]`) |

## 3. Descriptive Epoch-1 Comparison (Checkpoint-Selection Context)

Epoch 1 metrics are shown descriptively below for all three variants to separate feature-set differences from checkpoint-selection epoch differences (`baseline_303` selected Ep 1; `prior_300` selected Ep 5; `variant_301` selected Ep 5). No extra epoch was selected or retrained after viewing metrics.

| Variant (Epoch 1) | `Recall@5` | `Recall@10` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline `303D` (`uv_47`, Ep 1) | 0.64025 | 0.75656 | 0.57829 | 214 | 741 | 1037 | 1324 |
| Prior `300D` (`uv_44`, Ep 1) | 0.63909 | 0.75636 | 0.58073 | 219 | 736 | 1026 | 1337 |
| Variant `301D` (`uv_45`, Ep 1) | 0.63979 | 0.75581 | 0.57821 | 214 | 735 | 1016 | 1331 |

## 4. Per-Epoch History (`301D` Variant `repeat_no_month` / `uv_45`, `seed=42`)

| Epoch | Train Loss | Overall `Recall@5` | Overall `Recall@10` | Overall `Recall@25` | Overall `Recall@50` | Overall `Recall@100` | `NDCG@10` | `new_hits@10` | `new_hits@25` | `new_hits@50` | `new_hits@100` | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.09557 | 0.63979 | 0.75581 | 0.85009 | 0.89644 | 0.93629 | 0.57821 | 214 | 735 | 1016 | 1331 |  |
| 2 | 0.07358 | 0.63869 | 0.75417 | 0.84706 | 0.89516 | 0.93530 | 0.58147 | 190 | 705 | 1000 | 1329 |  |
| 3 | 0.07036 | 0.64245 | 0.75452 | 0.84773 | 0.89583 | 0.93508 | 0.58452 | 202 | 721 | 1005 | 1334 |  |
| 4 | 0.06908 | 0.64319 | 0.75468 | 0.84895 | 0.89645 | 0.93822 | 0.58502 | 207 | 741 | 1023 | 1345 |  |
| 5 | 0.06779 | 0.64335 | 0.75411 | 0.84850 | 0.89542 | 0.93733 | 0.58633 | 194 | 726 | 1029 | 1337 | YES |
| 6 | 0.06704 | 0.64507 | 0.75611 | 0.85136 | 0.89599 | 0.93646 | 0.58711 | 203 | 737 | 1022 | 1333 |  |
| 7 | 0.06628 | 0.64642 | 0.75502 | 0.84825 | 0.89485 | 0.93555 | 0.58789 | 205 | 740 | 1021 | 1321 |  |
| 8 | 0.06565 | 0.64385 | 0.75611 | 0.84792 | 0.89381 | 0.93529 | 0.58834 | 206 | 733 | 1014 | 1327 |  |
| 9 | 0.06516 | 0.64725 | 0.75730 | 0.84813 | 0.89462 | 0.93560 | 0.58985 | 208 | 729 | 1012 | 1334 |  |
| 10 | 0.06484 | 0.64684 | 0.75707 | 0.84772 | 0.89399 | 0.93489 | 0.58951 | 212 | 728 | 999 | 1325 |  |

## 5. Runtime Cost, Invariance, `is_repeat_sku` Influence & Candidate Artifacts

- **Device:** `mps`
- **Training time (10 epochs, 26,253 rows):** `113.11s`
- **Validation replay inference time (6,191 rows x 841 SKUs):** `0.816s`
- **Parameters:** `47,361` (`first_layer_in_features=301`, `-256 = -2 * 128` weights vs `303D`, `+128 = +1 * 128` weights vs `300D`)
- **Month omission invariance max abs diff (perturbing source cols 31, 32):** `0.0`
- **Retained feature sensitivity max abs diff (zeroing retained 39 cols):** `10.273620`
- **Retained `is_repeat_sku` influence (`1` vs `0` across 4,700,328 eligible val pairs):** max abs diff `8.673151`, mean abs diff `3.181254`, mean signed `(1 - 0)` `+3.180984`
- **Candidate checkpoint copy (not active reference):** `checkpoints/mlp-repeat-restored-v1/candidate/best_selector.pt` (SHA-256 `977241cf9fe8a36ae5c0b5afede6c8018e55eb3202e39d88fa7e3693bc0bc22a`, byte-identical=True, promoted=False)
- **Candidate manifest:** `checkpoints/mlp-repeat-restored-v1/candidate/manifest.json` (SHA-256 `f11bc21eed8767132e51b413ce1e42dc10626c4d2bb2d076f68af6e85d0837d5`)
- **Preserved audit/reference/prior-300 hashes before & after:** `ALL MATCHED`

## 6. Retained Feature Contract (`39` Context + `6` Customer Stats = `45` Scalars, `301D` Total Input)

Canonical source column `0: is_repeat_sku` is retained at MLP column `256`; only `31: month_sin` and `32: month_cos` are omitted. Legacy numeric normalization remains unchanged (`u`/`v` L2-normalized in `[-1, 1]`, `customer_stats_6` in `[0, 1]`, `31` retained context columns in `[0, 1]`, `8` signed columns in `[-1, 1]`).

| Retained Context Index | Source Cache Index | MLP Column | Feature Name |
|---:|---:|---:|---|
| 0 | 0 | 256 | `is_repeat_sku` |
| 1 | 1 | 257 | `user_sku_order_count_norm` |
| 2 | 2 | 258 | `user_sku_qty_share` |
| 3 | 3 | 259 | `days_since_last_sku_buy_norm` |
| 4 | 4 | 260 | `sku_urgency` |
| 5 | 5 | 261 | `user_cat_order_share` |
| 6 | 6 | 262 | `user_subcat_order_share` |
| 7 | 7 | 263 | `user_brand_order_share` |
| 8 | 8 | 264 | `global_sku_popularity_norm` |
| 9 | 9 | 265 | `industry_sku_affinity` |
| 10 | 10 | 266 | `is_repeat_category` |
| 11 | 11 | 267 | `is_repeat_brand` |
| 12 | 12 | 268 | `is_repeat_business_line` |
| 13 | 13 | 269 | `customer_age` |
| 14 | 14 | 270 | `customer_overdue_ratio` |
| 15 | 15 | 271 | `customer_30d_activity` |
| 16 | 16 | 272 | `customer_90d_activity` |
| 17 | 17 | 273 | `category_age` |
| 18 | 18 | 274 | `category_overdue_ratio` |
| 19 | 19 | 275 | `category_never_seen` |
| 20 | 20 | 276 | `family_age` |
| 21 | 21 | 277 | `family_overdue_ratio` |
| 22 | 22 | 278 | `family_never_seen` |
| 23 | 23 | 279 | `family_history_share` |
| 24 | 24 | 280 | `sku_orders_7d` |
| 25 | 25 | 281 | `sku_orders_30d` |
| 26 | 26 | 282 | `sku_orders_90d` |
| 27 | 27 | 283 | `sku_buyers_30d` |
| 28 | 28 | 284 | `sku_growth_30d` |
| 29 | 29 | 285 | `category_relative_share` |
| 30 | 30 | 286 | `global_sale_age` |
| 31 | 33 | 287 | `sku_last_lift` |
| 32 | 34 | 288 | `sku_recent3_lift` |
| 33 | 35 | 289 | `category_last_lift` |
| 34 | 36 | 290 | `family_last_lift` |
| 35 | 37 | 291 | `line_last_lift` |
| 36 | 38 | 292 | `sku_support_confidence` |
| 37 | 39 | 293 | `hierarchical_backoff` |
| 38 | 40 | 294 | `same_basket_lift` |

Plus `customer_stats_6` at MLP columns `295..300`: `distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`.

## 7. Reproduction Commands

```bash
./venv/bin/python -m recommender.experiments.mlp_feature_ablation.run_repeat_restored
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
```
