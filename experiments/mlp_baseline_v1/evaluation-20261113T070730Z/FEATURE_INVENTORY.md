# Single saved MLP baseline evaluation

Reference: `no-brand-v1`; saved MLP selected epoch 1. One inference evaluation, no training or upstream tower evaluation.

Validation: 2025-10-01 through 2026-02-18, 6,191 queries, 1,857 new targets. Full rolling pool: 735–771 eligible candidates per query; purchased development universe 841 products.

## Results

- `queries`: 6191
- `new_targets`: 1857
- `recall@10`: 0.7565593823119603
- `ndcg@10`: 0.5782861484383223
- `new_hits@10`: 214
- `new_recall@10`: 0.11523963381798599
- `new_hits@25`: 741
- `new_recall@25`: 0.3990306946688207
- `new_hits@50`: 1037
- `new_recall@50`: 0.5584275713516424
- `new_hits@100`: 1324
- `new_recall@100`: 0.7129779213785676

## Architecture

`[customer u:128, product v:128, contextual:41, customer statistics:6] -> Linear(303,128) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(128,64) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(64,1)`. Dropout is disabled during this evaluation. Output is a ranking logit. The `uv_47` recipe has no separate dot-score input, dot product, u*v or absolute-difference block.

Customer u encodes company text, city/region/industry/sales_branch and the last 32 purchase events. Product v encodes t1_fam text, category/subcategory/business_line/tier/unit/country_of_origin embeddings and static container one-hot inputs. Separate product brand embedding is absent; brand text and MLP brand features remain. No descriptions are included.

u and v are L2-normalized signed vectors, not scaled to [0,1].

## Exact ordered 41 contextual features

Indices below are zero-based. The original names are preserved, with misleading names clarified in the meaning column.

| Context index | MLP column | Name | Meaning | Normalization | Observed eligible validation range |
|---:|---:|---|---|---|---|
| 0 | 256 | `is_repeat_sku` | Candidate SKU purchased before | Binary 0/1 | [0, 1] |
| 1 | 257 | `user_sku_order_count_norm` | Prior customer orders containing this SKU | clip(log1p(count)/log1p(50),0,1) | [0, 1] |
| 2 | 258 | `user_sku_qty_share` | Share of customer purchased quantity allocated to this SKU | Ratio [0,1] | [0, 1] |
| 3 | 259 | `days_since_last_sku_buy_norm` | Elapsed days since customer bought this SKU | clip(log1p(days)/log1p(365),0,1); 1 if never bought | [0.117432, 1] |
| 4 | 260 | `sku_urgency` | SKU elapsed time relative to customer SKU median purchase gap | clip(days/max(median_gap,1)/5,0,1); 0 with fewer than two purchases | [0, 1] |
| 5 | 261 | `user_cat_order_share` | Share of customer orders containing candidate category | Ratio [0,1] | [0, 1] |
| 6 | 262 | `user_subcat_order_share` | Share of customer orders containing candidate subcategory | Ratio [0,1] | [0, 1] |
| 7 | 263 | `user_brand_order_share` | Share of customer orders containing candidate brand | Ratio [0,1] | [0, 1] |
| 8 | 264 | `global_sku_popularity_norm` | Global prior training-history sale count for candidate SKU | clip(log1p(count)/log1p(1000),0,1) | [0, 1] |
| 9 | 265 | `industry_sku_affinity` | Candidate share of sales in customer industry | Ratio [0,1] | [0, 0.357178] |
| 10 | 266 | `is_repeat_category` | Candidate category purchased before | Binary 0/1 | [0, 1] |
| 11 | 267 | `is_repeat_brand` | Candidate brand purchased before | Binary 0/1 | [0, 1] |
| 12 | 268 | `is_repeat_business_line` | Candidate business line purchased before | Binary 0/1 | [0, 1] |
| 13 | 269 | `customer_age` | Days since customer last purchased anything; not customer tenure | clip(days/365,0,1) | [0.00273895, 1] |
| 14 | 270 | `customer_overdue_ratio` | Elapsed time relative to customer median basket gap | clip(days/max(median_gap,1)/4,0,1) | [0.000351191, 1] |
| 15 | 271 | `customer_30d_activity` | Customer basket count in prior 30 days | clip(log1p(count)/log1p(1000),0,1) | [0, 0.44751] |
| 16 | 272 | `customer_90d_activity` | Customer basket count in prior 90 days | clip(log1p(count)/log1p(1000),0,1) | [0, 0.604004] |
| 17 | 273 | `category_age` | Days since customer bought candidate category | clip(days/365,0,1); 1 if category unseen | [0.00273895, 1] |
| 18 | 274 | `category_overdue_ratio` | Elapsed category time relative to customer category median gap | clip(days/max(median_gap,1)/4,0,1) | [0, 1] |
| 19 | 275 | `category_never_seen` | Candidate category absent from customer purchase history | Binary 0/1 | [0, 1] |
| 20 | 276 | `family_age` | Days since customer bought candidate product family | clip(days/365,0,1); 1 if family unseen | [0.00273895, 1] |
| 21 | 277 | `family_overdue_ratio` | Elapsed family time relative to customer family median gap | clip(days/max(median_gap,1)/4,0,1) | [0, 1] |
| 22 | 278 | `family_never_seen` | Candidate family absent from customer purchase history | Binary 0/1 | [0, 1] |
| 23 | 279 | `family_history_share` | Share of customer baskets containing candidate family | Ratio [0,1] | [0, 1] |
| 24 | 280 | `sku_orders_7d` | Global candidate sale-record count in 7-day window | clip(log1p(count)/log1p(1000),0,1) | [0, 0.42627] |
| 25 | 281 | `sku_orders_30d` | Global candidate sale-record count in 30-day window | clip(log1p(count)/log1p(1000),0,1) | [0, 0.604004] |
| 26 | 282 | `sku_orders_90d` | Global candidate sale-record count in 90-day window | clip(log1p(count)/log1p(1000),0,1) | [0, 0.739746] |
| 27 | 283 | `sku_buyers_30d` | Distinct candidate buyers in 30-day window | clip(log1p(count)/log1p(1000),0,1) | [0, 0.541016] |
| 28 | 284 | `sku_growth_30d` | Candidate sale growth: latest 30 days versus preceding 30 days | clip(log((n_latest30+1)/(n_previous30+1))/5,-1,1) | [-0.415771, 0.513184] |
| 29 | 285 | `category_relative_share` | Candidate share of global 30-day sales within its category; not customer category preference | Ratio [0,1] | [0, 1] |
| 30 | 286 | `global_sale_age` | Days since candidate last sold globally | clip(days/365,0,1); 1 if never sold | [0.00273895, 1] |
| 31 | 287 | `month_sin` | Sine encoding of prediction calendar month | sin(2*pi*(month-1)/12), [-1,1] | [-1, 0.5] |
| 32 | 288 | `month_cos` | Cosine encoding of prediction calendar month | cos(2*pi*(month-1)/12), [-1,1] | [-0, 1] |
| 33 | 289 | `sku_last_lift` | SKU transition association from customer last basket | Smoothed log-lift/5, clipped [-1,1] | [-0.804688, 1] |
| 34 | 290 | `sku_recent3_lift` | SKU transition association from customer last three baskets | Weighted average of signed lifts; weights 1,0.5,0.25 | [-0.804688, 1] |
| 35 | 291 | `category_last_lift` | Category transition association from last basket | Smoothed log-lift/5, clipped [-1,1] | [-1, 1] |
| 36 | 292 | `family_last_lift` | Family transition association from last basket | Smoothed log-lift/5, clipped [-1,1] | [-1, 1] |
| 37 | 293 | `line_last_lift` | Business-line transition association from last basket | Smoothed log-lift/5, clipped [-1,1] | [-1, 1] |
| 38 | 294 | `sku_support_confidence` | Evidence strength for SKU transition association | support/(support+20), [0,1] | [0, 0.981934] |
| 39 | 295 | `hierarchical_backoff` | SKU lift blended with family/category lifts according to support confidence | confidence*sku_lift+(1-confidence)*(family_lift+category_lift)/2, [-1,1] | [-1, 1] |
| 40 | 296 | `same_basket_lift` | Association with products co-purchased in the same basket | Smoothed log-lift/5, clipped [-1,1] | [-0.738281, 1] |

## Exact ordered six customer statistics

All six use their fitted log1p normalizer and clipping to [0,1], followed by the float16-to-float32 cache boundary. They are computed from strictly earlier customer history.

| MLP column | Name | Meaning | Observed validation range |
|---:|---|---|---|
| 297 | `distinct_orders` | Distinct prior purchase dates (orders) | [0.113953, 1] |
| 298 | `total_quantity` | Total prior purchased quantity | [0.0287628, 1] |
| 299 | `distinct_products` | Distinct previously purchased SKUs | [0.158203, 1] |
| 300 | `distinct_categories` | Distinct previously purchased categories | [0.356201, 1] |
| 301 | `median_order_gap` | Median days between distinct purchase dates | [0.110291, 1] |
| 302 | `median_purchase_quantity` | Median quantity per purchase row, not per basket | [0.0382385, 0.968262] |

## Interpretation and pending changes

- 31 contextual values and all six customer statistics use [0,1]. Ten contextual values use signed encodings: SKU growth, two month encodings, six transition/backoff scores and same-basket lift. Both 128D embedding blocks also have signed coordinates.
- Counts/shares/flags and relationship signals include overlapping information; no feature-importance claim was inferred from this baseline evaluation.
- `customer_age` means time since last purchase. `category_relative_share` is global product share within category, not a customer preference feature.
- Existing `user_cat_order_share` is a customer category share. Explicit favorite-category match, preference relative to the favorite category, and normalized customer category purchase count have not been added.
- Previously discussed removal of `is_repeat_sku` and month sine/cosine has not yet been applied.
- Global windows and relationship statistics freeze at the training cutoff for validation. Customer history advances only with earlier purchases. Catalogue metadata remains snapshot-unverified.
- The saved reference and its normalization were preserved. No Stage 2 execution or held-out test access occurred.

## Evidence

- `evaluation.json`: exact checkpoint hashes, input contract, observed input ranges, full metrics and scope.
- `predictions.npz`: per-query metrics for future paired comparisons.
- Source: `recommender/experiments/static_enrichment/model.py`, `train.py`, `config.py`; `recommender/experiments/dynamic_selector/feature_schema.py`, `feature_builder.py`; `recommender/candidates.py`, `features.py`.
