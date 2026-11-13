# Strictly-Causal Rolling Statistics (`rolling_observed_v2`), Candidate Self-Attention Prep, and Inference Replay

> **Execution Scope (`training_executed=false`, `heldout_access=false`, `promoted=false`):** No training, optimizer steps, or held-out test (`>= 2026-02-19`) access occurred. All frozen references (`repeat-no-month-v1`, `no-brand-v1`, historical Stage 1/2 aliases, `stage2_mlp_score_v1`) remain byte-identical. Old frozen weights were **not** retrained for rolling inputs.

## 1. Root Cause & Strictly-Causal `rolling_observed_v2` Fix
Why global/popularity features froze at `TRAIN_CUTOFF` (`2025-10-01`) during validation even though demand was disabled inside `ProductTower`:
1. `PointInTimeFeatureStore` ([`recommender/candidates.py`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/candidates.py)) previously indexed `_sku_sale_dates`, `_industry_sale_dates`, and `_industry_sku_sale_dates` only from `df < train_cutoff` and capped `popularity_candidates` at `min(as_of, train_cutoff)`.
2. `DynamicFeatureBuilder` ([`recommender/experiments/dynamic_selector/feature_builder.py`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/experiments/dynamic_selector/feature_builder.py)) previously filtered `self.sales` and sequential/co-occurrence relationship `observations`/`complements` at `< TRAIN_CUTOFF` and capped `global_features` / `build` relationship advancement at `min(date, TRAIN_CUTOFF)`.

**Fix (`rolling_observed_v2`) with Historical Separation & `v3` Cache Contract:**
- Added explicit `statistics_policy`: legacy `frozen_train_cutoff_v1` (historical-only replay) vs **`rolling_observed_v2`** (default for new feature preparation & request inference, schema `version=2` in [`feature_schema.py`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/experiments/dynamic_selector/feature_schema.py)).
- Under `rolling_observed_v2`, [`PointInTimeFeatureStore`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/candidates.py) still fits `_popularity_scale` strictly on `< train_cutoff`, while rolling `global_sku_popularity_norm`, `industry_sku_affinity`, and `popularity_candidates` using all strictly prior dev purchases (`< as_of_date`, never `>= as_of_date`, never held-out). [`DynamicFeatureBuilder`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/experiments/dynamic_selector/feature_builder.py) rolls 7/30/90d counts, 30d buyers, 30d growth, category share, global sale age, and relationship tables (`RelationshipTable.reset()` supports non-monotonic adapter queries), and explicitly rejects `pd.NaT` / missing dates (`pd.isna(date)`) in `__init__`, `global_features`, and `build` (as does [`RollingDynamicFeatureAdapter`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/experiments/rolling_candidate_v2/data.py)).
- Versioned `v3` cache namespace: [`data/interim/rolling_candidate_v2/v3/`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/data/interim/rolling_candidate_v2/v3/features_manifest.json) (`cache_version=3`, `namespace=rolling_candidate_v2_v3`, `train_rows=26,253`, `val_rows=6,191`, `teacher_score_included=false`, `heldout_access=false`, `observed_source_digest=e1ffa78c...`), leaving the prior [`data/interim/rolling_candidate_v2/`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/data/interim/rolling_candidate_v2/features_manifest.json) (`v2`) directory intact. `compute_rolling_source_digest` computes an order-invariant SHA-256 over date-gated (`< TEST_START`) dev purchases and used catalogue/company columns so any source edit invalidates the `v3` cache without reading held-out records.
- Recomputed rolling selector scores (`train_selector_scores.pt`, `val_selector_scores.pt`, [`selector_scores_manifest.json`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/data/interim/rolling_candidate_v2/v3/selector_scores_manifest.json)) drive `quota100_50` Stage 2 slate construction (`candidate_source=rolling_selector_scores`, [`slates_manifest.json`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/data/interim/rolling_candidate_v2/v3/slates_manifest.json)):
  - Train split (`< TRAIN_CUTOFF`): `max_abs_diff = 5.25e-6` vs frozen `train_scores.pt` (exact parity).
  - Validation split (`>= TRAIN_CUTOFF`): `max_abs_diff = 4.204` vs frozen `val_scores.pt`; `6,094 / 6,191` validation rows change their `quota100_50` candidate set (`mean_positive_per_slate` improves `1.5637 -> 1.5673`).
- Live adapter vs sealed `v3` cache parity on 16 spread validation rows in non-monotonic order (`RollingDynamicFeatureAdapter`): `max_abs_diff_vs_feature_cache = 0.0`, `max_abs_diff_vs_slate_cache = 0.0`.
- Notice `train_features.npy` (`< TRAIN_CUTOFF`) preserves the exact historical SHA-256 (`9e6aec1d...`), whereas `val_features.npy` changes (`7edef269... -> bd7b098c...`) because validation queries now observe strictly prior validation purchases.

## 2. Validation-Only Frozen `301D` Selector Rolling Replay (No Training)
> **Scope Disclaimer:** Inference-only replay (`training_executed=false`, `heldout_access=false`, `promoted=false`) using identical frozen `repeat-no-month-v1` `301D` selector weights on `N = 6,191` validation queries (`10,276` total targets, `1,857` new targets), comparing saved `frozen_train_cutoff_v1` scores against recomputed `rolling_observed_v2` scores. This isolates the bug-correction accuracy effect from changed validation inputs (`u`, `v`, and `customer_stats_6` remain from the frozen snapshot) and is **not** architecture or training-quality evidence. See [`rolling_selector_replay.json`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/experiments/rolling_candidate_v2/rolling_selector_replay.json).

| Validation Metric | Saved Frozen (`frozen_train_cutoff_v1`) | Recomputed Rolling (`rolling_observed_v2`) | Delta (`Rolling - Frozen`) | Customer-Clustered 95% Bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Overall `Recall@5` | 0.64335 | 0.64696 | +0.00361 | `[-0.00006, +0.00737]` |
| Overall `Recall@10` | 0.75411 | 0.75845 | +0.00433 | `[+0.00177, +0.00703]` |
| Overall `Recall@25` | 0.84850 | 0.85096 | +0.00246 | `[+0.00003, +0.00491]` |
| Overall `Recall@50` | 0.89542 | 0.89877 | +0.00335 | `[+0.00111, +0.00557]` |
| Overall `Recall@100` | 0.93733 | 0.94030 | +0.00297 | `[+0.00069, +0.00524]` |
| `NDCG@10` | 0.58633 | 0.58822 | +0.00189 | `[+0.00029, +0.00367]` |
| `MRR` | 0.56615 | 0.56708 | +0.00094 | — |
| Discovery@10 (`new_hits` / `new_recall@10`) | 194 / 0.10447 | 206 / 0.11093 | +12 / +0.00646 | `[+0.00000, +0.01296]` |
| Discovery@25 (`new_hits` / `new_recall@25`) | 726 / 0.39095 | 757 / 0.40765 | +31 / +0.01669 | `[+0.00646, +0.02709]` |
| Discovery@50 (`new_hits` / `new_recall@50`) | 1029 / 0.55412 | 1041 / 0.56058 | +12 / +0.00646 | `[-0.00484, +0.01729]` |
| Discovery@100 (`new_hits` / `new_recall@100`) | 1337 / 0.71998 | 1357 / 0.73075 | +20 / +0.01077 | `[+0.00155, +0.02026]` |

## 3. Candidate-to-Candidate Self-Attention Ranker (No Teacher Score)
- Module: [`RollingCandidateAttentionRanker`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/experiments/rolling_candidate_v2/model.py) over frozen `repeat-no-month-v1` `quota100_50` candidate slates (`candidate_source=rolling_selector_scores`) and `rolling_observed_v2` 41D features (`teacher_score_included=False`, `mlp_score` explicitly rejected).
- Architecture: unchanged 13D/4x14x17 `UPSCANRRanker` base cross-attention + preserved 425D `base_head` + opt-in zero-initialized residual unordered masked `TransformerEncoderLayer(128, 4, 256, norm_first=True)` without positional encodings.
- Zero-residual init parity on 32 cached validation slates vs 425D `CandidateContextRanker`: `max_abs_logit_diff = 0.0` (control, treatment, and `disable_candidate_attention=True`), `max_abs_attn_diff = 0.0`, reversed-slate diff `0.0`, NaN-padded invalid-candidate diff `0.0`, masked logits `-inf`.

## 4. CPU & MPS Vector Recomputation and Serving Latency Benchmark
- **Hardware**: Apple M5 (`arm64`, 10 CPU cores, `torch_num_threads=4`), macOS 27.2.1, PyTorch `2.14.1`, `mps_available=true`.
- **Non-timed parity vs frozen `val_purchase_cache.pt`**: product vectors max diff `2.09e-7`, val customer vectors `2.68e-7`.
- **Scope & Exclusions**: Steady-state timings measure pre-materialized device inputs after `warmup=3`, `repeats=15` with `sync()`. Cold disk load (`1,867.8 ms`) and host-to-device transfer (`13.68 ms` CPU / `44.33 ms` MPS) are separate. **`shape_only_probe_with_tower_stats_18d_6d` uses untrained weights strictly to probe forward-pass latency if 6D product demand and 6D customer tabular stats were restored; it carries zero quality or training claim.**
- **Unique customer vs transition scope**: Latest-snapshot deduping across dev yields `2,571` unique customers (`2,319` train, `1,676` val), distinct from scoring all `6,191` validation transitions.

| Workload / Component | Items | CPU Median (ms) | CPU P90 (ms) | MPS Median (ms) | MPS P90 (ms) | Scope / Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Product tower (`12D` active) | 841 | 0.566 | 0.570 | 0.519 | 0.542 | Full catalogue |
| Product tower (`12D` active) | 1 | 0.035 | 0.036 | 0.451 | 0.503 | Single-SKU refresh |
| Customer tower (`0D` + `sales_branch`, bs=256) | 2,571 | 130.227 | 134.866 | 46.915 | 47.242 | Unique dev customers |
| Customer tower (`0D` + `sales_branch`, bs=256) | 6,191 | 331.151 | 336.240 | 111.642 | 111.794 | All val transitions |
| Customer tower single refresh (`bs=1`) | 1 | 0.399 | 0.405 | 0.872 | 0.919 | Request-time `u_c` |
| Full dev recompute (`841p + 2,571c`, encode only) | 3,412 | 130.634 | 132.514 | 47.384 | 47.419 | Product then unique dev customers |
| Customer recompute + `841`-SKU dot/fam scoring (cached product vectors) | 2,571 | 136.426 | 136.632 | 51.064 | 51.100 | Cached product vectors; customer encoding and scoring |
| Shape-only probe (`18D` prod + `6D` cust, **untrained**) | 3,412 | 130.456 | 130.539 | 47.412 | 47.507 | Difference within measurement variation; no quality evidence |
| `RollingDynamicFeatureAdapter` cold date (CPU) | 841 | 8.534 | 8.652 | — | — | Cold init `4,794.2 ms`; global `5.99 ms` |
| `RollingDynamicFeatureAdapter` warm date (CPU) | 841 | 2.321 | 2.342 | — | — | Same-date global cache hit |
| Active `301D` MLP selector (`bs=1` / `bs=256` x 841) | 1 / 256 | 0.293 / 75.540 | 0.296 / 75.910 | 0.341 / 29.769 | 0.358 / 30.244 | Full 841-SKU scoring |
| Stage 2 Control `425D` (`bs=1` / `bs=64` x 100) | 1 / 64 | 0.660 / 10.123 | 0.697 / 10.149 | 1.637 / 5.037 | 2.029 / 5.411 | Pointwise baseline |
| Stage 2 Cand-Attention `425D` (`bs=1` / `bs=64` x 100) | 1 / 64 | 0.942 / 18.189 | 0.960 / 18.236 | 2.051 / 7.961 | 2.415 / 8.323 | Masked unordered set attention |

## 5. Limitations, Semantics & Next Runnable Entrypoints
- **Freshness & Upstream Snapshot Caveat**: `observed_source_digest` invalidates stale `v3` rolling caches when dev purchases or catalogue/company inputs change, so stale rolling features are never silently reused. However, `prepare_retrieval_data()` and frozen Stage 1 `u`/`v`/`customer_stats_6`/targets come from historical sealed caches (`e1_e5_retrieval_v1`, `embedding-ablation-v1`). Rebuilding rolling features into a new namespace after purchase edits while reusing historical `u`/`stats6`/targets would be inconsistent; actual source changes require a coherent upstream snapshot rebuild. Furthermore, `RollingDynamicFeatureAdapter` is development-only (`< TEST_START`), cannot prove the timestamp/freshness of externally supplied `u`/`v`/`customer_stats_6`, and production purchase ingestion/refresh is not deployed.
- **`user_cat_order_share` Semantics**: [`PointInTimeFeatureStore`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/recommender/candidates.py) computes `user_cat_order_share` (and `user_subcat_order_share`, `user_brand_order_share`) as purchase rows/events per attribute divided by total purchase rows/events (`snapshot.total_orders`), not distinct basket count; no separate basket-level category count feature exists.
- **`snapshot_unverified` Metadata**: Product/company attributes remain non-effective-dated; only purchase/market/relationship statistics roll with `< as_of_date`.
- **No Stage 2 Training Yet**: Training (`train.py --execute-training`) was **not** executed per instruction; quality impact of candidate self-attention or retrained rolling selectors remains unmeasured until explicitly authorized.
- **No-training prep/parity**: `./venv/bin/python -m recommender.experiments.rolling_candidate_v2.data --prepare-only --prepare-slates --verify-parity-rows 16`
- **Inference benchmark**: `./venv/bin/python -m recommender.experiments.rolling_candidate_v2.benchmark`
- **Preserved Hashes**: All 17 frozen reference/checkpoint/legacy slate SHA-256 digests verified identical before and after execution (see [`summary.json`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/experiments/rolling_candidate_v2/summary.json) and [`rolling_selector_replay.json`](file:///Users/kalyma123/programming/enterprise-product-recommendation-system/experiments/rolling_candidate_v2/rolling_selector_replay.json)).
