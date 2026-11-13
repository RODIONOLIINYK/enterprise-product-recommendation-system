# Stage 1 `FeatureAblatedStage1Selector` MLP Active Reference (`repeat-no-month-v1`)

- **Reference ID:** `repeat-no-month-v1` (upstream `E_no_brand` towers + `FeatureAblatedStage1Selector repeat_no_month` / `uv_45`, `301D`, `seed=42`)
- **Canonical Active Pointer:** [`experiments/stage1_mlp_reference/reference.json`](reference.json)
- **Immutable Reference Directory:** `checkpoints/stage1-mlp-reference/repeat-no-month-v1/`
  - **Tower (`best_retriever.pt`, epoch 16/30):** `checkpoints/stage1-mlp-reference/repeat-no-month-v1/best_retriever.pt`
    - SHA-256: `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981` (`3,432,974` bytes)
    - Source: `checkpoints/embedding-ablation-v1/runs/stage1-E_no_brand-seed42/best.pt` (byte-identical copy of `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt`)
  - **Selector (`best_selector.pt`, epoch 5/10):** `checkpoints/stage1-mlp-reference/repeat-no-month-v1/best_selector.pt`
    - SHA-256: `977241cf9fe8a36ae5c0b5afede6c8018e55eb3202e39d88fa7e3693bc0bc22a` (`198,789` bytes)
    - Source: `checkpoints/mlp-repeat-restored-v1/runs/selector-E_no_brand-repeat_no_month-uv_45-seed42/best.pt` (byte-identical to `checkpoints/mlp-repeat-restored-v1/candidate/best_selector.pt`)
    - Recorded `upstream_digest`: `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981` (matches copied tower SHA-256)
  - **Manifest:** `checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json`
    - SHA-256: `7f8d8d1e01fc210b0247851cea0364dc95dc45556d0dadea57d5ceec0389ab53`

## 1. Selection Authority, Historical Preservation & Scope

- **Explicit User Override (2026-11-13):** The user explicitly overrode the earlier non-promotion recommendation in [`../mlp_repeat_restored_v1/PROMOTION_DECISION.md`](../mlp_repeat_restored_v1/PROMOTION_DECISION.md) with: *"okay, save the Repeat restored, 301D as the new reference, commit and push"*.
- **Historical Campaign Evidence Preserved:** Original campaign outputs ([`../mlp_repeat_restored_v1/summary.json`](../mlp_repeat_restored_v1/summary.json), `checkpoints/mlp-repeat-restored-v1/candidate/manifest.json`, [`../embedding_ablation_v1/summary.json`](../embedding_ablation_v1/summary.json)) retain `promoted: false` as historical campaign evidence.
- **Frozen Prior Reference Pointer:** [`../embedding_ablation_v1/reference.json`](../embedding_ablation_v1/reference.json) (`no-brand-v1`, SHA-256 `148da7db8f7b50090e166b8863329b0b7830d6f1c289143c578ab194d0a5211f`) remains **byte-for-byte unchanged** because prior campaign reproduction modules (`recommender.experiments.mlp_feature_ablation`, `recommender.experiments.description_cleaning`) pin and hash-verify it directly.
- **Narrow Reference Scope:** Use this pair as the active default reference for **Stage 1 MLP selector work only**. It is **not** a Stage 2 (`UPS-CANR`) or production retriever default. Historical Stage 1 discovery-reference (`checkpoints/retriever-discovery-reference/best_retriever.pt`), Stage 2 aliases (`checkpoints/ranker/best_ranker.pt`, `best_discovery_ranker.pt`, `best_ndcg_ranker.pt`), and all seven `audit/artifacts.json` entries remain untouched.

## 2. Model Resolution, Architecture & 301D Input Contract

- **Strict Selector Loader:**
  - Class: `FeatureAblatedStage1Selector` in `recommender.experiments.mlp_feature_ablation.model` (`47,361` params, `301 -> 128 -> 64 -> 1`, `LayerNorm` + `GELU` + `Dropout(0.1)`).
  - Loader: `load_ablated_selector_checkpoint` in `recommender.experiments.mlp_feature_ablation.model`.
  - Expected spec: `build_feature_ablation_spec(REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES, name=REPEAT_RESTORED_VARIANT_NAME, recipe=REPEAT_RESTORED_RECIPE_NAME, expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES)` from `recommender.experiments.mlp_feature_ablation.{config,schema}`.
  - Contract digest: `8761650f542b7bdd7605065e686b3e61048c3656a313d3223c8ba4a0027535c2`.
- **Frozen Stage 1 Static Towers (`StaticEnrichedRetriever`, `E_no_brand`, `851,213` params):**
  - Class: `StaticEnrichedRetriever` in `recommender.experiments.static_enrichment.model` (`build_ablation_specs()["E_no_brand"].variant_spec` in `recommender.experiments.embedding_ablation.config`).
  - `CustomerTower`: `1024D` `BAAI/bge-m3` company text projection, `city_id`/`region_id`/`industry_id` (`16D` each), `sales_branch` customer embedding (`16D`, vocab 7), `customer_tabular_features = 0`, 32-event Transformer history over catalogue vectors (`history_gap_reference_days = 74.0`).
  - `ProductTower`: `t1_fam` product text (`1024D`, includes `brand` and `product_family`, omits `cleaned_description`), **6** learned `16D` attribute embeddings (`category`, `subcategory`, `business_line`, `tier`, `unit`, `country_of_origin`; separate `brand` embedding **absent**), static `12D` `container_type` one-hot (`demand_policy = disabled_v1`, zeroed/omitted demand scalars).
  - `brand` remains retained in `t1_fam` text, scalar features (`user_brand_order_share`, `is_repeat_brand`), and Stage 1 familiarity.
- **Ordered 301D Selector Input Contract (`uv_45`):**
  1. `u` (`128D`, cols `0..127`): L2-normalized customer vector in `[-1, 1]`.
  2. `v` (`128D`, cols `128..255`): L2-normalized product vector in `[-1, 1]`.
  3. `features_39` (`39D`, cols `256..294`): `is_repeat_sku` **PRESENT** at source index `0` / MLP col `256`; **ONLY** `month_sin` (source index `31`) and `month_cos` (source index `32`) are omitted (`39` context features retained).
  4. `customer_stats_6` (`6D`, cols `295..300`): `distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`.
  - Total scalar features: `39 + 6 = 45`; total input width: `128 + 128 + 45 = 301`.
- **Legacy Normalization Status & Pending Work:**
  - Inputs remain under legacy normalization (**NOT yet all `[0, 1]`**): `u`/`v` are unit-norm `[-1, 1]`; `customer_stats_6` are in `[0, 1]` (`float16 -> float32`); of `features_39`, `31` features are in `[0, 1]` and `8` signed features (`sku_growth_30d`, `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`, `line_last_lift`, `hierarchical_backoff`, `same_basket_lift`) remain in `[-1, 1]`.
  - Numeric normalization alignment and category-feature work remain pending.

## 3. Validation-Only Metrics (`2025-10-01`..`2026-02-18`, `6,191` Queries, `10,276` Total Targets, `1,857` New Targets)

No held-out test data (`>= 2026-02-19`) or Stage 2 ranker training/evaluation was executed (`validation_only = true`, `heldout_access = false`, `stage2_executed = false`, `metadata_policy = snapshot_unverified`).

| Metric (Full-Pool Selector) | Prior Reference `no-brand-v1` (`303D`, Ep 1) | Prior Ablation `300D` (`uv_44`, Ep 5) | Active Reference `repeat-no-month-v1` (`301D`, Ep 5) | Delta vs `no-brand-v1` |
|---|---:|---:|---:|---:|
| Overall Macro `Recall@5` | `0.640251` (`64.025%`) | `0.638847` (`63.885%`) | **`0.643349` (`64.335%`)** | `+0.310 pp` |
| Overall Macro `Recall@10` | `0.756559` (`75.656%`) | `0.756250` (`75.625%`) | **`0.754114` (`75.411%`)** | `-0.245 pp` |
| Overall Macro `Recall@25` | `0.851952` (`85.195%`) | `0.849901` (`84.990%`) | **`0.848498` (`84.850%`)** | `-0.345 pp` |
| Overall Macro `Recall@50` | `0.896959` (`89.696%`) | `0.896294` (`89.629%`) | **`0.895424` (`89.542%`)** | `-0.154 pp` |
| Overall Macro `Recall@100` | `0.936849` (`93.685%`) | `0.937029` (`93.703%`) | **`0.937329` (`93.733%`)** | `+0.048 pp` |
| Overall `NDCG@10` | `0.578286` | `0.583116` | **`0.586331`** | `+0.00804` |
| Overall `MRR` | `0.553445` | `0.561641` | **`0.566148`** | `+0.01270` |
| Ordinary Top-10 New Hits (`/ 1,857`) | `214` (`11.524%`) | `205` (`11.039%`) | **`194` (`10.447%`)** | `-20` (`-1.077 pp`) |
| Unseen-Only `hits@25` (`/ 1,857`) | `741` (`39.903%`) | `729` (`39.257%`) | **`726` (`39.095%`)** | `-15` (`-0.808 pp`) |
| Unseen-Only `hits@50` (`/ 1,857`) | `1,037` (`55.843%`) | `1,038` (`55.897%`) | **`1,029` (`55.412%`)** | `-8` (`-0.431 pp`) |
| Unseen-Only `hits@100` (`/ 1,857`) | `1,324` (`71.298%`) | `1,335` (`71.890%`) | **`1,337` (`71.998%`)** | `+13` (`+0.700 pp`) |

### Tradeoffs & Methodological Caveats

- **Accepted Tradeoffs vs `no-brand-v1`:** Higher `NDCG@10` (`0.586331` vs `0.578286`), higher `Recall@5` (`64.335%` vs `64.025%`), and higher unseen `@100` (`1,337` vs `1,324`), paired with lower overall `Recall@10` (`75.411%` vs `75.656%`), lower ordinary top-10 discovery (`194` vs `214`), and lower unseen `@25`/`@50` (`726`/`1,029` vs `741`/`1,037`).
- **Single-Seed & Epoch-Selection Limits:** Results are from a single training seed (`seed=42`) at selected epochs (tower epoch `16`, selector epoch `5` vs `no-brand-v1` selector epoch `1`). Paired customer-clustered bootstrap intervals in [`../mlp_repeat_restored_v1/summary.json`](../mlp_repeat_restored_v1/summary.json) quantify query/customer sampling variability conditional on those selected weights only; **do not claim statistical or multi-seed superiority**.
- **Pending Work:** Legacy numeric normalization (`[-1, 1]` lift/growth features and unit-norm `u`/`v`) and category-feature improvements remain open for future experiments against this baseline.
