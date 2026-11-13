# [SUPERSEDED / HISTORICAL] Stage 1 `CompactStage1Selector` MLP Reference (`no-brand-v1`)

> **Superseded Notice (2026-11-13):** This `no-brand-v1` (`303D`, `uv_47`) reference has been **superseded as the active Stage 1 MLP reference** by **`repeat-no-month-v1`** (`301D`, `uv_45`, `FeatureAblatedStage1Selector`) via explicit user override.
> - **New Canonical Active Pointer:** [`experiments/stage1_mlp_reference/reference.json`](../stage1_mlp_reference/reference.json) ([`experiments/stage1_mlp_reference/REFERENCE.md`](../stage1_mlp_reference/REFERENCE.md)) -> [`checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json`](../../checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json).
> - **Frozen Historical Pointer Preserved:** [`experiments/embedding_ablation_v1/reference.json`](reference.json) (SHA-256 `148da7db8f7b50090e166b8863329b0b7830d6f1c289143c578ab194d0a5211f`) and all `checkpoints/stage1-mlp-reference/no-brand-v1/` files remain **byte-for-byte unchanged** because prior campaign reproduction modules (`mlp_feature_ablation`, `description_cleaning`, `mlp_repeat_restored`) pin and hash-verify them directly. All historical values below are preserved untouched.

- **Historical Reference ID:** `no-brand-v1` (variant `E_no_brand` + `CompactStage1Selector uv_47`, `seed=42`)
- **Pinned Historical Pointer (Frozen):** [`experiments/embedding_ablation_v1/reference.json`](reference.json)
- **Immutable Historical Directory:** `checkpoints/stage1-mlp-reference/no-brand-v1/`
  - **Tower (`best_retriever.pt`, epoch 16/30):** `checkpoints/stage1-mlp-reference/no-brand-v1/best_retriever.pt`
    - SHA-256: `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981` (`3,432,974` bytes)
    - Source: `checkpoints/embedding-ablation-v1/runs/stage1-E_no_brand-seed42/best.pt` (byte-identical copy)
  - **Selector (`best_selector.pt`, epoch 1/10):** `checkpoints/stage1-mlp-reference/no-brand-v1/best_selector.pt`
    - SHA-256: `d083a5562d968e8e606f69d86e142e370567fa4f506f650e9e2234c53d66dad5` (`198,661` bytes)
    - Source: `checkpoints/embedding-ablation-v1/runs/selector-E_no_brand-uv_47-seed42/best.pt` (byte-identical copy)
    - Recorded `upstream_digest`: `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981` (matches tower SHA-256)
  - **Manifest:** `checkpoints/stage1-mlp-reference/no-brand-v1/manifest.json`
    - SHA-256: `4e223d3de158a79a1dfd08e65b372d699652239edb0d9462250a987e92e0be86`

## 1. Selection Authority & Preservation of Campaign Evidence

- **Explicit User Selection (2026-11-13):** The user explicitly selected *"remove brand and save this as the reference"* based on the completed `embedding_ablation_v1` evidence ([`REPORT.md`](REPORT.md), [`REVIEW.md`](REVIEW.md)).
- **Scope of "Remove Brand":** This removes **only the separate learned `brand` categorical embedding (`16D`) inside `StaticEnrichedProductTower`**, using the completed `E_no_brand` (`seed=42`) run. `brand` remains retained in:
  1. `t1_fam` product text embeddings (`BAAI/bge-m3`, `1024D`),
  2. Candidate pairwise/scalar features (`user_brand_order_share`, `is_repeat_brand`), and
  3. Stage 1 familiarity weights (`brand`, `category`, `business_line`, `family`).
- **Historical Campaign Record Preserved:** The original ablation campaign outputs ([`results.json`](results.json), [`summary.json`](summary.json), [`REPORT.md`](REPORT.md), [`REVIEW.md`](REVIEW.md)) are preserved unchanged, including `promoted: false` in `summary.json`. Saving `no-brand-v1` reflects explicit user authorization despite single-seed (`seed=42`) exploratory uncertainty.
- **Narrow Reference Scope:** This pair is the current default reference for **future Stage 1 `CompactStage1Selector` MLP work only**. It is **not** an untested Stage 2 (`UPS-CANR`) or production retriever default. Historical Stage 1 discovery-reference (`checkpoints/retriever-discovery-reference/best_retriever.pt`), Stage 2 aliases (`checkpoints/ranker/best_ranker.pt`, `best_discovery_ranker.pt`, `best_ndcg_ranker.pt`), and all seven `audit/artifacts.json` entries remain untouched.

## 2. Architecture, Input Contract & Normalization Status

- **Stage 1 Static Towers (`StaticEnrichedRetriever`, `851,213` params):**
  - `CustomerTower`: `t1_fam`/company `1024D` text projection, `city_id`/`region_id`/`industry_id` (`16D` each), `sales_branch` customer embedding (`16D`, vocab 7), `customer_tabular_features = 0` (all changing customer stats omitted from static tower), 32-event Transformer history over catalogue vectors (`history_gap_reference_days = 74.0`).
  - `ProductTower`: `t1_fam` product text (`1024D`, includes `brand` and `product_family`, omits `cleaned_description`), **6** learned `16D` attribute embeddings (`category`, `subcategory`, `business_line`, `tier`, `unit`, `country_of_origin`; `brand` absent), static `12D` `container_type` one-hot (`demand_policy = disabled_v1`, zeroed/omitted demand scalars).
- **Stage 1 Selector (`CompactStage1Selector`, `uv_47`, `47,617` params, width `303` -> `128 -> 64 -> 1`):**
  - Ordered blocks: `u (128)` + `v (128)` + `features_41 (41)` + `customer_stats_6 (6)`.
  - **Normalization Status (Unchanged Legacy `uv_47`):** Inputs are **NOT yet all `[0, 1]`**:
    - `u`, `v`: L2-normalized vectors in `[-1, 1]`.
    - `customer_stats_6`: train-fitted `Log1pScaler` maxima in `[0, 1]` (`float16 -> float32`).
    - `features_41`: legacy `dynamic_campaign_v1` scaling (`float16` on disk -> `float32` compute), where 31 features are in `[0, 1]` and 10 signed features (`sku_growth_30d`, `month_sin`, `month_cos`, `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`, `line_last_lift`, `hierarchical_backoff`, `same_basket_lift`) are in `[-1, 1]`.
  - **Pending Work:** Numeric normalization alignment (e.g., rescaling `[-1, 1]` features/embeddings if tested) and category-feature improvements remain pending future experiments against this `no-brand-v1` baseline.

## 3. Validation-Only Metrics (`2025-10-01`..`2026-02-18`, `6,191` Rows, `1,857` New Targets, `seed=42`)

No held-out test data (`>= 2026-02-19`) or Stage 2 ranker training/evaluation was executed.

| Metric (Full-Pool `uv_47` Selector) | Previous Baseline (`U0_t1_fam_clean`, Ep 26 / MLP Ep 1) | Active Reference (`no-brand-v1` / `E_no_brand`, Ep 16 / MLP Ep 1) | Delta |
|---|---:|---:|---:|
| Unseen `hits@25` (`/ 1,857`) | `737` (`39.69%`) | **`741`** (`39.90%`) | `+4` |
| Unseen `hits@50` (`/ 1,857`) | `1,012` (`54.50%`) | **`1,037`** (`55.84%`) | `+25` |
| Unseen `hits@100` (`/ 1,857`) | `1,320` (`71.08%`) | **`1,324`** (`71.30%`) | `+4` |
| Unseen `hits@10` (`/ 1,857`) | `196` (`10.55%`) | **`214`** (`11.52%`) | `+18` |
| Overall `Recall@10` (`6,191` rows) | `0.7505190641169014` | **`0.7565593823119603`** | `+0.00604` |
| Overall `NDCG@10` (`6,191` rows) | `0.5771271981708025` | **`0.5782861484383223`** | `+0.00116` |
