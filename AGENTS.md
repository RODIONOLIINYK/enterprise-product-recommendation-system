# Agent Guide & Reference Architecture (`AGENTS.md`)

This document is the authoritative source of truth for coding agents working in this repository. It describes **only the active, verified code and reference checkpoints** currently in the repository. See [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) for validation metrics and [`plan.md`](plan.md) for current project goals and next experiments.

---

## 1. Environment & Non-Negotiable Rules

- **Python & Test Runner:**
  - Always use `./venv/bin/python` and `PYTHONPATH=. ./venv/bin/pytest tests/`.
  - Hardware acceleration uses Apple Silicon `mps` when available (`recommender/train_retriever.py:get_device`).
- **Strict Temporal Split & Zero Test Leakage (`recommender/split.py`):**
  - **Train:** `< 2025-10-01` (`50,414` events, `26,264` transitions; `26,184` transitions have $\ge 50$ eligible rolling SKUs).
  - **Validation:** `2025-10-01` to `2026-02-18` (`10,634` events, `6,191` transitions, `1,857` new-to-customer target instances).
  - **Test:** `>= 2026-02-19` (`10,868` events, `6,327` transitions).
  - **NEVER** train on, tune on, inspect, or evaluate on the held-out Test split (`>= 2026-02-19`) unless the user explicitly instructs you to run final test evaluation.

---

## 2. Active Reference Architecture

### Stage 1: Discovery-Reference Two-Tower Retriever
- **Code:** [`recommender/models.py`](recommender/models.py), [`recommender/train_retriever.py`](recommender/train_retriever.py), [`recommender/familiarity.py`](recommender/familiarity.py), [`recommender/stage1_catalogue.py`](recommender/stage1_catalogue.py), [`recommender/evaluate_stage1.py`](recommender/evaluate_stage1.py)
- **Reference Checkpoint:** `checkpoints/retriever-discovery-reference/best_retriever.pt` (Epoch 23 of 30, `seed=42`, trained with `--discovery-reference`).
- **Key Architectural Settings:**
  - **Output dimension:** `128D` L2-normalized customer vectors $\mathbf{u}_c$ and product vectors $\mathbf{v}_p$.
  - **`CustomerTower`:**
    - Projects `1024D` `BAAI/bge-m3` company text $\to 128\text{D}$.
    - Embeds `city_id`, `region_id`, `industry_id` (`16D` each) + **7 point-in-time profile features** (`total_purchases`, `total_quantity`, `distinct_products`, `distinct_categories`, `days_since_first_order`, `days_since_last_order`, `median_purchase_quantity`) $\to 128\text{D}$.
    - Encodes the last 32 purchase events using `catalog_history_encoder = "attention"`: each historical token combines its frozen `128D` catalogue product vector, normalized event quantity (`quantity_reference = 12.0`), and normalized inter-purchase gap (`history_gap_reference_days = 74.0`) through a 2-layer Transformer (`4` heads) and learned attention pooling $\to 128\text{D}$.
    - Late-fusion MLP combines `[text (128), tabular (128), history (128)]` $\to 128\text{D}$.
  - **`ProductTower`:**
    - Uses `remove_product_attribute_embeddings = True` (no learned brand/category/subcategory/line ID embeddings inside `ProductTower`).
    - Combines `1024D` `BAAI/bge-m3` product text (without subcategory), one-hot `container_type` with an explicit `none` bucket (`14` buckets), and **7 point-in-time product demand features** (`total_volume`, `orders_30d`, `orders_90d`, `unique_buyers`, `category_market_share`, `days_since_last_sold`, `sku_median_quantity`) $\to 128\text{D}$.
  - **4 Familiarity Flags (`recommender/familiarity.py`):**
    - Adds learned weights for whether the customer has ever purchased the candidate's `brand`, `product_category`, `business_line`, or `product_family` strictly prior to `as_of_date`.
  - **Rolling Candidate Pool (`recommender/stage1_catalogue.py`):**
    - Candidates on `as_of_date` must be for-sale merchandise (`ТОВАР`), have at least one genuine paid sale (`ПРОДАЖА`) strictly before `as_of_date`, and not be blocked prior to `as_of_date` (`735–771` active SKUs per validation date out of `1,722` sellable catalogue items).
  - **Validation Discovery Performance:**
    - Retrieves **`652 / 1,857` (`35.11%`)** new-to-customer targets in Top-25 unseen and **`921 / 1,857` (`49.60%`)** in Top-50 unseen.

### Candidate Slate Construction & 13 Pairwise Features (`recommender/candidates.py`)
- For Stage 2 training and evaluation, `CandidateRetriever.generate_candidate_batch` builds a **50-candidate mixed slate** (`heuristic_fraction = 0.5`):
  - Up to **25 slots** from the customer's previously purchased SKUs ranked by personal repeat frequency and recency.
  - Remaining **25+ slots** filled by top unseen Stage 1 Discovery candidates from the rolling pool.
  - On the 6,191 validation baskets, this 50-item mixed slate covers **`87.48%` of all targets** (`97.96%` of repeat targets and `41.09%` = `763 / 1,857` of new-to-customer targets).
- Each candidate $p$ in the 50-item slate carries **13 point-in-time pairwise features** (`NUM_PAIRWISE_FEATURES = 13`):
  1. `is_repeat_sku` (index `0`)
  2. `user_sku_order_count_norm`
  3. `user_sku_qty_share`
  4. `days_since_last_sku_buy_norm`
  5. `sku_urgency`
  6. `user_cat_order_share`
  7. `user_subcat_order_share`
  8. `user_brand_order_share`
  9. `user_business_line_order_share`
  10. `global_sku_popularity_norm`
  11. `industry_sku_affinity`
  12. `stage1_score` (sigmoid retrieval logit)
  13. `stage1_rank_inv` ($1 / (1 + \text{rank})$)

### Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker (`UPS-CANR`)
- **Code:** [`recommender/ranker.py`](recommender/ranker.py), [`recommender/train_ranker.py`](recommender/train_ranker.py), [`recommender/evaluate.py`](recommender/evaluate.py)
- **Reference Checkpoint:** `checkpoints/ranker/best_ranker.pt` (promoted from the best ablation `excluded_statistics = ("history_repeat",)` with `statistics_attention = True`).
- **Architecture (`UPSCANRRanker`):**
  1. **Unified 33-Token Customer Memory (`B, 33, 128`):**
     - **Token 0 (Customer Profile):** Combines `1024D` company text projection, `city`/`region`/`industry` embeddings, and 8 profile features (zeroed in the token value builder when `statistics_attention=True`), plus `type_embedding[0]`.
     - **Tokens 1..32 (Purchase History):** Combines `product_embedding` (`64D`), `cat_embedding` (`32D`), `subcat_embedding` (`16D`), `line_embedding` (`16D`), `habit_proj` (`32D`, zeroed in the token value builder when `statistics_attention=True`), and `product_text_proj` (`32D`) $\to 128\text{D}$, plus `pos_embedding` and `type_embedding[1]`.
     - **Self-Attention:** 1-layer `TransformerEncoder` over the 33 tokens (Token 0 is never masked).
  2. **Statistics-Biased 4-Head Cross-Attention:**
     - Candidate queries $Q = \mathbf{z}_{\text{cand}} \in \mathbb{R}^{B \times 50 \times 128}$ attend to the 33 memory tokens.
     - With `statistics_attention = True` and `excluded_statistics = ("history_repeat",)`, each head adds a learned bilinear attention bias $B_h(c, m) = [1, \mathbf{s}_c]^\top W_h \mathbf{s}_m$ (`W_h` shape `4 x 14 x 17`) between the 13 candidate pairwise features (+ intercept) and the 9 history statistics / 8 company profile statistics (with `history_repeat` masked to `0`).
  3. **Scoring Head (`397D` MLP):**
     - Concatenates `[\mathbf{z}_{\text{cand}} (128), \mathbf{h}_{\text{attended}} (128), \mathbf{z}_{\text{cand}} \odot \mathbf{h}_{\text{attended}} (128), \text{pairwise\_features} (13)]` (`397D`) $\to$ MLP (`397 -> 256 -> 128 -> 64 -> 1`).
  4. **Validation Performance (`6,191` baskets):**
     - **New-to-Customer Recall@10:** **`11.25%` (`209 / 1,857`)**
     - **Overall Recall@10:** **`76.15%`**
     - **Overall NDCG@10:** **`0.5924`** (discovery-selected checkpoint Epoch 10; peak NDCG checkpoint achieves `0.5945`).

---

## 3. Clean Module Inventory (`recommender/`)

| Module | Purpose |
| :--- | :--- |
| [`recommender/data.py`](recommender/data.py), [`merging.py`](recommender/merging.py), [`scope.py`](recommender/scope.py), [`preparation.py`](recommender/preparation.py) | Raw workbook/CSV parsing, paid-sale filtering (`ПРОДАЖА` + `ТОВАР`), quantity-summing record deduplication, and `cleaned_purchases.csv` generation. |
| [`recommender/company.py`](recommender/company.py), [`catalogue.py`](recommender/catalogue.py), [`metadata.py`](recommender/metadata.py) | Company metadata (`city`, `region_code`, `industry_branch`) and product catalogue metadata (`brand`, `product_category`, `product_subcategory`, `business_line`, `container_type`, `product_family`). |
| [`recommender/split.py`](recommender/split.py) | Chronological split (`TRAIN_CUTOFF = 2025-10-01`, `TEST_START = 2026-02-19`) and leakage validation. |
| [`recommender/features.py`](recommender/features.py) | `Log1pScaler`, `PointInTimeNormalizer`, and customer lifetime feature extraction. |
| [`recommender/sequences.py`](recommender/sequences.py) | 32-event chronological purchase sequence builder (`ENRICHED_FEATURE_DIM = 9`). |
| [`recommender/text.py`](recommender/text.py) | `BAAI/bge-m3` (`1024D`) text embedding generation and parquet caching. |
| [`recommender/familiarity.py`](recommender/familiarity.py) | Point-in-time binary customer familiarity flags (`brand`, `category`, `business_line`, `family`). |
| [`recommender/models.py`](recommender/models.py) | Stage 1 `CustomerTower`, `ProductTower`, and `TwoTowerModel`. |
| [`recommender/stage1_catalogue.py`](recommender/stage1_catalogue.py) | Rolling prior-sale candidate pool (`RollingCandidatePool`) and `load_stage1_discovery`. |
| [`recommender/train_retriever.py`](recommender/train_retriever.py) | Stage 1 training pipeline (`--discovery-reference` and standard two-tower modes). |
| [`recommender/evaluate_stage1.py`](recommender/evaluate_stage1.py) | Stage 1 rolling discovery evaluation against repeat/popularity heuristics. |
| [`recommender/candidates.py`](recommender/candidates.py) | `PointInTimeFeatureStore`, `CandidateRetriever`, 50-item mixed slate builder, and 13 pairwise features. |
| [`recommender/ranker.py`](recommender/ranker.py) | Stage 2 `UPSCANRRanker` and `RankerLoss`. |
| [`recommender/train_ranker.py`](recommender/train_ranker.py) | Stage 2 slate caching, training loop, and checkpoint selection (`discovery` vs `ndcg`). |
| [`recommender/evaluate.py`](recommender/evaluate.py) | Full validation pipeline evaluation and conditional test-gate runner. |

---

## 4. Saved Checkpoints & Caches

| Path | Description |
| :--- | :--- |
| `checkpoints/retriever-discovery-reference/best_retriever.pt` | Active Stage 1 Discovery Reference checkpoint (Epoch 23, 128D, 4 familiarity flags). |
| `checkpoints/retriever/best_retriever.pt` | Standard 10-epoch Stage 1 checkpoint used by unit/integration tests. |
| `checkpoints/stage1-full-catalogue/text_embeddings/` | Cached `BAAI/bge-m3` text embeddings for the full merchandise catalogue used by `load_stage1_discovery`. |
| `checkpoints/stage1-rolling-discovery/` | Stage 1 rolling candidate pool evaluation outputs. |
| `checkpoints/ranker/best_ranker.pt` | Active Stage 2 `UPS-CANR` reference checkpoint (`statistics_attention=True`, `excluded_statistics=("history_repeat",)`). |
| `checkpoints/ranker/best_discovery_ranker.pt` | Discovery-selected Stage 2 reference checkpoint (`209 / 1,857` new-to-customer hits). |
| `checkpoints/ranker/best_ndcg_ranker.pt` | Peak-NDCG Stage 2 reference checkpoint (`0.5934` Val NDCG@10). |
| `checkpoints/evaluation_report.json` | Full validation evaluation report for the Stage 1 + Stage 2 reference pipeline. |
| `data/interim/ranker_train_slates_v3_repeat_0.5_rolling_v1.pt` | Cached 50-candidate training slates (`26,184` transitions) for the reference Stage 1 retriever. |
| `data/interim/ranker_val_slates_v3_repeat_0.5_rolling_v1.pt` | Cached 50-candidate validation slates (`6,191` transitions) for the reference Stage 1 retriever. |
