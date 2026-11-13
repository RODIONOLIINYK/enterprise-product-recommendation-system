# Agent Guide & Reference Architecture (`AGENTS.md`)

This document guides coding agents working in this repository. Source behavior and checkpoint metadata were audited on 2026-11-10; **local artifacts differ from the historical validation reference**. See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) and [`audit/artifacts.json`](audit/artifacts.json) before selecting a baseline. Validation numbers below are historical records, not reproduced by this audit. See [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) for validation metrics and [`plan.md`](plan.md) for current project goals and next experiments.

> **Current Reference for Future Stage 1 `FeatureAblatedStage1Selector` MLP Work (`repeat-no-month-v1`, 2026-11-13):**
> - **Canonical Active Pointer & Reference Pair:** [`experiments/stage1_mlp_reference/reference.json`](experiments/stage1_mlp_reference/reference.json) resolves [`checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json`](checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json) (`best_retriever.pt` epoch 16 + `best_selector.pt` epoch 5, `seed=42`, `FeatureAblatedStage1Selector` `uv_45`, `301D`; see [`experiments/stage1_mlp_reference/REFERENCE.md`](experiments/stage1_mlp_reference/REFERENCE.md)). Saved by explicit user override (`PROMOTION_DECISION.md` records the prior non-promotion recommendation and historical `promoted: false` campaign flags remain preserved).
> - **Frozen Historical Pointer (`no-brand-v1`):** [`experiments/embedding_ablation_v1/reference.json`](experiments/embedding_ablation_v1/reference.json) (`no-brand-v1`, `uv_47`, `303D`, SHA-256 `148da7db...`) is **frozen byte-for-byte** for prior campaign reproduction (`mlp_feature_ablation`, `description_cleaning`, `mlp_repeat_restored`). Do not repoint it.
> - **Configuration (`301D`):** Over frozen `E_no_brand` Stage 1 towers (separate learned `brand` categorical embedding absent from `ProductTower`; `brand` retained in `t1_fam` text with `product_family` and no descriptions, `user_brand_order_share` / `is_repeat_brand`, and Stage 1 `brand` familiarity; `sales_branch` customer embedding kept; 6 product embeddings `category`, `subcategory`, `business_line`, `tier`, `unit`, `country_of_origin`; demand disabled `12D` `container_type` only), selector `uv_45` **retains `is_repeat_sku`** (col `0` / MLP col `256`) and omits **ONLY `month_sin` and `month_cos`** (`39` context + `6` customer stats = `45` scalars + `256` `u`/`v` = `301`). Legacy numeric normalization unchanged (**NOT yet all `[0, 1]`** — `8` signed lift/growth features and unit-norm `u`/`v` remain in `[-1, 1]`).
> - **Validation-Only Metrics (`6,191` queries, `10,276` total targets, `1,857` new targets, `2025-10-01`..`2026-02-18`):** Recall@5 `0.64335`, Recall@10 `0.75411`, Recall@25 `0.84850`, Recall@50 `0.89542`, Recall@100 `0.93733`, NDCG@10 `0.58633`, ordinary new hits@10 `194/1,857`, unseen hits@25 `726/1,857`, @50 `1,029/1,857`, @100 `1,337/1,857`. Tradeoffs vs `no-brand-v1`: higher NDCG@10 / Recall@5 / unseen@100, lower Recall@10 (`-0.245 pp`) and ordinary top-10 discovery (`-20`); single seed (`seed=42`), selected epochs (`16`/`5`), pending normalization and category work — do not claim statistical superiority.
> - **Narrow Scope:** Use as the default baseline **only for future Stage 1 MLP selector experiments**. It is **NOT** a Stage 2 (`UPS-CANR`) or production retriever default; preserve all historical `checkpoints/retriever-discovery-reference/` and `checkpoints/ranker/` aliases.

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
    - Embeds `city_id`, `region_id`, `industry_id` (`16D` each) + **6 explicit profile features in the local checkpoint** (`distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`) $\to 128\text{D}$.
    - Encodes the last 32 purchase events using `catalog_history_encoder = "attention"`: each historical token combines its shared `128D` catalogue product vector, normalized event quantity (`quantity_reference = 60.0` in the local checkpoint), and normalized inter-purchase gap (`history_gap_reference_days = 74.0`) through a 2-layer Transformer (`4` heads) and masked-mean plus last-valid-event pooling followed by a learned projection $\to 128\text{D}$.
    - Late-fusion MLP combines `[text (128), tabular (128), history (128)]` $\to 128\text{D}$.
  - **`ProductTower`:**
    - Uses `remove_product_attribute_embeddings = True` (no learned brand/category/subcategory/line ID embeddings inside `ProductTower`).
    - Combines `1024D` `BAAI/bge-m3` product text (without subcategory), one-hot `container_type` with an explicit `none` bucket (`12` buckets in the local checkpoint), and **6 demand features** (`total_volume`, `orders_30d`, `orders_90d`, `unique_buyers`, `category_market_share`, `days_since_last_sold`); optional `sku_median_quantity` is disabled in the local checkpoint. Demand uses a fixed training-cutoff snapshot, which is not point-in-time for earlier training predictions (see audit) $\to 128\text{D}$.
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
  9. `global_sku_popularity_norm`
  10. `industry_sku_affinity`
  11. `is_repeat_category`
  12. `is_repeat_brand`
  13. `is_repeat_business_line`

### Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker (`UPS-CANR`)
- **Code:** [`recommender/ranker.py`](recommender/ranker.py), [`recommender/train_ranker.py`](recommender/train_ranker.py), [`recommender/evaluate.py`](recommender/evaluate.py)
- **Default Checkpoint Path:** `checkpoints/ranker/best_ranker.pt` (`statistics_attention=True`, `excluded_statistics=("history_repeat",)`). The local alias is epoch 7, NDCG-selected (`201 / 1,857` discovery hits; stored NDCG `0.593815`). The local discovery checkpoint is epoch 4 (`209 / 1,857`; stored NDCG `0.591893`). Preserve both; neither establishes the historical epoch-10 reference below.
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
  4. **Historical Documented Validation Performance (`6,191` baskets):**
     - **New-to-Customer Recall@10:** **`11.25%` (`209 / 1,857`)**
     - **Overall Recall@10:** **`76.15%`**
     - **Overall NDCG@10:** **`0.5924`** (discovery-selected checkpoint Epoch 10; historical reference peak NDCG checkpoint achieves `0.5934`; `0.5945` belongs to the rejected due-score experiment).

---

## 3. Clean Module Inventory (`recommender/`)

| Module | Purpose |
| :--- | :--- |
| [`recommender/data.py`](recommender/data.py), [`merging.py`](recommender/merging.py), [`scope.py`](recommender/scope.py), [`preparation.py`](recommender/preparation.py) | Raw workbook/CSV parsing, paid-sale filtering (`ПРОДАЖА` + `ТОВАР`), quantity-summing record deduplication, and `cleaned_purchases.csv` generation. |
| [`recommender/company.py`](recommender/company.py), [`catalogue.py`](recommender/catalogue.py) | Company metadata (`city`, `region_code`, `industry_branch`) and product catalogue metadata (`brand`, `product_category`, `product_subcategory`, `business_line`, `container_type`, `product_family`). |
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
| `checkpoints/ranker/best_ranker.pt` | Local NDCG-selected alias, epoch 7: `201 / 1,857`, stored NDCG `0.593815`; audit provenance applies. |
| `checkpoints/ranker/best_discovery_ranker.pt` | Local discovery selection, epoch 4: `209 / 1,857`, stored NDCG `0.591893`. |
| `checkpoints/ranker/best_ndcg_ranker.pt` | Local NDCG selection, epoch 7: stored NDCG `0.593815`, `201 / 1,857` discovery hits. |
| `checkpoints/evaluation_report.json` | Saved epoch-4 validation report; does not certify the current epoch-7 default alias. |
| `data/interim/ranker_train_slates_v3_repeat_0.5_rolling_v1.pt` | Cached 50-candidate training slates (`26,263` rows locally; historical documentation said `26,184`) for the Stage 1 retriever. |
| `data/interim/ranker_val_slates_v3_repeat_0.5_rolling_v1.pt` | Cached 50-candidate validation slates (`6,191` transitions) for the reference Stage 1 retriever. |

## 5. Audit and future experiment prerequisites

- Ordinary `PYTHONPATH=. ./venv/bin/pytest tests/` skips six unsplit real-purchase readers. `--allow-heldout-data` requires explicit final-test authorization; never use it for debugging or architecture selection.
- Slate identity now binds ordered feature names, scope policy and resolved product inputs. Old cache identities intentionally miss; the v3 payload format and saved checkpoint state remain compatible. No caches were regenerated by the audit.
- Enriched history **source** remains nine-dimensional. The experimental shared model selects source columns 0–7; its 176D tokens omit `real_event_indicator` and use three type dimensions. Its default 17D bias derives the validity column from `attention_mask`; reduced 16D bias is a separate ablation. See [SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md).
- Historical product-text input is zero in the normal Stage 2 path; its projection bias still contributes. Numerical memory inputs are zero in bias-only mode. The attached ProductTower is bypassed by direct candidate vectors; retain these checkpoint-compatible branches until a separately versioned change.
- The 2026-11-11 shared-representation experiments independently replayed both local reference selections. Stage 1 demand look-ahead remains unresolved: results are conditional on the frozen retriever. Preserve the default model and checkpoint aliases; shared vectors and reduced bias did not show a reproducible discovery gain. 192D/224D modes are implemented/tested but their training and later edge/scorer experiments are deferred.

## 6. Shared-representation experiments (2026-11-11)

- [SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md) records 16 controlled 10-epoch model/seed fits, selected-checkpoint metrics, bootstrap intervals, scope and reproduction commands.
- Separate implementation: `shared_ranker.py`, `shared_data.py`, `train_shared_ranker.py`, `shared_temporal_probe.py`, `summarize_shared_ranker.py`. Baseline integration in `train_ranker.py` is opt-in only.
- Caches: `data/interim/shared_stage1_v1/`; models/results: `checkpoints/shared-stage1-experiments/`; aggregate evidence: `experiments/shared_stage1/`. Never reuse these as unversioned legacy slates.
- All old checkpoints/caches were preserved. Source sequences remain 9D; shared token history is 8D with default history-repeat exclusion. Canonical SKU mapping distinguishes real unknown events from padding through the mask.

## 7. E1–E5 experiments (2026-11-11–12)

- [E1_E5_EXPERIMENTS.md](E1_E5_EXPERIMENTS.md) and [experiments/e1_e5/README.md](experiments/e1_e5/README.md) document 32 completed fits, reproduction commands, telemetry, seed variation and ablations. No model or default alias was promoted.
- E1/E2 remain conditional on the audited frozen retriever: three-seed discovery hits average 196.33 for fresh controls, 202.33 for the unseen scalar, 201.67 for the transition residual and 203.67 for the transition MLP feature. None establishes a gain over the preserved 209-hit discovery checkpoint.
- E3–E5 have separate scratch-weight, demand-disabled controls and prior-sale training masks. S0 averages 719/1,857 unseen hits@25 and 980/1,857@50, but its downstream mean is 194.67 ± 11.24 discovery hits. Longer memory and the tested multi-interest variants failed the single-seed screen; all multi-interest models collapsed. Do not interpret these controls as a promoted fix to the production retriever.
- Code: `experimental_signals.py`, `experiment_signals.py`, `experimental_retrieval.py`, `experiment_retrieval_data.py`, `experiment_retrieval.py`, `experiment_serving.py`, `experiment_evidence.py`. Inputs: `data/interim/e1_e5_retrieval_v1/`; artifacts: `checkpoints/e1-e5/`; aggregate evidence: `experiments/e1_e5/`. Preserve these namespaces and all older artifacts.
- `PurchaseEmbeddingCache` encodes only on purchase or explicit versioned rebuild; requests use cached vectors. `refresh_catalogue` creates a new namespace and remaps canonical SKU biases without mutating the old model. No request-clock-dependent tower computation is supported.

## 8. Follow-up hardening (2026-11-12)

- [HARDENING_REPORT.md](HARDENING_REPORT.md) records verified fixes, compatibility checks and synthetic timings. All seven inventoried checkpoint/cache hashes and reference replay metrics remain unchanged; no model was promoted.
- New base retriever training defaults to `disabled_v1` demand and writes exclusive `runs/<run-id>/` children. Missing policy on old checkpoints reconstructs legacy fixed-cutoff inputs. This closes the new trainer's dynamic-demand pathway, not historical metadata or legacy candidate-mask issues.
- Base ranker caches now use sealed format v4, separate from model/slate schema v3 and historical experimental caches. New ranker checkpoint containers require ordered input and originating cache provenance. Untagged legacy artifacts remain explicitly unverified; `--strict-provenance` rejects them.
- Optional `observed_only_v1` constrains pairwise metadata to strictly earlier purchase observations in a separate cache namespace. Default `snapshot_unverified` preserves the baseline. Neither policy supplies missing effective-date catalogue/text/block history.
- Both base trainers create run manifests; evaluation can resolve a completed ranker run directory and verifies the selected file hash. Never infer a successful new run from parent `best_*.pt` aliases. Default evaluation reports are unique, preserving the historical report.
- Production date-gated CSV reading is in `development.py`. `--synthetic-only` skips local-artifact tests and rejects repository private-file reads; CI contains no private artifacts. Normal tests still skip the six protected real-data readers.
- Named ranker stages and opt-in `forward_scores` preserve default output/state compatibility. New cached-only training freezes the attached ProductTower; raw-input mode and legacy projections remain supported. Purchase-triggered serving cache behavior is unchanged.

## 9. Dynamic selection / candidate context campaign (2026-11-12)

- [DYNAMIC_SELECTOR_EXPERIMENTS.md](DYNAMIC_SELECTOR_EXPERIMENTS.md) and [campaign commands](experiments/dynamic_campaign/README.md) document 68 completed validation-only fits, selected-weight/cache replays, three-seed comparisons, clustered intervals, ablations, CPU/MPS latency and memory costs. No model was promoted; old checkpoints/caches and reference replay remain unchanged.
- New code is isolated in `recommender/experiments/dynamic_selector/` and `candidate_context/`; artifacts use `checkpoints/dynamic-campaign-v1/` and `data/interim/dynamic_campaign_v1/`. The only core scoring extension is optional `memory_value_statistics=False`; default output/state behavior is preserved.
- All paired new fits share frozen demand-disabled S0-seed42 epoch 13. Common training cohort is 26,253 rows with at least 100 eligible products; validation retains all 6,191 transitions and 1,857 new targets. Globals are strictly prior in training and frozen at TRAIN_CUTOFF in validation. Metadata is explicitly snapshot-unverified; brand/subcategory/industry purchase-observed availability is zero in the measured sample.
- Three-seed new hits: original clean 50-item control 188.0 ± 4.4; dot/100 control 214.0 ± 5.3; dot/100 with scalar features 227.7 ± 4.9; dynamic MLP/100 with scalar features 233.3 ± 6.7. The latter increments over depth-matched controls remain uncertain. Combined candidate attention reaches 239.0 ± 5.3 but fails NDCG in every seed. Do not choose a model by discovery count alone.
- Conservative recommendation: retain single-vector dot selection and original UPS-CANR, evaluate the 100-item mixed policy for approval; retain scalar augmentation as experimental. Warm MPS request medians: dot/100 3.115 ms; dot/100 + scalar features 3.483 ms; full-pool MLP/100 10.634 ms. Resident masks/facts are assumed; network/storage/ingestion are excluded.
- New input features are quantized float16 then computed in float32 in both cache and request adapters. Candidate-subset building preserves full history. Towers encode only on purchase or explicit versioned rebuild. Unqualified runs retain diagnostic weights without a selected alias. Seeded MPS training is not claimed bitwise deterministic; selected-checkpoint inference was replayed.

## 10. Exploratory variant policy (2026-11-13)

- New exploratory model/feature variants run on a **single seed (`seed=42`)** by default (without mandatory 3-seed repeats) to keep iteration fast. Multi-seed repeats (`17`, `42`, `2026`) are reserved for final promotion candidates or when explicitly requested by the user.
- All non-negotiable rules remain unchanged: strict validation-only evaluation (`< 2026-02-19`), zero held-out test-split access, isolated experimental namespaces, and full test-suite verification (`PYTHONPATH=. ./venv/bin/pytest tests/` and `--synthetic-only`).
