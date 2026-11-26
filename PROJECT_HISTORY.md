# Project History & Model Evolution (`PROJECT_HISTORY.md`)

Chronological engineering log of the **Enterprise B2B Product Recommendation System** from initial prototype (`2026-07-16`) to the final promoted `Wave2BE2ERanker` (`w2_2mem_recency_hl365`, `2026-11-26`), grounded directly in git commits and saved evaluation reports.

---

## Executive Progression Summary (Full-Catalogue Validation)

| Phase | Period & Key Commits | What Was Introduced | Dataset / Evaluation Scope | Overall Recall@10 | NDCG@10 | New-to-Customer @10 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **0. Early CatBoost Prototypes** | `2026-07-16 .. 08-18`<br>(`afc66ef`, `0bd5a91`, `2a5eff1`, `0fdb00b`, `2cb7f75`) | Initial replenishment CatBoost Ranker → calibrated CatBoost Classifier prototypes before pivoting to full-catalogue Two-Tower neural retrieval. | Small sampled candidate subset *(not full-catalogue)* | — | — | — |
| **1. Global Popularity Baseline** | `2026-10-07 .. 11-07`<br>(`c9e8786`, `a0f5407`) | Strict chronological split (`zero future leakage`), paid-sale filtering (`ПРОДАЖА` + `ТОВАР`), full-catalogue evaluation (`6,191` val baskets). | v1 Val (`6,191` baskets, SKU) | **17.88%** | `0.0925` | — |
| **2. First Modular Two-Tower (Stage 1)** | `2026-11-07`<br>(`a0f5407`) | 128D `CustomerTower` (BGE-M3 + history Transformer) & `ProductTower` (BGE-M3 + catalogue attributes), 10 epochs. | v1 Val (`6,191` baskets, SKU) | **47.60%** | `0.3007` | — |
| **3. First `UPS-CANR` Neural Ranker (Stage 2)** | `2026-11-07`<br>(`a0f5407`) | 33-token Customer Memory (Token 0 profile + Tokens 1..32 history), 4-head cross-attention, 50-item rerank, BCE + listwise loss. | v1 Val (`6,191` baskets, SKU) | **66.69%** | `0.5347` | — |
| **4. Standalone CatBoost v1 (GBDT Benchmark)** | `2026-11-13`<br>(`EVALUATION_RESULTS.md`) | Full-catalogue tabular CatBoost (`52` features, `318` trees, inverse-probability sampled negatives) directly scoring all `4.7M` eligible pairs. | v1 Val (`6,191` baskets, SKU) | **75.82%** | `0.5937` | **10.88%** (`202 / 1,857`) |
| **5. Discovery Reference + Statistics Attention** | `2026-11-08 .. 11-10`<br>(`1fe9a65`, `d0cee7e`, `677ed97`, `4f3d326`) | 30-ep Stage 1 Discovery Reference (removed noisy SKU ID embeddings, added 4 familiarity flags + rolling prior-sale pool); Stage 2 bilinear `statistics_attention` (`excluded history_repeat`). | v1 Val (`6,191` baskets, SKU) | **76.15%** | `0.5924` | **11.25%** (`209 / 1,857`) |
| **6. Rolling Causality & 100-Item Slates** | `2026-11-10 .. 11-14`<br>(`be81789`, `3fbec83`, `464cb7c`, `22f2142`) | Fixed temporal look-ahead leaks, sealed artifact provenance, `rolling_observed_v2` strictly `< as_of_date` stats, 100-candidate rolling slates (beat CatBoost v1 by `+1.28 pp` & `+53` new hits). | v1 Val (`6,191` baskets, SKU) | **77.10%** | `0.6022` | **13.73%** (`255 / 1,857`) |
| **7. Canonical `base_name` + Full-Pool Stage 5** | `2026-11-21 .. 11-22`<br>(`5f66ac7`, `ce06109`, `eba3b4e`, `e0681bd`, `91763d5`) | Unified pack variants (`841` SKUs → `585` formulations), 60-day aux head (`λ=0.3`), financial + manager + `65D` parsed specs, mixed returning + cold-start baskets, **removed Stage 1 MLP** (`SharedUVStage5Ranker` scores all `585` products in `1.32 ms`). | v1 Val (`6,480` baskets, `base_name`) | **80.89%** | `0.6346` | **21.62%** (`372 / 1,721`) |
| **8. CatBoost v2_full (`cand_42` + Option A)** | `2026-11-24`<br>(`v2_full_catboost_wave2b`) | Full 3-year `v2_full` GBDT benchmark on exact `cand_42` features: mixed CatBoost (`81.40%` R@10 / `0.6426` NDCG / `1,108` New) + Option A OOF affine blend. | v2 Val (`16,623` baskets, `1,141` prod.) | **81.61%** | `0.6438` | **25.33%** (`1,139 / 4,496`) |
| **9. Full 3-Year `v2_full` E2E + Option A (`w2_2mem_blocks`)** | `2026-11-25`<br>(`1ba48b6`) | Scaled to full 3-year ERP ledger (`137,765` train / `16,623` val / `13,184` test, `1,141` products / `1,660` SKUs). Single-stage E2E `Wave2BE2ERanker`, pruned `cand_42 + CUST_7 + SEQ_5`, Pure-Q `Q=v_bn`, Full-42 bias, 2× memory layers, separated `OptionAUnseenHead` + OOF affine blend (beats CatBoost v2 by `+0.57 pp` R@10 / `+0.0104` NDCG). | v2 Val (`16,623` baskets, `1,141` prod.) | **82.18%** | `0.6542` | **27.31%** (`1,228 / 4,496`)<br>PureU: **51.67%** |
| **10. Final Promoted: 2× Memory + 365d Recency (`w2_2mem_recency_hl365`)** | `2026-11-26`<br>(`3ae0a3e` → `7421499`) | Added smooth exponential recency sample weighting (`half-life = 365 days`, mean-normalized `1.00×`: Jan 2026 `2.31×`, 1 yr `1.16×`, 3 yr `0.28×`). Massive `+3.01 pp` cold-start boost on Val & **`+5.98 pp`** on zero-shot unlearned products on Held-Out Test! | **v2 Val** (`16,623` all / `15,343` Trained C&P)<br>**v2 Held-Out Test** (`13,184` all / `9,839` Trained C&P, `7.4 mo` gap) | **82.26%** (all)<br>**84.27%** (Trained)<br>**76.63%** (all)<br>**83.03%** (Trained) | `0.6548`<br>`0.6718`<br>`0.5841`<br>`0.6349` | **27.91%** (PureU `51.78%`)<br>**20.52%**<br>**21.71%** (PureU `42.78%`)<br>**15.69%** (PureU `43.14%`) |

---

## Detailed Chronological Log by Phase

### Phase 0: Early Tabular CatBoost Prototypes (`2026-07-16 .. 2026-08-18`)
- **Key Commits:** `afc66ef`, `36f2c16`, `6b9a3bb`, `0bd5a91`, `2a5eff1`, `60594f6`, `97cfd30`, `29ab48b`, `0fdb00b`, `7ba1756`, `2cb7f75`, `d20ce22`.
- **What Was Built:** Initial replenishment feature notebooks, `CatBoostRanker` → `CatBoostClassifier` prototypes, and early package-variant unification (`97cfd30`, `7ba1756`).
- **Why Replaced:** Evaluated only over small sampled candidate groups rather than live full-catalogue chronological retrieval, motivating the pivot (`2cb7f75`) to a neural Two-Tower retriever + deep reranker.

### Phase 1: Clean Temporal Two-Stage Neural Foundation (`2026-10-07 .. 2026-11-07`)
- **Key Commits:** `c9e8786`, `753c53b`, `a0f5407`.
- **What Was Built:**
  - Retired legacy scripts and introduced strict chronological splitting (`TRAIN_CUTOFF = 2025-10-01`, `TEST_START = 2026-02-19`, zero test leakage) and genuine paid-sale eligibility (`ПРОДАЖА` + `ТОВАР`).
  - Built `BAAI/bge-m3` (`1024D`) semantic text embeddings for both companies and products, 32-event chronological customer sequences, **Stage 1 Two-Tower Retriever**, and **Stage 2 `UPS-CANR` Neural Ranker** (`33`-token customer memory + 4-head cross-attention).
- **Performance Impact (`6,191` Validation Baskets):**
  - Global Popularity Baseline: `Recall@10 = 17.88%`, `NDCG@10 = 0.0925`.
  - Stage 1 Two-Tower Retriever (10 epochs): `Recall@10 = 47.60%`, `Recall@50 = 75.74%`, `NDCG@10 = 0.3007` (**`+29.72 pp`** over popularity).
  - Stage 2 `UPS-CANR` Neural Ranker (Top-50 rerank): `Recall@10 = 66.69%`, `NDCG@10 = 0.5347` (**`+19.09 pp`** over Stage 1 alone).

### Phase 2: Discovery Reference & Statistics-Biased Cross-Attention (`2026-11-08 .. 2026-11-10`)
- **Key Commits:** `1fe9a65`, `8329d2b`, `d0cee7e`, `677ed97`, `4f3d326`, `4ce2b7b`, `4cbe764`.
- **What Was Built:**
  - Extended Stage 1 to 30 epochs, added container-type features and customer medians (`1fe9a65`), replaced noisy learned product brand/category/line ID embeddings in `ProductTower` with 4 explicit point-in-time familiarity flags (`brand`, `category`, `business_line`, `product_family`, `d0cee7e`), and introduced the strictly causal `RollingCandidatePool` (`677ed97`).
  - Upgraded Stage 2 `UPS-CANR` with **bilinear Statistics-Biased Cross-Attention** (`W_h` pairing 13 pairwise candidate features with memory statistics) and removed the harmful `history_repeat` bias flag (`4f3d326`).
- **Performance Impact (`6,191` Validation Baskets):**
  - Stage 1 Unseen Discovery: `35.11%` (`652 / 1,857`) @25 and `49.60%` (`921 / 1,857`) @50.
  - Stage 2 `UPS-CANR`: **`Recall@10 = 76.15%`** (**`+9.46 pp`** over initial `66.69%`), **`NDCG@10 = 0.5924`**, **New-to-Customer Recall@10 = `11.25%` (`209 / 1,857`)**.
  - Rejected hypotheses archived in `EVALUATION_RESULTS.md`: frozen Stage 1 vectors in Stage 2 (`-20` new hits), aligned 11D diagonal/full attention (`-22` to `-25` new hits), reorder `due_score` in bias-only mode (`-6` new hits).

### Phase 3: Causality Audit, Rolling Features & Stage 1 MLP (`2026-11-10 .. 2026-11-14`)
- **Key Commits:** `be81789`, `99bc1c5`, `c7788ec`, `3fbec83`, `39d9753`, `5deb5b7`, `da66fec`, `34e6980`, `464cb7c`, `22f2142`.
- **What Was Built:**
  - Full codebase audit (`99bc1c5`) discovered and closed fixed-cutoff demand look-ahead (`disabled_v1` demand), sealed cache/checkpoint provenance (`3fbec83`), and replaced frozen train-cutoff validation stats with strictly rolling `< as_of_date` statistics (`rolling_observed_v2`, `22f2142`).
  - Evaluated static metadata enrichment, Stage 1 MLP selectors (`repeat-no-month-v1`, `301D`, `464cb7c`), 100-candidate rolling slates, and head-to-head standalone CatBoost (`52` features, full eligible pool).
- **Performance Impact (`6,191` Validation Baskets):**
  - Rolling pointwise Stage 2 (100-item slates): **`Recall@10 = 77.10%`**, **`NDCG@10 = 0.6022`**, **New@10 = `13.73%` (`255 / 1,857`, `+46` new hits)**.
  - Standalone CatBoost benchmark on the exact same pool: `Recall@10 = 75.82%`, `NDCG@10 = 0.5937`, `New@10 = 202` — confirming the Transformer cross-attention architecture beats GBDTs on both overall ranking and discovery.

### Phase 4: Roadmap v2 & Full-Pool Stage 5 Formulations (`2026-11-21 .. 2026-11-22`)
- **Key Commits:** `5f66ac7`, `e8d4144`, `ce06109`, `0fc578a`, `5a6347d`, `eba3b4e`, `e0681bd`, `91763d5`.
- **What Was Built:**
  - Audit revealed `24.5%` (`455 / 1,857`) of "new-SKU" targets were simply **pack-size switches** of an already-bought formulation. Unified `841` SKUs into **`585` canonical `base_name` product formulations** across every stage + added a deterministic causal pack-size SKU resolver at the end.
  - Added **60-day training-only auxiliary multi-task loss** (`λ = 0.3`, multiplying unseen positives `3.34×` with zero inference overhead), plus strictly causal **financial (`ДатаПолнойОплати < t`), rolling 90d manager, and `65D` parsed technical spec features** (`container_type`, `volume` tiers, `application_domain`, `viscosity_grade`).
  - Added **first-observed (cold-start) baskets** directly into dated training and validation (`6,480` baskets).
  - **Eliminated Stage 1 MLP truncation completely (`ce06109`, `e0681bd`):** `SharedUVStage5Ranker` (`567K` params) scores **all `585` eligible formulations directly** in **`1.32 ms`** on CPU.
  - Optimized live demo serving latency from `1,845 ms → 4.6 ms` (`5a6347d`) and closed 11 pipeline audit items with 27 regression tests (`91763d5`).
- **Performance Impact (`6,480` Mixed Validation Baskets):**
  - **`Recall@10 = 80.89%`**, **`NDCG@10 = 0.6346`**, **New@10 = `21.62%` (`372 / 1,721`)**, First-Observed NDCG@10 `0.2523` (`149 / 354` hits).

### Phase 5: Full 3-Year `v2_full` E2E Architecture, Option A & Recency (`2026-11-25 .. 2026-11-26`)
- **Key Commits:** `1ba48b6`, `3ae0a3e`, `be8d9fd`, `b9dc6f5`, `4658ae3`, `89cafa3`, `7421499`.
- **What Was Built:**
  - Scaled to the full 3-year ERP export (`2023-01-03 .. 2026-11-20`): **`137,765` Train** (`< 2026-02-01`), **`16,623` Validation** (`2026-02-02 .. 2026-06-30`), **`13,184` Held-Out Test** (`2026-07-01 .. 2026-11-20`), **`1,141` canonical products (`1,660` SKUs)**.
  - **Single-Stage End-to-End `Wave2BE2ERanker` (`1ba48b6`):** Eliminated separate Stage 1 Two-Tower pretraining altogether — `ProductTower` and Customer Token 0 train end-to-end directly inside Stage 2 (`ProductTower` cached once at startup in `1.4 ms`, zero per-request overhead).
  - **Wave 2B Architectural Refinements:** Pruned dead features (`cand_42 + CUST_7 + SEQ_5`, adding `mgr_30d`, `mgr×industry`, `region_90d`, `full_cf_cosine`, `full_cf_recency_weighted`), switched to **Pure-Q (`Q = v_bn`)**, expanded bilinear attention bias from 13D → **Full-42**, and introduced the **separated `OptionAUnseenHead`** (dedicated cross-sell head with 2-parameter customer-grouped OOF affine blend `a, b`).
  - **Wave 2A Final Promotion (`3ae0a3e`):** Doubled memory depth (**2× Transformer layers**) and added **exponential 365-day recency sample weighting (`w2_2mem_recency_hl365`)**, beating both `v2_full` CatBoost (`81.61%` R@10 / `0.6438` NDCG) and the uniform-weight `w2_2mem_blocks` baseline.
  - **Authorized Final Held-Out Test & Delivery (`b9dc6f5` → `7421499`):**
    - Evaluated strictly once on all `13,184` held-out Test orders (**`7.4 months` / `225 days` average gap** after the last training purchase `2026-01-30`; up to **`9.7 months` / `294 days`** on `2026-11-20`):
      - **Trained Customers & Products (`9,839`):** **`83.03%` Recall@10**, `0.6349` NDCG@10 (stable monthly: `Jul 82.33% → Nov 82.88%`, proving zero overfitting vs. `84.27%` Val).
      - **Within `1,141` catalogue (`12,928`):** **`79.21%` Recall@10**, `0.6010` NDCG@10, `50.47%` Pure Unseen@10.
      - **Full Catalogue (`13,184`):** **`76.63%` Recall@10**, `0.5841` NDCG@10.
      - **Head-to-Head Test Proof (`w2_2mem_recency_hl365` vs. `w2_2mem_blocks` without recency):** `+0.79 pp` Full Catalogue (`76.63%` vs `75.84%`), **`+3.31 pp`** on All Unlearned (`57.80%` vs `54.49%`), **`+4.51 pp`** on Unlearned Products (`56.55%` vs `52.04%`, `+62` new hits), and **`+5.98 pp`** on zero-shot `ProductTower` unlearned products (`66.29%` vs `60.31%`, `+58` new hits).
    - Packaged standalone **Live Python PyTorch CPU server** (`live_demo_server.py`, `3.02 ms` median / `314 req/sec`), interactive 11-block 3D architecture docs (`DOCUMENTATION.html`), Google Slides + offline `PRESENTATION.html` / `PRESENTATION.pdf`, and portable ZIP (`B2B_Recommender_Demo_v2_final.zip`).
