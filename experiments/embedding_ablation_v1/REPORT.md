# Isolated Product-Text & Product-Embedding Ablation Report (`U0_t1_fam_clean` + `CompactStage1Selector uv_47`, `seed=42`)

## 1. Scope, Protocol & Exact Metric Semantics

- **Target Baseline:** `U0_t1_fam_clean` (clean static towers: `t1_fam` text, 0 changing customer stats, static 12D `container_type` only on `ProductTower`, demand disabled, 7 product embeddings `category/subcategory/business_line/brand/tier/unit/country_of_origin`, customer `sales_branch` embedding).
- **Downstream Scorer:** Current `CompactStage1Selector` (`uv_47`, 303D input, 128/64 hidden layers, 10 epochs `seed=42`, selection policy `new_hits@50 -> ndcg@10 -> recall@10`). No numeric normalization changes, no new features, no Stage 2 / `UPS-CANR` runs.
- **Validation Cohort & Denominators:** Strictly `< 2026-02-19` (`2025-10-01` to `2026-02-18`), **6,191** validation transitions, **1,857** new-to-customer target instances. Training uses `< 2025-10-01` (`8,465` Stage 1 discovery rows; `26,253` selector rows with `>= 100` eligible SKUs).
- **Full-Pool Metrics:**
  - **New-target hits/recall@25, @50, @100:** Computed over unseen eligible SKUs (`unseen = allowed & ~seen`), micro-averaged over denominator **1,857**.
  - **Overall Recall@10 / NDCG@10:** Computed over all eligible SKUs (`allowed`), macro-averaged across **6,191** transitions against full `positive_pids`.
- **Inapplicable Ablations:**
  - Text `unit`: unit is not an explicit field in t1_fam text rendering; do not invent a text-unit ablation.
  - Embedding `product_family`: There is no product_family categorical embedding in the U0_t1_fam_clean baseline.

## 2. Part A — Independent Field-Level Text Deletions from `t1_fam` (Full-Pool `uv_47` MLP-Only)

| Variant | Omitted Field | Changed Texts (/841) | Empty | S1 Ep (s) | MLP Ep (s) | MLP New@25 | MLP New@50 | MLP New@100 | MLP Recall@10 | MLP NDCG@10 | Δ New@50 vs Base (95% CI) | Δ NDCG@10 vs Base (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| U0_t1_fam_clean | `none` | 0 | 0 | 26 (49.4s) | 1 (251.8s) | 737 (39.69%) | 1012 (54.50%) | 1320 (71.08%) | 0.75052 | 0.57713 | — | — |
| T_no_base_name | `base_name` | 47 | 0 | 24 (48.2s) | 1 (143.3s) | 740 (39.85%) | 1010 (54.39%) | 1321 (71.14%) | 0.75024 | 0.57693 | -2 ([-0.63, -0.11, +0.41] pp) | -0.0002 ([-0.0011, -0.0002, +0.0007]) |
| T_no_subcategory | `product_subcategory` | 833 | 0 | 24 (49.7s) | 1 (120.3s) | 734 (39.53%) | 1022 (55.04%) | 1324 (71.30%) | 0.75150 | 0.57681 | +10 ([-0.16, +0.54, +1.25] pp) | -0.0003 ([-0.0013, -0.0003, +0.0008]) |
| T_no_family | `product_family` | 171 | 0 | 20 (45.8s) | 1 (113.7s) | 737 (39.69%) | 1015 (54.66%) | 1323 (71.24%) | 0.75088 | 0.57704 | +3 ([-0.52, +0.17, +0.85] pp) | -0.0001 ([-0.0011, -0.0001, +0.0010]) |
| T_no_category | `product_category` | 833 | 0 | 26 (45.4s) | 1 (120.9s) | 731 (39.36%) | 1013 (54.55%) | 1313 (70.71%) | 0.75091 | 0.57646 | +1 ([-0.69, +0.05, +0.87] pp) | -0.0007 ([-0.0018, -0.0006, +0.0005]) |
| T_no_brand | `brand` | 830 | 0 | 18 (44.8s) | 1 (114.0s) | 733 (39.47%) | 1020 (54.93%) | 1319 (71.03%) | 0.75113 | 0.57738 | +8 ([-0.53, +0.43, +1.43] pp) | +0.0003 ([-0.0012, +0.0003, +0.0016]) |
| T_no_business_line | `business_line` | 819 | 0 | 20 (48.5s) | 1 (119.6s) | 736 (39.63%) | 1008 (54.28%) | 1325 (71.35%) | 0.75254 | 0.57617 | -4 ([-1.09, -0.21, +0.73] pp) | -0.0010 ([-0.0027, -0.0009, +0.0007]) |

### Upstream Stage 1 (128D Dot + Familiarity) — Text Deletions

| Variant | Omitted Field | S1 Ep | S1 New@25 | S1 New@50 | S1 New@100 | S1 Recall@10 | S1 NDCG@10 | Δ S1 New@25 (95% CI) | Δ S1 New@50 (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| U0_t1_fam_clean | `none` | 26 | 737 (39.69%) | 1009 (54.33%) | 1341 (72.21%) | 0.37333 | 0.22104 | — | — |
| T_no_base_name | `base_name` | 24 | 744 (40.06%) | 1028 (55.36%) | 1346 (72.48%) | 0.37102 | 0.21950 | +7 ([-0.57, +0.37, +1.29] pp) | +19 ([+0.11, +1.02, +1.88] pp) |
| T_no_subcategory | `product_subcategory` | 24 | 740 (39.85%) | 1004 (54.07%) | 1350 (72.70%) | 0.36766 | 0.21979 | +3 ([-1.00, +0.16, +1.39] pp) | -5 ([-1.51, -0.28, +1.00] pp) |
| T_no_family | `product_family` | 20 | 739 (39.80%) | 1016 (54.71%) | 1337 (72.00%) | 0.37491 | 0.22358 | +2 ([-1.11, +0.11, +1.34] pp) | +7 ([-0.82, +0.38, +1.59] pp) |
| T_no_category | `product_category` | 26 | 740 (39.85%) | 1017 (54.77%) | 1335 (71.89%) | 0.37038 | 0.21839 | +3 ([-0.83, +0.16, +1.15] pp) | +8 ([-0.68, +0.41, +1.47] pp) |
| T_no_brand | `brand` | 18 | 715 (38.50%) | 991 (53.37%) | 1320 (71.08%) | 0.36242 | 0.21607 | -22 ([-2.58, -1.20, +0.16] pp) | -18 ([-2.36, -0.96, +0.34] pp) |
| T_no_business_line | `business_line` | 20 | 737 (39.69%) | 1004 (54.07%) | 1336 (71.94%) | 0.37550 | 0.22260 | +0 ([-1.35, +0.00, +1.40] pp) | -5 ([-1.63, -0.27, +1.09] pp) |

## 3. Part B — Independent Product Categorical Embedding Deletions (Full-Pool `uv_47` MLP-Only)

| Variant | Removed Emb | S1 Params | S1 Ep (s) | MLP Ep (s) | MLP New@25 | MLP New@50 | MLP New@100 | MLP Recall@10 | MLP NDCG@10 | Δ New@50 vs Base (95% CI) | Δ NDCG@10 vs Base (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| U0_t1_fam_clean | `none` | 855,469 | 26 (49.4s) | 1 (251.8s) | 737 (39.69%) | 1012 (54.50%) | 1320 (71.08%) | 0.75052 | 0.57713 | — | — |
| E_no_category | `category` | 851,181 | 17 (44.2s) | 4 (109.3s) | 706 (38.02%) | 1029 (55.41%) | 1311 (70.60%) | 0.75513 | 0.58413 | +17 ([-1.11, +0.90, +2.89] pp) | +0.0070 ([+0.0032, +0.0069, +0.0113]) |
| E_no_subcategory | `subcategory` | 850,637 | 20 (43.3s) | 1 (109.8s) | 743 (40.01%) | 1004 (54.07%) | 1318 (70.97%) | 0.75314 | 0.57878 | -8 ([-1.64, -0.42, +0.81] pp) | +0.0017 ([-0.0005, +0.0017, +0.0038]) |
| E_no_brand | `brand` | 851,213 | 16 (43.3s) | 1 (110.0s) | 741 (39.90%) | 1037 (55.84%) | 1324 (71.30%) | 0.75656 | 0.57829 | +25 ([+0.16, +1.35, +2.63] pp) | +0.0012 ([-0.0026, +0.0013, +0.0042]) |
| E_no_unit | `unit` | 851,293 | 19 (43.1s) | 6 (141.3s) | 737 (39.69%) | 1007 (54.23%) | 1320 (71.08%) | 0.75516 | 0.58669 | -5 ([-2.19, -0.28, +1.66] pp) | +0.0096 ([+0.0061, +0.0095, +0.0131]) |
| E_no_business_line | `business_line` | 851,053 | 30 (60.7s) | 1 (174.4s) | 729 (39.26%) | 995 (53.58%) | 1330 (71.62%) | 0.75261 | 0.57913 | -17 ([-2.29, -0.91, +0.43] pp) | +0.0020 ([-0.0007, +0.0019, +0.0052]) |

### Upstream Stage 1 (128D Dot + Familiarity) — Embedding Deletions

| Variant | Removed Emb | S1 Ep | S1 New@25 | S1 New@50 | S1 New@100 | S1 Recall@10 | S1 NDCG@10 | Δ S1 New@25 (95% CI) | Δ S1 New@50 (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| U0_t1_fam_clean | `none` | 26 | 737 (39.69%) | 1009 (54.33%) | 1341 (72.21%) | 0.37333 | 0.22104 | — | — |
| E_no_category | `category` | 17 | 725 (39.04%) | 994 (53.53%) | 1335 (71.89%) | 0.38001 | 0.22082 | -12 ([-2.44, -0.65, +1.14] pp) | -15 ([-2.63, -0.81, +0.91] pp) |
| E_no_subcategory | `subcategory` | 20 | 719 (38.72%) | 1015 (54.66%) | 1310 (70.54%) | 0.36634 | 0.21599 | -18 ([-2.55, -0.93, +0.65] pp) | +6 ([-1.36, +0.33, +1.99] pp) |
| E_no_brand | `brand` | 16 | 726 (39.10%) | 1016 (54.71%) | 1322 (71.19%) | 0.36716 | 0.22232 | -11 ([-2.42, -0.58, +1.28] pp) | +7 ([-1.42, +0.40, +2.10] pp) |
| E_no_unit | `unit` | 19 | 737 (39.69%) | 1016 (54.71%) | 1331 (71.67%) | 0.36116 | 0.21625 | +0 ([-1.80, +0.00, +1.85] pp) | +7 ([-1.42, +0.36, +2.10] pp) |
| E_no_business_line | `business_line` | 30 | 722 (38.88%) | 1012 (54.50%) | 1319 (71.03%) | 0.35681 | 0.21432 | -15 ([-2.46, -0.80, +0.95] pp) | +3 ([-1.65, +0.16, +1.94] pp) |

## 4. Exploratory Cumulative Combination Decision

- **Omitted:** Only one individual deletion (`E_no_brand`) achieved a statistically supported `uv_47` `New@50` gain (`+25`, 95% CI `[+0.16, +1.35, +2.63] pp`) without degrading `New@25` or `New@100`. All 6 `t1_fam` text deletions had small effects whose paired 95% CIs straddled zero (`New@50` delta `-4` to `+10`), and the remaining embedding deletions either reduced `New@50` (`E_no_subcategory`, `E_no_unit`, `E_no_business_line`) or reduced `New@25`/`New@100`/upstream `Stage 1` (`E_no_category`: `New@25 -31`, `New@100 -9`, `uv_47 New@50` CI `[-1.11, +0.90, +2.89] pp`). Because there were not multiple independently promising deletions, no cumulative combination was trained.

## 5. Reproduction Commands

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/embedding_ablation_v1/run.py
PYTHONPATH=. ./venv/bin/python experiments/embedding_ablation_v1/summarize.py
```
