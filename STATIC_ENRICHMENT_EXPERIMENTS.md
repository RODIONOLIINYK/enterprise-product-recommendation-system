# Static Product & Customer Metadata Enrichment Campaign (`STATIC_ENRICHMENT_EXPERIMENTS.md`)

**Status:** Validation-only campaign complete (`heldout_access = false`, `promoted = false`).  
**Code:** [`recommender/experiments/static_enrichment/`](recommender/experiments/static_enrichment/), [`experiments/static_enrichment/`](experiments/static_enrichment/)  
**Artifacts & Caches:** `checkpoints/static-enrichment-v1/runs/`, `data/interim/static_enrichment_v1/`  
**Aggregate Evidence:** [`experiments/static_enrichment/summary.json`](experiments/static_enrichment/summary.json), [`experiments/static_enrichment/COMPARISON.md`](experiments/static_enrichment/COMPARISON.md), [`experiments/static_enrichment/decisions.json`](experiments/static_enrichment/decisions.json)

---

## 1. Scope, Protocol & Data Provenance

- **Protocol (`S0` demand-disabled baseline):** Scratch-trained 128D Two-Tower retriever (`30` epochs, `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(eta_min=1e-5)`, `pos_weight=20.0`, batch size `256`), `disabled_v1` product demand (all 18 time-dependent product inputs zeroed), prior-sale/not-blocked rolling eligibility masks, 4 point-in-time familiarity flags, and checkpoint selection by `(new_hits@25, new_hits@50)`.
- **Validation Scope:** `6,191` transitions (`2025-10-01` to `2026-02-18`), containing `1,857` new-to-customer target instances. The held-out test split (`>= 2026-02-19`) was never accessed or evaluated (`--allow-heldout-data` never used).
- **Downstream Stage 2 Protocol (`dot/100`, `quota100_50`):** `10` epochs, `26,253` training rows with `>= 100` eligible products, `6,191` validation transitions, `41` scalar features (`float16 -> float32`) in `dynamic_features` mode, and selection by `(qualified, new_hits@10, ndcg@10)` subject to both preserved reference quality floors:
  - `Recall@10 >= 0.7558833119851873`
  - `NDCG@10 >= 0.5918928743676547`

### Metadata Inspection & Caveats (`snapshot_unverified`)

All raw static fields come from point-in-time-unverified snapshots (`items.csv`, `customers.csv`, and `data/raw/описание.csv` exported `2026-11-12`). No historical effective-date changelog exists for descriptions, product tiers, branches, or VAT status; all enriched runs carry `metadata_policy = "snapshot_unverified"`.

1. **Product Tier (`col 127` vs `col 124` in `data/raw/items.csv`, `19,346` rows):**
   - **`col 127` (chosen):** `780 / 19,346` (`4.03%`) filled, `5` clean tier labels (`PREMIUM: 367`, `MAINSTREAM: 348`, `TOP: 36`, `ENTRY: 24`, `SPECIALAPP: 5`).
   - **`col 124` (rejected):** `1,605 / 19,346` (`8.30%`) filled, `60` unique values, of which `1,007 / 1,605` duplicate `brand` (`col 83`) and `328 / 1,605` duplicate `product_family` (`col 125`) (e.g., `FLEETGUARD: 577`, `PALL: 366`, `SKY: 128`, `SKF: 62`).
2. **Excluded Fields:** Customs code (`col 98` in `items.csv`), all item price columns, and price `col 3` in `data/raw/описание.csv` are strictly excluded.
3. **Product Descriptions (`data/raw/описание.csv`):**
   - `276` headerless `;`-separated rows (`268` matched `ТОВАР` items in `items.csv`; `253` non-empty after stripping HTML tags, entities, and `"NULL"` strings; median cleaned length `1,262.0` chars across all `276` rows, `1,511.0` chars across the `253` non-empty rows).
   - Within the `841` development SKUs: `159` appear in `описание.csv` and `155` have non-empty cleaned descriptions.
   - Within the `1,857` validation new targets: `514` have non-empty cleaned descriptions (`526` appear in `описание.csv`) and `1,343` do not (`1,331` absent from CSV).
4. **Customer Static Columns (`data/raw/customers.csv`, `42,775` rows):**
   - `col 44` (`sales_branch`): `6` branches (`ДНЕПРГЛАВ: 18,406`, `КИЕВ: 12,049`, `ЛЬВОВ: 4,169`, `ХМЕЛЬНИЦК: 3,324`, `ОДЕССА: 3,294`, `ХАРГЛАВНЫЙ: 678`), `855` empty (`2.00%`).
   - `col 59` (`vat_status`): `6` raw values (`К_ПЛАТ: 38,761`, `К_НЕПЛАТ: 1,902`, `БЕЗ ПДВ: 1,110`, `П_ПЛАТ: 6`, `П_НЕПЛАТ: 2`, `ЕКСПОРТ: 1`), `993` empty (`2.32%`).
   - Both columns remain in `EXCLUDED_COMPANY_COLUMNS` in core `recommender/company.py` and are loaded only inside `recommender/experiments/static_enrichment/`.
5. **Vocabulary Fitting (Training-Only, `MIN_TRAIN_CATEGORY_COUNT = 3`):**
   - Bucket `0` = `<NONE>`, bucket `1` = `<OOV>`, remaining buckets fitted strictly on training-eligible products (`>= 3` training products) or training customers.
   - Fitted vocabulary sizes: `brand: 10`, `category: 12`, `subcategory: 46`, `business_line: 20`, `product_family: 51`, `tier: 5`, `unit: 5`, `country_of_origin: 15`, `sales_branch: 7`, `vat_status: 5`.
   - `BAAI/bge-m3` (`1024D`) product text caches are stored in `data/interim/static_enrichment_v1/text_embeddings/` and keyed by SHA-256 recipe hash (`t1`, `t2`, `t1_fam`, `t2_fam`).

---

## 2. Stage 1 Variants & Screening Decisions

| Variant | Text Recipe | Extra Product Embeddings (`16D` each) | Extra Customer Embeddings (`16D` each) |
|---|---|---|---|
| `C0` | `c0_s0` (exact S0 text) | None | None |
| `T1` | `t1`: deduplicated `brand + name + base_name + product_type + category + subcategory + business_line` | None | None |
| `T2` | `t2`: `T1 + cleaned_description` | None | None |
| `A` | Best `T` (`T2`) | `category`, `subcategory`, `business_line`, `brand` | None |
| `F_emb` | `T2` | `A + product_family` | None |
| `F_txt` | `t2_fam`: `T2 + product_family` | `category`, `subcategory`, `business_line`, `brand` | None |
| `P` | Best `F` (`F_txt`: `t2_fam`) | `A + tier + unit + country_of_origin` | None |
| `P_no_unit` | `t2_fam` | `A + tier + country_of_origin` | None |
| `U0` | `t2_fam` | `best_P` (`A + tier + unit + country_of_origin`) | `sales_branch` |
| `U1` | `t2_fam` | `best_P` (`A + tier + unit + country_of_origin`) | `sales_branch`, `vat_status` |

Chained screening selections (`experiments/static_enrichment/decisions.json`):  
`best_t = "T2"` (`703@25` vs `T1`'s `702@25`), `best_f = "F_txt"` (`709@25` vs `F_emb`'s `704@25`), `best_p = "P"` (`725@25` vs `P_no_unit`'s `715@25`), `best_screen_overall = "P"`, `best_stage1_3seed = "P"`.

---

## 3. Stage 1 Results (Validation: `1,857` New Targets)

### 3.1 Seed-42 Screen (All 10 Variants, 30/30 Epochs Verified)

| Variant | Best Ep | Params | Fit (s) | Hits@25 | Recall@25 | Hits@50 | Recall@50 | With-Desc (514) @25 / @50 | No-Desc (1,343) @25 / @50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0` | 13 | 825,133 | 44.51 | 707 | 38.07% | 1,005 | 54.12% | 192 / 284 | 515 / 721 |
| `T1` | 13 | 825,133 | 44.09 | 702 | 37.80% | 1,009 | 54.33% | 190 / 285 | 512 / 724 |
| `T2` | 19 | 825,133 | 43.83 | 703 | 37.86% | 985 | 53.04% | 187 / 269 | 516 / 716 |
| `A` | 23 | 842,925 | 44.71 | 711 | 38.29% | 1,001 | 53.90% | 186 / 286 | 525 / 715 |
| `F_emb` | 20 | 847,837 | 45.44 | 704 | 37.91% | 1,009 | 54.33% | 186 / 285 | 518 / 724 |
| `F_txt` | 13 | 842,925 | 47.29 | 709 | 38.18% | 1,004 | 54.07% | 204 / 292 | 505 / 712 |
| `P` | 20 | 855,613 | 49.34 | **725** | **39.04%** | **1,017** | **54.77%** | 190 / 277 | 535 / 740 |
| `P_no_unit` | 17 | 851,437 | 49.29 | 715 | 38.50% | 985 | 53.04% | 160 / 253 | 555 / 732 |
| `U0` | 20 | 857,773 | 49.79 | 715 | 38.50% | 1,015 | 54.66% | 193 / 282 | 522 / 733 |
| `U1` | 20 | 859,901 | 50.36 | 719 | 38.72% | 1,006 | 54.17% | 194 / 272 | 525 / 734 |

- **Missing-Brand & Missing-Family Cohorts:** Across all 18 Stage 1 runs (including all 10 screening fits and all 3-seed repeats), `brand_missing` (`3` targets) achieved `0 / 3` hits@25 and `@50`, and `family_missing` (`6` targets) achieved `0 / 6` hits@25 and `@50`.
- `C0-seed42` reproduced the historical `S0-seed42` epoch-13 baseline (`707` unseen hits@25, `1,005` unseen hits@50).

### 3.2 Three-Seed Stage 1 Summary (`seeds = 17, 42, 2026`)

| Variant | Epochs (17, 42, 2026) | Params | Fit (s) Mean ± SD | Hits@25 Mean ± SD (Values) | Recall@25 | Hits@50 Mean ± SD (Values) | Recall@50 | With-Desc @25 / @50 (514) | No-Desc @25 / @50 (1,343) |
|---|---|---|---|---|---|---|---|---|---|
| `C0` | 20, 13, 21 | 825,133 | 44.93 ± 0.49 | 719.00 ± 14.42 (735, 707, 715) | 38.72% | 980.00 ± 23.26 (976, 1005, 959) | 52.77% | 194.33 ± 2.08 / 273.67 ± 9.61 | 524.67 ± 13.43 / 706.33 ± 17.47 |
| `P` | 23, 20, 16 | 855,613 | 52.69 ± 2.90 | **733.67 ± 9.02** (733, 725, 743) | **39.51%** | 1,020.33 ± 16.26 (1006, 1017, 1038) | 54.95% | 204.00 ± 13.53 / 291.33 ± 13.20 | 529.67 ± 4.73 / 729.00 ± 14.93 |
| `U0` | 19, 20, 16 | 857,773 | 51.26 ± 3.03 | 731.33 ± 17.62 (729, 715, 750) | 39.38% | **1,023.00 ± 18.36** (1010, 1015, 1044) | **55.09%** | 193.00 ± 11.00 / 280.00 ± 7.21 | 538.33 ± 14.15 / 743.00 ± 13.23 |
| `U1` | 18, 20, 21 | 859,901 | 52.90 ± 2.21 | 728.33 ± 11.37 (725, 719, 741) | 39.22% | 1,008.33 ± 13.65 (1023, 1006, 996) | 54.30% | 195.00 ± 10.54 / 272.67 ± 6.03 | 533.33 ± 7.64 / 735.67 ± 19.55 |

### 3.3 Stage 1 Paired Customer-Clustered 95% Intervals vs `C0` (`percentage points`)

| Variant | Nested `Δ Recall@25` (low, mean, high) | Nested `Δ Recall@50` (low, mean, high) | Per-Seed `Δ Recall@25` (17 / 42 / 2026) | Per-Seed `Δ Recall@50` (17 / 42 / 2026) |
|---|---|---|---|---|
| `T1` (seed 42) | [-1.40, -0.27, +0.91] pp | [-0.89, +0.21, +1.39] pp | — / -0.27 pp / — | — / +0.21 pp / — |
| `T2` (seed 42) | [-1.93, -0.25, +1.51] pp | [-2.80, -1.09, +0.50] pp | — / -0.25 pp / — | — / -1.09 pp / — |
| `A` (seed 42) | [-1.52, +0.23, +1.88] pp | [-1.99, -0.21, +1.57] pp | — / +0.23 pp / — | — / -0.21 pp / — |
| `F_emb` (seed 42) | [-2.06, -0.16, +1.75] pp | [-1.77, +0.16, +2.21] pp | — / -0.16 pp / — | — / +0.16 pp / — |
| `F_txt` (seed 42) | [-1.35, +0.11, +1.56] pp | [-1.52, -0.05, +1.48] pp | — / +0.11 pp / — | — / -0.05 pp / — |
| `P_no_unit` (seed 42) | [-1.27, +0.39, +2.08] pp | [-2.81, -1.04, +0.66] pp | — / +0.39 pp / — | — / -1.04 pp / — |
| `P` (3 seeds) | [-0.76, +0.79, +2.40] pp | [-0.11, +2.18, +4.40] pp | -0.11 / +0.95 / +1.53 pp | +1.58 / +0.65 / +4.27 pp |
| `U0` (3 seeds) | [-1.08, +0.65, +2.37] pp | **[+0.00, +2.31, +4.62] pp** | -0.32 / +0.42 / +1.88 pp | +1.83 / +0.55 / +4.58 pp |
| `U1` (3 seeds) | [-1.16, +0.48, +2.23] pp | [-0.50, +1.54, +3.31] pp | -0.52 / +0.62 / +1.39 pp | +2.51 / +0.00 / +1.98 pp |

---

## 4. Downstream Stage 2 Results (`dot/100`, `quota100_50`, 3 Seeds Each)

### 4.1 Aggregate & Per-Seed Downstream Metrics

| Config | Qual | Params (Trainable) | Fit (s) Mean ± SD | New Hits@10 Mean ± SD (17, 42, 2026) | New Recall@10 | Overall Recall@10 Mean ± SD | Overall NDCG@10 Mean ± SD | Slate New Targets |
|---|---:|---|---:|---|---|---|---|---:|
| `C0-dot100-control` | 3/3 | 866,313 (664,329) | 147.84 ± 12.58 | 210.00 ± 8.19 (212, 217, 201) | 11.31% | 0.761272 ± 0.002866 | 0.592877 ± 0.000847 | 1,250.3 ± 15.6 |
| `C0-dot100-dynamic_features` | 3/3 | 1,017,482 (815,498) | 154.81 ± 8.70 | 234.67 ± 11.72 (248, 230, 226) | 12.64% | **0.762675 ± 0.000270** | 0.595475 ± 0.000693 | 1,250.3 ± 15.6 |
| `P-dot100-control` | 3/3 | 866,313 (664,329) | 152.50 ± 4.61 | 196.67 ± 7.09 (203, 189, 198) | 10.59% | 0.759312 ± 0.001769 | 0.592351 ± 0.000420 | 1,268.3 ± 20.1 |
| `P-dot100-dynamic_features` | 3/3 | 1,017,482 (815,498) | 155.13 ± 13.40 | **236.00 ± 3.61** (235, 233, 240) | **12.71%** | 0.762477 ± 0.002036 | **0.595793 ± 0.000945** | 1,268.3 ± 20.1 |

Per-seed detail (`10/10` epochs completed for every run):
- `C0-dot100-control`: seed 17 ep 7 (`new=212, rec=0.763614, ndcg=0.592694`), seed 42 ep 10 (`new=217, rec=0.762125, ndcg=0.593801`), seed 2026 ep 7 (`new=201, rec=0.758076, ndcg=0.592137`).
- `C0-dot100-dynamic_features`: seed 17 ep 10 (`new=248, rec=0.762369, ndcg=0.596269`), seed 42 ep 5 (`new=230, rec=0.762777, ndcg=0.595165`), seed 2026 ep 7 (`new=226, rec=0.762879, ndcg=0.594992`).
- `P-dot100-control`: seed 17 ep 8 (`new=203, rec=0.759281, ndcg=0.592816`), seed 42 ep 9 (`new=189, rec=0.757558, ndcg=0.592001`), seed 2026 ep 6 (`new=198, rec=0.761096, ndcg=0.592235`).
- `P-dot100-dynamic_features`: seed 17 ep 8 (`new=235, rec=0.760212, ndcg=0.594702`), seed 42 ep 6 (`new=233, rec=0.764156, ndcg=0.596367`), seed 2026 ep 8 (`new=240, rec=0.763063, ndcg=0.596311`).

### 4.2 Downstream Paired Customer-Clustered 95% Intervals vs `C0`

| Comparison | Nested `Δ Recall@10` | Nested `Δ NDCG@10` | Nested `Δ New Recall@10` | Per-Seed `Δ New Recall@10` (17 / 42 / 2026) |
|---|---|---|---|---|
| `P-dot100-control` vs `C0-dot100-control` | [-0.00687, -0.00200, +0.00295] | [-0.00282, -0.00057, +0.00163] | [-0.01825, -0.00714, +0.00309] | -0.00481 / -0.01509 / -0.00163 |
| `P-dot100-dynamic_features` vs `C0-dot100-dynamic_features` | [-0.00358, -0.00029, +0.00293] | [-0.00247, +0.00027, +0.00266] | [-0.00921, +0.00044, +0.01058] | -0.00708 / +0.00157 / +0.00721 |

---

## 5. Findings & Recommendations

1. **Text Rewrites & HTML Descriptions (`T1`, `T2`):** Replacing `c0_s0` text with deduplicated structured metadata (`T1`) or appending cleaned HTML descriptions (`T2`) did not beat `C0` at `@25` on seed 42 (`702` and `703` vs `707`), and `with_description_nonempty` (`514` targets) dropped slightly (`187` for `T2` vs `192` for `C0`).
2. **Static Product Embeddings (`P`):** Combining `t2_fam` text with 16D categorical embeddings for `category`, `subcategory`, `business_line`, `brand`, `tier`, `unit`, and `country_of_origin` (`P`) produced the highest 3-seed Stage 1 unseen discovery@25 (`733.67 ± 9.02`, `+14.67` over `C0`) and `+40.33` unseen hits@50 (`1,020.33 ± 16.26` vs `980.00 ± 23.26`), with `unit` contributing clearly (`P` `725` vs `P_no_unit` `715` @25, `1,017` vs `985` @50 on seed 42). However, the 3-seed customer-clustered 95% interval for `@25` still spans zero (`[-0.76, +0.79, +2.40] pp`), and `@50` is borderline (`[-0.11, +2.18, +4.40] pp`).
3. **Customer Static Embeddings (`U0`, `U1`):** Adding `sales_branch` (`U0`) kept `@25` flat vs `P` (`731.33 ± 17.62`) while slightly edging `@50` (`1,023.00 ± 18.36`, nested `@50` CI `[+0.00, +2.31, +4.62] pp`); adding `vat_status` (`U1`) slightly reduced both (`728.33` / `1,008.33`).
4. **Downstream `dot/100` Impact:**
   - Under `dot100-control`, `P` underperformed `C0` (`196.67 ± 7.09` vs `210.00 ± 8.19` new hits@10; Recall@10 `0.759312` vs `0.761272`), likely because the frozen Stage 2 control ranker retains `C0`-coupled projections/biases while `P`'s 100-item slate brings slightly more unseen candidates (`1,268.3` vs `1,250.3`).
   - Under `dot100-dynamic_features`, `P` (`236.00 ± 3.61` new hits@10, `NDCG@10 = 0.595793 ± 0.000945`) and `C0` (`234.67 ± 11.72`, `0.595475 ± 0.000693`) both pass all quality floors (`3/3` seeds) with virtually identical customer-clustered means (`Δ new = +0.04 pp`, 95% CI `[-0.92, +1.06] pp`).
   - **Decision:** Keep `promoted = false`. Retain `P` as an experimental static-enrichment candidate (and `snapshot_unverified` metadata), not a promoted replacement, until historical effective-date metadata governance and joint Stage 1/Stage 2 calibration are verified.

---

## 6. Reproduction Commands

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase prep
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase c0
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase screen
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase repeat
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase downstream
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/summarize.py
```

---

## 7. Clean Static Towers + Transferred 6-Stat Scoring MLP (`seed=42`)

**Namespace:** `checkpoints/static-enrichment-v1/clean_v2/runs/` (preserves all v1 artifacts untouched)  
**Evidence:** [`experiments/static_enrichment/clean_v2_results.json`](experiments/static_enrichment/clean_v2_results.json), [`experiments/static_enrichment/clean_v2_summary.json`](experiments/static_enrichment/clean_v2_summary.json), [`experiments/static_enrichment/clean_v2_COMPARISON.md`](experiments/static_enrichment/clean_v2_COMPARISON.md)

### 7.1 Architectural Changes

- **Zero Changeable Tower Inputs (Stage 1):**
  - **`CustomerTower` (`num_tabular_features=0`):** Deleted all 6 changing customer statistics (`distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`) from `h_tab`. `tabular_proj[0]` takes strictly static IDs (`city_id`, `region_id`, `industry_id` -> `48D`, plus `sales_branch` `16D` -> `64D` in `U0_*`), fused with `h_text` and `h_hist` (last 32 purchases) into precomputed `u_c`.
  - **`ProductTower` (`num_demand_features=12`):** Deleted the 6 demand columns (`demand[:, :6]`) from the tower input layer, leaving ONLY the `12D` static `container_type` one-hot buckets (`demand[:, 6:]`).
  - **Text (`t1_fam`):** Uses `T1 + product_family` deduplicated text with **no `описание.csv` descriptions**.
- **Transferred 6 Customer Stats & Precomputed MLP (Stage 2, `quota100_50`, 10 epochs, `seed=42`):**
  - The 6 point-in-time customer stats (log1p-scaled strictly on training `< 2025-10-01`, quantized `float16 -> float32`) are fed directly into the downstream scoring MLP (`41 + 6 = 47` scalar features).
  - **`best_clean-dot100_47`** (`TransferredStatsDynamicRanker`): `UPS-CANR` `dot100_41` (`425D` head) expanded to `431D` (`397 + 28 + 6`, zero-initialized on the 6 transferred columns).
  - **`best_clean-precomputed_mlp_47`** (`PrecomputedEmbeddingMLPRanker`): Lightweight `184,833`-param MLP (`560 -> 256 -> 128 -> 64 -> 1`) using ONLY precomputed `u_c`, precomputed `v_p`, `u_c * v_p`, `|u_c - v_p|`, `u_c · v_p`, and all `47` changeable features — NO `h_text`/`h_tab` recomputation and NO 33-token cross-attention.

### 7.2 Clean Stage 1 Results (`30` epochs verified, `seed=42`, `1,857` new targets)

| Variant | Ep | Params | Fit (s) | Hits@25 | Recall@25 | Hits@50 | Recall@50 | With-Desc @25/@50 (514) | No-Desc @25/@50 (1,343) | Δ@25 vs `C0_clean` (95% CI) | Δ@50 vs `C0_clean` (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `C0_clean` | 13 | 822,829 | 50.71 | 710 | 38.23% | 972 | 52.34% | 188 / 264 | 522 / 708 | — | — |
| `P_t1_fam_clean` (`best_clean`) | 21 | 853,309 | 49.62 | **749** | **40.33%** | **1,037** | **55.84%** | 190 / 279 | 559 / 758 | **[+0.11, +2.07, +4.09] pp** | **[+1.71, +3.48, +5.38] pp** |
| `U0_t1_fam_clean` | 26 | 855,469 | 49.36 | 737 | 39.69% | 1,009 | 54.33% | 206 / 281 | 531 / 728 | [-0.59, +1.45, +3.35] pp | [-0.05, +1.99, +4.06] pp |
| `U0_t1_fam_nosubcat` | 20 | 850,637 | 50.50 | 719 | 38.72% | 1,015 | 54.66% | 205 / 285 | 514 / 730 | [-1.40, +0.51, +2.48] pp | [+0.38, +2.33, +4.29] pp |

- **`subcategory` embedding ablation (`U0_t1_fam_clean` vs `U0_t1_fam_nosubcat`):** Keeping the `16D` `subcategory` embedding yielded `+18` unseen hits@25 (`737` vs `719`; customer-clustered 95% CI `[-0.65, +0.93, +2.55] pp`) and `-6` hits@50 (`1,009` vs `1,015`; 95% CI `[-1.99, -0.33, +1.36] pp`).
- **`P_t1_fam_clean` vs `C0_clean`:** Removing the 6 changing customer stats and 6 product demand inputs while using `t1_fam` (no `описание.csv`) + 7 static product attribute embeddings delivered a strictly positive seed-42 gain of `+39` unseen hits@25 and `+65` unseen hits@50.

### 7.3 Downstream Stage 2 Results (`quota100_50`, `10` epochs verified, `seed=42`)

| Config | Ep | Qual | Params (Trainable) | Fit (s) | New Hits@10 | New Recall@10 | Recall@10 | NDCG@10 | Slate New Targets |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `C0_clean-dot100_41` | 8 | True | 1,017,482 (815,498) | 134.38 | **241** | **12.98%** | **0.762081** | **0.597476** | 1,230 |
| `P_t1_fam_clean-dot100_41` | 7 | True | 1,017,482 (815,498) | 134.52 | 239 | 12.87% | 0.761552 | 0.593200 | 1,267 |
| `P_t1_fam_clean-dot100_47` | 7 | True | 1,019,018 (817,034) | 143.44 | 233 | 12.55% | 0.760275 | 0.592892 | 1,267 |
| `P_t1_fam_clean-precomputed_mlp_47` | 1 | **False** | 184,833 (184,833) | 59.87 | 198 | 10.66% | 0.745342 | 0.575605 | 1,267 |

Paired customer-clustered 95% intervals (`seed=42`):
- `P_t1_fam_clean-dot100_41` vs `C0_clean-dot100_41`: `Δ Recall@10 = [-0.00517, -0.00043, +0.00415]`, `Δ NDCG@10 = [-0.00702, -0.00427, -0.00140]`, `Δ New Recall@10 = [-0.01041, -0.00107, +0.00839]`.
- `P_t1_fam_clean-dot100_47` vs `P_t1_fam_clean-dot100_41`: `Δ Recall@10 = [-0.00432, -0.00124, +0.00159]`, `Δ NDCG@10 = [-0.00241, -0.00030, +0.00210]`, `Δ New Recall@10 = [-0.01131, -0.00331, +0.00479]`.
- `P_t1_fam_clean-precomputed_mlp_47` vs `P_t1_fam_clean-dot100_47`: `Δ Recall@10 = [-0.02063, -0.01501, -0.00924]`, `Δ NDCG@10 = [-0.02165, -0.01733, -0.01318]`, `Δ New Recall@10 = [-0.03363, -0.01870, -0.00485]`.

### 7.4 Conclusions (`clean_v2`, `seed=42`)

1. **Stage 1 clean static towers work cleanly:** Removing the 6 changeable customer stats and 6 demand dims preserves baseline quality (`C0_clean` `710@25`) and `P_t1_fam_clean` reaches `749@25` / `1,037@50` without `описание.csv` descriptions.
2. **Transferring the 6 customer stats to the Stage 2 `dot100` MLP (`47` vs `41`) is neutral-to-slightly-negative:** `UPS-CANR` already models customer profile statistics in its statistics-biased cross-attention (`dot100_41` `239` vs `dot100_47` `233`).
3. **Lightweight precomputed-embedding MLP (`precomputed_mlp_47`) is insufficient:** Replacing `UPS-CANR`'s 33-token cross-attention with an MLP over precomputed `(u_c, v_p)` + `47` features trains faster (`59.9s` vs `143.4s`) but drops `-0.0173` NDCG@10 and `-0.0150` Recall@10, failing both reference quality floors (`Recall@10 >= 0.7558833`, `NDCG@10 >= 0.5918929`).

### 7.5 Reproduction Commands (`clean_v2`)

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase clean_v2
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/summarize.py --campaign clean_v2
```

---

## 8. Compact Stage 1 Selector (`stage1_mlp_v3`) & Stage 2 Isolation Audit (`seed=42`, Validation-Only)

**Namespace:** `checkpoints/static-enrichment-v1/stage1_mlp_v3/runs/` (all `v1` and `clean_v2` checkpoints/caches preserved untouched; `promoted = false`, `heldout_access = false`)  
**Evidence:** [`experiments/static_enrichment/stage1_mlp_v3_results.json`](experiments/static_enrichment/stage1_mlp_v3_results.json), [`experiments/static_enrichment/stage1_mlp_v3_summary.json`](experiments/static_enrichment/stage1_mlp_v3_summary.json), [`experiments/static_enrichment/stage1_mlp_v3_COMPARISON.md`](experiments/static_enrichment/stage1_mlp_v3_COMPARISON.md)

### 8.1 Architecture Alignment & Contracts

1. **Clean Static Stage 1 Towers (128D `u_c` and `v_p`):**
   - `CustomerTower`: **`num_tabular_features=0`** (no changing customer stats), retaining `h_text`, static `h_tab`, and `h_hist` (last 32 purchases). **`U0_t1_fam_clean` (`64D` tabular: `48D` `city`/`region`/`industry` + `16D` `sales_branch`)** satisfies the user `sales_branch` requirement; `P_t1_fam_clean` (`48D`, no `sales_branch`) is retained strictly as a benchmark comparison.
   - `ProductTower`: **`num_demand_features=12`** (`container_type` one-hot only, zero demand stats), `t1_fam` deduplicated text (no `описание.csv` descriptions, `product_family` in text), and seven `16D` embeddings (`category`, `subcategory`, `business_line`, `brand`, `tier`, `unit`, `country_of_origin`).
2. **Stage 1 Full-Pool `CompactStage1Selector` (scores full eligible pool before `quota100_50` slate selection):**
   - Trained for `10` epochs (`batch_size=64`, `lr=1e-3`, `weight_decay=1e-4`, eligible-pool `BCEWithLogitsLoss(pos_weight=20.0)`), checkpoint selected by `(new_hits@50, ndcg@10, recall@10)`.
   - **`uv_47` (`303D`, `47,617` params):** `[u(128), v(128), features_41(41), customer_stats_6(6)] -> 128 -> 64 -> 1`.
   - **`uv_47_ref` (`304D`, `47,745` params):** `[u(128), v(128), features_41(41), customer_stats_6(6), reference_score/10(1)] -> 128 -> 64 -> 1`.
   - `customer_stats_6` contains all 6 transferred point-in-time stats (`distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`), `float16 -> float32` quantized consistently.
3. **Stage 2 Ranker (`UPS-CANR`):** Retained (`CandidateContextRanker(dynamic_features)`, `quota100_50` 100-item slates) to produce final top-10 rankings; BOTH discovery-selected (`best.pt`) and NDCG-selected (`best_ndcg.pt`) checkpoints are saved and replayed.

### 8.2 Latent Bug Fixes & `LEGACY_PROFILE_FEATURES` Accounting

- **Item D (`_make_downstream_ranker` bug fix):** Previously checked `source.shape[1] >= 48 + len(ref_config["profile_features"])` (`54`), which evaluated `True` for `U0_t1_fam_clean` (`64D = 48D + 16D sales_branch`), causing the last 6 coordinates of `sales_branch` weights to be miscopied into `UPS-CANR` profile stats. Fixed via `_resolve_stage1_customer_tabular_contract` in [`recommender/experiments/static_enrichment/train.py`](recommender/experiments/static_enrichment/train.py) to inspect explicit metadata (`customer_tabular_features`, `customer_profile_features`, `customer_embeddings`), validate exact widths, reject ambiguous checkpoints missing metadata with `ValueError`, and leave `tabular_proj[0].weight[:, 48:]` untouched when `customer_tabular_features=0`.
- **Item E (`LEGACY_PROFILE_FEATURES` overlap):** `UPSCANRRanker`'s 8-feature `LEGACY_PROFILE_FEATURES` (`total_purchases`, `total_quantity`, `distinct_products`, `distinct_orders`, `distinct_categories`, `days_since_first_order`, `days_since_last_order`, `median_order_gap`) overlaps **5 of the 6** transferred stats and omits `median_purchase_quantity`.

### 8.3 Stage 1 Clean Towers vs `CompactStage1Selector` (Full Pool & `quota100_50` Coverage, `6,191` Val Queries)

Note on denominators: all `6,191` validation transitions have `>= 100` eligible rolling SKUs; total new targets = `1,857` (used by `new_recall@10` and Stage 1 `@25/@50`), of which `1,810` belong to the rolling eligible pool (`(target & unseen).sum()`) and `47` are uneligible cold targets (`target & ~allowed`).

| Stage 1 Scorer | Input | Ep | `sales_branch` | Unseen@25 | Unseen@50 | Recall@10 | NDCG@10 | Slate New Targets (Elig Cov / All Cov) | Slate Overall Cov |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C0_clean + dot` | 128D dot | 13 | No (`48D`) | 710 | 972 | 0.34870 | 0.20980 | 1,230 (67.96% / 66.24%) | 93.60% |
| `P_t1_fam_clean + dot` | 128D dot | 21 | No (`48D`) | **749** | **1,037** | 0.35373 | 0.21049 | 1,267 (70.00% / 68.23%) | 94.02% |
| `U0_t1_fam_clean + dot` | 128D dot | 26 | **Yes (`16D`, total `64D`)** | 737 | 1,009 | 0.37333 | 0.22104 | **1,274 (70.39% / 68.61%)** | 93.99% |
| `U0 + CompactStage1Selector (uv_47)` | 303D MLP | 1 | Yes (`16D` in `u`) | 737 | 1,012 | **0.75052** | **0.57713** | 1,254 (69.28% / 67.53%) | **94.04%** |
| `U0 + CompactStage1Selector (uv_47_ref)` | 304D MLP | 1 | Yes (`16D` in `u`) | 703 | 993 | 0.74662 | 0.57582 | 1,251 (69.12% / 67.37%) | 93.96% |

**Feature-group zero ablations on `CompactStage1Selector` (epoch 1):**
- **`uv_47` (`303D`):** Full (`R@10=0.75052`, `NDCG=0.57713`, `unseen@25=737`, `unseen@50=1,012`, `slate_new=1,254`) vs zero `features_41` (`R@10=0.18806`, `NDCG=0.09200`, `unseen@25=547`, `unseen@50=839`, `slate_new=1,088`) vs zero `customer_stats_6` (`R@10=0.75275`, `NDCG=0.57460`, `unseen@25=723`, `unseen@50=1,011`, `slate_new=1,244`) vs zero both 47 (`R@10=0.21095`, `NDCG=0.10203`, `unseen@25=572`, `unseen@50=841`, `slate_new=1,084`).
- **`uv_47_ref` (`304D`):** Full (`R@10=0.74662`, `NDCG=0.57582`, `unseen@25=703`, `unseen@50=993`, `slate_new=1,251`) vs zero `features_41` (`R@10=0.37381`, `NDCG=0.22509`, `unseen@25=676`, `unseen@50=963`, `slate_new=1,205` — falls back to the dot anchor) vs zero `customer_stats_6` (`R@10=0.74785`, `NDCG=0.57560`, `unseen@25=702`, `unseen@50=1,004`, `slate_new=1,257`).
- By epoch 10, Stage 1 selector overall ranking increases (`uv_47` `NDCG=0.58683`, `uv_47_ref` `NDCG=0.58832`) while unseen recall@50 declines (`946` and `933`) as repeat purchases dominate eligible-pool BCE.

### 8.4 Stage 2 End-to-End `UPS-CANR` & Regression Isolation Diagnostics (`quota100_50`, `seed=42`)

| Config | Selector / Init / Slate | Ep (Disc/NDCG) | R@5 (Disc) | R@10 (Disc) | NDCG@10 (Disc) | Repeat R@10 | New@10 (Disc) | Cond New | Slate New | R@10 (NDCG) | NDCG@10 (NDCG) | New@10 (NDCG) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_clean-dot100_41` (v2) | `C0_clean` dot / `C0` / `C0` | 8 / 8 | 0.65251 | 0.76208 | **0.59748** | 0.88407 | 241 (12.98%) | **19.59%** | 1,230 | 0.76208 | **0.59748** | 241 |
| `U0-dot100_41` | `U0` dot / `U0` (fixed) / `U0` | 6 / 6 | 0.65343 | 0.76457 | 0.59606 | 0.88775 | 233 (12.55%) | 18.29% | **1,274** | 0.76457 | 0.59606 | 233 |
| `U0-mlp303_100_41` | `uv_47` (303D) / `U0` / `uv_47` | 7 / 6 | 0.65015 | 0.75973 | 0.59461 | 0.88348 | 226 (12.17%) | 18.02% | 1,254 | 0.76377 | 0.59665 | 224 |
| `U0-mlp304_100_41` | `uv_47_ref` (304D) / `U0` / `uv_47_ref` | 6 / 6 | **0.65525** | **0.76516** | 0.59667 | **0.88859** | 233 (12.55%) | 18.63% | 1,251 | 0.76516 | 0.59667 | 233 |
| `P-dot100_41` | `P` dot / `P` / `P` | 7 / 6 | 0.64782 | 0.76065 | 0.59311 | 0.88300 | 234 (12.60%) | 18.47% | 1,267 | 0.76342 | 0.59475 | 230 |
| `P-dot100_41-init_c0` | `P` dot / `C0` init / `P` | 7 / 6 | 0.65068 | 0.76189 | 0.59365 | 0.88419 | **243 (13.09%)** | 19.18% | 1,267 | 0.76469 | 0.59542 | 239 |
| `P-c0_slate_init_c0-41` | `C0` dot / `C0` init / `C0` (`P` vecs) | 10 / 6 | 0.65357 | 0.75920 | 0.59547 | 0.88312 | 233 (12.55%) | 18.94% | 1,230 | **0.76595** | 0.59606 | 231 |

**Diagnosis Findings (Items A–C):**
- **Item A (Selection objective):** In `clean_v2` `P_t1_fam_clean-dot100_41`, discovery selection picked epoch 7 (`NDCG=0.59320`, `R@10=0.76155`, `new=239`) while NDCG peaked at epoch 6 (`NDCG=0.59608`, `R@10=0.76413`, `new=227`, weights unsaved in `clean_v2`). In our v3 replay (`P-dot100_41`), saving BOTH checkpoints confirms epoch 6 (`NDCG=0.59475`, `R@10=0.76342`) vs epoch 7 (`NDCG=0.59311`, `R@10=0.76065`). Meanwhile `C0_clean-dot100_41`, `U0-dot100_41`, and `U0-mlp304_100_41` all unify discovery and NDCG peaks on the same epoch (`8`, `6`, `6`).
- **Item B (Conversion vs coverage):** Both `U0` (`1,274` slate new targets) and `P` (`1,267`) widen Stage 1 slate new-target coverage over `C0_clean` (`1,230`), yet Stage 2 converts a smaller fraction to Top-10 (`18.29%` for `U0-dot100_41`, `18.63%` for `U0-mlp304_100_41`, vs `19.59%` for `C0_clean-dot100_41`).
- **Item C (Copied init vs slates vs vectors):** On `P_t1_fam_clean`, holding `P` slates (`1,267`) and `P` vectors fixed while swapping ONLY Stage 1 copied initialization from `P` to `C0_clean` (`P-dot100_41-init_c0`) lifts discovery hits `+9` (`234 -> 243`, exceeding `C0_clean`'s `241`) and NDCG (`0.59311 -> 0.59365` disc; `0.59475 -> 0.59542` NDCG). Swapping slates back to `C0_clean` (`P-c0_slate_init_c0-41`) recovers `NDCG@10` (`0.59606`) and highest `R@10` (`0.76595` NDCG ep) at the cost of losing 37 slate new targets (`1,267 -> 1,230`). Paired customer-clustered 95% CIs within `seed=42` (`U0-mlp304_100_41` vs `U0-dot100_41`: `ΔR@10=[-0.00259, +0.00053, +0.00364]`, `ΔNDCG@10=[-0.00157, +0.00057, +0.00284]`, `ΔNewR@10=[-0.00860, 0.0, +0.00809]`) show query-level uncertainty only (not multi-seed variance) and cross zero.

### 8.5 Warm Single-Request Latency (`U0_t1_fam_clean` Cached Vectors, 50 repeats)

| Device | Pipeline | E2E Median (ms) | E2E p95 (ms) | Selector (ms) | Select100 (ms) | Ranker (ms) |
|---|---|---:|---:|---:|---:|---:|
| CPU | `U0-dot100_41` | **1.014** | 1.137 | 0.015 | 0.039 | 0.860 |
| CPU | `U0-mlp303_100_41` | 1.478 | 1.973 | 0.415 | 0.045 | 0.929 |
| CPU | `U0-mlp304_100_41` | 1.517 | 2.002 | 0.453 | 0.046 | 0.935 |
| MPS | `U0-dot100_41` | **4.117** | 10.638 | 0.269 | 0.965 | 2.636 |
| MPS | `U0-mlp303_100_41` | 4.197 | **9.001** | 0.491 | 0.994 | 2.644 |
| MPS | `U0-mlp304_100_41` | 4.174 | 10.603 | 0.573 | 1.067 | 2.623 |

### 8.6 Reproduction Commands (`stage1_mlp_v3`)

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase stage1_mlp_v3
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/summarize.py --campaign stage1_mlp_v3
```
