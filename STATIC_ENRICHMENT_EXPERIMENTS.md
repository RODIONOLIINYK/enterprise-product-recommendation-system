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
