# Roadmap v2 Final Report (`experiments/roadmap_v2`)

The seed-42 [warehouse exclusion follow-up](#stage-5-follow-up-without-warehouse-inputs) tested three Stage 5 recipes. The financial and manager recipe had the highest NDCG@10 and new-formulation hit count at the NDCG-selected checkpoints. The earlier warehouse results below document the original campaign.

> **Strict Validation-Only Policy:** All training ran strictly `< 2025-10-01` (`26,253` transitions); all evaluation ran strictly on `2025-10-01..2026-02-18` (`6,191` baskets, `10,276` SKU targets = `1,857` new-SKU targets [`455` pack-size switches + `1,402` truly new `base_name`s]; `9,809` unique `base_name` targets = `1,340` truly new `base_name` targets). **Held-out Test (`>= 2026-02-19`) was never accessed.** All preserved reference checkpoints and caches passed SHA-256 verification unchanged; full repository test suite passes (`622 passed, 6 skipped, 70 subtests passed`).

---

## Executive Decision (North-Star Selection Criterion)

Per the project's North-Star criterion (**consistently higher NDCG@10, higher truly new `base_name` discovery excluding same-`base_name` pack switches, and stronger overall Recall@10 across 3 seeds `42, 43, 44`**):

1. **Primary Recommended Architecture (Product-Formulation / `base_name` Level): Stage 5 Canonical `base_name` Pipeline (`585` entities + `60d aux` + `top4_3acde`)**
   - Unifies `841` packaging-variant SKUs into `585` canonical `base_name` product bodies across **all stages** (pack-agnostic Two-Tower → `rolling_observed_v2` Stage 1 MLP → `base_name`-indexed Stage 2 `UPS-CANR` Transformer) with `60-day` training-only auxiliary head (stripped at inference) and `3a_financial + 3c_manager + 3d_warehouse + 3e_parsed_specs`.
   - **Beats both the pre-roadmap baseline (`pointwise425_ndcg`) and our best SKU-level model (`top4_3acde`) on 3 out of 3 seeds (`42, 43, 44`) on all three North-Star metrics:**
     - `bn_NDCG@10`: **`0.65282 ± 0.00047`** (`+0.00270` over SKU `top4_3acde` collapsed `0.65012 ± 0.00048`; `+0.00077` over pre-roadmap `0.65205`)
     - `bn_Recall@10`: **`82.93% ± 0.08%`** (`+0.33 pp` over SKU `top4_3acde` `82.60% ± 0.17%`; `+0.09 pp` over pre-roadmap `82.84%`)
     - `true_new_bn@10` *(truly brand-new formulations, zero pack switches)*: **`220.3 ± 5.8 / 1,340`** (NDCG ckpt; **`228.0 ± 9.0`**, peak **`237`** on Discovery ckpt — **`+16` to `+25` more truly new product formulations** than pre-roadmap `212`)
     - Stage 1 100-item slate coverage: **`97.43%`** of all `base_name` targets and **`83.4–83.5%`** (`1,118–1,119 / 1,340`) of truly new `base_name` targets, with a **30% smaller product universe (`585` vs `841`)**.
2. **Best Exact-SKU / Pack-Size Architecture (whenever concrete pack-level `product_id`s are required in Top-10): SKU-Level `top4_3acde` (`sku_plus_base_name` + `60d aux` + `3a_financial + 3c_manager + 3d_warehouse + 3e_parsed_specs`)**
   - Beats the pre-roadmap SKU champion (`pointwise425_ndcg`: `R@5 66.01%`, `R@10 77.10%`, `NDCG@10 0.60223`, `New@10 255`) on **3 out of 3 seeds (`42, 43, 44`) on every metric**:
     - `Recall@5`: **`66.29% ± 0.13%`** (`+0.28 pp`)
     - `Recall@10`: **`77.43% ± 0.12%`** (`+0.33 pp`)
     - `NDCG@10`: **`0.60263 ± 0.00022`** (`+0.00040`)
     - `New@10`: **`311.0 ± 4.0 / 1,857`** (**`+56` hits average** = `178.3` pack-switches + `132.7` truly new `base_name`s).

---

## Stage-by-Stage Progression (Full `6,191` Validation Baskets)

### Stage 0 — Point-in-Time Causality Audit & Rolling Baseline
- Verified empirically that `100%` of checked multi-date customers (`168/168` train, `139/139` val) receive strictly date-varying `CustomerTower u(c, t)` and cumulative `stats6(c, t)` per `(customer, date)` instance, while `ProductTower v` disables demand (`disabled_v1`).
- Verified `0.0` exact diff on all `26,253` training rows between frozen vs. `rolling_observed_v2`, while switching validation features to `rolling_observed_v2` immediately lifted Stage 1 MLP (`Recall@10 75.41% → 75.85%`, `New@10 194 → 206`, `New@25 726 → 757`).

### Stage 1 — `base_name` 3-Way Ablation (`seed=42`, [stage1_base_name_summary.json](stage1_base_name_summary.json))
- Audit revealed **`455 / 1,857` (`24.5%`)** of validation "new-SKU" targets are pack-size switches of a previously purchased `base_name`.
- **Stage 1 MLP:** `sku_plus_base_name` beat `sku_only` significantly on every primary metric (`R@5 +0.98 pp`, `R@10 +0.58 pp`, `NDCG@10 +0.0111`, `New@10 +60` [`219 → 279`], `New@25 +89` [`766 → 855`]).
- **Stage 2 Transformer (E2E own slates):** `sku_plus_base_name` achieved `R@5 66.37%`, `R@10 77.38%`, `NDCG@10 0.6023`, `New@10 304` (and `311` on fixed slates), lifting pack-switch slate coverage `382 → 428 / 455`.
- `base_name_only` dropped `-3 pp` on SKU-level Recall/NDCG because removing exact-SKU history ties 4–5 pack variants of the same oil at Top-10 — directly confirming why Stage 5 needed to collapse the product entity itself.

### Stage 2 — Multi-Task 60-Day Auxiliary Head & History-SKU Dropout (`seed=42`, [stage2_augmentation_summary.json](stage2_augmentation_summary.json))
- Leak-free 60-day auxiliary labels (`[t, min(t+60d, TRAIN_CUTOFF))`) multiplied unseen training discovery positives **`3.34×`** (`0.420 → 1.402` per query).
- **`2A` (60d aux head `λ=0.3`, stripped at inference)** won cleanly on both stages: lifted Stage 1 slate coverage to `9,740 / 10,276` (`94.78%`, `1,331` new targets) and lifted Stage 2 to `NDCG@10 = 0.6025`, `R@10 = 77.39%`, `New@10 = 315` with zero inference overhead.
- **`2B/2C` (History-SKU Dropout `p=0.30`)** boosted Stage 1 `New@50` (`1,113`) and Stage 2 `true_new_bn` (`127–140`), but degraded Stage 2 `NDCG@10` (`-0.009`) and `R@10` (`-0.7 pp`) by mis-calibrating the repeat-vs-unseen threshold; not kept in Stage 2.

### Stage 3 — Causal Commercial, Rolling-Manager, Warehouse & Parsed-Spec Features (`seed=42`, [stage3_commercial_summary.json](stage3_commercial_summary.json))
- All 5 new strictly-causal feature groups beat control on Stage 1 `New@50` and Stage 2 `NDCG@10`:
  - **`3d_warehouse`** (primary warehouse 30d/90d availability): S2 `NDCG@10 = 0.6040`, `R@10 = 77.54%`, `129` true-new.
  - **`3c_manager`** (rolling 90d current-manager behavioral shares, zero static manager IDs): S1 `New@50 = 1,113 (+26)`, S2 `NDCG@10 = 0.6037`, `R@5 = 66.35%`, Disc ckpt `316` (`132` true-new).
  - **`3a_financial`** (payment deferrals, prepaid/cash liquidity, strictly `< t` `ДатаПолнойОплати`, overdue unpaid ratio, bonus rate): S2 `R@10 = 77.48%`, `NDCG@10 = 0.6021`.
  - **`3e_parsed_specs`** (viscosity grade, pack size, dominant container, company equipment matching): S2 `R@10 = 77.47%`, Disc ckpt `318` (`131` true-new).
  - **`3b_pricing`** (NAV negative-COGS-corrected margin, unit price, USD cost, spend, price fit): `+13` S1 `New@50`, flat on S2 `R@10` (`-0.08 pp`).
- **Combined End-to-End:** `top4_3acde` (`3a+3c+3d+3e`) beat `all5_3abcde` (`3a+3b+3c+3d+3e`) on slate coverage (`9,745`, `1,338` new), `NDCG@10` (`0.6028`), `R@5` (`66.57%`), and `R@10` (`77.58%`).

### Stage 4 & Stage 5 — 3-Seed Head-to-Head Verification (`seeds 42, 43, 44`, [stage4_multiseed_summary.json](stage4_multiseed_summary.json))

| Model (3-Seed Mean ± Std, `6,191` Val Baskets) | `bn_Recall@10` | `bn_NDCG@10` | `true_new_bn@10` *(of `1,340`)* | Per-Seed `true_new_bn@10` `[42, 43, 44]` |
| :--- | :---: | :---: | :---: | :---: |
| Pre-roadmap `pointwise425_ndcg` (collapsed to `base_name`) | `82.84%` | `0.65205` | `212.0` (`15.82%`) | `[212]` |
| SKU `top4_3acde` (collapsed, NDCG ckpt) | `82.60% ± 0.17%` | `0.65012 ± 0.00048` | `213.0 ± 4.6` (`15.90%`) | `[212, 218, 209]` |
| SKU `top4_3acde` (collapsed, Disc ckpt) | `82.60% ± 0.08%` | `0.65033 ± 0.00091` | `212.3 ± 4.5` (`15.85%`) | `[217, 208, 212]` |
| **Stage 5 Native `base_name` (NDCG ckpt — Winner)** | **`82.93% ± 0.08%`** | **`0.65282 ± 0.00047`** | **`220.3 ± 5.8`** (`16.44%`) | **`[217, 227, 217]`** |
| **Stage 5 Native `base_name` (Disc ckpt — Discovery Winner)** | **`82.73% ± 0.15%`** | **`0.65207 ± 0.00060`** | **`228.0 ± 9.0`** (`17.01%`) | **`[228, 237, 219]`** |

---

## Code, Tests & Reproduction

- **Modules (`recommender/experiments/roadmap_v2/`):** `config.py`, `base_name_features.py`, `stage1_mlp.py`, `stage2_ranker.py`, `augmentation.py`, `stage2_aug_runner.py`, `commercial_features.py`, `parsed_spec_features.py`, `stage5_base_entity.py`, `run_stage0_stage1.py`, `run_stage2.py`, `run_stage3.py`, `run_stage5.py`, `run_stage3_combined_and_stage5.py`, `run_stage4_multiseed.py`.
- **Causality & Parity Unit Tests (`tests/test_roadmap_v2_*.py`):** 10 new tests covering strict `< as_of_date` causality, `< TRAIN_CUTOFF` 60-day label capping, negative NAV COGS sign handling, causal `ДатаПолнойОплати < t` settlement/overdue logic, History-SKU Dropout consistency, and Stage 5 `base_name` collapsing/resolution.
- **Full Test Suite:** `PYTHONPATH=. ./venv/bin/pytest tests/ -q` → `622 passed, 6 skipped, 70 subtests passed`.

## Stage 5 Follow-Up Without Warehouse Inputs

All three variants reuse the seed-42 canonical Two-Tower checkpoint and vectors, train the Stage 1 selector and Stage 2 ranker for 10 epochs each with the 60-day auxiliary loss at weight 0.3, and rebuild the 100-formulation slate for each selector. Evaluation uses the same 6,191 validation baskets and all 1,340 truly new formulation targets. All three fits excluded warehouse features from both stages. The warehouse row is the archived seed-42 run.

| Stage 5 recipe | Recall@5 | Recall@10 | NDCG@10 | Truly new hits@10 | Truly new in slate | Hits@10 at discovery checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60-day auxiliary only | 72.81% | 82.82% | 0.65253 | 204 | 1,112 | 225 |
| + Financial and manager | 72.21% | 82.68% | **0.65423** | **226** | 1,119 | 226 |
| + Financial, manager, and parsed specs | 72.60% | **82.87%** | 0.65383 | 222 | **1,128** | **228** |
| Earlier financial, manager, parsed specs, and warehouse | 72.66% | 82.86% | 0.65235 | 217 | 1,119 | **228** |

The first four metrics use the NDCG-selected checkpoint. Relative to the auxiliary-only control, financial and manager features added 22 new-formulation hits and 1.64 percentage points of new-formulation recall (95% customer-clustered interval +0.30 to +2.98 points). Recall@5 fell 0.60 points (95% interval −1.05 to −0.14 points). The +0.00170 NDCG change had a 95% interval of −0.00079 to +0.00419.

Adding parsed specs covered nine more new formulations but lost four top-10 hits and 0.00040 NDCG at the NDCG-selected checkpoint. Recall@5 rose 0.39 points (95% interval +0.04 to +0.78 points). The lift depends on the metric, and the test adds financial and manager features as a pair. Removing warehouse from the otherwise matched three-feature recipe changed Recall@10 by +0.005 points, NDCG by +0.00148, and new hits by +5. The intervals for all three warehouse-removal differences included zero. These are validation estimates for selected checkpoints from one seed.

The shared training cohort has 26,253 company-date cases. At formulation level it has 43,613 eligible next-basket positives and 119,329 eligible 60-day positives, including 9,332 and 30,268 new-to-company positives, respectively, before slate restriction. Those are labels on the same cases. The first observed basket currently seeds the purchase history and cases start at the second observed basket.

Reproduce the comparison with `PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.compare_stage5_features --train-missing`. The [comparison summary](stage5_warehouse_exclusion_seed42.json) includes both checkpoint selections, paired intervals, and shared Two-Tower checksums.

## First Observed Baskets in the Same Pipeline

The user selected the warehouse-free financial, manager, and parsed-spec recipe as the [Stage 5 reference](stage5_reference.json). This experiment compares that saved selector and ranker against the same architecture and feature schema trained from scratch on a larger cohort. Both arms use the same frozen Two-Tower weights and 60-day auxiliary loss at weight 0.3. Checkpoints are selected on the original returning-customer validation set.

Training grows from 26,253 to 29,237 company-date cases. Of 3,167 first observed training baskets, 114 had fewer than 100 eligible formulations and 69 had no eligible target. The 2,984 added cases have 4,256 eligible next-basket formulation labels. Their histories have zero valid purchase tokens. Financial and manager features take their existing empty-history defaults, and parsed equipment match can still use the company profile. Rolling market statistics and eligibility use only earlier events.

Validation includes the original 6,191 returning-customer cases and a separate cohort of 289 first observed baskets with 354 modeled formulation targets. Of those targets, 349 are eligible on the prediction date. The first-basket recall denominator includes the other five. First observed baskets are defined within this export, and some companies may have bought before the dataset began. Catalogue and company metadata remain snapshot-based.

| Validation cohort | Metric | Pinned reference | Mixed training | Change |
| --- | --- | ---: | ---: | ---: |
| First observed baskets | Recall@10 | 37.71% | 41.25% | +3.55 pp |
| First observed baskets | NDCG@10 | 0.21252 | 0.25397 | +0.04145 |
| First observed baskets | Top-10 targets found | 136 / 354 | 149 / 354 | +13 |
| Returning baskets | Recall@10 | 82.87% | 82.66% | −0.20 pp |
| Returning baskets | NDCG@10 | 0.65383 | 0.65306 | −0.00077 |
| Returning baskets | Truly new top-10 hits | 222 / 1,340 | 218 / 1,340 | −4 |

The table uses the checkpoints selected on returning-customer NDCG. The first-basket NDCG gain has a 95% paired customer-bootstrap interval of +0.01765 to +0.06641. The first-basket Recall@10 interval is −1.47 to +8.59 percentage points. The returning-customer NDCG interval is −0.00310 to +0.00176 and its Recall@10 interval is −0.56 to +0.13 points. The selector's first-basket NDCG improved from 0.20090 to 0.24794 and its slate covered 315 targets versus 311. Most of the gain came from moving targets higher in the ranking. On the discovery-selected checkpoints, the mixed model found 149 versus 122 first-basket targets and 218 versus 228 new targets on returning baskets. This single-seed validation comparison uses the pinned Stage 5 reference as the control. See the [paired results](stage5_first_order_comparison_seed42.json).

```sh
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5 --first-order-mode control
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5 --first-order-mode mixed
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.compare_stage5_features --first-order
```

These commands require the local reference weights, cached company embeddings, and development feature bundles. The reference JSON records SHA-256 checksums. The per-customer arrays and model weights stay in the ignored checkpoint tree.
