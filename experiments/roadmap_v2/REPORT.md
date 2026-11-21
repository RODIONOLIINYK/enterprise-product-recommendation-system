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

The shared training cohort has 26,253 company-date cases. At formulation level it has 43,613 eligible next-basket positives and 119,329 eligible 60-day positives, including 9,332 and 30,268 new-to-company positives, respectively, before slate restriction. Those are labels on the same cases. In that experiment the first observed basket seeded the purchase history and cases started at the second observed basket.

Reproduce the comparison with `PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.compare_stage5_features --train-missing`. The [comparison summary](stage5_warehouse_exclusion_seed42.json) includes both checkpoint selections, paired intervals, and shared Two-Tower checksums.

## First Observed Baskets in the Same Pipeline

The user selected the warehouse-free financial, manager, and parsed-spec recipe as the [returning-basket reference](stage5_returning_reference.json). This experiment compares that saved selector and ranker against the same architecture and feature schema trained from scratch on a larger cohort. Both arms use the same frozen Two-Tower weights and 60-day auxiliary loss at weight 0.3. Checkpoints are selected on the original returning-customer validation set.

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

## Dated first-basket reference

The canonical run builds each training and validation basket from one development ledger, keyed by company and prediction date. All three stages were refit on seed 42. Checkpoint selection uses all 6,480 validation baskets in their observed proportions. The Two-Tower selects by unseen hits at 25 and 50, the MLP by new hits at 50, and the primary ranker by formulation NDCG at 10. The discovery checkpoint has a separate identity.

| Training population | Baskets |
| --- | ---: |
| First observed | 2,984 |
| Returning | 26,111 |
| Total | 29,095 |
| Of these, eligible for the discovery tower objective | 10,303 |
| Of these, with an entire 60-day auxiliary window | 26,182 |

All observed baskets contribute to later history, including those excluded from training. The 100-slot policy excludes 125 early training baskets with too few eligible formulations. Another 211 have no eligible target. Applying the same positive-eligibility rule to both cohorts removes 142 returning rows present in the previous mixed fit. Validation retains all modeled targets, including unavailable formulations. The returning cohort contains 1,367 new targets, of which 1,340 were eligible. The first cohort contains 354 new targets, of which 349 were eligible.

### Validation performance

Both columns use the same company/date baskets. Recall and NDCG are averaged per basket. New hits count distinct formulation targets within each basket.

| Cohort | Baskets | Previous mixed Recall@10 | Dated Recall@10 | Previous mixed NDCG@10 | Dated NDCG@10 | Previous → dated new hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 6,480 | 80.82% | 80.78% | 0.63526 | 0.63345 | 367 → 359 of 1,721 |
| First observed | 289 | 41.25% | 39.75% | 0.25397 | 0.23750 | 149 → 147 of 354 |
| Returning | 6,191 | 82.66% | 82.69% | 0.65306 | 0.65194 | 218 → 212 of 1,367 |

The 100-candidate slate covers 9,864 of 10,163 targets (97.06%) and 1,423 of 1,721 new targets (82.68%). After resolving formulations to eligible pack SKUs, this checkpoint reaches exact-SKU Recall@10 of 66.62% and NDCG@10 of 0.53376. See the [aggregate stage and checkpoint results](stage5_dated_first_baskets_seed42.json).

Against the previous mixed fit, pooled NDCG changed by −0.00181 (95% paired customer-bootstrap interval −0.00434 to +0.00077). First-basket NDCG changed by −0.01647 (−0.04359 to +0.00764). Intervals use 2,000 customer-clustered draws and seed 123. Multiple input and selection changes were made together. Neither these intervals nor a single training seed establishes which change accounts for the difference.

### Training and evaluation audit

- The previous mixed experiment reused embeddings from a returning-only tower and selected checkpoints on returning customers. The dated path trains the discovery tower on eligible first and returning baskets, re-encodes each customer from the selected weights and that row's history, and selects all stages on the mixed validation cohort.
- A first basket has zero valid history tokens, zero customer purchase counts, and zero brand/category/family familiarity. Cold cadence uses a fixed 14-day prior. The customer embedding can be nonzero because profile text and attributes still enter the tower. A masked placeholder keeps attention gradients finite with an empty history. A real history token must be at least one calendar day old.
- The historical formulation vector averaged the metadata of member packs, including packs first observed after a query. The new formulation vector takes its static text and attributes from its earliest paid pack, with ties resolved by SKU ID. New pack launches cannot change that vector retrospectively. Only formulations with an eligible, previously sold pack can be recommended. Candidate statistics, familiarity, and the final pack resolver use observations strictly before the sample date.
- The commercial export contains 70,030 lines. The paid-sale filter admits 65,048, whose quantities reconcile on all 61,048 cleaned company/date/SKU records. A multi-line dated basket counts as settled only once every paid line has a known earlier settlement. This prevents the absence of a recorded payment date from making a partial settlement look complete. Warehouse fields do not enter the feature schema.
- Customer and market statistics, target novelty, eligibility, and 32-event sequences are rebuilt in chronological order. The selector and ranker receive the financial, manager, and parsed-spec columns. The auxiliary loss uses only complete training-period windows. The ordinary next-basket loss still uses eligible cases through September 30, 2025.
- The permitted static side information includes company descriptions, profile attributes, product names, formulation identity, and catalogue attributes. The exports lack historical versions. Blocking uses the existing last-modified proxy. BGE-M3 text, training-fitted vocabularies and normalization scales are fixed inputs to this reference. The learned weights are fitted offline. Historical metadata accuracy and purchases before the export begins remain unobservable.
- The source manifest binds development event contents, static inputs, commercial export, ordered feature schema, canonical pack mapping, and ordered company/date keys. Private vectors and predictions are stored locally. The historical SKU retrieval and ranking references have their own scopes.

Reproduce the fit in a new isolated run, or replay the selected reference from the local sources and weights.

```sh
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5 --replay-reference
```

An independent source rebuild and checkpoint replay reproduced the selector metrics, coverage, formulation and resolved-SKU metrics, and every per-case metric for both ranker selections. The full workspace suite passed 630 tests (6 skipped), and the synthetic suite passed 620 (16 skipped). A clean export of the staged repository passed 552 synthetic tests (15 skipped). Each suite also passed 70 subtests. The workspace includes additional unpublished experiments.

---

## Final Held-Out Test Evaluation (`2026-02-19` to `2026-07-17`)

Authorized strictly inference-only evaluation of the pinned [`stage5-dated-first-baskets-v2`](stage5_reference.json) reference (`seed=42`, zero retraining, zero tuning, zero checkpoint re-selection) over the five-month held-out Test period (`2026-02-19..2026-07-17`). All rolling statistics, sequences, familiarity flags, eligibility masks, and multi-line payment settlements (`full_payment_date < t`) are computed strictly prior to each basket's date `t` from the reconciled commercial ledger (`76,797` paid lines matching all `71,916` cleaned rows), while keeping all frozen vocabularies, normalization scales, static formulation vectors (`585` dev formulations), and neural weights unchanged.

### Held-Out Test Population & Catalogue Shift

- **All test baskets:** `6,582` (`255` first observed + `6,327` returning), `10,406` distinct formulation targets (`1,913` truly new to customer) across `1,832` companies (`100%` covered by cached BGE-M3 company text).
- **Test-only catalogue launches (`>= 2026-02-19`):** `58` new formulations introduced only during the Test window fall outside the frozen `585`-formulation dev universe, accounting for `166` targets (`1.60%` of all targets; `134` truly new) across `150` baskets (`85` baskets contain only unmodeled targets: `6` first observed + `79` returning).
- **Both scopes are reported below:** **Full Catalogue (`6,582` baskets)** counts all `166` unmodeled formulations as misses in the denominator (unbiased production reality without catalogue refresh); **Modeled Universe (`6,497` baskets)** matches the `585`-formulation dev denominator used in validation.

### Primary Ranker (`best_ndcg.pt`, Epoch 7) — Validation vs. Held-Out Test

| Cohort | Validation (`6,480`) | Test — Modeled (`6,497`) | Test — Full Catalogue (`6,582`) |
| :--- | ---: | ---: | ---: |
| **All — Recall@5** | `70.51%` | `68.61%` | `67.49%` |
| **All — Recall@10** | `80.78%` | `79.89%` | `78.56%` |
| **All — Recall@50** | `93.36%` | `93.55%` | `92.00%` |
| **All — NDCG@10** | `0.63345` | `0.61852` | `0.60895` |
| **All — New hits@10** | `359 / 1,721 (20.86%)` | `353 / 1,779 (19.84%)` | `353 / 1,913 (18.45%)` |
| **Returning — Recall@10** | `82.69%` | `81.57%` | `80.24%` |
| **Returning — NDCG@10** | `0.65194` | `0.63408` | `0.62452` |
| **Returning — New hits@10** | `212 / 1,367 (15.51%)` | `226 / 1,445 (15.64%)` | `226 / 1,564 (14.45%)` |
| **First observed — Recall@10** | `39.75%` | `37.72%` | `36.81%` |
| **First observed — NDCG@10** | `0.23750` | `0.22811` | `0.22264` |
| **First observed — New hits@10** | `147 / 354 (41.53%)` | `127 / 334 (38.02%)` | `127 / 349 (36.39%)` |
| **100-Slate Coverage (Total)** | `9,864 / 10,163 (97.06%)` | `9,945 / 10,240 (97.12%)` | `9,945 / 10,406 (95.57%)` |
| **100-Slate Coverage (New)** | `1,423 / 1,721 (82.68%)` | `1,489 / 1,779 (83.70%)` | `1,489 / 1,913 (77.84%)` |
| **Resolved Exact-SKU Recall@10** | `66.62%` | `65.93%` | `64.15%` |
| **Resolved Exact-SKU NDCG@10** | `0.53376` | `0.51880` | `0.50595` |

### Stage-by-Stage & Discovery Checkpoint (`best_discovery.pt`, Epoch 3) Summary

- **Two-Tower (Epoch 7):** `910` unseen new hits@25 (`51.15%` modeled / `47.57%` full catalogue) and `1,222` hits@50 (`68.69%` modeled / `63.88%` full catalogue), vs. `908` / `1,183` on validation.
- **Stage 1 MLP (Epoch 8):** Modeled `Recall@10 = 79.67%`, `NDCG@10 = 0.61656`, `New@10 = 352 / 1,779`; Full Catalogue `Recall@10 = 78.34%`, `NDCG@10 = 0.60699`, `New@10 = 352 / 1,913`.
- **Stage 2 Discovery Checkpoint (`best_discovery.pt`, Epoch 3):** Modeled `Recall@10 = 80.04%`, `NDCG@10 = 0.61702`, `New@10 = 359 / 1,779 (20.18%)` (`234` returning + `125` first observed); Full Catalogue `Recall@10 = 78.70%`, `NDCG@10 = 0.60743`, `New@10 = 359 / 1,913 (18.77%)`; Resolved Exact-SKU `Recall@10 = 64.02%` (full) / `65.80%` (modeled). The validation trade-off (higher NDCG@10 on `best_ndcg.pt`, higher Recall@10 and new discovery on `best_discovery.pt`) replicated cleanly out-of-time.

### Integrity & Reproduction

- All preserved reference hashes, all 5 pinned `stage5_reference.json` artifact SHA-256 digests, and `commercial_dev_events.parquet` (`20fc9513...`) were verified unchanged before and after evaluation.
- Aggregate metrics are stored in [`stage5_dated_first_baskets_test_seed42.json`](stage5_dated_first_baskets_test_seed42.json); private per-query `.npz` outputs stay in the ignored checkpoint directory.
- Reproduce (authorized held-out flag required): `PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.evaluate_stage5_test --allow-heldout-test`.

---

## Stage 5 Follow-Up: Excluding Stage 1 MLP, CatBoost & Modest Ranker Scaling (Validation-Only)

All experiments below ran strictly on the `29,095` training (`< 2025-10-01`) and `6,480` validation (`2025-10-01..2026-02-18`) dated baskets (**zero held-out Test access**), using the exact same `55` causal `rolling_observed_v2` `base_name` features (`41` `base_name` + `3a_financial` + `3c_manager` + `3e_parsed_specs`) and frozen Two-Tower `v_bn` vectors where neural inputs are used.

### Validation Comparison (`6,480` baskets = `289` first observed + `6,191` returning)

| Model (All `585` Eligible Formulations unless noted) | Params / Trees | `R@5` | `R@10` | `NDCG@10` | `New@10` *(of `1,721`)* | First-Obs `NDCG@10` | Returning `New@10` |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived 3-Stage w/ Stage 1 MLP (NDCG, Ep 7) | 100-slate | `70.51%` | `80.78%` | `0.63345` | `359` | `0.23750` | `212` |
| Archived 3-Stage w/ Stage 1 MLP (Disc, Ep 3) | 100-slate | `70.54%` | `80.93%` | `0.63232` | `367` | `0.25371` | `211` |
| No-MLP Zero-Shot (Pinned NDCG ckpt, all 585) | `~600K` | `70.50%` | `80.74%` | `0.63330` | `355` | `0.23416` | `212` |
| No-MLP Full-Pool `UPS-CANR` (Std, NDCG Ep 7) | `~600K` | `70.53%` | `80.76%` | `0.63369` | `354` | `0.23543` | `217` |
| No-MLP Full-Pool `UPS-CANR` (Std, Disc Ep 4) | `~600K` | `70.47%` | `80.88%` | `0.63337` | `371` | `0.25717` | `219` |
| **No-MLP `modest_wider_mlp_384` (Ep 7 — Active Ref)** | **`769K`** | `70.54%` | `80.85%` | **`0.63428`** | `366` | `0.24414` | `222` |
| **No-MLP `modest_8h_ffn384_mlp384` (Disc Ep 5)** | **`803K`** | `70.62%` | **`80.96%`** | `0.63321` | **`378`** | `0.23983` | **`230`** |
| No-MLP `wider_16h_ffn384_mlp512` (Ep 4 / Disc Ep 3) | `933K` | `70.51%` | `80.74%` *(80.90% ep3)* | `0.63340` | `366` *(374 ep3)* | — | — |
| No-MLP `more_blocks_2mem_2xattn_4mlp384` (Ep 4 / Ep 5) | `1.08M` | `70.55%` | `80.69%` *(80.93% ep5)* | `0.63108` | `361` *(367 ep5)* | — | — |
| No-MLP Large `bigger_128d_8h_2L_4mlp` (Ep 6 / Ep 3) | `1.30M` | `70.64%` | `80.57%` | `0.63201` | `353` *(372 ep3)* | — | — |
| **CatBoost Standalone (`72` feats, 0 neural)** | `661 trees` | **`70.69%`** | `80.70%` | `0.63403` | `356` | **`0.27037`** | `204` |
| CatBoost + Two-Tower (`78` feats) | `501 trees` | `70.30%` | `80.35%` | `0.63291` | `357` | `0.23739` | `209` |

### Key Conclusions & New Active Reference (`stage5-fullpool-nomlp-v3`)

- **Excluding Stage 1 MLP simplifies the system and improves metrics:** With only `585` canonical `base_name` formulations, scoring the full eligible pool directly in Stage 2 eliminates both `RoadmapStage1Selector` and the 100-candidate slate bottleneck (`100%` eligible candidate coverage, `3.3s` total for `6,480` baskets / `1.16 ms` single-request CPU).
- **Modest capacity scaling hits the sweet spot (`~769K–803K` params):** While drastic scaling (`1.30M` params with 2 memory + 2 cross-attention layers) overfitted (`NDCG@10 = 0.63201`), gently widening the scoring MLP to `(384, 192, 96)` (`modest_wider_mlp_384`, Epoch 7) set a new overall validation peak (**`NDCG@10 = 0.63428`**, `Recall@10 = 80.85%`, `New@10 = 366/1,721`, resolved SKU `Recall@10 = 66.66%` / `NDCG@10 = 0.53410`), and adding 8 heads + FFN 384 (`modest_8h_ffn384_mlp384`, Epoch 5) set new peaks on **`Recall@10 = 80.96%`** and **`New@10 = 378/1,721`** (`230` returning + `148` first observed).
- **CatBoost Standalone vs. `UPS-CANR`:** Pure tabular CatBoost (`72` features, zero neural models, `0.55 ms` single-request CPU model-only / `1.39 ms` incl. `Pool`) wins cold-start (`First Observed NDCG@10 = 0.27037`) and `Recall@5` (`70.69%`), while `UPS-CANR` wins returning-customer discovery (`222–230` vs `204`) and pooled `NDCG@10` (`0.63428` vs `0.63403`).
- **Active Reference Promotion:** Per user selection, [`stage5_reference.json`](stage5_reference.json) now points to `stage5-fullpool-nomlp-v3` (`modest_wider_mlp_384` primary + `modest_8h_ffn384_mlp384` discovery companion), while the prior 3-stage reference evaluated on Test is preserved verbatim in [`stage5_3stage_reference.json`](stage5_3stage_reference.json). See [`stage5_nomlp_and_catboost_seed42.json`](stage5_nomlp_and_catboost_seed42.json) and [`stage5_modest_ranker_seed42.json`](stage5_modest_ranker_seed42.json).
