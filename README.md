# Enterprise Product Recommendation System

Next-basket B2B product formulation recommender for industrial distribution, built around **canonical `base_name` entity unification**, **strictly point-in-time rolling commercial/behavioral features (`cand_42 + CUST_7 + SEQ_5`)**, and a **Single-Stage End-to-End Full-Pool `Wave2BE2ERanker` with a dedicated `OptionAUnseenHead` Cross-Sell Head** (zero separate Stage 1 pretraining, zero top-K shortlist truncation, `~4–6 ms` median single-request CPU latency across all `1,141` formulations).

---

## Current Architecture & End-to-End Pipeline (`v2_full`: `v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6`)

In B2B industrial distribution (lubricants, coolants, chemicals), the same product formulation is sold across multiple package sizes (e.g., 4L jug, 20L pail, 209L drum). Rather than letting 4–5 pack variants of one oil crowd Top-10 slots, the active **`v2_full`** pipeline unifies **`1,660` development SKUs into `1,141` canonical `base_name` formulations** over 4 years of real paid sales (`2023-01-03` → `2026-06-30`), handles **both returning and cold-start (first-observed) companies** in a single dated pipeline, and trains a **single-stage end-to-end neural ranker** with cleanly separated specialist heads for repeat replenishment and new-product discovery.

```text
Raw 4-Year ERP / NAV Sales (dataset.xlsx) + Companies (customers.csv) + Catalogue (items.csv)
  │
  ▼  1. Paid-Sale & Multi-Line Settlement Reconciliation (Strictly < TEST_START_V2 = 2026-07-01)
     • Filters strictly to positive paid merchandise sales (ПРОДАЖА + ТОВАР; excludes gifts/returns)
     • 460,109 dev records → 137,765 Train baskets (< 2026-02-01) & 16,623 Val baskets (2026-02-01..2026-06-30)
     • Held-Out Test (>= 2026-07-01, ~13,190 baskets) remains STRICTLY UNTOUCHED
     • Unifies 1,660 dev SKUs → 1,141 canonical base_name formulations (1,056 train-observed)
  │
  ▼  2. Strictly Causal Feature & Sequence Engine (cand_42 + CUST_7 + SEQ_5, all < prediction_date)
     • 32-event chronological base_name history (SEQ_5 active stats: qty, live_age, prior_cnt, vol_share, valid)
     • Token 0 Customer Current State (CUST_7):
         customer_age, 30d/90d activity, prepaid/cash ratio, avg/last deferral days, overdue unpaid ratio
     • 42D Strictly Causal Candidate Features (cand_42 = pruned cand_38 + extra_cand_5):
         -  5D repeat-SKU cadence (has_bought, order_count, qty_share, days_since_last, urgency)
         - 32D non-repeat core/commercial/spec signals (cat/subcat/brand/family shares & cadence, global 7d/30d/90d
               velocity & share, cross-company sequential & complement lifts, 90d manager shares, viscosity/equipment)
         -  5D extra_cand_5 signals: mgr_base_name_share_30d, mgr_industry_base_name_share_90d,
               region_base_name_share_90d, exact incremental full_cf_cosine & 180d full_cf_recency_weighted
  │
  ▼  3. Single-Stage E2E Wave2BE2ERanker (Zero Stage 1 Pretraining; 963K per-req / 1.20M total params)
     • ProductTower (StaticEnrichedProductTower): pack-free BGE-M3 (1024D) + 65D multi-hot specs + 6 attrs → v_bn (1,141 × 128)
       Trained end-to-end inside Stage 2 and precomputed once per catalogue refresh (0 ms per-request overhead)
     • Token 0 (Customer): token0_fusion([BGE-M3 company_text → 128, city/region/industry + CUST_7 → 128])
     • Tokens 1..32 (History): direct E2E v_bn[seq_product_ids] + pos_emb + type_emb[1]
     • 1-layer 8-head TransformerEncoder (FFN=384) over 33 tokens → Pure-Q (Q = v_bn) 8-head cross-attention
       with Full-42 bilinear statistics bias (43 × 12 per head)
     • Cleanly Separated Specialist Heads (conflict-free targets, 0% tabular dropout, w_optA = 1.0):
         - Main Head (repeats):      MLP([v_bn, h_att, v_bn ⊙ h_att, cand_42]) → s_main  (trained on returning repeats)
         - OptionAUnseenHead (new):  MLP([v_bn, h_att, v_bn ⊙ h_att, cand_non_repeat_37]) → s_unseen (unseen only)
     • Training-only 60d forward aux heads (λ = 0.3) stripped at inference
  │
  ▼  4. 2-Parameter Out-of-Fold Affine Calibration & Full-Pool Ranking (~4–6 ms CPU)
     • score(p) = s_main(p) if previously bought else a · s_unseen(p) + b
     • Customer-grouped 2-fold OOF calibration (full-fit deployment: a = 1.200, b = -2.118)
     • Scores all 1,141 eligible formulations directly (also supports Pure Unseen ranking via OptionAUnseenHead)
  │
  ▼  5. Top-10 Ranked Formulations (+ Causal Pack-Size Resolver → Eligible Pack SKU, ~0.2 ms)
     • Per formulation: customer's most frequent/recent eligible pack < t → dominant container < t → popular pack < t
```

- **Active Canonical Reference:** [`experiments/v2_full/v2_full_reference.json`](experiments/v2_full/v2_full_reference.json) (`v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6`, Ep 6 unified `cal-Discovery == cal-Unseen` primary + Ep 10 `cal-NDCG` companion; `seed=42`, zero held-out Test access).
- **Frozen `v1` Reference (Preserved):** [`experiments/roadmap_v2/stage5_reference.json`](experiments/roadmap_v2/stage5_reference.json) (`stage5-v5-shared-uv-ep10`, `585` formulations) and archived 3-stage test-evaluated [`stage5_3stage_reference.json`](experiments/roadmap_v2/stage5_3stage_reference.json).

---

## Architecture Evolution & Milestone History

| Milestone | Dataset & Universe | Architecture | Key Change | Validation Headline |
| :--- | :--- | :--- | :--- | :--- |
| **1. Initial SKU Baseline** | `v1` (`841` SKUs, `6,191` returning val) | Two-Tower → 50-item slate → `UPS-CANR` (`13` feats) | First neural retriever + statistics-biased cross-attention ranker | `R@10 76.15%`, `NDCG@10 0.5924`, `New@10 209/1,857` |
| **2. Stage 5 Entity Unification (`v1`)** | `v1` (`585` `base_name`, `6,480` dated val incl. cold-start) | Two-Tower (`65D` specs) → Stage 1 MLP → `UPS-CANR` (`100`-slate) | Unified pack variants into canonical formulations; added commercial/financial/manager/spec features + first-observed baskets | `R@10 80.78%`, `NDCG@10 0.63345`, `New@10 359/1,721` *(Test Full Cat: `78.56%`)* |
| **3. No-MLP Full-Pool Shared-UV (`v1`)** | `v1` (`585` `base_name`, `6,480` val) | 2-Stage `SharedUVStage5Ranker` (`567K` params, `1.32 ms`) | Dropped Stage 1 MLP & top-100 truncation; grounded Stage 2 directly in Stage 1 `u_bn`/`v_bn` (`cand_38 + CUST_11`) | `R@10 80.89%`, `NDCG@10 0.63457`, `New@10 372/1,721` (`21.62%`) |
| **4. 4-Year `v2_full` Wave 1** | `v2_full` (`1,141` `base_name`, `137,765` train / `16,623` val) | 6-arm ablation: 2-stage vs. single-stage E2E (`arm4`) | Expanded to 4 years (`4.74×` baskets, `~6×` unseen positives); proved single-stage E2E matches/beats 2-stage with zero Stage 1 pretraining | `arm4`: `R@10 81.84%`, `NDCG@10 0.65232`, `New@10 1,168/4,496` (`25.98%`) |
| **5. `v2_full` Wave 2B** | `v2_full` (`1,141` `base_name`, `16,623` val) | `Wave2BE2ERanker` (`cand_42`, Pure-Q, Full-42 bias, Option A 70/30) | Pruned dead features + added `extra_cand_5` (CF cosine/recency, mgr 30d, mgr×ind, region 90d), removed `candidate_mlp` (`Q = v_bn`), added `OptionAUnseenHead` + OOF affine blend | `R@10 82.02%`, `NDCG@10 0.65139`, `New@10 1,212/4,496`, `Pure U@10 2,312` |
| **6. `v2_full` Wave 2C (Active Ref) ⭐** | `v2_full` (`1,141` `base_name`, `16,623` val) | `Wave2BE2ERanker` (`w2c_exp1_clean_sep_w10`, Ep 6) | Replaced 70/30 dropout with **clean conflict-free separated Main/Option A targets**, `0%` dropout, and `unseen_head_weight = 1.0` (eliminated synthetic masked-repeat augmentation after negative ablation) | **`R@10 82.12%`**, **`NDCG@10 0.65380`**, **`New@10 1,219/4,496 (27.11%)`**, **`Pure U@10 2,298 (51.11%)`** |

---

## Verified Performance

### 1. Active `v2_full` Validation Benchmarks (`2026-02-01` → `2026-06-30`, `16,623` baskets, `40,435` targets, `4,496` truly new — Zero Held-Out Test Access)

| Model (`1,141` Canonical Formulations) | `R@5` | `R@10` | `R@25` | `NDCG@10` | `New@10` *(Ret / FO)* | `Pure U@10` | `Pure U@25` | SKU `NDCG@10` (`New@10`) | 1-Req CPU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Wave 1 `baseline_2stage_v5` (2-stage, `cand_38`) | `68.18%` | `81.88%` | — | `0.65315` | `1,147` (`647 / 500`) | — | — | `0.45437` (`613`) | `6.70 ms` |
| Wave 1 `arm4_e2e_unseen_attn_aux` (E2E, `cand_38`) | `68.26%` | `81.84%` | — | `0.65232` | `1,168` (`667 / 501`) | — | — | `0.45306` (`625`) | `6.50 ms` |
| CatBoost `cb_mixed_cand42` (`1,500` trees) | — | `81.40%` | — | `0.64260` | `1,108` (`645 / 463`) | `2,171` | `3,051` | `0.44422` (`565`) | `1.78 ms` |
| CatBoost `cb_optionA_oof_blend` (`cand_42`) | — | `81.61%` | — | `0.64379` | `1,139` (`643 / 496`) | `2,278` | `3,162` | `0.44512` (`589`) | — |
| Wave 2B `w2b_cand42_optionA_pureQ_full42bias` (Ep 5) | — | `82.02%` | — | `0.65139` | `1,212` (`710 / 502`) | `2,312` | `3,197` | `0.45107` (`639`) | `6.80 ms` |
| Wave 2C `w2c_exp1_clean_sep_w05` (`0.5× L_optA`, Ep 10) | `68.26%` | `82.07%` | `92.56%` | `0.65439` | `1,193` (`691 / 502`) | `2,235` | `3,152` | `0.45462` (`623`) | `~6.4 ms` |
| **Active Primary: `w2c_exp1_clean_sep_w10` (`1.0× L_optA`, Ep 6) ⭐** | **`68.28%`** | **`82.12%`** | **`92.77%`** | `0.65380` | **`1,219`** (`721 / 498`) | **`2,298`** (`51.11%`) | **`3,218`** (`71.57%`) | `0.45456` (**`630`**) | **`6.40 ms`** |
| **Companion: `w2c_exp1_clean_sep_w10` (`cal-NDCG`, Ep 10)** | `68.17%` | `82.10%` | `92.68%` | **`0.65437`** | `1,160` (`665 / 495`) | `2,276` (`50.62%`) | `3,203` (`71.24%`) | **`0.45461`** (`613`) | `6.40 ms` |

- **Held-out Test Policy (`v2_full`):** All baskets `>= 2026-07-01` remain strictly locked and untouched while Wave 2A capacity/sequence experiments proceed.

### 2. Frozen `v1` Reference Summary (`585` Formulations — Preserved)
- **Validation (`2025-10-01` → `2026-02-18`, `6,480` baskets):** Active `v1` `v5_shared_uv` Ep 10 (`567K` params, `1.32 ms` CPU) achieves `R@10 = 80.89%`, `NDCG@10 = 0.63457`, `New@10 = 372/1,721 (21.62%)`, Resolved SKU `R@10 = 66.77%` / `NDCG@10 = 0.53504`.
- **Historical Held-Out Test (`2026-02-19` → `2026-07-17`, inference-only on archived 3-stage reference):** `79.89%` Modeled / `78.56%` Full-Catalogue `R@10` (`6,582` baskets). See [`experiments/roadmap_v2/REPORT.md`](experiments/roadmap_v2/REPORT.md).

---

## Repository Structure (Active Pipeline)

```text
.
├── configs/
│   ├── datasets_links.toml                  # Local paths to raw dataset.xlsx, customers.csv, items.csv
│   └── product_scope.toml                   # Paid-sale & sellable merchandise rules (ПРОДАЖА + ТОВАР)
├── recommender/
│   ├── data.py, merging.py, scope.py, preparation.py
│   │                                        # Raw Excel/CSV parsing, paid-sale filtering, quantity-summing merge
│   ├── company.py, catalogue.py             # Company & item master loading, RollingCandidatePool
│   ├── split.py, development.py             # Strict temporal split & development-only purchase loader
│   ├── features.py, sequences.py, text.py, familiarity.py, ranker.py
│   │                                        # Sequence utilities, BGE-M3 cache, RankerLoss & RankerOutput
│   └── experiments/
│       ├── v2_full/                         # ACTIVE 4-year v2_full single-stage E2E pipeline
│       │   ├── config.py                    # Splits (TRAIN_CUTOFF_V2=2026-02-01, TEST_START_V2=2026-07-01) & ref loader
│       │   ├── prepare_v2_full.py           # Leak-free 4-year ingestion, 1,141 base_name universe & cand_38 memmaps
│       │   ├── extra_features.py            # Strictly causal extra_cand_5 (CF, mgr 30d, mgr×ind, region 90d) & cand_42/CUST_7/SEQ_5
│       │   ├── wave2b_models.py             # Wave2BE2ERanker, OptionAUnseenHead, Pure Unseen metrics & 2-fold OOF affine blend
│       │   ├── e2e_ranker.py                # Wave 1 EndToEndStage2Ranker & TwoStageAdaptiveRanker
│       │   ├── run_v2_full_wave2c.py        # Wave 2C clean separated Option A runner (active reference trainer)
│       │   ├── run_v2_full_wave2b.py        # Wave 2B cand_42 + Pure-Q + Option A runner & shared batching
│       │   ├── run_v2_full_wave2.py         # Wave 2A capacity / history-length / recency experiment runner
│       │   ├── run_v2_full_catboost.py      # Standalone 42-feature CatBoost + Option A OOF blend & ensemble
│       │   ├── run_v2_full_experiments.py   # Wave 1 6-arm 2-stage vs. E2E benchmark
│       │   ├── train_v2_full_baseline.py    # AMP context, full-pool ranking & sync helpers
│       │   ├── wave2_utils.py, provenance.py# Shared Wave 2 utilities & dataset/artifact SHA-256 provenance
│       │   └── demo_engine.py               # Live V2FullLiveDemoEngine serving w2c_exp1_clean_sep_w10 on Validation
│       ├── roadmap_v2/                      # Shared base_name / commercial / spec builders & live demo UI
│       │   ├── demo_dashboard.py            # Interactive web dashboard (defaults to v2_full; --model v1 optional)
│       │   ├── dated_baskets.py, stage5_base_entity.py, native_base_name.py, commercial_features.py
│       │   ├── parsed_spec_features.py, augmentation.py, clean_ranker.py, stage5_artifacts.py
│       │   └── run_stage5.py, run_stage5_ablations.py, evaluate_stage5_test.py
│       └── static_enrichment/               # StaticEnrichedProductTower & enriched metadata loaders
├── experiments/
│   ├── v2_full/                             # Active v2_full reference manifest & Wave 1/2B/2C/CatBoost JSON results
│   │   ├── v2_full_reference.json           # Canonical active pointer (v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6)
│   │   ├── v2_full_wave2c_arm4_e2e_unseen_attn_aux_seed42.json
│   │   ├── v2_full_wave2b_arm4_e2e_unseen_attn_aux_seed42.json
│   │   ├── v2_full_catboost_wave2b_seed42.json
│   │   └── v2_full_experiments_seed42.json
│   └── roadmap_v2/                          # Frozen v1 Stage 5 reference manifests & REPORT.md
├── notebooks/
│   └── colab_v2_full_experiments.ipynb      # Native bf16 Colab L4/A100 training & Drive auto-sync notebook
├── tests/                                   # Unit, causality, parity & leakage-guard test suite
├── AGENTS.md                                # Agent engineering handbook & strict non-leakage rules
└── plan.md                                  # Active roadmap, completed Wave 1/2B/2C results & Wave 2A plan
```

---

## Setup, Live Demo & Commands

Use Python 3.12 (`./venv/bin/python`; Apple Silicon `mps` or CUDA `bf16` detected automatically):

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r requirements-dev.txt
# Optional: for CatBoost benchmarks
./venv/bin/pip install -r requirements-catboost.txt
```

### Interactive Live Demo Dashboard (`http://127.0.0.1:8765`)
Serves the active `v2_full` model (`w2c_exp1_clean_sep_w10`, Ep 6) live on CPU (~4–6 ms/req) over the `16,623` out-of-time Validation baskets (`2026-02-02` → `2026-06-30`, zero held-out Test access), with cohort filters, Case `#` jump, and a live **Ranking Mode** switcher (`Blended (Main + Option A OOF)`, `Pure Unseen Only (Option A Head)`, `Main Repeat Head Only`):

```bash
# Default: active v2_full Ep 6 cal-Discovery == cal-Unseen reference (zero Test access)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --port 8765

# Optional: Ep 10 cal-NDCG companion checkpoint
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --checkpoint-kind ndcg --port 8765

# Legacy v1 Stage 5 Shared-UV model on held-out Test baskets (>= 2026-02-19)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --model v1 --allow-heldout-test --port 8765
```

### `v2_full` Preparation, Training & Testing

```bash
# 1. Prepare leak-free 4-year v2_full dataset (< 2026-07-01) & seal extra_cand_5
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.prepare_v2_full
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.extra_features

# 2. Train Wave 2C clean separated Option A (use --amp --in-memory on native bf16 CUDA L4/A100)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.run_v2_full_wave2c \
    --base arm4_e2e_unseen_attn_aux --seed 42 --amp --in-memory --skip-v1-verify

# 3. Run standalone 42-feature CatBoost benchmark + OOF affine blend
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.run_v2_full_catboost \
    --task-type auto --seed 42 --in-memory --skip-v1-verify

# 4. Run active v2_full + Stage 5 audit tests (zero held-out Test access)
PYTHONPATH=. ./venv/bin/pytest tests/test_v2_full_pipeline.py tests/test_v2_full_wave2b.py \
    tests/test_v2_full_wave2c.py tests/test_v2_full_wave2.py tests/test_stage5_audit_fixes.py tests/test_data.py -q
```

---

## Data Handling

Raw data (`data/`), interim caches, BGE-M3 embeddings, precomputed predictions (`.npz`), Colab bundles (`exports/`), and model checkpoints (`checkpoints/`) remain strictly local and git-ignored. Only aggregate JSON summaries, SHA-256 reference manifests, code, notebook source (outputs cleared), and tests are tracked in Git.
