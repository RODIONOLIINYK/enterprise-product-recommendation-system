# Enterprise Product Recommendation System

Next-basket B2B product formulation recommender for industrial distribution, built around **canonical `base_name` entity unification**, **strictly point-in-time rolling commercial/behavioral features (`rolling_observed_v2`)**, and a **2-stage No-MLP Full-Pool `UPS-CANR` Neural Reranker** (with a standalone CatBoost tabular benchmark).

---

## Current Architecture & End-to-End Pipeline (`Stage 5: stage5-v5-shared-uv-ep10`)

In B2B industrial distribution (lubricants, coolants, chemicals), the same product formulation is sold in multiple package sizes (e.g., 4L jug, 20L pail, 209L drum). Rather than letting 4–5 packaging variants of the same oil compete for Top-10 slots, the active **Stage 5** pipeline unifies **`841` development SKUs into `585` canonical `base_name` product formulations** across every stage, handles **both returning customers and cold-start (first-observed) companies** in one dated pipeline, grounds all Stage 2 tokens and candidate queries directly in Stage 1's shared `128D` embedding space (`567K` params, zero scratch text/ID tables), and scores **all `585` eligible formulations directly** (`~1.3 ms` ranker on CPU).

```text
Raw ERP / NAV Sales (dataset.xlsx) + Companies (customers.csv) + Catalogue (items.csv)
  │
  ▼  1. Paid-Sale & Multi-Line Settlement Reconciliation
     • Filters to positive paid merchandise sales (ПРОДАЖА + ТОВАР; excludes gifts/returns)
     • Merges same-(company, date, SKU) lines (summed quantity); 76,797 paid lines ↔ 71,916 cleaned rows
     • Multi-line basket counts as paid only after EVERY paid line settles strictly < prediction date
     • Collapses 841 dev SKUs → 585 canonical base_name formulations; all statistics computed natively on merged base_name events
  │
  ▼  2. Strictly Causal Native 585-Base-Name Feature & Sequence Engine (~0.55 ms/req)
     • 32-event chronological base_name purchase sequence (same-day multi-packs collapsed to 1 event; 0 tokens for first orders; month & repeat excluded)
     • Token 0 Customer Current State (11D — NOT broadcast to candidates!):
         customer_age, customer_overdue_ratio, customer_30d/90d_activity, 6 financial/payment ratios (3a_financial), mgr_changed_recently
     • 38D Strictly Candidate / Customer-Candidate Features per (company, date, base_name):
         - 13D core pairwise/attention-bias features (repeat, recency, urgency, collapsed cat/subcat/brand shares, global pop, industry affinity)
         -  5D category & family cadence/share (no binary never_seen duplicates)
         -  7D global base_name velocity & share (7d, 30d, 90d, buyers_30d, growth_30d, category share, global sale age)
         -  7D cross-company sequential transitions & same-basket complements (no formula duplicate hierarchical_backoff)
         -  3D manager rolling 90d behavioral shares (base_name, brand, category)
         -  3D pack-agnostic parsed specs (multi-value user_viscosity_order_share, app_matches_company_equipment, user_application_order_share)
  │
  ▼  3. Stage 1 — Pure Pack-Free Two-Tower Encoder (No sales_branch, 65D Multi-Hot Specs)
     • ProductTower: pure pack-free BGE-M3 text (585 distinct base_names) + 65D multi-hot [12 container + 8 vol + 11 app + 34 visc] + 6 product attrs → cached v_bn (585 × 128)
     • CustomerTower: company BGE-M3 + city/region/industry + 32-event base_name Transformer → live u_bn (128D) recomputed per request
     • Zero Stage 1 MLP & zero top-50/top-100 shortlist truncation (100% eligible candidate coverage)
  │
  ▼  4. Stage 2 — Unified Shared-UV Full-Pool UPS-CANR (SharedUVStage5Ranker, 567K params, zero scratch text/ID tables)
     • Token 0 (Customer): LayerNorm(u_bn + MLP([u_bn (128), cust_11 (11)])) + type_emb[0]
     • Tokens 1..32 (History): direct precomputed Stage-1 v_bn[seq_product_ids] (128) + pos_emb + type_emb[1] (no redundant MLP)
     • Candidate Queries Q (all 585): LayerNorm(v_bn + MLP([v_bn (128), cand_38 (38)])) before 8-head statistics-biased cross-attention
     • 1-layer 8-head TransformerEncoder (FFN=384) over 33 tokens → 8-head cross-attention → MLP [z_cand, h_att, z_cand ⊙ h_att, 38D] = 422D → 384 → 192 → 96 → 1
     • Trained with RankerLoss + training-only 60-day forward auxiliary head (λ = 0.3, stripped at inference)
     • Scores all 585 eligible formulations directly in 1.32 ms median on CPU (zero late-epoch overfitting through Ep 10)
  │
  ▼  5. Top-10 Ranked Product Formulations (+ Optional Causal Pack-Size Resolver → Concrete Eligible SKU, 0.08 ms)
     • Per recommended base_name: customer's most frequent/recent eligible pack < t → dominant container type < t → rolling popular pack < t
```

- **Active Reference Manifest:** [`experiments/roadmap_v2/stage5_reference.json`](experiments/roadmap_v2/stage5_reference.json) (`stage5-v5-shared-uv-ep10`, `seed=42`, trained on `29,095` dated baskets = `2,984` first observed + `26,111` returning; validated on `6,480` baskets = `289` first observed + `6,191` returning).
- **Archived 3-Stage Reference (Evaluated on Held-Out Test):** [`experiments/roadmap_v2/stage5_3stage_reference.json`](experiments/roadmap_v2/stage5_3stage_reference.json) (`stage5-dated-first-baskets-v2`, which included Stage 1 MLP + `quota100_50` 100-item slate).

---

## Verified Performance

### 1. Validation Benchmarks (`2025-10-01` to `2026-02-18`, `6,480` baskets, `10,163` targets, `1,721` truly new — Zero Test Access)

| Model (`585` Canonical Formulations) | Pipeline | `R@5` | `R@10` | `NDCG@10` | `New@10` *(of `1,721`)* | First-Obs `New@10` / `NDCG@10` | Returning `New@10` | 1-Req CPU Latency |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Archived 3-Stage Reference (NDCG, Ep 7) | 2T → S1 MLP → S2 (`100`-slate) | `70.51%` | `80.78%` | `0.63345` | `359` (`20.86%`) | `147` / `0.23750` | `212` | — |
| Prior `v3` `modest_wider_mlp_384` (Ep 7) | 2T → Full-Pool S2 (`55D`, `769K`) | `70.54%` | `80.85%` | `0.63428` | `366` (`21.27%`) | `144` / `0.24414` | `222` | `~1.2 ms` |
| Clean `v4b` `v4b_fullspec65_nosb` (Ep 6 / Ep 8) | 2T (`65D`) → Clean S2 (`755K`) | **`70.80%`** | **`81.10%`** | **`0.63485`** | `373` *(**381** ep8)* | `151` / `0.24399` | `222` *(**239** ep8)* | **`1.13 ms`** |
| **`v5_shared_uv` (Active Ref, Ep 10 Disc)** ⭐ | **2T (`65D`) → Shared-UV S2 (`567K`)** | `70.54%` | `80.89%` | `0.63457` | **`372`** (`21.62%`) | `149` / `0.25231` | **`223`** | **`1.32 ms`** |
| **`v5_shared_uv` (Companion, Ep 7 NDCG)** ⭐ | **2T (`65D`) → Shared-UV S2 (`567K`)** | `70.69%` | `80.94%` | **`0.63485`** | `366` (`21.27%`) | `148` / **`0.25926`** | `218` | **`1.32 ms`** |
| **CatBoost Standalone (`72` feats, `661` trees)** | **Pure Tabular Full-Pool (0 neural)** | `70.69%` | `80.70%` | `0.63403` | `356` (`20.69%`) | `152` / **`0.27037`** | `204` | **`0.55 ms`** *(1.39ms w/ Pool)* |

- **Resolved Exact-SKU Validation (`v5_shared_uv`):** Active Ep 10 reaches `sku_Recall@10 = 66.77%`, `sku_NDCG@10 = 0.53504`, `sku_New@10 = 214`; Companion Ep 7 sets all-time resolved-SKU highs (`sku_Recall@10 = 66.93%`, `sku_NDCG@10 = 0.53624`, `sku_New@10 = 220`).
- **Takeaway:** `v5_shared_uv` deletes `188K` redundant Stage 2 parameters (`755K → 567K`), eliminates late-epoch overfitting (`NDCG@10` holds `0.63485 → 0.63457` through epoch 10), boosts cold-start `NDCG@10` (`+0.0153` over `v4b`), and sets new resolved exact-SKU records while scoring all `585` formulations in `1.32 ms` on CPU.

### 2. Final Held-Out Test Evaluation (`2026-02-19` to `2026-07-17`, Inference-Only on Archived 3-Stage Reference)

Strictly inference-only evaluation on the five-month forward Test split (`stage5-dated-first-baskets-v2`, primary NDCG checkpoint Ep 7; zero retraining/tuning; `stage5-fullpool-nomlp-v3` was developed strictly on validation and has not touched Test):

| Cohort / Metric | Validation (`6,480`) | Test — Modeled `585`-BN (`6,497`) | Test — Full Catalogue (`6,582`) |
| :--- | ---: | ---: | ---: |
| **All — Recall@10** | `80.78%` | `79.89%` | `78.56%` |
| **All — NDCG@10** | `0.63345` | `0.61852` | `0.60895` |
| **All — Truly New Hits@10** | `359 / 1,721` (`20.86%`) | `353 / 1,779` (`19.84%`) | `353 / 1,913` (`18.45%`) |
| **Returning — Recall@10 / NDCG@10** | `82.69%` / `0.65194` | `81.57%` / `0.63408` | `80.24%` / `0.62452` |
| **First Observed — Recall@10 / NDCG@10** | `39.75%` / `0.23750` | `37.72%` / `0.22811` | `36.81%` / `0.22264` |
| **100-Slate Coverage (Total / New)** | `97.06%` / `82.68%` | `97.12%` / `83.70%` | `95.57%` / `77.84%` |
| **Resolved Exact-SKU Recall@10 / NDCG@10** | `66.62%` / `0.53376` | `65.93%` / `0.51880` | `64.15%` / `0.50595` |

*Full-Catalogue Test counts all `166` targets from `58` new formulations launched after `2026-02-18` as misses (`85` test baskets contain only newly launched formulations). See [`experiments/roadmap_v2/REPORT.md`](experiments/roadmap_v2/REPORT.md) and [`stage5_dated_first_baskets_test_seed42.json`](experiments/roadmap_v2/stage5_dated_first_baskets_test_seed42.json).*

---

## Repository Structure

```text
.
├── configs/
│   ├── datasets_links.toml                # Local paths to raw dataset.xlsx, customers.csv, items.csv
│   └── product_scope.toml                 # Paid-sale & sellable merchandise rules (ПРОДАЖА + ТОВАР)
├── recommender/                           # Core library & active experimental pipelines
│   ├── data.py, merging.py, scope.py, preparation.py
│   │                                      # Raw Excel/CSV parsing, paid-sale filtering, quantity-summing merge
│   ├── company.py, catalogue.py           # Company & item master loading, RollingCandidatePool
│   ├── split.py, development.py           # Strict temporal splits (TRAIN_CUTOFF=2025-10-01, TEST_START=2026-02-19)
│   ├── features.py, sequences.py, text.py, familiarity.py
│   │                                      # Point-in-time profile normalizer, 32-event sequences, BGE-M3 cache, familiarity
│   ├── models.py, candidates.py, ranker.py
│   │                                      # Base TwoTowerModel, PointInTimeFeatureStore, UPSCANRRanker & RankerLoss
│   ├── train_retriever.py, train_ranker.py, evaluate.py, evaluate_stage1.py
│   │                                      # Legacy SKU-level training & evaluation entrypoints
│   └── experiments/
│       ├── roadmap_v2/                    # ACTIVE Stage 5 canonical base_name pipeline
│       │   ├── dated_baskets.py           # Unified dated first + returning basket builder & commercial reconciler
│       │   ├── stage5_base_entity.py      # 585 base_name universe, base_name sequences, evaluator, SKU resolver
│       │   ├── base_name_features.py      # 19D strictly-causal base_name rolling/transition/complement builder
│       │   ├── commercial_features.py     # 3A financial, 3B pricing, 3C rolling 90d manager, 3D warehouse builders
│       │   ├── parsed_spec_features.py    # 3E deterministic viscosity/pack/application & company equipment matcher
│       │   ├── augmentation.py            # Leak-free 60-day forward auxiliary label builder (< TRAIN_CUTOFF)
│       │   ├── stage1_mlp.py              # RoadmapStage1Selector (used in 3-stage comparison)
│       │   ├── run_stage5.py              # Canonical 3-stage Stage 5 trainer & --replay-reference verifier
│       │   ├── run_stage5_ablations.py    # No-MLP full-pool UPS-CANR & CatBoost (standalone / +Two-Tower) ablations
│       │   ├── run_stage5_bigger_ranker.py# Modest & scaled No-MLP full-pool UPS-CANR (active reference trainer)
│       │   ├── evaluate_stage5_test.py    # Locked (--allow-heldout-test) inference-only held-out Test evaluator
│       │   └── demo_dashboard.py          # Interactive local web dashboard sampling live held-out Test baskets
│       ├── dynamic_selector/              # 41D rolling feature schema, DynamicFeatureBuilder, slate selection
│       ├── rolling_candidate_v2/          # rolling_observed_v2 feature bundle & preserved-hash guardrails
│       ├── static_enrichment/             # StaticEnrichedRetriever (E_no_brand Two-Tower), t1_fam text cache
│       └── catboost_comparison/           # Historical & standalone SKU-level CatBoost comparisons
├── experiments/
│   └── roadmap_v2/                        # Tracked JSON summaries, reference manifests, and REPORT.md
│       ├── stage5_reference.json          # Active canonical reference (stage5-fullpool-nomlp-v3)
│       ├── stage5_3stage_reference.json   # Archived 3-stage reference (stage5-dated-first-baskets-v2)
│       ├── stage5_modest_ranker_seed42.json
│       ├── stage5_nomlp_and_catboost_seed42.json
│       ├── stage5_dated_first_baskets_seed42.json
│       ├── stage5_dated_first_baskets_test_seed42.json
│       └── REPORT.md                      # Complete Stage 0–5, Held-Out Test, No-MLP & CatBoost report
├── tests/                                 # 634 unit, causality, leakage-guard & parity tests (6 protected skips)
├── AGENTS.md                              # Agent engineering handbook & strict non-leakage rules
├── EVALUATION_RESULTS.md                  # Historical SKU validation record
└── plan.md                                # Project roadmap & milestone log
```

---

## Setup & Data Preparation

Use Python 3.12 (`./venv/bin/python`; Apple Silicon `mps` is detected automatically when available):

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r requirements-dev.txt
# Optional: for CatBoost benchmarks
./venv/bin/pip install -r requirements-catboost.txt
```

Private raw inputs live locally under `data/raw/` (git-ignored, configured in [`configs/datasets_links.toml`](configs/datasets_links.toml)):
`data/raw/dataset.xlsx`, `data/raw/customers.csv`, `data/raw/items.csv`.

```bash
# Export cleaned paid merchandise purchases to data/interim/cleaned_purchases.csv
./venv/bin/python -m recommender --export-purchases data/interim/cleaned_purchases.csv
```

- **Chronological splits ([`recommender/split.py`](recommender/split.py)):**
  - **Train:** `< 2025-10-01` (`50,414` events; `29,095` dated Stage 5 baskets = `2,984` first observed + `26,111` returning)
  - **Validation:** `2025-10-01` to `2026-02-18` (`10,634` events; `6,480` dated Stage 5 baskets = `289` first observed + `6,191` returning)
  - **Held-Out Test:** `>= 2026-02-19` (`10,868` events; `6,582` baskets = `255` first observed + `6,327` returning — **strictly locked behind explicit opt-in flags**)

---

## Training, Replay & Test Commands

Run from the repository root with `./venv/bin/python`:

```bash
# Train the active No-MLP Full-Pool UPS-CANR variants (modest_wider_mlp_384 & modest_8h_ffn384_mlp384)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5_bigger_ranker --mode modest

# Run No-MLP standard full-pool UPS-CANR + CatBoost (standalone & +Two-Tower) validation ablations
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5_ablations

# Replay and checksum-verify the archived 3-stage Stage 5 reference on validation
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.run_stage5 --replay-reference

# Run authorized inference-only held-out Test evaluation (refuses without --allow-heldout-test)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.evaluate_stage5_test --allow-heldout-test

# Launch the interactive live held-out Test demo dashboard (http://127.0.0.1:8765)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.roadmap_v2.demo_dashboard --allow-heldout-test --port 8765

# Run the test suite (ordinary pytest skips 6 protected real-data readers; zero held-out Test access)
PYTHONPATH=. ./venv/bin/pytest tests/ -q
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
```

---

## Documentation Map & Historical Audits

- **[experiments/roadmap_v2/REPORT.md](experiments/roadmap_v2/REPORT.md):** Full Stage 0–5 ablation progression, dated first/returning basket audit, final held-out Test evaluation, No-MLP full-pool vs. CatBoost comparison, and modest scaling results.
- **[AGENTS.md](AGENTS.md):** Active reference pointers, module map, checkpoint/cache provenance, and temporal-leakage guardrails.
- **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)** & **[plan.md](plan.md):** Historical SKU validation progression and roadmap context.
- **[AUDIT_REPORT.md](AUDIT_REPORT.md)**, **[HARDENING_REPORT.md](HARDENING_REPORT.md)**, **[DYNAMIC_SELECTOR_EXPERIMENTS.md](DYNAMIC_SELECTOR_EXPERIMENTS.md)**, **[SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md)**, **[E1_E5_EXPERIMENTS.md](E1_E5_EXPERIMENTS.md):** Historical SKU-level audits, hardening changes, and controlled ablation records (preserved untouched).

## Data Handling

Raw data (`data/raw/`), interim caches (`data/interim/`), cached embeddings, precomputed predictions (`.npz`), and model checkpoints (`checkpoints/`) remain strictly local and git-ignored. Only aggregate JSON summaries, reference manifests, code, and tests are tracked in Git.
