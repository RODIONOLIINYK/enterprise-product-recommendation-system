# Enterprise Product Recommendation System

Next-basket B2B product formulation recommender for industrial distribution, built around **canonical `base_name` entity unification**, **strictly point-in-time rolling commercial/behavioral features (`rolling_observed_v2`)**, and a **2-stage No-MLP Full-Pool `UPS-CANR` Neural Reranker** (with a standalone CatBoost tabular benchmark).

---

## Current Architecture & End-to-End Pipeline (`Stage 5: stage5-fullpool-nomlp-v3`)

In B2B industrial distribution (lubricants, coolants, chemicals), the same product formulation is sold in multiple package sizes (e.g., 4L jug, 20L pail, 209L drum). Rather than letting 4–5 packaging variants of the same oil compete for Top-10 slots, the active **Stage 5** pipeline unifies **`841` development SKUs into `585` canonical `base_name` product formulations** across every stage, handles **both returning customers and cold-start (first-observed) companies** in one dated pipeline, and eliminates the intermediate Stage 1 MLP shortlist by scoring all eligible formulations directly.

```text
Raw ERP / NAV Sales (dataset.xlsx) + Companies (customers.csv) + Catalogue (items.csv)
  │
  ▼  1. Paid-Sale & Multi-Line Settlement Reconciliation
     • Filters to positive paid merchandise sales (ПРОДАЖА + ТОВАР; excludes gifts/returns)
     • Merges same-(company, date, SKU) lines (summed quantity); 76,797 paid lines ↔ 71,916 cleaned rows
     • Multi-line basket counts as paid only after EVERY paid line settles strictly < prediction date
     • Collapses 841 dev SKUs → 585 canonical base_name formulations (earliest paid pack anchors static metadata)
  │
  ▼  2. Strictly Causal Point-in-Time Feature & Sequence Layer (rolling_observed_v2, strictly < t)
     • 32-event chronological base_name purchase sequence (same-day multi-packs collapsed to 1 event; 0 tokens for first orders)
     • 8D normalized customer lifetime profile + 1024D BAAI/bge-m3 company text + city/region/industry/sales_branch
     • 55 candidate features per (company, date, base_name):
         - 41D base_name rolling repeat, cadence/urgency, global/industry popularity, velocity, transition & same-basket lifts
         -  6D 3a_financial: prepaid/cash ratio, avg/last contract deferral, actual/contract pay ratio, overdue unpaid ratio, bonus rate
         -  4D 3c_manager: rolling 90-day behavioral shares of the customer's current manager (base_name, brand, category) + manager change flag
         -  4D 3e_parsed_specs: regex-parsed viscosity grade history/share, company equipment-to-application match, application order share
  │
  ▼  3. Stage 1 — Pack-Agnostic Two-Tower Encoder (E_no_brand recipe, zero demand features)
     • Encodes earliest-paid-pack t1_fam BGE-M3 text + 6 categorical attributes (category, subcategory, business_line, tier, unit, origin)
     • Produces cached 128D L2-normalized formulation vectors v_bn (585 × 128)
     • No Stage 1 MLP & no 100-item shortlist truncation (100% eligible candidate coverage)
  │
  ▼  4. Stage 2 — Direct Full-Pool UPS-CANR Transformer Reranker (active ref: modest_wider_mlp_384, 769K params)
     • 33-token Customer Memory: Token 0 (Company Profile) + Tokens 1..32 (base_name Purchase History) → 1-layer TransformerEncoder
     • 4-head Statistics-Biased Cross-Attention (candidate queries attend to 33 memory tokens with bilinear feature-to-history bias)
     • Scoring MLP trunk: [z_cand (128), h_attended (128), z_cand ⊙ h_attended (128), 55 features] = 439D → 384 → 192 → 96 → 1
     • Trained with RankerLoss + training-only 60-day forward auxiliary head (λ = 0.3, stripped at inference)
     • Scores all 585 eligible formulations directly in ~1.2 ms/request on CPU (3.3s for 6,480 baskets on MPS)
  │
  ▼  5. Top-10 Ranked Product Formulations (+ Optional Causal Pack-Size Resolver → Concrete Eligible SKU)
     • Per recommended base_name: customer's most frequent/recent eligible pack < t → dominant container type < t → rolling popular pack < t
```

- **Active Reference Manifest:** [`experiments/roadmap_v2/stage5_reference.json`](experiments/roadmap_v2/stage5_reference.json) (`stage5-fullpool-nomlp-v3`, `seed=42`, trained on `29,095` dated baskets = `2,984` first observed + `26,111` returning; validated on `6,480` baskets = `289` first observed + `6,191` returning).
- **Archived 3-Stage Reference (Evaluated on Held-Out Test):** [`experiments/roadmap_v2/stage5_3stage_reference.json`](experiments/roadmap_v2/stage5_3stage_reference.json) (`stage5-dated-first-baskets-v2`, which included Stage 1 MLP + `quota100_50` 100-item slate).

---

## Verified Performance

### 1. Validation Benchmarks (`2025-10-01` to `2026-02-18`, `6,480` baskets, `10,163` targets, `1,721` truly new — Zero Test Access)

| Model (`585` Canonical Formulations, `3a+3c+3e`) | Pipeline | `R@5` | `R@10` | `NDCG@10` | `New@10` *(of `1,721`)* | First-Obs `NDCG@10` | Returning `New@10` | 1-Req CPU Latency |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Archived 3-Stage Reference (NDCG, Ep 7) | 2T → S1 MLP → S2 (`100`-slate) | `70.51%` | `80.78%` | `0.63345` | `359` (`20.86%`) | `0.23750` | `212` | — |
| Archived 3-Stage Reference (Disc, Ep 3) | 2T → S1 MLP → S2 (`100`-slate) | `70.54%` | `80.93%` | `0.63232` | `367` (`21.32%`) | `0.25371` | `211` | — |
| No-MLP Full-Pool `UPS-CANR` (Std, Disc Ep 4) | 2T → Full-Pool S2 (`585` BNs) | `70.47%` | `80.88%` | `0.63337` | `371` (`21.56%`) | `0.25717` | `219` | `1.16 ms` |
| **No-MLP `modest_wider_mlp_384` (Active Ref, Ep 7)** | **2T → Full-Pool S2 (`585` BNs)** | `70.54%` | `80.85%` | **`0.63428`** | `366` (`21.27%`) | `0.24414` | `222` | `~1.2 ms` |
| **No-MLP `modest_8h_ffn384_mlp384` (Discovery Ref, Ep 5)** | **2T → Full-Pool S2 (`585` BNs)** | `70.62%` | **`80.96%`** | `0.63321` | **`378`** (`21.96%`) | `0.23983` | **`230`** | `~1.4 ms` |
| **CatBoost Standalone (`72` feats, `661` trees)** | **Pure Tabular Full-Pool (0 neural)** | **`70.69%`** | `80.70%` | `0.63403` | `356` (`20.69%`) | **`0.27037`** | `204` | **`0.55 ms`** *(1.39ms w/ Pool)* |

- **Resolved Exact-SKU Validation (Active Ref `modest_wider_mlp_384`, Ep 7):** `sku_Recall@10 = 66.66%`, `sku_NDCG@10 = 0.53410`, `sku_New@10 = 203`.
- **Takeaway on Neural vs. CatBoost:** Full-pool `UPS-CANR` (`modest_wider_mlp_384` / `modest_8h_ffn384_mlp384`) leads on pooled `NDCG@10` (`0.63428`), `Recall@10` (`80.96%`), and returning-customer true discovery (`222–230` vs `204`), while pure tabular CatBoost leads on cold-start (`First Observed NDCG@10 = 0.27037`) and CPU serving simplicity (zero PyTorch/BGE-M3 runtime).

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
