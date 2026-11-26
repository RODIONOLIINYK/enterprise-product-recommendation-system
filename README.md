# Enterprise B2B Product Recommendation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CPU Latency](https://img.shields.io/badge/CPU_Inference-3.02_ms_(314_req%2Fs)-059669)](#headline-results)
[![Held-Out Test](https://img.shields.io/badge/Held--Out_Test_Recall@10-83.03%25_(Trained)_%7C_76.63%25_(All)-2563EB)](#headline-results)

End-to-end **next-basket, cross-sell, and cold-start product recommendation system** built for B2B industrial distribution over **4 years of real ERP/NAV sales (`167,572` orders, `1,660` SKUs unified into `1,141` canonical product formulations)**.

The final production architecture (**`Wave2BE2ERanker` · `w2_2mem_recency_hl365`**) eliminates separate Stage 1 retrieval and top-K shortlist truncation altogether: a single end-to-end **2-layer Transformer + 8-head Statistics-Biased Cross-Attention** network scores **every eligible product in the catalogue in `3.02 ms` on CPU (no GPU)** while combining a repeat-replenishment head with a dedicated **Unseen Products Head** for first-time cross-sell discovery.

> **Full Engineering Timeline:** See [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) for the complete chronological progression across all `70+` commits — from **`17.88%`** Global Popularity and early CatBoost prototypes up to **`84.27%` Validation / `83.03%` Held-Out Test Recall@10**.

---

## Headline Results

Evaluated under **strict chronological splits with zero future leakage** (`Train < 2026-02-01` · `Validation 2026-02-02..2026-06-30` · `Held-Out Test 2026-07-01..2026-11-20`). Every feature, rolling counter, settlement status, and sequence token is computed strictly `< prediction_date`.

| Evaluation Split & Slice | Orders | Recall@10 | NDCG@10 | New-to-Customer @10 | Pure Cross-Sell @10 | Temporal Gap from Training |
| :--- | ---: | :---: | :---: | :---: | :---: | :--- |
| **Validation · Trained Customers & Products** | `15,343` | **84.27%** | **0.6718** | `20.52%` (`582`) | — | `0 – 5.0 months` |
| **Held-Out Test · Trained Customers & Products** | `9,839` | **83.03%** | **0.6349** | `15.69%` (`248`) | **43.14%** | **`7.3 months` avg** *(up to `9.7 mo`)* |
| **Validation · Full Catalogue (All Orders)** | `16,623` | **82.26%** | **0.6548** | **27.91%** (`1,255`) | **51.78%** | `0 – 5.0 months` |
| **Held-Out Test · Full Catalogue (All Orders)** | `13,184` | **76.63%** | **0.5841** | **21.71%** (`853`) | **42.78%** | **`7.4 months` avg** *(up to `9.7 mo`)* |

### Key Verified Findings
- **Zero Overfitting Across a 7.4-Month Horizon:** The average held-out Test order (`2026-09-12`) occurs **`225 days` (`7.4 months`)** after the last training purchase (`2026-01-30`), extending up to **`294 days` (`9.7 months`)** on `2026-11-20`. Trained-entity Recall@10 shows zero monthly decay: **`Jul 82.33% → Aug 83.07% → Sep 83.72% → Oct 83.20% → Nov 82.88%`**.
- **Outperforms Strong GBDT Baselines:** Beats full-catalogue CatBoost on both `v1` (`77.10%` vs `75.82%`, `+53` new hits) and `v2_full` (`82.26%` / `0.6548` NDCG vs CatBoost `81.61%` / `0.6438` NDCG, `+116` new hits).
- **365-Day Exponential Recency Weighting Delivers `+5.98 pp` Zero-Shot Gain on Held-Out Test:** Head-to-head against the identical 2-memory architecture trained with uniform weights (`w2_2mem_blocks`), `w2_2mem_recency_hl365` wins **`+0.79 pp`** on Full Catalogue (`76.63%` vs `75.84%`), **`+3.31 pp`** across all unlearned orders (`57.80%` vs `54.49%`), and **`+5.98 pp`** on zero-shot `ProductTower` unlearned products (`66.29%` vs `60.31%`, `+58` new hits) with zero extra parameters or latency.
- **Production Serving Speed (No GPU Required):** **`3.02 ms` median** neural net latency (**`3.38 ms`** full E2E including features + SKU pack resolution, **`314 orders/sec`** on 4 CPU threads) scoring all `1,141` products simultaneously.
- **Low-Cost Production Maintenance:** Retraining takes only minutes on a single GPU; scheduled retraining every **2 months** (or whenever **~10 new products** launch) minimizes unlearned entities and sustains production accuracy near the **`~84–85% Recall@10`** trained-entity regime.

---

## System Architecture (`Wave2BE2ERanker`)

In industrial distribution (lubricants, coolants, chemicals), a single formulation is sold in multiple containers (e.g., 4L jug, 20L pail, 209L drum). Unifying **`1,660` SKUs into `1,141` canonical `base_name` formulations** prevents pack variants from crowding Top-10 slots, while a deterministic causal pack resolver selects each customer's preferred SKU at the end (`< 0.15 ms`).

```mermaid
flowchart LR
    subgraph Inputs["Strictly Causal Inputs (< prediction_date)"]
        CF["Customer Profile\nBGE-M3 1024D + City/Region/Industry\n+ CUST_7 (activity & payment health)"]
        PD["Product Catalogue (1,141)\nPack-Free BGE-M3 1024D\n+ 65D Multi-Hot Tech Specs + 6 Attrs"]
        C42["42D Causal Candidate Features\nRepeat cadence, CF cosine/recency,\nMgr 30d/90d, Region 90d, Seq lift, Specs"]
    end

    subgraph Encoders["End-to-End Encoders (Zero Stage 1 Pretraining)"]
        CT["Customer Profile Encoder\ncust_text_proj + cust_tab_proj\n→ token0_fusion (128D)"]
        PT["ProductTower (Zero Item IDs)\n1,141 × 128D v_bn\n(cached at startup: 1.4 ms)"]
    end

    subgraph Memory["Customer Memory & Cross-Attention"]
        T0["Token 0: Customer Profile"]
        H32["Tokens 1..32: Purchase History\nv_bn[seq] + pos + SEQ_5"]
        TF["2-Layer 8-Head Transformer\n(33 × 128D, FFN=384)"]
        CA["8-Head Pure-Q Cross-Attention\nQ = v_bn · Full-42 Bilinear Bias\n(cand_42 × [SEQ_5 | CUST_7])"]
    end

    subgraph Heads["Specialist Scoring Heads & Calibration"]
        MH["Main Head (Repeats)\n[v_bn, h_att, v_bn⊙h_att, cand_42]"]
        UH["Unseen Products Head (Cross-Sell)\n[v_bn, h_att, v_bn⊙h_att, non_repeat_37]"]
        CAL["2-Param OOF Affine Blend\n+ Causal Pack-Size SKU Resolver\n→ Top-10 Recommendations (3.0 ms)"]
    end

    CF --> CT --> T0 --> TF
    CF -.->|CUST_7 key stats| CA
    PD --> PT
    PT -->|v_bn[seq]| H32 --> TF
    PT -->|Pure Query Q = v_bn| CA
    TF --> CA
    C42 -->|Full-42 Bias| CA
    CA --> MH --> CAL
    CA --> UH --> CAL
    C42 --> MH
    C42 -->|37 non-repeat| UH
```

### Architectural Highlights
- **Single-Stage End-to-End Design (`1.13M` per-request / `1.37M` total params):** `ProductTower` and the Token 0 customer encoder train jointly with the Transformer ranker — removing two-stage representation mismatch and offline Stage 1 pretraining.
- **Zero Per-Product ID Embeddings:** `ProductTower` relies purely on semantic text (`BAAI/bge-m3`), `65D` structured specs (container, volume tiers, application domains, viscosity grades), and 6 categorical attributes, enabling **zero-shot encoding of newly launched products** without retraining.
- **Cleanly Separated Specialist Heads:** Conflict-free targets let the **Main Head** optimize repeat replenishment while the **Unseen Products Head** (using strictly the 37 non-repeat candidate features) specializes in first-time product discovery and cold-start orders, merged via a 2-parameter customer-grouped out-of-fold affine calibration (`score = s_main` for prior items, `a · s_unseen + b` for unseen items).
- **Training-Only 60-Day Multi-Task Auxiliary Loss (`λ = 0.3`):** Multiplies sparse unseen positive supervision `3.3×` during training and is stripped completely at inference (`0` extra latency).

---

## Model Evolution (From Scratch to Final Release)

Full commit-by-commit breakdown with dates, ablations, and rejected hypotheses is in [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).

| Stage | Milestone & Key Innovation | Validation Recall@10 | NDCG@10 | New-to-Customer @10 |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Global Popularity Baseline (`a0f5407`, strict temporal split) | `17.88%` | `0.0925` | — |
| **2** | First Modular Two-Tower Retriever (`a0f5407`, BGE-M3 + history Transformer) | `47.60%` | `0.3007` | — |
| **3** | First `UPS-CANR` Neural Ranker (`a0f5407`, 33-token memory + cross-attention) | `66.69%` | `0.5347` | — |
| **4** | Standalone CatBoost v1 (`52` features, `318` trees, full-catalogue benchmark) | `75.82%` | `0.5937` | `10.88%` (`202`) |
| **5** | Discovery Reference (4 familiarity flags) + Statistics-Biased Attention (`4f3d326`) | `76.15%` | `0.5924` | `11.25%` (`209`) |
| **6** | Rolling `< as_of_date` Causality Audit & 100-Candidate Slates (`22f2142`) | `77.10%` | `0.6022` | `13.73%` (`255`) |
| **7** | Canonical `base_name` Unification + Full-Pool Stage 5 (`e0681bd`, `585` prod.) | `80.89%` | `0.6346` | `21.62%` (`372`) |
| **8** | CatBoost `v2_full` (`42` features + Unseen Products Head OOF blend) | `81.61%` | `0.6438` | `25.33%` (`1,139`) |
| **9** | Single-Stage E2E `Wave2BE2ERanker` + `cand_42` + 2× Memory (`w2_2mem_blocks`) | `82.18%` | `0.6542` | `27.31%` (`1,228`) |
| **10** | **Final Promoted: 2× Memory + 365d Recency (`w2_2mem_recency_hl365`) ⭐** | **`82.26%`** *(all)*<br>**`84.27%`** *(Trained)* | **`0.6548`**<br>**`0.6718`** | **`27.91%`** (`1,255`)<br>Pure Cross-Sell: **`51.78%`** |

---

## Interactive Live Python Demo & Engineering Stack

The repository includes a **zero-dependency standalone PyTorch web server** ([`live_demo_server.py`](recommender/experiments/v2_full/live_demo_server.py)) that loads `best_discovery.pt` (`strict=True`) and serves live single-order CPU inference over all **`13,184` held-out Test orders** (`0.09 s` startup via zero-copy `float16` memmap):

- **Cohort Inspection:** Switch seamlessly between **Trained Customers & Products (`9,839`)**, **Unlearned Entities (`3,345` with automatic `⚠️` deployment warning banner)**, **Cold-Start (`317`)**, and **Returning with New Items (`2,875`)**.
- **Live 100-Order CPU Benchmark:** Runs 100 real Test orders sequentially on local CPU right from the UI and reports live median/P95 latency, throughput (`orders/sec`), and accuracy.
- **Interactive 3D Architecture Explorer:** Clickable 11-block 3D canvas with expandable feature, tensor-shape, and regularization inspectors (`DOCUMENTATION.html`).

---

## Repository Structure

```text
.
├── PROJECT_HISTORY.md                         # Full chronological engineering & metric log (70+ commits)
├── EVALUATION_RESULTS.md                      # Detailed validation tables & archived rejected hypotheses
├── recommender/
│   ├── data.py, merging.py, scope.py, preparation.py
│   ├── company.py, catalogue.py, split.py, development.py
│   ├── features.py, sequences.py, text.py, familiarity.py, ranker.py
│   └── experiments/
│       ├── v2_full/                           # ACTIVE 4-year single-stage E2E pipeline
│       │   ├── config.py                      # Chronological splits & active reference loader
│       │   ├── prepare_v2_full.py             # Leak-free 4-year ingestion & 1,141 base_name universe
│       │   ├── extra_features.py              # Strictly causal extra_cand_5 & cand_42/CUST_7/SEQ_5
│       │   ├── wave2b_models.py               # Wave2BE2ERanker, Unseen Products Head & OOF calibration
│       │   ├── run_v2_full_wave2.py           # Wave 2A 2-memory & 365d recency training runner
│       │   ├── run_v2_full_catboost.py        # 42-feature CatBoost benchmark + OOF blend
│       │   ├── evaluate_v2_full_test.py       # Authorized inference-only held-out Test evaluator
│       │   ├── build_live_test_cache.py       # Builds memmap 13,184-order cache & 1,000-order bundle
│       │   ├── live_demo_server.py            # Standalone PyTorch CPU web server + live benchmark UI
│       │   ├── build_portable_demo.py         # Packages standalone Live App & offline demo ZIPs
│       │   └── demo_docs_uk.py                # Interactive 3D architecture docs & offline deck builder
│       ├── roadmap_v2/                        # Shared formulation, commercial, spec & Stage 5 modules
│       └── static_enrichment/                 # StaticEnrichedProductTower & enriched metadata loaders
├── experiments/
│   ├── v2_full/                               # Active reference manifest, validation & held-out Test JSONs
│   └── roadmap_v2/                            # Frozen v1 Stage 5 reference manifests & REPORT.md
├── notebooks/
│   └── colab_v2_full_experiments.ipynb        # Native bf16 CUDA L4/A100 training notebook
└── tests/                                     # Causality, parity, leakage-guard & regression test suite
```

---

## Quickstart & Commands

```bash
# 1. Environment setup (Python 3.12; Apple Silicon MPS / CUDA bf16 auto-detected)
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt -r requirements-dev.txt

# 2. Launch standalone Live Python PyTorch server over all 13,184 held-out Test orders (CPU ~3 ms/req)
./venv/bin/python recommender/experiments/v2_full/live_demo_server.py \
    --data-dir data/v2_full/interim/test_live_cache --port 8799

# 3. Package standalone runnable Live App ZIP (model + backend + website + all 13,184 Test data)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.build_portable_demo --mode live_app

# 4. Prepare data & train active Wave 2A model (use --amp --in-memory on CUDA bf16)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.prepare_v2_full
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.extra_features
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.run_v2_full_wave2 --seed 42 --amp --in-memory

# 5. Run causality, architecture & leakage-guard tests
PYTHONPATH=. ./venv/bin/pytest tests/test_v2_full_pipeline.py tests/test_v2_full_wave2b.py \
    tests/test_v2_full_wave2c.py tests/test_v2_full_wave2.py tests/test_stage5_audit_fixes.py tests/test_data.py -q
```

---

## Data Privacy & Reproducibility

Proprietary ERP exports (`data/`), BGE-M3 embedding caches, model checkpoints (`checkpoints/`), and packaged ZIP archives (`exports/`) remain strictly local and git-ignored. All code, unit tests, SHA-256 reference manifests, and aggregate JSON evaluation reports are tracked in Git.
