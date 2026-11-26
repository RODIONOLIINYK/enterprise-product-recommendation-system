# Enterprise B2B Product Recommendation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CPU Inference](https://img.shields.io/badge/CPU_Serving-3.02_ms_(314_req%2Fs)-059669)](#headline-results)
[![Held-Out Test](https://img.shields.io/badge/Held--Out_Test_Recall@10-83.03%25_(Trained)_%7C_76.63%25_(All)-2563EB)](#headline-results)

End-to-end **next-basket, cross-sell, and cold-start recommendation system** built for B2B industrial distribution over **4 years of real ERP/NAV sales (`167,572` dated orders, `711K` raw invoice lines, `1,660` SKUs unified into `1,141` canonical product formulations)**.

The final production model (**`Wave2BE2ERanker` · `w2_2mem_recency_hl365`**) replaces traditional multi-stage retrieval-and-rerank pipelines with a **single-stage end-to-end Transformer + 8-head Statistics-Biased Cross-Attention** network (`1.13M` per-request params). It scores **all `1,141` catalogue products simultaneously in `3.02 ms` on CPU (no GPU)**, handles returning and zero-history cold-start clients in one forward pass, and encodes newly launched products zero-shot via semantic embeddings and technical specs.

> **Full Engineering Log:** See [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) for the complete chronological commit-by-commit progression (`70+` commits) from **`17.88%`** Global Popularity to **`84.27%` Validation / `83.03%` Held-Out Test Recall@10**.

---

## Headline Results

Evaluated under **strict chronological splits with zero future leakage** (`Train < 2026-02-01` · `Validation 2026-02-02..2026-06-30` · `Held-Out Test 2026-07-01..2026-11-20`). Every feature, rolling counter, payment settlement, and history token is computed strictly `< prediction_date`.

| Evaluation Split & Slice | Orders | Recall@10 | NDCG@10 | New-to-Customer @10 | Pure Cross-Sell @10 | Gap After Last Training Sale |
| :--- | ---: | :---: | :---: | :---: | :---: | :--- |
| **Validation · Trained Customers & Products** | `15,343` | **84.27%** | **0.6718** | `20.52%` (`582`) | — | `0 – 5.0 months` |
| **Held-Out Test · Trained Customers & Products** | `9,839` | **83.03%** | **0.6349** | `15.69%` (`248`) | **43.14%** | **`7.3 months` avg** *(up to `9.7 mo`)* |
| **Validation · Full Catalogue (All Orders)** | `16,623` | **82.26%** | **0.6548** | **27.91%** (`1,255`) | **51.78%** | `0 – 5.0 months` |
| **Held-Out Test · Full Catalogue (All Orders)** | `13,184` | **76.63%** | **0.5841** | **21.71%** (`853`) | **42.78%** | **`7.4 months` avg** *(up to `9.7 mo`)* |

- **Zero Overfitting Across 7.4–9.7 Months:** The average held-out Test order (`2026-09-12`) lies **`225 days` (`7.4 months`)** after the last training sale (`2026-01-30`), extending to **`294 days` (`9.7 months`)** on `2026-11-20`. Trained-entity Recall@10 remains flat across all 5 test months: **`Jul 82.33% → Aug 83.07% → Sep 83.72% → Oct 83.20% → Nov 82.88%`**.
- **Beats Strong GBDT Baselines:** Outperforms full-catalogue CatBoost on both `v1` (`77.10%` vs `75.82%`, `+53` new hits) and `v2_full` (`82.26%` / `0.6548` NDCG vs CatBoost `81.61%` / `0.6438` NDCG, `+116` new hits).
- **Controlled Recency Ablation on Held-Out Test:** Compared head-to-head against the identical 2-memory architecture trained with uniform sample weights (`w2_2mem_blocks`), 365-day exponential recency weighting (`w2_2mem_recency_hl365`) improves Full Catalogue by **`+0.79 pp`**, All Unlearned orders by **`+3.31 pp`** (`57.80%` vs `54.49%`), and zero-shot `ProductTower` unlearned products by **`+5.98 pp`** (`66.29%` vs `60.31%`, `+58` new hits) with zero latency overhead.
- **CPU Serving & Cheap Maintenance:** **`3.02 ms` median** neural inference (**`3.38 ms`** full E2E with features + SKU pack resolution, **`314 orders/sec`** on 4 CPU threads without GPU). Retraining every **2 months** (or whenever **~10 new products** launch) takes minutes on one GPU and keeps live accuracy near the **`~84–85% Recall@10`** trained-entity regime.

---

## Technology Stack

| Layer | Technologies & Implementation Details |
| :--- | :--- |
| **Deep Learning & Training** | **PyTorch 2.x** (`nn.TransformerEncoder`, `MultiheadAttention`, custom bilinear attention bias), **CUDA `bf16` AMP** (Colab L4/A100), **Apple Silicon `MPS`**, `AdamW`, cosine annealing |
| **Multilingual NLP & Embeddings** | **HuggingFace Transformers / `BAAI/bge-m3`** (`1024D` L2-normalized semantic embeddings for Ukrainian/Russian/English product & company descriptions) |
| **Tabular & GBDT Benchmarks** | **CatBoost** (42-feature & 52-feature full-catalogue baselines + OOF calibration ensembles) |
| **Data & Feature Engineering** | **Python 3.12**, **NumPy** (zero-copy `float16` memory-mapped feature tensors), **Pandas**, **PyArrow / Parquet**, streaming `xml.etree.ElementTree` ERP workbook parser (`711K` rows in `45s`) |
| **Serving & Interactive Demo** | Standalone zero-dependency **Python `ThreadingHTTPServer`** + **PyTorch CPU** (`live_demo_server.py`), HTML5/CSS/JS frontend, interactive **3D `<canvas>`** architecture inspector, Matplotlib |
| **Quality & Reproducibility** | **pytest** (`600+` unit, causality, parity, and leakage-guard tests), **ruff**, deterministic seeding, **SHA-256** dataset/cache/checkpoint provenance manifests |

---

## System Architecture (`Wave2BE2ERanker`)

```mermaid
flowchart LR
    subgraph Inputs["Strictly Causal Inputs (< prediction_date)"]
        CF["Customer Profile\nBGE-M3 1024D + City/Region/Industry\n+ CUST_7 (activity & payment health)"]
        PD["Product Catalogue (1,141)\nPack-Free BGE-M3 1024D\n+ 65D Multi-Hot Specs + 6 Attrs"]
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

---

## Training Methodology, Data Augmentation & Loss Design

1. **60-Day Forward Multi-Task Auxiliary Supervision (`λ = 0.3`):**
   - Next-basket new-product purchases are sparse (`~0.42` unseen positives/order). For every training basket at date `t`, we construct strictly leak-free auxiliary labels over `[t, min(t + 60d, TRAIN_CUTOFF))`, multiplying unseen positive supervision **`3.34×`** (`0.42 → 1.40` positives/query). Auxiliary heads are discarded at inference (**`0` extra parameters, `0 ms` latency**).
2. **Composite Ranking Loss (`WeightedRankerLoss`):**
   - Combines **Multi-Label BCE** (`pos_weight = 10` to handle `~1:500` class imbalance across `1,141` products) + **Listwise Softmax Cross-Entropy** over all eligible formulations.
3. **Conflict-Free Specialist Head Separation:**
   - Standard single-head rankers suffer gradient interference between high-frequency repeat purchases and rare first-time discoveries. Separating the **Main Head** (supervised on returning repeats, `cand_42`) from the **Unseen Products Head** (supervised strictly on unseen eligible items using only the `37` non-repeat features) lifted New@10 from `11.25% → 27.91%` and Pure Cross-Sell@10 to **`51.78%`**.
   - Merged via **2-parameter customer-grouped out-of-fold affine calibration** (`score = s_main` for repeat items; `a · s_unseen + b` for unseen items, preventing data leakage across customers).
4. **365-Day Exponential Recency Sample Weighting:**
   - Smooth decay anchored at `TRAIN_CUTOFF` (`raw = max(0.1, 2^(-(T_cutoff - d)/365))`, mean-normalized to `1.00×`: `2.31×` for Jan 2026, `1.16×` at 1 year, `0.58×` at 2 years, `0.28×` at 3 years). Retains 3-year customer memory while adapting representations to the active product assortment.
5. **Regularization & Augmentation Ablations Tested:**
   - **Per-basket 70/30 Cold-Start Feature-Group Dropout** (zeroing 30 customer-specific features on 30% of training baskets to simulate cold-start) and **History-SKU Dropout (`p = 0.30`)** were both implemented and rigorously ablated. Controlled validation experiments proved that **clean conflict-free target separation with `0%` tabular dropout** outperformed stochastic masking on both NDCG@10 and cross-sell discovery.
   - Standard network regularization: `Dropout = 0.12` in Transformer/encoders, tiered `0.25 / 0.20 / 0.10` in MLP heads, `LayerNorm`, `AdamW` (`weight_decay = 1e-2`), cosine LR schedule, native `bf16` AMP on CUDA.

---

## Key Engineering Decisions & Lessons Learned

| Decision | Problem Solved | Measured Impact |
| :--- | :--- | :--- |
| **Canonical `base_name` Unification (`1,660 SKUs → 1,141 formulations`) + Causal Pack Resolver** | Audit revealed `24.5%` of "new SKUs" were pack-size switches (4L jug vs. 20L pail vs. 209L drum) crowding Top-10 slots. | **`+4.74 pp` Recall@10** (`76.15% → 80.89%`) and **nearly `2×` true new-product discovery** (`11.25% → 21.62%`). |
| **Eliminated Stage 1 Retrieval & Top-K Truncation (Full-Pool E2E)** | Separate Two-Tower + Stage 1 MLP created representation drift, capped recall via top-50/100 slates, and added pipeline complexity. | Removed Stage 1 training completely; `ProductTower` precomputes `1,141 × 128D` vectors in `1.4 ms`; live CPU serving runs in **`3.02 ms`**. |
| **Zero Item-ID Embeddings in `ProductTower`** | Learned product-ID tables fail on newly launched SKUs. | Enables **zero-shot product encoding** from BGE-M3 text + `65D` structured specs (**`66.29%` Recall@10** on zero-shot unlearned Test products). |
| **Pure-Q (`Q = v_bn`) + Full-42 Bilinear Cross-Attention Bias** | Passing tabular features into candidate queries created shortcuts that bypassed semantic history attention. | Kept queries strictly in semantic `v_bn` space while expanding bilinear attention bias `13D → 42D`, improving both discovery and latency. |
| **Strict Point-in-Time Causality & Multi-Line Settlement Audit** | Early fixed-cutoff demand snapshots and partial invoice settlement dates introduced subtle look-ahead leaks. | Rebuilt all features strictly `< prediction_date` (including `ДатаПолнойОплати < t` across full baskets), sealed SHA-256 manifests, and locked `600+` regression tests. |
| **Two Production Deployment Modes** | Held-out Test slice analysis isolated `83.03%` accuracy on trained entities vs. `57.80%` on unseen entities. | Documented two CRM/1C rollout modes: (1) high-precision trained-only mode, or (2) universal mode with live `⚠️` UI warning badge on unlearned entities + 2-month retraining. |

### Documented Negative Results (What Didn't Work & Why)
- **Early 25-Candidate Sampled CatBoost (`July 2026`):** Artificial `92.6%` Recall@10 on 25-item negative-sampled groups did not transfer to full-catalogue retrieval — motivated the pivot to strict chronological full-catalogue evaluation (`17.88%` popularity baseline).
- **Frozen Stage 1 History Vectors in Stage 2 (`-20` new hits) & Aligned 11D Diagonal Attention (`-22` to `-25` new hits):** Restricted cross-attention expressivity and broke coordinate alignment.
- **Multi-Interest Retrieval Heads:** Collapsed across all tested seeds during Stage 1 ablations.
- **History-SKU Dropout (`p = 0.30`) in Stage 2:** Miscalibrated the repeat-vs-unseen threshold (`-0.7 pp` R@10, `-0.009` NDCG).

---

## Model Evolution Path (`17.88% → 84.27%`)

| Stage | Milestone & Key Innovation | Val Recall@10 | NDCG@10 | New@10 |
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

## Repository Structure & Quickstart

```text
.
├── PROJECT_HISTORY.md                         # Full chronological engineering & metric log (70+ commits)
├── EVALUATION_RESULTS.md                      # Detailed validation tables & archived rejected hypotheses
├── recommender/
│   ├── data.py, merging.py, scope.py, preparation.py, company.py, catalogue.py
│   ├── split.py, development.py, features.py, sequences.py, text.py, familiarity.py, ranker.py
│   └── experiments/
│       ├── v2_full/                           # ACTIVE 4-year single-stage E2E pipeline & live PyTorch server
│       ├── roadmap_v2/                        # Shared base_name formulation, commercial & spec builders
│       └── static_enrichment/                 # StaticEnrichedProductTower & enriched metadata loaders
├── experiments/v2_full/                       # Active reference manifest, validation & held-out Test JSONs
├── notebooks/colab_v2_full_experiments.ipynb  # Native bf16 CUDA L4/A100 training notebook
└── tests/                                     # Causality, parity, leakage-guard & regression test suite
```

```bash
# 1. Setup (Python 3.12; Apple Silicon MPS / CUDA bf16 auto-detected)
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt -r requirements-dev.txt

# 2. Launch standalone Live Python PyTorch server (all 13,184 held-out Test orders, ~3 ms CPU/req + live benchmark)
./venv/bin/python recommender/experiments/v2_full/live_demo_server.py \
    --data-dir data/v2_full/interim/test_live_cache --port 8799

# 3. Package standalone runnable Live App ZIP (model + backend + website + all 13,184 Test data)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.build_portable_demo --mode live_app

# 4. Prepare data & train active Wave 2A model (CUDA bf16)
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.prepare_v2_full
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.extra_features
PYTHONPATH=. ./venv/bin/python -m recommender.experiments.v2_full.run_v2_full_wave2 --seed 42 --amp --in-memory

# 5. Run causality, architecture & leakage-guard test suite
PYTHONPATH=. ./venv/bin/pytest tests/test_v2_full_pipeline.py tests/test_v2_full_wave2b.py \
    tests/test_v2_full_wave2c.py tests/test_v2_full_wave2.py tests/test_stage5_audit_fixes.py tests/test_data.py -q
```

---

## Data Privacy & Reproducibility

Proprietary ERP exports (`data/`), BGE-M3 embedding caches, model checkpoints (`checkpoints/`), and packaged ZIP archives (`exports/`) remain strictly local and git-ignored. All source code, unit tests, SHA-256 reference manifests, and aggregate evaluation reports are tracked in Git.
