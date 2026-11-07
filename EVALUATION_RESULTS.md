# Evaluation & Training Report: Modular Two-Tower Retriever + UPS-CANR Ranker

This report records the original checkpoints. The retriever input schema, quantity scaling,
and slate construction have since changed. The tables below describe the earlier run.

## 1. Architecture Overview
**Official Architecture Name:**
> **Modular Late-Fusion Two-Tower Retriever + Unified Profile-Sequence Candidate-Aware Neural Ranker (`UPS-CANR`)**

### Implemented Pipeline Modules
- **Enriched Sequence Tokens (`recommender/sequences.py`):** 32 historical purchase tokens $P_1 \dots P_{32}$ encoding 4 categorical indices (`product_id`, `product_category`, `product_subcategory`, `business_line`) and 9 point-in-time continuous features (`qty_norm`, `days_ago_norm`, `inter_purchase_gap_norm`, `sin_month`, `cos_month`, `user_product_prior_count_norm`, `user_product_quantity_share`, `is_repeat_purchase`, `real_event_indicator`).
- **Stage 1 Retriever (`recommender/models.py`, `recommender/train_retriever.py`):**
  - `CustomerTower`: Modular Late Fusion combining frozen `BAAI/bge-m3` company text (`1024-d` -> `128-d`), customer tabular/lifetime features (`128-d`), and a 2-layer Transformer over the 32 enriched purchase tokens (`128-d` via dual masked-mean + last-event pooling) into an L2-normalized `128-d` customer vector $\mathbf{u}_c$.
  - `ProductTower`: Combines `BAAI/bge-m3` product text (`1024-d` -> `128-d`), catalog hierarchy embeddings (`brand`, `product_category`, `product_subcategory`, `business_line`), and 6 point-in-time demand velocity features into an L2-normalized `128-d` product vector $\mathbf{v}_p$.
- **Candidate Generation & Pairwise Features (`recommender/candidates.py`):** Retrieves top-$C=50$ candidates per transition and computes 12 point-in-time customer-candidate pairwise features strictly bounded in $[0.0, 1.0]$.
- **Stage 2 Ranker (`recommender/ranker.py`, `recommender/train_ranker.py`):**
  - `UPSCANRRanker`: Builds a unified 33-token Customer Memory $[\mathbf{z}_{\text{customer\_profile}}, P_1, \dots, P_{32}] \in \mathbb{R}^{B \times 33 \times 128}$, applies 4-head candidate-aware cross-attention for all $C=50$ candidates simultaneously, and scores each candidate via a `396-d` Residual Fusion MLP trained with Multi-Label BCE + Listwise Softmax Cross-Entropy loss.
- **Evaluation & Baselines (`recommender/evaluate.py`):** Compares Global Popularity, Personal Repeat-Purchase Heuristic, Stage 1 Retriever, and Stage 2 `UPS-CANR` Ranker across overall, Cold/Light ($\le 2$ prior baskets), and Warm ($> 2$ prior baskets) segments.

---

## 2. Saved Artifacts & Data Locations

- **Validation Evaluation Report (JSON):** `checkpoints/evaluation_report.json`
- **Stage 1 Retriever Training History (JSON):** `checkpoints/retriever/training_history.json`
- **Stage 1 Best Model Checkpoint:** `checkpoints/retriever/best_retriever.pt`
- **Stage 2 `UPS-CANR` Training History (JSON):** `checkpoints/ranker/training_history.json`
- **Stage 2 Best Model Checkpoint:** `checkpoints/ranker/best_ranker.pt`
- **Cached BGE-M3 Text Embeddings:**
  - `data/interim/embeddings/bge_m3_catalogue.parquet`
  - `checkpoints/retriever/bge_m3_company.parquet`
- **Precomputed Candidate Slates:**
  - `data/interim/ranker_train_slates.pt` (`26,264` train transitions)
  - `data/interim/ranker_val_slates.pt` (`6,191` validation transitions)

---

## 3. Epoch-by-Epoch Training Curves

### A. Stage 1: Modular Two-Tower Retriever (`checkpoints/retriever/training_history.json`)

| Epoch | Train Loss | Val Loss | Val Recall@10 | Val Recall@50 | Val Recall@100 | Val NDCG@10 | Val MRR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.1339 | 0.1043 | 25.76% | 57.70% | 73.60% | 0.1388 | 0.1373 |
| **2** | 0.0975 | 0.0945 | 33.85% | 65.62% | 78.61% | 0.1927 | 0.1876 |
| **3** | 0.0889 | 0.0895 | 37.44% | 68.73% | 80.91% | 0.2162 | 0.2044 |
| **4** | 0.0839 | 0.0874 | 40.74% | 70.88% | 82.08% | 0.2505 | 0.2409 |
| **5** | 0.0807 | 0.0834 | 43.46% | 72.25% | 83.09% | 0.2658 | 0.2534 |
| **6** | 0.0777 | 0.0819 | 44.71% | 73.51% | 83.59% | 0.2786 | 0.2656 |
| **7** | 0.0754 | 0.0810 | 46.14% | 74.65% | 84.38% | 0.2850 | 0.2683 |
| **8** | 0.0739 | 0.0798 | 46.91% | 75.30% | 84.58% | 0.2969 | 0.2825 |
| **9** | 0.0728 | 0.0795 | 47.33% | 75.54% | **85.05%** | 0.2973 | 0.2811 |
| **10** | **0.0722** | **0.0792** | **47.61%** | **75.67%** | 85.03% | **0.3005** | **0.2847** |

### B. Stage 2: `UPS-CANR` Neural Ranker (`checkpoints/ranker/training_history.json`)

| Epoch | Train Loss | Val Loss | Val Recall@5 | Val Recall@10 | Val Recall@20 | Val NDCG@10 | Val MRR | Token 0 Profile Attn | Tokens 1..32 History Attn |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.0658 | 2.5044 | 57.47% | 66.49% | 71.41% | 0.5216 | 0.5083 | 20.83% | 79.17% |
| **2** | 2.7796 | 2.4335 | 57.66% | 66.36% | 71.21% | 0.5257 | 0.5148 | 24.97% | 75.03% |
| **3** | 2.6637 | 2.4107 | 57.86% | 66.35% | 71.28% | 0.5287 | 0.5188 | 30.48% | 69.52% |
| **4** | 2.6004 | 2.3991 | 58.11% | 66.58% | 71.45% | 0.5339 | 0.5256 | 27.17% | 72.83% |
| **5** | 2.5606 | 2.4215 | 57.79% | 66.20% | 71.35% | 0.5284 | 0.5203 | 29.16% | 70.84% |
| **6** | 2.5151 | 2.4772 | 57.11% | 66.15% | 71.37% | 0.5201 | 0.5092 | 24.17% | 75.83% |
| **7** | 2.4755 | 2.4002 | 57.92% | 66.61% | 71.44% | 0.5325 | 0.5240 | 32.34% | 67.66% |
| **8** | 2.4419 | 2.4469 | 57.29% | 66.24% | 71.42% | 0.5245 | 0.5142 | 30.87% | 69.13% |
| **9 (Best)** | 2.4078 | **2.3816** | 58.09% | **66.69%** | 71.49% | **0.5347** | **0.5278** | **31.12%** | **68.88%** |
| **10** | **2.3846** | 2.3877 | **58.13%** | 66.69% | **71.56%** | 0.5337 | 0.5262 | 30.87% | 69.13% |

---

## 4. Validation Split Comparison (`2025-10-01` to `2026-02-18`, 6,191 Transitions)

### A. Overall Metrics

| Model / Baseline | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | HitRate@10 | NDCG@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Global Popularity Baseline** | 8.35% | 17.88% | 28.98% | 46.01% | 63.17% | 24.28% | 0.0925 | 0.0970 |
| **2. Stage 1: Modular Two-Tower Retriever** | 34.44% | 47.60% | 60.47% | 75.74% | 85.04% | 56.39% | 0.3007 | 0.2855 |
| **3. Stage 2: `UPS-CANR` Neural Ranker (Top-50 Rerank)** | **58.09%** | **66.69%** | **71.49%** | 75.74% | 85.04% | **73.41%** | **0.5347** | **0.5289** |
| **4. Personal Repeat + Pop Fallback Heuristic** | 62.60% | 73.60% | 80.69% | 85.11% | 88.49% | 79.52% | 0.5669 | 0.5471 |

### B. Stratified by Customer History Depth (`Cold/Light <= 2` vs `Warm > 2` Prior Baskets)

| Segment | Model | Recall@5 | Recall@10 | Recall@20 | Recall@50 | HitRate@10 | NDCG@10 | MRR | Token 0 Attn | History Attn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold / Light ($\le 2$ baskets, $N=444$)** | Global Popularity | 8.89% | 18.90% | 30.21% | 44.37% | 22.30% | 0.0919 | 0.0858 | — | — |
| | Stage 1 Two-Tower | 21.71% | 31.52% | 41.87% | 60.80% | 35.81% | 0.1788 | 0.1611 | — | — |
| | **Stage 2 `UPS-CANR`** | **40.81%** | **45.81%** | **52.64%** | **60.80%** | **50.23%** | **0.3740** | **0.3711** | **70.84%** | **29.16%** |
| | Personal Repeat Heuristic | 50.51% | 55.33% | 60.45% | 68.02% | 59.23% | 0.4750 | 0.4723 | — | — |
| **Warm ($> 2$ baskets, $N=5,747$)** | Global Popularity | 8.30% | 17.80% | 28.89% | 46.14% | 24.43% | 0.0925 | 0.0979 | — | — |
| | Stage 1 Two-Tower | 35.43% | 48.84% | 61.91% | 76.89% | 57.98% | 0.3101 | 0.2951 | — | — |
| | **Stage 2 `UPS-CANR`** | **59.43%** | **68.31%** | **72.94%** | **76.89%** | **75.20%** | **0.5471** | **0.5411** | **28.04%** | **71.96%** |
| | Personal Repeat Heuristic | 63.54% | 75.01% | 82.25% | 86.44% | 81.09% | 0.5741 | 0.5528 | — | — |

---

## 5. Underfitting & Overfitting Diagnostics

1. **Stage 1 Retrieval Bottleneck (`Recall@50 = 75.74%` vs `85.11%` Personal Repeat):**
   - Stage 1 shows smooth, monotonic loss decrease (`0.1339 -> 0.0722` train, `0.1043 -> 0.0792` val) without overfitting, but **underfits exact SKU identity**: `ProductTower` currently encodes text (`BAAI/bge-m3`), `brand`, `category`, `subcategory`, `business_line`, and 6 demand numbers, without a dedicated learnable `product_id` embedding table.
   - Because Stage 2 (`UPS-CANR`) only reranks the top 50 candidates from Stage 1, any repeat SKU ranked $>50$ by Stage 1 is unreachable by Stage 2 (`24.26%` of true items missed before Stage 2 runs).
2. **Stage 2 Rapid Convergence & Mild Overfitting After Epoch 4–9:**
   - Stage 2 achieves massive gains over Stage 1 (`Recall@10` `47.60% -> 66.69%`, `NDCG@10` `0.3007 -> 0.5347`), packing `88.1%` of Stage 1's top-50 recall into the top 10.
   - Training loss continues falling (`3.0658 -> 2.3846`) while validation loss plateaus around `2.38–2.40` after Epoch 4–9 due to the hard `75.74%` Stage 1 candidate ceiling and positive-injection distribution shift between training and inference slates.
3. **Recommended Next Steps (Pending User Approval):**
   - **Hybrid Candidate Slate Generation:** Construct the 50-candidate slate for Stage 2 by taking the union of the customer's previously purchased SKUs and the top Stage 1 Two-Tower semantic candidates (raising candidate slate recall from `75.74%` to `>90%`).
   - **Learnable `product_id` Embedding in `ProductTower`:** Add `nn.Embedding(num_products, 64)` to `ProductTower` so Stage 1 directly captures exact SKU-level collaborative signals alongside text and category hierarchy.
