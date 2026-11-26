# Evaluation & Training Report: Stage 1 Discovery Reference + Stage 2 `UPS-CANR` Reference

The historical sections record previously documented results on the chronological **Validation Split (`2025-10-01` to `2026-02-18`, `6,191` transitions, `1,857` new-to-customer targets)**. The held-out Test Split (`>= 2026-02-19`, `6,327` transitions) remains untouched.

The 2026-11-11 [shared-representation report](SHARED_STAGE1_EXPERIMENTS.md) independently reproduces both local selections below and records new controlled experiments. The later historical tables remain unchanged; their epoch-10/0.5924 reference is not the current local alias.

## Audit provenance note (2026-11-10)

The sections below preserve the **historical documented reference**. The audit did not rerun validation. Local files currently contain different selections, and the saved epoch-4 evaluation report cannot certify the epoch-7 default alias:

| Local checkpoint | Epoch / selection | Stored Recall@10 | Stored NDCG@10 | Stored new hits@10 |
|---|---|---:|---:|---:|
| `ranker/best_ranker.pt` | 7 / NDCG | 76.0417% | 0.593815 | 201 / 1,857 |
| `ranker/best_ndcg_ranker.pt` | 7 / NDCG | 76.0417% | 0.593815 | 201 / 1,857 |
| `ranker/best_discovery_ranker.pt` | 4 / discovery | 76.1505% | 0.591893 | 209 / 1,857 |

These are checkpoint metadata, not new measurements. Full hashes, configuration and caveats are in [audit/artifacts.json](audit/artifacts.json) and [AUDIT_REPORT.md](AUDIT_REPORT.md). No checkpoints were promoted, rewritten or removed. The historical reference peak is **0.5934**; **0.5945** is the rejected due-score ablation, not the same selected checkpoint as 209 hits.

---

## 1. End-to-End Validation Comparison (`checkpoints/evaluation_report.json`)

Evaluated across all `6,191` validation baskets using the rolling prior-sale candidate pool (`735–771` eligible SKUs per date):

| System | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | HitRate@10 | NDCG@10 | MRR | New-to-Customer Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Global Popularity Baseline** | 8.21% | 17.56% | 29.02% | 46.12% | 62.74% | 23.94% | 0.0910 | 0.0956 | — |
| **2. Personal Repeat + Popularity Heuristic** | 62.57% | 73.54% | 80.69% | 85.13% | 88.45% | 79.45% | 0.5666 | 0.5469 | 8.78% (`163 / 1,857`) |
| **3. Stage 1 Discovery Reference (Pure Unseen Ranking)** | 7.36% | 12.11% | 18.88% | 29.10% | 37.98% | 16.59% | 0.0566 | 0.0528 | — |
| **4. Stage 1 Mixed 50-Item Slate (25 Repeat + 25 Unseen)** | 62.38% | 72.64% | 78.79% | 87.48% | 87.48% | 78.60% | 0.5624 | 0.5446 | — |
| **5. Stage 2 `UPS-CANR` Reference (`best_ranker.pt`)** | **64.79%** | **76.15%** | **83.24%** | **87.48%** | **89.38%** | **81.88%** | **0.5924** | **0.5707** | **11.25% (`209 / 1,857`)** |

### Candidate Coverage & Discovery Breakdown on Validation (`6,191` Baskets)

| Target Subset | Total Target Instances | Stage 1 Slate Recall@50 (Mixed 50-Item Slate) | Stage 2 `UPS-CANR` Reranked Recall@10 |
| :--- | ---: | ---: | ---: |
| **Repeat Targets** (previously bought by customer) | `8,579` | **97.96%** (`8,404 / 8,579`) | **91.50%** (`7,850 / 8,579`) |
| **New-to-Customer Targets** (never bought before) | `1,857` | **41.09%** (`763 / 1,857`) | **11.25%** (`209 / 1,857`) |

### Stratified Performance: Cold/Light ($\le 2$ Prior Baskets) vs. Warm ($> 2$ Prior Baskets)

| Segment | Model | Recall@5 | Recall@10 | Recall@20 | Recall@50 | HitRate@10 | NDCG@10 | MRR | Token 0 Attn | History Attn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold / Light ($\le 2$ baskets, $N=444$)** | Personal Repeat Heuristic | 50.34% | 55.02% | 60.69% | 68.10% | 58.78% | 0.4731 | 0.4706 | — | — |
| | Stage 1 Mixed Slate | 50.34% | 55.73% | 64.96% | 74.16% | 59.46% | 0.4754 | 0.4713 | — | — |
| | **Stage 2 `UPS-CANR` Reference** | **51.55%** | **60.27%** | **68.02%** | **74.16%** | **63.74%** | **0.4932** | **0.4832** | **5.05%** | **94.95%** |
| **Warm ($> 2$ baskets, $N=5,747$)** | Personal Repeat Heuristic | 63.51% | 74.98% | 82.24% | 86.44% | 81.05% | 0.5738 | 0.5528 | — | — |
| | Stage 1 Mixed Slate | 63.31% | 73.95% | 79.86% | 88.50% | 80.08% | 0.5691 | 0.5503 | — | — |
| | **Stage 2 `UPS-CANR` Reference** | **65.81%** | **77.38%** | **84.41%** | **88.50%** | **83.28%** | **0.6001** | **0.5774** | **3.51%** | **96.49%** |

---

## 2. Stage 1 Discovery Reference (`checkpoints/retriever-discovery-reference/`)

- Trained with `python -m recommender.train_retriever --discovery-reference` (`30` epochs, `seed=42`, `discovery_only=True` over `8,773` training baskets containing at least one first-time purchase).
- Uses 128D towers, 2-layer Transformer over 32 purchase tokens with shared catalogue product vectors and inter-purchase gaps (`history_gap_reference_days = 74.0`), one-hot `container_type` with explicit `none`, no learned brand/category/line ID embeddings in `ProductTower` (`remove_product_attribute_embeddings = True`), and 4 binary familiarity flags (`brand`, `product_category`, `business_line`, `product_family`).
- Selected at **Epoch 23**:
  - **Rolling Pool New-to-Customer Recall@25:** **`35.11%` (`652 / 1,857`)**
  - **Rolling Pool New-to-Customer Recall@50:** **`49.60%` (`921 / 1,857`)**
  - **Mixed 50-Item Slate New-to-Customer Recall@50:** **`41.09%` (`763 / 1,857`)**

---

## 3. Stage 2 `UPS-CANR` Reference Training Curve (`checkpoints/ranker/training_history.json`)

Trained for 10 epochs (`seed=42`, `heuristic_fraction=0.5`, `statistics_attention=True`, `excluded_statistics=("history_repeat",)`):

| Epoch | Train Loss | Val Loss | Val Recall@5 | Val Recall@10 | Val Recall@20 | Val NDCG@10 | Val MRR | Val New Recall@10 (`Hits / 1,857`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.6626 | 2.0942 | 63.56% | 75.59% | 82.97% | 0.5819 | 0.5582 | 10.77% (`200 / 1,857`) |
| **2** | 2.4191 | 2.0637 | 63.93% | 75.95% | 83.20% | 0.5877 | 0.5653 | 10.61% (`197 / 1,857`) |
| **3** | 2.3422 | 2.0548 | 64.59% | 75.87% | 83.21% | 0.5917 | 0.5712 | 10.29% (`191 / 1,857`) |
| **4** | 2.2995 | 2.0537 | 64.26% | 75.97% | 83.22% | 0.5902 | 0.5687 | 10.82% (`201 / 1,857`) |
| **5** | 2.2644 | 2.0446 | 64.68% | 76.02% | 83.26% | 0.5929 | 0.5718 | 10.29% (`191 / 1,857`) |
| **6** | 2.2400 | 2.0489 | 64.55% | 76.00% | 83.20% | 0.5913 | 0.5698 | 10.50% (`195 / 1,857`) |
| **7 (Best NDCG)** | 2.2124 | 2.0422 | **64.86%** | 76.09% | 83.24% | **0.5934** | **0.5721** | 10.93% (`203 / 1,857`) |
| **8** | 2.1890 | 2.0468 | 64.71% | 75.99% | **83.28%** | 0.5922 | 0.5707 | 10.39% (`193 / 1,857`) |
| **9** | 2.1695 | 2.0472 | 64.79% | 76.08% | 83.22% | 0.5924 | 0.5705 | 10.72% (`199 / 1,857`) |
| **10 (Best Discovery / Reference)** | **2.1563** | 2.0484 | 64.79% | **76.15%** | 83.24% | 0.5924 | 0.5707 | **11.25% (`209 / 1,857`)** |

---

## 4. Archive of Tested & Rejected Experimental Hypotheses

The older experiments in the first table had their implementation paths removed. Later versioned campaigns retain isolated reproduction code and evidence, without enabling new production paths. Their results are recorded here so agents do not repeat completed tests:

| Experiment (Removed from Codebase) | New-to-Customer Hits@10 (`/ 1,857`) | Overall Recall@10 | Peak Val NDCG@10 | Why It Failed / Post-Mortem |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 2 Reference (`excluded_statistics=("history_repeat",)` )** | **`209` (`11.25%`)** | **`76.15%`** | `0.5934` | Active reference baseline (`checkpoints/ranker/`). |
| Remove `history_month` + `history_repeat` + `company_distinct_orders` | `203` (`10.93%`) | `76.09%` | `0.5934` | Slightly reduced discovery hits without improving overall recall. |
| Introduce reorder `due_score = gap_norm - age_norm` | `203` (`10.93%`) | `76.05%` | `0.5945` | Entered only through attention bias while memory token values zeroed out `tabular_features` and `sequence`, so due-ness could not propagate into attended values `h_attended`. |
| Frozen Precompiled Stage 1 Vectors for Token 0 & History (`1..32`) | `189` (`10.18%`) | `75.82%` | `0.5947` (`0.5880` at discovery ckpt) | Freezing Stage 1 retrieval vectors prevented history tokens from adapting to Stage 2 ranking and passed them through trainable Transformer/QKV layers that destroyed direct coordinate alignment. |
| Aligned 11-D Statistics (Diagonal Attention) | `187` (`10.07%`) | `75.82%` | `0.5922` | Restricted the existing $14 \times 17$ learned interaction matrix $s_c^\top W s_m$ and forced artificial pairings (e.g. customer `distinct_products` paired with both `popularity` and `industry_affinity`). |
| Aligned 11-D Statistics (Full $11 \times 11$ Interactions) | `184` (`9.91%`) | `75.84%` | `0.5930` | Compressed the $14 \times 17$ statistics representation into $11 \times 11$ without adding new information. |
| Combined Precompiled Vectors + Aligned 11-D + Shared Favorite Attributes | `181` (`9.75%`) | `75.81%` | `0.5928` | Compounded the representational restrictions of frozen history vectors and compressed attention statistics (`-13.4%` relative drop in new-product discovery). |
| Stage 1 Price / Turnover / Co-Purchase Complementarity | Below reference | Below reference | — | Inferred invoice prices, turnover weights, and monthly co-purchase complementarity lowered Stage 1 unseen discovery recall below the 4-flag reference. |

### Rolling ranker capacity and standalone CatBoost (2026-11-13)

Seed 42; all numbers in each row come from the same NDCG-selected checkpoint. Neural arms use identical 100-candidate rolling slates, 32 events and a ten-epoch budget. Standalone CatBoost uses 52 categorical/statistical inputs and directly ranks the full eligible pool, with no neural representations or selection. Its training samples uniform negatives with inverse-probability weights; validation scores all 4,700,328 eligible query/product pairs. This is a complete-pipeline comparison for CatBoost, rather than a matched architecture-only ablation.

| Configuration | Recall@10 | NDCG@10 | New hits@10 / 1,857 |
| --- | ---: | ---: | ---: |
| Saved original rolling pointwise A | 0.77104 | 0.60223 | 255 |
| Fresh original control | 0.76941 | 0.60214 | 253 |
| Wider head: 425 → 512 → 256 → 128 → 1 | 0.76632 | 0.60052 | 239 |
| Wider Transformer FFN: 256 → 512 | 0.76682 | 0.59975 | 245 |
| Two history self-attention layers | 0.76899 | 0.59955 | 245 |
| Standalone CatBoost, 318 trees | 0.75815 | 0.59373 | 202 |

Discovery-selected wider-head / wider-FFN / deeper-history checkpoints produce 250 / 249 / 247 new hits respectively; none exceeds saved A or its fresh control. No model was promoted or held-out data evaluated. Single-seed results, paired intervals, learning curves, replay checks and latency are in [the authoritative report](experiments/ranker_capacity_v1/REPORT.md).

### Direct statistical/categorical access and underfitting audit (2026-11-14)

All 41 rolling candidate statistics were already direct Stage 2 scoring inputs. Profile8 previously entered through attention bias; candidate identity/category/line were indirect through frozen candidate vectors. Four matched fresh seed42/10-epoch fits isolated additional direct scoring paths, preserving original head widths, 32-event memory, loss, schedule and 100-candidate slates. Added input weights were zero-initialized with exact initial logit parity.

| NDCG-selected model | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: |
| Saved pointwise A | 0.77104 | 0.60223 | 255 |
| Fresh control | 0.76954 | 0.60105 | 251 |
| Direct profile8 | 0.76910 | 0.59877 | 238 |
| Direct candidate SKU/category/line embeddings | 0.76604 | 0.59693 | 238 |
| Both direct input paths | 0.76633 | 0.59774 | 242 |

No treatment improved the control and no reference was promoted. A separate train-only 128-query probe reached 100% conditional top-10 target recovery after 300 updates with dropout disabled; this verifies optimization/capacity on a tiny cohort, not full-data generalization. Training support is only 2,319 distinct customers, median 8 target instances/SKU; 448/841 SKUs have at most 10 training target instances. Same-mode training/validation diagnostics, feature-by-feature mapping, paired intervals and limitations are in [the authoritative report](experiments/direct_feature_access_v1/REPORT.md). Held-out purchases were not accessed.

### Fixed-input regularization and rotating negatives (2026-11-14)

Seed42, ten epochs per fit; the standalone CatBoost52 schema, numeric inputs, trained model and results were frozen. The new small MLP receives exactly those 49 numeric and three categorical fields (model-specific learned categorical encodings), with no pretrained towers or Transformer. Fixed-pair arms use the identical 2,670,525 sampled pairs/importance weights and binary objective as CatBoost. The augmentation arm changes only negative draws relative to the unregularized MLP, preserving all 45,225 eligible positive instances per epoch. Validation ranks the complete eligible pool.

| NDCG-selected model | Recall@10 | NDCG@10 | New hits@10 |
| --- | ---: | ---: | ---: |
| Frozen standalone CatBoost52 | 0.75815 | 0.59373 | 202 |
| MLP52, fixed pairs, dropout .2/.1 and WD1e-4 | 0.76130 | 0.58915 | 221 |
| MLP52, fixed pairs, zero dropout/WD | 0.76293 | 0.59143 | 223 |
| MLP52, rotating negatives, zero dropout/WD | 0.76263 | 0.59372 | 224 |

The MLP has32,953 parameters. Rotating negatives exposes13,240,017 distinct training pairs instead of2,670,525 fixed pairs, without synthetic positive purchases. The CatBoost comparison's paired intervals include zero for NDCG/Recall@10/new recall; no superiority is claimed.

Seven matched Stage2 fits separately halve scoring-head, profile-projection or attention dropout; remove weight decay; remove all active dropout/weight decay; or combine direct52 input access with zero regularization. None of the NDCG-selected treatments beats saved A overall. Best treatment NDCG is0.60138 (zero regularization,235 new hits); direct52+zero regularization gives0.59705/232, versus saved A0.60223/255. The discovery-selected zero-regularization checkpoint gives258 hits but lower R@10=0.76785/NDCG@10=0.60077. No reference was promoted or held-out purchases accessed. Full protocols, all selections, exact transition-input column mapping, uncertainty and replay evidence are in [the authoritative report](experiments/regularization_signal_v1/REPORT.md).

## 5. Shared Stage 1 representations (2026-11-11)

Sixteen controlled 10-epoch model/seed runs are documented in [SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md). Three-seed mean new-target hits@10 are 196.67 for fresh R0, 196.33 for shared R3, 195.67 for reduced-bias R4 and 200.00 without customer-token information. Paired customer-bootstrap intervals do not establish a discovery benefit for sharing or indicator removal. The preserved discovery checkpoint replays 209/1,857, 76.1505% overall Recall@10 and 0.591893 NDCG@10 at the same epoch-4 checkpoint.

Default architecture, bias and checkpoint aliases are unchanged. Wider/edge/scoring experiments were deferred; all results retain the known frozen Stage 1 training-demand look-ahead limitation. The held-out Test split was not evaluated.

## E1–E5 campaign — 2026-11-11–12

Full evidence: [E1_E5_EXPERIMENTS.md](E1_E5_EXPERIMENTS.md), [aggregate metrics](experiments/e1_e5/summary.json), and [commands](experiments/e1_e5/README.md). Thirty-two controlled fits were completed without held-out access or default promotion. Three-seed frozen-retriever discovery hits: fresh control 196.33 ± 5.69; E1 202.33 ± 0.58; E2 residual 201.67 ± 6.11; E2 MLP 203.67 ± 1.53. The separately trained demand-disabled single-vector control averages 719/1,857 at Stage 1 Top-25 unseen and 980/1,857 at Top-50, with downstream 194.67 ± 11.24 discovery hits. E3–E5 extensions failed the screen; all tested multi-interest outputs collapsed. These comparisons do not replace the independently replayed local discovery checkpoint (209 hits) or NDCG alias (201 hits).
