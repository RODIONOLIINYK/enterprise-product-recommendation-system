# Stage 2 Candidate-Set Self-Attention & Pre-Attention Token Enrichment (3-Arm Seed-42 Campaign)

> **Execution Scope:** `training_executed=True`, `heldout_access=False`, `promoted=False`, `statistics_policy=rolling_observed_v2`, `teacher_score_included=False`, `seed=42`, `epochs=10`, `batch_size=64`.
> All 17 frozen reference hashes + 14 rolling v3 cache/evidence hashes verified identical before and after execution.

## 1. Controlled 3-Arm Architecture & Parameter Summary

All three arms share identical `rolling_observed_v2` `quota100_50` slates (`26,253` train rows / `6,191` val rows, `10,276` total targets, `1,857` new targets), identical Stage 1 `repeat-no-month-v1` `u`/`v` vectors, identical initial `base` + `base_head` + `candidate_attention` weights, and identical `AdamW(lr=1e-3, wd=1e-4)` + `CosineAnnealingLR(eta_min=1e-5)` + `RankerLoss` optimization.

| Arm | Candidate Self-Attention | Pre-Attention Token Enrichment | Total Params | Trainable Params | Qualified Discovery | Discovery Epoch | NDCG Epoch | Train Time (s) |
| --- | :---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| Arm A (Pointwise 425D Control) | `False` | `False` | 1,237,515 | 671,497 | `True` | 8 | 8 | 138.2 |
| Arm B (Cand-Attn 425D) | `True` | `False` | 1,237,515 | 891,530 | `True` | 9 | 9 | 193.4 |
| Arm C (Cand-Attn + Enriched Tokens 128D) | `True` | `True` | 1,344,779 | 998,794 | `True` | 6 | 9 | 247.8 |

### Architecture Dimensions & Placement
- **Base UPS-CANR Cross-Attention & Bilinear Bias (Common A/B/C):** `33`-token customer memory width `128`, `4`-head cross-attention, `statistics_attention_weights` `(4, 14, 17)`, `excluded_statistics=('history_repeat',)`, `397D` encoded output + `28D` raw dynamic context -> **`425D`** `base_head` (`425 -> 256 -> 128 -> 64 -> 1`).
- **Candidate-Set Self-Attention (Common B/C):** `425D -> Linear(425, 128)` -> masked unordered `TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, norm_first=True, activation='gelu')` (no candidate positional encodings) -> `residual_head(256 -> 128 -> 1)` zero-initialized.
- **Pre-Attention Token Enrichment (Arm C only):** Base semantic tokens computed with zeros in numeric branches; three zero-output-initialized residual MLPs (`Linear -> LayerNorm(128) -> GELU -> Linear(128)`) inject `[0, 1]`-normalized statistics BEFORE `memory_encoder` and BEFORE `cross_attention`:
  - Profile Token 0: `[z_prof (128), profile8] (136) -> 128 -> 128`, added BEFORE `type_embedding[0]`.
  - History Tokens 1..32: `[t_hist (128), event9] (137) -> 128 -> 128`, added BEFORE `pos_embedding + type_embedding[1]` (padded tokens sanitized/masked to zero before embeddings and MLP).
  - Candidate Queries: `[v_cand (128), all41] (169) -> 128 -> 128`, added BEFORE `cross_attention` (no positional encodings; padded candidates masked to zero before MLP).
- **Zero-Residual Init Parity:** `max_abs_logit_diff` vs `CandidateContextRanker(425D)` at init: Arm A `0.0e+00`, Arm B `0.0e+00`, Arm C `0.0e+00`; reversed-slate diff `0.0e+00`, NaN-padded diff `0.0e+00`.

## 2. Scalar Normalization, Sources & Limitations
- **Transform Version:** `unit_interval_signed_shift_v1` (`transform_digest=6cb7bbc599bc...`).
- **Exact Scalar Channel Counts:** `41` candidate context + `8` profile lifetime + `9` history event = **`58`** enrichment channels (plus unchanged raw `41D`/`8D`/`9D` paths to bilinear bias and `425D` scoring/attention head, and preserved signed `128D` `u`/`v` latent vectors).
- **10 Signed Candidate Context Columns Mapped via `(x + 1) / 2 -> [0, 1]`:** `sku_growth_30d`, `month_sin`, `month_cos`, `sku_last_lift`, `sku_recent3_lift`, `category_last_lift`, `family_last_lift`, `line_last_lift`, `hierarchical_backoff`, `same_basket_lift` (remaining 31 columns validated in `[0, 1]`).
- **Stage 2 vs Stage 1 `uv_45` Month Note:** Stage 2 retains all 41 pairwise/context features (including month_sin and month_cos, so 10 candidate context columns are shifted from [-1,1] to [0,1]), whereas Stage 1 repeat-no-month-v1 uv_45 omits month_sin and month_cos.
- **Profile 8D (`LEGACY_PROFILE_FEATURES`):** `total_purchases`, `total_quantity`, `distinct_products`, `distinct_orders`, `distinct_categories`, `days_since_first_order`, `days_since_last_order`, `median_order_gap` (all `[0, 1]`).
- **History 9D (`ENRICHED_FEATURE_NAMES`):** `days_since_purchase_norm` recomputed to `prediction_day` from `seq_purchase_dates` (`age/(age+90)`); `purchase_month_sin`/`purchase_month_cos` mapped via `(x + 1) / 2 -> [0, 1]`; `real_event_indicator` derived from `attention_mask`; `is_repeat_purchase` (`history_repeat`, col 7) strictly zeroed (`= 0.0`) across all enriched paths.
- **History Text & Metadata Limitations:** Normal Stage 2 slates pass seq_product_text=None with catalog_text_embeddings=None, so product_text_proj receives zero vectors and contributes only its learned bias. Metadata policy remains `snapshot_unverified` (non-effective-dated catalogue/company attributes).

## 3. Validation Performance (N = 6,191 Queries; 10,276 Total Targets; 1,857 New Targets)

### 3.1 NDCG-Selected Checkpoints (`best_ndcg.pt`) & Discovery-Selected Checkpoints (`best.pt` / `best_unqualified.pt`)

| Arm & Selection | Ep | Qual | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | NDCG@10 | MRR | Repeat Micro R@10 | Repeat Macro R@10 | New Hits@10 | New Recall@10 | Mixed Cov (Micro) | New Cov |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm A (NDCG-sel) | 8 | `True` | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 | 0.89298 | 0.92067 | 255 | 0.13732 | 0.94424 | 0.69736 |
| Arm B (NDCG-sel) | 9 | `True` | 0.65307 | 0.76744 | 0.85725 | 0.90077 | 0.94010 | 0.59890 | 0.57919 | 0.88775 | 0.91588 | 252 | 0.13570 | 0.94424 | 0.69736 |
| Arm C (NDCG-sel) | 9 | `True` | 0.66033 | 0.76647 | 0.85867 | 0.90248 | 0.94010 | 0.60012 | 0.58102 | 0.88811 | 0.91735 | 239 | 0.12870 | 0.94424 | 0.69736 |
| Arm A (Disc-sel) | 8 | `True` | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 | 0.89298 | 0.92067 | 255 | 0.13732 | 0.94424 | 0.69736 |
| Arm B (Disc-sel) | 9 | `True` | 0.65307 | 0.76744 | 0.85725 | 0.90077 | 0.94010 | 0.59890 | 0.57919 | 0.88775 | 0.91588 | 252 | 0.13570 | 0.94424 | 0.69736 |
| Arm C (Disc-sel) | 6 | `True` | 0.65956 | 0.77100 | 0.85906 | 0.90267 | 0.94010 | 0.59963 | 0.57879 | 0.89191 | 0.92023 | 255 | 0.13732 | 0.94424 | 0.69736 |

### 3.2 Customer-Clustered Paired Bootstrap 95% CIs (2,000 draws, seed=123)

| Selection | Comparison | Recall@10 Delta (95% CI) | NDCG@10 Delta (95% CI) | New Recall@10 Delta (95% CI) | Recall@5 Delta (95% CI) |
| --- | --- | --- | --- | --- | --- |
| `ndcg` | A -> B (B - A) | -0.00360 ([-0.00742, +0.00007]) | -0.00333 ([-0.00627, -0.00050]) | -0.00162 ([-0.01255, +0.00858]) | -0.00705 ([-0.01133, -0.00247]) |
| `ndcg` | B -> C (C - B) | -0.00097 ([-0.00509, +0.00317]) | +0.00122 ([-0.00227, +0.00444]) | -0.00700 ([-0.01744, +0.00429]) | +0.00726 ([+0.00242, +0.01207]) |
| `ndcg` | A -> C (C - A) | -0.00457 ([-0.00817, -0.00085]) | -0.00211 ([-0.00499, +0.00060]) | -0.00862 ([-0.01807, +0.00149]) | +0.00021 ([-0.00390, +0.00418]) |
| `discovery` | A -> B (B - A) | -0.00360 ([-0.00742, +0.00007]) | -0.00333 ([-0.00627, -0.00050]) | -0.00162 ([-0.01255, +0.00858]) | -0.00705 ([-0.01133, -0.00247]) |
| `discovery` | B -> C (C - B) | +0.00356 ([-0.00096, +0.00779]) | +0.00073 ([-0.00273, +0.00404]) | +0.00162 ([-0.00934, +0.01195]) | +0.00649 ([+0.00170, +0.01108]) |
| `discovery` | A -> C (C - A) | -0.00004 ([-0.00347, +0.00320]) | -0.00260 ([-0.00540, +0.00010]) | +0.00000 ([-0.00917, +0.00859]) | -0.00056 ([-0.00479, +0.00368]) |

### 3.3 History-Length & Validation-Month Cohorts (NDCG-Selected Checkpoints)
> **Population Caveat:** Validation months (`2025-10`..`2026-02`) and history-length buckets contain different customer populations, basket sizes, and seasonal mixes; cohort deltas should not be conflated with stationary longitudinal effects.

| Cohort | Rows | New Targets | A Recall@10 | B Recall@10 | C Recall@10 | A NDCG@10 | B NDCG@10 | C NDCG@10 | A New@10 | B New@10 | C New@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hist:cold_0` | 0 | 0 | — | — | — | — | — | — | 0 | 0 | 0 |
| `hist:light_1_8` | 1,116 | 574 | 0.72426 | 0.71721 | 0.71201 | 0.60206 | 0.59442 | 0.59540 | 165 | 157 | 153 |
| `hist:warm_9_31` | 1,808 | 594 | 0.80690 | 0.80674 | 0.80393 | 0.63782 | 0.63648 | 0.63334 | 79 | 82 | 75 |
| `hist:truncated_32` | 3,267 | 689 | 0.76718 | 0.76285 | 0.76435 | 0.58259 | 0.57963 | 0.58335 | 11 | 13 | 11 |
| `month:2025-10` | 1,616 | 533 | 0.76444 | 0.75921 | 0.75113 | 0.60039 | 0.59278 | 0.59384 | 73 | 71 | 63 |
| `month:2025-11` | 1,314 | 358 | 0.77791 | 0.77701 | 0.78152 | 0.60606 | 0.60596 | 0.60699 | 47 | 56 | 53 |
| `month:2025-12` | 1,353 | 380 | 0.79118 | 0.78879 | 0.79179 | 0.62098 | 0.61816 | 0.62028 | 60 | 54 | 52 |
| `month:2026-01` | 1,149 | 351 | 0.75573 | 0.75095 | 0.74963 | 0.58241 | 0.57989 | 0.57987 | 42 | 42 | 39 |
| `month:2026-02` | 759 | 235 | 0.76050 | 0.75532 | 0.75346 | 0.59609 | 0.59413 | 0.59634 | 33 | 29 | 32 |

## 4. Mechanism Diagnostics (Fitted Checkpoints, No Retraining)

| Checkpoint & Ablation | Max Abs Logit Diff | Mean Abs Logit Diff | Max Abs z_cand Diff | Max Abs Memory Diff | Recall@10 Delta | NDCG@10 Delta | New Hits@10 Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm B (ndcg) `disable_cand_attn` | 2.5313 | 0.39522 | 0.0000 | 0.0000 | -0.00128 | -0.00239 | -1 |
| Arm B (discovery) `disable_cand_attn` | 2.5313 | 0.39522 | 0.0000 | 0.0000 | -0.00128 | -0.00239 | -1 |
| Arm C (ndcg) `disable_cand_attn` | 0.5237 | 0.04763 | 0.0000 | 0.0000 | -0.00038 | -0.00024 | -1 |
| Arm C (ndcg) `disable_all_enrich` | 7.7360 | 2.81743 | 2.0919 | 5.7682 | -0.03695 | -0.04205 | -38 |
| Arm C (ndcg) `disable_cand_enrich` | 7.4780 | 2.72492 | 2.0919 | 0.0000 | -0.03982 | -0.04583 | -38 |
| Arm C (ndcg) `disable_prof_enrich` | 1.6732 | 0.10189 | 0.0000 | 4.3895 | -0.00060 | -0.00033 | -2 |
| Arm C (ndcg) `disable_hist_enrich` | 2.7169 | 0.26537 | 0.0000 | 4.9072 | -0.00086 | -0.00084 | -1 |
| Arm C (discovery) `disable_cand_attn` | 0.8079 | 0.12726 | 0.0000 | 0.0000 | -0.00019 | +0.00018 | +0 |
| Arm C (discovery) `disable_all_enrich` | 6.3102 | 2.67027 | 1.9791 | 5.9385 | -0.04294 | -0.04311 | -51 |
| Arm C (discovery) `disable_cand_enrich` | 5.9777 | 2.65102 | 1.9791 | 0.0000 | -0.05149 | -0.05000 | -60 |
| Arm C (discovery) `disable_prof_enrich` | 0.7369 | 0.05502 | 0.0000 | 2.6237 | -0.00026 | -0.00036 | +0 |
| Arm C (discovery) `disable_hist_enrich` | 1.4048 | 0.21757 | 0.0000 | 5.0035 | -0.00046 | +0.00027 | -8 |

### Trained-Weight Stability Checks (Nonzero Weights, Reversed Slate & NaN Candidate/History Padding)

| Checkpoint | Reversed Slate Max Diff | NaN Cand/Hist Pad Max Valid Diff | Masked Logits `-inf` |
| --- | ---: | ---: | :---: |
| Arm B (ndcg) | 9.537e-07 | 0.000e+00 | `True` |
| Arm B (discovery) | 9.537e-07 | 0.000e+00 | `True` |
| Arm C (ndcg) | 9.537e-07 | 0.000e+00 | `True` |
| Arm C (discovery) | 9.537e-07 | 0.000e+00 | `True` |

## 5. Real-Input Stage 2 Latency Benchmark (CPU & MPS, Slate=100, Batch=1 & 64)

> Hardware: `Apple M5`, PyTorch `2.14.1`, `mps_available=True`. Scope: `prepared_on_device_forward` measures pre-materialized tensors on target device; `host_to_device_transfer_plus_forward` includes CPU->device tensor transfer.

| Device | Arm | Batch 1 Prepared Med/P90 (ms) | Batch 1 H2D+Fwd Med/P90 (ms) | Batch 64 Prepared Med/P90 (ms) | Batch 64 H2D+Fwd Med/P90 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `cpu` | `arm_a_control_pointwise_425d` | 0.643 / 0.697 | 0.673 / 0.692 | 10.631 / 10.681 | 10.906 / 11.019 |
| `cpu` | `arm_b_cand_attn_425d` | 0.995 / 1.013 | 1.014 / 1.031 | 18.802 / 19.066 | 18.862 / 18.967 |
| `cpu` | `arm_c_cand_attn_enriched_128d` | 1.309 / 1.322 | 1.312 / 1.332 | 22.247 / 22.471 | 22.435 / 22.666 |
| `mps` | `arm_a_control_pointwise_425d` | 2.999 / 3.145 | 6.112 / 6.253 | 6.639 / 6.752 | 9.844 / 9.936 |
| `mps` | `arm_b_cand_attn_425d` | 3.228 / 3.293 | 6.344 / 6.490 | 8.969 / 9.320 | 12.220 / 12.307 |
| `mps` | `arm_c_cand_attn_enriched_128d` | 9.844 / 10.129 | 13.052 / 13.260 | 16.832 / 16.937 | 19.860 / 20.107 |

## 6. Reproduction CLI
```bash
# 1. Synthetic unit & regression tests
PYTHONPATH=. ./venv/bin/pytest tests/test_candidate_token_enrichment.py --synthetic-only
# 2. Full synthetic suite
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
# 3. Full pytest suite
PYTHONPATH=. ./venv/bin/pytest tests/
# 4. Zero-init parity check (no training)
./venv/bin/python -m recommender.experiments.candidate_token_enrichment.train --verify-init-only --seed 42
# 5. Full 3-arm seed-42 10-epoch campaign + evaluation + benchmark
./venv/bin/python -m recommender.experiments.candidate_token_enrichment.train --execute-training --seed 42 --epochs 10 --batch-size 64
```
