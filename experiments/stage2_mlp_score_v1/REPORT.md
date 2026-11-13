# Stage 2 UPS-CANR Integration with Frozen Stage 1 301D MLP Score

> **User decision (2026-11-13): Exclude the MLP score from Stage 2 attention and scoring. Training is on hold.** See [DECISION.md](DECISION.md). Results below remain historical evidence.

## 1. Experiment Contract & Setup
- **Stage 1 Reference (`active_reference_id`)**: `repeat-no-month-v1` (`uv_45`, `301D`, selected epoch 5, `heldout_access=False`)
- **Score Feature**: `stage1_mlp_sigmoid_score` via `sigmoid_float32_masked_v1` (`sigmoid(raw_logit)` in `[0, 1]` float32 after pre-sigmoid eligibility masking)
- **Stage 2 Integration Path**: `Stage2MLPScoreRanker` wrapping `CandidateContextRanker('dynamic_features')` around `UPSCANRRanker` (`excluded_statistics=('history_repeat',)`, `num_pairwise_features=14` -> cross-attention `W[4, 15, 17]` + `398D` encoded features + `28D` dynamic cols `13:41` -> `426D` head)
- **Candidate Slate Policy**: `quota100_50` (100 SKUs: up to 50 eligible repeat + remaining top eligible unseen by frozen `301D` MLP score)
- **Controlled Arms (`seed=42`, `epochs=10`, `batch_size=64`)**:
  - **Matched Control (`score_channel_enabled=False`)**: identical 426D architecture, identical initial weights (zero-init on new score columns), score channel forced to `0.0` before attention and head; `1,018,062` params (`671,821` trainable), `121.6s`
  - **Treatment (`score_channel_enabled=True`)**: identical initial weights and minibatch sequence, actual masked `sigmoid(mlp_logit)` fed to attention and head; `1,018,062` params (`671,821` trainable), `121.0s`
- **Quality Guardrails (`checkpoints/ranker/best_discovery_ranker.pt`)**: `Recall@10 >= 0.75588`, `NDCG@10 >= 0.59189`

## 2. Quota100_50 Validation Slate Coverage (Frozen 301D MLP Selector)
- Validation queries: `6191` (`2025-10-01..2026-02-18`), total targets: `10276`, repeat targets: `8419`, new targets: `1857`
- **Overall targets in slate**: `9681/10276` (`94.21%` micro, `93.70%` macro)
- **Repeat targets in slate**: `8409/8419` (`99.88%`)
- **New targets in slate**: `1272/1857` (`68.50%`)

## 3. Main Validation Results

### A. Discovery-Selected Checkpoints (Guardrail-Constrained)

| Model / Arm | Ep | Qual | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | New R@10 | Cond New R@10 | Repeat R@10 (mic/mac) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 301D Teacher (full pool) | 5 | - | 0.64335 | 0.75411 | 0.84850 | 0.89542 | 0.93733 | 0.58633 | 0.56615 | 194/1857 | 10.45% | 10.72% | 88.41% / 91.04% |
| Stage 1 301D Teacher (`quota100_50` order) | 5 | - | 0.64335 | 0.75411 | 0.84850 | 0.89542 | 0.93700 | 0.58633 | 0.56590 | 194/1857 | 10.45% | 15.25% | 88.41% / 91.04% |
| Stage 2 Control (`score=0`) | 9 | True | 0.65343 | 0.76477 | 0.85467 | 0.89776 | 0.93700 | 0.59713 | 0.57888 | 244/1857 | 13.14% | 19.18% | 88.73% / 91.49% |
| Stage 2 Treatment (`+301D MLP score`) | 8 | False | 0.65020 | 0.75875 | 0.84782 | 0.89768 | 0.93700 | 0.59155 | 0.57282 | 211/1857 | 11.36% | 16.59% | 88.42% / 91.26% |

### B. NDCG-Selected Checkpoints (`best_ndcg.pt`)

| Model / Arm | Ep | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | New R@10 | Cond New R@10 | Repeat R@10 (mic/mac) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 2 Control (`score=0`, `best_ndcg`) | 8 | 0.65404 | 0.76320 | 0.85617 | 0.89857 | 0.93700 | 0.59735 | 0.58008 | 237/1857 | 12.76% | 18.63% | 88.70% / 91.42% |
| Stage 2 Treatment (`+301D MLP score`, `best_ndcg`) | 10 | 0.65026 | 0.75703 | 0.84677 | 0.89495 | 0.93700 | 0.59189 | 0.57420 | 195/1857 | 10.50% | 15.33% | 88.40% / 91.27% |

## 4. Paired Customer-Clustered 95% Bootstrap CIs (Treatment - Control)

| Comparison | Metric | Point Delta | 95% CI Low | Median | 95% CI High |
| --- | --- | ---: | ---: | ---: | ---: |
| Discovery vs Discovery | `recall@5_delta` | -0.00323 | -0.00753 | -0.00318 | +0.00128 |
| Discovery vs Discovery | `recall@10_delta` | -0.00603 | -0.01032 | -0.00597 | -0.00169 |
| Discovery vs Discovery | `recall@25_delta` | -0.00685 | -0.01102 | -0.00677 | -0.00269 |
| Discovery vs Discovery | `recall@50_delta` | -0.00008 | -0.00445 | -0.00012 | +0.00419 |
| Discovery vs Discovery | `recall@100_delta` | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| Discovery vs Discovery | `ndcg@10_delta` | -0.00558 | -0.00839 | -0.00557 | -0.00281 |
| Discovery vs Discovery | `new_recall@10_delta` | -0.01777 | -0.03056 | -0.01766 | -0.00551 |
| NDCG vs NDCG | `recall@5_delta` | -0.00378 | -0.00870 | -0.00376 | +0.00096 |
| NDCG vs NDCG | `recall@10_delta` | -0.00617 | -0.01045 | -0.00619 | -0.00193 |
| NDCG vs NDCG | `recall@25_delta` | -0.00940 | -0.01366 | -0.00928 | -0.00507 |
| NDCG vs NDCG | `recall@50_delta` | -0.00362 | -0.00805 | -0.00367 | +0.00073 |
| NDCG vs NDCG | `recall@100_delta` | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| NDCG vs NDCG | `ndcg@10_delta` | -0.00547 | -0.00834 | -0.00548 | -0.00280 |
| NDCG vs NDCG | `new_recall@10_delta` | -0.02262 | -0.03596 | -0.02256 | -0.01003 |

## 5. Epoch-by-Epoch Validation Trajectory

| Ep | Control Loss | Control R@10 | Control NDCG@10 | Control New@10 | Treatment Loss | Treatment R@10 | Treatment NDCG@10 | Treatment New@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.1788 | 0.75005 | 0.58126 | 183 | 3.0924 | 0.75459 | 0.58588 | 205 |
| 2 | 3.0700 | 0.75826 | 0.58792 | 213 | 2.9543 | 0.75732 | 0.58851 | 208 |
| 3 | 3.0338 | 0.76080 | 0.59222 | 225 | 2.9186 | 0.75640 | 0.58909 | 202 |
| 4 | 3.0134 | 0.76031 | 0.59307 | 228 | 2.9040 | 0.75603 | 0.59138 | 205 |
| 5 | 2.9893 | 0.76298 | 0.59486 | 233 | 2.8845 | 0.75604 | 0.59154 | 205 |
| 6 | 2.9709 | 0.76537 | 0.59696 | 242 | 2.8711 | 0.75626 | 0.59179 | 195 |
| 7 | 2.9403 | 0.76391 | 0.59637 | 243 | 2.8528 | 0.75781 | 0.59162 | 201 |
| 8 | 2.9090 | 0.76320 | 0.59735 | 237 | 2.8345 | 0.75875 | 0.59155 | 211 |
| 9 | 2.8802 | 0.76477 | 0.59713 | 244 | 2.8099 | 0.75666 | 0.59157 | 199 |
| 10 | 2.8580 | 0.76379 | 0.59606 | 243 | 2.7958 | 0.75703 | 0.59189 | 195 |

## 6. Mechanism, Observability & Parity Verification
- **Zero-extension initialization parity (425D `CandidateContextRanker` vs 426D `Stage2MLPScoreRanker`)**: max abs logit diff with actual score `0.00e+00`, with zero score `0.00e+00`, attn diff `0.00e+00`.
- **Control exact score-invariance**: max abs val logit diff with vs without `mlp_score` = `0.0e+00` (discovery) and `0.0e+00` (NDCG); `attention_score_weight_l2=0.0000`, `head_score_weight_l2=0.0000`.
- **Treatment score observability (Discovery ckpt)**: `attention_score_weight_l2=1.0917` (`max_abs=0.2278`), `head_score_weight_l2=4.6050` (`max_abs=0.8890`); toggling `mlp_score` vs `0`: `max_abs_attn_diff=0.14351`, `max_abs_h_att_diff=0.17058`, `max_abs_logit_diff=8.59835` (full-val max abs logit diff `8.83452`).
- **Treatment score observability (NDCG ckpt)**: `attention_score_weight_l2=1.1012`, `head_score_weight_l2=4.5998`, `max_abs_attn_diff=0.14379`, `max_abs_logit_diff=8.88141`.
- **Live request-time adapter vs sealed cache parity**: max abs `mlp_score` diff `1.79e-07`, max abs logit diff `2.62e-06`; slate-order reversal diff `mlp_score=0.00e+00`, `logit=0.00e+00`; NaN-padded invalid-candidate masking diff `mlp_score=0.00e+00`, `logit=0.00e+00` (`masked_score_zero=True`, `masked_logit_neginf=True`).
- **Teacher checkpoint replay vs saved `val_scores.pt`**: max abs logit diff `3.34e-06` across `64` validation rows.

## 7. History Cohort Breakdown (Discovery Checkpoints)

| Cohort | Rows | New Targets | Control R@10 | Control New@10 | Treatment R@10 | Treatment New@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_0` | 0 | 0 | N/A | 0 | N/A | 0 |
| `light_1_8` | 1116 | 574 | 0.71668 | 156 | 0.69982 | 133 |
| `warm_9_31` | 1808 | 594 | 0.80403 | 78 | 0.79346 | 63 |
| `truncated_32` | 3267 | 689 | 0.75947 | 10 | 0.75966 | 15 |

## 8. Methodological Disclosures & Scope Guardrails
- **In-sample teacher score caveat**: `train_scores.pt` predictions were generated by the frozen `301D` MLP selector trained on the `< 2025-10-01` training split, so training-split MLP scores can appear sharper/better calibrated than out-of-sample validation MLP scores; Stage 2 can partially over-rely on `mlp_score` unless out-of-fold validation scores are evaluated in future work.
- **Teacher feature transform**: `mlp_score = sigmoid(raw_logit)` is a bounded `[0, 1]` float32 feature (with `-1e9` / ineligible positions masked to `0.0` before sigmoid), **not** a calibrated probability claim.
- **Metadata policy**: `snapshot_unverified` — catalog attributes (`product_family`, `brand`, etc.) reflect current snapshot metadata.
- **Single-seed conditional CIs**: Results are for a single `seed=42` fit on fixed `quota100_50` slates; paired customer-clustered 95% CIs capture query/customer sampling uncertainty given this single seed and frozen Stage 1 slate, not multi-seed variance.
- **Promotion & held-out status**: `heldout_access = false`; held-out split (`>= 2026-02-19`) was never touched. No Stage 2 model was promoted and all historical/production aliases (`checkpoints/ranker/`, `repeat-no-month-v1`) remain byte-identical.

## 9. Preserved Hashes & Reproduction
- Preserved reference/upstream hashes verified identical before and after execution (`14` files).
- Re-run command: `./venv/bin/python -m recommender.experiments.stage2_mlp_score.run`
- Verification tests: `PYTHONPATH=. ./venv/bin/pytest tests/test_stage2_mlp_score.py` and `PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only`
