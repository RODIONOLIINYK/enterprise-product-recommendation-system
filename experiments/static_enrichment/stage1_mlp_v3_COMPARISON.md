# Compact Stage 1 Selector (`uv_47` / `uv_47_ref`) & Stage 2 Isolation Report (`seed=42`)

All metrics computed strictly on validation transitions (`< 2026-02-19`, `6,191` transitions, `1,857` new targets). Zero test-split access.

## 1. Stage 1 Full-Pool Ranking & `quota100_50` Slate Coverage

| Stage 1 Scorer | Input Dim | Ep | Sales Branch | Unseen@25 | Unseen@50 | Recall@10 | NDCG@10 | Slate New Targets (Cov) | Slate Overall Cov |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| C0_clean + dot | 128D dot | 13 | No | 710 | 972 | 0.34870 | 0.20980 | 1230 (67.96%) | 93.60% |
| P_t1_fam_clean + dot (NO sales_branch) | 128D dot | 21 | No | 749 | 1037 | 0.35373 | 0.21049 | 1267 (70.00%) | 94.02% |
| U0_t1_fam_clean + dot (WITH sales_branch) | 128D dot | 26 | Yes (16D) | 737 | 1009 | 0.37333 | 0.22104 | 1274 (70.39%) | 93.99% |
| U0 + CompactStage1Selector (`uv_47`) | 303D MLP | 1 | Yes (16D in u) | 737 | 1012 | 0.75052 | 0.57713 | 1254 (69.28%) | 94.04% |
| U0 + CompactStage1Selector (`uv_47_ref`) | 304D MLP | 1 | Yes (16D in u) | 703 | 993 | 0.74662 | 0.57582 | 1251 (69.12%) | 93.96% |

## 2. Stage 2 End-to-End `UPS-CANR` Results (Discovery vs NDCG Checkpoints)

| Config | Sel | Ep (Disc/NDCG) | R@5 (Disc) | R@10 (Disc) | NDCG@10 (Disc) | Repeat R@10 | New@10 (Disc) | Cond New (Disc) | Slate New | R@10 (NDCG) | NDCG@10 (NDCG) | New@10 (NDCG) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_clean-dot100_41 (v2 ref) | disc=ndcg | 8 / 8 | 0.65251 | 0.76208 | 0.59748 | 0.88407 | 241 (12.98%) | 19.59% | 1230 | 0.76208 | 0.59748 | 241 |
| U0-dot100_41 | dot | 6 / 6 | 0.65343 | 0.76457 | 0.59606 | 0.88775 | 233 (12.55%) | 18.29% | 1274 | 0.76457 | 0.59606 | 233 |
| U0-mlp303_100_41 | uv_47 | 7 / 6 | 0.65015 | 0.75973 | 0.59461 | 0.88348 | 226 (12.17%) | 18.02% | 1254 | 0.76377 | 0.59665 | 224 |
| U0-mlp304_100_41 | uv_47_ref | 6 / 6 | 0.65525 | 0.76516 | 0.59667 | 0.88859 | 233 (12.55%) | 18.63% | 1251 | 0.76516 | 0.59667 | 233 |
| P-dot100_41 | dot | 7 / 6 | 0.64782 | 0.76065 | 0.59311 | 0.88300 | 234 (12.60%) | 18.47% | 1267 | 0.76342 | 0.59475 | 230 |
| P-dot100_41-init_c0 | dot | 7 / 6 | 0.65068 | 0.76189 | 0.59365 | 0.88419 | 243 (13.09%) | 19.18% | 1267 | 0.76469 | 0.59542 | 239 |
| P-c0_slate_init_c0-41 | dot | 10 / 6 | 0.65357 | 0.75920 | 0.59547 | 0.88312 | 233 (12.55%) | 18.94% | 1230 | 0.76595 | 0.59606 | 231 |

## 3. Warm Single-Request Latency (`U0_t1_fam_clean` Cached Vectors -> 100-Item Slate -> `UPS-CANR`)

| Device | Pipeline | E2E Median (ms) | E2E p95 (ms) | Selector (ms) | Select100 (ms) | Ranker (ms) |
|---|---|---:|---:|---:|---:|---:|
| cpu | U0-dot100_41 | 1.014 | 1.137 | 0.015 | 0.039 | 0.860 |
| cpu | U0-mlp303_100_41 | 1.478 | 1.973 | 0.415 | 0.045 | 0.929 |
| cpu | U0-mlp304_100_41 | 1.517 | 2.002 | 0.453 | 0.046 | 0.935 |
| mps | U0-dot100_41 | 4.117 | 10.638 | 0.269 | 0.965 | 2.636 |
| mps | U0-mlp303_100_41 | 4.197 | 9.001 | 0.491 | 0.994 | 2.644 |
| mps | U0-mlp304_100_41 | 4.174 | 10.603 | 0.573 | 1.067 | 2.623 |
