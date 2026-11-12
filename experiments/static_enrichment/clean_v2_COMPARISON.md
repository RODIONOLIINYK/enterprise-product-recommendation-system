# Clean Static Towers + Transferred 6-Stat Scoring MLP (seed=42)

All metrics computed strictly on the 6,191 validation transitions / 1,857 new-to-customer targets. No test-split access.

## Stage 1 Clean Towers (0 customer tabular stats, 12D container_type only, 30 epochs, seed=42)

| Variant | Ep | Params | Fit (s) | Unseen Hits@25 | Recall@25 | Unseen Hits@50 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0_clean | 13 | 822,829 | 50.7 | 710 | 38.23% | 972 | 52.34% |
| P_t1_fam_clean | 21 | 853,309 | 49.6 | 749 | 40.33% | 1037 | 55.84% |
| U0_t1_fam_clean | 26 | 855,469 | 49.4 | 737 | 39.69% | 1009 | 54.33% |
| U0_t1_fam_nosubcat | 20 | 850,637 | 50.5 | 719 | 38.72% | 1015 | 54.66% |

Selected `best_clean = P_t1_fam_clean`.

## Downstream Stage 2 (quota100_50, 10 epochs, seed=42)

| Config | Ep | Qual | Params (Trainable) | Fit (s) | New Hits@10 | New Recall@10 | Recall@10 | NDCG@10 | Slate New Targets |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| C0_clean-dot100_41 | 8 | True | 1,017,482 (815,498) | 134.4 | 241 | 12.98% | 0.762081 | 0.597476 | 1230 |
| P_t1_fam_clean-dot100_41 | 7 | True | 1,017,482 (815,498) | 134.5 | 239 | 12.87% | 0.761552 | 0.593200 | 1267 |
| P_t1_fam_clean-dot100_47 | 7 | True | 1,019,018 (817,034) | 143.4 | 233 | 12.55% | 0.760275 | 0.592892 | 1267 |
| P_t1_fam_clean-precomputed_mlp_47 | 1 | False | 184,833 (184,833) | 59.9 | 198 | 10.66% | 0.745342 | 0.575605 | 1267 |
