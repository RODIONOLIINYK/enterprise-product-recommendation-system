# Current Project Goals & Experimental Roadmap (`plan.md`)

This document tracks the current project state, root-cause findings from our latest architectural review, and the concrete experimental roadmap for agents working on the **Enterprise Product Recommendation System**.

---

## 0. Current Project State & Active Plan (2026-11-22)

### A. Canonical `v1` Formulation Reference (`stage5-v5-shared-uv-ep10`) — Frozen & Preserved
- **Reference Pointer:** [`experiments/roadmap_v2/stage5_reference.json`](experiments/roadmap_v2/stage5_reference.json) (`SharedUVStage5Ranker`, `567,442` params, `seed=42`, full-pool over all `585` eligible formulations, zero Stage 1 MLP / zero top-50 truncation, `1.32 ms` steady CPU single-request latency). Prior 3-stage test-evaluated reference is preserved verbatim in [`experiments/roadmap_v2/stage5_3stage_reference.json`](experiments/roadmap_v2/stage5_3stage_reference.json).
- **Stage 1 (`StaticEnrichedRetriever`, pack-free `585` `base_name` BGE-M3 + `65D` structured multi-hot specs, no `sales_branch`):** `895/1,721` unseen @25, `1,186/1,721` @50.
- **Stage 2 Validation (`6,480` baskets: `314` first-observed + `6,166` returning, `2025-10-01..2026-02-18`, zero held-out Test `>= 2026-02-19` access):**
  - **Discovery Primary (Ep 10):** `R@10 = 80.89%`, `NDCG@10 = 0.63457`, `New@10 = 372/1,721 (21.62%)`, Resolved Exact-SKU `R@10 = 66.77%` / `NDCG@10 = 0.53504` / `New@10 = 214`.
  - **NDCG Companion (Ep 7):** `NDCG@10 = 0.63485`, First-Obs `NDCG@10 = 0.25926`, Resolved Exact-SKU `R@10 = 66.93%` / `NDCG@10 = 0.53624` / `New@10 = 220`.
- **Completed `v1` Stage 2 Architecture Ablations (`v5b`–`v5e`, all verified, `seed=42`):**
  - `v5b` ([`stage5_v5b_self_attn_seed42.json`](experiments/roadmap_v2/stage5_v5b_self_attn_seed42.json)): `self_attn_bias` matched control (`0.63492` vs `0.63491`); `history_residual_mlp` on Tokens `1..32` regressed slightly (`0.63466`, `+34K` params, `+0.24 ms`) — rejected.
  - `v5c` ([`stage5_v5c_self_attn_seed42.json`](experiments/roadmap_v2/stage5_v5c_self_attn_seed42.json)): 5-feature cadence-normalized self-attention bias (`earlier_hh`, `later_hh`, `t0_key_age`, `key_rel_qty`, `key_is_token0`, `wd=0.0`, no `same_base_name` / no `self_attn_mlp`): `NDCG@10 = 0.63484`, `New@10 = 362` (Disc Ep 8 `366`, `New@25 = 740`).
  - `v5d` ([`stage5_v5d_no_memory_encoder_seed42.json`](experiments/roadmap_v2/stage5_v5d_no_memory_encoder_seed42.json)): Removing `memory_encoder` (`402K` params) dropped peak `NDCG@10` to `0.63341` (`-0.00149`), `R@5 -0.34 pp`, Resolved SKU NDCG `-0.00178` — **keep 1-layer `memory_encoder`**.
  - `v5e` ([`stage5_v5e_nomem_mlp_seed42.json`](experiments/roadmap_v2/stage5_v5e_nomem_mlp_seed42.json)): Parameter-matched `no_mem_wide_mlp` (`572K`, `0.63439`, `1.65 ms`) and `no_mem_deep_mlp` (`598K`, `0.63421`, `1.84 ms`) both lost to `control` (`567K`, `0.63491`, `1.33 ms`) and ran `24–38%` slower on CPU.
  - **Root cause on `v1`:** `77.8%` (`5,041/6,480`) of `v1` validation baskets get identical `NDCG@10` across Stage 2 variants because `77.5%` are pure-repeat (`94.9%` `R@10`) and `56.1%` of returning customers have `<= 10` prior products (median `9`), while upstream `ProductTower` (`v_bn`) remained frozen and Stage 2 main loss (~83% repeats) allowed the scoring MLP to coast on `cand_38`.

### B. New 4-Year `v2_full` Dataset (Ingested & Prepared — `data/v2_full/`, `v1` Untouched)
- **Source:** `/Users/kalyma123/Downloads/{dataset.xlsx, customers.csv, items.csv}` copied into `data/v2_full/raw/`, all real paid merchandise lines (`ПРОДАЖА + ТОВАР`, all business lines, `2023-01-03..2026-11-20`).
- **Strict Chronological Split (`recommender/experiments/v2_full/config.py`):**
  - **Train (`< 2026-02-01`):** **`137,765` baskets** (`7,792` first-observed, `129,973` returning), `340,948` total targets, **`59,031` truly new targets** (`4.74×` `v1` baskets, `~6×` unseen training positives).
  - **Validation (`2026-02-01..2026-06-30`):** **`16,623` baskets** (`408` first-observed, `16,215` returning), `40,435` total targets, **`4,496` truly new targets** (`2.61×` `v1` new targets, `0` excluded).
  - **Held-Out Test (`>= 2026-07-01`):** **STRICTLY UNTOUCHED** (and completely excluded from the Colab training bundle).
  - **Universe (`< 2026-07-01`):** `460,109` dev paid records, `1,660` SKUs → **`1,141` canonical formulations** (`1,056` train-observed).
- **Completed on `v2_full`:**
  1. Full leak-free preparation (`data/v2_full/interim/prepared_v2_full/manifest.json`) with memory-mapped `float16` `cand_38` matrices and BGE-M3 embeddings.
  2. **Stage 1 Two-Tower (`checkpoints/v2_full/baseline_v5_shared_uv-seed42/two_tower/`):** 30 epochs completed on MPS — best Ep 11 reached **`2,770 / 4,496` (`61.61%`) unseen @25**, **`3,305 / 4,496` (`73.51%`) @50**, **`3,690 / 4,496` (`82.07%`) @100** (up `+9.6 pp` @25 over `v1` despite a `2×` larger catalogue).
  3. Early local MPS baseline check (Ep 1–2 before moving to Colab Pro): Ep 2 already reached `R@10 = 81.79%`, `NDCG@10 = 0.64689`, `New@10 = 1,144/4,496 (25.44%)`.

### C. Colab setup and completed Wave 1 experiments
- **Packaged & Verified:**
  - Apply the [reviewed code patch](exports/v2_full_reviewed_code.tar.gz) over the original bundle using the updated notebook. The patch contains source, synthetic tests and the product-scope configuration used to build longer histories. The existing feature matrices are reused.
  - Wave 2B also needs the v2 development commercial parquet and customer CSV beside the two archives in Drive. The notebook identifies both source paths and runs Wave 2B first.
  - Compressed leak-free bundle (`1.83 GB`, `< 2026-07-01` only, includes code + `prepared_v2_full` + Stage 1 vectors): `exports/v2_full_colab_bundle.tar.zst`
  - Ready-to-run Colab notebook (`--amp` CUDA `bf16` + `fp32` loss/eval + `--in-memory` RAM loading + `--skip-v1-verify` + per-arm Drive auto-sync/auto-resume): [`notebooks/colab_v2_full_experiments.ipynb`](notebooks/colab_v2_full_experiments.ipynb)
  - Implementation & unit tests (`4/4` passing): [`e2e_ranker.py`](recommender/experiments/v2_full/e2e_ranker.py), [`train_v2_full_baseline.py`](recommender/experiments/v2_full/train_v2_full_baseline.py), [`run_v2_full_experiments.py`](recommender/experiments/v2_full/run_v2_full_experiments.py), [`test_v2_full_pipeline.py`](tests/test_v2_full_pipeline.py).
- **Exact 6-Model Experiment Matrix (all `seed=42`, 10 epochs, full `1,141`-formulation pool, paired 95% customer-clustered bootstrap CIs over `16,623` val baskets):**

  | Model | Stage 1 Pretraining? | `ProductTower` (`v_bn`) in Stage 2 | Unseen-Only Attn Aux (`Head_attn`, no `cand_38`) | Qty & Days Before/After (Self + Cross Attn) |
  | :--- | :---: | :---: | :---: | :---: |
  | **`baseline_2stage_v5`** | Yes (frozen `u_bn`/`v_bn`) | Frozen | No | No |
  | **`arm1_2stage_unseen_attn_aux`** | Yes (frozen `u_bn`/`v_bn`) | Frozen | **Yes (`β=0.25`)** | No |
  | **`arm2_2stage_unfreeze_vbn_unseen_attn`** | Yes (frozen `u_bn`, warm-start `ProductTower`) | **Unfrozen Ep 2–10 (`LR=1e-4`)** | **Yes (`β=0.25`)** | No |
  | **`arm3_e2e_stage2_only`** | **No (Stage 2 only)** | **From scratch in Stage 2** | No | No |
  | **`arm4_e2e_unseen_attn_aux`** | **No (Stage 2 only)** | **From scratch in Stage 2** | **Yes (`β=0.25`)** | No |
  | **`arm5_e2e_unseen_attn_aux_qty_time`** | **No (Stage 2 only)** | **From scratch in Stage 2** | **Yes (`β=0.25`)** | **Yes (Self + Cross)** |

  - **Arm details:**
    - `Head_attn([v_bn, h_att, v_bn * h_att] → 128 → 1)` takes zero direct `cand_38` inputs (note: `cand_38` still shapes cross-attention weights via query `z_cand` and the 13-feature statistics bias, but not the memory values `V`), masks previously bought items out, supervises strictly on 60-day unseen adoptions (`β = 0.25`), and is discarded at inference (`0` extra params, `0 ms` latency).
    - `ProductTower` (`v_bn`) in Arms 2–5 is precomputed & cached once per catalogue refresh at inference (`0 ms` product-tower overhead per request).
    - Arm 5 adds cadence-normalized quantity & days before/after to **both** self-attention (`earlier_hh`, `later_hh`, `t0_key_age`, `key_rel_qty`, `key_is_token0`, `wd=0.0`) and cross-attention (key-side `key_rel_qty`, `key_cadence_age`, `key_gap_before`, `key_gap_after` + pairwise candidate-history `cand_before_hist`, `cand_after_hist`, `wd=0.0`).
- **Completed Wave 1 Validation Results (`2026-11-23`, Colab L4 `bf16`, `seed=42`, `16,623` val baskets, `1,141` formulations, zero held-out Test access — [`v2_full_experiments_seed42.json`](experiments/v2_full/v2_full_experiments_seed42.json)):**

  | Model (NDCG-selected) | Ep | `R@5` | `R@10` | `NDCG@10` | `New@10` (`/4,496`) | Returning `New@10` (`/3,630`) | First-Obs `New@10` (`/866`) | `New@25` | SKU `NDCG@10` (`New@10`) | CPU ms |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **`baseline_2stage_v5`** | 10 | `68.18%` | **`81.88%`** | **`0.65315`** | `1,147` (`25.51%`) | `647` (`17.82%`) | `500` (`57.74%`) | `2,351` | **`0.45437`** (`613`) | `6.70` |
  | `arm1_2stage_unseen_attn_aux` | 8 | `68.21%` | `81.77%` | `0.65251` | `1,146` (`25.49%`) | `644` (`17.74%`) | `502` (`57.97%`) | `2,357` | `0.45365` (`605`) | `6.15` |
  | `arm2_2stage_unfreeze_vbn_unseen_attn` | 10 | `68.12%` | `81.84%` | `0.65236` | `1,150` (`25.58%`) | `651` (`17.93%`) | `499` (`57.62%`) | `2,355` | `0.45312` (`614`) | `5.15` |
  | `arm3_e2e_stage2_only` | 10 | `68.05%` | `81.87%` | `0.65270` | `1,152` (`25.62%`) | `663` (`18.26%`) | `489` (`56.47%`) | **`2,405`** | `0.45300` (`607`) | `6.17` |
  | **`arm4_e2e_unseen_attn_aux`** | 10 | **`68.26%`** | `81.84%` | `0.65232` | **`1,168` (`25.98%`)** | **`667` (`18.37%`)** | `501` (`57.85%`) | `2,319` | `0.45306` (**`625`**) | `6.50` |
  | `arm5_e2e_unseen_attn_aux_qty_time` *(rejected)* | 9 | `68.15%` | `81.82%` | `0.65192` | `1,151` (`25.60%`) | `664` (`18.29%`) | `487` (`56.24%`) | `2,388` | `0.45344` (`607`) | `8.95` |

  - **Key Wave 1 & Forensic Findings:**
    1. All 5 arms' overall `NDCG@10` deltas vs. `baseline_2stage_v5` are statistically indistinguishable from zero (all 95% CIs straddle `0`, `-0.00045` to `-0.00123`). `baseline_2stage_v5` leads overall `NDCG@10` (`0.65315`); `arm4_e2e_unseen_attn_aux` (zero Stage 1 pretraining) leads `R@5` (`68.26%`), NDCG-checkpoint `New@10` (`1,168`, `+21`), returning `New@10` (`667` at Ep 10 / `673` at Ep 6), and Resolved SKU `New@10` (`625`). `arm5` regressed vs. `arm4` (`-17` New@10) and ran `38%` slower on CPU (`8.95 ms`) — **reject `arm5`**.
    2. **Causal ablation on trained `baseline` & `arm4` checkpoints (zero retraining) proved `cand_38` tabular dominance + repeat crowding:**
       - `80.5%` (`13,377/16,623`) of val baskets are pure-repeat (`88.9%` of targets), where models already hit `91.0–91.2%` R@10 / `0.728–0.729` NDCG. Returning customers average only `2.65` unseen slots in Top-10 (`7.35` repeats), yet `baseline` already achieves **`43.1%` (`1,564/3,630`) Pure Unseen@10** (`61.6%` @25) when repeats are masked out.
       - Zeroing `v_bn & u_bn` drops `baseline` only to `0.6335` (`arm4` to `0.6405`), whereas zeroing `cand_38` collapses `baseline` to `0.2213` and `arm4` to `0.0432`. `cand_38` enters the scoring MLP twice (directly + inside `z_cand = LN(v_bn + MLP([v_bn, cand_38]))`), and `h_att` across-candidate variance is small (`0.280/2.033` in `baseline`, `0.471/1.927` in `arm4`).
       - Wave 2 will therefore be run on **both `baseline_2stage_v5` and `arm4_e2e_unseen_attn_aux`** (plus CatBoost) and prioritize **new verified information + imbalance/shortcut fixes** alongside capacity/history/recency.

### D. Wave 2B (Run FIRST) & Wave 2A / Wave 3 Controlled Experiments (Implemented Locally, Ready for Audit)
- **Execution Order (Updated `2026-11-24`):**
  1. **Wave 2B (COMPLETED `2026-11-24`):** `extra_cand_5` + `cand_42` pruning + Option A 70/30 Unseen-Only Discovery Head + Pure-Q `Q = v_bn` / Full-42 cross-attention bias + CatBoost `v2_full` benchmark & ensemble → promoted **`w2b_cand42_optionA_pureQ_full42bias` (`cal-Unseen` Ep 5)** as canonical `v2_full` reference.
  2. **User Review of Option A (COMPLETED `2026-11-24`):** Reviewed `OptionAUnseenHead`, the Wave 2B 70/30 unscaled tabular feature-group dropout, cold-branch attention masking, head separation, and 2-param OOF affine calibration. Decided to replace the Wave 2B 30% tabular dropout with a clean **two-experiment Option A revision (`Wave 2C`)** before running Wave 2A.
  3. **Wave 2C — Clean Separated Option A & Loss-Weighting Sweep (COMPLETED `2026-11-25`, [`v2_full_wave2c_arm4_e2e_unseen_attn_aux_seed42.json`](experiments/v2_full/v2_full_wave2c_arm4_e2e_unseen_attn_aux_seed42.json)):**
     - **Clean inverted-U optimum at `unseen_head_weight = 1.0` (`w ∈ {0.5, 1.0, 2.0}`, `cand_37` 0% dropout, conflict-free separated targets, real unseen only, native L4 `bf16`):**
       - **`w2c_exp1_clean_sep_w10` (`1.0 * L_optA`) — PROMOTED NEW CANONICAL `v2_full` REFERENCE (`v2-full-arm4-w2c-clean-sep-w10-cal-disc-ep6`):**
         - **Primary `cal-Discovery == cal-Unseen` (Ep 6, single unified checkpoint):** `R@5 = 68.28%`, **All-time high `R@10 = 82.12%`**, **All-time high `R@25 = 92.77%`**, `NDCG@10 = 0.65380` (statistically tied vs `w05` `-0.00059 [-0.00185, +0.00068]`, `+0.00241` above Wave 2B), **All-time high `New@10 = 1,219/4,496 (27.11%)*`** (`Ret 721 / FO 498`; `+26` hits / `+0.58 pp [+0.02 pp, +1.15 pp]*` vs `w05`, `+7` vs Wave 2B), **All-time high `New@25 = 2,584`**, `PureU@10 = 2,298`, **All-time high `PureU@25 = 3,218`**, `PU NDCG@10 = 0.30730`, Resolved `SKU NDCG@10 = 0.45456` (`SKU New@10 = 630`), `6.40 ms` median CPU 1-req.
         - **Companion `cal-NDCG` (Ep 10):** `R@10 = 82.10%`, `NDCG@10 = 0.65437`, `New@10 = 1,160`, `PureU@10 = 2,276`, `PureU@25 = 3,203`, `SKU R@10 = 55.22%` / `SKU NDCG@10 = 0.45461`.
       - **`w2c_exp1_clean_sep_w05` (`0.5 * L_optA`, preserved in `prior_v2_full_baseline`):** `cal-Discovery` Ep 10 `R@10 = 82.07%`, `NDCG@10 = 0.65439`, `New@10 = 1,193`, `PureU@10 = 2,235`; `cal-NDCG` Ep 9 `NDCG@10 = 0.65465`, `SKU NDCG@10 = 0.45518`, `New@10 = 1,178`.
       - **`w2c_exp1_clean_sep_w20` (`2.0 * L_optA`, REJECTED):** Over-weighted Option A crowded the `1.0` global grad-clip norm (worse train loss on *both* heads); significantly lost vs `w10` on both `R@10` (`-0.18 pp [-0.30, -0.06]*`) and `New@10` (`-31` hits / `-0.69 pp [-1.10, -0.30]*`, `cal-Disc 1,188`), and lost `NDCG@10` vs `w05` (`0.65340` Ep 6 / `0.65405` Ep 9).
       - **`w2c_exp2_clean_sep_syn7030_w05` (REJECTED on base-size model at `w=0.5` — Ep 1–9):** Regressed across every metric (`CalNDCG 0.65332`, `CalNew@10 1,131`, `PureU@10 2,152`) on the 1-block `128D` model; retained strictly as a **separate bigger-model Wave 2A arm** (`w2_2mem_wider_mlp_syn7030`, `w=1.0`, user request) to test whether added capacity benefits from masked-$b^*$ + FO synthetic unseen rows.
  4. **Wave 2A (NEXT — TPU `v6e-1` / native-`bf16` GPU):** Clean 11-arm isolated experiment matrix built on the promoted `w=1.0` clean separated `0%`-dropout Option A `Wave2BE2ERanker` base, including an 8-arm **Model-Size Scaling Ladder** (`--preset size`: `w2_2mem_blocks`, `w2_wider_mlp`, `w2_2mem_wider_mlp`, `w2_wider_towers`, `w2_dim192`, `w2_dim192_2mem_wider_mlp`, `w2_dim256`, `w2_dim256_2mem_4mlp`) plus `w2_hist_len64`, `w2_recency_hl365`, and separate `w2_2mem_wider_mlp_syn7030`.
  5. **Wave 3 (Conditional):** `w2_hist_len96` (`--include-hist96`, OFF by default) runs ONLY if `w2_hist_len64` improves over `seq_len=32`.
- **Explicitly Dropped & Removed From Code (Do NOT Re-Add):**
  - **Arm 5 (`arm5_e2e_unseen_attn_aux_qty_time`):** Regressed in Wave 1 (`0.65192`, `-17` New@10 vs `arm4`, `8.95 ms` = `38%` slower) — removed from `e2e_ranker.py`, `run_v2_full_experiments.py`, and `run_v2_full_wave2.py` (Wave 1 `arm4` checkpoints remain `strict=True` compatible).
  - **Seasonality (`month_sin/cos`, same-time-last-year):** Verified absent on Train (`P(rebuy | dormant > 90d)` decays monotonically with zero 365-day spike; only `2.89%` of lines seasonal, already tracked by rolling 30d demand).
  - **Price-tier (`-0.56 pp`) and `warehouse90` (`+0.33 pp`):** Verified weak/negative on Train-only probe.
  - **Learned Customer / Product ID embeddings & 3-seed replication:** Rejected (rely on paired 95% customer-clustered bootstrap CIs over `16,623` val baskets on `seed=42`).

- **Canonical `v2_full` Reference (`v2-full-arm4-w2a-2mem-recency-hl365-cal-disc-ep10`, promoted `2026-11-26`):**
  - **Reference Pointer:** [`experiments/v2_full/v2_full_reference.json`](experiments/v2_full/v2_full_reference.json) -> [`checkpoints/v2_full/wave2a_arm4_e2e_unseen_attn_aux_clean_sep_w10-seed42/w2_2mem_recency_hl365/best_discovery.pt`](checkpoints/v2_full/wave2a_arm4_e2e_unseen_attn_aux_clean_sep_w10-seed42/w2_2mem_recency_hl365/best_discovery.pt) (Ep 10 `cal-Discovery` primary; Ep 9 `best_ndcg.pt == best_unseen.pt` `cal-NDCG == cal-Unseen` companion), single-stage E2E **`Wave2BE2ERanker`** (`arm4_e2e_unseen_attn_aux`, **`memory_layers=2`** [2× `Self-Attn -> FFN(384)`], **exponential recency sample weighting `half_life=365d, min_weight=0.1`**, `1,370,564` trainable / `1,370,370` inference / `1,128,498` per-request params, `46.9 min` CUDA `bf16` train, `7.17 ms` Colab CPU / `2.86 ms` Apple Silicon CPU 1-req over all `1,141` formulations, zero Stage 1 pretraining, strictly **42** candidate features `cand_42 + CUST_7 + SEQ_5`, Pure-Q `Q = v_bn`, Full-42 cross-attention bias, clean conflict-free separated Main/Option A targets, `OptionAUnseenHead` on `cand_37` with `0%` dropout, **`unseen_head_weight=1.0`**, and 2-parameter customer-grouped OOF affine blend, zero held-out Test `>= 2026-07-01` access).
  - **Reference Validation Metrics (`16,623` baskets, `40,435` targets, `4,496` truly new targets: `3,630` Ret + `866` FO):**
    - **Primary `cal-Discovery` (Ep 10, `best_discovery.pt` + OOF affine blend):** `R@5 = 68.24%`, **All-time high `R@10 = 82.26%`**, `R@25 = 92.88%`, `NDCG@10 = 0.65476`, **All-time high `New@10 = 1,255/4,496 (27.91%)`** (`Ret 761/3,630` + `FO 494/866`; `+27` hits / `+0.60 pp` vs `w2_2mem_blocks`), `New@25 = 2,627/4,496 (58.43%)`, `Pure Unseen@10 = 2,328/4,496 (51.78%)`, `Pure Unseen@25 = 3,220/4,496 (71.62%)`, `Pure Unseen NDCG@10 = 0.32592`, **All-time high Resolved SKU `R@10 = 55.24%` / `NDCG@10 = 0.45513` / `New@10 = 663`**.
    - **Companion `cal-NDCG == cal-Unseen` (Ep 9, `best_ndcg.pt == best_unseen.pt` + OOF affine blend):** `R@5 = 68.27%`, **`R@10 = 82.26%`**, **All-time high `R@25 = 92.91%`**, **All-time high `NDCG@10 = 0.65477`**, `New@10 = 1,246*` (`+40` hits / `+0.89 pp [+0.32, +1.46]*` vs prior `cal-NDCG`; `Ret 754` / `FO 492`), **All-time high `New@25 = 2,630 (58.50%)`**, **All-time high `Pure Unseen@10 = 2,350 (52.27%)`**, **All-time high `Pure Unseen@25 = 3,239 (72.04%)`**, **All-time high `Pure Unseen NDCG@10 = 0.32629`**, Resolved SKU `R@10 = 55.21%` / `NDCG@10 = 0.45502` / `New@10 = 653`.
    - **Prior Historical Baseline (Preserved in `prior_v2_full_baseline`):** `w2_2mem_blocks` (`cal-Discovery` Ep 10: `R@10 = 82.18%`, `NDCG@10 = 0.65420`, `New@10 = 1,228`, `Pure Unseen@10 = 2,323`; `cal-NDCG == cal-Unseen` Ep 9: `NDCG@10 = 0.65453`, `New@10 = 1,206`, `Pure Unseen@10 = 2,337`).
  - **Final `w2_2mem_blocks` `2×2` Factorial (`seq_len=64` × `half_life=365d`, COMPLETED `2026-11-26`, zero held-out Test access):**
    - `w2_2mem_recency_hl365` (`seq=32, hl=365d`, `46.9m`): **PROMOTED** — beats `w2_2mem_blocks` across every headline metric at 0 extra params/latency (`cal-Disc` Ep 10: `R@10 82.26% / NDCG 0.65476 / New@10 1,255 / PU NDCG 0.32592 / SKU New 663`; `cal-NDCG==Unseen` Ep 9: `R@25 92.91% / NDCG 0.65477 / New@10 1,246* / PureU@10 2,350 / PureU@25 3,239 / PU NDCG 0.32629`).
    - `w2_2mem_hist64` (`seq=64`, `49.9m`, REJECTED): regressed across the board (`R@5 -0.20 pp*`, `R@10 82.07%`, `NDCG 0.65367`, `New@10 1,214`, `PureU@10 2,248`) — skip Wave 3 `seq_len=96`.
    - `w2_2mem_hist64_recency_hl365` (`seq=64, hl=365d`, `50.0m`, REJECTED): `hl=365d` rescued `hist64` (`New@10 +0.69 pp*` vs `hist64`), but remained strictly below `w2_2mem_recency_hl365` (`R@10 82.20%`, `NDCG 0.65440`, `New@10 1,245`, `PureU@10 2,336`).
  - **Wave 2A Model-Size Scaling Interim Results (`5/8` arms completed on Colab CUDA `bf16`, `2026-11-25`, zero held-out Test access):**
    - `w2_2mem_blocks` (`1.37M`, `47.1m`): **PROMOTED** (`cal-Disc` Ep 10 `R@10 82.18% / NDCG 0.65420 / New@10 1,228 / PureU@10 2,323`; `cal-NDCG==Unseen` Ep 9 `NDCG 0.65453 / New@10 1,206* / PureU@10 2,337 / PU NDCG 0.31382`).
    - `w2_wider_mlp` (`1.49M`, `52.6m`): `cal-Disc` Ep 6 `82.03% / 0.65353 / 1,213`; `cal-NDCG` Ep 9 `82.11% / 0.65474 / 1,188* / SKU NDCG 0.45505` (`PureU@10` peaked Ep 5 `2,290`).
    - `w2_2mem_wider_mlp` (`1.69M`, `53.2m`): `cal-Disc` Ep 6 `82.02% / 0.65340 / 1,208`; `cal-NDCG` Ep 9 `82.05% / 0.65438 / 1,205*` (`PureU@10` peaked Ep 2 `2,258`).
    - `w2_wider_towers` (`1.82M`, `47.1m`): `cal-NDCG` Ep 9 hit highest `NDCG@10 0.65524*` (`+0.00143*` vs `w10` Ep 6) but dropped discovery (`New@10 1,167`, `PureU@10` peaked Ep 1 `2,269`).
    - `w2_dim192` (`1.94M`, `50.4m`, `10.24ms` CPU): `cal-Disc` Ep 7 `81.98% / 0.65275 / 1,204`; `cal-NDCG` Ep 10 `82.10% / 0.65405 / 1,183` (`PureU@10` peaked Ep 1 `2,280`).
    - **Finding:** Depth (`memory_layers: 1 -> 2`) improved both discovery and recall with healthy late-epoch generalization (Ep 9–10) at `+3.4%` GPU train cost, while width scaling (`wider_mlp`, `wider_towers`, `dim192`) caused early Option A overfitting (Ep 1–5).
  - **Active Development Standard (All Wave 2C, Wave 2A, Wave 2B & CatBoost runners):**
    - **Colab Precision Policy (`bf16` REQUIRED):** All future Colab training runs MUST use `--amp` on a **native hardware `bf16` GPU (L4 or A100, `torch.cuda.is_bf16_supported(including_emulation=False) == True`)** — forward matmuls in `bf16` with `fp32` master weights, AdamW, and `RankerLoss`. Never train on Tesla T4 (lacks hardware `bf16`; PyTorch 2.6+ software-emulates `bf16` at `~6–8×` slower speed).
    - Strictly **42 candidate features** (`cand_42 + CUST_7 + SEQ_5`; dropped dead `base_name_growth_30d` AND weak `mgr_base_name_growth_30d_vs_90d` (`RetU AUC 0.5626`); `extra_cand_5` cached under `prepared_v2_full/extra_cand_5/<identity>/`).
    - All new neural experiments (including Wave 2A) build directly on **`Wave2BE2ERanker`** (`arm4_e2e_unseen_attn_aux`) with **`pure_v_query=True`, `num_pairwise_bias_features=42`, `option_a_unseen_head=True`, `separated_main_targets=True`, `unseen_dropout_cold_prob=0.0`, `unseen_head_weight=1.0`, and per-epoch OOF affine blend selection**, comparing paired 95% CIs against `v2_full_ref_primary` / `v2_full_ref_cal_ndcg`.
    - CatBoost runs strictly the 42-feature models (`cb_mixed_cand42`, `cb_optionA_unseen_cand42`) + OOF blends/ensembles and evaluates paired 95% CIs against `v2_full_ref_primary`.

- **Wave 2B — COMPLETED Validation Results (`arm4_e2e_unseen_attn_aux`, `cand_42 + CUST_7 + SEQ_5`, `seed=42`, L4 GPU, `2026-11-24`, ZERO held-out Test `>= 2026-07-01` access):**
  - Evidence artifacts: [`experiments/v2_full/v2_full_wave2b_arm4_e2e_unseen_attn_aux_seed42.json`](experiments/v2_full/v2_full_wave2b_arm4_e2e_unseen_attn_aux_seed42.json), [`experiments/v2_full/v2_full_catboost_wave2b_seed42.json`](experiments/v2_full/v2_full_catboost_wave2b_seed42.json).

  | Model (Validation, `16,623` baskets) | `R@10` | `NDCG@10` | `New@10` (`Ret / FO`) | `Pure U@10` | `Pure U@25` | `PU NDCG@10` | `SKU NDCG@10` (`New@10`) | CPU 1-req |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | *Prior Baseline: W2B `cand_43` pruned (Disc Ep 5)* | `81.99%` | `0.65290` | `1,172` (`681 / 491`) | `2,237` | `3,139` | `0.29954` | `0.45225` (`617`) | `5.79 ms` |
  | `w2b_cand42_pruned` (NDCG Ep 8) | `81.92%` | `0.65213` | `1,171` (`677 / 494`) | `2,028` | `2,901` | `0.26960` | `0.45321` (`627`) | `5.21 ms` |
  | `w2b_cand42_optionA_70_30` (cal-NDCG Ep 10) | `81.96%` | `0.65202` | `1,206*` (`709 / 497`) | `2,226` | `3,088` | `0.30159` | `0.45279` (`641`) | `7.65 ms` |
  | `w2b_cand42_pureQ_full42bias` (NDCG Ep 7) | **`82.10%`** | **`0.65440*`** | `1,175` (`681 / 494`) | `2,089` | `3,002` | `0.27861` | **`0.45466`** (`612`) | `5.37 ms` |
  | **`w2b_cand42_optionA_pureQ_full42bias` (Ep 9 Main)** | `81.98%` | `0.65396` | `1,192` (`694 / 498`) | `2,161` | `3,097` | `0.28916` | `0.45462` (`632`) | `6.80 ms` |
  | **`w2b_cand42_optionA_pureQ_full42bias` (cal-NDCG Ep 9)** | `81.96%` | `0.65360` | `1,177` (`679 / 498`) | `2,267` | `3,160` | `0.30180` | `0.45434` (`614`) | `6.80 ms` |
  | **PROMOTED REF: `w2b_cand42_optionA_pureQ_full42bias` (cal-Unseen Ep 5)** | **`82.02%`** | `0.65139` | **`1,212*`** (`710 / 502`) | **`2,312`** | **`3,197`** | **`0.30801`** | `0.45107` (`639`) | `6.80 ms` |
  | `cb_mixed_cand42` (CatBoost 1500 trees) | `81.40%` | `0.64260` | `1,108` (`645 / 463`) | `2,171` | `3,051` | `0.28515` | `0.44422` (`565`) | `1.78 ms` |
  | `cb_optionA_unseen_cand42` (Unseen-Only CB) | `69.66%` | `0.50889` | `1,587` (`1,091 / 496`) | `2,278` | `3,162` | `0.30412` | `0.33513` (`871`) | `1.89 ms` |
  | `cb_optionA_oof_blend` (CB OOF affine) | `81.61%` | `0.64379` | `1,139` (`643 / 496`) | `2,278` | `3,162` | `0.30412` | `0.44512` (`589`) | — |
  | `ensemble (w2b_cand42_pruned + 0.2*cb_mixed_cand42)` | `82.00%` | `0.65314` | `1,171` (`674 / 497`) | `2,054` | `2,952` | `0.27592` | `0.45288` (`623`) | — |

- **Wave 2A Core Arm Matrix (NEXT — built on the promoted `w2c_exp1_clean_sep_w10` Option A `Wave2BE2ERanker` base with `cand_42 + CUST_7 + SEQ_5`, `pure_v_query=True`, `num_pairwise_bias_features=42`, `option_a_unseen_head=True`, `0%` dropout, `1.0 * L_optA` + OOF affine blend; static-shape `v6e-1 TPU` & native-`bf16` GPU support; `--preset size` runs the 8 model-size scaling arms; `w2_hist_len96` moved to conditional Wave 3):**

  | Arm Name | `d_model` / `heads` | `memory_layers` / `ffn` | `mlp_widths` | Towers (`p_hid / c_hid / emb`) | `seq_len` / Extra |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **`w2_2mem_blocks`** | `128 / 8` | **`2 / 384`** (`2× [Self-Attn→FFN]`) | `(384, 192, 96)` | `128 / 128 / 16` | `32` |
  | **`w2_wider_mlp`** | `128 / 8` | `1 /` **`512`** | **`(512, 256, 128)`** | `128 / 128 / 16` | `32` |
  | **`w2_2mem_wider_mlp`** | `128 / 8` | **`2 / 512`** | **`(512, 256, 128)`** | `128 / 128 / 16` | `32` |
  | **`w2_wider_towers`** | `128 / 8` | `1 / 384` | `(384, 192, 96)` | **`256 / 256 / 32`** | `32` (`~0 ms` prod tower) |
  | **`w2_dim192`** | **`192 / 8`** (`24D/head`) | `1 /` **`576`** | `(384, 192, 96)` | `192 / 192 / 16` | `32` |
  | **`w2_dim192_2mem_wider_mlp`** | **`192 / 8`** (`24D/head`) | **`2 / 576`** | **`(512, 256, 128)`** | **`192 / 192 / 24`** | `32` |
  | **`w2_dim256`** | **`256 / 8`** (`32D/head`) | `1 /` **`768`** | **`(512, 256, 128)`** | **`256 / 256 / 32`** | `32` |
  | **`w2_dim256_2mem_4mlp`** | **`256 / 8`** (`32D/head`) | **`2 / 1024`** | **`(768, 512, 256, 128)`** | **`384 / 384 / 32`** | `32` |
  | **`w2_hist_len64`** | `128 / 8` | `1 / 384` | `(384, 192, 96)` | `128 / 128 / 16` | **`64`** |
  | **`w2_recency_hl365`** | `128 / 8` | `1 / 384` | `(384, 192, 96)` | `128 / 128 / 16` | `32` (**`HL=365d, min=0.1`**) |
  | **`w2_2mem_wider_mlp_syn7030`** *(separate)* | `128 / 8` | **`2 / 512`** | **`(512, 256, 128)`** | `128 / 128 / 16` | `32` (**70/30 masked-$b^*$ + FO syn**) |
  | **`w2_2mem_hist64`** *(`--preset final_2mem`)* | `128 / 8` | **`2 / 384`** | `(384, 192, 96)` | `128 / 128 / 16` | **`64`** |
  | **`w2_2mem_recency_hl365`** *(`--preset final_2mem`)* | `128 / 8` | **`2 / 384`** | `(384, 192, 96)` | `128 / 128 / 16` | `32` (**`HL=365d, min=0.1`**) |
  | **`w2_2mem_hist64_recency_hl365`** *(`--preset final_2mem`)* | `128 / 8` | **`2 / 384`** | `(384, 192, 96)` | `128 / 128 / 16` | **`64` + `HL=365d, min=0.1`** |
  | **`w2_hist_len96`** *(Wave 3 conditional, `--include-hist96`)* | `128 / 8` | `1 / 384` | `(384, 192, 96)` | `128 / 128 / 16` | **`96`** |

- **File-by-File Code Intent & Auditing Contracts (What the Reviewing Agent Must Check):**
  1. **Wave 2A Runner (`run_v2_full_wave2.py`, `wave2_utils.py`, `synthetic_unseen.py`):**
     - **Intent:** Run Wave 2A exclusively on the promoted `Wave2BE2ERanker` architecture (`cand_42 + CUST_7 + SEQ_5`, `pure_v_query=True`, `num_pairwise_bias_features=42`, `option_a_unseen_head=True`, `0%` dropout, `1.0 * L_optA` + OOF affine blend) on `v6e-1 TPU` (static-shape `(B, C)` path) or native-`bf16` GPU, applying `WeightedRankerLoss` to separated `L_main`, `L_60d`, and Option A losses (`L_unseen_next`, `L_unseen_60d`), saving `best_unseen.pt` / `best_ndcg.pt` / `best_discovery.pt` + OOF calibrated predictions + `val_logits_ndcg.npy` / `val_unseen_logits_ndcg.npy`, and comparing paired 95% CIs against `v2_full_reference.json` (`v2_full_ref_primary`, `v2_full_ref_cal_ndcg`) plus direct model-size scaling comparisons across the ladder (`w2_2mem_wider_mlp`, `w2_dim192_2mem_wider_mlp`, `w2_dim256`, `w2_dim256_2mem_4mlp`, and `w2_2mem_wider_mlp_syn7030`).
     - **Check:** Verify `build_wave2_arm_specs()` returns the 11 clean `cand_42` E2E Option A + Pure-Q + Full-42 arms by default (12 with `include_hist96=True`, 8 model-size arms under `--preset size`), `v1` `SharedUVStage5Ranker` remains untouched, and `run_v2_full_wave2` enforces both calibrated reference pointers (`v2_full_ref_primary`, `v2_full_ref_cal_ndcg`).
  2. **[`recommender/experiments/v2_full/extra_features.py`](recommender/experiments/v2_full/extra_features.py):**
     - **Intent:** Define canonical pruned schemas (`CAND_42`, `CUST_7`, `SEQ_5`, `CAND_42_NON_REPEAT_INDICES` (`37` cols), `UNSEEN_37_GLOBAL_LOCAL_INDICES` (`7` cols)) and compute strictly point-in-time (`event_day < query_day`, rejecting any event/query `>= TEST_START_V2`) float16 memmaps `train_extra_cand_5.npy` & `val_extra_cand_5.npy` sealed under `prepared_v2_full/extra_cand_5/<identity>/` (weak `mgr_base_name_growth_30d_vs_90d` and dead `mgr_90` counter permanently removed).
     - **Check:** Verify strict `< query_day` causality (zero same-day leakage), exact parity of incremental `cooc`/`deg` cosine CF with brute-force `X^T X` over `< query_day`, L2-normalized 180d-half-life recency CF matching unit CF when all prior purchases share the same age, `region_base_name_share_90d` populated for first-observed customers while `mgr_*`/`cf_*` stay `0.0`, and `ValueError` on `>= TEST_START_V2`.
  3. **[`recommender/experiments/v2_full/wave2b_models.py`](recommender/experiments/v2_full/wave2b_models.py):**
     - **Intent:** Dedicated Wave 2B/2A E2E classes (`OptionAUnseenHead`, `Wave2BE2ERanker`; deprecated `Wave2BTwoStageRanker` removed) + `evaluate_pure_unseen_predictions` + `fit_oof_affine_unseen_blend` without modifying `v1` `clean_ranker.py`.
     - **Check:**
       - `OptionAUnseenHead` MUST receive pure `v_exp` (NEVER `z_cand`!) and mask `cold_features` before `_attend_candidates` on the 30% cold branch so customer-specific features cannot leak through `z_cand` or `h_att`.
       - 70/30 per-basket dropout MUST be **unscaled** (`1.0` on 70% full branch, `0.0` on 30 customer-specific cols / `1.0` on 7 global cols on 30% cold branch, `1.0` on all 37 cols at `eval()`), matching real returning vs cold-start feature distributions.
       - `strip_wave2b_training_aux` MUST strip `aux_60d_head`, `unseen_attn_head`, and `option_a_head.aux_60d_head` while PRESERVING `option_a_head.mlp_blocks` and `option_a_head.main_head` for inference.
       - `fit_oof_affine_unseen_blend` MUST use deterministic SHA-256 customer-grouped 2-fold OOF CV (`a > 0`) and exact Top-25 repeat + Top-25 unseen pre-extraction mathematically identical to full-pool ranking.
  4. **[`recommender/experiments/v2_full/run_v2_full_wave2b.py`](recommender/experiments/v2_full/run_v2_full_wave2b.py):**
     - **Intent:** End-to-end Wave 2B runner exclusively for `arm4_e2e_unseen_attn_aux`, assembling `cand_42` on-the-fly from `cand_38` + `extra_cand_5`, pruning `CUST_11 -> CUST_7`, tracking `best_ndcg.pt`, `best_discovery.pt`, and (for Option A) `best_unseen.pt` with exact replay assertions, saving `val_logits_ndcg.npy` / `val_unseen_logits_ndcg.npy` (`float32`), running 2-fold OOF affine calibration on Option A, computing paired 95% CIs vs. `v2_full_ref_primary`, CPU latency, and SHA-256 verified resume/sync.
     - **Check:** Verify Option A loss masks strictly to `unseen_bn & allowed_bn`, epoch selection never uses calibrated `(a, b)`, and `v1` preserved hashes remain untouched.
  5. **[`recommender/experiments/v2_full/run_v2_full_catboost.py`](recommender/experiments/v2_full/run_v2_full_catboost.py):**
     - **Intent:** Standalone CatBoost `v2_full` benchmark strictly on 42 candidate features (`cb_mixed_cand42`, `cb_optionA_unseen_cand42`; deprecated `cb_mixed_cand38` removed) + 2-fold customer-grouped OOF Option A affine blend + CatBoost/Neural OOF linear ensemble (`fit_oof_linear_ensemble`) + paired 95% CIs vs. `v2_full_ref_primary`.
     - **Check:** Verify exact Horvitz-Thompson importance weights (`sum(w_neg) == total_eligible_negatives`), zero early stopping on validation, zero repeat features/targets in `cb_optionA_unseen_cand42`, full `1,141`-formulation validation scoring (`~allowed_bn` masked to `-inf`), and disjoint calibration customers in both folds.
  6. **[`tests/test_v2_full_wave2b.py`](tests/test_v2_full_wave2b.py), [`tests/test_v2_full_wave2.py`](tests/test_v2_full_wave2.py), [`tests/test_v2_full_pipeline.py`](tests/test_v2_full_pipeline.py):**
     - **Intent:** Full regression coverage for `cand_42`, `extra_cand_5`, Wave 2B E2E + CatBoost, Wave 2A E2E, `v2_full_reference.json` checksum verification, temporal boundaries, and resume.
- **Auditing Agent Verification Commands:**
  ```bash
  PYTHONPATH=. ./venv/bin/pytest tests/test_v2_full_wave2b.py tests/test_v2_full_wave2.py tests/test_v2_full_pipeline.py tests/test_stage5_v5b_self_attn.py tests/test_stage5_native_bn.py -v
  PYTHONPATH=. ./venv/bin/pytest tests/ -q
  ./venv/bin/ruff check recommender/experiments/v2_full/ tests/test_v2_full_wave2.py tests/test_v2_full_wave2b.py tests/test_v2_full_pipeline.py
  ```
- **Example Colab / CLI Invocations (Wave 2B First):**
  ```bash
  # 1. Precompute & seal extra_cand_5 (strictly < 2026-07-01)
  PYTHONPATH=. python -m recommender.experiments.v2_full.extra_features

  # 2. Wave 2B Neural Arms (exclusively arm4_e2e_unseen_attn_aux on cand_42)
  PYTHONPATH=. python -m recommender.experiments.v2_full.run_v2_full_wave2b \
      --base arm4_e2e_unseen_attn_aux --seed 42 --amp --in-memory --skip-v1-verify \
      --sync-dir "/content/drive/MyDrive/v2_full_colab/outputs"

  # 3. Wave 2B CatBoost (cand_42) + CatBoost/arm4 Ensemble
  PYTHONPATH=. python -m recommender.experiments.v2_full.run_v2_full_catboost \
      --task-type auto --seed 42 --in-memory --skip-v1-verify \
      --neural-logits checkpoints/v2_full/wave2b_arm4_e2e_unseen_attn_aux-seed42/w2b_cand42_pruned/val_logits_ndcg.npy \
      --sync-dir "/content/drive/MyDrive/v2_full_colab/outputs"
  ```

---

## Historical Pre-Stage-5 Campaign Notes (`v1`, 2026-11-10..2026-11-14)

> **Current Reference for Future Stage 1 `FeatureAblatedStage1Selector` MLP Work (`repeat-no-month-v1`, 2026-11-13):**
> - **Canonical Active Pointer & Reference Pair:** [`experiments/stage1_mlp_reference/reference.json`](experiments/stage1_mlp_reference/reference.json) -> [`checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json`](checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json) (`best_retriever.pt` epoch 16 + `best_selector.pt` epoch 5, `seed=42`, `FeatureAblatedStage1Selector` `uv_45`, `301D`, superseding `no-brand-v1` via explicit user override; see [`experiments/stage1_mlp_reference/REFERENCE.md`](experiments/stage1_mlp_reference/REFERENCE.md) and [`experiments/mlp_repeat_restored_v1/PROMOTION_DECISION.md`](experiments/mlp_repeat_restored_v1/PROMOTION_DECISION.md)).
> - **Frozen Historical Pointer:** [`experiments/embedding_ablation_v1/reference.json`](experiments/embedding_ablation_v1/reference.json) (`no-brand-v1`, `uv_47`, `303D`) remains **frozen byte-for-byte** for prior campaign reproduction (`mlp_feature_ablation`, `description_cleaning`, `mlp_repeat_restored`); historical campaign `promoted: false` flags remain preserved.
> - **What Changed (`301D` vs `no-brand-v1` `303D`):** Keeps the frozen `E_no_brand` Stage 1 towers (separate learned `brand` categorical embedding absent; `category`, `subcategory`, `business_line`, `tier`, `unit`, `country_of_origin` kept; `sales_branch` customer embedding kept; `t1_fam` text with `brand`/`product_family` and no descriptions kept; `brand` scalar/familiarity features kept; demand disabled `12D` `container_type` only), **retains `is_repeat_sku`**, and omits **ONLY `month_sin` and `month_cos`** (`39` context + `6` customer stats = `45` scalars + `256` `u`/`v` = `301D`; legacy numeric normalization unchanged, **NOT yet all `[0, 1]`**).
> - **Validation-Only Metrics (`6,191` queries, `10,276` total targets, `1,857` new targets, `2025-10-01`..`2026-02-18`):** Recall@5 `0.64335`, Recall@10 `0.75411`, Recall@25 `0.84850`, Recall@50 `0.89542`, Recall@100 `0.93733`, NDCG@10 `0.58633`, ordinary new hits@10 `194/1,857`, unseen hits@25 `726/1,857`, @50 `1,029/1,857`, @100 `1,337/1,857` (explicit user selection accepting higher NDCG@10 / Recall@5 / unseen@100 and lower Recall@10 / top-10 discovery vs `no-brand-v1` on single `seed=42`, selected epochs `16`/`5`; do not claim statistical superiority; numeric normalization and category-feature work remain pending).
> - **Narrow Scope:** Default baseline **only for future Stage 1 MLP selector work** — **NOT** an untested Stage 2 (`UPS-CANR`) or production retriever default; historical Stage 1 discovery-reference and Stage 2 aliases remain untouched.

> **Stage 2 decision (2026-11-13):** Exclude the per-item Stage 1 MLP score from Stage 2 attention and scoring; retain the existing Stage 2 reference and the `repeat-no-month-v1` Stage 1 selector. The user instructed not to run training yet. See [experiments/stage2_mlp_score_v1/DECISION.md](experiments/stage2_mlp_score_v1/DECISION.md).

> **Rolling Statistics Policy (`rolling_observed_v2`, 2026-11-13):** Historical caches (`dynamic_campaign_v1`, `frozen_train_cutoff_v1`) freeze globals/relationship counts at `TRAIN_CUTOFF` during validation and are **prohibited as defaults for new MLP or Stage 2 work**. Use **`rolling_observed_v2`** ([`recommender/experiments/rolling_candidate_v2/`](recommender/experiments/rolling_candidate_v2/)) for new preparation/inference (`< as_of_date` strictly prior observed purchases only, zero held-out access, sealed v3 caches); keep `frozen_train_cutoff_v1` strictly for historical reproduction. Existing frozen weights (`repeat-no-month-v1`) were not retrained for rolling inputs.

> **Completed follow-up (2026-11-13):** Historical CatBoostClassifier recovery/current-data retraining, separate company-text/attribute tokens, and 64-event unpooled self-attention memory have now been evaluated on matched rolling candidates (seed 42). The original 32-event pointwise control remains strongest overall; no reference was promoted. See [the combined comparison](experiments/current_model_comparison_v1/REPORT.md) before proposing these experiments again.

> **Completed capacity / standalone-tree follow-up (2026-11-13):** A wider scoring MLP, wider Transformer FFN, and two-layer history Transformer were each tested separately (seed 42, 10 epochs, unchanged rolling 100-item slates). None beat saved Arm A (`R@10=0.77104`, `NDCG@10=0.60223`, `255` new hits). Standalone CatBoost now directly scores every eligible SKU using 52 categorical/statistical features, with no neural vectors, scores or candidate selection: `R@10=0.75815`, `NDCG@10=0.59373`, `202/1,857` new hits. Training uses uniformly sampled negatives with inverse-probability weights; validation evaluates the full eligible pool. See [the capacity/standalone report](experiments/ranker_capacity_v1/REPORT.md). No model was promoted; do not suggest these configurations as untried experiments.

---

Audit prerequisite (2026-11-10): resolve the artifact mismatch and fixed train-cutoff demand issue in [AUDIT_REPORT.md](AUDIT_REPORT.md) before treating the historical baseline below as locally reproducible. The audit preserved all checkpoints and implemented no new architecture. The shared-vector experiment now excludes source index 8 from tokens, uses three type dimensions, and preserves its effect only in the default mask-derived legacy bias. The nine-feature source format remains unchanged; use the ordered schema constants.

Follow-up (2026-11-12): [hardening implementation](HARDENING_REPORT.md) preserves/replays the references, defaults new Stage 1 training to versioned demand-free inputs, and introduces validated v4 base caches and provenance-bound isolated runs. This does not retroactively repair the historical retriever or establish a new architecture winner. Effective-date metadata and basket-aware encoding remain separate work; the completed E1–E5 results remain in their report.

Completed P0–P7 (2026-11-12): [68-fit dynamic-selector/candidate-context campaign](DYNAMIC_SELECTOR_EXPERIMENTS.md) provides paired seeds, full denominator metrics, depth-matched controls, ablations and latency/cache costs. Wider dot-selected slates provide the clearest conservative improvement; additional MLP selection and candidate attention do not establish a reliable incremental benefit. Added scalar ranker features improve means but remain optional pending a retention decision. Existing default aliases remain unchanged. This supersedes speculative priorities below where those hypotheses have now been tested.

> **Completed direct-feature / underfitting audit (2026-11-14):** All 41 rolling CatBoost context scalars already reach Stage 2 directly. A new four-arm seed42/10-epoch screen added direct profile8 inputs, independent candidate SKU/category/line embeddings, or both; none beat its fresh control (`NDCG@10=0.60105`, `251` new hits) or saved A (`0.60223`, `255`). The original model memorized all reachable targets in a 128-query training-only probe; larger-model curves show lower training loss without improved validation. Training has 2,319 customers, median 8 target instances/SKU, and 448/841 SKUs with at most 10. This does not establish a capacity shortage. See [the feature/fit audit](experiments/direct_feature_access_v1/REPORT.md); preserve references and do not suggest the same direct-input configurations as untried.

> **Completed fixed-input regularization / augmentation screen (2026-11-14):** Transition/co-purchase scalars were verified as direct inputs to the active Stage1 MLP and Stage2 scorer. Three standalone MLP fits use exactly the frozen standalone CatBoost52 raw fields, fixed sampled pairs/importance weights and binary objective; zero dropout/weight decay plus rotating negatives reaches `R@10=0.76263`, `NDCG@10=0.59372`, `224` new hits, versus CatBoost `0.75815/0.59373/202`, with 32,953 MLP parameters. Paired uncertainty does not establish superiority over CatBoost. Rotating negatives covers 13.24M distinct comparisons while preserving all 45,225 eligible positives. Seven Stage2 stage-wise/combined dropout/weight-decay fits do not beat saved A overall; all direct52 inputs plus zero regularization reaches `0.76403/0.59705/232`. A zero-regularization discovery checkpoint has 258 new hits with lower overall metrics. No defaults were promoted. Historical17, retrieved53, and standalone52 CatBoost recipes are explicitly separated in [the authoritative report](experiments/regularization_signal_v1/REPORT.md). These tested configurations should not be proposed again as untried; using the new small MLP to rebuild Stage2 candidates remains untested.

## 1. Current Baseline & Core Objective

### Active Reference Baseline (Verified on Validation: `6,191` Baskets, `1,857` New-to-Customer Targets)
- **Stage 1 Retriever (`checkpoints/retriever-discovery-reference/best_retriever.pt`):**
  - 128D two-tower model with 4 familiarity flags (`brand`, `product_category`, `business_line`, `product_family`) over the rolling prior-sale candidate pool. Includes `container_type` as one-hot packaging with an explicit `none` value following discussion with the user.
  - Retrieves **`652 / 1,857` (`35.11%`)** new-to-customer targets in Top-25 unseen and **`921 / 1,857` (`49.60%`)** in Top-50 unseen.
  - Combined with up to 25 personal repeat SKUs (`heuristic_fraction = 0.5`), the 50-item candidate slate fed to Stage 2 contains **`763 / 1,857` (`41.09%`)** new-to-customer targets and **`8,404 / 8,579` (`97.96%`)** repeat targets (`87.48%` overall Recall@50).
- **Stage 2 Ranker (`checkpoints/ranker/best_ranker.pt`):**
  - `UPSCANRRanker` with `statistics_attention = True` and `excluded_statistics = ("history_repeat",)`.
  - **Overall Recall@10:** `76.15%` (`+2.61` pp over the Personal Repeat heuristic's `73.54%`)
  - **Overall NDCG@10:** `0.5924` (`+0.0258` over the Personal Repeat heuristic's `0.5666`)
  - **New-to-Customer Recall@10:** **`11.25%` (`209 / 1,857` hits)**

### Primary Project Goal
> **Substantially increase Stage 2 new-to-customer product discovery in the Top 10 (currently `209 / 1,857 = 11.25%`, out of `763 / 1,857 = 41.09%` present in the 50-item candidate slate) while preserving or improving overall Recall@10 (`>= 76.15%`) and NDCG@10 (`>= 0.5924`).**

---

## 2. Key Architectural Diagnosis (Why Recent Stage 2 Changes Did Not Improve Discovery)

The behavioral observations below are verified in source; explanations for why past experiments lost discovery hits remain hypotheses, not isolated causal evidence. Before designing any new Stage 2 experiment, agents must understand the five structural findings identified in our review of [`recommender/ranker.py`](recommender/ranker.py):

1. **Statistics-Attention Zeros Out Memory Token Values (`ranker.py` lines 513–514):**
   - In `UPSCANRRanker.forward()`, when `statistics_attention=True` (`statistics_in_scores=True`), `tabular_features` and `sequence` are replaced with `torch.zeros_like(...)` before being passed into `_compute_customer_memory()`.
   - Consequently, customer lifetime statistics and historical purchase habit/cadence statistics affect **only the cross-attention weights $\alpha$**, not the memory value vectors $V$!
   - Even if a feature (such as reorder due-ness) shifts attention toward a specific historical token, the resulting attended context vector $\mathbf{h}_{\text{attended}}$ does not carry that numerical statistic into the final scoring MLP.
2. **The Existing $14 \times 17$ Bilinear Attention Bias Already Captures Cross-Feature Interactions:**
   - `statistics_attention_weights` has shape `(num_heads=4, 14, 17)`, computing $B_h(c, m) = [1, \mathbf{s}_c]^\top W_h \mathbf{s}_m$ across all 13 candidate pairwise features (+ intercept) and all 17 memory statistics (9 history + 8 company).
   - Forcing statistics into identical 11D coordinates ("aligned statistics") restricted expressive power and paired unrelated quantities without changing the final scoring MLP inputs.
3. **Freezing Pretrained Stage 1 Vectors Inside History Tokens Hurts Adaptation:**
   - Replacing trainable Stage 2 history embeddings with frozen Stage 1 product vectors reduced new-product discovery from `209` to `189` hits (`-9.6%`) because Stage 1 vectors compress SKU information for broad retrieval rather than fine-grained Stage 2 repeat-vs-discovery ranking, and subsequent Transformer/QKV projections destroy direct coordinate alignment anyway.
4. **Default `RankerLoss` Does Not Prioritize New-to-Customer Positives (`new_positive_weight = 1.0`):**
   - In a 50-item slate with 25 repeat SKUs and 25 unseen SKUs, repeat targets (`8,404` in validation slates) are ~11× more numerous and much easier to predict from direct SKU history than unseen targets (`763` in validation slates).
   - With `new_positive_weight = 1.0`, the network optimizes overall BCE + listwise loss primarily by ranking repeat items accurately.
5. **Candidate Pairwise Features Lack Explicit Co-Purchase / Transition Signals and Family Familiarity:**
   - For an unseen candidate SKU (`is_repeat_sku = 0`), its SKU-specific repeat features (`user_sku_order_count_norm`, `user_sku_qty_share`, `days_since_last_sku_buy_norm`, `sku_urgency`) are all constant (`0.0` or `1.0`).
   - To rank which of the 25 unseen candidates in the slate a customer will actually buy first, the final scoring MLP needs **new predictive information** about sequential product transitions (e.g., *customers who bought historical SKU $A$ frequently buy candidate SKU $B$ next*) and the missing product-family familiarity signal in the scoring head. Category, brand and business-line repeat flags are already present at candidate indices 10–12.

---

## 3. Prioritized Experimental Roadmap for Agents

Future experiments must be run **one controlled change at a time** against the clean Stage 2 reference (`209 / 1,857` new-to-customer hits, `76.15%` Recall@10, `0.5924` NDCG@10 on validation, `seed=42`, 10 epochs). Keep only changes that demonstrably improve validation metrics; revert any change that does not.

### Priority 1: Isolate Where Statistics Enter `UPSCANRRanker` (Attention Bias vs. Token Values vs. Both)
- **Hypothesis:** Zeroing `tabular_features` and `sequence` inside `_compute_customer_memory()` when `statistics_attention=True` starves the value vectors $V$ and `h_attended` of quantitative cadence and habit information.
- **Controlled Experiment:**
  - Compare three clean modes on the exact same cached slates:
    1. `statistics_in_bias_only` (current reference: `209` discovery hits, `76.15%` Recall@10)
    2. `statistics_in_values_only` (`statistics_attention=False`)
    3. `statistics_in_both` (keep `statistics_attention=True` for the cross-attention bias **and** pass the real `tabular_features` and `sequence` into `_compute_customer_memory()`).

### Priority 2: Residual Adaptation of Stage 1 Vectors Instead of Freezing
- **Hypothesis:** If Stage 1 product vectors are used in Stage 2 history tokens, they should be combined via a zero-initialized residual adapter rather than replacing trainable Stage 2 embeddings with frozen vectors:
  $$\mathbf{h}_i = \mathbf{h}_i^{\text{trainable}} + \alpha \cdot \text{Proj}(\mathbf{v}_i^{\text{Stage 1}})$$
- **Controlled Experiment:** Test whether adding a residual projection from `catalog_product_vectors` to the trainable history token representation improves discovery without losing trainable SKU plasticity.

### Priority 3: Discovery-Aware Objective Weighting (`new_positive_weight > 1.0`) & Calibration

> **User-reported completed experiment (2026-11-13):** Repeat/new objective weighting has already been tried without improvement. Do not rerun this sweep by default. Historical numeric evidence has not yet been recovered in the current working tree; this status records the user’s correction.
- **Hypothesis:** Upweighting unseen positive targets (`new_positive_weight` in `{1.25, 1.5, 2.0}`) in `RankerLoss` forces the scoring MLP and cross-attention head to allocate capacity to differentiating among unseen candidates, while `discovery_min_overall_recall` guards overall Recall@10.
- **Controlled Experiment:** Sweep `new_positive_weight` in `{1.25, 1.5, 2.0}` on the clean reference architecture and inspect the Pareto frontier between New-to-Customer Recall@10 and Overall Recall@10 / NDCG@10.

### Priority 4: Add True New-Information Signals for Unseen Candidates in Stage 2
- **Hypothesis:** Changing how the network routes existing features has diminishing returns; improving unseen SKU discovery requires **new predictive features** for unseen candidates (`is_repeat_sku == 0`):
  1. **Point-in-time sequential co-purchase / transition affinity ($P(p_{\text{cand}} \mid \text{history}_{<t})$):** Empirical conditional probability or PMI that candidate SKU $p$ is bought in the same or next basket after the customer's recent historical SKUs, computed strictly from training baskets (`< 2025-10-01`).
  2. **Add the missing family familiarity flag:** Candidate indices 10–12 already carry category, brand and business-line repeat flags. Test adding product-family familiarity as a new feature with explicit cache/checkpoint schema versioning; do not duplicate the three existing flags.

---

## 4. Workflow Rules for Agents

1. **Always benchmark against the clean reference:** Every experiment must report:
   - Validation New-to-Customer Hits@10 and Recall@10 (`out of 1,857`)
   - Validation Overall Recall@10 and NDCG@10 at the **same checkpoint** (as well as peak NDCG@10 if different)
2. **Never leave dead experimental flags or unused modules in the repo:** If an experiment fails to beat the reference, log its numbers in Section 4 of [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) and revert the code changes immediately.
3. **Never touch the Test split (`>= 2026-02-19`)** until the user explicitly signs off on the final validation model.

## 5. Completed shared-representation experiment (2026-11-11)

See [SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md) and `experiments/shared_stage1/`. Three-seed mean discovery hits: R0 196.67, full shared R3 196.33, reduced-bias R4 195.67, no-customer-token R3P0 200.00. The preserved discovery checkpoint independently replays 209 hits, 76.1505% Recall@10 and 0.591893 NDCG@10. These are distinct selected checkpoints, not mixed historical peaks.

Keep the existing ranker/bias defaults. The later isolated rolling-data capacity screen is recorded above; the previously proposed shared-vector edge/scorer experiments remain deferred. Establish a separately versioned rolling-demand Stage 1 control before claiming a temporally clean shared-vector comparison; do not mix that retraining into these frozen-retriever ablations. Input losslessness and working gradients alone did not produce a discovery improvement.

## 6. Completed E1–E5 campaign (2026-11-11–12)

See [E1_E5_EXPERIMENTS.md](E1_E5_EXPERIMENTS.md) for 32 fits and the architecture decision. Keep the original single-vector/32-event Transformer and 13-feature UPS-CANR shape. E1 and E2 showed small conditional fresh-control gains without evidence sufficient for promotion; E3 failed the screen and E4/E5 produced duplicate interests. The demand-disabled scratch control improved Stage 1 coverage but not reproducible downstream discovery. Default checkpoints and all historical/reference caches remain unchanged; no test evaluation was performed. Reproduction and cache-serving contracts are in [experiments/e1_e5/README.md](experiments/e1_e5/README.md).
