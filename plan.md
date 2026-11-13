# Current Project Goals & Experimental Roadmap (`plan.md`)

This document tracks the current project state, root-cause findings from our latest architectural review, and the concrete experimental roadmap for agents working on the **Enterprise Product Recommendation System**.

> **Current Reference for Future Stage 1 `FeatureAblatedStage1Selector` MLP Work (`repeat-no-month-v1`, 2026-11-13):**
> - **Canonical Active Pointer & Reference Pair:** [`experiments/stage1_mlp_reference/reference.json`](experiments/stage1_mlp_reference/reference.json) -> [`checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json`](checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json) (`best_retriever.pt` epoch 16 + `best_selector.pt` epoch 5, `seed=42`, `FeatureAblatedStage1Selector` `uv_45`, `301D`, superseding `no-brand-v1` via explicit user override; see [`experiments/stage1_mlp_reference/REFERENCE.md`](experiments/stage1_mlp_reference/REFERENCE.md) and [`experiments/mlp_repeat_restored_v1/PROMOTION_DECISION.md`](experiments/mlp_repeat_restored_v1/PROMOTION_DECISION.md)).
> - **Frozen Historical Pointer:** [`experiments/embedding_ablation_v1/reference.json`](experiments/embedding_ablation_v1/reference.json) (`no-brand-v1`, `uv_47`, `303D`) remains **frozen byte-for-byte** for prior campaign reproduction (`mlp_feature_ablation`, `description_cleaning`, `mlp_repeat_restored`); historical campaign `promoted: false` flags remain preserved.
> - **What Changed (`301D` vs `no-brand-v1` `303D`):** Keeps the frozen `E_no_brand` Stage 1 towers (separate learned `brand` categorical embedding absent; `category`, `subcategory`, `business_line`, `tier`, `unit`, `country_of_origin` kept; `sales_branch` customer embedding kept; `t1_fam` text with `brand`/`product_family` and no descriptions kept; `brand` scalar/familiarity features kept; demand disabled `12D` `container_type` only), **retains `is_repeat_sku`**, and omits **ONLY `month_sin` and `month_cos`** (`39` context + `6` customer stats = `45` scalars + `256` `u`/`v` = `301D`; legacy numeric normalization unchanged, **NOT yet all `[0, 1]`**).
> - **Validation-Only Metrics (`6,191` queries, `10,276` total targets, `1,857` new targets, `2025-10-01`..`2026-02-18`):** Recall@5 `0.64335`, Recall@10 `0.75411`, Recall@25 `0.84850`, Recall@50 `0.89542`, Recall@100 `0.93733`, NDCG@10 `0.58633`, ordinary new hits@10 `194/1,857`, unseen hits@25 `726/1,857`, @50 `1,029/1,857`, @100 `1,337/1,857` (explicit user selection accepting higher NDCG@10 / Recall@5 / unseen@100 and lower Recall@10 / top-10 discovery vs `no-brand-v1` on single `seed=42`, selected epochs `16`/`5`; do not claim statistical superiority; numeric normalization and category-feature work remain pending).
> - **Narrow Scope:** Default baseline **only for future Stage 1 MLP selector work** — **NOT** an untested Stage 2 (`UPS-CANR`) or production retriever default; historical Stage 1 discovery-reference and Stage 2 aliases remain untouched.

> **Stage 2 decision (2026-11-13):** Exclude the per-item Stage 1 MLP score from Stage 2 attention and scoring; retain the existing Stage 2 reference and the `repeat-no-month-v1` Stage 1 selector. The user instructed not to run training yet. See [experiments/stage2_mlp_score_v1/DECISION.md](experiments/stage2_mlp_score_v1/DECISION.md).

> **Rolling Statistics Policy (`rolling_observed_v2`, 2026-11-13):** Historical caches (`dynamic_campaign_v1`, `frozen_train_cutoff_v1`) freeze globals/relationship counts at `TRAIN_CUTOFF` during validation and are **prohibited as defaults for new MLP or Stage 2 work**. Use **`rolling_observed_v2`** ([`recommender/experiments/rolling_candidate_v2/`](recommender/experiments/rolling_candidate_v2/)) for new preparation/inference (`< as_of_date` strictly prior observed purchases only, zero held-out access, sealed v3 caches); keep `frozen_train_cutoff_v1` strictly for historical reproduction. Existing frozen weights (`repeat-no-month-v1`) were not retrained for rolling inputs.

---

Audit prerequisite (2026-11-10): resolve the artifact mismatch and fixed train-cutoff demand issue in [AUDIT_REPORT.md](AUDIT_REPORT.md) before treating the historical baseline below as locally reproducible. The audit preserved all checkpoints and implemented no new architecture. The shared-vector experiment now excludes source index 8 from tokens, uses three type dimensions, and preserves its effect only in the default mask-derived legacy bias. The nine-feature source format remains unchanged; use the ordered schema constants.

Follow-up (2026-11-12): [hardening implementation](HARDENING_REPORT.md) preserves/replays the references, defaults new Stage 1 training to versioned demand-free inputs, and introduces validated v4 base caches and provenance-bound isolated runs. This does not retroactively repair the historical retriever or establish a new architecture winner. Effective-date metadata and basket-aware encoding remain separate work; the completed E1–E5 results remain in their report.

Completed P0–P7 (2026-11-12): [68-fit dynamic-selector/candidate-context campaign](DYNAMIC_SELECTOR_EXPERIMENTS.md) provides paired seeds, full denominator metrics, depth-matched controls, ablations and latency/cache costs. Wider dot-selected slates provide the clearest conservative improvement; additional MLP selection and candidate attention do not establish a reliable incremental benefit. Added scalar ranker features improve means but remain optional pending a retention decision. Existing default aliases remain unchanged. This supersedes speculative priorities below where those hypotheses have now been tested.

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

Keep the existing ranker/bias defaults. Larger-width training and edge/scorer experiments remain deferred. Establish a separately versioned rolling-demand Stage 1 control before claiming a temporally clean shared-vector comparison; do not mix that retraining into these frozen-retriever ablations. Input losslessness and working gradients alone did not produce a discovery improvement.

## 6. Completed E1–E5 campaign (2026-11-11–12)

See [E1_E5_EXPERIMENTS.md](E1_E5_EXPERIMENTS.md) for 32 fits and the architecture decision. Keep the original single-vector/32-event Transformer and 13-feature UPS-CANR shape. E1 and E2 showed small conditional fresh-control gains without evidence sufficient for promotion; E3 failed the screen and E4/E5 produced duplicate interests. The demand-disabled scratch control improved Stage 1 coverage but not reproducible downstream discovery. Default checkpoints and all historical/reference caches remain unchanged; no test evaluation was performed. Reproduction and cache-serving contracts are in [experiments/e1_e5/README.md](experiments/e1_e5/README.md).
