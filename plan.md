# Current Project Goals & Experimental Roadmap (`plan.md`)

This document tracks the current project state, root-cause findings from our latest architectural review, and the concrete experimental roadmap for agents working on the **Enterprise Product Recommendation System**.

---

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

Before designing any new Stage 2 experiment, agents must understand the five structural findings identified in our review of [`recommender/ranker.py`](recommender/ranker.py):

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
5. **Candidate Pairwise Features Lack Explicit Co-Purchase / Transition Signals for Unseen SKUs:**
   - For an unseen candidate SKU (`is_repeat_sku = 0`), its SKU-specific repeat features (`user_sku_order_count_norm`, `user_sku_qty_share`, `days_since_last_sku_buy_norm`, `sku_urgency`) are all constant (`0.0` or `1.0`).
   - To rank which of the 25 unseen candidates in the slate a customer will actually buy first, the final scoring MLP needs **new predictive information** about sequential product transitions (e.g., *customers who bought historical SKU $A$ frequently buy candidate SKU $B$ next*) and explicit category/brand/family familiarity signals in the scoring head.

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
  2. **Explicit 4 familiarity flags in candidate pairwise features:** Pass the 4 binary familiarity flags (`brand`, `category`, `business_line`, `product_family`) directly into the candidate pairwise features so the Stage 2 scoring MLP sees the exact same family/brand/category familiarity signals that helped Stage 1.

---

## 4. Workflow Rules for Agents

1. **Always benchmark against the clean reference:** Every experiment must report:
   - Validation New-to-Customer Hits@10 and Recall@10 (`out of 1,857`)
   - Validation Overall Recall@10 and NDCG@10 at the **same checkpoint** (as well as peak NDCG@10 if different)
2. **Never leave dead experimental flags or unused modules in the repo:** If an experiment fails to beat the reference, log its numbers in Section 4 of [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) and revert the code changes immediately.
3. **Never touch the Test split (`>= 2026-02-19`)** until the user explicitly signs off on the final validation model.
