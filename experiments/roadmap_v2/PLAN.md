# Master Improvement Plan — Strictly Causal Rolling Pipeline, `base_name` Statistics, Augmentation & Commercial Signals (`roadmap_v2`)

> **Scope & Non-Negotiable Rules**
> - **Chronological split:** Train `< 2025-10-01` (`26,253` queries), Validation `2025-10-01..2026-02-18` (`6,191` queries, `10,276` total targets, `1,857` new targets). **Held-out Test (`>= 2026-02-19`) is strictly untouched** until final sign-off.
> - **Zero data leakage:** Every tower input, rolling statistic, co-occurrence/lift counter, payment settlement (`ДатаПолнойОплати < t`), manager rolling window, and warehouse shipment signal is computed strictly `< as_of_date` (`t`). During training, 60-day auxiliary labels are additionally capped strictly `< 2025-10-01` (`TRAIN_CUTOFF`) so no validation purchases leak into training targets.
> - **Frozen references preserved:** All existing checkpoints and caches (`repeat-no-month-v1`, `pointwise425_ndcg` = `R@10 77.10% / NDCG@10 0.6022 / 255 new hits`, `checkpoints/ranker/*`, `semantic_oracle_v1`) remain untouched. Every stage writes to its own versioned namespace and is evaluated **one controlled change at a time** with paired customer-clustered bootstrap 95% CIs on full validation.
> - **Explicitly Excluded (per user domain decisions):** `Компания`, `ПОДАРОК` / `ВОЗВРАТ` / `АКЦИЯ` / `БОНУС КЛН`, fixed per-manager ID vectors/embeddings, repeat-vs-discovery ensemble calibration, and already-disproved Stage 2 architecture tweaks (wider/deeper MLPs/Transformers, 64-event memory, candidate-set attention, Stage 1 score injection).

---

## Stage 0 — Leak-Free Rolling Baseline & Stage 1 Native Rolling Retrain

**Goal:** Guarantee that **both** Stage 1 and Stage 2 share one unified, strictly point-in-time `rolling_observed_v2` foundation with zero frozen-cutoff bias or timestamp mismatch.

1. **Point-in-time tower & rolling audit:**
   - Verify that during training (`26,253` timestamps) and validation (`6,191` timestamps), `CustomerTower` computes each transition's cumulative stats (`distinct_orders`, `total_quantity`, `distinct_products`, `distinct_categories`, `median_order_gap`, `median_purchase_quantity`) and 32-event sequence strictly from `< t`, while `ProductTower` uses `demand = disabled_v1` (zero future demand leakage into $\mathbf{v}_p$).
   - Verify that all 41 rolling statistics roll strictly `< t` across both train and validation (never frozen at `2025-10-01`).
   - Document exact compute cost for full point-in-time training vs. purchase-triggered incremental inference (`PurchaseEmbeddingCache`).
2. **Retrain Stage 1 MLP (`repeat-no-month-v1` recipe) natively on `rolling_observed_v2`:**
   - Currently Stage 2 (`pointwise425_ndcg`) was trained on `rolling_observed_v2`, while Stage 1's `best_selector.pt` was still trained on legacy `frozen_train_cutoff_v1`.
   - Train a clean `rolling_observed_v2` Stage 1 control (`S1-R0`), rebuild matched `rolling100` slates (`quota100_50`), and train a matched Stage 2 control (`S2-R0`).
   - **Exit criterion:** Verified zero-leakage unit tests + clean paired baseline metrics for both `S1-R0` and `S2-R0`.

---

## Stage 1 — `base_name` Product-Body Statistics (Controlled 3-Way Ablation)

**Verified data fact:** `100%` (`841 / 841`) of the active sellable product universe already has a clean `base_name` in `items.csv`, forming **`585` unique product bodies** with **`193` multi-pack groups covering `449 / 841` (`53.4%`) of all active SKUs** (e.g., `SHELL Rimula R4 L 15W-40` cleanly unifies `209л` drum `ТОВ-У001141` and `20л` pail `ТОВ-У001142`; `SHELL Helix Ultra 5W-40` unifies `1л`, `4л`, `5л`, `209л`).

**New strictly-causal `< t` `base_name`-level statistics:**
- **Customer × `base_name` history (5):** `has_bought_base_sku`, `user_base_sku_order_count_norm`, `user_base_sku_qty_share`, `days_since_last_base_sku_buy_norm`, `base_sku_urgency`.
- **Global & industry `base_name` rolling demand (7):** `base_name_global_pop_norm`, `base_name_industry_affinity`, `base_name_orders_7d`, `base_name_orders_30d`, `base_name_orders_90d`, `base_name_buyers_30d`, `base_name_growth_30d`.
- **Cross-company `base_name` transitions (3):** `base_name_last_lift`, `base_name_recent3_lift`, `base_name_same_basket_lift`.

**Exact 3-arm controlled experiment (run on BOTH Stage 1 MLP and Stage 2 Transformer):**
1. **Arm A — `sku_only` (Control):** Exact `product_id` statistics only (no `base_name` stats).
2. **Arm B — `sku + base_name`:** Exact `product_id` statistics **plus** all 15 `base_name` statistics.
3. **Arm C — `base_name_only` (Exact-SKU-Excluded Ablation):** Remove exact `product_id`-level repeat/popularity/demand/lift features (`is_repeat_sku`, `user_sku_*`, `days_since_last_sku_buy_norm`, `sku_urgency`, `global_sku_popularity_norm`, `industry_sku_affinity`, `sku_orders_*`, `sku_buyers_30d`, `sku_growth_30d`, `sku_last_lift`, `sku_recent3_lift`, `same_basket_lift`), keeping **only** the `base_name`-level equivalents (plus packaging/container & category/brand/family features).

- **Decision rule:** Compare Arms A, B, and C on Recall@5, Recall@10, Recall@50/100 (Stage 1), NDCG@10, New@10, and Repeat@10. Carry forward the winning feature set into Stages 2–4.

---

## Stage 2 — Training Augmentation & Multi-Task 60-Day Auxiliary Head (Stage 1 MLP & Stage 2 Transformer)

**Goal:** Directly address discovery data scarcity (`~11:1` repeat-to-new target imbalance; `448 / 841` SKUs have `<= 10` training positives) without distorting inference.

1. **Stage 2A — Multi-Task Next-60-Day Auxiliary Head (Training-Only, Deleted at Inference):**
   - **Label construction (strictly leak-free):** For every training transition `(customer_id, t)` with `t < TRAIN_CUTOFF`, define auxiliary binary target $y_{60\text{d}}(p) = 1$ if customer buys product $p$ (or `base_name`, matching Stage 1 winner) in **any** paid order in `[t, min(t + 60d, TRAIN_CUTOFF))`.
   - **Architecture:** Add a lightweight second head `Linear(hidden_dim -> 1)` branching off the shared penultimate representation of:
     - **Stage 2 `UPSCANRRanker` scoring trunk** (`64 -> 1` next-basket main head + `64 -> 1` 60-day auxiliary head)
     - **Stage 1 MLP selector trunk** (penultimate hidden layer → main head + 60-day auxiliary head)
   - **Loss:** $\mathcal{L} = \mathcal{L}_{\text{next}} + \lambda \cdot \mathcal{L}_{60\text{d}}$ (sweep $\lambda \in \{0.2, 0.4\}$).
   - **Inference:** Discard the auxiliary head completely; score strictly with Head 1 (zero latency overhead).

2. **Stage 2B — History-SKU Dropout Augmentation (Training-Only):**
   - During training epochs only, with probability $p_{\text{drop}}$ (e.g., `0.25–0.35` per multi-item/repeat transition), randomly select 1–2 previously purchased SKUs (or `base_name`s) that appear as positives in the current basket and **mask them out of the customer's input history**:
     - Zero/mask matching tokens in the 32-event sequence `attention_mask`.
     - Reset that candidate's personal repeat/order-count/qty-share/urgency features to exact unseen defaults (`is_repeat=0`, `days_since=1.0`, `count=0`, `qty_share=0`, `urgency=0`), and optionally adjust its `base_name` repeat features coherently.
   - Effect: Converts real repeat purchases into **synthetic first-time discovery examples** with authentic customer context, multiplying discovery gradients several-fold without needing synthetic data generation.

3. **Stage 2C — Combined (`60-Day Auxiliary Head + History-SKU Dropout`):**
   - Test both together if each shows non-negative / positive signal individually.

---

## Stage 3 — New Causal Commercial, Financial, Rolling-Manager, Warehouse & Parsed Features

All features extracted strictly `< t` from existing raw files (`data/raw/dataset.xlsx`, `data/raw/items.csv`, `data/raw/customers.csv`) — **one feature group added and ablated at a time** so noisy groups are immediately rejected.

### 3A. Payment Deferrals (`отсрочки по платежам`) & Customer Financial Health
- **Sources:** `УсловияОплаты` (98.2% filled: `ПРЕДОПЛАТА`/`ФАКТ`/`НАЛОЖКА` = `0d`, `ДН2`..`ДН90` = `2..90d`), `ТипОплаты` (`БЕЗНАЛ` vs `НАЛ`), `БонусДокумента`, and strictly causal `ДатаПолнойОплати < t`.
- **Customer financial-state features (strictly `< t`):**
  - `is_prepaid_or_cash_buyer`: fraction of customer's recent orders with `0d` deferral (`ПРЕДОПЛАТА` / `ФАКТ` / `НАЛ`) — signals immediate payment capability / strong liquidity.
  - `avg_contractual_deferral_days_norm` & `last_contractual_deferral_days_norm`: normalized typical & latest payment deferral terms (`0..90d`).
  - `actual_vs_contract_pay_ratio`: from strictly settled prior invoices (`ДатаПолнойОплати < t`), median ratio of actual days-to-pay vs. contractual terms (early/on-time vs. chronic late payer).
  - `has_unpaid_overdue_invoice`: binary/count indicator as of date `t` of prior invoices where `purchase_date + deferral_days < t` AND (`ДатаПолнойОплати` is null or `>= t`) — directly signals cashflow stress or credit hold.
  - `customer_bonus_rate`: rolling ratio of `БонусДокумента` to order revenue `< t`.

### 3B. Revenue, Cost, Margin & Pricing (`Оборот`, `Себестоимость`, `СебестоимостьУЕ`)
- **Candidate & customer-fit features (strictly `< t`):**
  - `sku_unit_price_norm` (median `Оборот / Количество` over last 90d `< t`, fallback lifetime `< t`) and `sku_unit_cost_usd_norm` (`СебестоимостьУЕ / Количество`).
  - `sku_margin_ratio`: `(Оборот - Себестоимость) / max(Оборот, eps)`.
  - `sku_price_change_90d`: recent 30d vs. prior 60–90d unit price change.
  - `customer_recent_spend_90d_norm` & `customer_avg_order_revenue_norm`.
  - `sku_price_vs_customer_avg_unit_price`: log-ratio comparing candidate's price tier to the customer's historical average price per liter/kg (budget vs. premium synthetic fit).

### 3C. Rolling Current-Manager Behavioral Features (`МенеджерКод` — Zero Fixed Manager Vectors)
- **Design for frequent manager rotation:** Never learn a static embedding for `МенеджерКод`. Instead, look up *whoever is the customer's most recent manager strictly before `t`*, and compute rolling behavioral statistics from **that manager's paid sales in `[t - 90d, t)`**:
  - `current_manager_sku_share_90d` & `current_manager_base_sku_share_90d`
  - `current_manager_brand_share_90d` & `current_manager_cat_share_90d`
  - `manager_changed_recently` (`1` if customer's last order had a different manager than the order before it).

### 3D. Regional Warehouse Availability (`Склад`)
- Look up customer's primary shipping warehouse (`Склад`) from orders `< t`, and compute:
  - `sku_shipped_from_customer_warehouse_30d` & `_90d`
  - `base_sku_shipped_from_customer_warehouse_30d`
  - Acts as a causal regional stockout / assortment-availability signal.

### 3E. Offline Parsed Product Specs & Company Equipment Tags
- Run a one-time offline extraction (regex + LLM verification) over the `841` active SKUs and `~3,500` companies:
  - **Product specs:** `viscosity_grade` (`15W-40`, `5W-30`, `VG 46`, `VG 220`, `NLGI 2`), `pack_liters_kg`, `application` (`diesel`, `hydraulic`, `gear`, `compressor`, `food_grade`, `coolant`, `grease`).
  - **Customer equipment flags** (from `COMPANY_DESC`): `trucks`, `agro_tractors`, `compressors`, `hydraulics`, `food_grade`, `metalworking`, `quarry`.
  - **Derived causal pairwise features:** `same_viscosity_grade`, `matches_customer_dominant_pack`, `application_matches_company_equipment`.

---

## Stage 4 — Integrated Retrain, Slate Rebuild & Multi-Seed Verification

1. **Combine only the individually validated winners** from Stages 1, 2, and 3.
2. **Stage 1 full retrain:** Retrain Stage 1 (point-in-time Two-Tower if warranted + Rolling MLP with winning features + 60d auxiliary head + History-SKU Dropout), then rebuild sealed causal `rolling100` candidate slates.
3. **Stage 2 full retrain:** Train Stage 2 Transformer (`UPSCANRRanker`) on the upgraded slates with winning features + 60d auxiliary head + History-SKU Dropout.
4. **Statistical validation:** Run **3 seeds (`42, 43, 44`)** on full `6,191` validation baskets, compute paired customer-clustered bootstrap 95% CIs vs. `pointwise425_ndcg` (`R@10 77.10%`, `NDCG@10 0.6022`, `255` new hits), verify zero regression in serving latency, and record a promotion recommendation (leaving held-out Test untouched until explicit user authorization).

---

## Stage 5 — Full `base_name` Canonical Product Entity Across All Stages (Mandatory Architectural Test & Head-to-Head Decision)

**Core idea:** Instead of modeling every packaging variant (`product_id`, e.g., `209L` drum vs. `20L` pail vs. `4L` jug of the same lubricant) as a separate product entity, promote **`base_name`** (`585` unique active product bodies vs. `841` active SKUs) to be the **canonical product unit across every stage of the pipeline**, combined with the best-performing augmentations and commercial features discovered in Stages 2–3 (specifically optimizing for **truly new `base_name` discovery**):
1. **Product representation & embeddings:** Build one canonical `ProductTower` embedding, one text embedding, and one product ID embedding per `base_name` (packaging/container size no longer splits the same chemical formulation into separate embeddings or fragmented IDs).
2. **Purchase history sequences:** Index customer history tokens in both Stage 1 (`CustomerTower`) and Stage 2 (`UPSCANRRanker`) by `base_name` (summing same-day quantities across pack variants of the same `base_name` into unified product-body events).
3. **Candidate pool, slates & ranking:** Run rolling eligibility, Stage 1 retrieval, candidate slate construction, and Stage 2 ranking directly over `base_name` candidates (eliminating near-duplicate packaging collisions that currently consume multiple top-10 / top-100 slots).
4. **Head-to-head evaluation & architecture selection:**
   - Evaluate BOTH the best SKU-level (`sku_plus_base_name`) pipeline (deduplicated to `base_name` at top-K for apples-to-apples comparison) AND the native `base_name`-canonical pipeline on:
     - **NDCG@10**
     - **Truly new `base_name` discovery (`true_new_bn@10`, NOT same-`base_name` pack-size switches!)**
     - **Overall Recall@10** (and Recall@5 / Recall@50–100)
   - Also measure optional lightweight **pack-size resolution** (customer's dominant/last container type for that `base_name` or category, fallback to rolling popular pack) when concrete SKU output is needed.
   - **Carry forward whichever approach consistently wins on NDCG@10 + truly new `base_name` discovery + Recall@10.**

---

## Verification & Success Criteria
- Every new feature builder and augmentation gets deterministic unit tests in `tests/` verifying:
  1. Strict `purchase_date < t` and `ДатаПолнойОплати < t` causality (zero same-day or post-cutoff leakage).
  2. Strict `< TRAIN_CUTOFF` truncation for 60-day auxiliary training targets.
  3. Zero mutation of `manifest["preserved_hashes"]` and zero access to held-out Test (`>= 2026-02-19`).
- **North-Star Selection Criterion:** Every stage and the final architecture choice are judged on **consistently higher NDCG@10, higher truly new `base_name` discovery (excluding same-`base_name` pack switches), and strong overall Recall@10** on matched validation controls.
