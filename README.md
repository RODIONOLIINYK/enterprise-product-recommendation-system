# Enterprise Product Recommendation System

Next-basket recommendations for B2B enterprise customers using **Stage 1 Two-Tower Retrieval** and **Stage 2 Candidate-Aware Neural Reranking (`UPS-CANR`)**.

## Current Status & Documentation Map

- **[AGENTS.md](AGENTS.md):** Agent handbook with the exact active reference architecture, module map, checkpoint/cache inventory, and strict temporal-leakage guardrails.
- **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md):** Verified validation metrics for the active Stage 1 and Stage 2 reference models, plus a summary of tested-and-rejected experimental hypotheses.
- **[plan.md](plan.md):** Current project goals, diagnostic findings, and the prioritized experimental roadmap for improving new-to-customer product discovery in Stage 2.

## Active Pipeline

```text
Raw sales + company profiles + product catalogue
  → Stage 1 Discovery Reference Two-Tower (128D vectors + 4 familiarity flags, rolling prior-sale pool)
  → 50-item candidate slate (~25 Personal Repeat + ~25 Unseen Stage 1 Discovery candidates)
  → Stage 2 UPS-CANR Ranker (33-token Customer Memory + Statistics-Biased Cross-Attention + 397D MLP)
  → Top-10 ranked products
```

- **Stage 1 (`recommender/models.py`, `recommender/stage1_catalogue.py`, `recommender/familiarity.py`):**
  - 128-dimensional `CustomerTower` and `ProductTower`.
  - `CustomerTower` encodes each historical purchase using its shared 128D product vector, normalized quantity, and normalized inter-purchase gap (`history_gap_reference_days = 74.0`), processed by a 2-layer Transformer with attention pooling, fused with 1024D `BAAI/bge-m3` company text and 7 normalized customer profile features.
  - `ProductTower` encodes 1024D `BAAI/bge-m3` product text (without subcategory), one-hot `container_type` (with explicit `none`), and 7 point-in-time product demand features (`remove_product_attribute_embeddings = True`).
  - Scoring adds 4 point-in-time binary familiarity flags (`brand`, `product_category`, `business_line`, `product_family`) with learned weights over a rolling candidate pool of merchandise with at least one prior paid sale and no active block.
- **Stage 2 (`recommender/ranker.py`, `recommender/train_ranker.py`):**
  - `UPSCANRRanker` with `statistics_attention = True` and `excluded_statistics = ("history_repeat",)`.
  - Builds a 33-token Customer Memory (Token 0 = Customer Profile, Tokens 1..32 = Historical Purchases contextualized by a 1-layer TransformerEncoder).
  - Scores 50 candidates per transition using 4-head cross-attention biased by candidate-to-memory bilinear statistics interactions ($14 \times 17$ per head) and a `397 -> 256 -> 128 -> 64 -> 1` MLP over `[z_candidate (128), h_attended (128), z_candidate * h_attended (128), pairwise_features (13)]`.

## Setup

Use Python 3.12 (`./venv/bin/python`).

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Data Preparation

Private raw inputs live locally under `data/raw/` (configured in `configs/datasets_links.toml`):

```text
data/raw/dataset.xlsx
data/raw/customers.csv
data/raw/items.csv
```

Run `notebooks/01_clean_purchases.ipynb` or use the Python preparation API to produce `data/interim/cleaned_purchases.csv`:

```bash
./venv/bin/python -m recommender
```

- **Paid-sale eligibility (`configs/product_scope.toml`):** Only positive finite `ПРОДАЖА` records with `Gen_ Prod_ Posting Group = ТОВАР` and item-master classification `ТОВАР` count as purchases. Gifts (`ПОДАРОК`), returns, and accounting rows are excluded.
- **Record merging (`recommender/merging.py`):** Same-company, same-calendar-date, same-`product_id` lines are merged and their quantities summed. Package variants remain distinct product IDs.
- **Chronological splits (`recommender/split.py`):**
  - **Train:** `< 2025-10-01` (`26,264` next-basket transitions; `26,184` with $\ge 50$ rolling candidates)
  - **Validation:** `2025-10-01` to `2026-02-18` (`6,191` next-basket transitions, `1,857` new-to-customer target instances)
  - **Test:** `>= 2026-02-19` (`6,327` transitions — **strictly held out**)

## Training & Evaluation Commands

Run from the repository root using `./venv/bin/python`:

```bash
# Train Stage 1 Discovery Reference Retriever (saves to checkpoints/retriever-discovery-reference/)
PYTHONPATH=. ./venv/bin/python -m recommender.train_retriever --discovery-reference

# Evaluate Stage 1 on rolling prior-sale discovery pool (saves to checkpoints/stage1-rolling-discovery/)
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate_stage1

# Train Stage 2 UPS-CANR Reference Ranker (saves to checkpoints/ranker/)
PYTHONPATH=. ./venv/bin/python -m recommender.train_ranker

# Run full Stage 1 + Stage 2 Validation Evaluation (saves to checkpoints/evaluation_report.json)
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate

# Run the unit and integration test suite
PYTHONPATH=. ./venv/bin/pytest tests/
```

## Data Handling

Raw data, generated tables, cached embeddings, precomputed slates, and model checkpoints remain local and are ignored by Git. Clear notebook outputs before committing.
