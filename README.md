<div align="center">

# Enterprise Product Recommendation System

### Privacy-first, point-in-time product ranking with CatBoost

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-FFCC00)](https://catboost.ai/)
[![Model](https://img.shields.io/badge/Task-Learning--to--Rank-7C3AED)](#why-learning-to-rank)
[![Privacy](https://img.shields.io/badge/Data-Local%20Only-059669)](#privacy-by-design)

Turn truthful customer-product history into a ranked list of products a customer is likely to purchase next.

</div>

---

## Overview

This repository contains a complete local recommendation baseline:

1. clean paid purchases and received products;
2. construct point-in-time customer-product history;
3. generate positive and negative ranking candidates;
4. keep every customer in exactly one data split;
5. train a configurable `CatBoostRanker`;
6. evaluate recommendation quality against random and purchase-history baselines.

The implementation is designed for a real business dataset without publishing the dataset, trained model, customer identifiers, product identifiers, or company-derived results.

## Why learning to rank?

A customer can purchase several products in one event. Predicting one class is therefore the wrong shape of problem.

For every customer scoring event, the pipeline creates a candidate group:

| Candidate | Meaning | Label |
|---|---|---:|
| Product purchased in this event | Positive example, including a first purchase | `1` |
| Product purchased earlier but not now | Truthful historical negative | `0` |
| Existing product never paid for by this customer | Sampled catalogue negative | `0` |

CatBoost receives the candidates and their historical features, then learns to put the positive candidates above the negative ones.

```mermaid
flowchart LR
    A["Private purchase data"] --> B["Clean and aggregate"]
    B --> C["Point-in-time history"]
    C --> D["Candidate groups"]
    D --> E["Customer-disjoint split"]
    E --> F["CatBoostRanker"]
    F --> G["Ranked products"]
    G --> H["Ranking metrics"]
```

## Core guarantees

- **Truthful history** — every historical feature uses only events strictly before the scoring event.
- **Same customer, same product** — product-history statistics are never copied from another customer or product.
- **First purchases are retained** — a positive product does not need earlier purchase history.
- **Realistic negatives** — previously purchased products can correctly have `label = 0` when they were not purchased in the current event.
- **No identity memorization** — `customer_id` and `group_id` organize the data but never enter the model.
- **No target leakage** — current-event quantities and dates are excluded from training features.
- **Unseen-customer evaluation** — all records for one customer stay in one split.

## Pipeline

### 1. Clean purchases

[`notebooks/01_clean_purchases.ipynb`](notebooks/01_clean_purchases.ipynb) converts the private source data into one customer-date-product record and keeps paid quantity separate from total received quantity.

### 2. Build ranking candidates

[`notebooks/02_build_historical_features.ipynb`](notebooks/02_build_historical_features.ipynb):

- sorts source events chronologically;
- creates one ranking group per customer purchase date;
- keeps all products actually paid for on that date as positives;
- samples products previously purchased by the same customer as hard negatives;
- samples a small number of real catalogue products never paid for by that customer;
- calculates every feature from the candidate customer’s real prior history;
- removes target-day quantities and purchase dates from the final training table.

The scoring date is written to separate group metadata for validation and auditing. It is not a model input.

### 3. Train and evaluate

[`scripts/train_catboost.py`](scripts/train_catboost.py):

- validates the training and metadata contracts;
- creates deterministic `70 / 15 / 15` customer-disjoint splits;
- trains a CatBoost YetiRank model with early stopping;
- compares it with random and purchase-history baselines;
- exports local metrics, feature importance, test predictions, and the model.

## Model contract

### Input

Each model row means:

> “Should this product rank highly for this customer at this scoring event, given only information available beforehand?”

| Feature family | Examples |
|---|---|
| Product identity | `product_id`, `product_category`, `business_line` |
| Purchase history | prior purchase count, prior paid quantity, last paid quantity |
| Receipt history | prior receipt count, prior received quantity, last received quantity |
| Recency | days since the last paid purchase or receipt |
| Frequency | average days between purchases of this exact customer-product pair |
| Affinity | previously purchased this category or business line |

`business_line` supplies a population-level signal for customers with little or no history. The historical business-line flag adds a personalized signal once prior purchases exist.

### Output

CatBoost outputs one numeric relevance score per candidate product. Products are sorted by that score inside the customer’s group:

```text
customer scoring group
├── Product A  score 2.41  → rank 1
├── Product C  score 1.76  → rank 2
└── Product B  score 0.93  → rank 3
```

The score is not a calibrated purchase probability. Its purpose is ordering candidates.

## Evaluation, without the jargon

Suppose the customer actually purchased `A` and `B`, while the model ranks:

```text
1. A  ✓
2. C  ✗
3. B  ✓
```

| Metric | Result | Question it answers |
|---|---:|---|
| HitRate@1 | `1.00` | Was at least one correct product first? |
| Precision@1 | `1.00` | Was the single first recommendation correct? |
| Recall@1 | `0.50` | How much of the two-product basket appeared in the first slot? |
| Precision@3 | `0.67` | What fraction of the first three recommendations was correct? |
| Recall@3 | `1.00` | How much of the purchased basket appeared in the first three slots? |
| MRR | `1.00` | How early did the first correct result appear? |
| NDCG@3 | `< 1.00` | Were all correct products placed as high as possible? |

The evaluator also reports:

- **catalogue coverage** — how much of the candidate catalogue is ever recommended;
- **first-purchase Recall@K** — recall for products the customer had never paid for before;
- **repeat-purchase Recall@K** — recall for purchased products with earlier paid history.

Metrics are always compared with simple baselines. A sophisticated model is useful only if it beats an understandable alternative.

## Quick start

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- the private source data in the expected local path

### Prepare the data

Run the notebooks in order:

```text
notebooks/01_clean_purchases.ipynb
notebooks/02_build_historical_features.ipynb
```

### Train

```bash
uv run --with-requirements requirements-training.txt \
  python scripts/train_catboost.py \
  --config configs/catboost_training.json
```

Hyperparameters, categorical features, excluded fields, split fractions, metric cutoffs, and output paths live in [`configs/catboost_training.json`](configs/catboost_training.json).

## Repository layout

```text
.
├── configs/
│   └── catboost_training.json       # Model, split, feature, and output settings
├── notebooks/
│   ├── 01_clean_purchases.ipynb     # Private-data cleaning
│   └── 02_build_historical_features.ipynb
├── scripts/
│   └── train_catboost.py            # Reproducible training and evaluation
├── requirements-training.txt        # Pinned training dependencies
└── README.md
```

Generated outputs remain local and ignored:

```text
data/
models/catboost_ranker.cbm
artifacts/catboost/metrics.json
artifacts/catboost/feature_importance.csv
artifacts/catboost/test_predictions.csv
```

## Privacy by design

The business dataset is confidential. This public repository intentionally excludes:

- raw and cleaned sales records;
- customer and product identifier values;
- product catalogues derived from company data;
- trained models and intermediate feature tables;
- company-derived evaluation results;
- executed notebook outputs.

Tracked notebooks contain source code only. Public tests and examples should use synthetic data.

## Current limitations

- Candidate groups are sampled and may be smaller than the production catalogue.
- Strong offline results on small candidate groups may overstate production performance.
- Repeat-purchase ranking should be evaluated separately from first-purchase discovery.
- A future production evaluation should use the exact candidate-generation policy used at inference time.

## Roadmap

- [x] Privacy-safe cleaning pipeline
- [x] Point-in-time customer-product features
- [x] Truthful positive and negative candidates
- [x] Customer-disjoint CatBoost ranking baseline
- [x] Ranking metrics and simple baselines
- [ ] Synthetic-data regression tests
- [ ] Production-scale candidate retrieval
- [ ] Probability calibration and serving interface
- [ ] Monitoring for drift, coverage, and recommendation quality

---

<div align="center">

Built as a transparent recommendation baseline: correct history first, model complexity second.

</div>
