<div align="center">

# Enterprise Product Recommendation System

### Point-in-time purchase probabilities and explainable product recommendations

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-FFCC00)](https://catboost.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Model](https://img.shields.io/badge/Model-CatBoostClassifier-7C3AED)](#model)
[![Evaluation](https://img.shields.io/badge/Split-Customer--Disjoint-0EA5E9)](#evaluation)
[![Privacy](https://img.shields.io/badge/Data-Local%20Only-059669)](#privacy-by-design)

An end-to-end recommendation pipeline that transforms private purchase history into ranked product probabilities using leakage-safe features and a production-oriented `CatBoostClassifier` serving path.

**Purchase probabilities · Point-in-time features · Product unification · First/repeat purchase evaluation**

</div>

---

## At a glance

| Capability | Implementation |
|---|---|
| Business problem | Recommend the products a customer is most likely to purchase next |
| Data preparation | Product cleaning, package-volume unification, receipt aggregation |
| Feature design | Strictly prior customer-product history, cadence, affinity, and demand |
| Model | `CatBoostClassifier` with direct purchase probabilities |
| Validation | Customer-disjoint train, validation, and test splits |
| Serving output | One classifier-sorted, product-level recommendation table |
| Privacy | Raw data and customer-specific outputs remain local |

## Why this project matters

Real recommendation systems require more than training a model. Historical features must be leakage-safe, negatives must represent products that were genuinely available but not purchased, first purchases must remain learnable, and inference must reproduce the same feature definitions used during training.

This repository implements that complete path:

```mermaid
flowchart LR
    A["Private purchase workbook"] --> B["Clean and unify products"]
    B --> C["Point-in-time history"]
    C --> D["Candidate generation"]
    D --> E["Customer-disjoint split"]
    E --> F["CatBoostClassifier"]
    F --> G["Purchase probabilities"]
    G --> H["Classifier-sorted products.csv"]
```

## Verified results

The current model uses 17 features and is evaluated on customers never seen during training. The test split contains 371 customers, 4,247 ranking groups, 106,175 candidate rows, and zero customer overlap with training or validation.

### Recommendation quality

| Test metric | Result |
|---|---:|
| MRR | **0.6363** |
| Hit Rate@1 | **47.96%** |
| Recall@5 | **77.64%** |
| Recall@10 | **93.07%** |
| NDCG@10 | **0.6909** |
| First-purchase Recall@10 | **81.04%** |
| Repeat-purchase Recall@10 | **96.04%** |
| Catalogue Coverage@10 | **86.03%** |

### Probability quality

| Test metric | Result |
|---|---:|
| ROC AUC | **0.9017** |
| Log loss | **0.1563** |
| Brier score | **0.0447** |
| Expected calibration error | **0.0019** |
| Mean predicted probability | 6.43% |
| Observed positive rate | 6.31% |

The probabilities are conditional on the candidate-generation policy used during training. They should not be interpreted as unconditional probabilities across the entire product catalogue.

## Model

Every candidate row is a binary decision:

> Given only information available before the scoring date, will this customer purchase this product in the current event?

The classifier:

- optimizes binary `Logloss`;
- handles product, category, and business-line identifiers as categorical features;
- returns `predict_proba()[:, 1]` directly;
- sorts recommendations by purchase probability;
- uses early-stopping metadata from a separate validation customer set;
- is evaluated with both probability and recommendation-ranking metrics.

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("models/catboost_classifier.cbm")
probabilities = model.predict_proba(candidate_features)[:, 1]
```

## Serving output

The usage pipeline writes one local product table:

```text
deployment_pipeline/data/products.csv
```

Each row represents one unique product. Product identity and historical features come first, `expected_days_before_next_order` is the final feature column, and all scoring fields follow it:

```text
product identity
→ customer-product history
→ category and business-line affinity
→ catalogue demand
→ expected_days_before_next_order
→ historical_score
→ probability
```

Rows are sorted by `probability` descending and displayed as percentages. Probabilities are calculated on the same row that supplies the product ID and product name, preventing positional joins or product-probability misalignment.

## Feature engineering

| Feature family | Examples |
|---|---|
| Product context | `product_id`, `product_category`, `business_line` |
| Purchase depth | prior purchase count, cumulative quantity, last quantity |
| Recency | days since the last paid purchase |
| Reorder cadence | average and variability of prior purchase intervals |
| Replenishment | expected days before the next order |
| Customer affinity | prior category and business-line counts and shares |
| Product demand | lifetime purchases, unique customers, recent 30-day purchases |

### Leakage controls

- Historical features use only information available before the modeled outcome.
- Customer and group identifiers organize examples but never enter the model.
- Current-event quantities and outcomes do not enter the feature set.
- First purchases remain valid positive labels.
- Previously purchased products can be truthful negatives when absent from the current event.
- Customers never overlap across train, validation, and test.

### Product unification

Package variants such as `1 L`, `2 L`, `500 ml`, and Cyrillic unit equivalents are grouped by normalized base name. The smallest package becomes canonical, larger packages are converted into canonical-unit quantities, and product identity is remapped consistently in both training and usage pipelines.

## Repository structure

```text
.
├── configs/
│   └── catboost_training.json
├── deployment_pipeline/
│   └── usage_pipeline.py
├── notebooks/
│   ├── 01_clean_purchases.ipynb
│   └── 02_build_historical_features.ipynb
├── scripts/
│   └── train_catboost.py
├── models/
│   └── catboost_classifier.cbm
├── artifacts/
│   └── catboost_classifier/
│       ├── metrics.json
│       └── feature_importance.csv
├── requirements.txt
└── README.md
```

## Run locally

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- The private source workbook in `data/raw/`

### 1. Prepare the data

Run the notebooks in order:

```text
notebooks/01_clean_purchases.ipynb
notebooks/02_build_historical_features.ipynb
```

### 2. Train the classifier

```bash
uv run --with-requirements requirements.txt \
  python scripts/train_catboost.py \
  --config configs/catboost_training.json
```

### 3. Generate recommendations

```bash
uv run --with-requirements requirements.txt \
  python deployment_pipeline/usage_pipeline.py [customer_id]
```

When `customer_id` is omitted, the script chooses a known customer. The customer-specific recommendation table is saved locally to `deployment_pipeline/data/products.csv`.

## Evaluation

Recommendation quality is measured at K = 1, 3, 5, and 10 using:

- **Hit Rate** — whether at least one relevant product appears in the top K;
- **Precision** — how much of the recommendation list is relevant;
- **Recall** — how much of the purchased basket is recovered;
- **MRR** — how early the first relevant product appears;
- **NDCG** — whether relevant products are ordered near the top;
- **Catalogue coverage** — how broadly the model recommends across products;
- **First/repeat recall** — discovery and replenishment performance separately.

Probability quality is evaluated with log loss, Brier score, ROC AUC, and expected calibration error.

## Privacy by design

The following stay local and are excluded from publication:

- raw and cleaned purchase records;
- customer and product-level training tables;
- private workbooks;
- generated customer recommendation tables;
- executed notebook outputs.

Only explicitly reviewed code, the classifier model, aggregate metrics, and aggregate feature importance are published. A trained model still encodes patterns learned from confidential data and should receive the same review before every release.

## Limitations and next steps

- Candidate groups are smaller than the full production catalogue.
- Offline quality depends on production candidate retrieval matching evaluation conditions.
- Probabilities are conditional on sampled candidates.
- Current serving inspection does not contain future purchase outcomes.
- Production monitoring for drift, coverage, probability quality, and realized conversion remains future work.

## Resume-ready summary

> Built an end-to-end enterprise product recommendation system using leakage-safe point-in-time features, customer-disjoint evaluation, CatBoost classification, direct probability scoring, package-variant unification, and a shared training/inference feature contract; achieved 93.1% Recall@10 and 0.902 ROC AUC on held-out customers.

<div align="center">

---

**A trustworthy recommendation system keeps its history, candidates, evaluation, and serving definitions aligned.**

</div>
