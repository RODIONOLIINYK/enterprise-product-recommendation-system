# Enterprise Product Recommendation System

Preparation for next-basket recommendations using **two-tower retrieval and a hybrid cross-encoder ranker**.

## Current status

Implemented: workbook loading with header detection, actual paid-sale filtering, item classification loading, shared historical candidate eligibility, exact product-ID aggregation and validation, and an executable cleaning notebook. These reuse the useful preparation logic from the previous code.

Upcoming: company enrichment, point-in-time feature bundles, text embeddings, tower training, cross-encoder training, evaluation and serving. This branch currently has no trained model or inference command.

Read [plan.md](plan.md) for the complete sequential implementation and training plan, shared feature contract, defaults and validation criteria.

## Intended pipeline

```text
Raw sales + company profiles + product catalogue
  → shared point-in-time company and product features
  → company tower + product tower
  → retrieve the top 100 eligible products
  → hybrid cross-encoder using the same complete feature bundles
  → top 10 products with relevance scores
```

The ranker jointly encodes company/product text, separately processes the identical purchase sequence, and combines every categorical and numeric input in its scoring head.

## Setup

Use Python 3.12. Retain the existing environment for purchase preparation and create a separate environment for future neural training.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependency versions will be locked when the training environment is verified. The requirements file currently lists direct dependencies only. Planned local training uses PyTorch MPS on a supported Apple Silicon Mac, with CPU fallback.

## Prepare purchase history

Keep the private inputs locally:

```text
data/raw/dataset.xlsx
data/raw/customers.csv
data/raw/items.csv
```

Run `notebooks/01_clean_purchases.ipynb` top to bottom in your environment's kernel. The cleaning step reads the sales workbook and item export. Company-profile enrichment is upcoming.

The notebook writes `data/interim/cleaned_purchases.csv`, one row per company, calendar date and **exact product ID**. Package variants remain separate. Quantities retain their source units and duplicate event quantities are summed without package conversion.

Actual purchases require a positive finite `ПРОДАЖА` sales record, valid company/product IDs and a date, and `Gen_ Prod_ Posting Group = ТОВАР`. The exact product ID must also be classified as for sale (`ТОВАР`) in column 51 of `items.csv`. Product-ID prefixes, brands and category names do not establish eligibility.

Training and inference share `configs/product_scope.toml`. At a scoring time, candidates must additionally have at least one genuine purchase strictly before the cutoff. A gift-only or never-sold product cannot be retrieved, ranked or returned. Gifts (`ПОДАРОК`), returns and accounting movements never count as purchase evidence. Gift-only dates create no target basket; mixed dates keep paid products only. The prepared history retains sale classification columns to enforce this rule.

```python
from recommender.catalogue import load_eligible_products

# history is the actual-purchase table produced by the cleaning notebook.
candidates = load_eligible_products("data/raw/items.csv", history, "2026-07-18")
```

Historical availability and stock are separate from the for-sale accounting classification. Newly sold products become candidates only at a later cutoff; their first sale must not leak into earlier training catalogues.

## Validation

```bash
python -m unittest discover -s tests -v
```

Synthetic tests check header detection, actual-sale filtering, gift-only/mixed dates, for-sale and prior-sale eligibility, future-sale invariance, package identity, event uniqueness and quantity preservation. The notebook validates real purchases and their CSV round trip.

## Repository layout

```text
recommender/   Shared purchase preparation and catalogue eligibility
configs/       Business eligibility policy
notebooks/    Cleaning notebook, without published execution outputs
tests/        Synthetic data preparation tests
plan.md       Sequential feature, training, evaluation and serving plan
data/         Ignored private source and generated tables
private/      Ignored backups of pre-cleanup notebooks
```

## Data handling

Raw data, generated tables, backups, embeddings, checkpoints, model weights and recommendations remain local and are ignored by Git. Clear notebook outputs before committing because they can contain company-specific records. The current cleaning notebook prints aggregate counts only.
