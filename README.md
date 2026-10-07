# Enterprise Product Recommendation System

Preparation for next-basket recommendations using **two-tower retrieval and a hybrid cross-encoder ranker**.

## Current status

Implemented: workbook loading with header detection, positive merchandise-sale filtering, exact product-ID aggregation and validation, and an executable cleaning notebook. These reuse the useful preparation logic from the previous code.

Upcoming: metadata joins, point-in-time feature bundles, text embeddings, tower training, cross-encoder training, evaluation and serving. This branch currently has no trained model or inference command.

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

Run `notebooks/01_clean_purchases.ipynb` top to bottom in your environment's kernel. Only the workbook is consumed by the current cleaning step; the customer/item joins are upcoming.

The notebook writes `data/interim/cleaned_purchases.csv`, one row per company, calendar date and **exact product ID**. Package variants remain separate. Quantities retain their source units and duplicate event quantities are summed without package conversion.

The existing merchandise scope is retained: positive `ПРОДАЖА` transactions, item type `ТОВАР`, and product IDs beginning with `ТОВ`. Gifts, returns, accounting rows, invalid dates, incomplete required fields and non-finite quantities are excluded.

## Validation

```bash
python -m unittest discover -s tests -v
```

Synthetic tests check header detection, filtering, package identity, event uniqueness and quantity preservation. The notebook validates real purchases and their CSV round trip.

## Repository layout

```text
recommender/   Shared purchase preparation helpers
notebooks/    Cleaning notebook, without published execution outputs
tests/        Synthetic data preparation tests
plan.md       Sequential feature, training, evaluation and serving plan
data/         Ignored private source and generated tables
private/      Ignored backups of pre-cleanup notebooks
```

## Data handling

Raw data, generated tables, backups, embeddings, checkpoints, model weights and recommendations remain local and are ignored by Git. Clear notebook outputs before committing because they can contain company-specific records. The current cleaning notebook prints aggregate counts only.
