# Enterprise Product Recommendation System

Next-basket recommendations using **two-tower retrieval and candidate-aware reranking**.

## Current status

The code prepares paid purchase history, trains both recommendation stages, and evaluates
them against popularity and personal-repeat baselines. Checkpoints stay local. The
[evaluation report](EVALUATION_RESULTS.md) records an earlier architecture, and
[plan.md](plan.md) retains the original design notes.

## Current pipeline

```text
Raw sales + company profiles + product catalogue
  → product vectors and company history/profile vector
  → retrieve the top 50 eligible products
  → candidate-aware neural reranker
  → top 10 products with relevance scores
```

Stage 1 represents each historical purchase using its shared product vector, scaled
quantity, and scaled gap to the latest observed purchase. New product vectors use
product text, brand, category, business line, and demand features. Stage 2 uses
candidate-specific attention and pairwise features to reorder the retrieved list.
Its training and validation slates use the same retrieval policy.

## Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training uses PyTorch on MPS, CUDA, or CPU, depending on availability.

## Prepare purchase history

Keep the private inputs locally:

```text
data/raw/dataset.xlsx
data/raw/customers.csv
data/raw/items.csv
```

Run `notebooks/01_clean_purchases.ipynb` top to bottom in your environment's kernel. The cleaning step reads the sales workbook and item export. `recommender.company` loads and normalizes company metadata.

Dataset paths and preparation order are configured in `configs/datasets_links.toml`:

```toml
companies = "../data/raw/customers.csv"
purchases = "../data/raw/dataset.xlsx"
items = "../data/raw/items.csv"

[pipeline]
order = ["companies", "items", "purchases"]
```

Paths are relative to the TOML file's directory, so they work regardless of the working directory. Absolute paths are also supported. Company exports use zero-based columns 1, 2, 82, 88 and 90 for customer ID, company name, legal name, address and description respectively. Empty strings and `NULL` become missing values; missing customer IDs raise an error. Duplicate company and product metadata are merged by exact ID when their fields agree or fill missing values. Conflicting fields raise an error rather than choosing a value silently.

The company loader preserves `address` and derives a nullable `city` column with `process_address()`. Explicit Ukrainian/Russian locality markers (`м.`, `г.`, `місто`, `город`, `смт.`, `пгт.`, `с.`, etc.) take precedence; prefixes and trailing street/number/administrative details are removed. Villages and settlements also populate this locality field. Without a marker, the rule excludes streets, house numbers, regions, districts, countries and account-like strings, then selects the plausible name nearest before a region/district or requires a single remaining name. Conflicting marked localities and ambiguous unmarked addresses remain missing. Source spelling is retained, so Ukrainian/Russian aliases and historical city names are not merged. These are deterministic parsing rules, not a verified locality registry.

For Customer Tower training, `load_companies()` keeps companies with **at least one valid paid purchase** by default. It matches exact `customer_id` values to purchase history and adds `purchase_count`, counting distinct company/calendar-date/product events. Companies with no paid purchases, including gift-only companies, are excluded. When `purchases` is omitted, the loader prepares the configured sales workbook using the configured item catalogue. Pass history already produced by `load_purchases()` to reuse it, or set `minimum_purchases=0` to retain all company profiles with their counts.

All loaders share `recommender/merging.py`. `merge_purchase_events()` combines matching customer/calendar-date/exact-product lines and **sums every quantity**, including identical repeated lines. Purchase counting and historical product eligibility use the same merged events, so repeated source lines count as one event while their quantities survive. Different products, package variants, dates and customers remain separate. Optional purchase name/category/business-line variants are retained as JSON-array strings when they differ; classification and other metadata must agree. Already merged history can be passed through the function again without changing its totals. `aggregate_purchases()` remains available to the notebook and delegates to this shared implementation.

The reusable helpers can also be called directly:

```python
from recommender.merging import merge_purchase_events, merge_records

history = merge_purchase_events(paid_purchase_lines)
products = merge_records(product_records, keys=["product_id"])
# For another table, explicitly name the additive amounts:
merged = merge_records(records, keys=["customer_id", "product_id"], sum_columns=["quantity"])
```

`merge_records()` preserves all input columns. Only named `sum_columns` are added; attributes such as unit, price and classification are coalesced when consistent. Use `combine_columns=["description"]` to explicitly retain differing text values as JSON arrays. A merged quantity is never obtained by dropping duplicate rows or taking the first quantity.

Load company metadata without passing a dataset or file path:

```python
from recommender import load_companies, run_preparation

companies = load_companies()
tables = run_preparation()  # Runs each configured stage from left to right.
history = tables["purchases"]
companies = load_companies(purchases=history)  # Reuse prepared history.

# Override the order or run only selected stages.
tables = run_preparation(order=["purchases", "companies"])
# Use another config: load_companies(config_path="configs/local_datasets.toml")
```

`__init__.py` exposes these functions when you import `recommender`. Keep execution in an explicit function rather than calling the loaders during import: Python executes `__init__.py` once on the first import, which is unsuitable for controlling repeated preparation runs. Change `[pipeline].order` to control execution, or pass `order=` as above. Purchases read the item export internally to validate sales even when `items` is omitted from the order. Companies also need purchase history for the training filter; placing `purchases` before `companies` lets the runner reuse that table instead of reading the sales workbook again. The returned `items` table is the parsed master catalogue; use `load_eligible_products` below for scoring-time candidates.

You can also execute the configured stages from the project root:

```bash
python -m recommender
python -m recommender --order purchases companies
python -m recommender --config configs/datasets_links.toml
```

To load only company metadata, run `python -m recommender --order companies` from the project root. Direct execution with `python recommender/company.py` also works and prints the company row count; an absolute script path works from any directory.

These commands print row counts and return no output files. In Python, `run_preparation()` returns the prepared DataFrames; the notebook still writes cleaned purchase history as described below. Dataset configuration is separate from the business eligibility policy in `configs/product_scope.toml`.

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

After preparing purchase history, train and evaluate from the project root:

```bash
python -m recommender.train_retriever
python -m recommender.train_ranker
python -m recommender.evaluate
```

Evaluation defaults to the validation split. Run the test suite with:

```bash
python -m pytest tests/
```

Synthetic tests check shared record merging, metadata preservation/conflicts, repeated-merge stability, company normalization and IDs, dataset path resolution and stage order, header detection, actual-sale filtering, gift-only/mixed dates, for-sale and prior-sale eligibility, future-sale invariance, package identity, event uniqueness and quantity preservation. The notebook validates real purchases and their CSV round trip.

## Repository layout

```text
recommender/   Data preparation, retrieval, reranking, and evaluation
configs/       Dataset paths, preparation order and business eligibility policy
notebooks/    Cleaning notebook, without published execution outputs
tests/        Preparation, training, scoring, and regression tests
plan.md       Sequential feature, training, evaluation and serving plan
data/         Ignored private source and generated tables
private/      Ignored backups of pre-cleanup notebooks
```

## Data handling

Raw data, generated tables, backups, embeddings, checkpoints, model weights and recommendations remain local and are ignored by Git. Clear notebook outputs before committing because they can contain company-specific records. The current cleaning notebook prints aggregate counts only.
