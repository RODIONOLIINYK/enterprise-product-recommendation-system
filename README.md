# Enterprise Product Recommendation System

Next-basket recommendations for B2B enterprise customers using **Stage 1 Two-Tower Retrieval** and **Stage 2 Candidate-Aware Neural Reranking (`UPS-CANR`)**.

## Current Status & Documentation Map

- **[AGENTS.md](AGENTS.md):** Agent handbook with the exact active reference architecture, module map, checkpoint/cache inventory, and strict temporal-leakage guardrails.
- **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md):** Verified validation metrics for the active Stage 1 and Stage 2 reference models, plus a summary of tested-and-rejected experimental hypotheses.
- **[plan.md](plan.md):** Current project goals, diagnostic findings, and the prioritized experimental roadmap for improving new-to-customer product discovery in Stage 2.

The 2026-11-10 [audit](AUDIT_REPORT.md) found that local checkpoint selections differ from the historical reference metrics. Check [artifact metadata](audit/artifacts.json) before training or evaluation. Saved models and caches were preserved.

The [2026-11-12 follow-up](HARDENING_REPORT.md) implements versioned demand-free training, validated caches, provenance-bound isolated runs, and compatible scoring/preprocessing refactors. Historical default aliases remain unchanged.

[DYNAMIC_SELECTOR_EXPERIMENTS.md](DYNAMIC_SELECTOR_EXPERIMENTS.md) records the completed 68-fit P0–P7 campaign. A wider dot-selected slate provides the clearest conservative gain; full-pool MLP selection and candidate attention do not establish a reliable incremental benefit. Scalar scoring features remain an optional accuracy-oriented candidate. All default aliases are preserved.

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
  - `CustomerTower` encodes each historical purchase using its shared 128D product vector, normalized quantity, and normalized inter-purchase gap (`history_gap_reference_days = 74.0`), processed by a 2-layer Transformer with masked-mean plus last-valid-event pooling and a learned projection, fused with 1024D `BAAI/bge-m3` company text and 6 normalized customer profile features in the local checkpoint.
  - `ProductTower` encodes 1024D `BAAI/bge-m3` product text (without subcategory), one-hot `container_type` (with explicit `none`), and 6 demand features from a fixed training-cutoff snapshot in the local checkpoint (`remove_product_attribute_embeddings = True`).
  - Scoring adds 4 point-in-time binary familiarity flags (`brand`, `product_category`, `business_line`, `product_family`) with learned weights over a rolling candidate pool of merchandise with at least one prior paid sale and no active block.
- **Stage 2 (`recommender/ranker.py`, `recommender/train_ranker.py`):**
  - `UPSCANRRanker` with `statistics_attention = True` and `excluded_statistics = ("history_repeat",)`.
  - Builds a 33-token Customer Memory (Token 0 = Customer Profile, Tokens 1..32 = Historical Purchases contextualized by a 1-layer TransformerEncoder).
  - Scores 50 candidates per transition using 4-head cross-attention biased by candidate-to-memory bilinear statistics interactions ($14 \times 17$ per head) and a `397 -> 256 -> 128 -> 64 -> 1` MLP over `[z_candidate (128), h_attended (128), z_candidate * h_attended (128), pairwise_features (13)]`.

## Shared-representation experiment

[SHARED_STAGE1_EXPERIMENTS.md](SHARED_STAGE1_EXPERIMENTS.md) documents the separate 176D model, its eight-feature history input, original/reduced bias comparison, three-seed results and reproduction commands. The existing ranker remains the default: the new model did not show a reproducible discovery gain. Optional 192D/224D layouts are tested but their training is deferred. All original checkpoints and caches are preserved.

## Setup

Use Python 3.12 (`./venv/bin/python`).

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
# Tests and lint
./venv/bin/pip install -r requirements-dev.txt
```

Direct dependencies are pinned to tested versions. On the tested macOS arm64/Python 3.12 platform, add `-c constraints-macos-arm64.txt` for the recorded transitive dependency versions. This platform snapshot is not a universal wheel lock. Notebook dependencies are separate in `requirements-notebook.txt`.

## Data Preparation

Private raw inputs live locally under `data/raw/` (configured in `configs/datasets_links.toml`):

```text
data/raw/dataset.xlsx
data/raw/customers.csv
data/raw/items.csv
```

The preparation API returns tables; the default CLI prints row counts. Explicitly export the purchases table when preparing data for training:

```bash
./venv/bin/python -m recommender
./venv/bin/python -m recommender --export-purchases data/interim/cleaned_purchases.csv
```

Exports refuse to replace an existing file unless `--overwrite` is supplied. Preparation reads configured raw data; it is not part of validation-only model selection. Development training/evaluation uses a date gate before materializing purchase fields.

- **Paid-sale eligibility (`configs/product_scope.toml`):** Only positive finite `ПРОДАЖА` records with `Gen_ Prod_ Posting Group = ТОВАР` and item-master classification `ТОВАР` count as purchases. Gifts (`ПОДАРОК`), returns, and accounting rows are excluded.
- **Record merging (`recommender/merging.py`):** Same-company, same-calendar-date, same-`product_id` lines are merged and their quantities summed. Package variants remain distinct product IDs.
- **Chronological splits (`recommender/split.py`):**
  - **Train:** `< 2025-10-01` (historically documented `26,264` transitions and `26,184` full rolling slates; local cache has `26,263` rows—see audit)
  - **Validation:** `2025-10-01` to `2026-02-18` (`6,191` next-basket transitions, `1,857` new-to-customer target instances)
  - **Test:** `>= 2026-02-19` (`6,327` transitions — **strictly held out**)

## Training & Evaluation Commands

Run from the repository root using `./venv/bin/python`:

```bash
# Train the reference-shaped Stage 1 architecture in a fresh run, with dynamic demand disabled
PYTHONPATH=. ./venv/bin/python -m recommender.train_retriever --discovery-reference --demand-policy disabled_v1 --run-id demand-free-01

# Evaluate Stage 1 on rolling prior-sale discovery pool (saves to checkpoints/stage1-rolling-discovery/)
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate_stage1

# Train Stage 2 in a new checkpoints/ranker/runs/<run-id>/ directory
PYTHONPATH=. ./venv/bin/python -m recommender.train_ranker

# Validate the preserved legacy default; report goes to checkpoints/evaluations/<unique-id>.json
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate

# Validate an explicitly selected new run with complete provenance required
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate --ranker-checkpoint checkpoints/ranker/runs/RUN_ID --strict-provenance

# Run the unit and integration test suite
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
```

Both trainers create exclusive `runs/<run-id>/` subdirectories; they never replace parent reference aliases. Use the matching retriever `.pt` path for training and evaluation when selecting a new retriever. New ranker caches use format v4; historical v3 files remain untouched. Legacy checkpoint evaluation is explicitly labeled as having an unverified input contract. `--pairwise-metadata-policy observed_only_v1` opts new ranker training into strictly prior purchase-observed pairwise attributes; this does not recover missing historical catalogue metadata.

Six legacy tests read the unsplit real purchase CSV and are skipped by default. Run them with `--allow-heldout-data` only after explicit authorization for final held-out testing. Synthetic split-boundary tests remain enabled. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for remaining temporal and cache limitations.

## Data Handling

Raw data, generated tables, cached embeddings, precomputed slates, and model checkpoints remain local and are ignored by Git. Clear notebook outputs before committing.

### E1–E5 controlled experiments

The [E1–E5 report](E1_E5_EXPERIMENTS.md) records 32 validation-only fits: trainable unseen/transition signals, older-history pooling, and hard/smooth multi-interest retrieval, with downstream rankers, seed variation, ablations and latency/cache costs. [Reproduction commands](experiments/e1_e5/README.md) use isolated artifacts and purchase-triggered versioned caches. No experiment was promoted to the default system.
