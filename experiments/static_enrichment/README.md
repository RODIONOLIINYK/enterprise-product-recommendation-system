# Static Product & Customer Metadata Enrichment Campaign

Isolated, demand-disabled (`disabled_v1`) Stage 1 + downstream Stage 2 (`quota100_50`) evaluation of static metadata enrichment.

## Protocol & Isolation
- Code: `recommender/experiments/static_enrichment/` (core modules untouched).
- Caches: `data/interim/static_enrichment_v1/` (BGE-M3 text caches keyed by recipe hash, training-only vocabularies, cohort masks).
- Checkpoints: `checkpoints/static-enrichment-v1/runs/`.
- Evidence: `experiments/static_enrichment/`.
- Metadata policy: `snapshot_unverified` (`data/raw/описание.csv` created 2026-11-12; `items.csv` / `customers.csv` snapshot). Time-dependent price columns (`col 3` in `описание.csv`, item price cols) and customs code (`col 98`) are strictly excluded.
- Held-out test split (`>= 2026-02-19`) is never read, tuned on, or evaluated.

## Commands
```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase prep
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase c0
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase screen
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase repeat
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase downstream
PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/summarize.py
```
