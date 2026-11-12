# Experimental module index

Experimental code stays separate from the production model by opt-in entry points.
Historical module paths remain stable so recorded commands and checkpoint reconstruction continue to work. Importing production modules does not eagerly import these experiments.

| Area | Existing stable modules |
|---|---|
| Shared representations | `shared_ranker`, `shared_data`, `train_shared_ranker`, `summarize_shared_ranker`, `shared_temporal_probe` |
| E1/E2 signals | `experimental_signals`, `experiment_signals` |
| E3–E5 retrieval | `experimental_retrieval`, `experiment_retrieval_data`, `experiment_retrieval` |
| Serving and evidence | `experiment_serving`, `experiment_evidence`, `experiments/e1_e5/*.py` |
| Dynamic selector / candidate context | `experiments/dynamic_selector/`, `experiments/candidate_context/`; campaign in `experiments/dynamic_campaign/` at repository root |

Core contracts, sealed base slate caches, run manifests and the development-only CSV reader live in production-neutral `artifacts` and `development`. The experimental training caches retain their historical schemas; base ranker cache format v4 uses a separate filename. Do not migrate historical caches or update recorded metrics in place.
