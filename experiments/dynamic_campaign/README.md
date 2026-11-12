# P0–P7 dynamic selection and candidate context campaign

Authorized on 2026-11-12 after reviewing the attached architecture plan. All new code is opt-in under `recommender/experiments/`; no default aliases, old results, or historical caches are replaced.

`protocol.json` records budgets, temporal policy, paired seeds, screening and final acceptance criteria. The common upstream is the **frozen demand-free S0-seed42 epoch-13 model**, not the leaky historical retriever. Historical checkpoint metadata supplies compatible shapes and vocabulary only when constructing new rankers; all learned branches are reset before transferring declared clean upstream projections. Seeds vary selector/ranker fits, not that fixed upstream.

All comparable new fits use the same 26,253 training rows with at least 100 rolling eligible products, and all 6,191 validation rows. The ten excluded early rows are excluded from every new fit, not only the depth-100 fits. The source target denominators remain complete, including unavailable products. Global observations are strictly earlier during training and frozen at the training cutoff during validation; customer observations continue to update strictly before each prediction. Current catalogue/text/company metadata is explicitly snapshot-based. Missing effective dates are not fabricated.

The pointwise selector trains against **all eligible products**, including repeat and unseen targets, rather than fitting old selected slates. Its static control uses purchase-stable pair features; dynamic adds clock/recency/global features; relations adds recent SKU and hierarchical/family/category/line relationships and complementarity. The 41-feature contract is ordered and hashed, and the float16 feature cache has file checksums. Computation uses float32. Product demand remains disabled in cached vectors. No request date is fed into either tower.

The request adapter deliberately reproduces the float16 feature quantization boundary before float32 scoring, matching training inputs. The first 13 features retain their production formulas except global popularity uses a fixed `log1p(1000)` denominator, preventing later training-count maxima from changing earlier features. The declared new count/age/lift scales apply to the additional feature groups; global sale age is measured at the frozen aggregate refresh.

Candidate policies use identical eligibility and deterministic ordering across selector comparisons. P2/P6 quota policies sort the repeat and unseen subsets by selector score; they differ from the historical personal-repeat-frequency heuristic. P3–P5 retain the original S0 personal-repeat 50-item slates and remain independent of P2. A separate completed `fixed50` probe records the poorer dot-ranked repeat policy. `top50` has no quota; `quota100_50` uses the same eligible pool.

Candidate-context models use one position-free 128D four-head attention block. Tests cover ID-aligned permutation equivariance, padding, scalar heads, gradients, residual initialization and repeat-feature ablation. Direct scoring, projected customer-dot scoring and zero-initialized residual scoring are independent variants. Additional routing experiments compare bias-only, values-only and both. These branches are not defaults.

```bash
# Regressions before training; never authorize held-out data for model selection
PYTHONPATH=. ./venv/bin/pytest tests/

# References, feature preparation, independent screens, paired repeats, combined/depth fits
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/run.py --phase all

# Feature availability and temporal contract evidence
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/inventory.py

# Cheap scoring control requested by the later review, on identical original slates
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/pointwise_control.py

# After all fits: selected-checkpoint replay, full warm CPU/MPS telemetry and cohorts
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/telemetry.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/summarize.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/verify.py
```

Intermediate phase commands are `--phase p0`, `--phase selectors`, and `--phase rankers`. A matching completed run is reusable only if its settings digest and selected checkpoint hash agree. An incomplete run fails rather than reusing stale files; archive it and choose an explicitly new namespace to retry. New model/cache artifacts live in ignored `checkpoints/dynamic-campaign-v1/` and `data/interim/dynamic_campaign_v1/`. Aggregates live here; per-customer predictions and IDs remain local.

To retrain the matrix without replacing this campaign, set fresh output namespaces before running the same commands. The frozen upstream inputs and checked feature cache are intentionally shared:

```bash
export RECSYS_CAMPAIGN_ROOT=checkpoints/dynamic-campaign-rerun
export RECSYS_CAMPAIGN_EVIDENCE=experiments/dynamic_campaign_rerun
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/run.py --phase all
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/pointwise_control.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/telemetry.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/summarize.py
PYTHONPATH=. ./venv/bin/python experiments/dynamic_campaign/verify.py
```

Keep these variables unset when inspecting the recorded campaign. `summarize.py` and `verify.py` can regenerate derived analysis/audits; diagnostic inventory, coverage and timing outputs refuse replacement. For a fresh inference replay with the recorded models, copy `results.json` and `decisions.json` into an empty evidence directory, set only `RECSYS_CAMPAIGN_EVIDENCE` to that directory, and run `telemetry.py`. This leaves the published timing evidence unchanged.

Before downstream training, direct checkpoint inspection corrected the initial recall shorthand to the actual stored guard `0.7558833119851873`, and fixed the simultaneous NDCG floor at that same artifact's `0.5918928743676547`. Selector fits are unaffected. Every downstream run uses both floors. A run that never qualifies keeps explicitly named `best_unqualified.pt` diagnostic weights and no selected checkpoint in its manifest; it cannot resolve as a successful serving selection.

A setup probe exposed the inferior repeat-slot policy before any component screen completed. Its complete control is retained as `fixed50-control`; one partially trained repeat-flag probe was archived, and the independent component screens restarted on original personal-repeat slates under the `production50` name. This is a recorded methodology correction, not an unreported checkpoint-selection change. `production_control.py` supplies the shared slate adapter; its standalone training entry point is optional and duplicates that baseline.

During seed-42 component screening, `supplementary_protocol.json` added bounded near-miss repetitions: at least ten extra new hits and recall within 0.003, with the original NDCG screen tolerance. This measures tradeoffs for configurations that narrowly miss the initial 0.002 recall tolerance. Checkpoint selection and final acceptance remain unchanged. The supplement is exploratory, disclosed separately, and executed by `pointwise_control.py` along with the later review's cheap matched-slate control.

The later review's pretrained residual proposal is a separate matched fine-tuning study: both the clean control and zero-initialized residual start from the same selected clean ranker, receive ten additional epochs at `0.0001`, and are compared with each other. A real development batch must reproduce the pretrained logits exactly before residual training. This is labeled as a 10+10-epoch budget; it is not credited as a fair ten-epoch architecture improvement.

The P7 depth-matched factorial adds dot selection at 100 candidates, with and without added scalar ranker signals, and the same added-signal ranker after the dynamic selector. All receive three seeds. This prevents attributing a wider-slate gain to an unnecessary MLP selector. The dot-serving benchmark computes candidate features only for the chosen 100 products; feature-subset tests prove this preserves the complete customer history and exact input values.

`DynamicSelectorService` wraps the existing versioned purchase cache: `on_purchase` is its only tower encoding path; `score` uses cached vectors and cheap dated features. Catalogue/model refresh creates an empty new namespace and requires explicit purchase-snapshot replay. The original cache remains usable. No experiment is promoted without separate user approval.

The serving adapter takes an external as-of feature provider and eligibility masks; application event ingestion must update those facts alongside the purchase cache. The full benchmark uses resident development facts, not a deployed feature service. `DynamicFeatureBuilder(..., relationships=False)` omits transition tables and work for the independent dynamic selector plus ordinary UPS-CANR path; its first 33 features match the full provider exactly. Combined models that use relationship inputs retain those tables. CPU/MPS full-request timing distinguishes these costs.
