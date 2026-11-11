# E1–E5 experiment campaign

All artifacts are local. Reference aliases, checkpoint files and legacy/shared slate caches are preserved. Nothing in this campaign changes the default ranker or serving path.

## Implementations

- `recommender.experimental_signals`: E1 zero-initialized unconstrained unseen scalar; E2 point-in-time basket transition counts, smoothed log-lift, zero-initialized residual and additional MLP feature.
- `recommender.experiment_signals`: matched control, 10-epoch Stage 2 fits, independent discovery/NDCG selections, selected-weight replay, zero-signal ablations, per-basket evidence and synchronized accelerator latency.
- `recommender.experimental_retrieval`: E3 older-event mean/attention pooling, E4 independent 1/2/3 query interests over the unchanged recent-history Transformer, E5 hard max versus log-mean-exp; versioned purchase embedding cache.
- `recommender.experiment_retrieval_data`: isolated development inputs with six demand columns disabled, strictly earlier history, prior-sale/blocked eligibility, full unseen target denominators and compressed older-event multiplicities.
- `recommender.experiment_retrieval`: 30-epoch fresh Stage 1 fits, purchase-vector caches, production mixed-slate construction and fresh 10-epoch downstream ranker fits.
- `recommender.experiment_evidence`: aggregate mean/sample SD, paired customer-cluster bootstrap, resources, source hashes and original-artifact preservation checks.

## Commands

Run from the repository root; no command authorizes held-out test evaluation.

```sh
PYTHONPATH=. ./venv/bin/pytest tests/ -q

# First screen E1/E2 at one seed.
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_signals --seeds 42

# Repeat small configurations with matched controls; completed identical runs reuse results.
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_signals --seeds 42 17 2026

# Screen every E3/E4/E5 configuration and train its downstream ranker.
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_retrieval --seeds 42

# The campaign applies a documented screening rule, repeats promising families
# and paired controls at seeds 17 and 2026, and writes its selection decision.
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/run_retrieval_campaign.py

# Confirm the demand-disabled control at three seeds even when no extension passes.
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_retrieval --runs S0 --seeds 17 2026

# Explicit alternative to the automatic repetition rule:
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_retrieval --runs S0 E3mean E3attention --seeds 17 2026
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/replay_references.py
PYTHONPATH=. ./venv/bin/python -m recommender.experiment_evidence
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/scoring_ablations.py
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/attention_diagnostics.py
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/benchmark_pipeline.py
PYTHONPATH=. ./venv/bin/python experiments/e1_e5/finalize_report.py
```

`--output` selects a separate checkpoint directory. Existing completed runs must have exactly matching settings; incomplete runs are rejected rather than overwritten. `--epochs` and `--ranker-epochs` change budgets and require a new output path. `--skip-downstream` is available only for implementation smoke checks, not complete comparisons.

## Contracts and interpretation

E1/E2 use exactly the audited frozen retriever and fixed 50-candidate slates. Results are conditional on that retriever, which inherited training-cutoff demand look-ahead. They do not certify a temporally clean Stage 1 model. Historical epoch-10 metrics are not mixed with local epoch-4/epoch-7 checkpoints. A fresh same-seed control is the comparator for every signal fit.

For E2, `N_AB` counts adjacent date-grouped baskets, with a destination SKU counted at most once per basket. Both source and destination must occur strictly before the prediction date. Global counts freeze at 2025-10-01 throughout validation. `P(B)=(N_B+0.5)/(N_transitions+1)` prevents undefined log-lift. Conditional smoothing uses alpha=20, fixed before screening. Customer weights are normalized event frequencies over all strictly earlier purchases; there is no manual recency decay. MLP injection adds a column only to the first scoring layer and does not widen the reference attention bias. E1 and E2 are never combined.

E3–E5 use a separate, freshly initialized Stage 1 control, S0. No learned reference weights initialize the experimental towers; the reference loader supplies metadata, eligibility and existing feature scales only. All six demand columns are disabled both in cached inputs and in the product forward. Packaging features remain. This removes the identified training-demand look-ahead path without introducing date-dependent product vectors. Training loss also respects the same rolling prior-sale and blocked-product masks as evaluation. Snapshot catalogue/text metadata and training-fitted scales remain inherited inputs, so this is not an audit of historical metadata availability.

S0 has the original mean-plus-last history pooling. E4k1 is an extra one-query pooling control; E4k2/E4k3 add only more interest queries. Each query-pooled summary is concatenated with the same last-valid-event vector and passed through the existing history projection before text/profile fusion; this preserves the original last-event route in the one-query control. E4/E5 never use E3's older-event memory. Hard maximum and smooth maximum operate on normalized cosine similarities before the common 0.1 temperature and existing product/familiarity terms. Smooth scoring uses tau=0.1 and subtracts tau*log(K), avoiding a K-dependent BCE intercept. K=1 hard/smooth equivalence is tested.

Stage 1 selection maximizes new-target hits@25, breaking ties by hits@50 and retaining the earliest exact tie. Stage 2 selects discovery hits with the preserved local discovery checkpoint's fixed overall-recall guardrail (0.7558833119851873); independent peak-NDCG checkpoints are saved. A downstream run with no qualifying epoch is reported explicitly as a guardrail failure, with its unconstrained checkpoint for diagnosis only. Such a model is not a recommendation.

All models retain every validation basket and all 1,857 unseen target instances in denominators, including products that are absent, blocked or globally first-sale at prediction time. Conditional new recall divides hits by unseen targets present in the actual mixed slate, not 1,857. Different Stage 1 models produce different slates; each downstream ranker is trained on that model's own training slates. The exact production mixed-slate constructor and pairwise-feature store are reused.

## Serving and memory

`PurchaseEmbeddingCache.on_purchase` encodes the history once per purchase version and caches normalized customer vectors plus a separate long-term vector when E3 is active. `recommend` performs a version check, cache lookup and product similarities; it never invokes a tower or accepts a recommendation clock. Changing model/catalogue versions requires an explicit cache rebuild. State can be saved with `state_dict` and restored only into a matching namespace. `experiment_serving.refresh_catalogue` creates a separate model/cache namespace, remaps existing product biases by canonical SKU and initializes new-SKU biases at zero without mutating the previous model. Cold-start initialization is explicit.

Frozen training/evaluation purchase caches bind the selected checkpoint digest and input identity. At a global model/catalogue refresh, product vectors and customer caches are rebuilt under a new version. Runtime rolling eligibility and familiarities can filter/adjust cached scores without changing the tower inputs. No time-dependent tower recomputation is introduced.

Per customer, float32 vectors cost 512 bytes for S0/E4k1, 1,024 bytes for E3's final vector plus long-term summary or E4k2/E5k2, and 1,536 bytes for three interests. Dictionary keys, purchase IDs, history-store data and allocator overhead are additional. Offline training caches deliberately retain per-transition snapshots and scores for audit; these are not the production per-customer cache footprint. Repeated older events are stored as SKU/count pairs and yield exactly the same attention/mean pooling as expanded repeated product vectors.

Timing covers training plus per-epoch validation. Serving timings synchronize the accelerator and report warmed median/p95 for cached similarity and purchase encoding separately. Stage 2 timings include its candidate-aware neural forward on cached product vectors. Measurements share one workstation and are operational telemetry, not isolated hardware benchmarks. Parameter counts include checkpoint-compatible branches, including the Stage 2 ProductTower that is bypassed by direct candidate vectors.

Paired 95% intervals resample customers 2,000 times on matched validation baskets. They characterize selected fixed models; they do not incorporate model-selection or training-seed uncertainty. Repeated-seed means/sample standard deviations are reported separately. A small validation difference does not automatically select a more complicated architecture.
