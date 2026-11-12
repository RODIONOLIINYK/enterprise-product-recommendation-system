# Verified audit fixes and compatible refactors — 2026-11-12

Implemented against `c7788ec`, following the user's approval to implement verified fixes and compatible refactors. No model was promoted, no real training run was launched, and no held-out test evaluation was performed. The existing E1–E5 and shared-representation experiment results remain the architecture evidence; this follow-up repairs infrastructure and preserves their reconstruction paths.

## Correctness changes

| Finding | Implemented behavior | Compatibility and limits |
|---|---|---|
| Fixed training-cutoff demand leaks later training sales into earlier examples | New Stage 1 training defaults to versioned `disabled_v1`: all six dynamic demand inputs and the optional SKU median quantity are zero; packaging inputs remain. Loaders reconstruct the saved policy and sanitize externally supplied demand tensors without mutating callers. | Missing policy in old checkpoints means `legacy_train_cutoff_v1`. Explicit legacy training remains available for reproduction. This removes this demand pathway; it does not resolve historical metadata or change the legacy trainer's candidate-mask policy. The separate E3–E5 S0 protocol remains the evaluated prior-sale-masked control. |
| Metadata may use future purchase fallbacks or current masters | Optional `observed_only_v1` pairwise policy ignores current catalogue/company snapshots and uses purchase-record attributes strictly before the prediction time. It has a distinct cache identity/namespace. | Default `snapshot_unverified` retains historical behavior. Industry affinity is unknown/zero without verifiable history. Current product texts, familiarity metadata and block snapshots still lack effective dates; no full historical correctness claim is made. |
| Slate identity alone is insufficient | Base cache format v4 validates ordered feature names and units, tensor shapes/dtypes/finite values, candidate mappings, duplicate IDs, labels, history validity/padding, strictly prior history, split dates, exact transition order/targets and a sealed payload checksum. | v3 model schema remains unchanged. Historical cache files are never migrated in place. A conflicting destination produces a content-addressed alternate filename or an error, never replacement. |
| Checkpoint origin is incomplete | New ranker container format v4 records the named input contract and digest, retriever digest, both originating cache identities and payload digests, copied model configuration plus digest, metadata policy and selection metric. Evaluation validates the contract and actual validation payload. | Untagged legacy checkpoints still load with an explicit `legacy_unverified_input_contract` result; strict evaluation rejects them. A tagged checkpoint with missing provenance is rejected. |
| Reused output directory can expose stale selected aliases | Each trainer creates an exclusive child run directory and manifest. Evaluation resolves only a completed run's named selected checkpoint and verifies its SHA-256. A failed discovery guardrail leaves no selected alias. | Parent `best_*.pt` aliases are preserved. Interrupted runs remain incomplete and cannot resolve as successful selections. Checkpoints may be replaced while that new run is training; completed runs are never implicitly reused. |
| Validation reads more purchase fields than needed | Production development readers share the date-gated CSV loader formerly confined to experimental code. Held-out record fields are not materialized into development DataFrames. | The reader scans CSV dates to partition records. Explicit final-test code remains gated; no final-test option was used. |
| `Log1pScaler` ignores custom clip bounds | Custom bounds now clip the unbounded nonnegative log ratio; invalid bounds raise. | Default `[0, 1]` behavior is exactly retained. |
| Transition score depends on other candidates when all are unknown | Unknown products now receive the same smoothed score whether alone or in a mixed candidate list; smoothing must be finite and positive. | Historical E2 evaluated candidates belong to its catalogue, so recorded results do not change. |

## Refactors and performance

`UPSCANRRanker` now exposes `encode_customer_memory`, `encode_candidate_features` returning a named `RankerFeatures`, and `score_candidates`. The compatible `forward` composes those stages. The E2 MLP feature wrapper uses the named tensor instead of intercepting an identity MLP and storing 397-dimensional features in `raw_scores`; output logits/raw scores remain scalar per candidate. Historical state dictionary keys load strictly and produce identical outputs.

Cached-only ranker training freezes the attached ProductTower and excludes frozen parameters from the optimizer. The tower remains in the state dictionary and is available when raw-input mode is explicitly enabled. Historical product-text projection, numeric projection branches, legacy history modes and profile features are retained because supported checkpoints depend on them. The unassigned catalogue-vector fallback emits a deprecation warning rather than disappearing.

Training and validation share batch-to-model assembly. It avoids transferring candidate indices when direct vectors suffice, and keeps unused retriever profiles/reporting metadata off the device. Shared-model inputs remain conditional. Public collator fields and experimental reconstruction paths remain compatible.

Customer snapshots use incremental prefix statistics, exact two-heap SKU gap medians, and bounded LRU state. Backdated queries rebuild safely; later updates cannot mutate previously returned snapshots. Rolling catalogue eligibility caches are bounded and return defensive copies. Neither change adds request-time customer tower recomputation. Existing purchase-triggered, versioned embedding-cache semantics remain unchanged.

`forward_scores` optionally disables cross-attention weights and summaries with `need_weights=False`. Default diagnostic `forward` is unchanged. This fast path remains opt-in because kernel timing and numerical rounding vary. These APIs operate on supplied customer inputs; serving continues to use the existing purchase-triggered embedding cache.

The reproducible synthetic benchmark compares the old source loaded with `git show c7788ec:...` against this source, without private data or checkpoints. See [raw measurements](audit/hardening_benchmarks.json):

| Check | Before / diagnostic | After / score-only |
|---|---:|---:|
| 100 chronological snapshot queries over 1,000 synthetic events, median of 5 fresh-store samples | 43.819 ms | 4.200 ms |
| Ranker MPS median, B=1 / C=50 / L=32, 100 samples | 1.150 ms | 1.040 ms |
| Ranker MPS p95, same run | 1.524 ms | 1.070 ms |
| Default logits maximum absolute difference | — | 0 |
| Historical E2 wrapper logits maximum absolute difference | — | 0 |
| Score-only logits maximum absolute difference | — | 4.77e-7 |

Snapshots and pairwise features match exactly, including backdated queries. An earlier timing invocation measured score-only p95 at 3.042 ms versus 1.601 ms with diagnostics, so the latest faster tail is not evidence of a guaranteed tail-latency improvement. All measurements are warm local microbenchmarks; they exclude storage, network, serialization and deployment load.

## Workflow cleanup

- CLI preparation has an explicit atomic `--export-purchases PATH` command, refusing replacement unless `--overwrite` is supplied. Ordinary preparation still returns tables/prints counts.
- Default evaluation writes unique report paths. Explicit report replacement requires `--overwrite-report`; historical `evaluation_report.json` is preserved.
- Runtime, test/development and notebook requirements are separated and pinned. `constraints-macos-arm64.txt` records 51 installed transitive runtime/test versions for CPython 3.12/macOS arm64; it is not a cross-platform, hash-locked wheel manifest.
- Added a synthetic-only GitHub Actions job and undefined-symbol Ruff checks. The job has no private data/checkpoints and disables model downloads. Workflow setup follows the [official setup-python examples](https://github.com/actions/setup-python). Linux Actions execution is still pending; local execution used the existing tested macOS environment.
- `recommender/experiments/README.md` logically groups experimental modules without breaking recorded import paths or importing experiments in normal production startup. The production CSV reader now lives in `development.py` with a compatibility re-export.
- Removed the exact duplicate `campaign_complete.json`; `repetition_selection.json` is the canonical completion/selection record, and its writer now maintains the status. Historical metrics and original experiment source digests remain historical evidence.

## Verification

The full suite passes **457 tests and 70 subtests**, with the six unsplit real-purchase readers skipped. New tests cover corrupt schemas/dates/labels/candidates/checksums, strict and legacy provenance, versioned demand invariance, metadata future invariance, run guardrail failure and successful manifest resolution, exact/default scoring, E2 state compatibility, cached/raw tower modes, required batch transfers, bounded/backdated prefix state, safe eligibility copies, clipping, transition smoothing, atomic exports and the development date gate.

Synthetic-only tests also pass **451 tests and 70 subtests**, with 12 skipped, in a temporary source checkout containing no `data/` or `checkpoints/` directories. The Python file-access guard is an additional test aid, not an OS sandbox. Six tests requiring local checkpoints are skipped in this mode in addition to the six real-purchase tests. The installed virtual environment was reused; this verifies source/data isolation, not a fresh Linux dependency installation.

The reference replay matches its recorded JSON exactly: Stage 1 unseen hits remain **652@25 / 921@50**, and the preserved ranker selections remain **201 / 209 / 201** new-to-customer hits for alias/discovery/NDCG respectively. All seven checkpoint/cache SHA-256 values from `audit/artifacts.json` are unchanged. See [verification record](audit/hardening_validation.json). Existing historical and E1–E5 numerical reports were not rewritten.

```bash
# Safe regression suite; never add --allow-heldout-data for development
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
./venv/bin/python -m ruff check recommender tests experiments tools --select F821,F822,F823

# Synthetic old/new compatibility and latency; writes audit/hardening_benchmarks.json
PYTHONPATH=. ./venv/bin/python tools/benchmark_hardening.py

# New Stage 1 architecture training, isolated run; does not replace reference aliases
PYTHONPATH=. ./venv/bin/python -m recommender.train_retriever --discovery-reference --demand-policy disabled_v1 --run-id demand-free-01

# New Stage 2 run; choose the intended retriever .pt explicitly
PYTHONPATH=. ./venv/bin/python -m recommender.train_ranker --retriever-checkpoint checkpoints/retriever-discovery-reference/best_retriever.pt --run-id ranker-01 --pairwise-metadata-policy observed_only_v1
PYTHONPATH=. ./venv/bin/python -m recommender.evaluate --ranker-checkpoint checkpoints/ranker/runs/ranker-01 --strict-provenance
```

The last two example runs deliberately use the preserved retriever and therefore remain conditional on its audited inputs. To use a newly trained retriever, supply its selected `.pt` path to both Stage 2 training and evaluation. Old experiment drivers retain their original output/cache conventions for faithful reproduction; the isolated base-trainer API is the new default for new work.

Basket-aware sequence encoding, effective-date metadata collection, rolling historical demand snapshots and changes to numerical memory values remain separate behavioral experiments. They cannot be established as compatible correctness refactors. The existing architecture recommendation is unchanged until new controlled validation evidence and explicit promotion approval are available.
