# Rolling MLP input and latency audit — November 13, 2026

This audit uses the frozen `repeat-no-month-v1` model, development purchases only, and inference only. No training, held-out evaluation, promotion, commit or push is authorized by this audit.

## Findings

1. The original `rolling_candidate_v2` slate builder attached rolling features but selected candidates using historical `reference_selector_scores`. Those scores were generated with frozen validation globals. New selection must recompute the frozen 301D selector with the rolling features and bind score/slate caches to their exact feature and model identities.
2. Feature cache identities originally bound a historical retrieval identity and transition keys without fingerprinting the current observed purchase and metadata inputs. Checksums verified the cached file, but could not establish its freshness relative to the current source snapshot. Changed sources must invalidate cache reuse.
3. Historical/default production aliases remain distinct from this experimental rolling path. The new development adapter is not a deployed purchase ingestion service and does not automatically discover new sales.

## Independent temporal checks

See `independent_checks.json` for the actual results and timing samples. On 24 training queries and 24 validation queries selected across chronological quantiles and evaluated in reverse order:

- All 41 regenerated, quantized context features match the prepared rolling feature cache exactly.
- All six lifetime customer scalars, freshly computed from strictly earlier purchases with the source checkpoint normalizer, match the cached source values exactly.
- All valid history timestamps precede the prediction date.
- Rebuilding entire feature providers using only purchases strictly before October 1, 2025; December 1, 2025; and February 18, 2026 yields exactly the same features as the full development observation store queried at those dates. Same-day and future purchases are therefore unnecessary for these checked outputs.
- All cached customer and product vectors are finite and have L2 norms within approximately 2e-7 of one.

Sampling and synthetic boundary tests support the temporal implementation; they are not an exhaustive proof over every possible event stream. Metadata remains `snapshot_unverified`: category/brand/company metadata are not effective-dated historical records.

## Input contract and semantic limits

The active MLP has 301 inputs: customer vector 128, product vector 128, retained contextual scalars 39, and customer lifetime scalars 6. `is_repeat_sku` remains present; the two month features are omitted. Eight retained growth/lift scalars and latent vectors remain signed. All-input `[0,1]` normalization is unfinished; changing it requires a separately trained/versioned model, not reinterpretation of the saved weights.

`PointInTimeFeatureStore.total_orders` and category share counts use purchase rows/events. Thus `user_cat_order_share` is a share of SKU purchase events, not a share of distinct dated baskets. The six customer lifetime features separately include `distinct_orders`, which does use distinct purchase dates. A dedicated customer/category basket-count scalar is absent. The audit preserves these saved-model semantics.

Externally supplied customer vectors, product vectors, and six customer statistics must belong to the same coherent observed snapshot and model version. A source fingerprint does not regenerate an upstream vector cache. After source corrections or new data, rebuild/refresh affected upstream inputs together; do not combine rebuilt context features with stale historical customer snapshots.

## Performance and architecture judgment

The new request measurements use real cached vectors, real rolling features, eligibility masking and top-100 sorting for one customer against all 841 products. They exclude raw purchase ingestion, lifetime/history preparation, vector recomputation, storage and networking. Warm global-date and customer snapshot caches are assumed.

The older vector refresh benchmark separately measured 841 products and 2,571 development customer snapshots, with BGE text embeddings and model inputs already prepared. It excludes aggregation and persistence. It does not measure all 1,722 sellable catalogue records or a complete production refresh.

At this scale, full-pool MLP inference costs milliseconds, while refreshing every customer vector costs tens to hundreds of milliseconds of encoder computation. These costs make statistics inside the vectors technically feasible, but establish no accuracy benefit. Keep the corrected rolling MLP as the comparison baseline. A future controlled variant can put point-in-time product demand/customer size statistics into towers while retaining useful direct pairwise scalars. Existing historical fixed-cutoff demand must not be restored as a causal input.

If product vectors depend on dynamic statistics, product updates also affect customer history encodings that consume those vectors. Category/global denominators can affect multiple products, and rolling windows expire without a new purchase. Refresh must account for these dependencies and clock-driven changes; it is not always one product and one customer per purchase.

## Completed fixes and final validation

Gemini 4 Hydrogen implemented the fixes in the existing isolated package:

- New sealed caches live under `data/interim/rolling_candidate_v2/v3/`; earlier caches remain unchanged.
- Candidate slates default to `rolling_selector_scores`, recomputed from the frozen 301D model and the exact rolling feature cache. Historical scores require explicit selection for diagnostics.
- Feature, score and slate identities bind a fingerprint of the observed development purchase and metadata snapshot. Changed observed sources invalidate cache reuse.
- Missing prediction dates are rejected. The live scoring adapter also validates model contract, canonical product ordering, candidate IDs, tensor devices, finite values and eligibility-mask shapes.

The same frozen selector on rolling validation inputs improves overall Recall@10 from 0.754114 to 0.758447 (approximately +0.43 percentage points) and NDCG@10 from 0.586331 to 0.588225. New-product hits in the ordinary top 10 increase from 194 to 206; unseen-only hits@25/50/100 change from 726/1029/1337 to 757/1041/1357. The paired customer-clustered 95% interval for the Recall@10 change excludes zero. These are fixed-weight validation input corrections; they do not estimate retraining variability or the benefit of a new architecture. Exact values, denominators, monthly metrics and intervals are in `../rolling_candidate_v2/rolling_selector_replay.json`.

Independent final checks are recorded in `final_verification.json`: saved rolling scores reproduce reported key metrics; every selected validation slate index matches selection from those rolling scores; v3 feature hashes match the independently audited feature values; all 17 preserved upstream/reference hashes match. Both required suites pass: 544 passed / 6 skipped for full tests; 535 passed / 15 skipped for synthetic-only tests, with 70 subtests in each. `git diff --check` passes.

No training, held-out evaluation, promotion, commit or push occurred. Historical reference weights and pointers remain unchanged.
