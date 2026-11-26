# Current CatBoost and separate-profile memory comparison

All new fits are development-only, seed 42, with no held-out evaluation, promotion, commit or push. Current comparisons share 26,253 training / 6,191 validation queries and the same 100 retrieved candidates per query, all 10,276 target instances and 1,857 new targets. Statistics roll strictly before prediction dates. No targets are injected into current candidate sets.

## Historical CatBoost evidence

The final historical model was a **CatBoostClassifier**, following an earlier CatBoostRanker. The classifier was introduced in `29ab48b` and made the only model in `0fdb00b`. Configuration and saved validation results were recovered from `2cb7f75`; the binary was last changed in `535e211` without an accompanying metrics report update. Historical saved numbers therefore do not independently certify that last binary.

Saved validation: Recall@10=0.92590, NDCG@10=0.68607, MRR=0.63382, first-purchase Recall@10=0.80056.

That benchmark had 4,247 validation groups with approximately 25 candidates, inserting all targets plus sampled negatives. It split customers 80/10/10, and training covered 2024-01-03 through 2026-07-17. These historical scores must not be compared directly with the current temporal, retrieved-slate, complete-target benchmark. The old weights were not loaded or evaluated: they were trained on dates later than current validation.

The original 17-feature recipe was rebuilt causally on current candidates: 3 categorical features (SKU/category/business line), personal quantity/frequency/recency/cadence, category/line affinity and rolling global popularity/buyers. Missing cadence remains NaN. Category/line metadata use the declared current snapshot rather than invented historical effective dates. The current 53-feature variant uses 3 categoricals +41 rolling candidate features +8 profile features +1 tower dot similarity. It does not use the rejected Stage 1 MLP score as a Stage 2 input.

Both classifiers reuse the historical 500-tree maximum, depth 8, learning rate 0.05 and Logloss objective/early stopping, with 4 CPU threads. The original Logloss-selected results are retained separately. For the common ranking comparison, prefixes are evaluated every 50 trees (at most 10 checkpoints), using the same discovery guardrails and NDCG ordering as neural fits.

## Current matched-candidate results

Recall is macro over complete target baskets. New recall is micro over all 1,857 new target instances, including targets absent from the slate. All models share the same reachability ceiling. This is a practical model comparison: CatBoost and the transformer have different inputs, objectives and training algorithms. It does not isolate a pure architecture effect.

### NDCG-selected weights

| Model | Epoch/trees | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | Repeat micro R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 MLP on identical slate | fixed | 0.64696 | 0.75845 | 0.85096 | 0.89877 | 0.94010 | 0.58822 | 0.56685 |206/1,857 |0.88740 |
| A / fused profile /32 events | 8 | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 |255/1,857 |0.89298 |
| CatBoost / original 17 features | 300 | 0.65224 | 0.75793 | 0.84455 | 0.89638 | 0.94010 | 0.59457 | 0.57917 |143/1,857 |0.89251 |
| CatBoost / current 53 features | 400 | 0.65282 | 0.75783 | 0.84608 | 0.89805 | 0.94010 | 0.59512 | 0.57783 |201/1,857 |0.88728 |
| Two fused profile tokens /32 events | 6 | 0.65767 | 0.76911 | 0.85814 | 0.90302 | 0.94010 | 0.60056 | 0.58170 |243/1,857 |0.89155 |
| Separate text+attributes /32 events | 6 | 0.65943 | 0.76676 | 0.85830 | 0.90271 | 0.94010 | 0.59948 | 0.58096 |235/1,857 |0.89013 |
| Separate text+attributes / 64 events | 9 | 0.65717 | 0.76646 | 0.85731 | 0.90213 | 0.94010 | 0.60007 | 0.58217 |254/1,857 |0.88847 |
| A / fused profile / 64 events | 9 | 0.65936 | 0.76617 | 0.85548 | 0.90089 | 0.94010 | 0.59997 | 0.58183 |243/1,857 |0.88799 |

Paired customer-clustered95% bootstrap intervals (2,000 draws, seed 123). Delta is treatment minus comparator; intervals are conditional on fitted weights.

| Comparison | Metric | Delta |95% CI |
| --- | --- | ---: | --- |
|A to CatBoost / original 17 features |recall@10_delta |-0.01311 |[-0.01780,-0.00833] |
|A to CatBoost / original 17 features |ndcg@10_delta |-0.00766 |[-0.01172,-0.00355] |
|A to CatBoost / original 17 features |new_recall@10_delta |-0.06031 |[-0.07564,-0.04582] |
|A to CatBoost / current 53 features |recall@10_delta |-0.01322 |[-0.01772,-0.00865] |
|A to CatBoost / current 53 features |ndcg@10_delta |-0.00711 |[-0.01044,-0.00377] |
|A to CatBoost / current 53 features |new_recall@10_delta |-0.02908 |[-0.04274,-0.01588] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |recall@10_delta |-0.00428 |[-0.00733,-0.00130] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |ndcg@10_delta |-0.00275 |[-0.00517,-0.00050] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |new_recall@10_delta |-0.01077 |[-0.01917,-0.00275] |
|fused_two_profile_32_to_split_text_attributes_32 |recall@10_delta |-0.00235 |[-0.00470,-0.00011] |
|fused_two_profile_32_to_split_text_attributes_32 |ndcg@10_delta |-0.00108 |[-0.00291,+0.00074] |
|fused_two_profile_32_to_split_text_attributes_32 |new_recall@10_delta |-0.00431 |[-0.01024,+0.00163] |
|split_text_attributes_32_to_split_text_attributes_64 |recall@10_delta |-0.00030 |[-0.00407,+0.00348] |
|split_text_attributes_32_to_split_text_attributes_64 |ndcg@10_delta |+0.00058 |[-0.00223,+0.00337] |
|split_text_attributes_32_to_split_text_attributes_64 |new_recall@10_delta |+0.01023 |[+0.00000,+0.02050] |
|A 32_to_A64 |recall@10_delta |-0.00488 |[-0.00827,-0.00164] |
|A 32_to_A64 |ndcg@10_delta |-0.00226 |[-0.00474,+0.00015] |
|A 32_to_A64 |new_recall@10_delta |-0.00646 |[-0.01540,+0.00213] |
|A 64_to_split64 |recall@10_delta |+0.00029 |[-0.00341,+0.00415] |
|A 64_to_split64 |ndcg@10_delta |+0.00010 |[-0.00236,+0.00255] |
|A 64_to_split64 |new_recall@10_delta |+0.00592 |[-0.00420,+0.01599] |
|A 32_to_split64 |recall@10_delta |-0.00459 |[-0.00836,-0.00114] |
|A 32_to_split64 |ndcg@10_delta |-0.00216 |[-0.00459,+0.00039] |
|A 32_to_split64 |new_recall@10_delta |-0.00054 |[-0.01034,+0.00913] |

### DISCOVERY-selected weights

| Model | Epoch/trees | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | Repeat micro R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 MLP on identical slate | fixed | 0.64696 | 0.75845 | 0.85096 | 0.89877 | 0.94010 | 0.58822 | 0.56685 |206/1,857 |0.88740 |
| A / fused profile /32 events | 8 | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 |255/1,857 |0.89298 |
| CatBoost / original 17 features | 350 | 0.65229 | 0.75759 | 0.84557 | 0.89693 | 0.94010 | 0.59363 | 0.57723 |144/1,857 |0.89298 |
| CatBoost / current 53 features | 200 | 0.64971 | 0.75709 | 0.84787 | 0.90010 | 0.94010 | 0.59299 | 0.57445 |206/1,857 |0.88514 |
| Two fused profile tokens /32 events | 6 | 0.65767 | 0.76911 | 0.85814 | 0.90302 | 0.94010 | 0.60056 | 0.58170 |243/1,857 |0.89155 |
| Separate text+attributes /32 events | 5 | 0.65798 | 0.77002 | 0.85798 | 0.90297 | 0.94010 | 0.59896 | 0.57846 |246/1,857 |0.89346 |
| Separate text+attributes / 64 events | 9 | 0.65717 | 0.76646 | 0.85731 | 0.90213 | 0.94010 | 0.60007 | 0.58217 |254/1,857 |0.88847 |
| A / fused profile / 64 events | 7 | 0.65977 | 0.76761 | 0.85800 | 0.90383 | 0.94010 | 0.59896 | 0.57917 |249/1,857 |0.89037 |

Paired customer-clustered95% bootstrap intervals (2,000 draws, seed 123). Delta is treatment minus comparator; intervals are conditional on fitted weights.

| Comparison | Metric | Delta |95% CI |
| --- | --- | ---: | --- |
|A to CatBoost / original 17 features |recall@10_delta |-0.01346 |[-0.01814,-0.00858] |
|A to CatBoost / original 17 features |ndcg@10_delta |-0.00860 |[-0.01254,-0.00458] |
|A to CatBoost / original 17 features |new_recall@10_delta |-0.05977 |[-0.07433,-0.04565] |
|A to CatBoost / current 53 features |recall@10_delta |-0.01395 |[-0.01864,-0.00929] |
|A to CatBoost / current 53 features |ndcg@10_delta |-0.00924 |[-0.01294,-0.00571] |
|A to CatBoost / current 53 features |new_recall@10_delta |-0.02639 |[-0.03955,-0.01349] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |recall@10_delta |-0.00103 |[-0.00430,+0.00224] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |ndcg@10_delta |-0.00327 |[-0.00580,-0.00095] |
|arm_a_control_pointwise_425d_to_split_text_attributes_32 |new_recall@10_delta |-0.00485 |[-0.01388,+0.00380] |
|fused_two_profile_32_to_split_text_attributes_32 |recall@10_delta |+0.00090 |[-0.00174,+0.00373] |
|fused_two_profile_32_to_split_text_attributes_32 |ndcg@10_delta |-0.00160 |[-0.00377,+0.00044] |
|fused_two_profile_32_to_split_text_attributes_32 |new_recall@10_delta |+0.00162 |[-0.00499,+0.00885] |
|split_text_attributes_32_to_split_text_attributes_64 |recall@10_delta |-0.00356 |[-0.00760,+0.00051] |
|split_text_attributes_32_to_split_text_attributes_64 |ndcg@10_delta |+0.00111 |[-0.00182,+0.00393] |
|split_text_attributes_32_to_split_text_attributes_64 |new_recall@10_delta |+0.00431 |[-0.00623,+0.01562] |
|A 32_to_A64 |recall@10_delta |-0.00343 |[-0.00635,-0.00037] |
|A 32_to_A64 |ndcg@10_delta |-0.00327 |[-0.00560,-0.00082] |
|A 32_to_A64 |new_recall@10_delta |-0.00323 |[-0.01237,+0.00559] |
|A 64_to_split64 |recall@10_delta |-0.00115 |[-0.00509,+0.00250] |
|A 64_to_split64 |ndcg@10_delta |+0.00111 |[-0.00154,+0.00369] |
|A 64_to_split64 |new_recall@10_delta |+0.00269 |[-0.00709,+0.01258] |
|A 32_to_split64 |recall@10_delta |-0.00459 |[-0.00836,-0.00114] |
|A 32_to_split64 |ndcg@10_delta |-0.00216 |[-0.00459,+0.00039] |
|A 32_to_split64 |new_recall@10_delta |-0.00054 |[-0.01034,+0.00913] |

## What the separated memory contains

- A: one 128D fused company profile token plus 32 historical event tokens.
- Token-count control: two 128D copies of the fused profile plus 32 events.
- Split 32: company-text 128D token, company-attribute 128D token, plus 32 event tokens.
- A 64: the original one fused profile token, plus 64 separate event tokens (65 tokens total).
- Split 64: the same two profile tokens, plus 64 separate event tokens. All 66 memory tokens pass through self-attention and remain available to each candidate’s cross-attention. There is no global older-history attention-pooling step.
- The two-token arms assign profile 8 statistics to the attribute token and zero statistics to the text token. Both profile tokens are always valid; each history position follows its event validity mask. Attention diagnostics aggregate both profile weights correctly.
- The 425D scoring head, 41 raw candidate features, 13 candidate bias features and frozen candidate vectors remain unchanged. Candidate-to-candidate self-attention and prior 128D statistical enrichers remain disabled.
- Numerical profile/history inputs retain the reference bias-only routing. Separating the company attribute representation does not silently enable the previously tested statistical enrichers.
- Before self-attention, the text token depends on company text and the attribute token on city/region/industry. Self-attention is then allowed to mix these views. This is source separation, not a guarantee of orthogonal or equally useful embeddings.
- The split bypasses and freezes the old profile-fusion projection. The two-fused-token control accounts for token count/statistics placement but does not make every parameter or optimization path identical.
- Product text and structured product attributes could be separate views. This run leaves the candidate product vector unchanged to isolate customer memory. Product text includes overlapping category/brand/family semantics; raw cosine/max operations between separately projected branches are not automatically meaningful.

## 64-event data checks

The original training-fitted vocabularies and quantity reference 60 are reused. All events are strictly before their query date. Every original 32-event suffix was verified exactly, including event statistics, dates and categorical IDs, before the 64-event model was trained. Queries, targets, slate selection and rolling candidate features do not change. The existing learned 32-position embedding is linearly interpolated for 64 positions, preserving the parameter layout. The 64-event configuration includes this interpolated position mapping; recent raw features are unchanged, but their positional representation changes. It does not test every possible long-history encoding or change the number of distinct baskets.

```json
{
  "train": {
    "queries": 26253,
    "queries_with_more_than32_events": 8510,
    "mean_events": 25.600502014160156,
    "exact32_suffix_parity": true
  },
  "val": {
    "queries": 6191,
    "queries_with_more_than32_events": 3211,
    "mean_events": 36.67388153076172,
    "exact32_suffix_parity": true
  }
}
```

## Completed corrections and limits

The user confirmed earlier repeat/new objective-weighting experiments did not improve results. That sweep was not repeated, and its user-reported status is now recorded in plan.md. The current matched fits retain the original loss weighting.

The new optional core profile/statistics/diagnostic hooks preserve legacy state keys and default arithmetic. Saved ArmA CPU replay still matches its saved MPS logits within 1.91e-6. Snapshot/hash verification preserves all 31 inventoried reference/rolling evidence files. The earlier masked-loss fix remains in place. Production ingestion/automatic refresh, source-coherent external inference snapshots, and effective-dated availability remain separate work.

A single training seed is exploratory. Repeated validation use and differing checkpoint selection/feature sets constrain generalization claims. No model was promoted.

## Evidence and reproduction

- [CatBoost summary](../catboost_current_v1/summary.json)
- [CatBoost historical validation metadata](../catboost_current_v1/historical_validation.json)
- [CatBoost common ranking selections](../catboost_current_v1/selected_checkpoints.json)
- [Separate-profile/64-event summary](../separate_profile_v1/summary.json)
- [Interactive comparison](comparison.html)

```bash
./venv/bin/python -m recommender.experiments.catboost_comparison.run --execute-training
./venv/bin/python -m recommender.experiments.catboost_comparison.select
./venv/bin/python -m recommender.experiments.separate_profile.train --execute-training
./venv/bin/python -m recommender.experiments.separate_profile.fused64 --execute-training
./venv/bin/python -m recommender.experiments.separate_profile.report
PYTHONPATH=. ./venv/bin/pytest tests/ -q
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
```


## Verification

All 12 selected checkpoint records reproduce the saved metrics. Neural CPU replays on 32 validation rows match saved MPS logits within 2.62e-06; CatBoost selected models reproduce the complete validation metrics after reload.

Full safe suite: 569 passed, 6 skipped. Synthetic-only: 559 passed, 16 skipped.

[Independent verification record](verification.json)
