# Later review: disposition during the active campaign

The additional pasted review was read while the explicitly authorized P0–P7 campaign was running. Its recommendations are review proposals, not an instruction to discard the accepted campaign or repeat already completed infrastructure work.

| Review item | Verified disposition |
|---|---|
| Fixed-cutoff demand, weak base cache checks, incomplete provenance and stale aliases | Already addressed for new base training in `3fbec83`; historical artifacts intentionally retain their old semantics. New campaign inputs use demand-disabled S0 and separately versioned/checksummed features and run manifests. |
| Ignored clipping bounds, identity-MLP interception, bypassed ProductTower, repeated snapshots, duplicate result file, missing export and CI | Already addressed in `3fbec83`. The old duplicate file is gone. Linux CI for that commit was independently confirmed successful, including dependency installation. |
| Metadata has no effective-date history | Still unresolved as data provenance. P1 measures zero brand/subcategory/industry features in strict purchase-observed mode and labels the primary snapshot assumption. No historical metadata is invented. |
| Arbitrary within-basket SKU ordering | Preserved in the unchanged upstream baseline; basket-aware encoding remains a separate behavioral experiment, outside P0–P7. |
| Numerical statistics excluded from memory values | Independent bias-only / values-only / both experiments are included. The default still uses its original route. |
| Candidate self-attention | P5 includes direct, learned customer-dot and zero-initialized residual heads, one 128D four-head layer, no absolute candidate positions, scalar/gradient/padding/permutation tests. A separate matched 10+10-epoch study tests the review's pretrained residual proposal. |
| Cheap pointwise model on the same 50-item slates | Added as a separate matched-slate control in `pointwise_control.py`; it uses cached embeddings and the original 13 pairwise features. |
| More unseen-positive loss sweeps | Not added as a new primary grid: the accepted later architecture plan deprioritizes this after the user's unsuccessful 2× trial. All comparisons retain matched losses. |
| Optional CatBoost, initial-rank features, full PRM replication | Suggestions rather than requirements of the accepted matrix; no additional library or positional shortcut is introduced. |
| Candidate depth and replacement of dot scoring | P2/P6 evaluate full eligible-pool cheap scoring and fixed 50/100 quotas independently from P3–P5 fixed-slate component experiments. Replacing candidate scoring never removes eligibility-constrained shortlisting. |

No experimental model is promoted. The completed fits, complete-denominator metrics, paired repeats and deployment costs support the recommendation in `DYNAMIC_SELECTOR_EXPERIMENTS.md`; architectural plausibility alone did not determine it.
