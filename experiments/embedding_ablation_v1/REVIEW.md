# Review of the embedding-first campaign

All 12 independent baseline/ablation results were inspected after Gemini completed. The seven historical audit artifact hashes and all six implementation source hashes recorded in summary.json match.

## Interpretation

Removing the brand categorical embedding is the most promising single-seed result: unseen hits@50 rise from 1,012 to 1,037, while brand remains available in product text and scalar features. Its nominal paired customer-bootstrap interval for the recall change is +0.16 to +2.63 percentage points. This interval is conditional on the fitted models and selected epochs; it does not include training-seed variation, epoch/variant selection uncertainty, or adjustment for the eleven deletion comparisons. Treat this as an exploratory candidate, not a confirmed promotion.

All six text deletion intervals for unseen Recall@50 include zero. Removing category or unit embeddings improves overall NDCG@10 in these selected fits but has discovery tradeoffs. Different selected epochs are part of these model-level comparisons; an isolated causal explanation for their NDCG changes has not been demonstrated.

## Original Antigravity conversation

The T3 thread reader could not resolve ff8dc551-a440-40b3-857f-def28e234ff3, but its local saved user and assistant messages were found and read after the delegate finished. Relevant instructions confirm: sales_branch is required in the customer tower; the six changing customer statistics belong in the scoring MLP; product demand statistics should be excluded from static product vectors; t1_fam omits descriptions and includes product family in text. The chosen U0_t1_fam_clean baseline satisfies those requirements.

Only user-visible messages were used for this review. The local conversation transcript is outside the repository and is not copied into these experimental artifacts.

## Verification recorded in the completed delegate

- Normal suite: 506 passed, 6 skipped.
- Synthetic-only suite: 499 passed, 13 skipped.
- No Stage 2 fits/evaluation or held-out test evaluation in this campaign.
- Existing checkpoint aliases are preserved; no model was promoted.
- Previously agreed numeric feature and normalization changes remain deferred.
