# Latest Decision: Promoted as `repeat-no-month-v1` by Explicit User Override (2026-11-13)

> **User Override & Active Promotion (2026-11-13):** Following review of the comparison below, the user explicitly overrode the non-promotion recommendation with: *"okay, save the Repeat restored, 301D as the new reference, commit and push"*.
> - **New Canonical Active Pointer:** [`experiments/stage1_mlp_reference/reference.json`](../stage1_mlp_reference/reference.json) ([`REFERENCE.md`](../stage1_mlp_reference/REFERENCE.md))
> - **Immutable Reference Directory:** `checkpoints/stage1-mlp-reference/repeat-no-month-v1/` ([`manifest.json`](../../checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json)) — `best_retriever.pt` (epoch 16, SHA-256 `2ad87d114731015d9a4ddec992d5425b776774a962b6f365da6da6b87182c981`) + `best_selector.pt` (epoch 5, SHA-256 `977241cf9fe8a36ae5c0b5afede6c8018e55eb3202e39d88fa7e3693bc0bc22a`).
> - **Scope & Historical Preservation:** Active reference scope is Stage 1 `FeatureAblatedStage1Selector` MLP work only (`validation_only=true`, `heldout_access=false`, `stage2_executed=false`). Explicit user selection accepts the reported tradeoffs (higher NDCG@10 `0.586331`, Recall@5 `64.335%`, unseen@100 `1,337`; lower Recall@10 `75.411%`, ordinary top-10 discovery `194`, unseen@25/`@50` `726`/`1,029` vs `no-brand-v1`) on single `seed=42`; do not claim statistical superiority. Prior campaign artifacts ([`summary.json`](summary.json), `checkpoints/mlp-repeat-restored-v1/candidate/manifest.json` with `promoted: false`, and frozen [`experiments/embedding_ablation_v1/reference.json`](../embedding_ablation_v1/reference.json)) are preserved unchanged as historical evidence.

---

## Historical Assessment (Prior to User Override): retain no-brand-v1

The repeat-restored, month-free 301-input candidate was evaluated but not promoted on November 13, 2026. The user authorized saving a reference and committing/pushing if the results were appropriate. The results show a tradeoff rather than a clear improvement in the requested overall recall and discovery measures.

Compared with the active 303-input reference, overall Recall@10 falls from 75.65594% to 75.41144% (−0.24450 percentage points; customer-clustered 95% CI −0.676 to +0.152). New targets in the ordinary top 10 fall from 214 to 194 (20 fewer, approximately 9.35% fewer hits), and unseen top-50 hits fall from 1,037 to 1,029. NDCG@10 improves from 0.578286 to 0.586331. Unseen top-100 hits improve from 1,324 to 1,337.

This is one seed with selected checkpoints, and it does not establish a general recall regression or repeatable superiority. A higher NDCG alone does not justify replacing the saved reference given lower overall Recall@10 and fewer top-10 discovery hits. The immutable candidate is preserved for future comparison, while the active pointer remains no-brand-v1. No commit or push was performed because the condition for those actions was not met.

The report uses the observed metric differences as point estimates, with bootstrap quantiles supplying the confidence interval endpoints. The bootstrap medians remain available in summary.json.
