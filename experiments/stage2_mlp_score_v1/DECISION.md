# Stage 2 MLP score excluded — 2026-11-13

The user explicitly selected exclusion of the Stage 1 MLP score from Stage 2 after reviewing this experiment and instructed that training must not run yet.

- Exclude the per-item MLP score from both the transformer cross-attention bias and final scoring head in subsequent Stage 2 work.
- New `Stage2MLPScoreRanker` instances default to `score_channel_enabled=False`; the standard Stage 2 path already has no MLP-score input.
- Retain the existing Stage 2 reference and the selected `repeat-no-month-v1` 301D Stage 1 MLP reference. The MLP can still select candidates; its score is excluded as a Stage 2 input.
- Preserve the completed experimental results, caches and checkpoints as historical evidence. Historical checkpoints load their saved explicit channel setting, and the isolated comparison runner retains explicit treatment settings for reproduction.
- No model fitting, retraining or checkpoint promotion is part of this exclusion change.

The measured score-enabled variant reduced overall Recall@10, NDCG@10 and new-product hits@10 relative to its matched control. See [REPORT.md](REPORT.md) for the historical validation evidence.
