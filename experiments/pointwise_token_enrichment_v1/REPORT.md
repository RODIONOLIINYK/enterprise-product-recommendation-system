# Pointwise Stage 2 with 128D statistical token enrichment

Only Arm D was trained in this follow-up: seed 42, 10 epochs, batch size 64, AdamW, MPS. Arms A-C are read-only results from the preceding matched experiment. No held-out access, reference promotion, commit or push occurred.

## Result

The enrichment-only model selects epoch 8 for both criteria. It does not improve over A: Recall@10 is 0.76885 versus 0.77104, NDCG@10 is 0.60034 versus 0.60223, and new hits@10 are 248 versus 255. D has slightly higher Recall@25 (0.85978 versus 0.85816). The paired intervals for Recall@10, NDCG@10 and new recall@10 versus A all include zero. This run supplies no evidence to replace A; it does not establish that enrichment is universally worse. D is smaller/faster than C but slower than A.

## Architecture and controlled inputs

D uses the same three pre-attention residual MLPs as C: candidate `[128 + 41] -> 128 -> 128`, profile Token 0 `[128 + 8] -> 128 -> 128`, and history Tokens 1-32 `[128 + 9] -> 128 -> 128`. Candidate-set attention is disabled and frozen. Customer-memory self-attention and candidate-to-memory cross-attention remain active. The original 425D scoring head and bilinear scalar bias remain unchanged. Stage 1 MLP scores are excluded.

Common initial weights match A exactly; enrichment initialization matches C. All residual enrichers initially output zero, giving exact initial logit parity with A. Candidate statistics use the same [0,1] enrichment normalization as C, while legacy raw scalar/bias paths and signed latent vectors retain their previous representation.

All arms use the same sealed `rolling_observed_v2` `quota100_50` slates: 26,253 training / 6,191 validation queries, 10,276 total validation targets, 1,857 new targets. The follow-up refuses mismatched observed-source, feature, score, upstream or slate identities. A-C selected checkpoints, logits, prediction arrays and histories were verified before reuse; all 31 reference/rolling-cache evidence hashes and prior selected artifacts remained unchanged.

## Validation metrics

Discovery checkpoints maximize new hits@10 subject to the same Recall@10/NDCG@10 guardrails. NDCG checkpoints maximize NDCG@10 under the original ordering. Results below are complete-basket macro recalls; new recall uses all 1,857 new target instances, including missing-slate targets.

### NDCG-selected checkpoints

| Arm | Epoch | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | NDCG@10 | MRR | New hits@10 | Repeat micro R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Pointwise control | 8 | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 | 255/1,857 | 0.89298 |
| B · Candidate attention | 9 | 0.65307 | 0.76744 | 0.85725 | 0.90077 | 0.94010 | 0.59890 | 0.57919 | 252/1,857 | 0.88775 |
| C · Attention + enrichment | 9 | 0.66033 | 0.76647 | 0.85867 | 0.90248 | 0.94010 | 0.60012 | 0.58102 | 239/1,857 | 0.88811 |
| D · Pointwise + enrichment | 8 | 0.65729 | 0.76885 | 0.85978 | 0.90295 | 0.94010 | 0.60034 | 0.58091 | 248/1,857 | 0.89084 |

Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. D minus the named comparator; these intervals do not measure training-seed variation.

| Comparator | Metric | Delta | 95% CI |
| --- | --- | ---: | --- |
| A · Pointwise control | recall@10_delta | -0.00219 | [-0.00531, +0.00087] |
| A · Pointwise control | ndcg@10_delta | -0.00189 | [-0.00438, +0.00058] |
| A · Pointwise control | new_recall@10_delta | -0.00377 | [-0.01226, +0.00480] |
| B · Candidate attention | recall@10_delta | +0.00141 | [-0.00290, +0.00573] |
| B · Candidate attention | ndcg@10_delta | +0.00144 | [-0.00186, +0.00463] |
| B · Candidate attention | new_recall@10_delta | -0.00215 | [-0.01249, +0.00879] |
| C · Attention + enrichment | recall@10_delta | +0.00238 | [-0.00104, +0.00583] |
| C · Attention + enrichment | ndcg@10_delta | +0.00022 | [-0.00198, +0.00258] |
| C · Attention + enrichment | new_recall@10_delta | +0.00485 | [-0.00464, +0.01313] |

### DISCOVERY-selected checkpoints

| Arm | Epoch | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | NDCG@10 | MRR | New hits@10 | Repeat micro R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Pointwise control | 8 | 0.66012 | 0.77104 | 0.85816 | 0.90476 | 0.94010 | 0.60223 | 0.58329 | 255/1,857 | 0.89298 |
| B · Candidate attention | 9 | 0.65307 | 0.76744 | 0.85725 | 0.90077 | 0.94010 | 0.59890 | 0.57919 | 252/1,857 | 0.88775 |
| C · Attention + enrichment | 6 | 0.65956 | 0.77100 | 0.85906 | 0.90267 | 0.94010 | 0.59963 | 0.57879 | 255/1,857 | 0.89191 |
| D · Pointwise + enrichment | 8 | 0.65729 | 0.76885 | 0.85978 | 0.90295 | 0.94010 | 0.60034 | 0.58091 | 248/1,857 | 0.89084 |

Paired customer-clustered bootstrap intervals: 2,000 draws, seed 123. D minus the named comparator; these intervals do not measure training-seed variation.

| Comparator | Metric | Delta | 95% CI |
| --- | --- | ---: | --- |
| A · Pointwise control | recall@10_delta | -0.00219 | [-0.00531, +0.00087] |
| A · Pointwise control | ndcg@10_delta | -0.00189 | [-0.00438, +0.00058] |
| A · Pointwise control | new_recall@10_delta | -0.00377 | [-0.01226, +0.00480] |
| B · Candidate attention | recall@10_delta | +0.00141 | [-0.00290, +0.00573] |
| B · Candidate attention | ndcg@10_delta | +0.00144 | [-0.00186, +0.00463] |
| B · Candidate attention | new_recall@10_delta | -0.00215 | [-0.01249, +0.00879] |
| C · Attention + enrichment | recall@10_delta | -0.00215 | [-0.00549, +0.00124] |
| C · Attention + enrichment | ndcg@10_delta | +0.00071 | [-0.00164, +0.00299] |
| C · Attention + enrichment | new_recall@10_delta | -0.00377 | [-0.01187, +0.00431] |

## Latency

Median prepared-input forward time in ms. Measured again for all four selected NDCG models in the same benchmark; excludes feature extraction, Stage 1 retrieval and disk. MPS timings include the synchronization cost of runtime input guards.

| Arm | CPU batch 1 | CPU batch 64 | MPS batch 1 | MPS batch 64 | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Pointwise control | 0.650 | 10.679 | 3.082 | 6.715 | 671,497 |
| B · Candidate attention | 1.048 | 18.976 | 3.243 | 9.208 | 891,530 |
| C · Attention + enrichment | 1.183 | 22.271 | 9.984 | 16.824 | 998,794 |
| D · Pointwise + enrichment | 1.005 | 14.230 | 9.162 | 14.018 | 778,761 |

## Correctness review response

- **Masked RankerLoss defect confirmed and fixed.** Invalid entries are excluded before BCE and softmax arithmetic. BCE is averaged over valid entries; listwise loss is averaged over rows with valid positives. All-masked rows contribute differentiable zero. Masked positive labels are rejected, and invalid-candidate gradients are zero. The six masked-ranker training/evaluation call sites now pass the candidate mask. Regression tests cover NaN/-inf padding, all-masked rows, no positives, variable lengths, unseen weighting, and exact all-valid value/gradient parity. Historical all-valid slate results are unaffected.
- **Upstream snapshot coherence remains a production prerequisite.** This experiment verifies the same sealed source/upstream artifacts as A-C. It does not implement automatic refresh after ingestion or prove arbitrary external vectors share the feature snapshot. A complete inference snapshot contract remains separate work.
- **Legacy frozen defaults and historical availability limits remain.** New experiment entry points explicitly select rolling statistics. Historical `CandidateRetriever.from_checkpoint` still uses the compatibility default; catalogue last-modified timestamps still cannot reconstruct effective-dated blocking history. Neither policy was silently changed.
- **The review's candidate-attention training status is outdated relative to local work.** A-C completed ten-epoch fits before this follow-up. Their original evidence remains at [the three-arm report](../candidate_token_enrichment_v1/REPORT.md).
- Broad data-module refactoring, feature-preparation optimization, documentation cleanup and basket-aware modeling were not mixed into this architecture comparison.

## Limits and artifacts

This is one exploratory training seed. Selected-checkpoint inference perturbations measure dependence on a fitted module, not the isolated causal value of statistics. Frozen towers and snapshot-unverified metadata remain unchanged. Do not infer final-test performance or promote a model solely from one metric.

- [Machine-readable summary](summary.json)
- [Interactive comparison](comparison.html)
- [Latency measurements](benchmark.json)
- New checkpoints: `checkpoints/pointwise-token-enrichment-v1/runs/`.

Reproduction:

```bash
./venv/bin/python -m recommender.experiments.candidate_token_enrichment.pointwise_enrichment --execute-training --device mps
./venv/bin/python -m recommender.experiments.candidate_token_enrichment.pointwise_report
PYTHONPATH=. ./venv/bin/pytest tests/ -q
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q
```


## Independent verification

All 8 selected checkpoint records were hash-verified and their saved logits reproduced the ranking metrics. CPU replay on 32 rows per record matched saved MPS logits within 1.91e-06. All 31 frozen reference/rolling cache evidence hashes matched. No additional training or held-out access occurred.

- Full safe suite: 560 passed, 6 skipped.
- Synthetic-only suite: 550 passed, 16 skipped.
- Masked-loss CPU regression and manual MPS check: finite losses/gradients, zero padding gradients.
- [Independent verification record](independent_verification.json)
