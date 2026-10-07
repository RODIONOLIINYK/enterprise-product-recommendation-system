# Implementation and training plan

## Objective and current state

Build a local next-basket recommendation system: **two-tower retrieval → hybrid cross-encoder ranking → top-K products**. Train on the Apple Silicon Mac with 16 GB of memory. Preserve exact sellable product IDs, including separate package variants. Default to 100 retrieval candidates and 10 final recommendations.

This is a future implementation plan. The current branch implements actual-purchase preparation, item-catalogue loading, shared historical product eligibility and their tests. It does not yet implement company metadata features, neural models, training or recommendation serving. Do not present the old model's results as results for this approach.

The ranker must receive the **same complete company and product input features** as the towers, at the same scoring time. It receives the original selected purchase sequence and structured fields, not merely two final retrieval embeddings. Its text branch processes company and candidate text together, its history branch processes the identical event sequence, and its scoring head combines those outputs with every categorical and numeric feature.

## 1. Reuse and verify the useful existing preparation code

The former `deployment_pipeline/utilities.py` and cleaning notebook already had useful source mappings, string normalization, paid-sale filtering, company/date/product aggregation and validation. These are now adapted in `recommender/data.py`; the notebook calls these helpers rather than duplicating their implementation.

Keep these behaviours:

- Detect the workbook header in the first 20 rows, including the existing fix for a blank first row.
- Read identifiers as strings, strip surrounding whitespace, parse purchase dates and quantities, and normalize dates to calendar days.
- Retain positive finite ordinary-sale records (`Gen_ Bus_ Posting Group = ПРОДАЖА`) with valid company/product IDs and dates. Require the sales classification `Gen_ Prod_ Posting Group = ТОВАР` and membership in the item master's for-sale (`ТОВАР`) products. IDs, brands and category names are not evidence of a sale.
- Exclude gifts (`ПОДАРОК`), returns and accounting rows before deriving history, order dates, cadence, popularity or labels. A gift-only date creates no purchase basket or target. For mixed dates, keep only actual paid-sale products.
- Aggregate repeated lines only by company, date and exact product ID. Preserve raw quantity units and separate package variants.
- Validate unique event keys, unchanged product IDs, positive quantities and quantity preservation per event and in total.

Run `python -m unittest discover -s tests -v`, then execute `notebooks/01_clean_purchases.ipynb` top to bottom. It writes the ignored local `data/interim/cleaned_purchases.csv`. The notebook also reads `data/raw/items.csv`. Shared eligibility is configured in `configs/product_scope.toml`: candidates must be classified as for sale and have at least one genuine purchase strictly before the scoring time. The original product-ID-prefix heuristic is removed. Missing optional product descriptions/categories do not erase a genuine purchase.

Do not reuse the old package-ID remapping, old 25-candidate training tables, old labels anchored to the current purchase event, old fitted-model metrics, or model-specific scoring formulas. The old historical-feature notebook is backed up locally: its strictly-prior counting and cumulative-history ideas are useful references, but its training table is not an input to the new system.

**Completion:** exact product IDs and per-event quantities match eligible workbook lines; source files remain unchanged; published notebooks have no execution outputs.

## 2. Set up a reproducible local training environment

Use Python 3.12. Retain the existing environment, but create a separate ignored `.venv` for neural training so installing the new stack does not change that environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The direct dependencies are:

| Library | Role |
| --- | --- |
| pandas, numpy, openpyxl | Source loading and feature calculations |
| pyarrow | Parquet preparation and feature tables |
| torch | Towers, history encoder, hybrid ranker, optimizers and data loaders |
| transformers | Joint text encoder and tokenizer |
| sentence-transformers | Frozen standalone multilingual text embeddings |
| scikit-learn | Scaling and diagnostic metrics |
| pytest | Automated checks; existing unittest tests remain compatible |
| ipykernel | Execute notebooks in the selected environment |

Check `torch.backends.mps.is_available()` and use MPS if available, otherwise CPU. Use float32 initially and `num_workers=0` for local data loaders. Run a forward/backward smoke test for each actual model on the selected device before starting a long run. Use seed 42 for sampling and initialization; record device, versions and seed without promising identical floating-point results across devices.

After imports and device tests pass, create an ignored exact-version environment lock under `private/` and record the resolved package versions in each experiment's manifest. No model downloads, installation or training are part of the current cleanup task.

**Completion:** all planned dependencies import, device smoke tests pass, and the resolved environment is recorded.

## 3. Prepare company and product metadata

Create company enrichment in `recommender/metadata.py` and a `prepare_metadata` module command. Reuse the implemented `recommender.catalogue.load_product_catalogue` for item loading and its validated identities/classification; do not create a second item parser. Read the company CSV with `sep=";"`, `header=None`, `dtype="string"` and `encoding="utf-8-sig"`. Normalize whitespace and treat empty strings and literal `NULL` as missing. Define the column mappings centrally rather than scattering numeric indices.

The inspected export positions below are **one-based**. Meanings are inferred from values and checked joins; record that status rather than claiming a provided source schema exists.

| Entity | Columns to map |
| --- | --- |
| Company | ID 2; activity notes 91; address 89; city-like code 99 |
| Product | ID 2; short name 4; generic product type 6; unit 7; for-sale accounting classification 51; business line 60; brand-like field 84; category 85; finer category 87; Ukrainian name 122; family-like field 126; expanded name 133; base name 134 |

Use column 2 as the join identity, not company names or an unverified alias column. Validate unique keys and many-to-one joins against exact purchase IDs. Report unmatched records; retain purchases with missing optional metadata using unknown/missing values rather than silently losing events.

Use the city code as a categorical location identifier. Populate readable city/region values only through a reviewed deterministic mapping or reliable address parsing; unresolved locations remain unknown. Do not reinterpret sales-office codes as company cities. External geocoding or company enrichment is not required.

Build product text from available expanded/Ukrainian names and type, adding brand and technical attributes only when present. Deduplicate repeated text and preserve model numbers and specifications. Extract explicit package amount/unit and viscosity grades such as `5W-40` from names; leave unrecognized specifications missing. Use the base name/family as a relationship feature, never as a replacement sellable ID. No long product description is assumed to exist.

Historical purchase names/categories remain available from purchase events. Current company notes and master-data fields are snapshot enrichment: use them as approximate historical profiles, as selected, record the caveat, and include a notes-free experiment. Store missingness indicators and never invent company descriptions.

Write ignored `data/interim/companies.parquet` and `products.parquet`, with an input-hash/column-map manifest. Use `select_eligible_products` at the recorded preparation cutoff to keep only for-sale products with actual prior sales in the prepared product table. Reapply the rule at every earlier historical scoring time; eligibility at the end of the dataset must not leak into earlier examples. The full item master is read only to validate and classify identities, not directly as a training or inference catalogue.

**Completion:** joins cannot multiply events; all optional-field gaps and inferred mappings are recorded; package variants remain distinct.

## 4. Build scoring snapshots and next-basket labels

Create `recommender/examples.py`. Generate Monday-midnight scoring dates within the observed purchase period. A company becomes eligible for a snapshot after its first observed purchase. Features use records with `purchase_date < scoring_time`; the label is the set of exact products in that company's first observed basket on or after the snapshot.

Companies with no observed future basket are censored at that snapshot: exclude them from supervised labels rather than treating every product as a negative. Retain all same-date genuinely purchased products as a single multi-positive basket. Build dates and targets from the filtered actual-purchase history, never raw movement dates: receiving a present is not an order. Weekly snapshots may precede a later paid basket, but a gift itself is never their outcome. Last-basket and first-appearance rules must be explicit in the example manifest.

Use these fixed temporal splits:

- Training: snapshots and target baskets before January 1, 2026.
- Validation: both dates from January 1 through March 31, 2026.
- Test: both dates from April 1 through July 17, 2026, the observed data endpoint.

Exclude examples that cross split boundaries. Split examples before fitting vocabularies, scalers or model parameters. A validation/test company's prior purchase history is usable at scoring time; future purchases are never features.

Use the implemented `select_eligible_products(catalogue, actual_purchase_history, scoring_time)` for every training, validation and test catalogue. A candidate must be `ТОВАР` in the item master and have at least one actual `ПРОДАЖА` purchase strictly before that snapshot. Gifts, zero quantities and future sales never satisfy this condition. Keep a product's first real sale in factual history so it becomes eligible at later cutoffs, but never insert that product into an earlier candidate list. A newly sold product outside a preceding snapshot's catalogue is an unretrievable outcome: include it in end-to-end recall and eligibility-coverage reporting. Train only on eligible positives, skip groups with none, and report skipped groups. This policy is not evidence of stock availability; add a separate stock constraint only when reliable inventory data is supplied.

Write `data/processed/examples.parquet` with example ID, company ID, scoring time, target-basket date, positive product IDs and split. IDs/dates organize examples; actual target dates and quantities never enter the model.

**Completion:** every target is a later observable basket, no target crosses its split, and excluded/unretrievable outcomes are counted.

## 5. Implement one shared point-in-time feature builder

Create `recommender/features.py`. Both stages use the following interfaces; do not implement a separate ranker feature formula:

```python
build_company_features(company_id, scoring_time)
build_product_features(product_id, scoring_time)
build_feature_pair(company_id, product_id, scoring_time)
```

Use `actual_purchase_mask` and the prepared classification-preserving history for purchase evidence; use `select_eligible_products` for both retrieval and ranker candidates. These eligibility functions already exist in `recommender.scope` / `recommender.catalogue`. Do not maintain independent training/inference filters. The history schema retains `item_type` and `transaction_type` so inference can verify paid-sale evidence rather than treating arbitrary movements as purchases.

The pair builder returns the exact company and product bundles. Version the schema, preprocessing, product-scope policy hash, input hashes and cutoff together. The company ID is a lookup key, not a learned company-ID feature; the model must work from content/history for an unseen company.

### Company bundle

- Company activity text; city/region categorical values and missingness.
- Category, business-line and brand purchase-share vectors, with training-vocabulary unknown buckets. Count product events consistently for these shares; use a missing flag for empty history.
- Total prior basket count; basket counts in the preceding 30 and 90 days.
- Days since the last basket; median positive interval between distinct prior basket dates.
- Estimated days until the next basket: median interval minus days since the last basket. This may be negative. It is missing when no interval is observed.
- The most recent 32 prior product events, ordered by purchase date and product ID for deterministic same-date ties.

Each selected event includes purchased-product text/identity/attributes, quantity, unit, age in days, gap from the preceding distinct basket and sequence position. Products in the same basket share its basket gap; do not fabricate gaps between same-basket lines. Use all prior history for aggregate statistics even when the sequence is capped.

### Product bundle

- Product text, identity, brand, family, category, finer category and business line.
- Explicit technical specifications, package amount/unit, and base-product relationship fields.
- Lifetime purchase-event count; event counts over the preceding 30/90 days.
- Distinct purchasing-company count and days since the last prior purchase by any company.
- Missingness for absent metadata/history. Popularity counts genuine purchases only, never gifts, inventory movements, quantities or synthetic training pairs.

Windows are `[scoring_time - window, scoring_time)`. Derive all history and demand from events before the cutoff, including when scoring a previously purchased product. Implement cumulative date-indexed histories or grouped window calculations; avoid rescanning the full raw table per pair. These reuse the useful counting concepts from the old feature code.

Fit categorical vocabularies and numeric scales on training examples only. Apply `log1p` to nonnegative counts and quantities before standardization. Standardize signed days-until-order separately; preserve missing flags and replace missing numeric tensor values with zero after scaling. Keep per-product unit information beside quantities; do not add litres and kilograms into one company-volume feature.

**Completion:** adding future purchases changes no earlier feature; the tower and ranker loaders produce equal raw feature bundles for the same key/cutoff.

## 6. Cache multilingual text embeddings

Create `recommender/text.py`. Use `intfloat/multilingual-e5-small` through Sentence Transformers to produce normalized 384-dimensional standalone text embeddings. Follow the checkpoint's `query:` prefix for company text and `passage:` prefix for product text. Use a token limit of 384, text-encoding batches of eight and `eval()`/no gradients for this frozen step.

Encode each unique company text and each eligible, previously sold product text once, including product texts needed for actual historical purchase events. Never build candidate caches from never-sold or gift-only item records. Missing text uses an explicit missing-text flag and a zero cached vector. Cache keys include normalized text, checkpoint revision, prefix convention and token limit; changing any of these invalidates the cache.

Save local caches under `text_cache/`. These standalone vectors serve the towers and the history branch. They do **not** replace joint company–candidate encoding inside the ranker.

**Completion:** repeated unchanged text reuses its cache, dimensions are checked, and all embedding values are finite.

## 7. Implement and train the two towers

Create `recommender/models.py` and `train_retriever`. Both towers output L2-normalized 128-dimensional vectors.

**History encoder:** concatenate purchased-product text embeddings with learned ID/attribute embeddings and normalized event numbers, project to width 128, add position/time information, and use a two-layer Transformer with four heads, feed-forward width 256 and dropout 0.1. Pool only valid event positions. Use one unmasked learned missing-history token for an empty sequence so padding cannot produce NaNs.

**Company tower:** concatenate company text embedding, pooled history representation, city/region embeddings, preference vectors and all company numeric fields/missing flags. Apply a 256-unit GELU layer, dropout 0.1 and a 128-unit output projection.

**Product tower:** concatenate product text embedding, product-ID embedding (width 32), other categorical embeddings (width 16 each), technical/package fields, and every product numeric field/missing flag. Apply the same 256-to-128 projection structure.

Use width 16 for history attribute embeddings and width 32 for history product IDs. Keep the pretrained text encoder frozen. Train categorical embeddings, history encoder and tower projections. Unknown product IDs still retain product content and attributes.

For each scoring-date batch, compute vectors only for products returned by the shared for-sale-and-prior-purchase filter, using that date's demand features. Score company/product pairs by dot product divided by temperature 0.1. Use a multi-positive catalogue-softmax loss with uniform target mass over each company's eligible next-basket products; do not treat basket co-positives as negatives.

Training defaults: seed 42, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch up to 64 companies from the same scoring date, gradient clipping at 1.0 and a maximum of 10 epochs. Evaluate validation Recall@100 each epoch and stop after two epochs without improvement. Save the best checkpoint, not just the final one.

Record model configuration, schema/preprocessing versions, checkpoint revision, input hashes, epoch and metrics. Store weights locally under `checkpoints/retriever/`. Cache candidate input features by date, not trainable product vectors across optimizer updates.

**Completion:** the model can overfit a small synthetic batch, validation retrieval runs on the actual eligible catalogue, and the best checkpoint reloads with identical scores within floating-point tolerance.

## 8. Construct ranker training pairs

Create `recommender/candidates.py`. Apply `select_eligible_products` at each example's cutoff, freeze the selected retriever and retrieve up to 100 eligible products using exact matrix multiplication. Record real candidate recall before augmenting anything.

For training only, include all eligible paid-basket positives, up to eight top retrieved non-positive products as hard negatives, and up to eight random eligible non-positive products. Gifts never become positives and gift-only dates never generate target groups. Remove duplicate pairs and exclude all basket positives from both negative pools. Previously purchased products may be negatives if absent from the next basket.

Use seed 42 to sample complete example groups until approximately 50,000 training pairs for the first Mac experiment; record the sampled group IDs. Label sampling as conditional on next-basket outcomes, not proof that a company dislikes a negative product forever.

Validation and test candidates are the actual retrieved top 100 without positive injection. Their missed positives remain missed in end-to-end metrics. Keep feature bundles referenced by key/cutoff rather than duplicating private text into every pair file.

**Completion:** every pair has the correct basket label, no positive enters a negative pool, and evaluation candidate lists are unaltered retrieval outputs.

## 9. Implement and train the hybrid cross-encoder

Create `recommender/ranker.py` and `train_ranker`. Use the **same company and product bundles**, but process them through three branches:

1. **Joint natural-language branch:** tokenize company activity text and candidate product text as a pair with `AutoTokenizer`, including separator/boundary tokens. Load the multilingual E5-small backbone with `AutoModel`. Limit the combined input to 384 tokens, truncating longer text first. Pool its final non-padding token representations into one joint text vector. Missing text is represented explicitly in both stages.
2. **History branch:** process the identical selected 32-event sequence with a copy of the trained company history encoder. Copy its required event categorical embeddings/projections as well. Subsequent ranker updates never alter retriever weights or its vector cache.
3. **Structured branch:** embed every company/product categorical field and concatenate all preference vectors, technical/package fields, numeric fields and missingness indicators; project this concatenation through a 128-unit GELU layer.

Concatenate the joint text vector, history vector and structured vector. Use a 128-unit GELU scoring layer with dropout 0.1 followed by one scalar logit. Do not replace text interaction with a cosine-similarity feature, and do not substitute final retrieval vectors for the full feature bundles. The candidate does not enter the separate history Transformer; candidate/history relationships can be learned by the scoring head in this chosen hybrid design.

Train with `torch.nn.BCEWithLogitsLoss` on purchase labels and AdamW. Use float32, physical batch size four, gradient accumulation over eight batches, weight decay `1e-4` and gradient clipping at 1.0. Average losses within each example group so companies with larger candidate groups do not dominate solely because of group size.

- Epoch 1: freeze the entire text backbone and train history, structured and scoring components.
- Epochs 2–3: unfreeze only the final two text-encoder layers. Use learning rate `2e-5` for these layers and `1e-3` for the other trainable components.
- Evaluate validation NDCG@10 after each epoch and keep the best checkpoint. Do not select models on the test set.
- A short physical-batch-size-one smoke test precedes full training. If the physical batch of four exceeds memory, reduce to two then one while increasing accumulation to retain an effective batch of 32; record the actual setting.

The unfreezing schedule requires train/eval-mode handling: frozen text layers remain in evaluation mode in epoch 1, while enabled trainable layers use training dropout thereafter. Joint pair representations must be recomputed when encoder weights change; standalone cached E5 vectors are not a cache for these pair-specific outputs.

Save the ranker under `checkpoints/ranker/` with its configuration, feature contract, retriever checkpoint identity and tokenizer revision. Use score/logit terminology; a sigmoid value from sampled negatives is not automatically a calibrated purchase probability.

**Completion:** all features reach the forward pass, scores are finite, a small labelled batch can be overfit, checkpoints reload, and retriever weights remain unchanged.

## 10. Evaluate both stages and their feature contributions

Create `recommender/evaluate.py`. Run validation throughout development and evaluate the held-out test period once the architecture and checkpoints are selected.

Report:

- Eligible-catalogue coverage and retrieval Recall@100.
- End-to-end Recall@5/10, NDCG@10 and MRR, with true baskets as the denominator, including retrieval misses and unretrievable products.
- Retrieval-only versus reranked ordering using identical candidate sets; also show within-candidate ranking quality without confusing it with end-to-end recall.
- First purchase by this company versus repeat purchase, short histories, missing notes and product popularity bands.
- Inference time per company, embedding-refresh time and peak memory on the Mac.

Train a notes-free control with the same splits and settings, replacing notes with the missing-note representation throughout both stages. Also run popularity-only and company-repeat/recency baselines; these reveal whether the neural models add value over the strong historical demand signal already seen in the old feature analysis. Use equal evaluation populations and record training seeds. No old classifier scores are a baseline for this changed catalogue/target without a fresh comparable evaluation.

Do not claim causality or verified historical company descriptions from snapshot-enriched results. Report absolute metrics and deltas; bootstrap uncertainty by company when interpreting small improvements. The ranker is adopted only if it improves held-out NDCG@10 without lowering Recall@10 versus retrieval-only ordering; otherwise keep it experimental and retain the retriever's ordering.

**Completion:** improvements are measured on unmodified retrieved candidates, limitations are visible, and the report distinguishes eligibility, retrieval and ranking failures.

## 11. Implement the complete local serving pipeline

Create `recommender/recommend.py` with a module command accepting company ID, scoring date, top K and checkpoint paths. Load schema, vocabularies, scalers, text caches and both selected model checkpoints together and reject incompatible feature versions.

Use one daily midnight cutoff for company features, product features and ranking features. Normalize a requested scoring date to that boundary and expose the effective cutoff in output metadata. Rebuild candidate vectors once per day or whenever the model/schema changes; select historical demand strictly before that cutoff. Do not mix current demand in the ranker with yesterday's demand inside retrieval vectors.

Serving sequence:

1. Resolve company metadata and purchases prior to the effective cutoff. For a known profile with no purchases, use the missing-history representation; reject completely unknown company IDs with a clear error.
2. Load for-sale, previously sold candidates through `load_eligible_products(items_path, actual_purchase_history, effective_cutoff)`, then build the complete company and eligible product bundles. Unsold items and products received only as gifts cannot enter retrieval, ranking or final output.
3. Encode the company and compare with the matching daily product-vector matrix.
4. Select at most 100 eligible candidates. Never exclude all previously purchased products: replenishment remains a valid recommendation.
5. Pass the same company bundle and each candidate's same product bundle to the hybrid ranker in evaluation/no-gradient mode.
6. Sort by descending ranker score, breaking ties by product ID, and return up to K unique products. If no eligible products exist, return an empty result with the cutoff and reason.
7. Save ignored local output with rank, product ID/name, retrieval score and ranker score, plus cutoff/checkpoint metadata. Do not label scores as probabilities.

This is a local command in v1; an API service, online-learning jobs and production deployment are future additions. No external publication of company data or model weights is implied.

## 12. Automate the stages and acceptance checks

Implement and verify the future commands in this order; these entrypoints are planned and do not exist yet except purchase preparation through the notebook:

```text
1. notebooks/01_clean_purchases.ipynb
2. python -m recommender.prepare_metadata
3. python -m recommender.build_examples
4. python -m recommender.build_features
5. python -m recommender.cache_text
6. python -m recommender.train_retriever
7. python -m recommender.build_candidates
8. python -m recommender.train_ranker
9. python -m recommender.evaluate
10. python -m recommender.recommend --company-id <ID> --scoring-date <YYYY-MM-DD> --top-k 10
```

Use one checked-in pipeline configuration for paths, split dates, feature windows, sequence cap and model defaults, and reference the existing `configs/product_scope.toml` as the single business-eligibility policy. Generated state stays in ignored folders. Each stage records its source/schema/model hashes and refuses stale incompatible upstream artifacts. Restarting model training requires rebuilding downstream candidates and ranker compatibility metadata.

Add meaningful tests for:

- Exact package identity, source-header detection, sale filtering and quantity preservation (already implemented).
- Unique metadata joins, string IDs, missing optional fields and unknown categorical values.
- Strict timestamp boundaries, 30/90-day windows, basket gaps, censoring and split-crossing labels.
- For-sale classification plus prior actual-sale evidence in both stages; exclude never-sold, gift-only and future-only products. Test that adding future sales cannot change past candidates.
- Gift-only dates produce no basket, popularity or cadence; mixed dates keep paid products only.
- Future-purchase invariance and equality of the company/product bundles supplied to each stage.
- Stable 32-event selection, padding and empty-history handling without NaNs.
- Multi-positive retrieval loss, exclusion of positives from negative pools and no evaluation positive injection.
- Matching daily vector/feature cutoffs, checkpoint round trips, unchanged retriever weights during ranker training, and deterministic top-K tie handling.
- One end-to-end synthetic smoke run from prepared data through both models to recommendation output, without private fixtures or downloads in unit tests.

Keep raw data, generated tables, notebook backups, text caches, checkpoint weights and company recommendations local. Clear notebook execution outputs before publishing source. After the first complete run, record completed stages and observed results separately from this plan rather than changing planned metrics into claims of success.

## How the cross-encoder works in this pipeline

The towers read company/product inputs separately and produce reusable vectors. Their dot product is a fast retrieval score. A cross-encoder instead reads company and candidate product text **together** and creates a representation that depends on that specific pair.

The tokenizer splits both texts into tokens and marks their boundaries. Transformer attention computes learned query/key/value representations: each token receives a weighted combination of information from other tokens, including tokens on the other side of the company/product boundary. Several layers build contextual representations; pooling produces the joint text vector. Purchase labels teach the scoring model which interactions are useful for the purchase task, rather than merely general text similarity.

In the selected **hybrid** design, a separate history Transformer performs the same kind of attention across prior purchase events, using explicit product, quantity, position and timing inputs. The structured branch processes the same company/product categories and numbers used by the towers. The final nonlinear head combines all three branches and produces one candidate score. Dates and quantities therefore have explicit numerical representations instead of relying on the language model to infer arithmetic from prose.

This means identical input information with different computation: company/product text has joint token interaction, purchase events have sequence interaction, and all branches influence the final score. The ranker does not consume only two cached final tower vectors. Joint text encoding must run for each candidate pair, which is why the two-tower stage retrieves a bounded set first.

Primary references: [E5-small checkpoint](https://huggingface.co/intfloat/multilingual-e5-small), [cross-encoder principles](https://www.sbert.net/examples/cross_encoder/applications/README.html), [cross-encoder training](https://www.sbert.net/docs/cross_encoder/training_overview.html), and [PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html).
