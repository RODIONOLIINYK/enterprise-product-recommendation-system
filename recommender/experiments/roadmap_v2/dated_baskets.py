"""Build the canonical first and returning basket cohorts directly from development events."""

from collections import Counter
from pathlib import Path
from typing import NamedTuple
import hashlib

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import file_digest, json_digest
from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.catalogue import RollingCandidatePool, load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.feature_builder import (
    DynamicFeatureBuilder,
)
from recommender.experiments.dynamic_selector.feature_schema import NAMES
from recommender.experiments.rolling_candidate_v2.config import (
    DEFAULT_RETRIEVER,
    SLATE_SIZE,
)
from recommender.experiments.static_enrichment.text_cache import (
    build_product_text,
    compute_text_recipe_hash,
)
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    load_enriched_companies,
)
from recommender.experiments.static_enrichment.config import (
    TRANSFERRED_CUSTOMER_STAT_NAMES,
)
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.features import (
    LEGACY_PROFILE_FEATURES,
    PointInTimeNormalizer,
    compute_point_in_time_features,
)
from recommender.scope import actual_purchase_mask, get_datasets_locs
from recommender.sequences import DAYS_PER_YEAR, PURCHASE_DATE_ORIGIN
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata
from .base_name_features import (
    BaseNameRollingFeatureBuilder,
    assemble_arm_features_numpy,
)
from .commercial_features import (
    COMMERCIAL_FEATURE_NAMES,
    COMMERCIAL_PARQUET_PATH,
    CommercialRollingFeatureBuilder,
)
from .config import ARM_BASE_NAME_ONLY, feature_names_for_arm
from .parsed_spec_features import (
    PARSED_SPEC_FEATURE_NAMES,
    ParsedSpecRollingFeatureBuilder,
)
from .stage5_base_entity import build_base_name_sequences, build_base_name_universe

COMM_COLUMNS = (0, 1, 2, 3, 4, 5, 14, 15, 16, 17)
SPEC_COLUMNS = (0, 1, 6, 7)
ENRICHMENT = Path("data/interim/static_enrichment_v1/enrichment_bundle.pt")
VOCABS = Path("data/interim/shared_stage1_v1/manifest.pt")
FEATURE_NAMES = (
    feature_names_for_arm(ARM_BASE_NAME_ONLY)
    + tuple(COMMERCIAL_FEATURE_NAMES[i] for i in COMM_COLUMNS)
    + tuple(PARSED_SPEC_FEATURE_NAMES[i] for i in SPEC_COLUMNS)
)


class DatedBasket(NamedTuple):
    customer: str
    date: pd.Timestamp
    first: bool
    target: np.ndarray
    new: np.ndarray
    allowed: np.ndarray
    unseen: np.ndarray
    sku_target: np.ndarray
    sku_allowed: np.ndarray


def aligned_commercial_events(events, purchases, *, test_start: pd.Timestamp = TEST_START):
    """Require the commercial lines to reconstruct exactly the cleaned paid-sale timeline."""
    paid = events.loc[actual_purchase_mask(events)].copy()
    keys = ["customer_id", "purchase_date", "product_id"]
    paid["customer_id"] = paid["customer_id"].astype(str).str.strip()
    paid["product_id"] = paid["product_id"].astype(str).str.strip()
    paid["purchase_date"] = pd.to_datetime(paid["purchase_date"], format="mixed").dt.normalize().astype("datetime64[ns]")
    purch = purchases[keys + ["quantity"]].copy()
    purch["customer_id"] = purch["customer_id"].astype(str).str.strip()
    purch["product_id"] = purch["product_id"].astype(str).str.strip()
    purch["purchase_date"] = pd.to_datetime(purch["purchase_date"], format="mixed").dt.normalize().astype("datetime64[ns]")
    totals = paid.groupby(keys).quantity.sum().sort_index()
    expected = purch.set_index(keys).quantity.sort_index()
    if not totals.index.equals(expected.index) or not np.allclose(
        totals, expected, rtol=1e-7, atol=1e-7
    ):
        raise ValueError("Commercial and cleaned purchase snapshots disagree.")
    # A dated basket is fully settled only after all its paid lines have settled.
    # An unknown settlement must behave exactly like a settlement still in the future.
    settlements = pd.to_datetime(paid["full_payment_date"], errors="coerce")
    incomplete = settlements.isna() | settlements.ge(test_start)
    incomplete_basket = incomplete.groupby(
        [paid.customer_id, paid.purchase_date]
    ).transform("any")
    paid["full_payment_date"] = settlements.mask(incomplete_basket)
    return paid


def first_observed_pack_indices(product_ids, members, first_sales):
    """Anchor each formulation's metadata at introduction, breaking same-day ties by SKU ID."""
    return np.array(
        [
            min(group, key=lambda i: (first_sales[product_ids[i]], product_ids[i]))
            for group in members
        ],
        dtype=np.int64,
    )


def basket_cases(
    purchases,
    product_ids,
    universe,
    pool,
    *,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
    test_start: pd.Timestamp = TEST_START,
):
    """Keep each dated basket once. Targets and prior ownership are independent of eligibility."""
    if (
        purchases.purchase_date.isna().any()
        or purchases.purchase_date.ge(test_start).any()
    ):
        raise ValueError("Expected dated development events only.")
    by_bn = universe["bn_to_idx"]
    mapping = {p: by_bn[b] for p, b in universe["all_pid_to_bn"].items() if b in by_bn}
    idx = universe["pid_idx_to_bn_idx"]
    pids = np.asarray(product_ids)
    n = len(by_bn)
    seen = {}
    counts = {s: Counter() for s in ("train", "val")}
    cases = {s: [] for s in counts}
    for date, daily in purchases.groupby("purchase_date", sort=True):
        allowed_sku = np.isin(pids, list(pool.eligible_on(date)))
        allowed = np.zeros(n, bool)
        allowed[idx[allowed_sku]] = True
        split = "train" if date < train_cutoff else "val"
        for cid, basket in daily.groupby("customer_id", sort=True):
            first = cid not in seen
            prior = seen.setdefault(cid, np.zeros(n, bool))
            target = np.zeros(n, bool)
            target[[mapping[p] for p in basket.product_id if p in mapping]] = True
            sku_target = np.isin(pids, basket.product_id)
            if not target.any():
                counts[split]["no_modeled_target"] += 1
            elif allowed.sum() < SLATE_SIZE:
                counts[split]["insufficient_pool"] += 1
            elif split == "train" and not (target & allowed).any():
                counts[split]["no_eligible_positive"] += 1
            else:
                cases[split].append(
                    DatedBasket(
                        cid,
                        date,
                        first,
                        target,
                        target & ~prior,
                        allowed,
                        allowed & ~prior,
                        sku_target,
                        allowed_sku,
                    )
                )
            prior |= target
    return cases, counts


def validate_dated_split(
    data,
    split,
    *,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
    test_start: pd.Timestamp = TEST_START,
):
    """Verify row dates and the paid-history contract before fitting or replay."""
    days = data["prediction_days"]
    if split not in ("train", "val") or train_cutoff >= test_start:
        raise ValueError("Invalid development partition or split boundaries.")
    if days.ndim != 1 or not torch.isfinite(days).all() or (days != days.round()).any():
        raise ValueError("Prediction days must be finite integer calendar days.")
    dates = PURCHASE_DATE_ORIGIN + pd.to_timedelta(days.numpy(), unit="D")
    if (dates >= test_start).any() or (
        (dates < train_cutoff) != (split == "train")
    ).any():
        raise ValueError("Basket outside its development partition.")
    if len(set(zip(data["customer_ids"], days.tolist(), strict=True))) != len(days):
        raise ValueError("Duplicate customer/date prediction.")
    valid = data["attention_mask"].bool()
    event_days = (data["seq_purchase_dates"] * DAYS_PER_YEAR).round()
    if not torch.isfinite(event_days[valid]).all() or ((event_days >= days[:, None]) & valid).any():
        raise ValueError("History contains a same-day or future event.")
    cold = data["first_observed"]
    if valid[cold].any() or data["familiarity_bn"][cold].any():
        raise ValueError("First observed basket contains customer history.")
    target, new = data["target_bn"], data["new_target_bn"]
    allowed, unseen = data["allowed_bn"], data["unseen_bn"]
    if (unseen & ~allowed).any() or (new & ~target).any() or not torch.equal(new & allowed, target & unseen):
        raise ValueError("Target novelty and eligible unseen masks disagree.")
    if not torch.equal(new[cold], target[cold]) or not torch.equal(
        unseen[cold], allowed[cold]
    ):
        raise ValueError("First basket repeat labels disagree with its empty history.")


def dated_candidate_features(
    customer, date, representatives, dynamic, base, commercial, specs
):
    sku = dynamic.build(customer, date)
    formulation = base.build(
        customer,
        date,
        category_last_lift=sku[:, NAMES.index("category_last_lift")],
        family_last_lift=sku[:, NAMES.index("family_last_lift")],
    )
    return np.concatenate(
        (
            assemble_arm_features_numpy(
                sku[representatives], formulation[representatives], ARM_BASE_NAME_ONLY
            ),
            commercial.build(customer, date)[representatives][:, COMM_COLUMNS],
            specs.build(customer, date)[representatives][:, SPEC_COLUMNS],
        ),
        axis=1,
    ).astype(np.float16)


def prepare_dated_baskets(*, historical_recipe=False):
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_enriched_companies(purchases=purchases, minimum_purchases=0)
    enrichment = torch.load(ENRICHMENT, map_location="cpu", weights_only=False)
    product_ids = enrichment["product_ids"]
    if set(product_ids) != set(purchases.product_id) or len(set(product_ids)) != len(
        product_ids
    ):
        raise ValueError(
            "Rebuild the static product bundle for this development SKU universe."
        )
    indexed = catalogue.set_index("product_id")
    text_inputs = [
        (pid, build_product_text(indexed.loc[pid], "t1_fam")) for pid in product_ids
    ]
    if (
        compute_text_recipe_hash(text_inputs, "t1_fam")
        != enrichment["text_recipe_hashes"]["t1_fam"]
    ):
        raise ValueError("Rebuild the product text cache for the current metadata.")
    vocabs = torch.load(VOCABS, map_location="cpu", weights_only=False)["vocabs"]
    config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)[
        "config"
    ]
    text_file = company_embeddings_path(DEFAULT_RETRIEVER, config["company_cache_name"])
    texts = load_cached_embeddings(text_file, "customer_id")
    if (set(purchases.customer_id) & set(companies.customer_id)) - texts.keys():
        raise ValueError("A company profile is missing from the static text cache.")
    normalizer = PointInTimeNormalizer.from_state(config["profile_normalizer"])
    profiles = build_customer_metadata(companies, vocabs)
    sales_branch = companies.set_index("customer_id").sales_branch.to_dict()
    pool = RollingCandidatePool(catalogue, purchases)
    universe = build_base_name_universe(product_ids, catalogue, purchases)
    reps = first_observed_pack_indices(
        product_ids, universe["bn_to_member_pid_indices"], pool.first_sales
    )
    representatives = (
        catalogue.set_index("product_id")
        .loc[np.asarray(product_ids)[reps]]
        .reset_index()
    )
    # Snapshot master fields only. Disallow the legacy all-dates transaction metadata fallback.
    dynamic = DynamicFeatureBuilder(
        purchases[["purchase_date", "customer_id", "product_id", "quantity"]],
        catalogue,
        companies,
        product_ids,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    base = BaseNameRollingFeatureBuilder(purchases, catalogue, companies, product_ids)
    commercial = aligned_commercial_events(
        pd.read_parquet(
            COMMERCIAL_PARQUET_PATH, filters=[("purchase_date", "<", TEST_START)]
        ),
        purchases,
    )
    comm = CommercialRollingFeatureBuilder(commercial, catalogue, product_ids, historical_recipe=historical_recipe)
    specs = ParsedSpecRollingFeatureBuilder(
        purchases, catalogue, companies, product_ids, historical_recipe=historical_recipe
    )
    familiarity = FamiliarityStore(
        purchases, catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    histories = dict(tuple(purchases.groupby("customer_id", sort=False)))
    cases, counts = basket_cases(purchases, product_ids, universe, pool)
    splits, sku_data = {}, {}
    for split, rows in cases.items():
        cids, dates, first, targets, new, allowed, unseen, sku_targets, sku_allowed = (
            zip(*rows, strict=True)
        )
        days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]
        sequences = build_base_name_sequences(
            purchases,
            cids,
            days,
            universe,
            representatives,
            vocabs["category_vocab"],
            vocabs["subcategory_vocab"],
            vocabs["business_line_vocab"],
        )
        features = np.empty((len(rows), len(reps), len(FEATURE_NAMES)), np.float16)
        profile8, profile6 = [], []
        for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
            features[i] = dated_candidate_features(
                cid, date, reps, dynamic, base, comm, specs
            )
            stats = compute_point_in_time_features(
                histories[cid],
                cid,
                date,
                normalizer=normalizer,
                include_median_purchase_quantity=True,
            )
            profile8.append([stats[k] for k in LEGACY_PROFILE_FEATURES])
            profile6.append([stats[k] for k in TRANSFERRED_CUSTOMER_STAT_NAMES])
            if (i + 1) % 5000 == 0:
                print(f"[dated baskets] {split} {i + 1}/{len(rows)}", flush=True)
        flags = torch.tensor(first, dtype=torch.bool)
        splits[split] = {
            "customer_ids": list(cids),
            "prediction_days": torch.tensor(days, dtype=torch.float32),
            "first_observed": flags,
            "target_bn": torch.from_numpy(np.stack(targets)),
            "new_target_bn": torch.from_numpy(np.stack(new)),
            "allowed_bn": torch.from_numpy(np.stack(allowed)),
            "unseen_bn": torch.from_numpy(np.stack(unseen)),
            "features_bn": torch.from_numpy(features),
            "feature_names": FEATURE_NAMES,
            "familiarity_bn": torch.cat(
                [
                    familiarity.batch_features(cids[i : i + 256], dates[i : i + 256])
                    for i in range(0, len(rows), 256)
                ]
            ),
            "text_embedding": torch.from_numpy(
                np.stack([texts.get(cid, np.zeros(1024, np.float32)) for cid in cids])
            ),
            "tabular_features": torch.tensor(profile8, dtype=torch.float32),
            "customer_stats_6": quantize_transferred_stats(
                np.asarray(profile6, np.float32)
            ),
            **{
                k: torch.tensor([profiles.get(cid, {}).get(k, 0) for cid in cids])
                for k in ("city_id", "region_id", "industry_id")
            },
            "sales_branch": torch.tensor(
                encode_categorical_values(
                    [sales_branch.get(cid) for cid in cids],
                    enrichment["customer_vocabs"]["sales_branch"],
                )
            ),
            **sequences,
        }
        validate_dated_split(splits[split], split)
        if split == "val":
            # New SKU status for pack-level validation uses the company's full prior SKU history.
            first_sales = (
                purchases.groupby(["customer_id", "product_id"])
                .purchase_date.min()
                .unstack()
            )
            known_at = first_sales.reindex(index=cids, columns=product_ids).to_numpy(
                dtype="datetime64[D]"
            )
            target = np.stack(sku_targets)
            sku_data[split] = {
                "target": torch.from_numpy(target),
                "allowed": torch.from_numpy(np.stack(sku_allowed)),
                "new_target": torch.from_numpy(
                    target
                    & ~(known_at < np.asarray(dates, dtype="datetime64[D]")[:, None])
                ),
            }
        counts[split].update(
            first_observed=sum(first), returning=len(rows) - sum(first)
        )
    inputs = {
        "text_embedding": enrichment["text_embeddings"]["t1_fam"][reps].float(),
        "attr_ids": {k: v[reps] for k, v in enrichment["product_attr_ids"].items()},
        "demand_features": torch.zeros(len(reps), 12),
    }
    manifest = {
        "development_events_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(purchases, index=False).values.tobytes()
        ).hexdigest(),
        "source_files_sha256": {
            Path(p).name: file_digest(p)
            for p in (
                get_datasets_locs("items"),
                get_datasets_locs("companies"),
                ENRICHMENT,
                VOCABS,
                DEFAULT_RETRIEVER,
                text_file,
                COMMERCIAL_PARQUET_PATH,
            )
        },
        "cohorts": {s: dict(v) for s, v in counts.items()},
        "features": list(FEATURE_NAMES),
        "row_keys_sha256": {
            s: json_digest(
                list(zip(d["customer_ids"], d["prediction_days"].int().tolist()))
            )
            for s, d in splits.items()
        },
        "catalogue_sha256": json_digest(
            list(zip(universe["base_names"], np.asarray(product_ids)[reps].tolist()))
        ),
    }
    return (
        splits,
        inputs,
        universe,
        enrichment,
        vocabs,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        manifest,
    )
