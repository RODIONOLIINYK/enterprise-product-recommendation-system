"""Build point-in-time Stage 5 examples for the first observed company basket."""

from collections import Counter

import numpy as np
import pandas as pd
import torch

from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.catalogue import RollingCandidatePool
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import quantize_features
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER, SLATE_SIZE
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    load_enriched_companies,
)
from recommender.features import LEGACY_PROFILE_FEATURES, PointInTimeNormalizer, compute_point_in_time_features
from recommender.models import TEXT_DIM
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata

from .base_name_features import BaseNameRollingFeatureBuilder, assemble_arm_features_numpy
from .commercial_features import CommercialRollingFeatureBuilder
from .config import ARM_BASE_NAME_ONLY
from .parsed_spec_features import ParsedSpecRollingFeatureBuilder
from .stage5_base_entity import build_base_name_sequences


FIRST_ORDER_GROUPS = ("3a_financial", "3c_manager", "3e_parsed_specs")
_ORIGIN = pd.Timestamp("2025-01-01")


def first_baskets(purchases: pd.DataFrame):
    """Yield each company's earliest observed dated basket, including single-basket companies."""
    dated = purchases.assign(
        customer_id=purchases["customer_id"].astype(str).str.strip(),
        purchase_date=pd.to_datetime(purchases["purchase_date"]).dt.normalize(),
    )
    if dated["purchase_date"].isna().any() or dated["purchase_date"].ge(TEST_START).any():
        raise ValueError("First-basket preparation requires development purchases only.")
    first = dated.groupby("customer_id")["purchase_date"].transform("min")
    rows = dated.loc[dated["purchase_date"].eq(first)]
    yield from rows.groupby(["purchase_date", "customer_id"], sort=True)["product_id"].agg(set).items()


def prepare_first_baskets(
    purchases,
    catalogue,
    product_ids,
    universe,
    vocabs,
    sales_branch_vocab,
    pool_extra,
):
    companies = load_enriched_companies(purchases=purchases, minimum_purchases=1)
    profile = build_customer_metadata(companies, vocabs)
    sales_branch = companies.set_index("customer_id")["sales_branch"].to_dict()
    config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["config"]
    normalizer = PointInTimeNormalizer.from_state(config["profile_normalizer"])
    text_path = company_embeddings_path(DEFAULT_RETRIEVER, config.get("company_cache_name", "bge_m3_company.parquet"))
    texts = load_cached_embeddings(text_path, "customer_id")
    empty_text = np.zeros(TEXT_DIM, dtype=np.float32)
    profile_stats = compute_point_in_time_features(
        purchases.iloc[:0], "", _ORIGIN, normalizer=normalizer, include_median_purchase_quantity=True,
    )

    pool = RollingCandidatePool(catalogue, purchases)
    dynamic = DynamicFeatureBuilder(
        purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    base_name = BaseNameRollingFeatureBuilder(purchases, catalogue, companies, product_ids)
    # With no observed orders, both commercial groups have only their existing
    # empty-history defaults. No manager assignment is inferred from the first sale.
    commercial = CommercialRollingFeatureBuilder(purchases.iloc[:0], catalogue, product_ids)
    specs = ParsedSpecRollingFeatureBuilder(purchases.iloc[:0], catalogue, companies, product_ids)
    members = universe["bn_to_member_pid_indices"]
    representative = universe["bn_rep_pid_idx"]
    pids = np.asarray(product_ids)
    pid_to_bn = dict(zip(product_ids, universe["pid_idx_to_bn_idx"], strict=True))
    rows = {"train": [], "val": []}
    counts = {split: Counter() for split in rows}

    for (date, customer), targets in first_baskets(purchases):
        split = "train" if date < TRAIN_CUTOFF else "val"
        counts[split]["first_baskets"] += 1
        target = np.zeros(len(members), dtype=bool)
        for pid in targets & pid_to_bn.keys():
            target[pid_to_bn[pid]] = True
        if not target.any():
            counts[split]["no_modeled_target"] += 1
            continue
        allowed_sku = np.isin(pids, list(pool.eligible_on(date)))
        allowed = np.asarray([allowed_sku[indices].any() for indices in members])
        if allowed.sum() < SLATE_SIZE:
            counts[split]["insufficient_eligible_formulations"] += 1
            continue
        if split == "train" and not (target & allowed).any():
            counts[split]["no_eligible_target"] += 1
            continue

        f41 = quantize_features(dynamic.build(customer, date)[representative])
        f19 = quantize_features(base_name.build(customer, date)[representative])
        features = assemble_arm_features_numpy(f41, f19, ARM_BASE_NAME_ONLY)
        extra, _ = pool_extra(
            commercial.build(customer, date)[None, ...].astype(np.float16),
            specs.build(customer, date)[None, ...].astype(np.float16),
            slice(None), FIRST_ORDER_GROUPS, members,
        )
        rows[split].append((customer, date, target, allowed, np.concatenate([features, extra[0]], axis=1)))
        counts[split]["included"] += 1
        counts[split]["targets"] += int(target.sum())
        counts[split]["eligible_targets"] += int((target & allowed).sum())
        counts[split]["with_company_text"] += int(customer in texts)

    result = {}
    for split, cases in rows.items():
        if not cases:
            raise ValueError(f"No usable first-order cases in {split}.")
        customers, dates, targets, allowed, features = zip(*cases)
        days = [(date - _ORIGIN).days for date in dates]
        target = torch.from_numpy(np.stack(targets))
        eligible = torch.from_numpy(np.stack(allowed))
        metadata = [profile.get(customer, {}) for customer in customers]
        sequences = build_base_name_sequences(
            purchases, customers, days, universe, catalogue,
            vocabs["category_vocab"], vocabs["subcategory_vocab"], vocabs["business_line_vocab"],
        )
        if sequences["attention_mask"].any():
            raise ValueError("A first-order example contains purchase history.")
        count = len(cases)
        result[split] = {
            "customer_ids": list(customers),
            "prediction_days": torch.tensor(days, dtype=torch.float32),
            "text_embedding": torch.from_numpy(np.stack([texts.get(customer, empty_text) for customer in customers])),
            **{key: torch.tensor([row.get(key, 0) for row in metadata], dtype=torch.long)
               for key in ("city_id", "region_id", "industry_id")},
            "tabular_features": torch.tensor([profile_stats[name] for name in LEGACY_PROFILE_FEATURES]).repeat(count, 1),
            "customer_stats_6": quantize_transferred_stats(
                torch.tensor([profile_stats[name] for name in config["profile_features"]]).repeat(count, 1)
            ),
            "sales_branch": torch.tensor(encode_categorical_values(
                [sales_branch.get(customer) for customer in customers], sales_branch_vocab,
            ), dtype=torch.long),
            "target_bn": target,
            "allowed_bn": eligible,
            "unseen_bn": eligible.clone(),
            "new_target_bn": target.clone(),
            "features_bn": torch.from_numpy(np.stack(features)),
            **sequences,
        }
    return result, {split: dict(value) for split, value in counts.items()}


def append_first_baskets(train, first_train):
    """Append cold rows while preserving the existing returning rows and feature schema."""
    result = dict(train)
    result["customer_ids"] = train["customer_ids"] + first_train["customer_ids"]
    for key, value in train.items():
        if isinstance(value, torch.Tensor):
            rows = first_train[key]
            if rows.shape[0] != len(first_train["customer_ids"]):
                raise ValueError(f"First-basket rows do not match customer count for {key}.")
            result[key] = torch.cat([value, rows], dim=0)
    return result
