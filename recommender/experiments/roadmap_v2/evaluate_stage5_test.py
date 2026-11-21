"""Authorized final inference-only held-out Test evaluation (>= 2026-02-19) for Stage 5 reference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.catalogue import RollingCandidatePool, load_product_catalogue, select_sellable_products
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.rolling_candidate_v2.config import (
    DEFAULT_RETRIEVER,
    SLATE_POLICY,
    SLATE_SIZE,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.experiments.static_enrichment.config import TRANSFERRED_CUSTOMER_STAT_NAMES
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    load_enriched_companies,
)
from recommender.experiments.static_enrichment.text_cache import (
    build_product_text,
    compute_text_recipe_hash,
)
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.features import (
    LEGACY_PROFILE_FEATURES,
    PointInTimeNormalizer,
    compute_point_in_time_features,
)
from recommender.scope import actual_purchase_mask, get_datasets_locs
from recommender.sequences import DAYS_PER_YEAR, PURCHASE_DATE_ORIGIN
from recommender.split import TEST_START
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata, get_device

from .base_name_features import BaseNameRollingFeatureBuilder
from .commercial_features import COMMERCIAL_PARQUET_PATH, CommercialRollingFeatureBuilder
from .config import EVIDENCE_ROOT
from .dated_baskets import (
    ENRICHMENT,
    FEATURE_NAMES,
    VOCABS,
    dated_candidate_features,
    first_observed_pack_indices,
)
from .first_baskets import FIRST_ORDER_GROUPS
from .parsed_spec_features import ParsedSpecRollingFeatureBuilder
from .run_stage5 import Stage5BaseNameRanker, _stage5_tower, _verify_reference_files
from .stage1_mlp import RoadmapStage1Selector
from .stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    evaluate_base_name_predictions,
    resolve_base_names_to_skus,
)


RAW_WORKBOOK_PATH = Path("data/raw/dataset.xlsx")
CLEANED_PURCHASES_PATH = Path("data/interim/cleaned_purchases.csv")
REFERENCE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"
TEST_SUMMARY_PATH = EVIDENCE_ROOT / "stage5_dated_first_baskets_test_seed42.json"

EXCEL_COMMERCIAL_MAPPING: Dict[str, str] = {
    "Gen_ Bus_ Posting Group": "transaction_type",
    "Gen_ Prod_ Posting Group": "item_type",
    "ДатаПродажи": "purchase_date",
    "БизнесЛиния": "business_line",
    "Категория": "product_category",
    "КлиентДляОплатыКод": "customer_id",
    "Склад": "warehouse",
    "ТоварКод": "product_id",
    "Товар": "product_name",
    "Количество": "quantity",
    "Себестоимость": "cost",
    "СебестоимостьУЕ": "cost_usd",
    "Оборот": "revenue",
    "ДатаПолнойОплати": "full_payment_date",
    "БонусДокумента": "doc_bonus",
    "МенеджерКод": "manager_id",
    "УсловияОплаты": "payment_terms",
    "ТипОплаты": "payment_type",
}


class HeldoutTestBasket(NamedTuple):
    customer: str
    date: pd.Timestamp
    first: bool
    has_modeled_target: bool
    target: np.ndarray
    new: np.ndarray
    allowed: np.ndarray
    unseen: np.ndarray
    extra_bn_targets: int
    extra_new_bn_targets: int
    sku_target: np.ndarray
    sku_allowed: np.ndarray
    extra_sku_targets: int
    extra_new_sku_targets: int


def load_all_cleaned_purchases(path: Path = CLEANED_PURCHASES_PATH) -> pd.DataFrame:
    """Load the complete cleaned paid purchase timeline (dev + held-out test)."""
    df = pd.read_csv(path)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if df["purchase_date"].isna().any() or df["quantity"].isna().any():
        raise ValueError("Cleaned purchases contain missing dates or quantities.")
    return df


def load_aligned_full_commercial_events(
    workbook_path: Path,
    all_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    """Read raw commercial lines (dev + test) and verify exact quantity reconciliation."""
    with pd.ExcelFile(workbook_path) as wb:
        preview = pd.read_excel(wb, sheet_name=0, header=None, nrows=20)
        header_row = next(
            (
                idx
                for idx, row in preview.iterrows()
                if set(EXCEL_COMMERCIAL_MAPPING).issubset(set(row.dropna()))
            ),
            None,
        )
        if header_row is None:
            raise ValueError("Expected commercial headers not found in first 20 rows of dataset.xlsx.")
        raw = pd.read_excel(
            wb,
            sheet_name=0,
            header=header_row,
            usecols=lambda col: col in EXCEL_COMMERCIAL_MAPPING,
            dtype={"КлиентДляОплатыКод": "string", "ТоварКод": "string", "МенеджерКод": "string"},
        ).rename(columns=EXCEL_COMMERCIAL_MAPPING)

    text_cols = (
        "transaction_type",
        "item_type",
        "business_line",
        "product_category",
        "customer_id",
        "warehouse",
        "product_id",
        "product_name",
        "manager_id",
        "payment_terms",
        "payment_type",
    )
    for col in text_cols:
        raw[col] = raw[col].astype("string").str.strip()
    for col in ("transaction_type", "item_type"):
        raw[col] = raw[col].str.upper()
    raw["purchase_date"] = pd.to_datetime(raw["purchase_date"], errors="coerce").dt.normalize()
    raw["full_payment_date"] = pd.to_datetime(raw["full_payment_date"], errors="coerce").dt.normalize()
    for col in ("quantity", "cost", "cost_usd", "revenue", "doc_bonus"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    sellable_ids = set(select_sellable_products(catalogue)["product_id"].astype(str))
    paid_mask = (
        raw["customer_id"].notna()
        & raw["customer_id"].ne("")
        & raw["product_id"].notna()
        & raw["product_id"].ne("")
        & raw["purchase_date"].notna()
        & np.isfinite(raw["quantity"].fillna(np.nan))
        & actual_purchase_mask(raw)
        & raw["product_id"].isin(sellable_ids)
    )
    paid = raw.loc[paid_mask].copy()

    keys = ["customer_id", "purchase_date", "product_id"]
    totals = paid.groupby(keys)["quantity"].sum().sort_index()
    expected = all_purchases.set_index(keys)["quantity"].sort_index()
    if not totals.index.equals(expected.index) or not np.allclose(
        totals.to_numpy(), expected.to_numpy(), rtol=1e-7, atol=1e-7
    ):
        raise ValueError("Full commercial export and cleaned purchases disagree.")

    # Multi-line basket counts as settled only once EVERY paid line has a known settlement date.
    # Each query date t strictly checks settle_date < t inside CommercialRollingFeatureBuilder.
    settlements = pd.to_datetime(paid["full_payment_date"], errors="coerce").dt.normalize()
    incomplete = settlements.isna()
    incomplete_basket = incomplete.groupby(
        [paid["customer_id"], paid["purchase_date"]]
    ).transform("any")
    paid["full_payment_date"] = settlements.mask(incomplete_basket)
    return paid


def build_test_basket_cases(
    all_purchases: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    pool: RollingCandidatePool,
) -> Tuple[List[HeldoutTestBasket], Dict[str, int]]:
    """Build chronological held-out Test baskets (>= TEST_START), retaining full-catalogue denominators."""
    by_bn = universe["bn_to_idx"]
    all_pid_to_bn = universe["all_pid_to_bn"]
    mapping = {p: by_bn[b] for p, b in all_pid_to_bn.items() if b in by_bn}
    idx = universe["pid_idx_to_bn_idx"]
    pids = np.asarray(product_ids)
    pid_set = set(product_ids)
    n = len(by_bn)

    seen_any_order: set[str] = set()
    seen_modeled_bn: Dict[str, np.ndarray] = {}
    seen_unmodeled_bn: Dict[str, set[str]] = {}
    seen_unmodeled_sku: Dict[str, set[str]] = {}

    cases: List[HeldoutTestBasket] = []
    counts: Counter[str] = Counter()

    for date, daily in all_purchases.groupby("purchase_date", sort=True):
        is_test = bool(date >= TEST_START)
        if is_test:
            allowed_sku = np.isin(pids, list(pool.eligible_on(date)))
            allowed = np.zeros(n, dtype=bool)
            allowed[idx[allowed_sku]] = True
            if int(allowed.sum()) < SLATE_SIZE:
                raise ValueError(
                    f"Test date {date.date()} has fewer than {SLATE_SIZE} eligible formulations ({int(allowed.sum())})."
                )
        for cid, basket in daily.groupby("customer_id", sort=True):
            first = cid not in seen_any_order
            prior_mod = seen_modeled_bn.setdefault(cid, np.zeros(n, dtype=bool))
            prior_unmod_bn = seen_unmodeled_bn.setdefault(cid, set())
            prior_unmod_sku = seen_unmodeled_sku.setdefault(cid, set())

            basket_pids = set(basket["product_id"].tolist())
            mod_idxs = sorted({mapping[p] for p in basket_pids if p in mapping})
            unmod_bns = {
                all_pid_to_bn.get(p, "")
                for p in basket_pids
                if all_pid_to_bn.get(p, "") and all_pid_to_bn.get(p, "") not in by_bn
            }
            unmod_skus = basket_pids - pid_set

            target = np.zeros(n, dtype=bool)
            if mod_idxs:
                target[mod_idxs] = True

            if is_test:
                sku_target = np.isin(pids, list(basket_pids))
                extra_bn = len(unmod_bns)
                extra_new_bn = len(unmod_bns - prior_unmod_bn)
                extra_sku = len(unmod_skus)
                extra_new_sku = len(unmod_skus - prior_unmod_sku)
                has_modeled = bool(target.any())

                counts["all_baskets"] += 1
                counts["first_observed_all" if first else "returning_all"] += 1
                if has_modeled:
                    counts["modeled_baskets"] += 1
                    counts["first_observed_modeled" if first else "returning_modeled"] += 1
                else:
                    counts["no_modeled_target_baskets"] += 1
                if extra_bn > 0:
                    counts["baskets_with_unmodeled_bn"] += 1
                counts["modeled_bn_targets"] += int(target.sum())
                counts["unmodeled_bn_targets"] += extra_bn
                counts["modeled_new_bn_targets"] += int((target & ~prior_mod).sum())
                counts["unmodeled_new_bn_targets"] += extra_new_bn

                cases.append(
                    HeldoutTestBasket(
                        customer=cid,
                        date=date,
                        first=first,
                        has_modeled_target=has_modeled,
                        target=target,
                        new=target & ~prior_mod,
                        allowed=allowed,
                        unseen=allowed & ~prior_mod,
                        extra_bn_targets=extra_bn,
                        extra_new_bn_targets=extra_new_bn,
                        sku_target=sku_target,
                        sku_allowed=allowed_sku,
                        extra_sku_targets=extra_sku,
                        extra_new_sku_targets=extra_new_sku,
                    )
                )

            seen_any_order.add(cid)
            prior_mod |= target
            prior_unmod_bn |= unmod_bns
            prior_unmod_sku |= unmod_skus

    return cases, dict(counts)


def validate_heldout_test_split(data: Mapping[str, Any]) -> None:
    """Assert strict temporal boundaries, causality, and label integrity on the test split."""
    days = data["prediction_days"]
    dates = PURCHASE_DATE_ORIGIN + pd.to_timedelta(days.numpy(), unit="D")
    if (dates < TEST_START).any():
        raise ValueError("Held-out test split contains a pre-TEST_START basket.")
    if len(set(zip(data["customer_ids"], days.tolist(), strict=True))) != len(days):
        raise ValueError("Duplicate customer/date row in test split.")
    valid = data["attention_mask"].bool()
    event_days = (data["seq_purchase_dates"] * DAYS_PER_YEAR).round()
    if ((event_days >= days[:, None]) & valid).any():
        raise ValueError("Test sequence history contains a same-day or future purchase!")
    cold = data["first_observed"]
    if valid[cold].any() or data["familiarity_bn"][cold].any():
        raise ValueError("First observed test basket contains non-empty history or familiarity!")
    target, new = data["target_bn"], data["new_target_bn"]
    allowed, unseen = data["allowed_bn"], data["unseen_bn"]
    if (new & ~target).any() or not torch.equal(new & allowed, target & unseen):
        raise ValueError("Test target novelty and eligible unseen masks disagree.")
    if not torch.equal(new[cold], target[cold]) or not torch.equal(unseen[cold], allowed[cold]):
        raise ValueError("First observed test basket novelty/unseen masks disagree with empty history.")


def evaluate_full_catalogue_predictions(
    ranked_bn_indices: np.ndarray,
    target_bn: torch.Tensor,
    new_target_bn: torch.Tensor,
    extra_total_targets: np.ndarray,
    extra_new_targets: np.ndarray,
    customer_ids: Sequence[str],
    *,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate formulation predictions with unmodeled test-only targets included as misses in denominator."""
    n_rows = ranked_bn_indices.shape[0]
    t_np = target_bn.detach().cpu().numpy().astype(bool)
    nt_np = new_target_bn.detach().cpu().numpy().astype(bool)
    ext_t = np.asarray(extra_total_targets, dtype=np.int64)
    ext_nt = np.asarray(extra_new_targets, dtype=np.int64)

    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    per_q_r5 = np.zeros(n_rows, dtype=np.float64)
    per_q_r10 = np.zeros(n_rows, dtype=np.float64)
    per_q_r25 = np.zeros(n_rows, dtype=np.float64)
    per_q_r50 = np.zeros(n_rows, dtype=np.float64)
    per_q_ndcg10 = np.zeros(n_rows, dtype=np.float64)
    per_q_new_hits10 = np.zeros(n_rows, dtype=np.float64)
    per_q_new_targets = nt_np.sum(axis=1).astype(np.float64) + ext_nt.astype(np.float64)
    per_q_targets = t_np.sum(axis=1).astype(np.float64) + ext_t.astype(np.float64)

    hits_at = {5: 0, 10: 0, 25: 0, 50: 0}
    new_hits_at = {5: 0, 10: 0, 25: 0, 50: 0}

    for i in range(n_rows):
        n_pos = int(t_np[i].sum()) + int(ext_t[i])
        if n_pos == 0:
            continue
        preds = ranked_bn_indices[i]
        preds_v = preds[preds >= 0]
        rel = t_np[i, preds_v]
        new_rel = nt_np[i, preds_v]

        for k in (5, 10, 25, 50):
            hits_at[k] += int(rel[:k].sum())
            new_hits_at[k] += int(new_rel[:k].sum())

        per_q_r5[i] = float(rel[:5].sum()) / n_pos
        per_q_r10[i] = float(rel[:10].sum()) / n_pos
        per_q_r25[i] = float(rel[:25].sum()) / n_pos
        per_q_r50[i] = float(rel[:50].sum()) / n_pos
        per_q_new_hits10[i] = float(new_rel[:10].sum())

        top10_rel = rel[:10].astype(np.float64)
        dcg = float((top10_rel * discounts[: len(top10_rel)]).sum())
        idcg = float(discounts[: min(n_pos, 10)].sum())
        per_q_ndcg10[i] = dcg / idcg if idcg > 0 else 0.0

    total_targets = int(per_q_targets.sum())
    total_new = int(per_q_new_targets.sum())
    metrics = {
        "queries": int(n_rows),
        "total_bn_targets": total_targets,
        "true_new_bn_targets": total_new,
        "unmodeled_bn_targets": int(ext_t.sum()),
        "unmodeled_new_bn_targets": int(ext_nt.sum()),
        "bn_recall@5": float(per_q_r5.mean()) if n_rows else 0.0,
        "bn_recall@10": float(per_q_r10.mean()) if n_rows else 0.0,
        "bn_recall@25": float(per_q_r25.mean()) if n_rows else 0.0,
        "bn_recall@50": float(per_q_r50.mean()) if n_rows else 0.0,
        "bn_micro_recall@10": float(hits_at[10] / max(total_targets, 1)),
        "bn_ndcg@10": float(per_q_ndcg10.mean()) if n_rows else 0.0,
        "true_new_bn_hits@5": int(new_hits_at[5]),
        "true_new_bn_hits@10": int(new_hits_at[10]),
        "true_new_bn_hits@25": int(new_hits_at[25]),
        "true_new_bn_hits@50": int(new_hits_at[50]),
        "true_new_bn_recall@10": float(new_hits_at[10] / max(total_new, 1)),
        "true_new_bn_recall@25": float(new_hits_at[25] / max(total_new, 1)),
        "true_new_bn_recall@50": float(new_hits_at[50] / max(total_new, 1)),
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            customer_ids=np.asarray(list(customer_ids), dtype=object),
            recall_5=per_q_r5,
            recall_10=per_q_r10,
            recall_25=per_q_r25,
            recall_50=per_q_r50,
            ndcg_10=per_q_ndcg10,
            new_hits_10=per_q_new_hits10,
            new_targets=per_q_new_targets,
            targets=per_q_targets,
        )
    return metrics


def _evaluate_resolved_skus(
    resolved_skus: np.ndarray,
    t_sku: np.ndarray,
    nt_sku: np.ndarray,
    extra_sku_targets: Optional[np.ndarray] = None,
    row_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute exact-SKU Recall@10, NDCG@10, and new-SKU hits@10."""
    disc_arr = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    rows = np.arange(len(resolved_skus)) if row_mask is None else np.where(row_mask)[0]
    sku_r10, sku_ndcg10 = [], []
    sku_new10 = 0
    total_sku_targets = 0
    for r_i in rows:
        row_p = resolved_skus[r_i]
        valid_p = row_p[row_p >= 0]
        rel = t_sku[r_i, valid_p].astype(np.float64)
        extra_n = int(extra_sku_targets[r_i]) if extra_sku_targets is not None else 0
        n_pos = int(t_sku[r_i].sum()) + extra_n
        total_sku_targets += n_pos
        if n_pos > 0:
            sku_r10.append(float(rel.sum()) / n_pos)
            dcg = float((rel * disc_arr[: len(rel)]).sum())
            idcg = float(disc_arr[: min(n_pos, 10)].sum())
            sku_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        sku_new10 += int(nt_sku[r_i, valid_p].sum())
    return {
        "queries": int(len(sku_r10)),
        "total_sku_targets": int(total_sku_targets),
        "sku_recall@10": float(np.mean(sku_r10)) if sku_r10 else 0.0,
        "sku_ndcg@10": float(np.mean(sku_ndcg10)) if sku_ndcg10 else 0.0,
        "sku_new_hits@10": int(sku_new10),
    }


def _snapshot_dev_source_hashes() -> Dict[str, str]:
    config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["config"]
    text_file = company_embeddings_path(DEFAULT_RETRIEVER, config["company_cache_name"])
    paths = (
        get_datasets_locs("items"),
        get_datasets_locs("companies"),
        ENRICHMENT,
        VOCABS,
        DEFAULT_RETRIEVER,
        text_file,
        COMMERCIAL_PARQUET_PATH,
    )
    return {Path(p).name: file_digest(p) for p in paths}


def run_heldout_test_evaluation(
    *,
    allow_heldout_test: bool = False,
    device: Optional[str] = None,
    reference_path: Path = REFERENCE_PATH,
    summary_output_path: Path = TEST_SUMMARY_PATH,
) -> Dict[str, Any]:
    """Run strictly inference-only final held-out Test evaluation on pinned Stage 5 reference."""
    if not allow_heldout_test:
        raise PermissionError(
            "Held-out Test evaluation (>= 2026-02-19) is locked by default. "
            "Pass --allow-heldout-test only when explicitly authorized for final evaluation."
        )

    t0 = time.perf_counter()
    dev = get_device(device)
    before_preserved = snapshot_preserved_hashes()
    before_dev_sources = _snapshot_dev_source_hashes()

    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    _verify_reference_files(reference)
    saved_val = json.loads(Path(reference["checkpoints"]["summary"]["path"]).read_text(encoding="utf-8"))
    if (
        reference["feature_groups"] != list(FIRST_ORDER_GROUPS)
        or reference["candidate_policy"] != SLATE_POLICY
        or reference["aux_lambda"] != 0.3
        or reference["heldout_access"] is not False
    ):
        raise ValueError("Pinned Stage 5 reference recipe mismatch.")

    # 1. Verify dev timeline & frozen static inputs match the pinned reference bit-for-bit
    dev_purchases = load_development_purchases(str(CLEANED_PURCHASES_PATH))
    dev_digest = hashlib.sha256(
        pd.util.hash_pandas_object(dev_purchases, index=False).values.tobytes()
    ).hexdigest()
    if dev_digest != saved_val["data_manifest"]["development_events_sha256"]:
        raise ValueError("Development events SHA-256 differs from pinned validation reference!")
    if before_dev_sources != saved_val["data_manifest"]["source_files_sha256"]:
        raise ValueError("Static/dev source file SHA-256 differs from pinned validation reference!")

    all_purchases = load_all_cleaned_purchases(CLEANED_PURCHASES_PATH)
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_enriched_companies(purchases=all_purchases, minimum_purchases=0)
    enrichment = torch.load(ENRICHMENT, map_location="cpu", weights_only=False)
    product_ids = list(enrichment["product_ids"])

    indexed_cat = catalogue.set_index("product_id")
    text_inputs = [(pid, build_product_text(indexed_cat.loc[pid], "t1_fam")) for pid in product_ids]
    if compute_text_recipe_hash(text_inputs, "t1_fam") != enrichment["text_recipe_hashes"]["t1_fam"]:
        raise ValueError("Product text recipe hash mismatch.")

    vocabs = torch.load(VOCABS, map_location="cpu", weights_only=False)["vocabs"]
    config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["config"]
    text_file = company_embeddings_path(DEFAULT_RETRIEVER, config["company_cache_name"])
    texts = load_cached_embeddings(text_file, "customer_id")

    test_cids_all = set(all_purchases.loc[all_purchases["purchase_date"] >= TEST_START, "customer_id"])
    missing_text = (test_cids_all & set(companies["customer_id"])) - set(texts.keys())
    if missing_text:
        raise ValueError(f"{len(missing_text)} test companies are missing from cached BGE-M3 company texts.")

    # Frozen dev universe & earliest-paid-pack formulation representatives (zero test influence!)
    dev_pool = RollingCandidatePool(catalogue, dev_purchases)
    universe = build_base_name_universe(product_ids, catalogue, dev_purchases)
    reps = first_observed_pack_indices(
        product_ids, universe["bn_to_member_pid_indices"], dev_pool.first_sales
    )
    cat_sha = json_digest(list(zip(universe["base_names"], np.asarray(product_ids)[reps].tolist())))
    if cat_sha != saved_val["data_manifest"]["catalogue_sha256"]:
        raise ValueError("Canonical formulation catalogue SHA-256 mismatch vs pinned reference!")

    representatives = (
        catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()
    )
    prod_inputs = {
        "text_embedding": enrichment["text_embeddings"]["t1_fam"][reps].float(),
        "attr_ids": {k: v[reps] for k, v in enrichment["product_attr_ids"].items()},
        "demand_features": torch.zeros(len(reps), 12),
    }

    # 2. Load full reconciled commercial events in-memory (NEVER mutating commercial_dev_events.parquet!)
    print("[stage5/test] Loading & reconciling full commercial timeline from dataset.xlsx...", flush=True)
    full_commercial = load_aligned_full_commercial_events(RAW_WORKBOOK_PATH, all_purchases, catalogue)

    # 3. Build strictly causal (< t) rolling feature builders over full timeline (opt-in allow_heldout_test=True)
    print("[stage5/test] Initializing strictly-causal rolling feature builders...", flush=True)
    normalizer = PointInTimeNormalizer.from_state(config["profile_normalizer"])
    profiles = build_customer_metadata(companies, vocabs)
    sales_branch = companies.set_index("customer_id")["sales_branch"].to_dict()
    test_pool = RollingCandidatePool(catalogue, all_purchases)

    dynamic = DynamicFeatureBuilder(
        all_purchases[["purchase_date", "customer_id", "product_id", "quantity"]],
        catalogue,
        companies,
        product_ids,
        statistics_policy=STATISTICS_POLICY_ROLLING,
        allow_heldout_test=True,
    )
    base = BaseNameRollingFeatureBuilder(
        all_purchases, catalogue, companies, product_ids, allow_heldout_test=True
    )
    comm = CommercialRollingFeatureBuilder(
        full_commercial, catalogue, product_ids, allow_heldout_test=True
    )
    specs = ParsedSpecRollingFeatureBuilder(
        all_purchases, catalogue, companies, product_ids, allow_heldout_test=True
    )
    familiarity = FamiliarityStore(
        all_purchases, catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    histories = dict(tuple(all_purchases.groupby("customer_id", sort=False)))

    # 4. Build chronological held-out Test baskets (>= 2026-02-19)
    cases, cohort_counts = build_test_basket_cases(all_purchases, product_ids, universe, test_pool)
    print(f"[stage5/test] Built {len(cases)} held-out test baskets: {json.dumps(cohort_counts)}", flush=True)

    (
        cids,
        dates,
        first_flags,
        has_modeled_flags,
        targets,
        new_targets,
        allowed_arr,
        unseen_arr,
        extra_bn_arr,
        extra_new_bn_arr,
        sku_targets,
        sku_allowed,
        extra_sku_arr,
        extra_new_sku_arr,
    ) = zip(*cases, strict=True)
    days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]

    sequences = build_base_name_sequences(
        all_purchases,
        cids,
        days,
        universe,
        representatives,
        vocabs["category_vocab"],
        vocabs["subcategory_vocab"],
        vocabs["business_line_vocab"],
        allow_heldout_test=True,
    )

    print(f"[stage5/test] Computing strictly-causal features for {len(cases)} test baskets...", flush=True)
    features = np.empty((len(cases), len(reps), len(FEATURE_NAMES)), dtype=np.float16)
    profile8, profile6 = [], []
    for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
        features[i] = dated_candidate_features(cid, date, reps, dynamic, base, comm, specs)
        stats = compute_point_in_time_features(
            histories[cid],
            cid,
            date,
            normalizer=normalizer,
            include_median_purchase_quantity=True,
        )
        profile8.append([stats[k] for k in LEGACY_PROFILE_FEATURES])
        profile6.append([stats[k] for k in TRANSFERRED_CUSTOMER_STAT_NAMES])
        if (i + 1) % 2000 == 0 or (i + 1) == len(cases):
            print(f"[stage5/test] features {i + 1}/{len(cases)}", flush=True)

    test_data: Dict[str, Any] = {
        "customer_ids": list(cids),
        "prediction_days": torch.tensor(days, dtype=torch.float32),
        "first_observed": torch.tensor(first_flags, dtype=torch.bool),
        "has_modeled_target": torch.tensor(has_modeled_flags, dtype=torch.bool),
        "target_bn": torch.from_numpy(np.stack(targets)),
        "new_target_bn": torch.from_numpy(np.stack(new_targets)),
        "allowed_bn": torch.from_numpy(np.stack(allowed_arr)),
        "unseen_bn": torch.from_numpy(np.stack(unseen_arr)),
        "extra_bn_targets": np.asarray(extra_bn_arr, dtype=np.int64),
        "extra_new_bn_targets": np.asarray(extra_new_bn_arr, dtype=np.int64),
        "features_bn": torch.from_numpy(features),
        "feature_names": FEATURE_NAMES,
        "familiarity_bn": torch.cat(
            [
                familiarity.batch_features(cids[i : i + 256], dates[i : i + 256])
                for i in range(0, len(cases), 256)
            ]
        ),
        "text_embedding": torch.from_numpy(
            np.stack([texts.get(cid, np.zeros(1024, dtype=np.float32)) for cid in cids])
        ),
        "tabular_features": torch.tensor(profile8, dtype=torch.float32),
        "customer_stats_6": quantize_transferred_stats(np.asarray(profile6, dtype=np.float32)),
        **{
            k: torch.tensor([profiles.get(cid, {}).get(k, 0) for cid in cids], dtype=torch.long)
            for k in ("city_id", "region_id", "industry_id")
        },
        "sales_branch": torch.tensor(
            encode_categorical_values(
                [sales_branch.get(cid) for cid in cids],
                enrichment["customer_vocabs"]["sales_branch"],
            ),
            dtype=torch.long,
        ),
        **sequences,
    }
    validate_heldout_test_split(test_data)

    # Full prior SKU history for pack-level new-SKU metrics
    first_sku_sales = (
        all_purchases.groupby(["customer_id", "product_id"])["purchase_date"].min().unstack()
    )
    known_at = first_sku_sales.reindex(index=cids, columns=product_ids).to_numpy(dtype="datetime64[D]")
    sku_target_np = np.stack(sku_targets)
    sku_allowed_t = torch.from_numpy(np.stack(sku_allowed))
    sku_new_target_np = sku_target_np & ~(
        known_at < np.asarray(dates, dtype="datetime64[D]")[:, None]
    )
    extra_sku_np = np.asarray(extra_sku_arr, dtype=np.int64)
    extra_new_sku_np = np.asarray(extra_new_sku_arr, dtype=np.int64)

    # 5. Strictly inference-only forward passes through pinned Two-Tower -> Stage 1 MLP -> Stage 2 Rankers
    out_pred_dir = Path(saved_val["run_directory"]) / "heldout_test_eval"
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    print("[stage5/test] Running frozen Two-Tower inference...", flush=True)
    tower = _stage5_tower(universe, enrichment).to(dev)
    tt_ckpt = torch.load(reference["checkpoints"]["two_tower"]["path"], map_location="cpu", weights_only=False)
    tower.load_state_dict(tt_ckpt["model_state_dict"], strict=True)
    tower.eval()

    with torch.no_grad():
        p_dev = {
            "text_embedding": prod_inputs["text_embedding"].to(dev),
            "demand_features": prod_inputs["demand_features"].to(dev),
            "attr_ids": {k: v.to(dev) for k, v in prod_inputs["attr_ids"].items()},
        }
        v_bn = tower.product_vectors(p_dev)
        idxs = torch.arange(len(v_bn), device=dev)
        u_chunks, dot_chunks = [], []
        for rows in torch.arange(len(cids)).split(256):
            c_inp = {
                "text_embedding": test_data["text_embedding"][rows].to(dev),
                "city_id": test_data["city_id"][rows].to(dev),
                "region_id": test_data["region_id"][rows].to(dev),
                "industry_id": test_data["industry_id"][rows].to(dev),
                "tabular_features": test_data["tabular_features"][rows].to(dev),
                "sequence": test_data["sequence"][rows].to(dev),
                "attention_mask": test_data["attention_mask"][rows].to(dev),
                "seq_purchase_dates": test_data["seq_purchase_dates"][rows].to(dev),
                "seq_product_ids": test_data["seq_product_ids"][rows].to(dev),
                "extra_attr_ids": {"sales_branch": test_data["sales_branch"][rows].to(dev)},
            }
            u_b = tower.encode_purchase(c_inp, v_bn)
            sc_b = tower.score_cached(u_b, v_bn, idxs, test_data["familiarity_bn"][rows].to(dev))
            u_chunks.append(u_b.squeeze(1).cpu())
            dot_chunks.append(sc_b.cpu())
        u_test = torch.cat(u_chunks, dim=0)
        dot_test = torch.cat(dot_chunks, dim=0)
        if not torch.isfinite(u_test).all() or not torch.isfinite(dot_test).all():
            raise ValueError("Non-finite Two-Tower embeddings/scores on held-out test split!")
        v_bn_cpu = v_bn.cpu()

    unseen_ord = dot_test.masked_fill(~test_data["unseen_bn"], -torch.inf).argsort(dim=-1, descending=True, stable=True)
    avail_new = test_data["new_target_bn"] & test_data["unseen_bn"]
    tt_h25 = int(avail_new.gather(1, unseen_ord[:, :25]).sum().item())
    tt_h50 = int(avail_new.gather(1, unseen_ord[:, :50]).sum().item())

    print("[stage5/test] Running frozen Stage 1 MLP inference...", flush=True)
    s1_ckpt = torch.load(reference["checkpoints"]["selector"]["path"], map_location="cpu", weights_only=False)
    s1_model = RoadmapStage1Selector(FEATURE_NAMES, aux_60d_enabled=False).to(dev)
    s1_model.load_state_dict(s1_ckpt["model_state_dict"], strict=True)
    s1_model.eval()
    v_dev = v_bn_cpu.to(dev)

    with torch.no_grad():
        s1_chunks = []
        for st in range(0, len(cids), 128):
            en = min(st + 128, len(cids))
            lg = s1_model(
                u=u_test[st:en].to(dev),
                v=v_dev,
                features=test_data["features_bn"][st:en].to(dev),
                customer_stats_6=test_data["customer_stats_6"][st:en].to(dev),
            )
            s1_chunks.append(lg.masked_fill(~test_data["allowed_bn"][st:en].to(dev), -1e9).cpu())
        s1_scores = torch.cat(s1_chunks, dim=0)

    s1_ranked = s1_scores.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
    test_idx = select_candidates(s1_scores, test_data["allowed_bn"], test_data["unseen_bn"], SLATE_POLICY)
    test_mask = test_data["allowed_bn"].gather(1, test_idx)

    slate_cov_tot = int(test_data["target_bn"].gather(1, test_idx).sum().item())
    slate_cov_new = int(test_data["new_target_bn"].gather(1, test_idx).sum().item())

    def _eval_scopes(ranked_bn: np.ndarray, prefix: str) -> Dict[str, Any]:
        cold_mask = test_data["first_observed"]
        ret_mask = ~cold_mask
        mod_mask = test_data["has_modeled_target"]

        def _full_cohort(mask: Optional[torch.Tensor], save_name: Optional[str] = None) -> Dict[str, Any]:
            if mask is None:
                m_np = np.ones(len(cids), dtype=bool)
                m_t = torch.ones(len(cids), dtype=torch.bool)
            else:
                m_np = mask.numpy()
                m_t = mask
            sub_cids = [c for c, keep in zip(cids, m_np.tolist()) if keep]
            return evaluate_full_catalogue_predictions(
                ranked_bn[m_np],
                test_data["target_bn"][m_t],
                test_data["new_target_bn"][m_t],
                test_data["extra_bn_targets"][m_np],
                test_data["extra_new_bn_targets"][m_np],
                sub_cids,
                save_path=(out_pred_dir / save_name) if save_name else None,
            )

        def _mod_cohort(mask: torch.Tensor) -> Dict[str, Any]:
            m_t = mask & mod_mask
            m_np = m_t.numpy()
            sub_cids = [c for c, keep in zip(cids, m_np.tolist()) if keep]
            return evaluate_base_name_predictions(
                ranked_bn[m_np],
                test_data["target_bn"][m_t],
                test_data["new_target_bn"][m_t],
                sub_cids,
            )

        return {
            "full_catalogue": {
                "all": _full_cohort(None, save_name=f"{prefix}_full_catalogue.npz"),
                "first_observed": _full_cohort(cold_mask),
                "returning": _full_cohort(ret_mask),
            },
            "modeled_universe": {
                "all": _mod_cohort(torch.ones(len(cids), dtype=torch.bool)),
                "first_observed": _mod_cohort(cold_mask),
                "returning": _mod_cohort(ret_mask),
            },
        }

    s1_eval = _eval_scopes(s1_ranked, "stage1_mlp")

    print("[stage5/test] Running frozen Stage 2 Ranker inference (NDCG & Discovery checkpoints)...", flush=True)
    orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]
    n_bn = universe["num_base_names"]
    s2_results: Dict[str, Any] = {}
    mod_mask_np = test_data["has_modeled_target"].numpy()

    for sel_name, ref_key in (("ndcg", "ranker"), ("discovery", "discovery_ranker")):
        ck = torch.load(reference["checkpoints"][ref_key]["path"], map_location="cpu", weights_only=False)
        eval_m = Stage5BaseNameRanker(
            num_base_names=n_bn,
            num_features=len(FEATURE_NAMES),
            num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
            num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
            num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
            num_categories=len(vocabs["category_vocab"]) + 1,
            num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
            num_business_lines=len(vocabs["business_line_vocab"]) + 1,
            aux_60d_enabled=False,
            seed=reference["seed"],
        ).to(dev)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        eval_m.eval()

        with torch.no_grad():
            chunks = []
            for rows in torch.arange(len(cids)).split(128):
                c_b = test_idx[rows]
                row_ar = rows[:, None]
                pf = test_data["features_bn"][row_ar, c_b].to(dev, dtype=torch.float32)
                c_dev = c_b.to(dev)
                inp = {
                    "text_embedding": test_data["text_embedding"][rows].to(dev),
                    "city_id": test_data["city_id"][rows].to(dev),
                    "region_id": test_data["region_id"][rows].to(dev),
                    "industry_id": test_data["industry_id"][rows].to(dev),
                    "tabular_features": test_data["tabular_features"][rows].to(dev),
                    "sequence": test_data["sequence"][rows].to(dev),
                    "attention_mask": test_data["attention_mask"][rows].to(dev),
                    "seq_product_ids": test_data["seq_product_ids"][rows].to(dev),
                    "seq_category_ids": test_data["seq_category_ids"][rows].to(dev),
                    "seq_subcategory_ids": test_data["seq_subcategory_ids"][rows].to(dev),
                    "seq_business_line_ids": test_data["seq_business_line_ids"][rows].to(dev),
                    "seq_purchase_dates": test_data["seq_purchase_dates"][rows].to(dev),
                    "prediction_day": test_data["prediction_days"][rows].to(dev),
                    "pairwise_features": pf,
                    "candidate_vectors": v_dev[c_dev],
                    "candidate_mask": test_mask[rows].to(dev),
                    "candidate_indices": c_dev,
                }
                chunks.append(eval_m(**inp).logits.cpu())
            s2_logits = torch.cat(chunks, dim=0)

        ranked_bn = test_idx.gather(1, s2_logits.argsort(dim=-1, descending=True, stable=True)[:, :50]).numpy()
        scopes = _eval_scopes(ranked_bn, f"stage2_{sel_name}")

        resolved_skus = resolve_base_names_to_skus(
            ranked_bn,
            cids,
            days,
            sku_allowed_t,
            all_purchases,
            catalogue,
            product_ids,
            universe,
            top_k=10,
        )
        sku_full = _evaluate_resolved_skus(
            resolved_skus, sku_target_np, sku_new_target_np, extra_sku_targets=extra_sku_np
        )
        sku_full["total_new_sku_targets"] = int(sku_new_target_np.sum() + extra_new_sku_np.sum())
        sku_mod = _evaluate_resolved_skus(
            resolved_skus, sku_target_np, sku_new_target_np, extra_sku_targets=None, row_mask=mod_mask_np
        )
        sku_mod["total_new_sku_targets"] = int(sku_new_target_np[mod_mask_np].sum())

        s2_results[sel_name] = {
            "epoch": int(ck["epoch"]),
            "checkpoint_path": reference["checkpoints"][ref_key]["path"],
            "checkpoint_sha256": reference["checkpoints"][ref_key]["sha256"],
            **scopes,
            "resolved_sku_metrics": {
                "full_catalogue": sku_full,
                "modeled_universe": sku_mod,
            },
        }

    # 6. Post-execution integrity verification (zero mutation of checkpoints, dev caches, or preserved hashes)
    if snapshot_preserved_hashes() != before_preserved:
        raise AssertionError("Preserved reference hashes changed during test evaluation!")
    if _snapshot_dev_source_hashes() != before_dev_sources:
        raise AssertionError("Development source/cache hashes changed during test evaluation!")
    _verify_reference_files(reference)

    total_all_targets = int(test_data["target_bn"].sum().item() + test_data["extra_bn_targets"].sum())
    total_all_new = int(test_data["new_target_bn"].sum().item() + test_data["extra_new_bn_targets"].sum())
    total_mod_targets = int(test_data["target_bn"].sum().item())
    total_mod_new = int(test_data["new_target_bn"].sum().item())

    summary: Dict[str, Any] = {
        "evaluation": "stage5_heldout_test_final",
        "reference_id": reference["reference_id"],
        "seed": int(reference["seed"]),
        "heldout_access": True,
        "inference_only": True,
        "test_date_range": [
            str(min(dates).date()),
            str(max(dates).date()),
        ],
        "num_base_names": int(n_bn),
        "trained_base_names": int(universe["trained_base_names"]),
        "cohort_counts": cohort_counts,
        "two_tower": {
            "selected_epoch": int(tt_ckpt["epoch"]),
            "true_new_bn_hits@25": tt_h25,
            "true_new_bn_hits@50": tt_h50,
            "true_new_bn_targets_modeled": total_mod_new,
            "true_new_bn_targets_full_catalogue": total_all_new,
            "true_new_bn_recall@25_modeled": float(tt_h25 / max(total_mod_new, 1)),
            "true_new_bn_recall@50_modeled": float(tt_h50 / max(total_mod_new, 1)),
            "true_new_bn_recall@25_full_catalogue": float(tt_h25 / max(total_all_new, 1)),
            "true_new_bn_recall@50_full_catalogue": float(tt_h50 / max(total_all_new, 1)),
        },
        "stage1_mlp": {
            "selected_epoch": int(s1_ckpt["epoch"]),
            **s1_eval,
        },
        "stage2_slate_coverage": {
            "total_covered": slate_cov_tot,
            "total_targets_modeled": total_mod_targets,
            "total_targets_full_catalogue": total_all_targets,
            "total_coverage_rate_modeled": float(slate_cov_tot / max(total_mod_targets, 1)),
            "total_coverage_rate_full_catalogue": float(slate_cov_tot / max(total_all_targets, 1)),
            "true_new_covered": slate_cov_new,
            "true_new_targets_modeled": total_mod_new,
            "true_new_targets_full_catalogue": total_all_new,
            "true_new_coverage_rate_modeled": float(slate_cov_new / max(total_mod_new, 1)),
            "true_new_coverage_rate_full_catalogue": float(slate_cov_new / max(total_all_new, 1)),
        },
        "stage2_ranker": s2_results,
        "validation_reference": {
            "summary_path": reference["checkpoints"]["summary"]["path"],
            "summary_sha256": reference["checkpoints"]["summary"]["sha256"],
            "ndcg_all": saved_val["stage2_ranker"]["ndcg"]["bn_metrics"],
            "ndcg_cohorts": saved_val["stage2_ranker"]["ndcg"]["cohorts"],
            "ndcg_resolved_sku": saved_val["stage2_ranker"]["ndcg"]["resolved_sku_metrics"],
            "discovery_all": saved_val["stage2_ranker"]["discovery"]["bn_metrics"],
            "discovery_cohorts": saved_val["stage2_ranker"]["discovery"]["cohorts"],
            "discovery_resolved_sku": saved_val["stage2_ranker"]["discovery"]["resolved_sku_metrics"],
            "slate_coverage": saved_val["stage2_slate_coverage"],
        },
        "integrity_audit": {
            "preserved_hashes_verified": True,
            "reference_checkpoints_verified": True,
            "dev_sources_unchanged": True,
            "dev_events_sha256": dev_digest,
            "catalogue_sha256": cat_sha,
            "test_row_keys_sha256": json_digest(
                list(zip(test_data["customer_ids"], test_data["prediction_days"].int().tolist()))
            ),
            "commercial_dev_parquet_sha256": before_dev_sources[COMMERCIAL_PARQUET_PATH.name],
            "reconciled_paid_commercial_lines": int(len(full_commercial)),
            "reconciled_cleaned_purchase_rows": int(len(all_purchases)),
            "test_companies_with_cached_text": int(len(test_cids_all & set(texts.keys()))),
            "test_companies_total": int(len(test_cids_all)),
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    atomic_json(summary, summary_output_path, overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-heldout-test",
        action="store_true",
        help="Explicitly authorize final inference-only evaluation on held-out Test (>= 2026-02-19).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=TEST_SUMMARY_PATH)
    args = parser.parse_args()
    summary = run_heldout_test_evaluation(
        allow_heldout_test=args.allow_heldout_test,
        device=args.device,
        summary_output_path=args.output,
    )
    print("\n================ STAGE 5 HELD-OUT TEST SUMMARY ================", flush=True)
    print(f"Reference: {summary['reference_id']} | Test window: {summary['test_date_range']}", flush=True)
    print("Slate coverage:", json.dumps(summary["stage2_slate_coverage"], indent=2))
    for sel in ("ndcg", "discovery"):
        print(f"\n--- Stage 2 [{sel}] (Full Catalogue, {summary['cohort_counts']['all_baskets']} baskets) ---")
        print(json.dumps(summary["stage2_ranker"][sel]["full_catalogue"]["all"], indent=2))
        print(f"--- Stage 2 [{sel}] (Modeled 585-BN Universe, {summary['cohort_counts']['modeled_baskets']} baskets) ---")
        print(json.dumps(summary["stage2_ranker"][sel]["modeled_universe"]["all"], indent=2))
        print(f"--- Stage 2 [{sel}] Resolved SKU ---")
        print(json.dumps(summary["stage2_ranker"][sel]["resolved_sku_metrics"], indent=2))


if __name__ == "__main__":
    main()
