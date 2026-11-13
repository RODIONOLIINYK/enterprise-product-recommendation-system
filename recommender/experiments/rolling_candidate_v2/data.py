"""Strictly-causal rolling feature preparation, request adapter, and Stage 2 slate cache (v2)."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.candidates import (
    STATISTICS_POLICY_FROZEN,
    STATISTICS_POLICY_ROLLING,
    VALID_STATISTICS_POLICIES,
)
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import sliced
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import (
    DEFAULT_NEW_STATISTICS_POLICY,
    NAMES as DYNAMIC_FEATURE_NAMES,
    contract as feature_contract_for_policy,
    quantize_features,
    schema_digest as feature_schema_digest_for_policy,
)
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.mlp_feature_ablation.config import (
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
)
from recommender.experiments.mlp_feature_ablation.model import (
    FeatureAblatedStage1Selector,
    load_ablated_selector_checkpoint,
)
from recommender.experiments.mlp_feature_ablation.schema import (
    FeatureAblationSpec,
    build_feature_ablation_spec,
    build_input_contract,
    validate_input_contract,
)
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_ranker import RankerDataset, collate_ranker_batch

from .config import (
    BATCH_SIZE,
    CACHE_ROOT,
    CANDIDATE_SOURCE_REFERENCE_DOT,
    CANDIDATE_SOURCE_REFERENCE_SELECTOR,
    CANDIDATE_SOURCE_ROLLING_SELECTOR,
    DEFAULT_DISCOVERY_RANKER,
    DEFAULT_RETRIEVER,
    DEFAULT_SLATE_CANDIDATE_SOURCE,
    E1_E5_RETRIEVAL_CACHE_DIR,
    EXPECTED_CONTRACT_DIGEST,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PRODUCT_UNIVERSE,
    EXPECTED_REFERENCE_ID,
    EXPECTED_RETRIEVER_SHA256,
    EXPECTED_SELECTOR_DIM,
    EXPECTED_SELECTOR_RECIPE,
    EXPECTED_SELECTOR_SHA256,
    EXPECTED_SELECTOR_VARIANT,
    EXPECTED_TRAIN_PURCHASE_CACHE_SHA256,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SCORES_SHA256,
    EXPECTED_UPSTREAM_VARIANT,
    EXPECTED_VAL_NEW_TARGETS,
    EXPECTED_VAL_PURCHASE_CACHE_SHA256,
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SCORES_SHA256,
    EXPECTED_VAL_TOTAL_TARGETS,
    MIN_ELIGIBLE_SKUS,
    SEED,
    SELECTOR_SOURCE_RUN_DIR,
    SLATE_POLICY,
    SLATE_SIZE,
    STAGE1_MLP_REF_MANIFEST,
    STAGE1_MLP_REF_POINTER,
    STAGE1_MLP_REF_RETRIEVER,
    STAGE1_MLP_REF_SELECTOR,
    UPSTREAM_STAGE1_RUN_DIR,
)


FEATURE_CACHE_VERSION = 3
SELECTOR_SCORE_CACHE_VERSION = 1
SLATE_CACHE_VERSION = 3
ROLLING_CACHE_NAMESPACE = "rolling_candidate_v2_v3"

_PURCHASE_DIGEST_COLUMNS = (
    "customer_id",
    "purchase_date",
    "product_id",
    "quantity",
    "product_category",
    "business_line",
    "product_name",
    "item_type",
    "transaction_type",
)
_CATALOGUE_DIGEST_COLUMNS = (
    "product_id",
    "brand",
    "product_category",
    "product_subcategory",
    "business_line",
    "product_family",
)
_COMPANY_DIGEST_COLUMNS = (
    "customer_id",
    "industry_branch",
    "city",
    "region_code",
)


def expected_selector_spec() -> FeatureAblationSpec:
    """Canonical 301D FeatureAblatedStage1Selector spec for repeat-no-month-v1."""
    return build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )


def _hash_canonical_dataframe(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    numeric_cols: Sequence[str] = (),
    date_cols: Sequence[str] = (),
) -> str:
    """Order-invariant deterministic SHA-256 over canonicalized DataFrame columns."""
    norm = pd.DataFrame(index=df.index)
    numeric_set = set(numeric_cols)
    date_set = set(date_cols)
    for col in columns:
        if col not in df.columns:
            if col in numeric_set:
                norm[col] = np.zeros(len(df), dtype=np.float64)
            elif col in date_set:
                norm[col] = np.zeros(len(df), dtype=np.int64)
            else:
                norm[col] = pd.Series([""] * len(df), index=df.index, dtype=object)
            continue
        if col in date_set:
            ts = pd.to_datetime(df[col], format="mixed")
            if bool(ts.isna().any()):
                raise ValueError(f"Missing/invalid timestamp in source digest column {col!r}.")
            norm[col] = ts.astype("datetime64[ns]").astype(np.int64)
        elif col in numeric_set:
            arr = pd.to_numeric(df[col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=True)
            arr[arr == 0.0] = 0.0
            norm[col] = arr
        else:
            s = df[col].fillna("").astype(str).str.strip()
            norm[col] = s

    sorted_df = norm.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"rows:{len(sorted_df)}\n".encode("utf-8"))
    for col in columns:
        h.update(f"col:{col}\n".encode("utf-8"))
        if col in date_set:
            arr = np.ascontiguousarray(sorted_df[col].to_numpy(dtype="<i8"))
            h.update(b"dtype:i8:")
            h.update(arr.tobytes())
        elif col in numeric_set:
            arr = np.ascontiguousarray(sorted_df[col].to_numpy(dtype="<f8"))
            h.update(b"dtype:f8:")
            h.update(arr.tobytes())
        else:
            h.update(b"dtype:str:")
            for val in sorted_df[col].tolist():
                raw = val.encode("utf-8")
                h.update(struct.pack("<I", len(raw)))
                h.update(raw)
    return h.hexdigest()


def compute_rolling_source_digest(
    purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute stable SHA-256 digest of date-gated (< TEST_START) dev purchases and used catalogue/company inputs."""
    if not purchases.empty:
        dates = pd.to_datetime(purchases["purchase_date"], format="mixed")
        if bool(dates.isna().any()) or not bool(dates.lt(TEST_START).all()):
            raise ValueError(
                f"Source digest refuses held-out or invalid purchase dates (>= {TEST_START.date()})."
            )
        max_date = str(dates.max().date())
    else:
        max_date = None

    purchases_sha = _hash_canonical_dataframe(
        purchases,
        _PURCHASE_DIGEST_COLUMNS,
        numeric_cols=("quantity",),
        date_cols=("purchase_date",),
    )
    catalogue_sha = _hash_canonical_dataframe(catalogue, _CATALOGUE_DIGEST_COLUMNS)
    companies_sha = _hash_canonical_dataframe(companies, _COMPANY_DIGEST_COLUMNS)
    combined_payload = {
        "version": 1,
        "source_purchases_sha256": purchases_sha,
        "source_catalogue_sha256": catalogue_sha,
        "source_companies_sha256": companies_sha,
        "dev_purchase_rows": int(len(purchases)),
        "catalogue_rows": int(len(catalogue)),
        "company_rows": int(len(companies)),
        "max_purchase_date": max_date,
    }
    return {
        **combined_payload,
        "observed_source_digest": json_digest(combined_payload),
    }



def assert_dev_dates_strict(split: str, prediction_days: torch.Tensor) -> None:
    """Enforce zero held-out access (>= 2026-02-19) and split boundary invariance."""
    origin = pd.Timestamp("2025-01-01")
    train_cutoff_day = (TRAIN_CUTOFF - origin).days
    test_start_day = (TEST_START - origin).days
    if bool((prediction_days >= test_start_day).any()):
        raise ValueError(f"Held-out dates >= {TEST_START.date()} detected in {split} split.")
    if split == "train" and bool((prediction_days >= train_cutoff_day).any()):
        raise ValueError(f"Validation/test dates >= {TRAIN_CUTOFF.date()} detected in train split.")
    if split == "val" and bool((prediction_days < train_cutoff_day).any()):
        raise ValueError(f"Training dates < {TRAIN_CUTOFF.date()} detected in val split.")


def _validate_new_statistics_policy(statistics_policy: str, *, allow_legacy_frozen: bool = False) -> str:
    if statistics_policy not in VALID_STATISTICS_POLICIES:
        raise ValueError(
            f"Unknown statistics_policy {statistics_policy!r}; expected one of {VALID_STATISTICS_POLICIES}."
        )
    if statistics_policy == STATISTICS_POLICY_FROZEN and not allow_legacy_frozen:
        raise ValueError(
            "statistics_policy='frozen_train_cutoff_v1' is legacy-only and prohibited for new feature preparation "
            "or inference. Use 'rolling_observed_v2' (or pass allow_legacy_frozen=True for historical replay)."
        )
    return statistics_policy


def verify_active_reference(*, load_selector: bool = True) -> Dict[str, Any]:
    """Verify frozen repeat-no-month-v1 pointer, manifest, selector, retriever, and source caches."""
    required_paths = (
        STAGE1_MLP_REF_POINTER,
        STAGE1_MLP_REF_MANIFEST,
        STAGE1_MLP_REF_RETRIEVER,
        STAGE1_MLP_REF_SELECTOR,
        SELECTOR_SOURCE_RUN_DIR / "train_scores.pt",
        SELECTOR_SOURCE_RUN_DIR / "val_scores.pt",
        UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
        UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
        DEFAULT_RETRIEVER,
        DEFAULT_DISCOVERY_RANKER,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")

    ref_ptr = json.loads(STAGE1_MLP_REF_POINTER.read_text(encoding="utf-8"))
    if ref_ptr.get("active_reference_id") != EXPECTED_REFERENCE_ID:
        raise ValueError(
            f"Expected active_reference_id={EXPECTED_REFERENCE_ID!r}, got {ref_ptr.get('active_reference_id')!r}."
        )
    if ref_ptr.get("heldout_access") is not False:
        raise ValueError("Reference pointer must record heldout_access=false.")
    if ref_ptr.get("upstream_variant") != EXPECTED_UPSTREAM_VARIANT:
        raise ValueError(f"Unexpected upstream_variant: {ref_ptr.get('upstream_variant')!r}")
    if ref_ptr.get("variant") != EXPECTED_SELECTOR_VARIANT:
        raise ValueError(f"Unexpected selector variant: {ref_ptr.get('variant')!r}")
    if ref_ptr.get("selector_recipe") != EXPECTED_SELECTOR_RECIPE:
        raise ValueError(f"Unexpected selector_recipe: {ref_ptr.get('selector_recipe')!r}")
    if int(ref_ptr.get("selector_input_dim", -1)) != EXPECTED_SELECTOR_DIM:
        raise ValueError(f"Unexpected selector_input_dim: {ref_ptr.get('selector_input_dim')}")

    expected_hashes = {
        "reference_manifest": (STAGE1_MLP_REF_MANIFEST, EXPECTED_MANIFEST_SHA256),
        "best_retriever": (STAGE1_MLP_REF_RETRIEVER, EXPECTED_RETRIEVER_SHA256),
        "best_selector": (STAGE1_MLP_REF_SELECTOR, EXPECTED_SELECTOR_SHA256),
        "train_scores": (SELECTOR_SOURCE_RUN_DIR / "train_scores.pt", EXPECTED_TRAIN_SCORES_SHA256),
        "val_scores": (SELECTOR_SOURCE_RUN_DIR / "val_scores.pt", EXPECTED_VAL_SCORES_SHA256),
        "train_purchase_cache": (
            UPSTREAM_STAGE1_RUN_DIR / "train_purchase_cache.pt",
            EXPECTED_TRAIN_PURCHASE_CACHE_SHA256,
        ),
        "val_purchase_cache": (
            UPSTREAM_STAGE1_RUN_DIR / "val_purchase_cache.pt",
            EXPECTED_VAL_PURCHASE_CACHE_SHA256,
        ),
    }
    actual_hashes: Dict[str, str] = {"reference_json": file_digest(STAGE1_MLP_REF_POINTER)}
    for label, (path, exp_sha) in expected_hashes.items():
        actual = file_digest(path)
        if actual != exp_sha:
            raise ValueError(f"Hash mismatch for {label} ({path}): {actual} != {exp_sha}")
        actual_hashes[label] = actual

    manifest = json.loads(STAGE1_MLP_REF_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("reference_id") != EXPECTED_REFERENCE_ID:
        raise ValueError("Reference manifest reference_id mismatch.")

    spec = expected_selector_spec()
    expected_contract = build_input_contract(spec)
    if expected_contract["contract_digest"] != EXPECTED_CONTRACT_DIGEST:
        raise AssertionError(
            f"Contract digest mismatch: {expected_contract['contract_digest']} != {EXPECTED_CONTRACT_DIGEST}."
        )
    selector = None
    selector_ckpt = None
    if load_selector:
        selector, selector_ckpt = load_ablated_selector_checkpoint(
            STAGE1_MLP_REF_SELECTOR,
            expected_spec=spec,
            map_location="cpu",
        )
        selector.eval()
        selector.requires_grad_(False)
        selector_upstream = selector_ckpt.get("upstream_digest") or selector_ckpt.get("settings", {}).get("upstream_digest")
        if selector_upstream != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(
                f"Selector upstream_digest mismatch: {selector_upstream} != {EXPECTED_RETRIEVER_SHA256}."
            )
        if int(selector_ckpt.get("epoch", -1)) != int(ref_ptr["selector"]["selected_epoch"]):
            raise ValueError(
                f"Selector epoch mismatch: {selector_ckpt.get('epoch')} != {ref_ptr['selector']['selected_epoch']}."
            )

    actual_hashes["default_retriever"] = file_digest(DEFAULT_RETRIEVER)
    actual_hashes["default_discovery_ranker"] = file_digest(DEFAULT_DISCOVERY_RANKER)
    return {
        "reference_pointer": ref_ptr,
        "manifest": manifest,
        "selector": selector,
        "selector_ckpt": selector_ckpt,
        "selector_spec": spec,
        "selector_input_contract": expected_contract,
        "hashes": actual_hashes,
    }


class RollingDynamicFeatureAdapter:
    """Request-time strictly-causal rolling feature adapter with float16->float32 cache parity."""

    def __init__(
        self,
        purchases: pd.DataFrame,
        catalogue: pd.DataFrame,
        companies: pd.DataFrame,
        product_ids: Sequence[str],
        *,
        relationships: bool = True,
        statistics_policy: str = DEFAULT_NEW_STATISTICS_POLICY,
        allow_legacy_frozen: bool = False,
    ):
        policy = _validate_new_statistics_policy(statistics_policy, allow_legacy_frozen=allow_legacy_frozen)
        self.statistics_policy = policy
        self.feature_contract = feature_contract_for_policy(policy)
        self.schema_digest = feature_schema_digest_for_policy(policy)
        self.product_ids = tuple(product_ids)
        self.builder = DynamicFeatureBuilder(
            purchases,
            catalogue,
            companies,
            self.product_ids,
            relationships=relationships,
            statistics_policy=policy,
            allow_non_monotonic=True,
        )

    @classmethod
    def from_development_data(
        cls,
        product_ids: Sequence[str],
        *,
        purchases_path: Union[str, Path] = "data/interim/cleaned_purchases.csv",
        catalogue_path: Optional[Union[str, Path]] = None,
        companies_path: Optional[Union[str, Path]] = None,
        relationships: bool = True,
        statistics_policy: str = DEFAULT_NEW_STATISTICS_POLICY,
        allow_legacy_frozen: bool = False,
    ) -> "RollingDynamicFeatureAdapter":
        """Construct adapter strictly from development purchases (< TEST_START).

        Note: Development-only adapter; externally supplied u/v/stats timestamps cannot be
        verified here and callers must ensure they reflect state strictly prior to as_of_date.
        """
        purchases = load_development_purchases(str(purchases_path))
        cat_path = Path(catalogue_path) if catalogue_path is not None else get_datasets_locs("items")
        comp_path = Path(companies_path) if companies_path is not None else get_datasets_locs("companies")
        catalogue = load_product_catalogue(cat_path)
        companies = load_companies(comp_path, purchases=purchases, minimum_purchases=1)
        return cls(
            purchases,
            catalogue,
            companies,
            product_ids,
            relationships=relationships,
            statistics_policy=statistics_policy,
            allow_legacy_frozen=allow_legacy_frozen,
        )

    def build_raw(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        *,
        include_relationships: Optional[bool] = None,
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Build raw float32 41D rolling features strictly prior to as_of_date."""
        date_ts = pd.Timestamp(as_of_date)
        if pd.isna(date_ts) or date_ts >= TEST_START:
            raise ValueError("Development predictions only")
        return self.builder.build(
            customer_id,
            date_ts,
            include_relationships=include_relationships,
            candidate_ids=candidate_ids,
        )

    def build_quantized(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        *,
        include_relationships: Optional[bool] = None,
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Build float16->float32 quantized 41D rolling features matching sealed disk cache."""
        raw = self.build_raw(
            customer_id,
            as_of_date,
            include_relationships=include_relationships,
            candidate_ids=candidate_ids,
        )
        return quantize_features(raw)

    def build_tensor(
        self,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        *,
        include_relationships: Optional[bool] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
        unsqueeze_batch: bool = True,
    ) -> torch.Tensor:
        """Build quantized float32 tensor shaped (1, C, 41) or (C, 41)."""
        arr = self.build_quantized(
            customer_id,
            as_of_date,
            include_relationships=include_relationships,
            candidate_ids=candidate_ids,
        )
        tensor = torch.from_numpy(np.array(arr, dtype=np.float32))
        if unsqueeze_batch:
            tensor = tensor.unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @torch.no_grad()
    def score_active_selector(
        self,
        selector: FeatureAblatedStage1Selector,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        customer_stats_6: torch.Tensor,
        customer_id: str,
        as_of_date: Union[str, pd.Timestamp],
        candidate_ids: Optional[Sequence[str]] = None,
        allowed_mask: Optional[torch.Tensor] = None,
        v_product_ids: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Evaluate frozen active selector (301D) with live strictly-causal rolling features."""
        date_ts = pd.Timestamp(as_of_date)
        if pd.isna(date_ts) or date_ts >= TEST_START:
            raise ValueError("Development predictions only")
        if not isinstance(selector, FeatureAblatedStage1Selector):
            raise TypeError(
                f"Expected FeatureAblatedStage1Selector, got {type(selector).__name__}."
            )
        expected_contract = build_input_contract(expected_selector_spec())
        validate_input_contract(selector.input_contract(), expected_contract)
        selector.eval()

        if not isinstance(u, torch.Tensor):
            raise TypeError(f"Expected u torch.Tensor, got {type(u).__name__}.")
        if u.ndim == 1:
            u_batch = u.unsqueeze(0)
        elif u.ndim == 2 and u.shape[0] == 1:
            u_batch = u
        else:
            raise ValueError(
                f"Expected single-customer u shape ({selector.embedding_dim},) or (1, {selector.embedding_dim}), "
                f"got {tuple(u.shape)}."
            )
        if u_batch.shape[1] != selector.embedding_dim:
            raise ValueError(
                f"Expected u embedding dim {selector.embedding_dim}, got {u_batch.shape[1]}."
            )
        if not bool(torch.isfinite(u_batch).all()):
            raise ValueError("u contains non-finite values.")

        if not isinstance(customer_stats_6, torch.Tensor):
            raise TypeError(
                f"Expected customer_stats_6 torch.Tensor, got {type(customer_stats_6).__name__}."
            )
        if customer_stats_6.ndim == 1:
            stats_batch = customer_stats_6.unsqueeze(0)
        elif customer_stats_6.ndim == 2 and customer_stats_6.shape[0] == 1:
            stats_batch = customer_stats_6
        else:
            raise ValueError(
                f"Expected single-customer customer_stats_6 shape ({selector.num_customer_stats},) or "
                f"(1, {selector.num_customer_stats}), got {tuple(customer_stats_6.shape)}."
            )
        if stats_batch.shape[1] != selector.num_customer_stats:
            raise ValueError(
                f"Expected customer_stats_6 width {selector.num_customer_stats}, got {stats_batch.shape[1]}."
            )
        if not bool(torch.isfinite(stats_batch).all()):
            raise ValueError("customer_stats_6 contains non-finite values.")

        if v_product_ids is not None:
            v_pids_tuple = tuple(str(pid) for pid in v_product_ids)
            if v_pids_tuple != self.product_ids:
                raise ValueError(
                    "v_product_ids does not match canonical adapter product_ids order or membership."
                )
        if not isinstance(v, torch.Tensor) or v.ndim != 2 or v.shape != (
            len(self.product_ids),
            selector.embedding_dim,
        ):
            raise ValueError(
                f"Expected canonical full-universe v shape ({len(self.product_ids)}, {selector.embedding_dim}), "
                f"got {tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__}."
            )
        if not bool(torch.isfinite(v).all()):
            raise ValueError("v contains non-finite values.")

        target_device = u_batch.device
        if v.device != target_device:
            raise ValueError(f"v device ({v.device}) must match u device ({target_device}).")
        if stats_batch.device != target_device:
            raise ValueError(
                f"customer_stats_6 device ({stats_batch.device}) must match u device ({target_device})."
            )
        first_param = next(selector.parameters(), None)
        if first_param is not None and first_param.device != target_device:
            raise ValueError(
                f"selector device ({first_param.device}) must match u device ({target_device})."
            )

        if candidate_ids is not None:
            cand_list = [str(pid) for pid in candidate_ids]
            if not cand_list:
                raise ValueError("candidate_ids must be non-empty.")
            if len(cand_list) != len(set(cand_list)):
                raise ValueError("Duplicate candidate_ids provided to score_active_selector.")
            idx_map = self.builder.index
            unknown = [pid for pid in cand_list if pid not in idx_map]
            if unknown:
                raise ValueError(f"Unknown candidate_ids not in canonical product_ids: {unknown[:5]}")
            idx = torch.tensor([idx_map[pid] for pid in cand_list], dtype=torch.long, device=v.device)
            v_cand = v[idx]
            num_cands = len(cand_list)
        else:
            cand_list = None
            v_cand = v
            num_cands = len(self.product_ids)

        mask: Optional[torch.Tensor] = None
        if allowed_mask is not None:
            if not isinstance(allowed_mask, torch.Tensor):
                raise TypeError(f"allowed_mask must be a torch.Tensor, got {type(allowed_mask).__name__}.")
            if allowed_mask.dtype != torch.bool:
                raise ValueError(f"allowed_mask must have dtype torch.bool, got {allowed_mask.dtype}.")
            if allowed_mask.ndim == 1:
                mask = allowed_mask.unsqueeze(0)
            elif allowed_mask.ndim == 2 and allowed_mask.shape[0] == 1:
                mask = allowed_mask
            else:
                raise ValueError(
                    f"Expected allowed_mask shape ({num_cands},) or (1, {num_cands}), "
                    f"got {tuple(allowed_mask.shape)}."
                )
            if mask.shape != (1, num_cands):
                raise ValueError(
                    f"Expected allowed_mask shape (1, {num_cands}) matching candidate count, "
                    f"got {tuple(mask.shape)}."
                )

        features_41 = self.build_tensor(
            customer_id,
            as_of_date,
            candidate_ids=cand_list,
            device=target_device,
            unsqueeze_batch=True,
        )
        logits = selector(
            u=u_batch.to(dtype=torch.float32),
            v=v_cand.to(dtype=torch.float32),
            features_41=features_41,
            customer_stats_6=stats_batch.to(dtype=torch.float32),
        )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Active selector produced non-finite logits.")
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), -torch.inf)
        return logits


def _cohort_slices(
    train_ret: Mapping[str, Any],
    val_ret: Mapping[str, Any],
    train_src: Mapping[str, Any],
    val_src: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[torch.Tensor]]:
    sources, retrieval, row_masks = [], [], []
    for split, source, data, expected_rows in (
        ("train", train_src, train_ret, EXPECTED_TRAIN_ROWS),
        ("val", val_src, val_ret, EXPECTED_VAL_ROWS),
    ):
        rows = torch.where(data["allowed"].sum(1) >= MIN_ELIGIBLE_SKUS)[0]
        if len(rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} {split} rows, got {len(rows)}.")
        sliced_src = sliced(source, rows, len(source["customer_ids"]))
        sliced_ret = sliced(data, rows, len(source["customer_ids"]))
        assert_dev_dates_strict(split, sliced_src["prediction_days"])
        sources.append(sliced_src)
        retrieval.append(sliced_ret)
        row_masks.append(rows)
    return sources, retrieval, row_masks


def validate_rolling_feature_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_identity: str,
    expected_policy: str = STATISTICS_POLICY_ROLLING,
    expected_observed_source_digest: Optional[str] = None,
) -> None:
    """Strictly verify rolling feature manifest and reject legacy frozen/stale policies/caches."""
    if int(manifest.get("version", -1)) != FEATURE_CACHE_VERSION:
        raise ValueError(
            f"Feature cache version mismatch: {manifest.get('version')!r} != {FEATURE_CACHE_VERSION}. "
            "Refusing to mix legacy frozen or unsealed v2 cache with versioned v3 rolling cache."
        )
    if manifest.get("namespace") != ROLLING_CACHE_NAMESPACE:
        raise ValueError(
            f"Feature cache namespace mismatch: {manifest.get('namespace')!r} != {ROLLING_CACHE_NAMESPACE!r}."
        )
    if manifest.get("test_access") is not False or manifest.get("heldout_access") is not False:
        raise ValueError("Feature cache manifest must record test_access=false and heldout_access=false.")
    if manifest.get("statistics_policy") != expected_policy:
        raise ValueError(
            f"Feature cache statistics_policy mismatch: {manifest.get('statistics_policy')!r} != {expected_policy!r}."
        )
    if manifest.get("feature_contract") != feature_contract_for_policy(expected_policy):
        raise ValueError("Feature cache feature_contract mismatch (legacy frozen contracts are incompatible).")
    if manifest.get("schema_digest") != feature_schema_digest_for_policy(expected_policy):
        raise ValueError("Feature cache schema_digest mismatch.")
    obs_dig = manifest.get("observed_source_digest")
    if not isinstance(obs_dig, str) or not obs_dig:
        raise ValueError("Feature cache manifest missing required observed_source_digest.")
    if expected_observed_source_digest is not None and obs_dig != expected_observed_source_digest:
        raise ValueError(
            f"Feature cache observed_source_digest mismatch: {obs_dig!r} != {expected_observed_source_digest!r}. "
            "Observed dev purchases or catalogue/company source inputs changed."
        )
    if manifest.get("identity") != expected_identity:
        raise ValueError("Feature cache identity mismatch.")


def prepare_or_load_rolling_feature_bundle(
    *,
    cache_root: Path = CACHE_ROOT,
    build: bool = True,
    statistics_policy: str = DEFAULT_NEW_STATISTICS_POLICY,
    allow_legacy_frozen: bool = False,
    purchases_path: Union[str, Path] = "data/interim/cleaned_purchases.csv",
    catalogue_path: Optional[Union[str, Path]] = None,
    companies_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build or strictly validate sealed rolling_observed_v2 41D feature bundle tied to repeat-no-month-v1 and current dev inputs."""
    policy = _validate_new_statistics_policy(statistics_policy, allow_legacy_frozen=allow_legacy_frozen)
    ref_info = verify_active_reference(load_selector=True)
    for name in ("train.pt", "val.pt", "manifest.pt"):
        if not (E1_E5_RETRIEVAL_CACHE_DIR / name).is_file():
            raise FileNotFoundError(
                f"Required retrieval cache {E1_E5_RETRIEVAL_CACHE_DIR / name} missing; refusing raw regeneration."
            )

    train_ret, val_ret, info, train_src, val_src = prepare_retrieval_data()
    if len(info["product_ids"]) != EXPECTED_PRODUCT_UNIVERSE:
        raise ValueError(
            f"Expected {EXPECTED_PRODUCT_UNIVERSE} product universe, got {len(info['product_ids'])}."
        )
    sources, retrieval, cohort_rows = _cohort_slices(train_ret, val_ret, train_src, val_src)

    purchases = load_development_purchases(str(purchases_path))
    cat_path = Path(catalogue_path) if catalogue_path is not None else get_datasets_locs("items")
    comp_path = Path(companies_path) if companies_path is not None else get_datasets_locs("companies")
    catalogue = load_product_catalogue(cat_path)
    companies = load_companies(comp_path, purchases=purchases, minimum_purchases=1)
    source_digest_info = compute_rolling_source_digest(purchases, catalogue, companies)
    observed_source_digest = source_digest_info["observed_source_digest"]

    expected_contract = feature_contract_for_policy(policy)
    expected_schema_digest = feature_schema_digest_for_policy(policy)
    upstream_digest = ref_info["hashes"]["best_retriever"]
    selector_digest = ref_info["hashes"]["best_selector"]

    identity = json_digest(
        {
            "cache_version": FEATURE_CACHE_VERSION,
            "namespace": ROLLING_CACHE_NAMESPACE,
            "statistics_policy": policy,
            "feature_schema": expected_schema_digest,
            "reference_id": EXPECTED_REFERENCE_ID,
            "upstream": upstream_digest,
            "selector": selector_digest,
            "source": info["identity"],
            "observed_source_digest": observed_source_digest,
            "train_rows": sources[0]["transition_keys"],
            "val_rows": sources[1]["transition_keys"],
            "minimum_eligible": MIN_ELIGIBLE_SKUS,
        }
    )

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "features_manifest.json"
    legacy_manifest_path = root / "manifest.json"
    if legacy_manifest_path.is_file() and not manifest_path.is_file():
        raise ValueError(
            f"Incompatible legacy manifest in {root}; refuse to reuse historical cache in v3 namespace."
        )

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_rolling_feature_manifest(
            manifest,
            expected_identity=identity,
            expected_policy=policy,
            expected_observed_source_digest=observed_source_digest,
        )
        if manifest.get("source_input_digests") != source_digest_info:
            raise ValueError("Feature cache source_input_digests mismatch.")
        for split in ("train", "val"):
            feat_path = root / f"{split}_features.npy"
            if not feat_path.is_file():
                raise FileNotFoundError(f"Missing feature array: {feat_path}")
            if file_digest(feat_path) != manifest["feature_sha256"][split]:
                raise ValueError(f"Rolling feature checksum mismatch for {split}.")
    else:
        if not build:
            raise FileNotFoundError(
                f"Rolling feature bundle missing in {root}; run preparation (--prepare-only) first."
            )
        if any(root.glob("*_features.npy")) or (root / "slates_manifest.json").is_file():
            raise ValueError(f"Incomplete or legacy files in {root}; refusing silent overwrite.")

        builder = DynamicFeatureBuilder(
            purchases,
            catalogue,
            companies,
            info["product_ids"],
            statistics_policy=policy,
            allow_non_monotonic=False,
        )
        arrays = [
            np.lib.format.open_memmap(
                root / f"{split}_features.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(src["customer_ids"]), len(info["product_ids"]), len(DYNAMIC_FEATURE_NAMES)),
            )
            for split, src in zip(("train", "val"), sources)
        ]
        queries = sorted(
            (int(day), split_idx, row_idx)
            for split_idx, src in enumerate(sources)
            for row_idx, day in enumerate(src["prediction_days"].tolist())
        )
        started = time.perf_counter()
        origin = pd.Timestamp("2025-01-01")
        for i, (day, split_idx, row_idx) in enumerate(queries):
            date = origin + pd.Timedelta(days=day)
            arrays[split_idx][row_idx] = builder.build(sources[split_idx]["customer_ids"][row_idx], date)
            if i % 5000 == 0 and i > 0:
                print(
                    f"[rolling_features] {i}/{len(queries)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        for array in arrays:
            array.flush()
        prep_seconds = time.perf_counter() - started
        manifest = {
            "version": FEATURE_CACHE_VERSION,
            "namespace": ROLLING_CACHE_NAMESPACE,
            "identity": identity,
            "statistics_policy": policy,
            "schema_digest": expected_schema_digest,
            "feature_contract": expected_contract,
            "active_reference_id": EXPECTED_REFERENCE_ID,
            "upstream_digest": upstream_digest,
            "selector_digest": selector_digest,
            "source_identity": info["identity"],
            "enrichment_identity": ref_info["manifest"]["source_identities"]["enrichment_identity"],
            "observed_source_digest": observed_source_digest,
            "source_input_digests": source_digest_info,
            "feature_sha256": {
                split: file_digest(root / f"{split}_features.npy") for split in ("train", "val")
            },
            "train_rows": len(sources[0]["customer_ids"]),
            "val_rows": len(sources[1]["customer_ids"]),
            "preparation_seconds": float(prep_seconds),
            "test_access": False,
            "heldout_access": False,
            "metadata_policy": "snapshot_unverified",
            "minimum_eligible": MIN_ELIGIBLE_SKUS,
            "product_ids": info["product_ids"],
        }
        atomic_json(manifest, manifest_path, overwrite=False)

    diagnostic_catalogue = catalogue.set_index("product_id")
    family_values = diagnostic_catalogue["product_family"].reindex(info["product_ids"]).fillna("").astype(str)
    family_known = torch.tensor(family_values.str.strip().ne("").to_numpy(dtype=bool, copy=True))
    expected_enrichment_identity = ref_info["manifest"]["source_identities"]["enrichment_identity"]

    campaign_data: Dict[str, Dict[str, Any]] = {}
    cache_lineage: Dict[str, Any] = {}
    for split, src_data, ret_data, rows in zip(("train", "val"), sources, retrieval, cohort_rows):
        purchase_cache_path = UPSTREAM_STAGE1_RUN_DIR / f"{split}_purchase_cache.pt"
        payload = torch.load(purchase_cache_path, map_location="cpu", weights_only=False)
        if payload.get("model_digest") != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(f"{purchase_cache_path} model_digest mismatch.")
        if payload.get("data_identity") != info["identity"]:
            raise ValueError(f"{purchase_cache_path} data_identity mismatch.")
        if split == "val" and payload.get("enrichment_identity") != expected_enrichment_identity:
            raise ValueError(f"{purchase_cache_path} enrichment_identity mismatch.")

        scores_path = SELECTOR_SOURCE_RUN_DIR / f"{split}_scores.pt"
        scores_payload = torch.load(scores_path, map_location="cpu", weights_only=False)
        if scores_payload.get("selector_digest") != EXPECTED_SELECTOR_SHA256:
            raise ValueError(f"{scores_path} selector_digest mismatch.")
        if scores_payload.get("upstream_digest") != EXPECTED_RETRIEVER_SHA256:
            raise ValueError(f"{scores_path} upstream_digest mismatch.")
        if scores_payload.get("data_identity") != info["identity"]:
            raise ValueError(f"{scores_path} data_identity mismatch.")
        validate_input_contract(scores_payload.get("input_contract"), ref_info["selector_input_contract"])

        feat_mmap = np.load(root / f"{split}_features.npy", mmap_mode="r")
        if feat_mmap.shape != (len(src_data["customer_ids"]), EXPECTED_PRODUCT_UNIVERSE, 41):
            raise ValueError(f"Unexpected {split}_features.npy shape: {feat_mmap.shape}.")

        u = payload["customer_vectors"][rows].squeeze(1).contiguous()
        v = payload["product_vectors"].contiguous()
        ref_scores = scores_payload["scores"].detach().cpu().to(dtype=torch.float32).contiguous()
        customer_stats_6 = quantize_transferred_stats(src_data["retriever_tabular_features"])

        ret_data.update(
            source=src_data,
            cohort_rows=rows,
            u=u,
            v=v,
            dot_scores=payload["scores"][rows].contiguous(),
            reference_selector_scores=ref_scores,
            features=feat_mmap,
            customer_stats_6=customer_stats_6,
            family_known=family_known,
        )
        campaign_data[split] = ret_data
        cache_lineage[split] = {
            "purchase_cache_path": str(purchase_cache_path),
            "purchase_cache_sha256": ref_info["hashes"][f"{split}_purchase_cache"],
            "reference_selector_scores_path": str(scores_path),
            "reference_selector_scores_sha256": ref_info["hashes"][f"{split}_scores"],
            "rolling_features_path": str(root / f"{split}_features.npy"),
            "rolling_features_sha256": manifest["feature_sha256"][split],
            "rows": int(len(src_data["customer_ids"])),
        }

    if not torch.equal(campaign_data["train"]["v"], campaign_data["val"]["v"]):
        raise ValueError("Train and val cached product_vectors differ.")

    total_val_targets = int(sum(len(x) for x in campaign_data["val"]["source"]["positive_pids"]))
    total_val_new = int(sum(len(x) for x in campaign_data["val"]["source"]["new_positive_pids"]))
    if total_val_targets != EXPECTED_VAL_TOTAL_TARGETS or total_val_new != EXPECTED_VAL_NEW_TARGETS:
        raise ValueError("Validation target count mismatch.")

    meta = {
        "reference": ref_info,
        "data_identity": info["identity"],
        "enrichment_identity": expected_enrichment_identity,
        "rolling_features_identity": manifest["identity"],
        "observed_source_digest": observed_source_digest,
        "source_input_digests": source_digest_info,
        "statistics_policy": manifest["statistics_policy"],
        "feature_schema_digest": manifest["schema_digest"],
        "product_ids": list(info["product_ids"]),
        "cache_lineage": cache_lineage,
        "upstream_snapshot_assumption": (
            "observed_source_digest invalidates stale rolling feature/score/slate caches when dev purchases "
            "or catalogue/company inputs change; however, prepare_retrieval_data and frozen Stage 1 u/v/stats6/targets "
            "come from historical sealed caches and require a coherent upstream snapshot rebuild after actual source edits."
        ),
    }
    return campaign_data, info, manifest, meta


def _score_frozen_selector_over_rolling_features(
    selector: FeatureAblatedStage1Selector,
    split_data: Mapping[str, Any],
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    """Inference-only replay of frozen active selector over rolling 41D features (CPU float32)."""
    selector_cpu = selector.cpu().eval()
    u_all = split_data["u"].cpu().to(dtype=torch.float32)
    v_all = split_data["v"].cpu().to(dtype=torch.float32)
    stats_all = split_data["customer_stats_6"].cpu().to(dtype=torch.float32)
    allowed_all = split_data["allowed"].cpu()
    feat_mmap = split_data["features"]

    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(u_all), batch_size):
            end = min(start + batch_size, len(u_all))
            f41 = torch.from_numpy(np.array(feat_mmap[start:end], dtype=np.float32))
            logits = selector_cpu(
                u=u_all[start:end],
                v=v_all,
                features_41=f41,
                customer_stats_6=stats_all[start:end],
            )
            masked = logits.masked_fill(~allowed_all[start:end], -1e9)
            chunks.append(masked)
    return torch.cat(chunks, dim=0).contiguous()


def prepare_or_load_rolling_selector_scores(
    campaign_data: Dict[str, Dict[str, Any]],
    info: Mapping[str, Any],
    meta: Dict[str, Any],
    *,
    cache_root: Path = CACHE_ROOT,
    batch_size: int = 256,
    train_parity_atol: float = 1e-4,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Recompute or strictly validate frozen 301D selector scores over rolling 41D features (no training)."""
    if meta.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"Rolling selector scores require statistics_policy={STATISTICS_POLICY_ROLLING!r}."
        )
    selector = meta["reference"].get("selector")
    if not isinstance(selector, FeatureAblatedStage1Selector):
        raise ValueError("meta['reference']['selector'] must be loaded to score rolling features.")

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "selector_scores_manifest.json"

    expected_shared = {
        "cache_version": SELECTOR_SCORE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "reference_id": EXPECTED_REFERENCE_ID,
        "upstream_sha256": meta["reference"]["hashes"]["best_retriever"],
        "selector_sha256": meta["reference"]["hashes"]["best_selector"],
        "data_identity": meta["data_identity"],
        "rolling_features_identity": meta["rolling_features_identity"],
        "observed_source_digest": meta["observed_source_digest"],
        "feature_schema_digest": meta["feature_schema_digest"],
        "rolling_features_sha256": {
            s: meta["cache_lineage"][s]["rolling_features_sha256"] for s in ("train", "val")
        },
        "training_executed": False,
        "heldout_access": False,
    }
    scores_identity = json_digest(expected_shared)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != scores_identity:
            raise ValueError("Rolling selector scores manifest identity mismatch.")
        for key, val in expected_shared.items():
            if manifest.get(key) != val:
                raise ValueError(f"Rolling selector scores manifest mismatch on {key!r}.")
        scores_by_split: Dict[str, torch.Tensor] = {}
        for split in ("train", "val"):
            score_path = root / f"{split}_selector_scores.pt"
            if not score_path.is_file():
                raise FileNotFoundError(f"Missing rolling selector scores file: {score_path}")
            actual_sha = file_digest(score_path)
            if actual_sha != manifest["split_sha256"][split]:
                raise ValueError(f"Rolling selector scores SHA-256 mismatch for {split}.")
            payload = torch.load(score_path, map_location="cpu", weights_only=False)
            if payload.get("scores_identity") != scores_identity:
                raise ValueError(f"Rolling selector scores payload identity mismatch for {split}.")
            scores = payload["scores"].detach().cpu().to(dtype=torch.float32).contiguous()
            if scores.shape != tuple(campaign_data[split]["allowed"].shape):
                raise ValueError(f"Unexpected {split} rolling selector scores shape: {tuple(scores.shape)}.")
            scores_by_split[split] = scores
            campaign_data[split]["rolling_selector_scores"] = scores
            meta["cache_lineage"][split]["rolling_selector_scores_path"] = str(score_path)
            meta["cache_lineage"][split]["rolling_selector_scores_sha256"] = actual_sha
        meta["rolling_selector_scores_identity"] = scores_identity
        return scores_by_split, manifest

    if any(root.glob("*_selector_scores.pt")):
        raise ValueError(f"Incomplete rolling selector score cache in {root}; refusing silent overwrite.")

    scores_by_split = {}
    split_sha256: Dict[str, str] = {}
    train_max_diff_allowed = 0.0
    val_max_diff_allowed = 0.0
    val_slates_changed_count = 0
    for split in ("train", "val"):
        scores = _score_frozen_selector_over_rolling_features(
            selector, campaign_data[split], batch_size=batch_size
        )
        allowed = campaign_data[split]["allowed"]
        ref_scores = campaign_data[split]["reference_selector_scores"]
        diff_allowed = float((scores[allowed] - ref_scores[allowed]).abs().max().item())
        if split == "train":
            train_max_diff_allowed = diff_allowed
            if diff_allowed > train_parity_atol:
                raise AssertionError(
                    f"Train rolling selector replay diverged from frozen train_scores.pt: "
                    f"max_abs_diff={diff_allowed} > {train_parity_atol}"
                )
        else:
            val_max_diff_allowed = diff_allowed
            unseen = campaign_data[split]["unseen"]
            idx_old = select_candidates(ref_scores, allowed, unseen, SLATE_POLICY)
            idx_new = select_candidates(scores, allowed, unseen, SLATE_POLICY)
            val_slates_changed_count = int(
                sum(set(a) != set(b) for a, b in zip(idx_old.tolist(), idx_new.tolist()))
            )

        out_path = root / f"{split}_selector_scores.pt"
        atomic_torch_save(
            {
                "cache_version": SELECTOR_SCORE_CACHE_VERSION,
                "split": split,
                "scores_identity": scores_identity,
                "observed_source_digest": meta["observed_source_digest"],
                "rolling_features_identity": meta["rolling_features_identity"],
                "rolling_features_sha256": meta["cache_lineage"][split]["rolling_features_sha256"],
                "selector_digest": EXPECTED_SELECTOR_SHA256,
                "upstream_digest": EXPECTED_RETRIEVER_SHA256,
                "data_identity": info["identity"],
                "input_contract": meta["reference"]["selector_input_contract"],
                "scores": scores,
            },
            out_path,
        )
        actual_sha = file_digest(out_path)
        scores_by_split[split] = scores
        split_sha256[split] = actual_sha
        campaign_data[split]["rolling_selector_scores"] = scores
        meta["cache_lineage"][split]["rolling_selector_scores_path"] = str(out_path)
        meta["cache_lineage"][split]["rolling_selector_scores_sha256"] = actual_sha

    manifest = {
        **expected_shared,
        "identity": scores_identity,
        "split_sha256": split_sha256,
        "train_parity_max_abs_diff_allowed": train_max_diff_allowed,
        "val_rolling_vs_frozen_max_abs_diff_allowed": val_max_diff_allowed,
        "val_quota100_50_candidate_set_changed_rows": val_slates_changed_count,
        "val_total_rows": int(len(campaign_data["val"]["u"])),
    }
    atomic_json(manifest, manifest_path, overwrite=False)
    meta["rolling_selector_scores_identity"] = scores_identity
    return scores_by_split, manifest


def _build_rolling_slate_payload(
    split: str,
    data: Mapping[str, Any],
    product_ids: Sequence[str],
    meta: Mapping[str, Any],
    *,
    candidate_source: str = DEFAULT_SLATE_CANDIDATE_SOURCE,
) -> Dict[str, Any]:
    if candidate_source == CANDIDATE_SOURCE_ROLLING_SELECTOR:
        if "rolling_selector_scores" not in data:
            raise ValueError(
                "campaign_data missing rolling_selector_scores; call prepare_or_load_rolling_selector_scores first."
            )
        cand_scores = data["rolling_selector_scores"]
        cand_scores_sha = meta["cache_lineage"][split]["rolling_selector_scores_sha256"]
    elif candidate_source == CANDIDATE_SOURCE_REFERENCE_SELECTOR:
        cand_scores = data["reference_selector_scores"]
        cand_scores_sha = meta["cache_lineage"][split]["reference_selector_scores_sha256"]
    elif candidate_source == CANDIDATE_SOURCE_REFERENCE_DOT:
        cand_scores = data["dot_scores"]
        cand_scores_sha = meta["cache_lineage"][split]["purchase_cache_sha256"]
    else:
        raise ValueError(f"Unknown candidate_source: {candidate_source!r}")

    indices = select_candidates(cand_scores, data["allowed"], data["unseen"], SLATE_POLICY)
    if indices.shape != (len(data["u"]), SLATE_SIZE):
        raise ValueError(f"Expected {split} slate shape ({len(data['u'])}, {SLATE_SIZE}), got {tuple(indices.shape)}.")
    eligible_selected = data["allowed"].gather(1, indices)
    if not bool(eligible_selected.all()):
        raise ValueError(f"Ineligible candidate selected in {split} slates.")
    if any(len(set(row)) != SLATE_SIZE for row in indices.tolist()):
        raise ValueError(f"Duplicate candidate index in {split} slates.")

    src = data["source"]
    row_arange = np.arange(len(indices))[:, None]
    pairwise_features = torch.from_numpy(
        np.array(data["features"][row_arange, indices.numpy()], dtype=np.float32)
    ).contiguous()
    if not bool(torch.isfinite(pairwise_features).all()):
        raise ValueError(f"Non-finite rolling pairwise_features in {split} slates.")

    # Deliberately omit mlp_score / teacher_raw_logit / candidate rank/position:
    # Stage 2 receives ONLY rolling 41D features and frozen customer/product vectors.
    payload: Dict[str, Any] = {
        "cache_version": SLATE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "split": split,
        "policy": SLATE_POLICY,
        "slate_size": SLATE_SIZE,
        "candidate_source": candidate_source,
        "statistics_policy": meta["statistics_policy"],
        "feature_schema_digest": meta["feature_schema_digest"],
        "teacher_score_included": False,
        "reference_id": EXPECTED_REFERENCE_ID,
        "upstream_sha256": meta["reference"]["hashes"]["best_retriever"],
        "selector_sha256": meta["reference"]["hashes"]["best_selector"],
        "candidate_source_scores_sha256": cand_scores_sha,
        "rolling_features_sha256": meta["cache_lineage"][split]["rolling_features_sha256"],
        "data_identity": meta["data_identity"],
        "rolling_features_identity": meta["rolling_features_identity"],
        "observed_source_digest": meta["observed_source_digest"],
        "rolling_selector_scores_identity": (
            meta.get("rolling_selector_scores_identity")
            if candidate_source == CANDIDATE_SOURCE_ROLLING_SELECTOR
            else None
        ),
        "product_ids": list(product_ids),
        "transition_keys": list(src["transition_keys"]),
        "customer_ids": list(src["customer_ids"]),
        "positive_pids": list(src["positive_pids"]),
        "new_positive_pids": list(src["new_positive_pids"]),
        "prediction_days": src["prediction_days"].clone(),
        "text_embedding": src["text_embedding"].clone(),
        "city_id": src["city_id"].clone(),
        "region_id": src["region_id"].clone(),
        "industry_id": src["industry_id"].clone(),
        "tabular_features": src["tabular_features"].clone(),
        "retriever_tabular_features": src["retriever_tabular_features"].clone(),
        "sequence": src["sequence"].clone(),
        "attention_mask": src["attention_mask"].clone(),
        "seq_product_ids": src["seq_product_ids"].clone(),
        "seq_purchase_dates": src["seq_purchase_dates"].clone(),
        "seq_category_ids": src["seq_category_ids"].clone(),
        "seq_subcategory_ids": src["seq_subcategory_ids"].clone(),
        "seq_business_line_ids": src["seq_business_line_ids"].clone(),
        "stage1_customer_vectors": data["u"].clone(),
        "candidate_indices": indices.to(dtype=torch.long).contiguous(),
        "candidate_mask": eligible_selected.to(dtype=torch.bool).contiguous(),
        "candidate_pids": [[product_ids[i] for i in row] for row in indices.tolist()],
        "pairwise_features": pairwise_features,
        "labels": data["target"].gather(1, indices).to(dtype=torch.float32).contiguous(),
    }
    payload["cache_identity"] = json_digest(
        {
            "cache_version": SLATE_CACHE_VERSION,
            "namespace": ROLLING_CACHE_NAMESPACE,
            "split": split,
            "policy": SLATE_POLICY,
            "slate_size": SLATE_SIZE,
            "candidate_source": candidate_source,
            "statistics_policy": payload["statistics_policy"],
            "feature_schema_digest": payload["feature_schema_digest"],
            "teacher_score_included": False,
            "upstream_sha256": payload["upstream_sha256"],
            "selector_sha256": payload["selector_sha256"],
            "candidate_source_scores_sha256": cand_scores_sha,
            "rolling_features_sha256": payload["rolling_features_sha256"],
            "data_identity": payload["data_identity"],
            "rolling_features_identity": payload["rolling_features_identity"],
            "observed_source_digest": payload["observed_source_digest"],
            "rolling_selector_scores_identity": payload["rolling_selector_scores_identity"],
            "product_ids": payload["product_ids"],
            "transition_keys": payload["transition_keys"],
            "indices": indices.tolist(),
        }
    )
    return payload


def _validate_loaded_rolling_slate(
    split: str,
    payload: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    expected_rows: int,
) -> None:
    if payload.get("cache_version") != SLATE_CACHE_VERSION:
        raise ValueError(f"Rolling slate cache version mismatch for {split}.")
    if payload.get("namespace") != ROLLING_CACHE_NAMESPACE:
        raise ValueError(f"Rolling slate namespace mismatch for {split}.")
    if payload.get("split") != split or payload.get("policy") != SLATE_POLICY:
        raise ValueError(f"Rolling slate split/policy mismatch for {split}.")
    if payload.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"Rolling slate statistics_policy={payload.get('statistics_policy')!r} != {STATISTICS_POLICY_ROLLING!r}."
        )
    if payload.get("teacher_score_included") is not False:
        raise ValueError("Rolling slate cache must enforce teacher_score_included=False.")
    if "mlp_score" in payload or "teacher_raw_logit" in payload:
        raise ValueError("Rolling slate cache must NOT contain mlp_score or teacher_raw_logit.")
    for key, val in expected_identity.items():
        if payload.get(key) != val:
            raise ValueError(f"Rolling slate {split} provenance mismatch on {key!r}: {payload.get(key)!r} != {val!r}.")
    assert_dev_dates_strict(split, payload["prediction_days"])
    if payload["candidate_indices"].shape != (expected_rows, SLATE_SIZE):
        raise ValueError(f"Rolling slate {split} candidate_indices shape mismatch.")
    if payload["pairwise_features"].shape != (expected_rows, SLATE_SIZE, 41):
        raise ValueError(f"Rolling slate {split} pairwise_features shape mismatch.")


def prepare_or_load_rolling_slate_cache(
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    cache_root: Path = CACHE_ROOT,
    candidate_source: str = DEFAULT_SLATE_CANDIDATE_SOURCE,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Build or strictly validate sealed quota100_50 rolling caches with NO Stage 1 teacher score."""
    if meta.get("statistics_policy") != STATISTICS_POLICY_ROLLING:
        raise ValueError(
            f"Stage 2 rolling slates require statistics_policy={STATISTICS_POLICY_ROLLING!r}."
        )
    if candidate_source == CANDIDATE_SOURCE_ROLLING_SELECTOR and (
        "rolling_selector_scores" not in campaign_data.get("train", {})
        or "rolling_selector_scores" not in campaign_data.get("val", {})
    ):
        raise ValueError(
            "candidate_source='rolling_selector_scores' requires prepare_or_load_rolling_selector_scores first."
        )

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "slates_manifest.json"
    product_ids = list(info["product_ids"])

    expected_shared = {
        "namespace": ROLLING_CACHE_NAMESPACE,
        "upstream_sha256": meta["reference"]["hashes"]["best_retriever"],
        "selector_sha256": meta["reference"]["hashes"]["best_selector"],
        "data_identity": meta["data_identity"],
        "rolling_features_identity": meta["rolling_features_identity"],
        "observed_source_digest": meta["observed_source_digest"],
        "rolling_selector_scores_identity": (
            meta.get("rolling_selector_scores_identity")
            if candidate_source == CANDIDATE_SOURCE_ROLLING_SELECTOR
            else None
        ),
        "feature_schema_digest": meta["feature_schema_digest"],
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "candidate_source": candidate_source,
        "teacher_score_included": False,
        "product_ids": product_ids,
    }

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("heldout_access") is not False:
            raise ValueError("Slates manifest must record heldout_access=false.")
        if manifest.get("policy") != SLATE_POLICY or manifest.get("cache_version") != SLATE_CACHE_VERSION:
            raise ValueError("Slates manifest policy/version mismatch.")
        for key, val in expected_shared.items():
            if key == "product_ids":
                continue
            if manifest.get(key) != val:
                raise ValueError(f"Slates manifest mismatch on {key!r}: {manifest.get(key)!r} != {val!r}.")
        slates: Dict[str, Dict[str, Any]] = {}
        for split in ("train", "val"):
            split_path = root / f"{split}_slates.pt"
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing rolling slate split file: {split_path}")
            actual_sha = file_digest(split_path)
            if actual_sha != manifest["split_sha256"][split]:
                raise ValueError(f"Rolling slate SHA-256 mismatch for {split}.")
            payload = torch.load(split_path, map_location="cpu", weights_only=False)
            exp_split = {
                **expected_shared,
                "rolling_features_sha256": meta["cache_lineage"][split]["rolling_features_sha256"],
                "transition_keys": list(campaign_data[split]["source"]["transition_keys"]),
            }
            _validate_loaded_rolling_slate(split, payload, exp_split, len(campaign_data[split]["u"]))
            if payload.get("cache_identity") != manifest["split_cache_identity"][split]:
                raise ValueError(f"Rolling slate cache_identity mismatch for {split}.")
            slates[split] = payload
        return slates, manifest

    if any(root.glob("*_slates.pt")):
        raise ValueError(f"Incomplete rolling slate cache in {root}; refusing silent overwrite.")

    slates = {}
    split_sha256: Dict[str, str] = {}
    split_cache_identity: Dict[str, str] = {}
    split_stats: Dict[str, Any] = {}
    for split in ("train", "val"):
        payload = _build_rolling_slate_payload(
            split,
            campaign_data[split],
            product_ids,
            meta,
            candidate_source=candidate_source,
        )
        out_path = root / f"{split}_slates.pt"
        atomic_torch_save(payload, out_path)
        slates[split] = payload
        split_sha256[split] = file_digest(out_path)
        split_cache_identity[split] = payload["cache_identity"]
        split_stats[split] = {
            "rows": int(len(payload["customer_ids"])),
            "slate_size": SLATE_SIZE,
            "mean_positive_per_slate": float(payload["labels"].sum(dim=1).mean().item()),
        }

    manifest = {
        "cache_version": SLATE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "heldout_access": False,
        "policy": SLATE_POLICY,
        "slate_size": SLATE_SIZE,
        "candidate_source": candidate_source,
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "teacher_score_included": False,
        "reference_id": EXPECTED_REFERENCE_ID,
        "upstream_sha256": expected_shared["upstream_sha256"],
        "selector_sha256": expected_shared["selector_sha256"],
        "data_identity": expected_shared["data_identity"],
        "rolling_features_identity": expected_shared["rolling_features_identity"],
        "observed_source_digest": expected_shared["observed_source_digest"],
        "rolling_selector_scores_identity": expected_shared["rolling_selector_scores_identity"],
        "feature_schema_digest": expected_shared["feature_schema_digest"],
        "rolling_features_sha256": {
            s: meta["cache_lineage"][s]["rolling_features_sha256"] for s in ("train", "val")
        },
        "split_sha256": split_sha256,
        "split_cache_identity": split_cache_identity,
        "split_stats": split_stats,
        "product_universe": len(product_ids),
    }
    atomic_json(manifest, manifest_path, overwrite=False)
    return slates, manifest


class RollingSlateDataset(RankerDataset):
    """RankerDataset extended with boolean candidate_mask (strictly NO mlp_score)."""

    def __init__(self, data: Mapping[str, Any]):
        if "mlp_score" in data or "teacher_raw_logit" in data:
            raise ValueError("RollingSlateDataset forbids mlp_score / teacher_raw_logit.")
        super().__init__(dict(data))
        self.candidate_mask = data.get("candidate_mask")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item["candidate_mask"] = (
            self.candidate_mask[idx]
            if self.candidate_mask is not None
            else torch.ones_like(self.candidate_indices[idx], dtype=torch.bool)
        )
        return item


def collate_rolling_slate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate Stage 2 slate batch preserving boolean candidate_mask."""
    collated = collate_ranker_batch(batch)
    collated["candidate_mask"] = torch.stack([x["candidate_mask"] for x in batch], dim=0)
    return collated


def rolling_ranker_batch_inputs(
    batch: Mapping[str, Any],
    product_vectors: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Prepare device tensors for RollingCandidateAttentionRanker.forward (no teacher score)."""
    names = (
        "text_embedding",
        "city_id",
        "region_id",
        "industry_id",
        "tabular_features",
        "sequence",
        "attention_mask",
        "seq_product_ids",
        "seq_category_ids",
        "seq_subcategory_ids",
        "seq_business_line_ids",
        "pairwise_features",
        "seq_purchase_dates",
        "prediction_day",
        "candidate_mask",
    )
    inputs: Dict[str, Any] = {
        name: batch[name].to(device) if batch.get(name) is not None else None
        for name in names
    }
    vectors = batch.get("candidate_vectors")
    if vectors is None:
        cand_idx = batch["candidate_indices"].to(device)
        mask = inputs["candidate_mask"]
        safe_idx = torch.where(mask, cand_idx, cand_idx.new_zeros(())) if mask is not None else cand_idx
        inputs["candidate_vectors"] = product_vectors[safe_idx]
    else:
        inputs["candidate_vectors"] = vectors.to(device)
    if batch.get("stage1_customer_vectors") is not None:
        inputs["cached_customer_vectors"] = batch["stage1_customer_vectors"].to(device)
    return inputs


def make_rolling_slate_loader(
    slate: Mapping[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = SEED,
) -> DataLoader:
    """Create deterministic DataLoader over Stage 2 rolling slates."""
    return DataLoader(
        RollingSlateDataset(slate),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collate_rolling_slate_batch,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def verify_rolling_cache_vs_adapter_parity(
    campaign_data: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
    *,
    slates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    n_rows: int = 16,
) -> Dict[str, Any]:
    """Spot-check exact float16->float32 parity between RollingDynamicFeatureAdapter and sealed caches."""
    adapter = RollingDynamicFeatureAdapter.from_development_data(
        info["product_ids"],
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    origin = pd.Timestamp("2025-01-01")
    val = campaign_data["val"]
    src = val["source"]
    total = len(src["customer_ids"])
    # Pick spread indices across validation months to exercise non-monotonic/rolling dates cleanly.
    step = max(1, total // max(int(n_rows), 1))
    indices = list(range(0, total, step))[: max(int(n_rows), 1)]
    max_cache_diff = 0.0
    max_slate_diff = 0.0
    for row_idx in reversed(indices):  # Reverse order checks non-monotonic parity too!
        cid = src["customer_ids"][row_idx]
        as_of = origin + pd.Timedelta(days=int(src["prediction_days"][row_idx]))
        live = adapter.build_quantized(cid, as_of)
        cached = np.array(val["features"][row_idx], dtype=np.float32)
        diff = float(np.max(np.abs(live - cached)))
        max_cache_diff = max(max_cache_diff, diff)
        if slates is not None and "val" in slates:
            slate_idx = slates["val"]["candidate_indices"][row_idx].numpy()
            slate_cached = slates["val"]["pairwise_features"][row_idx].numpy()
            slate_diff = float(np.max(np.abs(live[slate_idx] - slate_cached)))
            max_slate_diff = max(max_slate_diff, slate_diff)
    if max_cache_diff > 0.0 or max_slate_diff > 0.0:
        raise AssertionError(
            f"Rolling adapter vs cache parity mismatch: max_cache_diff={max_cache_diff}, max_slate_diff={max_slate_diff}"
        )
    return {
        "rows_checked": len(indices),
        "non_monotonic_query_order": True,
        "max_abs_diff_vs_feature_cache": max_cache_diff,
        "max_abs_diff_vs_slate_cache": max_slate_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare sealed strictly-causal rolling_observed_v2 features, selector scores, and Stage 2 slates."
    )
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare feature bundle and exit without training.")
    parser.add_argument("--prepare-selector-scores", action="store_true", help="Also recompute/validate frozen 301D selector scores on rolling features.")
    parser.add_argument("--prepare-slates", action="store_true", help="Also prepare Stage 2 quota100_50 rolling slates.")
    parser.add_argument(
        "--candidate-source",
        type=str,
        default=DEFAULT_SLATE_CANDIDATE_SOURCE,
        choices=(
            CANDIDATE_SOURCE_ROLLING_SELECTOR,
            CANDIDATE_SOURCE_REFERENCE_SELECTOR,
            CANDIDATE_SOURCE_REFERENCE_DOT,
        ),
    )
    parser.add_argument("--verify-parity-rows", type=int, default=16, help="Check live adapter vs cache parity on N rows.")
    args = parser.parse_args()

    campaign_data, info, feat_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=args.cache_root,
        build=True,
        statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    scores_manifest = None
    need_scores = (
        args.prepare_selector_scores
        or ((args.prepare_slates or not args.prepare_only) and args.candidate_source == CANDIDATE_SOURCE_ROLLING_SELECTOR)
    )
    if need_scores:
        _, scores_manifest = prepare_or_load_rolling_selector_scores(
            campaign_data,
            info,
            meta,
            cache_root=args.cache_root,
        )
    slates = None
    slates_manifest = None
    if args.prepare_slates or not args.prepare_only:
        slates, slates_manifest = prepare_or_load_rolling_slate_cache(
            campaign_data,
            info,
            meta,
            cache_root=args.cache_root,
            candidate_source=args.candidate_source,
        )
    parity = (
        verify_rolling_cache_vs_adapter_parity(
            campaign_data,
            info,
            slates=slates,
            n_rows=args.verify_parity_rows,
        )
        if args.verify_parity_rows > 0
        else None
    )
    summary = {
        "cache_version": feat_manifest["version"],
        "namespace": feat_manifest["namespace"],
        "statistics_policy": feat_manifest["statistics_policy"],
        "active_reference_id": EXPECTED_REFERENCE_ID,
        "feature_bundle_identity": feat_manifest["identity"],
        "observed_source_digest": feat_manifest["observed_source_digest"],
        "feature_schema_digest": feat_manifest["schema_digest"],
        "train_rows": feat_manifest["train_rows"],
        "val_rows": feat_manifest["val_rows"],
        "selector_scores_prepared": scores_manifest is not None,
        "selector_scores_identity": scores_manifest["identity"] if scores_manifest is not None else None,
        "slates_prepared": slates_manifest is not None,
        "candidate_source": slates_manifest["candidate_source"] if slates_manifest is not None else None,
        "slate_policy": SLATE_POLICY if slates_manifest is not None else None,
        "teacher_score_included": False,
        "parity_check": parity,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
