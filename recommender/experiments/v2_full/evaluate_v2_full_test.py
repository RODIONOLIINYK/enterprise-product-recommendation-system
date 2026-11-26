"""Authorized final inference-only held-out Test evaluation (>= 2026-07-01) for v2_full reference.

Evaluates the frozen promoted `w2_2mem_recency_hl365` Wave2BE2ERanker checkpoints
(Primary Ep 10 `best_discovery.pt` + Companion Ep 9 `best_ndcg.pt`) with frozen
validation-fitted affine calibration parameters (`deployment_full_fit a, b`) over
all strictly out-of-time `v2_full` held-out Test baskets (`2026-07-01 .. 2026-11-20`).

Zero retraining, zero hyperparameter tuning, zero calibration fitting on Test.
Also exports a compact stratified set of ~280 real Test baskets (45 highlights)
for the portable Ukrainian interactive demo (`build_portable_demo.py --split test`).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import RollingCandidatePool, select_sellable_products
from recommender.experiments.roadmap_v2.evaluate_stage5_test import (
    EXCEL_COMMERCIAL_MAPPING,
    HeldoutTestBasket,
    evaluate_full_catalogue_predictions,
    load_all_cleaned_purchases,
)
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    observed_pack_ids,
)
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    resolve_base_names_to_skus,
)
from recommender.experiments.static_enrichment.metadata import (
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.scope import actual_purchase_mask
from recommender.sequences import DAYS_PER_YEAR, PURCHASE_DATE_ORIGIN
from recommender.text import get_company_text, load_text_encoder
from recommender.train_retriever import build_customer_metadata, get_device

from .build_portable_demo import _extract_drivers
from .config import (
    CLEANED_PURCHASES_PATH,
    EVIDENCE_ROOT,
    INTERIM_ROOT,
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    RAW_ITEMS_PATH,
    RAW_WORKBOOK_PATH,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
    V2_FULL_REFERENCE_PATH,
)
from .demo_engine import U11_IDX, _verify_v2_checkpoint
from .extra_features import (
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    SEQ_5_ACTIVE_INDICES,
    assemble_cand_42,
    compute_extra_cand_5_arrays,
    prune_cust_7,
)
from .wave2b_models import Wave2BE2ERanker


TEST_REPORT_PATH = EVIDENCE_ROOT / "v2_full_heldout_test_results.json"
TEST_DEMO_EXPORT_PATH = INTERIM_ROOT / "test_demo_cases_w2_2mem_recency_hl365.json"


def stream_read_xlsx_all_commercial(workbook_path: Path) -> pd.DataFrame:
    """Fast streaming XML parser for all commercial rows (dev + held-out test)."""
    t0 = time.perf_counter()
    with zipfile.ZipFile(workbook_path) as zf:
        ss: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            with zf.open("xl/sharedStrings.xml") as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag.endswith("}si"):
                        ss.append("".join(elem.itertext()))
                        elem.clear()
        sheet_xml = next(
            n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )

        col_letter_to_name: Dict[str, str] = {}
        header_row_idx: int | None = None
        records: Dict[str, List[Any]] = {v: [] for v in EXCEL_COMMERCIAL_MAPPING.values()}
        want_letters: set[str] = set()

        with zf.open(sheet_xml) as f:
            row_idx = 0
            cur: Dict[str, str] = {}
            for _, elem in ET.iterparse(f, events=("end",)):
                tag = elem.tag
                if tag.endswith("}c"):
                    ref = elem.attrib.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    if header_row_idx is None or col in want_letters:
                        t = elem.attrib.get("t")
                        v = elem.find("{*}v")
                        if v is not None and v.text is not None:
                            cur[col] = ss[int(v.text)] if t == "s" else v.text
                        else:
                            is_el = elem.find("{*}is")
                            if is_el is not None:
                                cur[col] = "".join(is_el.itertext())
                elif tag.endswith("}row"):
                    row_idx += 1
                    if header_row_idx is None:
                        vals_set = {str(v).strip() for v in cur.values() if v is not None}
                        if set(EXCEL_COMMERCIAL_MAPPING).issubset(vals_set):
                            header_row_idx = row_idx
                            for col_l, hval in cur.items():
                                hs = str(hval).strip()
                                if hs in EXCEL_COMMERCIAL_MAPPING:
                                    col_letter_to_name[col_l] = EXCEL_COMMERCIAL_MAPPING[hs]
                            want_letters = set(col_letter_to_name)
                        elif row_idx >= 20:
                            raise ValueError("Expected commercial headers not found in first 20 rows.")
                    elif cur:
                        for col_l, cname in col_letter_to_name.items():
                            records[cname].append(cur.get(col_l))
                    cur.clear()
                    elem.clear()

    if header_row_idx is None:
        raise ValueError("Header row not found in workbook.")

    df = pd.DataFrame(records)
    for col in (
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
    ):
        df[col] = df[col].astype("string").str.strip()
    for col in ("transaction_type", "item_type"):
        df[col] = df[col].str.upper()

    for dcol in ("purchase_date", "full_payment_date"):
        raw_ser = df[dcol]
        num = pd.to_numeric(raw_ser, errors="coerce")
        parsed = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
        non_num_mask = raw_ser.notna() & num.isna()
        if bool(non_num_mask.any()):
            parsed.loc[non_num_mask] = pd.to_datetime(
                raw_ser.loc[non_num_mask], format="mixed", errors="coerce"
            ).dt.normalize()
        df[dcol] = parsed

    for ncol in ("quantity", "cost", "cost_usd", "revenue", "doc_bonus"):
        df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    print(f"[v2-test] Stream-parsed {len(df):,} full workbook rows in {time.perf_counter() - t0:.1f}s.", flush=True)
    return df


def load_reconciled_full_commercial(
    workbook_path: Path,
    all_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    raw = stream_read_xlsx_all_commercial(workbook_path)
    sellable_ids = set(select_sellable_products(catalogue)["product_id"].astype(str).str.strip())
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

    lhs = paid.groupby(["customer_id", "product_id", "purchase_date"], sort=True)["quantity"].sum()
    rhs = all_purchases.groupby(["customer_id", "product_id", "purchase_date"], sort=True)["quantity"].sum()
    if len(lhs) != len(rhs) or not lhs.index.equals(rhs.index) or not np.allclose(lhs.to_numpy(), rhs.to_numpy()):
        raise ValueError("Full commercial export and cleaned_purchases.csv disagree!")

    settlements = pd.to_datetime(paid["full_payment_date"], errors="coerce").dt.normalize()
    incomplete = settlements.isna()
    incomplete_basket = incomplete.groupby([paid["customer_id"], paid["purchase_date"]]).transform("any")
    paid["full_payment_date"] = settlements.mask(incomplete_basket)
    return paid


def build_v2_test_baskets(
    all_purchases: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    pool: RollingCandidatePool,
    test_start: pd.Timestamp = TEST_START_V2,
) -> Tuple[List[HeldoutTestBasket], Dict[str, int]]:
    """Build chronological held-out Test baskets (>= test_start) with full-catalogue denominators."""
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
        is_test = bool(date >= test_start)
        if is_test:
            allowed_sku = np.isin(pids, list(pool.eligible_on(date)))
            allowed = np.zeros(n, dtype=bool)
            allowed[idx[allowed_sku]] = True
        for cid, basket in daily.groupby("customer_id", sort=True):
            first = cid not in seen_any_order
            prior_mod = seen_modeled_bn.setdefault(cid, np.zeros(n, dtype=bool))
            prior_unmod_bn = seen_unmodeled_bn.setdefault(cid, set())
            prior_unmod_sku = seen_unmodeled_sku.setdefault(cid, set())

            basket_pids = set(basket["product_id"].tolist())
            mod_idxs = sorted({mapping[p] for p in basket_pids if p in mapping})
            unmod_bns = {
                all_pid_to_bn.get(p, p)
                for p in basket_pids
                if all_pid_to_bn.get(p, "") not in by_bn
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
                        unmodeled_novelty=tuple((bn, bn not in prior_unmod_bn) for bn in sorted(unmod_bns)),
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


def validate_v2_test_split(data: Mapping[str, Any], test_start: pd.Timestamp = TEST_START_V2) -> None:
    days = data["prediction_days"]
    dates = PURCHASE_DATE_ORIGIN + pd.to_timedelta(days.numpy(), unit="D")
    if (dates < test_start).any():
        raise ValueError("Held-out test split contains a pre-TEST_START_V2 basket.")
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


def _encode_test_company_texts(
    test_cids: Sequence[str],
    companies: pd.DataFrame,
    val_meta_path: Path,
) -> torch.Tensor:
    """Reuse cached BGE-M3 embeddings from val_meta.pt and encode remaining Test companies offline."""
    known: Dict[str, np.ndarray] = {}
    if val_meta_path.is_file():
        vm = torch.load(val_meta_path, map_location="cpu", weights_only=False)
        v_embs = vm["text_embedding"].cpu().numpy()
        for cid, emb in zip(vm["customer_ids"], v_embs, strict=True):
            c_str = str(cid).strip()
            if c_str not in known:
                known[c_str] = emb
        del vm

    unique_needed = sorted({str(c).strip() for c in test_cids if str(c).strip() not in known})
    if unique_needed:
        comp_idx = companies.drop_duplicates("customer_id").set_index("customer_id")
        texts: List[str] = []
        valid_cids: List[str] = []
        for cid in unique_needed:
            if cid in comp_idx.index:
                texts.append(get_company_text(comp_idx.loc[cid], include_region=True))
                valid_cids.append(cid)
            else:
                known[cid] = np.zeros(1024, dtype=np.float32)
        if valid_cids:
            print(f"[v2-test] Encoding {len(valid_cids):,} unseen Test company profiles with BAAI/bge-m3...", flush=True)
            t0 = time.perf_counter()
            encoder = load_text_encoder()
            embs = encoder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            embs = np.asarray(embs, dtype=np.float32)
            for cid, emb in zip(valid_cids, embs, strict=True):
                known[cid] = emb
            print(f"[v2-test] Encoded {len(valid_cids):,} company profiles in {time.perf_counter() - t0:.1f}s.", flush=True)

    return torch.from_numpy(
        np.stack([known.get(str(c).strip(), np.zeros(1024, dtype=np.float32)) for c in test_cids]).astype(np.float32)
    )


def _load_ranker(
    reference: Mapping[str, Any],
    shared_meta: Mapping[str, Any],
    ckpt_path: Path,
    dev: torch.device,
) -> Tuple[Wave2BE2ERanker, torch.Tensor, int]:
    arch = dict(reference["active_architecture"])
    enrichment = shared_meta["enrichment"]
    vocabs = shared_meta["vocabs"]
    n_demand = int(shared_meta["prod_inputs"]["demand_features"].shape[1])
    model = Wave2BE2ERanker(
        product_attr_vocab_sizes={k: len(v) for k, v in enrichment["product_vocabs"].items()},
        num_cities=len(vocabs["city_vocab"]) + 1,
        num_regions=len(vocabs["region_vocab"]) + 1,
        num_industries=len(vocabs["industry_vocab"]) + 1,
        num_demand_features=n_demand,
        num_candidate_features=int(arch.get("num_candidate_features", NUM_CAND_42_FEATURES)),
        num_customer_features=int(arch.get("num_customer_features", NUM_CUST_7_FEATURES)),
        num_pairwise_bias_features=int(arch.get("num_pairwise_bias_features", NUM_CAND_42_FEATURES)),
        active_seq_stat_indices=tuple(arch.get("active_seq_stat_indices", SEQ_5_ACTIVE_INDICES)),
        pure_v_query=bool(arch.get("pure_v_query", True)),
        option_a_unseen_head=bool(arch.get("option_a_unseen_head", True)),
        unseen_dropout_cold_prob=0.0,
        unseen_attn_aux_enabled=False,
        d_model=int(arch["d_model"]),
        num_heads=int(arch["num_heads"]),
        memory_ffn_dim=int(arch["memory_ffn_dim"]),
        memory_layers=int(arch.get("memory_layers", 1)),
        mlp_widths=tuple(int(w) for w in arch["mlp_widths"]),
        dropout=float(arch.get("dropout", 0.12)),
        aux_60d_enabled=False,
    ).to(dev)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    prod_inputs_dev = {
        "text_embedding": shared_meta["prod_inputs"]["text_embedding"].float().to(dev),
        "demand_features": shared_meta["prod_inputs"]["demand_features"].float().to(dev),
        "attr_ids": {k: v.to(dev) for k, v in shared_meta["prod_inputs"]["attr_ids"].items()},
    }
    with torch.no_grad():
        v_bn = model.precompute_product_vectors(prod_inputs_dev).detach()
    return model, v_bn, int(ck.get("epoch", 0))


def _fmt_sku(pid: str, cat_idx: pd.DataFrame) -> str:
    if pid not in cat_idx.index:
        return pid
    row = cat_idx.loc[pid]
    ct = str(row.get("container_type", "")).strip() if pd.notna(row.get("container_type", "")) else ""
    vol = row.get("volume", None)
    unit = str(row.get("unit", "")).strip() if pd.notna(row.get("unit", "")) else ""
    parts: List[str] = []
    if ct and ct.upper() != "NONE":
        parts.append(ct)
    if vol is not None and pd.notna(vol):
        try:
            vf = float(vol)
            if vf > 0:
                parts.append(f"{int(vf) if vf.is_integer() else vf:g}{unit}")
        except (TypeError, ValueError):
            pass
    return f"{' · '.join(parts)} ({pid})" if parts else pid


def _score_all_test(
    model: Wave2BE2ERanker,
    v_bn: torch.Tensor,
    test_meta: Mapping[str, Any],
    cand38_arr: np.ndarray,
    extra5_arr: np.ndarray,
    dev: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    n_rows = len(test_meta["customer_ids"])
    n_bn = int(v_bn.shape[0])
    s_main = np.empty((n_rows, n_bn), dtype=np.float32)
    s_unseen = np.empty((n_rows, n_bn), dtype=np.float32)
    cust_7_all = prune_cust_7(test_meta["tabular_features"].float())
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(n_rows, start + batch_size)
            sl = slice(start, end)
            c42 = torch.from_numpy(assemble_cand_42(cand38_arr[sl], extra5_arr[sl])).to(dev)
            res = model(
                company_text_embedding=test_meta["text_embedding"][sl].float().to(dev),
                city_id=test_meta["city_id"][sl].long().to(dev),
                region_id=test_meta["region_id"][sl].long().to(dev),
                industry_id=test_meta["industry_id"][sl].long().to(dev),
                customer_features=cust_7_all[sl].to(dev),
                sequence=test_meta["sequence"][sl].float().to(dev),
                attention_mask=test_meta["attention_mask"][sl].long().to(dev),
                seq_product_ids=test_meta["seq_product_ids"][sl].long().to(dev),
                seq_purchase_dates=test_meta["seq_purchase_dates"][sl].float().to(dev),
                prediction_day=test_meta["prediction_days"][sl].float().to(dev),
                candidate_features=c42,
                candidate_mask=test_meta["allowed_bn"][sl].bool().to(dev),
                precomputed_v_bn=v_bn,
                return_aux=False,
            )
            s_main[sl] = res["output"].logits.float().cpu().numpy()
            s_unseen[sl] = res["unseen_logits"].float().cpu().numpy()
    return s_main, s_unseen


def _rank_top_k(scores: np.ndarray, mask: np.ndarray, top_k: int = 50) -> np.ndarray:
    n_rows, n_bn = scores.shape
    masked = np.where(mask, scores, -1e9)
    order = np.argsort(-masked, axis=1)[:, :top_k]
    out = np.full((n_rows, top_k), -1, dtype=np.int64)
    for i in range(n_rows):
        row = order[i]
        valid = mask[i, row]
        cnt = int(valid.sum())
        out[i, :cnt] = row[valid]
    return out


def _compute_pure_unseen_metrics(
    ranked_unseen: np.ndarray,
    new_target: np.ndarray,
    extra_new: np.ndarray,
    first_obs: np.ndarray,
) -> Dict[str, Any]:
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    h10 = h25 = ret10 = ret25 = fo10 = 0
    ndcgs: List[float] = []
    for i in range(len(ranked_unseen)):
        n_pos = int(new_target[i].sum()) + int(extra_new[i])
        if n_pos == 0:
            continue
        preds = ranked_unseen[i]
        pv = preds[preds >= 0]
        rel = new_target[i, pv]
        r10 = int(rel[:10].sum())
        r25 = int(rel[:25].sum())
        h10 += r10
        h25 += r25
        if first_obs[i]:
            fo10 += r10
        else:
            ret10 += r10
            ret25 += r25
        dcg = float((rel[:10].astype(np.float64) * discounts[: len(rel[:10])]).sum())
        idcg = float(discounts[: min(n_pos, 10)].sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    total_new = int(new_target.sum()) + int(extra_new.sum())
    return {
        "pure_unseen_hits@10": int(h10),
        "pure_unseen_recall@10": float(h10 / max(total_new, 1)),
        "pure_unseen_hits@25": int(h25),
        "pure_unseen_recall@25": float(h25 / max(total_new, 1)),
        "pure_unseen_ndcg@10_new_baskets": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "pure_unseen_returning_hits@10": int(ret10),
        "pure_unseen_returning_hits@25": int(ret25),
        "pure_unseen_first_observed_hits@10": int(fo10),
    }


def _evaluate_sku_full_catalogue(
    resolved_skus: np.ndarray,
    sku_target: np.ndarray,
    sku_new_target: np.ndarray,
    extra_sku: np.ndarray,
) -> Dict[str, Any]:
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    n_rows = len(resolved_skus)
    r10s = np.zeros(n_rows, dtype=np.float64)
    ndcgs = np.zeros(n_rows, dtype=np.float64)
    new_hits = 0
    for i in range(n_rows):
        n_pos = int(sku_target[i].sum()) + int(extra_sku[i])
        if n_pos == 0:
            continue
        preds = resolved_skus[i][:10]
        pv = preds[preds >= 0]
        rel = sku_target[i, pv]
        r10s[i] = float(rel.sum()) / n_pos
        new_hits += int(sku_new_target[i, pv].sum())
        dcg = float((rel.astype(np.float64) * discounts[: len(rel)]).sum())
        idcg = float(discounts[: min(n_pos, 10)].sum())
        ndcgs[i] = dcg / idcg if idcg > 0 else 0.0
    return {
        "sku_recall@10": float(r10s.mean()) if n_rows else 0.0,
        "sku_ndcg@10": float(ndcgs.mean()) if n_rows else 0.0,
        "sku_new_hits@10": int(new_hits),
    }


def _evaluate_checkpoint_on_test(
    label: str,
    s_main: np.ndarray,
    s_unseen: np.ndarray,
    a: float,
    b: float,
    test_meta: Mapping[str, Any],
    cases: Sequence[HeldoutTestBasket],
    all_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
) -> Tuple[Dict[str, Any], np.ndarray]:
    allowed = test_meta["allowed_bn"].bool().cpu().numpy()
    unseen = test_meta["unseen_bn"].bool().cpu().numpy()
    first_obs = test_meta["first_observed"].bool().cpu().numpy()
    target_t = test_meta["target_bn"]
    new_t = test_meta["new_target_bn"]
    new_np = new_t.bool().cpu().numpy()

    extra_t = np.array([c.extra_bn_targets for c in cases], dtype=np.int64)
    extra_nt = np.array([c.extra_new_bn_targets for c in cases], dtype=np.int64)
    has_mod = np.array([c.has_modeled_target for c in cases], dtype=bool)
    extra_sku = np.array([c.extra_sku_targets for c in cases], dtype=np.int64)

    is_repeat = allowed & ~unseen
    blended = np.where(is_repeat, s_main, a * s_unseen + b)
    ranked_blended = _rank_top_k(blended, allowed, top_k=50)
    ranked_unseen = _rank_top_k(s_unseen, unseen, top_k=25)

    # Full-catalogue metrics (all Test baskets, unmodeled targets counted as misses)
    fc = evaluate_full_catalogue_predictions(
        ranked_blended, target_t, new_t, extra_t, extra_nt, test_meta["customer_ids"]
    )
    fc_ret = evaluate_full_catalogue_predictions(
        ranked_blended[~first_obs],
        target_t[~first_obs],
        new_t[~first_obs],
        extra_t[~first_obs],
        extra_nt[~first_obs],
        [c for c, f in zip(test_meta["customer_ids"], first_obs) if not f],
    )
    fc_fo = evaluate_full_catalogue_predictions(
        ranked_blended[first_obs],
        target_t[first_obs],
        new_t[first_obs],
        extra_t[first_obs],
        extra_nt[first_obs],
        [c for c, f in zip(test_meta["customer_ids"], first_obs) if f],
    )
    fc_pu = _compute_pure_unseen_metrics(ranked_unseen, new_np, extra_nt, first_obs)

    # Causal pack-size SKU resolution on Full Catalogue
    pred_days = test_meta["prediction_days"].cpu().numpy().astype(np.int64).tolist()
    sku_allowed = torch.from_numpy(np.stack([c.sku_allowed for c in cases]))
    sku_target = np.stack([c.sku_target for c in cases])
    resolved_skus = resolve_base_names_to_skus(
        ranked_blended[:, :10],
        test_meta["customer_ids"],
        pred_days,
        sku_allowed,
        all_purchases,
        catalogue,
        product_ids,
        universe,
        top_k=10,
    )
    first_sku_sales = all_purchases.groupby(["customer_id", "product_id"]).purchase_date.min().unstack()
    dates_arr = np.asarray([c.date for c in cases], dtype="datetime64[D]")
    known_at = first_sku_sales.reindex(index=test_meta["customer_ids"], columns=product_ids).to_numpy(dtype="datetime64[D]")
    sku_new_target = sku_target & ~(known_at < dates_arr[:, None])
    fc_sku = _evaluate_sku_full_catalogue(resolved_skus, sku_target, sku_new_target, extra_sku)

    # Modeled-only subset metrics (baskets with >=1 target in the 1,141 frozen universe, zero extra denominator)
    zeros_mod = np.zeros(int(has_mod.sum()), dtype=np.int64)
    mod_m = evaluate_full_catalogue_predictions(
        ranked_blended[has_mod],
        target_t[has_mod],
        new_t[has_mod],
        zeros_mod,
        zeros_mod,
        [c for c, m in zip(test_meta["customer_ids"], has_mod) if m],
    )
    mod_pu = _compute_pure_unseen_metrics(ranked_unseen[has_mod], new_np[has_mod], zeros_mod, first_obs[has_mod])
    mod_sku = _evaluate_sku_full_catalogue(
        resolved_skus[has_mod], sku_target[has_mod], sku_new_target[has_mod], zeros_mod
    )

    result = {
        "label": label,
        "frozen_calibration": {"a": float(a), "b": float(b)},
        "full_catalogue": {
            **fc,
            "returning_bn_recall@10": fc_ret["bn_recall@10"],
            "returning_bn_ndcg@10": fc_ret["bn_ndcg@10"],
            "returning_true_new_bn_hits@10": fc_ret["true_new_bn_hits@10"],
            "returning_true_new_bn_targets": fc_ret["true_new_bn_targets"],
            "first_observed_bn_recall@10": fc_fo["bn_recall@10"],
            "first_observed_bn_ndcg@10": fc_fo["bn_ndcg@10"],
            "first_observed_true_new_bn_hits@10": fc_fo["true_new_bn_hits@10"],
            "first_observed_true_new_bn_targets": fc_fo["true_new_bn_targets"],
            **fc_pu,
            **fc_sku,
        },
        "modeled_only": {
            **mod_m,
            **mod_pu,
            **mod_sku,
        },
    }
    return result, resolved_skus


def _export_stratified_test_demo_cases(
    *,
    cases: Sequence[HeldoutTestBasket],
    test_meta: Mapping[str, Any],
    cand38_arr: np.ndarray,
    extra5_arr: np.ndarray,
    s_main: np.ndarray,
    s_unseen: np.ndarray,
    resolved_skus: np.ndarray,
    a: float,
    b: float,
    all_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    companies: pd.DataFrame,
    product_ids: Sequence[str],
    universe: Mapping[str, Any],
    reps: np.ndarray,
    cpu_ranker: Wave2BE2ERanker,
    cpu_v_bn: torch.Tensor,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Export ~280 stratified real held-out Test baskets (modeled targets >= 1) for OPEN_DEMO.html."""
    rng = np.random.default_rng(seed)
    first_obs = np.array([c.first for c in cases], dtype=bool)
    has_mod = np.array([c.has_modeled_target for c in cases], dtype=bool)
    new_cnt = np.array([int(c.new.sum()) for c in cases], dtype=np.int64)
    tgt_cnt = np.array([int(c.target.sum()) for c in cases], dtype=np.int64)

    buckets = [
        (np.flatnonzero(has_mod & (~first_obs) & (new_cnt > 0)).tolist(), 105),
        (np.flatnonzero(has_mod & first_obs).tolist(), 50),
        (np.flatnonzero(has_mod & (~first_obs) & (tgt_cnt >= 2) & (new_cnt == 0)).tolist(), 65),
        (np.flatnonzero(has_mod & (~first_obs)).tolist(), 60),
    ]
    selected: List[int] = []
    seen: set[int] = set()
    for pool_ids, quota in buckets:
        perm = rng.permutation(pool_ids).tolist() if pool_ids else []
        for idx in perm:
            if len([x for x in selected if x in set(pool_ids)]) >= quota and idx not in seen:
                break
            if idx not in seen:
                seen.add(idx)
                selected.append(int(idx))

    company_map = companies.drop_duplicates("customer_id").set_index("customer_id").to_dict("index")
    cat_idx = catalogue.drop_duplicates("product_id").set_index("product_id")
    histories = dict(tuple(all_purchases.groupby("customer_id", sort=False)))
    bn_meta: List[Dict[str, str]] = []
    for b_i, bn_name in enumerate(universe["base_names"]):
        rep_pid = product_ids[int(reps[b_i])]
        row = cat_idx.loc[rep_pid] if rep_pid in cat_idx.index else {}
        bn_meta.append(
            {
                "base_name": str(bn_name),
                "brand": str(row.get("brand", "")) if pd.notna(row.get("brand", "")) else "",
                "category": str(row.get("product_category", "")) if pd.notna(row.get("product_category", "")) else "",
            }
        )

    all_pid_to_bn = universe["all_pid_to_bn"]
    cust_7_all = prune_cust_7(test_meta["tabular_features"].float())
    cust_11_all = test_meta["tabular_features"].float().cpu().numpy()

    # Warmup CPU ranker once for realistic single-request CPU latency measurement
    with torch.no_grad():
        i0 = selected[0]
        c42_0 = torch.from_numpy(assemble_cand_42(cand38_arr[i0], extra5_arr[i0])).unsqueeze(0)
        cpu_ranker(
            company_text_embedding=test_meta["text_embedding"][i0 : i0 + 1].float(),
            city_id=test_meta["city_id"][i0 : i0 + 1].long(),
            region_id=test_meta["region_id"][i0 : i0 + 1].long(),
            industry_id=test_meta["industry_id"][i0 : i0 + 1].long(),
            customer_features=cust_7_all[i0 : i0 + 1],
            sequence=test_meta["sequence"][i0 : i0 + 1].float(),
            attention_mask=test_meta["attention_mask"][i0 : i0 + 1].long(),
            seq_product_ids=test_meta["seq_product_ids"][i0 : i0 + 1].long(),
            seq_purchase_dates=test_meta["seq_purchase_dates"][i0 : i0 + 1].float(),
            prediction_day=test_meta["prediction_days"][i0 : i0 + 1].float(),
            candidate_features=c42_0,
            candidate_mask=test_meta["allowed_bn"][i0 : i0 + 1].bool(),
            precomputed_v_bn=cpu_v_bn,
            return_aux=False,
        )

    exported: List[Dict[str, Any]] = []
    for idx in selected:
        case = cases[idx]
        cid = case.customer
        date = case.date
        c42 = assemble_cand_42(cand38_arr[idx], extra5_arr[idx])

        # Measure real single-request CPU neural latency
        t_inf0 = time.perf_counter()
        with torch.no_grad():
            cpu_ranker(
                company_text_embedding=test_meta["text_embedding"][idx : idx + 1].float(),
                city_id=test_meta["city_id"][idx : idx + 1].long(),
                region_id=test_meta["region_id"][idx : idx + 1].long(),
                industry_id=test_meta["industry_id"][idx : idx + 1].long(),
                customer_features=cust_7_all[idx : idx + 1],
                sequence=test_meta["sequence"][idx : idx + 1].float(),
                attention_mask=test_meta["attention_mask"][idx : idx + 1].long(),
                seq_product_ids=test_meta["seq_product_ids"][idx : idx + 1].long(),
                seq_purchase_dates=test_meta["seq_purchase_dates"][idx : idx + 1].float(),
                prediction_day=test_meta["prediction_days"][idx : idx + 1].float(),
                candidate_features=torch.from_numpy(c42).unsqueeze(0),
                candidate_mask=test_meta["allowed_bn"][idx : idx + 1].bool(),
                precomputed_v_bn=cpu_v_bn,
                return_aux=False,
            )
        inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        allowed = case.allowed
        unseen = case.unseen
        is_rep = allowed & ~unseen
        sm = s_main[idx]
        su = s_unseen[idx]
        blended = np.where(is_rep, sm, a * su + b)

        rep_idxs = np.flatnonzero(is_rep)
        uns_idxs = np.flatnonzero(unseen)
        top_rep = rep_idxs[np.argsort(-sm[rep_idxs])[:15]] if len(rep_idxs) else np.empty(0, dtype=np.int64)
        top_uns = uns_idxs[np.argsort(-su[uns_idxs])[:15]] if len(uns_idxs) else np.empty(0, dtype=np.int64)
        union_bns = sorted({int(x) for x in top_rep.tolist()} | {int(x) for x in top_uns.tolist()})

        full_blended_order = np.argsort(-np.where(allowed, blended, -1e9))
        full_unseen_order = np.argsort(-np.where(unseen, su, -1e9))
        blended_rank = {int(bn): r + 1 for r, bn in enumerate(full_blended_order) if allowed[bn]}
        unseen_rank = {int(bn): r + 1 for r, bn in enumerate(full_unseen_order) if unseen[bn]}

        # Map top-10 resolved SKUs for fast display
        top10_bns = [int(bn) for bn in full_blended_order[:10] if allowed[bn]]
        sku_by_bn: Dict[int, int] = {}
        for pos, bn in enumerate(top10_bns):
            if pos < len(resolved_skus[idx]) and int(resolved_skus[idx][pos]) >= 0:
                sku_by_bn[bn] = int(resolved_skus[idx][pos])

        cands: List[Dict[str, Any]] = []
        for bn in union_bns:
            is_r = bool(is_rep[bn])
            pidx = sku_by_bn.get(bn, -1)
            if pidx < 0:
                # Fallback to first allowed member SKU in this formulation
                members = [p for p in universe["bn_to_member_pid_indices"][bn] if case.sku_allowed[p]]
                pidx = int(members[0]) if members else -1
            sku_lbl = _fmt_sku(product_ids[pidx], cat_idx) if pidx >= 0 else "—"
            sku_hit = bool(pidx >= 0 and case.sku_target[pidx])
            bm = bn_meta[bn]
            cands.append(
                {
                    "b": bn,
                    "n": bm["base_name"],
                    "br": bm["brand"],
                    "cat": bm["category"],
                    "r": 1 if is_r else 0,
                    "t": 1 if bool(case.target[bn]) else 0,
                    "nt": 1 if bool(case.new[bn]) else 0,
                    "sm": round(float(sm[bn]), 4),
                    "su": round(float(su[bn]), 4),
                    "sku": sku_lbl,
                    "sh": 1 if sku_hit else 0,
                    "dr": _extract_drivers(c42[bn], is_new_candidate=(not is_r)),
                }
            )

        gt: List[Dict[str, Any]] = []
        for bn in np.flatnonzero(case.target).tolist():
            bm = bn_meta[bn]
            gt.append(
                {
                    "b": int(bn),
                    "n": bm["base_name"],
                    "br": bm["brand"],
                    "cat": bm["category"],
                    "nw": 1 if bool(case.new[bn]) else 0,
                    "rb": blended_rank.get(int(bn)),
                    "ru": unseen_rank.get(int(bn)),
                }
            )
        gt.sort(key=lambda g: (g["rb"] if g["rb"] is not None else 9999, g["n"]))

        # Strictly prior customer profile & purchase history (< date)
        comp = company_map.get(cid, {})
        c_name = str(comp.get("company_name") or comp.get("name") or cid).strip()
        c_city = str(comp.get("city") or "—").strip()
        c_reg = str(comp.get("region_code") or "—").strip()
        c_ind = str(comp.get("industry_branch") or "—").strip()
        c_desc = str(comp.get("company_description") or "").strip()[:260]

        hist_df = histories.get(cid)
        prior_hist: List[Dict[str, Any]] = []
        n_prior_orders = 0
        n_prior_bns = 0
        med_gap: int | None = None
        days_since_last: int | None = None
        if hist_df is not None and not hist_df.empty:
            prior = hist_df.loc[hist_df["purchase_date"] < date].sort_values("purchase_date", kind="stable")
            if not prior.empty:
                udates = sorted(prior["purchase_date"].unique())
                n_prior_orders = len(udates)
                days_since_last = int((date - pd.Timestamp(udates[-1])).days)
                if len(udates) >= 2:
                    gaps = [(pd.Timestamp(udates[i]) - pd.Timestamp(udates[i - 1])).days for i in range(1, len(udates))]
                    med_gap = int(round(float(np.median(gaps))))
                n_prior_bns = len({all_pid_to_bn.get(str(p), "") for p in prior["product_id"] if all_pid_to_bn.get(str(p))})
                for _, hrow in prior.iloc[-24:].iloc[::-1].iterrows():
                    pid_s = str(hrow["product_id"])
                    bn_s = all_pid_to_bn.get(pid_s, pid_s)
                    prow = cat_idx.loc[pid_s] if pid_s in cat_idx.index else {}
                    prior_hist.append(
                        {
                            "dt": str(pd.Timestamp(hrow["purchase_date"]).date()),
                            "ago": int((date - pd.Timestamp(hrow["purchase_date"])).days),
                            "n": str(bn_s),
                            "br": str(prow.get("brand", "")) if pd.notna(prow.get("brand", "")) else "",
                            "cat": str(prow.get("product_category", "")) if pd.notna(prow.get("product_category", "")) else "",
                            "q": round(float(hrow["quantity"]), 1),
                        }
                    )

        cu = cust_11_all[idx]
        prep_pct = int(round(100.0 * float(cu[U11_IDX["cust_prepaid_or_cash_ratio"]])))
        def_days = int(round(float(math.expm1(max(0.0, float(cu[U11_IDX["cust_avg_deferral_days_norm"]])) * math.log1p(180.0)))))
        ovd_pct = int(round(100.0 * float(cu[U11_IDX["cust_overdue_unpaid_ratio"]])))

        top10_set = set(top10_bns)
        h10 = int(sum(1 for bn in top10_set if case.target[bn]))
        nh10 = int(sum(1 for bn in top10_set if case.new[bn]))

        exported.append(
            {
                "id": int(idx),
                "dt": str(date.date()),
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "el": int(allowed.sum()),
                "elu": int(unseen.sum()),
                "ms": [0.2, round(float(inf_ms), 2), 0.1],
                "fo": 1 if case.first else 0,
                "hn": 1 if (not case.first and int(case.new.sum()) > 0) else 0,
                "mi": 1 if int(case.target.sum()) >= 2 else 0,
                "h10": h10,
                "ntgt": int(case.target.sum()),
                "nh10": nh10,
                "nnew": int(case.new.sum()),
                "c": {
                    "cid": cid,
                    "nm": c_name,
                    "ct": c_city,
                    "rg": c_reg,
                    "ind": c_ind,
                    "desc": c_desc,
                    "ord": n_prior_orders,
                    "bns": n_prior_bns,
                    "gap": med_gap,
                    "last": days_since_last,
                    "prep": prep_pct,
                    "def": def_days,
                    "ovd": ovd_pct,
                },
                "hist": prior_hist,
                "cands": cands,
                "gt": gt,
            }
        )

    # Mark top 45 clean showcase highlights (strong recall + true new discoveries / cold-start hits)
    showcase_scored = sorted(
        exported,
        key=lambda d: (
            (d["nh10"] > 0 and d["h10"] >= min(d["ntgt"], 2)),
            d["nh10"],
            d["h10"] / max(d["ntgt"], 1),
            d["ntgt"],
        ),
        reverse=True,
    )
    sc_ids = {d["id"] for d in showcase_scored[:45]}
    for d in exported:
        d["sc"] = d["id"] in sc_ids
    exported.sort(key=lambda d: (not d["sc"], -d["nh10"], -(d["h10"] / max(d["ntgt"], 1)), d["id"]))
    return exported


def run_v2_full_heldout_test(*, allow_heldout_test: bool = False) -> Dict[str, Any]:
    if not allow_heldout_test:
        raise RuntimeError(
            "Refusing to run held-out Test evaluation without explicit --allow-heldout-test authorization."
        )
    t_all = time.perf_counter()
    reference = json.loads(V2_FULL_REFERENCE_PATH.read_text(encoding="utf-8"))
    primary_ckpt = _verify_v2_checkpoint(reference["checkpoints"]["ranker"], "primary_ranker")
    ndcg_ckpt = _verify_v2_checkpoint(reference["checkpoints"]["ndcg_ranker"], "ndcg_ranker")
    shared_meta = torch.load(PREPARED_ROOT / "shared_meta.pt", map_location="cpu", weights_only=False)

    universe = shared_meta["universe"]
    product_ids: List[str] = list(shared_meta["product_ids"])
    reps: np.ndarray = np.asarray(shared_meta["reps"], dtype=np.int64)
    vocabs = shared_meta["vocabs"]
    n_bn = int(universe["num_base_names"])

    catalogue = load_enriched_catalogue(RAW_ITEMS_PATH, descriptions_path=None)
    all_purchases = load_all_cleaned_purchases(CLEANED_PURCHASES_PATH)
    companies = load_enriched_companies(RAW_CUSTOMERS_PATH, purchases=all_purchases, minimum_purchases=0)
    full_commercial = load_reconciled_full_commercial(RAW_WORKBOOK_PATH, all_purchases, catalogue)

    pool = RollingCandidatePool(catalogue, all_purchases)
    allowed_train_pids = observed_pack_ids(pool.first_sales, TRAIN_CUTOFF_V2)
    assert allowed_train_pids is not None

    cases, case_counts = build_v2_test_baskets(all_purchases, product_ids, universe, pool, test_start=TEST_START_V2)
    n_test = len(cases)
    cids = [c.customer for c in cases]
    dates = [c.date for c in cases]
    days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]
    print(
        f"[v2-test] Built {n_test:,} held-out Test baskets ({dates[0].date()}..{dates[-1].date()}) | "
        f"modeled={case_counts['modeled_baskets']:,}, first_obs={case_counts['first_observed_all']:,}, "
        f"modeled_targets={case_counts['modeled_bn_targets']:,} (+{case_counts['unmodeled_bn_targets']:,} unmodeled), "
        f"new_targets={case_counts['modeled_new_bn_targets']:,} (+{case_counts['unmodeled_new_bn_targets']:,} unmodeled)",
        flush=True,
    )

    observed_catalogue = catalogue.loc[catalogue["product_id"].isin(allowed_train_pids)]
    representatives = catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()
    representatives = representatives.loc[representatives["product_id"].isin(allowed_train_pids)]

    print(f"[v2-test] Building strictly causal (< query_day) sequences for {n_test:,} Test baskets...", flush=True)
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
        frozen_v5_replay=False,
        test_start=TEST_START_V2,
    )
    sequences["sequence"][..., 3:5] = 0.0

    print(f"[v2-test] Building strictly causal (< query_day) cand_38 & CUST_11 for {n_test:,} Test baskets...", flush=True)
    engine = NativeBaseNameFeatureEngine(
        all_purchases,
        full_commercial,
        catalogue,
        companies,
        product_ids,
        universe,
        reps,
        allowed_member_pids=allowed_train_pids,
        frozen_v5_replay=False,
        allow_heldout_test=True,
        test_start=TEST_START_V2,
    )
    cand38 = np.empty((n_test, n_bn, len(CANDIDATE_38_FEATURE_NAMES)), dtype=np.float16)
    cust_11 = np.empty((n_test, len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)), dtype=np.float32)
    t_feat = time.perf_counter()
    for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
        c38, cu11 = engine.build_split(cid, date)
        cand38[i] = np.clip(c38, -65500.0, 65500.0).astype(np.float16)
        cust_11[i] = cu11
        if (i + 1) % 3000 == 0 or (i + 1) == n_test:
            print(f"[v2-test] cand_38 {i + 1:,}/{n_test:,} ({time.perf_counter() - t_feat:.1f}s)", flush=True)
    cust_11 = np.nan_to_num(cust_11, nan=0.0, posinf=1.0, neginf=-1.0)

    familiarity = FamiliarityStore(
        all_purchases, observed_catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    profiles = build_customer_metadata(companies, vocabs)
    text_emb = _encode_test_company_texts(cids, companies, PREPARED_ROOT / "val_meta.pt")

    test_meta: Dict[str, Any] = {
        "customer_ids": cids,
        "prediction_days": torch.tensor(days, dtype=torch.float32),
        "first_observed": torch.tensor([c.first for c in cases], dtype=torch.bool),
        "target_bn": torch.from_numpy(np.stack([c.target for c in cases])),
        "new_target_bn": torch.from_numpy(np.stack([c.new for c in cases])),
        "allowed_bn": torch.from_numpy(np.stack([c.allowed for c in cases])),
        "unseen_bn": torch.from_numpy(np.stack([c.unseen for c in cases])),
        "tabular_features": torch.from_numpy(cust_11),
        "familiarity_bn": torch.cat(
            [familiarity.batch_features(cids[i : i + 512], dates[i : i + 512]) for i in range(0, n_test, 512)]
        ),
        "text_embedding": text_emb,
        **{
            k: torch.tensor([profiles.get(cid, {}).get(k, 0) for cid in cids], dtype=torch.long)
            for k in ("city_id", "region_id", "industry_id")
        },
        **sequences,
    }
    validate_v2_test_split(test_meta, test_start=TEST_START_V2)

    print(f"[v2-test] Computing strictly causal (< query_day) extra_cand_5 for {n_test:,} Test baskets...", flush=True)
    reg_map = {str(cid).strip(): int(prof.get("region_id", 0)) for cid, prof in profiles.items()}
    ind_map = {str(cid).strip(): int(prof.get("industry_id", 0)) for cid, prof in profiles.items()}
    extra5 = compute_extra_cand_5_arrays(
        commercial_events=full_commercial,
        universe=universe,
        customer_region_map=reg_map,
        customer_industry_map=ind_map,
        splits={"test": test_meta},
        out_paths=None,
        test_start=pd.Timestamp("2099-01-01"),
    )["test"]

    dev = get_device()
    print(f"[v2-test] Scoring frozen Wave2BE2ERanker checkpoints on {dev}...", flush=True)

    # 1. Primary Ep 10 cal-Discovery (frozen deployment_full_fit a, b)
    prim_dep = reference["validation_metrics_primary_cal_discovery"]["affine_oof_calibration"]["deployment_full_fit"]
    prim_a, prim_b = float(prim_dep["a"]), float(prim_dep["b"])
    model_p, v_bn_p, ep_p = _load_ranker(reference, shared_meta, primary_ckpt, dev)
    s_main_p, s_unseen_p = _score_all_test(model_p, v_bn_p, test_meta, cand38, extra5, dev)
    res_primary, resolved_skus_p = _evaluate_checkpoint_on_test(
        f"primary_cal_discovery_ep{ep_p}",
        s_main_p,
        s_unseen_p,
        prim_a,
        prim_b,
        test_meta,
        cases,
        all_purchases,
        catalogue,
        product_ids,
        universe,
    )

    # 2. Companion Ep 9 cal-NDCG == cal-Unseen (frozen deployment_full_fit a, b)
    comp_dep = reference["validation_metrics_companion_cal_ndcg"]["affine_oof_calibration"]["deployment_full_fit"]
    comp_a, comp_b = float(comp_dep["a"]), float(comp_dep["b"])
    model_c, v_bn_c, ep_c = _load_ranker(reference, shared_meta, ndcg_ckpt, dev)
    s_main_c, s_unseen_c = _score_all_test(model_c, v_bn_c, test_meta, cand38, extra5, dev)
    res_companion, _ = _evaluate_checkpoint_on_test(
        f"companion_cal_ndcg_ep{ep_c}",
        s_main_c,
        s_unseen_c,
        comp_a,
        comp_b,
        test_meta,
        cases,
        all_purchases,
        catalogue,
        product_ids,
        universe,
    )

    # Export stratified Test cases for OPEN_DEMO.html (using CPU model for real single-request CPU timing)
    print("[v2-test] Exporting stratified held-out Test cases for OPEN_DEMO.html...", flush=True)
    cpu_model, cpu_v_bn, _ = _load_ranker(reference, shared_meta, primary_ckpt, torch.device("cpu"))
    demo_cases = _export_stratified_test_demo_cases(
        cases=cases,
        test_meta=test_meta,
        cand38_arr=cand38,
        extra5_arr=extra5,
        s_main=s_main_p,
        s_unseen=s_unseen_p,
        resolved_skus=resolved_skus_p,
        a=prim_a,
        b=prim_b,
        all_purchases=all_purchases,
        catalogue=catalogue,
        companies=companies,
        product_ids=product_ids,
        universe=universe,
        reps=reps,
        cpu_ranker=cpu_model,
        cpu_v_bn=cpu_v_bn,
        seed=42,
    )

    report = {
        "reference_id": reference["reference_id"],
        "arm": reference["active_architecture"]["arm"],
        "split": "heldout_test",
        "authorized_final_test_evaluation": True,
        "train_cutoff": str(TRAIN_CUTOFF_V2.date()),
        "test_start": str(TEST_START_V2.date()),
        "test_date_range": [str(dates[0].date()), str(dates[-1].date())],
        "counts": case_counts,
        "checkpoints": {
            "primary": {"path": str(primary_ckpt), "epoch": ep_p, "sha256": file_digest(primary_ckpt)},
            "companion": {"path": str(ndcg_ckpt), "epoch": ep_c, "sha256": file_digest(ndcg_ckpt)},
        },
        "primary_cal_discovery": res_primary,
        "companion_cal_ndcg": res_companion,
        "elapsed_seconds": float(time.perf_counter() - t_all),
    }
    atomic_json(report, TEST_REPORT_PATH, overwrite=True)

    demo_export = {
        "meta": {
            "reference_id": reference["reference_id"],
            "arm": reference["active_architecture"]["arm"],
            "epoch": ep_p,
            "split": "heldout_test",
            "train_params": int(reference["active_architecture"]["trainable_params"]),
            "per_request_params": int(reference["active_architecture"]["per_request_params"]),
            "num_formulations": n_bn,
            "test_baskets": n_test,
            "modeled_test_baskets": int(case_counts["modeled_baskets"]),
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "checkpoint_bytes": int(primary_ckpt.stat().st_size),
            "checkpoint_sha256": file_digest(primary_ckpt),
            "test_metrics_full_catalogue": res_primary["full_catalogue"],
            "test_metrics_modeled": res_primary["modeled_only"],
            "test_metrics_companion_full_catalogue": res_companion["full_catalogue"],
        },
        "cases": demo_cases,
    }
    atomic_json(demo_export, TEST_DEMO_EXPORT_PATH, overwrite=True)

    fc = res_primary["full_catalogue"]
    mo = res_primary["modeled_only"]
    fc9 = res_companion["full_catalogue"]
    print(
        f"\n[v2-test] FINAL HELD-OUT TEST COMPLETE in {report['elapsed_seconds']:.1f}s -> {TEST_REPORT_PATH}\n"
        f"  Primary Ep {ep_p} (a={prim_a:.3f}, b={prim_b:.3f}):\n"
        f"    Full Catalogue ({fc['queries']:,} baskets): "
        f"R@5={fc['bn_recall@5']:.2%} | R@10={fc['bn_recall@10']:.2%} | R@25={fc['bn_recall@25']:.2%} | "
        f"NDCG@10={fc['bn_ndcg@10']:.5f} | New@10={fc['true_new_bn_hits@10']:,}/{fc['true_new_bn_targets']:,} ({fc['true_new_bn_recall@10']:.2%}) | "
        f"PureU@10={fc['pure_unseen_hits@10']:,} ({fc['pure_unseen_recall@10']:.2%}) | "
        f"SKU R@10={fc['sku_recall@10']:.2%} / New={fc['sku_new_hits@10']:,}\n"
        f"    Modeled Only   ({mo['queries']:,} baskets): "
        f"R@10={mo['bn_recall@10']:.2%} | NDCG@10={mo['bn_ndcg@10']:.5f} | "
        f"New@10={mo['true_new_bn_hits@10']:,}/{mo['true_new_bn_targets']:,} ({mo['true_new_bn_recall@10']:.2%}) | "
        f"PureU@10={mo['pure_unseen_hits@10']:,} ({mo['pure_unseen_recall@10']:.2%})\n"
        f"  Companion Ep {ep_c} (a={comp_a:.3f}, b={comp_b:.3f}):\n"
        f"    Full Catalogue: R@10={fc9['bn_recall@10']:.2%} | NDCG@10={fc9['bn_ndcg@10']:.5f} | "
        f"New@10={fc9['true_new_bn_hits@10']:,}/{fc9['true_new_bn_targets']:,} ({fc9['true_new_bn_recall@10']:.2%}) | "
        f"PureU@10={fc9['pure_unseen_hits@10']:,} ({fc9['pure_unseen_recall@10']:.2%})\n"
        f"  Exported {len(demo_cases)} Test demo cases -> {TEST_DEMO_EXPORT_PATH}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-heldout-test", action="store_true", help="Explicitly authorize final held-out Test evaluation.")
    args = parser.parse_args()
    run_v2_full_heldout_test(allow_heldout_test=args.allow_heldout_test)


if __name__ == "__main__":
    main()
