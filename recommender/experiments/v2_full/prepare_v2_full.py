"""Ingest & prepare the 4-year `v2_full` dataset cleanly into `data/v2_full/` (zero v1 mutation).

Pipeline:
  1. Copy `/Users/kalyma123/Downloads/{dataset.xlsx, customers.csv, items.csv}` -> `data/v2_full/raw/`.
  2. Stream-parse `dataset.xlsx` via fast XML reader (verified against `pd.read_excel` on sample rows),
     filter strictly to paid sellable merchandise (`ПРОДАЖА + ТОВАР` in `select_sellable_products`),
     reconcile quantities, and export:
       - `data/v2_full/interim/cleaned_dev_purchases.csv` (reconciled development purchases)
       - `data/v2_full/interim/commercial_dev_events.parquet` (strictly `< TEST_START_V2 = 2026-07-01`)
  3. Encode BGE-M3 `1024D` company text embeddings (`company_text_bge_m3.parquet`) and pack-free
     formulation embeddings + `65D` structured specs strictly `< TRAIN_CUTOFF_V2 = 2026-02-01`
     (`bn_static_v2_full.pt`).
  4. Fit strictly-train (`< 2026-02-01`) categorical vocabularies and build memory-mapped
     `float16` `cand_38` feature matrices (`train_cand_38.npy`, `val_cand_38.npy`) plus compact
     metadata tensors (`train_meta.pt`, `val_meta.pt`, `shared_meta.pt`) in `data/v2_full/interim/prepared_v2_full/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import RollingCandidatePool, select_sellable_products
from recommender.data import HISTORY_COLUMNS, aggregate_purchases, validate_purchases
from recommender.experiments.roadmap_v2.augmentation import build_strict_60d_aux_labels
from recommender.experiments.roadmap_v2.dated_baskets import (
    aligned_commercial_events,
    basket_cases,
    first_observed_pack_indices,
    validate_dated_split,
)
from recommender.experiments.roadmap_v2.evaluate_stage5_test import EXCEL_COMMERCIAL_MAPPING
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    build_base_name_static_representations,
    observed_pack_ids,
    representative_attr_ids,
)
from recommender.experiments.roadmap_v2.run_stage5 import _collapse_bool_matrix_to_bn
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
)
from recommender.experiments.static_enrichment.metadata import (
    encode_categorical_values,
    fit_categorical_vocab,
    load_enriched_catalogue,
    load_enriched_companies,
)
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.preparation import export_purchases
from recommender.scope import actual_purchase_mask
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.text import get_company_text, load_cached_embeddings, load_text_encoder
from recommender.train_retriever import build_customer_metadata

from .config import (
    BN_STATIC_CACHE_PATH,
    CLEANED_PURCHASES_PATH,
    DEV_PURCHASES_PATH,
    load_v2_development_purchases,
    COMMERCIAL_DEV_PARQUET_PATH,
    COMPANY_TEXT_PARQUET_PATH,
    DOWNLOADS_ROOT,
    INTERIM_ROOT,
    PREPARED_ROOT,
    RAW_CUSTOMERS_PATH,
    RAW_ITEMS_PATH,
    RAW_ROOT,
    RAW_WORKBOOK_PATH,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
)


PRODUCT_ATTR_FIELDS: Tuple[str, ...] = (
    "product_category",
    "product_subcategory",
    "business_line",
    "tier",
    "unit",
    "country_of_origin",
)
PRODUCT_ATTR_KEY_MAP: Dict[str, str] = {
    "product_category": "category",
    "product_subcategory": "subcategory",
    "business_line": "business_line",
    "tier": "tier",
    "unit": "unit",
    "country_of_origin": "country_of_origin",
}


def copy_raw_sources(downloads_root: Path = DOWNLOADS_ROOT) -> None:
    """Copy raw v2_full sources from Downloads into data/v2_full/raw/ (idempotent)."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for fname, dst in (
        ("dataset.xlsx", RAW_WORKBOOK_PATH),
        ("customers.csv", RAW_CUSTOMERS_PATH),
        ("items.csv", RAW_ITEMS_PATH),
    ):
        if dst.is_file():
            continue
        src = downloads_root / fname
        if not src.is_file():
            raise FileNotFoundError(f"Expected source file missing: {src}")
        print(f"[v2-prep] Copying {src} -> {dst}...", flush=True)
        shutil.copy2(src, dst)


def _parse_excel_date(raw_val: Any) -> pd.Timestamp:
    if raw_val is None or raw_val == "":
        return pd.NaT
    try:
        ser = float(raw_val)
        if not np.isfinite(ser) or ser <= 0:
            return pd.NaT
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(ser))).normalize()
    except ValueError:
        ts = pd.to_datetime(str(raw_val).strip(), errors="coerce")
        return ts.normalize() if pd.notna(ts) else pd.NaT


def _parse_float(raw_val: Any) -> float:
    if raw_val is None or raw_val == "":
        return float("nan")
    try:
        return float(raw_val)
    except ValueError:
        return float("nan")


def stream_read_xlsx_commercial(workbook_path: Path) -> pd.DataFrame:
    """Fast streaming XML parser for NAV Excel commercial exports matching EXCEL_COMMERCIAL_MAPPING."""
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
        date_column = ""

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
                            date_column = next(k for k, v in col_letter_to_name.items() if v == "purchase_date")
                        elif row_idx >= 20:
                            raise ValueError("Expected commercial headers not found in first 20 rows.")
                    elif cur and _parse_excel_date(cur.get(date_column)) < TEST_START_V2:
                        for col_l, cname in col_letter_to_name.items():
                            records[cname].append(cur.get(col_l))
                    cur.clear()
                    elem.clear()

    if header_row_idx is None:
        raise ValueError("Header row not found in workbook.")

    df = pd.DataFrame(records)
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

    print(f"[v2-prep] Stream-parsed {len(df):,} workbook rows in {time.perf_counter() - t0:.1f}s.", flush=True)
    return df


def build_cleaned_and_commercial(catalogue: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate cleaned_purchases.csv & commercial_dev_events.parquet (strictly < TEST_START_V2)."""
    INTERIM_ROOT.mkdir(parents=True, exist_ok=True)
    if (DEV_PURCHASES_PATH.is_file() or CLEANED_PURCHASES_PATH.is_file()) and COMMERCIAL_DEV_PARQUET_PATH.is_file():
        dev_p = load_v2_development_purchases()
        comm_dev = pd.read_parquet(COMMERCIAL_DEV_PARQUET_PATH, filters=[("purchase_date", "<", TEST_START_V2)])
        return dev_p, comm_dev

    raw = stream_read_xlsx_commercial(RAW_WORKBOOK_PATH)
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

    paid_core = paid[HISTORY_COLUMNS].copy()
    cleaned_dev = aggregate_purchases(paid_core)
    validate_purchases(cleaned_dev, paid_core, products=catalogue)
    export_purchases(cleaned_dev, DEV_PURCHASES_PATH, overwrite=DEV_PURCHASES_PATH.is_file())
    print(f"[v2-prep] Saved {len(cleaned_dev):,} development purchases -> {DEV_PURCHASES_PATH}", flush=True)

    # Commercial dev parquet strictly < TEST_START_V2 (clamp settlement dates >= TEST_START_V2)
    comm_dev = paid.loc[paid["purchase_date"] < TEST_START_V2].copy()
    comm_dev.loc[comm_dev["full_payment_date"] >= TEST_START_V2, "full_payment_date"] = pd.NaT
    comm_dev.to_parquet(COMMERCIAL_DEV_PARQUET_PATH, index=False)
    print(
        f"[v2-prep] Saved {len(comm_dev):,} dev commercial lines (< {TEST_START_V2.date()}) -> {COMMERCIAL_DEV_PARQUET_PATH}",
        flush=True,
    )

    dev_p = cleaned_dev
    dev_p["purchase_date"] = pd.to_datetime(dev_p["purchase_date"], format="mixed").dt.normalize()
    dev_p["customer_id"] = dev_p["customer_id"].astype(str).str.strip()
    dev_p["product_id"] = dev_p["product_id"].astype(str).str.strip()
    return dev_p, comm_dev


def ensure_company_text_embeddings(companies: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Encode or load BAAI/bge-m3 1024D company embeddings for all v2_full companies."""
    if COMPANY_TEXT_PARQUET_PATH.is_file():
        return load_cached_embeddings(COMPANY_TEXT_PARQUET_PATH, "customer_id")

    cids = companies["customer_id"].astype(str).str.strip().tolist()
    texts = [get_company_text(row, include_region=True) for _, row in companies.iterrows()]
    print(f"[v2-prep] Encoding {len(cids):,} company profiles with BAAI/bge-m3...", flush=True)
    t0 = time.perf_counter()
    encoder = load_text_encoder()
    embs = encoder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    out_df = pd.DataFrame({"customer_id": cids, "embedding": [e.tolist() for e in embs]})
    COMPANY_TEXT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(COMPANY_TEXT_PARQUET_PATH, index=False)
    print(f"[v2-prep] Encoded & cached company BGE-M3 in {time.perf_counter() - t0:.1f}s.", flush=True)
    return {cid: embs[i] for i, cid in enumerate(cids)}


def fit_train_only_vocabs(
    train_purchases: pd.DataFrame,
    companies: pd.DataFrame,
    enriched_catalogue: pd.DataFrame,
    product_ids: Sequence[str],
    allowed_train_pids: set[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, torch.Tensor]]:
    """Fit customer & product categorical vocabularies strictly on < TRAIN_CUTOFF_V2."""
    train_cids = set(train_purchases["customer_id"].astype(str).str.strip().unique())
    train_cos = companies.loc[companies["customer_id"].isin(train_cids)]

    def _simple_vocab(series: pd.Series) -> Dict[str, int]:
        vals = sorted({str(v).strip() for v in series.dropna() if str(v).strip() and str(v).strip().upper() != "NULL"})
        return {v: i + 1 for i, v in enumerate(vals)}

    seq_and_cust_vocabs: Dict[str, Dict[str, int]] = {
        "city_vocab": _simple_vocab(train_cos["city"]),
        "region_vocab": _simple_vocab(train_cos["region_code"]),
        "industry_vocab": _simple_vocab(train_cos["industry_branch"]),
    }

    cat_idx = enriched_catalogue.drop_duplicates("product_id").set_index("product_id")
    train_cat = cat_idx.loc[cat_idx.index.isin(allowed_train_pids)]
    seq_and_cust_vocabs["category_vocab"] = _simple_vocab(train_cat["product_category"])
    seq_and_cust_vocabs["subcategory_vocab"] = _simple_vocab(train_cat["product_subcategory"])
    seq_and_cust_vocabs["business_line_vocab"] = _simple_vocab(train_cat["business_line"])

    product_vocabs: Dict[str, Dict[str, int]] = {}
    product_attr_ids: Dict[str, torch.Tensor] = {}
    for col in PRODUCT_ATTR_FIELDS:
        key = PRODUCT_ATTR_KEY_MAP[col]
        vocab = fit_categorical_vocab(train_cat[col].tolist(), min_count=1)
        product_vocabs[key] = vocab
        vals = [cat_idx.at[p, col] if p in cat_idx.index else None for p in product_ids]
        product_attr_ids[key] = torch.tensor(encode_categorical_values(vals, vocab), dtype=torch.long)

    return seq_and_cust_vocabs, product_vocabs, product_attr_ids


def load_prepared_metadata(root=PREPARED_ROOT, *, train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2):
    """Check the prepared partitions and ordered schemas at each experiment entry point."""
    shared = torch.load(root / "shared_meta.pt", map_location="cpu", weights_only=False)
    if pd.Timestamp(shared["train_cutoff"]) != train_cutoff or pd.Timestamp(shared["test_start"]) != test_start:
        raise ValueError("Prepared data uses different split boundaries.")
    splits = []
    for name in ("train", "val"):
        split = torch.load(root / f"{name}_meta.pt", map_location="cpu", weights_only=False)
        validate_dated_split(split, name, train_cutoff=train_cutoff, test_start=test_start)
        if tuple(split["feature_names"]) != CANDIDATE_38_FEATURE_NAMES or tuple(split["customer_feature_names"]) != CUSTOMER_DYNAMIC_11_FEATURE_NAMES:
            raise ValueError(f"{name} feature schema differs from the prepared model inputs.")
        n_rows = len(split["customer_ids"])
        n_items = shared["universe"]["num_base_names"]
        candidates = np.load(split["cand_38_path"], mmap_mode="r")
        if candidates.shape != (n_rows, n_items, len(CANDIDATE_38_FEATURE_NAMES)) or candidates.dtype != np.float16:
            raise ValueError(f"{name} candidate array has the wrong shape or dtype.")
        if split["tabular_features"].shape != (n_rows, len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)):
            raise ValueError(f"{name} customer features have the wrong shape.")
        splits.append(split)
    return shared, *splits


def prepare_v2_full_dataset(*, force_features: bool = False) -> Dict[str, Any]:
    """Build & cache all v2_full development artifacts cleanly (strictly < TEST_START_V2)."""
    t_all = time.perf_counter()
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = PREPARED_ROOT / "manifest.json"
    if manifest_path.is_file() and not force_features:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("dataset") != "v2_full" or pd.Timestamp(manifest["train_cutoff"]) != TRAIN_CUTOFF_V2
                or pd.Timestamp(manifest["test_start"]) != TEST_START_V2):
            raise ValueError("Prepared manifest uses a different dataset or split policy.")
        for partition in ("train", "val"):
            for artifact in (PREPARED_ROOT / "shared_meta.pt", PREPARED_ROOT / f"{partition}_meta.pt",
                             Path(manifest["splits"][partition]["cand_38_memmap"])):
                if not artifact.is_file():
                    raise FileNotFoundError(f"Incomplete prepared dataset, missing {artifact}")
        print(f"[v2-prep] Prepared v2_full dataset already cached at {PREPARED_ROOT}", flush=True)
        return manifest
    copy_raw_sources()

    catalogue = load_enriched_catalogue(RAW_ITEMS_PATH, descriptions_path=None)
    dev_purchases, comm_dev_raw = build_cleaned_and_commercial(catalogue)
    if dev_purchases["purchase_date"].ge(TEST_START_V2).any():
        raise ValueError("Held-out test event leaked into development purchases!")

    companies = load_enriched_companies(RAW_CUSTOMERS_PATH, purchases=dev_purchases, minimum_purchases=0)
    company_texts = ensure_company_text_embeddings(companies)

    pool = RollingCandidatePool(catalogue, dev_purchases)
    allowed_train_pids = observed_pack_ids(pool.first_sales, TRAIN_CUTOFF_V2)
    assert allowed_train_pids is not None

    # Canonical product_ids = all dev-sold sellable SKUs ordered deterministically
    product_ids = sorted(dev_purchases["product_id"].unique().tolist())
    universe = build_base_name_universe(
        product_ids, catalogue, dev_purchases, train_cutoff=TRAIN_CUTOFF_V2
    )
    reps = first_observed_pack_indices(product_ids, universe["bn_to_member_pid_indices"], pool.first_sales)
    n_bn = int(universe["num_base_names"])
    print(
        f"[v2-prep] Dev universe (< {TEST_START_V2.date()}): {len(dev_purchases):,} records, "
        f"{len(product_ids):,} SKUs -> {n_bn:,} canonical base_names ({universe['trained_base_names']:,} train-observed).",
        flush=True,
    )

    train_purchases = dev_purchases.loc[dev_purchases["purchase_date"] < TRAIN_CUTOFF_V2]
    vocabs, product_vocabs, product_attr_ids_all = fit_train_only_vocabs(
        train_purchases, companies, catalogue, product_ids, allowed_train_pids
    )
    enrichment = {
        "product_ids": product_ids,
        "product_vocabs": product_vocabs,
        "product_attr_ids": product_attr_ids_all,
        "customer_vocabs": {"sales_branch": {"<NONE>": 0, "<OOV>": 1}},
    }

    observed_catalogue = catalogue.loc[catalogue["product_id"].isin(allowed_train_pids)]
    representatives = catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()
    representatives = representatives.loc[representatives["product_id"].isin(allowed_train_pids)]

    commercial = aligned_commercial_events(comm_dev_raw, dev_purchases, test_start=TEST_START_V2)
    engine = NativeBaseNameFeatureEngine(
        dev_purchases,
        commercial,
        catalogue,
        companies,
        product_ids,
        universe,
        reps,
        allowed_member_pids=allowed_train_pids,
        frozen_v5_replay=False,
        test_start=TEST_START_V2,
    )
    familiarity = FamiliarityStore(
        dev_purchases, observed_catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    cases, case_counts = basket_cases(
        dev_purchases,
        product_ids,
        universe,
        pool,
        train_cutoff=TRAIN_CUTOFF_V2,
        test_start=TEST_START_V2,
    )
    print(
        f"[v2-prep] Basket cases: train={len(cases['train']):,}, val={len(cases['val']):,} "
        f"(excluded={dict(train=dict(case_counts['train']), val=dict(case_counts['val']))})",
        flush=True,
    )

    profiles = build_customer_metadata(companies, vocabs)
    bn_static = build_base_name_static_representations(
        universe,
        product_ids,
        catalogue,
        reps,
        allowed_member_pids=allowed_train_pids,
        frozen_v5_specs=False,
        cache_path=BN_STATIC_CACHE_PATH,
    )
    bn_static["attr_ids"] = representative_attr_ids(enrichment, product_ids, reps, allowed_train_pids)

    split_summaries: Dict[str, Any] = {}
    for split in ("train", "val"):
        rows = cases[split]
        n_rows = len(rows)
        cids, dates, first, targets, new, allowed, unseen, sku_targets, sku_allowed = zip(*rows, strict=True)
        days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]

        print(f"[v2-prep] Building {split} sequences ({n_rows:,} baskets)...", flush=True)
        sequences = build_base_name_sequences(
            dev_purchases,
            cids,
            days,
            universe,
            representatives,
            vocabs["category_vocab"],
            vocabs["subcategory_vocab"],
            vocabs["business_line_vocab"],
            frozen_v5_replay=False,
            test_start=TEST_START_V2,
        )
        sequences["sequence"][..., 3:5] = 0.0

        # Memory-mapped float16 candidate features (keeps peak RAM < 2.5 GB on 16 GB Apple Silicon!)
        cand_mm_path = PREPARED_ROOT / f"{split}_cand_38.npy"
        cand_mm = np.lib.format.open_memmap(
            cand_mm_path,
            mode="w+",
            dtype=np.float16,
            shape=(n_rows, n_bn, len(CANDIDATE_38_FEATURE_NAMES)),
        )
        cust_11 = np.empty((n_rows, len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)), dtype=np.float32)

        t_feat = time.perf_counter()
        for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
            c38, cu11 = engine.build_split(cid, date)
            cand_mm[i] = np.clip(c38, -65500.0, 65500.0).astype(np.float16)
            cust_11[i] = cu11
            if (i + 1) % 10000 == 0 or (i + 1) == n_rows:
                cand_mm.flush()
                print(
                    f"[v2-prep] {split} features {i + 1:,}/{n_rows:,} ({time.perf_counter() - t_feat:.1f}s)",
                    flush=True,
                )
        del cand_mm
        cust_11 = np.nan_to_num(cust_11, nan=0.0, posinf=1.0, neginf=-1.0)

        split_payload: Dict[str, Any] = {
            "customer_ids": list(cids),
            "prediction_days": torch.tensor(days, dtype=torch.float32),
            "first_observed": torch.tensor(first, dtype=torch.bool),
            "target_bn": torch.from_numpy(np.stack(targets)),
            "new_target_bn": torch.from_numpy(np.stack(new)),
            "allowed_bn": torch.from_numpy(np.stack(allowed)),
            "unseen_bn": torch.from_numpy(np.stack(unseen)),
            "cand_38_path": str(cand_mm_path),
            "feature_names": CANDIDATE_38_FEATURE_NAMES,
            "tabular_features": torch.from_numpy(cust_11),
            "customer_feature_names": CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
            "familiarity_bn": torch.cat(
                [familiarity.batch_features(cids[i : i + 512], dates[i : i + 512]) for i in range(0, n_rows, 512)]
            ),
            "text_embedding": torch.from_numpy(
                np.stack([company_texts.get(cid, np.zeros(1024, np.float32)) for cid in cids])
            ),
            **{
                k: torch.tensor([profiles.get(cid, {}).get(k, 0) for cid in cids], dtype=torch.long)
                for k in ("city_id", "region_id", "industry_id")
            },
            "sales_branch": torch.zeros(n_rows, dtype=torch.long),
            **sequences,
        }
        validate_dated_split(
            split_payload, split, train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2
        )

        if split == "train":
            print("[v2-prep] Computing strictly leak-free 60-day auxiliary labels (< TRAIN_CUTOFF_V2)...", flush=True)
            aux_60d_sku = build_strict_60d_aux_labels(
                dev_purchases,
                cids,
                days,
                product_ids,
                train_cutoff=TRAIN_CUTOFF_V2,
                test_start=TEST_START_V2,
            )
            split_payload["aux_60d_bn"] = _collapse_bool_matrix_to_bn(
                aux_60d_sku, universe["bn_to_member_pid_indices"]
            )
            split_payload["full_window"] = (
                split_payload["prediction_days"] <= (TRAIN_CUTOFF_V2 - PURCHASE_DATE_ORIGIN).days - 60
            )
        else:
            first_sales = dev_purchases.groupby(["customer_id", "product_id"]).purchase_date.min().unstack()
            known_at = first_sales.reindex(index=cids, columns=product_ids).to_numpy(dtype="datetime64[D]")
            target_np = np.stack(sku_targets)
            split_payload["sku_val"] = {
                "target": torch.from_numpy(target_np),
                "allowed": torch.from_numpy(np.stack(sku_allowed)),
                "new_target": torch.from_numpy(
                    target_np & ~(known_at < np.asarray(dates, dtype="datetime64[D]")[:, None])
                ),
            }

        atomic_torch_save(split_payload, PREPARED_ROOT / f"{split}_meta.pt")
        split_summaries[split] = {
            "baskets": n_rows,
            "first_observed_baskets": int(split_payload["first_observed"].sum().item()),
            "returning_baskets": int((~split_payload["first_observed"]).sum().item()),
            "total_bn_targets": int(split_payload["target_bn"].sum().item()),
            "new_bn_targets": int(split_payload["new_target_bn"].sum().item()),
            "cand_38_memmap": str(cand_mm_path),
            "cand_38_shape": [n_rows, n_bn, len(CANDIDATE_38_FEATURE_NAMES)],
        }

    shared_meta = {
        "universe": universe,
        "product_ids": product_ids,
        "reps": reps,
        "vocabs": vocabs,
        "enrichment": enrichment,
        "prod_inputs": {
            "text_embedding": bn_static["text_pack_free"].float(),
            "demand_features": bn_static["full_specs_multihot"].float(),
            "attr_ids": bn_static["attr_ids"],
        },
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
    }
    atomic_torch_save(shared_meta, PREPARED_ROOT / "shared_meta.pt")

    manifest = {
        "dataset": "v2_full",
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "num_skus": len(product_ids),
        "num_base_names": n_bn,
        "trained_base_names": int(universe["trained_base_names"]),
        "source_sha256": {
            "dataset_xlsx": file_digest(RAW_WORKBOOK_PATH),
            "customers_csv": file_digest(RAW_CUSTOMERS_PATH),
            "items_csv": file_digest(RAW_ITEMS_PATH),
            "cleaned_development_csv": file_digest(DEV_PURCHASES_PATH) if DEV_PURCHASES_PATH.is_file() else None,
            "commercial_dev_parquet": file_digest(COMMERCIAL_DEV_PARQUET_PATH),
        },
        "splits": split_summaries,
        "elapsed_seconds": float(time.perf_counter() - t_all),
    }
    atomic_json(manifest, manifest_path, overwrite=True)
    print(f"[v2-prep] Finished preparing v2_full in {manifest['elapsed_seconds']:.1f}s -> {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    prepare_v2_full_dataset(force_features=args.force_features)


if __name__ == "__main__":
    main()
