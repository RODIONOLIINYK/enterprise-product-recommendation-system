"""Versioned E3–E5 training inputs, with no demand look-ahead or held-out reads."""
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.familiarity import FamiliarityStore
from recommender.scope import get_datasets_locs
from recommender.shared_data import DEFAULT_RETRIEVER, load_development_purchases, atomic_torch_save
from recommender.experiment_signals import load_inputs
from recommender.train_ranker import load_ranker_retriever, checkpoint_digest
from recommender.split import TRAIN_CUTOFF, TEST_START

CACHE = Path("data/interim/e1_e5_retrieval_v1")
FIELDS = ("text_embedding", "city_id", "region_id", "industry_id", "sequence", "attention_mask", "seq_purchase_dates")


def older_history(purchases, data, product_ids):
    index = {sku: i + 1 for i, sku in enumerate(product_ids)}
    histories = {str(cid): group.sort_values(["purchase_date", "product_id"], kind="stable")
                 for cid, group in purchases.groupby("customer_id", sort=False)}
    rows = []
    for cid, day in zip(data["customer_ids"], data["prediction_days"].tolist()):
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(day))
        history = histories[str(cid)]
        prior = history.loc[history.purchase_date < date]
        older = prior.iloc[:-32]
        rows.append(Counter(index[str(sku)] for sku in older.product_id if str(sku) in index))
    width = max(1, max(map(len, rows), default=0))
    ids, counts = torch.zeros(len(rows), width, dtype=torch.long), torch.zeros(len(rows), width)
    for row, counter in enumerate(rows):
        if counter:
            ids[row, :len(counter)] = torch.tensor(list(counter))
            counts[row, :len(counter)] = torch.tensor(list(counter.values()), dtype=torch.float32)
    return ids, counts


def prepare_retrieval_data():
    train, val, manifest = load_inputs()
    identity = f"e3e5-v1:no-demand:scratch:{manifest['identity']}"
    paths = [CACHE / f"{name}.pt" for name in ("train", "val", "manifest")]
    if all(path.exists() for path in paths):
        loaded = [torch.load(path, map_location="cpu") for path in paths]
        if any(data["identity"] != identity for data in loaded):
            raise ValueError("Experimental retrieval cache mismatch")
        return *loaded, train, val
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(purchases=purchases, minimum_purchases=1)
    retriever, _ = load_ranker_retriever(DEFAULT_RETRIEVER, purchases, catalogue, companies, torch.device("cpu"))
    original = torch.load(DEFAULT_RETRIEVER, map_location="cpu")
    all_ids = manifest["product_ids"]
    # Sparsity optimization only: unobserved catalogue SKUs cannot pass prior-sale
    # eligibility on any development date. No counts/features/targets from a
    # validation basket are ever supplied to an earlier training row.
    trained = list(original["product_ids"])
    ids = trained + sorted((set(purchases.product_id.astype(str)) & set(all_ids)) - set(trained))
    index = {sku: i for i, sku in enumerate(ids)}
    expanded_indices = torch.tensor([all_ids.index(sku) for sku in ids])
    inputs = {key: value[expanded_indices].clone() for key, value in retriever.product_inputs.items()}
    inputs["demand_features"][:, :6] = 0  # independent of all observation dates
    fields = tuple(original["config"]["familiarity_fields"])
    familiarity = FamiliarityStore(purchases, catalogue, ids, fields)
    outputs = []
    for split, source in (("train", train), ("val", val)):
        data = {key: source[key] for key in FIELDS}
        data["tabular_features"] = source["retriever_tabular_features"]
        data["seq_product_ids"] = torch.tensor([[index.get(sku, -1) + 1 for sku in row] for row in source["seq_canonical_skus"]])
        data["older_ids"], data["older_counts"] = older_history(purchases, source, ids)
        data["target"] = torch.zeros(len(source["customer_ids"]), len(ids), dtype=torch.bool)
        data["new_target"] = data["target"].clone()
        data["allowed"] = data["target"].clone()
        data["unseen"] = data["target"].clone()
        dates = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(day)) for day in source["prediction_days"].tolist()]
        by_date = {date: retriever.rolling_pool.eligible_on(date) for date in set(dates)}
        for row, (cid, date, positives, new) in enumerate(zip(source["customer_ids"], dates, source["positive_pids"], source["new_positive_pids"])):
            seen = set(retriever.feature_store.get_customer_snapshot(cid, date).sku_stats)
            allowed = by_date[date]
            data["target"][row, [index[sku] for sku in positives if sku in index]] = True
            data["new_target"][row, [index[sku] for sku in new if sku in index]] = True
            data["allowed"][row, [index[sku] for sku in allowed if sku in index]] = True
            data["unseen"][row, [index[sku] for sku in allowed - seen if sku in index]] = True
        data["familiarity"] = familiarity.batch_features(source["customer_ids"], dates).bool()
        data["identity"] = identity
        atomic_torch_save(data, CACHE / f"{split}.pt")
        outputs.append(data)
        print(f"Prepared {split}: rows={len(dates)}, products={len(ids)}, older_width={data['older_ids'].shape[1]}", flush=True)
    info = {"identity": identity, "product_ids": ids, "expanded_indices": expanded_indices, "product_inputs": inputs,
            "reference_digest": checkpoint_digest(DEFAULT_RETRIEVER), "trained_products": len(trained),
            "demand_policy": "six demand inputs disabled, training and serving; packaging retained",
            "initialization": "all learned tower/attention/bias weights reset; reference metadata and feature scales only",
            "train_cutoff": str(TRAIN_CUTOFF.date()), "heldout_excluded": str(TEST_START.date())}
    atomic_torch_save(info, CACHE / "manifest.pt")
    return *outputs, info, train, val
