"""Frozen-weight snapshot sensitivity, separate from every Stage 2 ablation.

This does not train Stage 1 or establish a temporally valid pretrained model.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch

from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs
from recommender.shared_data import DEFAULT_RETRIEVER, load_development_purchases
from recommender.train_ranker import checkpoint_digest, load_ranker_retriever
from recommender.train_retriever import compute_product_demand_features


def main():
    before = checkpoint_digest(DEFAULT_RETRIEVER)
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(purchases=purchases, minimum_purchases=1)
    retriever, _ = load_ranker_retriever(DEFAULT_RETRIEVER, purchases, catalogue, companies, torch.device("cpu"))
    retriever.model.requires_grad_(False).eval()
    table = retriever.get_product_vectors()
    cache = torch.load("data/interim/shared_stage1_v1/train.pt", map_location="cpu")
    checkpoint = torch.load(DEFAULT_RETRIEVER, map_location="cpu")
    trained = set(checkpoint["product_ids"])
    index = {sku: i for i, sku in enumerate(retriever.eligible_product_ids)}
    categories = catalogue.set_index("product_id")["product_category"].dropna().to_dict()
    rows = []
    unique_days = cache["prediction_days"].unique()
    with torch.no_grad():
        for wanted in ("2025-03-01", "2025-06-01", "2025-09-01"):
            desired = (pd.Timestamp(wanted) - pd.Timestamp("2025-01-01")).days
            day = unique_days[(unique_days - desired).abs().argmin()]
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(day))
            started = time.perf_counter()
            demand = compute_product_demand_features(purchases, date, retriever.eligible_product_ids, categories)
            aggregation_seconds = time.perf_counter() - started
            inputs = {key: tensor.clone() for key, tensor in retriever.product_inputs.items()}
            inputs["demand_features"][:, :6] = demand
            started = time.perf_counter()
            changed = retriever.model.product_tower(
                inputs["text_embedding"], inputs["brand_id"], inputs["category_id"], inputs.get("subcategory_id"),
                inputs["business_line_id"], inputs["demand_features"], container_id=inputs.get("container_id"),
            )
            product_forward_seconds = time.perf_counter() - started
            eligible = sorted(index[sku] for sku in retriever.rolling_pool.eligible_on(date) & trained)
            batch_rows = torch.where(cache["prediction_days"] == day)[0][:64]
            customer_inputs = {name: cache[name][batch_rows] for name in (
                "text_embedding", "city_id", "region_id", "industry_id", "sequence", "attention_mask", "seq_product_ids", "seq_purchase_dates",
            )}
            customer_inputs["tabular_features"] = cache["retriever_tabular_features"][batch_rows]
            fixed_customer = retriever.model.customer_tower(**customer_inputs, catalog_product_vectors=table)
            rolling_customer = retriever.model.customer_tower(**customer_inputs, catalog_product_vectors=changed)
            product_distances = 1 - torch.nn.functional.cosine_similarity(table[eligible], changed[eligible])
            customer_distances = 1 - torch.nn.functional.cosine_similarity(fixed_customer, rolling_customer)
            rows.append({
                "date": str(date.date()), "eligible_trained_products": len(eligible), "customer_transitions": len(batch_rows),
                "products_with_changed_demand": int((demand[eligible] != retriever.product_inputs["demand_features"][eligible, :6]).any(1).sum()),
                "product_mean_cosine_distance": float(product_distances.mean()),
                "product_max_cosine_distance": float(product_distances.max()),
                "customer_mean_cosine_distance": float(customer_distances.mean()),
                "customer_max_cosine_distance": float(customer_distances.max()),
                "demand_snapshot_seconds_cpu": aggregation_seconds, "product_forward_seconds_cpu": product_forward_seconds,
            })
    assert checkpoint_digest(DEFAULT_RETRIEVER) == before
    output = Path("experiments/shared_stage1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_sensitivity.json").write_text(json.dumps({
        "stage1_digest": before, "mode": "frozen-weight sensitivity only; no retraining or quality evaluation",
        "comparison": "train-cutoff demand versus strictly prior-to-date demand; same weights/text/packaging/customer inputs",
        "limitation": "Weights still inherit Stage 1 training look-ahead. A clean rolling-demand Stage 1 training run remains separate future work.",
        "rows": rows,
    }, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
