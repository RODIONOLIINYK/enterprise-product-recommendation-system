"""Compare Stage 1 discovery and heuristic slates on chronological validation."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import chain, combinations
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.candidates import rank_personal_repeat, rank_unseen_candidates
from recommender.development import load_development_purchases
from recommender.catalogue import catalogue_blocked_mask, load_product_catalogue
from recommender.company import load_companies
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.stage1_catalogue import load_stage1_discovery
from recommender.text import load_cached_embeddings
from recommender.train_retriever import (
    RetrieverDataset, build_customer_metadata, build_retriever_transitions, get_device,
)


DISCOVERY_K = 25
SLATE_K = 50
DEFAULT_POOL = "prior_sold_without_known_block"


def take_unique(ranking, exclude, limit):
    """Take the first eligible distinct products from an ordered source."""
    result = []
    seen = set(exclude)
    for sku in ranking:
        if sku in seen:
            continue
        seen.add(sku)
        result.append(sku)
        if len(result) == limit:
            break
    return result


class Heuristics:
    """Fit popularity, industry adoption, and basket co-purchases on training sales."""

    def __init__(self, train: pd.DataFrame, industries: dict[str, str]):
        self.sales = Counter(train["product_id"])
        first = train.drop_duplicates(["customer_id", "product_id"])
        self.industry_sales = defaultdict(Counter)
        for customer, sku in first[["customer_id", "product_id"]].itertuples(index=False, name=None):
            if industry := industries.get(customer):
                self.industry_sales[industry][sku] += 1
        basket_counts = Counter()
        pair_counts = defaultdict(Counter)
        for _, basket in train.groupby(["customer_id", "purchase_date"], sort=False):
            products = sorted(set(basket["product_id"]))
            basket_counts.update(products)
            for left, right in combinations(products, 2):
                pair_counts[left][right] += 1
                pair_counts[right][left] += 1
        self.complements = {
            sku: {other: count / basket_counts[sku] for other, count in counts.items()}
            for sku, counts in pair_counts.items()
        }

    def shortlist_sources(self, eligible, seen, industry, popularity, popularity_rank):
        industry_ranking = sorted(
            self.industry_sales.get(industry, {}).keys() & eligible,
            key=lambda sku: (-self.industry_sales[industry][sku], popularity_rank[sku]),
        )
        complement_scores = {}
        for anchor in seen:
            for sku, score in self.complements.get(anchor, {}).items():
                if sku in eligible and sku not in seen:
                    complement_scores[sku] = max(score, complement_scores.get(sku, 0.0))
        co_purchase = sorted(complement_scores, key=lambda sku: (-complement_scores[sku], popularity_rank[sku]))
        return {
            "unseen_popularity": popularity,
            "industry_adoption": chain(industry_ranking, popularity),
            "co_purchase": chain(co_purchase, popularity),
        }


class Coverage:
    def __init__(self):
        self.baskets = 0
        self.basket_recall = 0.0
        self.hits = Counter()
        self.targets = Counter()

    def add(self, ranking, targets, groups):
        selected = set(ranking)
        self.baskets += 1
        self.basket_recall += len(targets & selected) / len(targets)
        for name, subset in groups.items():
            self.targets[name] += len(subset)
            self.hits[name] += len(subset & selected)

    def report(self):
        return {
            "overall_basket_recall": self.basket_recall / self.baskets if self.baskets else 0.0,
            "by_target": {
                name: {"hits": self.hits[name], "count": count,
                       "recall": self.hits[name] / count if count else 0.0}
                for name, count in self.targets.items()
            },
        }


def evaluate_stage1(
    checkpoint_path: Path, output_dir: Path, device: str | None = None,
    *, embedding_cache: Path = Path("checkpoints/stage1-full-catalogue/text_embeddings"),
    compare_pools: bool = False,
) -> dict:
    """Replay catalogue policies using fixed weights and strictly prior observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    history = load_development_purchases("data/interim/cleaned_purchases.csv")
    history["purchase_date"] = history["purchase_date"].dt.normalize()
    train = history.loc[history["purchase_date"] < TRAIN_CUTOFF]
    _, validation, _ = build_retriever_transitions(history)
    if not validation:
        raise ValueError("Stage 1 replay requires validation baskets.")

    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(purchases=history, minimum_purchases=1)
    retriever, ckpt = load_stage1_discovery(
        checkpoint_path, history, catalogue, companies, embedding_cache, dev,
    )
    config = ckpt["config"]
    trained_ids = ckpt["product_ids"]
    trained_set = set(trained_ids)
    vocabs = ckpt["vocabs"]
    expanded_ids = retriever.eligible_product_ids
    pool = retriever.rolling_pool
    merchandise = pool.merchandise
    vectors = retriever.get_product_vectors()
    history_vectors = vectors[:len(trained_ids)]
    metadata = catalogue.set_index("product_id")[["product_category", "product_subcategory", "business_line"]]
    company_embeddings = load_cached_embeddings(checkpoint_path.parent / retriever.company_cache_name, "customer_id")
    dataset = RetrieverDataset(
        validation, history, ckpt["product_to_idx"], build_customer_metadata(companies, retriever.customer_vocabs),
        company_embeddings, retriever.profile_normalizer, vocabs, metadata.to_dict("index"),
        max_seq_len=config["max_seq_len"], profile_features=retriever.profile_features,
        quantity_reference=retriever.quantity_reference,
    )
    industries = companies.set_index("customer_id")["industry_branch"].dropna().astype(str).to_dict()
    heuristics = Heuristics(train, industries)
    pool_sizes = {DEFAULT_POOL: len(expanded_ids)}
    if compare_pools:
        pool_sizes.update(training_sold=len(trained_ids), merchandise_without_known_block=len(expanded_ids))
    discovery_metrics = {name: defaultdict(Coverage) for name in pool_sizes}
    slate_metrics = {name: defaultdict(Coverage) for name in pool_sizes}
    first_sales = pool.first_sales
    popularity = sorted(expanded_ids, key=lambda sku: (-heuristics.sales[sku], sku))
    popularity_rank = {sku: i for i, sku in enumerate(popularity)}
    counts_by_pool = defaultdict(list)

    @lru_cache(maxsize=None)
    def pools_on(date):
        blocked = pool.blocked_on(date)
        eligible_by_pool = {DEFAULT_POOL: pool.eligible_on(date)}
        if compare_pools:
            eligible_by_pool["training_sold"] = trained_set - blocked
            eligible_by_pool["merchandise_without_known_block"] = pool.product_ids - blocked
        dated = {}
        for name, eligible in eligible_by_pool.items():
            dated[name] = (eligible, [sku for sku in popularity if sku in eligible])
            counts_by_pool[name].append(len(eligible))
        return dated, blocked

    ranges = {}

    def check_range(name, values):
        low, high = float(values.min()), float(values.max())
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"{name} contains out-of-range numeric model inputs: {low}, {high}")
        prev_low, prev_high = ranges.get(name, (1.0, 0.0))
        ranges[name] = [min(prev_low, low), max(prev_high, high)]

    demand = retriever.product_inputs["demand_features"]
    check_range("product_demand", demand[:, :6])
    if demand.shape[1] > 6:
        check_range("product_container_one_hot", demand[:, 6:])

    def check_history_inputs(_layer, args):
        check_range("purchase_quantity", args[0][..., -2])
        check_range("purchase_gap", args[0][..., -1])

    hook = retriever.model.customer_tower.token_proj.register_forward_pre_hook(check_history_inputs)
    fields = ("text_embedding", "city_id", "region_id", "industry_id", "tabular_features", "sequence",
              "attention_mask", "seq_product_ids", "seq_purchase_dates")
    offset = 0
    try:
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=128):
                rows = validation[offset:offset + len(batch["target"])]
                customer_ids = [t["customer_id"] for t in rows]
                dates = [t["as_of_date"] for t in rows]
                snapshots = [retriever.feature_store.get_customer_snapshot(cid, date)
                             for cid, date in zip(customer_ids, dates)]
                seen_sets = [set(snapshot.sku_stats) for snapshot in snapshots]
                for i, feature in enumerate(retriever.profile_features):
                    check_range(feature, batch["tabular_features"][:, i])
                customers = retriever.model.customer_tower(
                    **{field: batch[field].to(dev) for field in fields}, catalog_product_vectors=history_vectors,
                )
                logits = retriever.compute_logits(customers, vectors, customer_ids=customer_ids, as_of_dates=dates)
                dated_pools = [pools_on(date) for date in dates]
                ranked = {
                    name: rank_unseen_candidates(
                        logits[:, :size], expanded_ids[:size], [dated[name][0] for dated, _ in dated_pools],
                        seen_sets, SLATE_K,
                    )
                    for name, size in pool_sizes.items()
                }
                for i, (row, snapshot, seen) in enumerate(zip(rows, snapshots, seen_sets)):
                    targets = set(row["positive_product_ids"])
                    new = targets - seen
                    outside = new - trained_set
                    dated, blocked = dated_pools[i]
                    groups = {
                        "new": new, "repeat": targets & seen, "outside_training_catalogue": outside,
                        "globally_first_sale": {sku for sku in outside if first_sales[sku] == row["as_of_date"]},
                        "known_blocked_at_target_date": targets & blocked,
                    }
                    stats = {sku: (data["order_count"], data["last_date"]) for sku, data in snapshot.sku_stats.items()}
                    for name, (eligible, available_popularity) in dated.items():
                        sources = heuristics.shortlist_sources(
                            eligible, seen, industries.get(row["customer_id"]), available_popularity, popularity_rank,
                        )
                        sources["two_tower"] = ranked[name][i]
                        repeat = rank_personal_repeat(stats, available_popularity, SLATE_K)
                        slate_metrics[name]["global_popularity"].add(available_popularity[:SLATE_K], targets, groups)
                        slate_metrics[name]["personal_repeat_plus_popularity"].add(repeat, targets, groups)
                        for method, source in sources.items():
                            discovery = take_unique(source, seen, SLATE_K)
                            discovery_metrics[name][method].add(discovery[:DISCOVERY_K], targets, groups)
                            reserved = repeat[:DISCOVERY_K]
                            slate = reserved + take_unique(discovery, reserved, DISCOVERY_K)
                            slate_metrics[name][f"mixed_{method}"].add(slate, targets, groups)
                offset += len(rows)
    finally:
        hook.remove()
    if retriever.familiarity is not None:
        flags = retriever.familiarity.batch_features(customer_ids, dates)
        check_range("familiarity_flags", flags)

    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": ckpt["epoch"],
        "validation_baskets": len(validation), "product_statistics_before": str(TRAIN_CUTOFF.date()),
        "default_candidate_pool": DEFAULT_POOL,
        "discovery_k": DISCOVERY_K, "slate_k": SLATE_K, "numeric_input_ranges": ranges,
        "catalogue_status": {
            "merchandise_count": len(merchandise),
            "snapshot_blocked_count": int(catalogue_blocked_mask(merchandise).sum()),
            "known_blocked_before_validation": int(catalogue_blocked_mask(merchandise, TRAIN_CUTOFF).sum()),
            "historical_policy": "Apply snapshot block flags only when last modification predates the basket.",
        },
        "catalogues": {
            name: {"candidate_count_min": min(counts_by_pool[name]),
                   "candidate_count_max": max(counts_by_pool[name]),
                   "discovery_at_25": {key: metric.report() for key, metric in discovery_metrics[name].items()},
                   "slate_at_50": {key: metric.report() for key, metric in slate_metrics[name].items()}}
            for name in pool_sizes
        },
    }
    (output_dir / "validation_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/retriever-discovery-reference/best_retriever.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage1-rolling-discovery"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("checkpoints/stage1-full-catalogue/text_embeddings"))
    parser.add_argument("--device")
    parser.add_argument("--compare-pools", action="store_true", help="Include the two alternative catalogue pools.")
    args = parser.parse_args()
    report = evaluate_stage1(
        args.checkpoint, args.output_dir, args.device,
        embedding_cache=args.embedding_cache, compare_pools=args.compare_pools,
    )
    for name, catalogue in report["catalogues"].items():
        print(f"{name} ({catalogue['candidate_count_min']:,} to {catalogue['candidate_count_max']:,} candidates)")
        for evaluation in ("discovery_at_25", "slate_at_50"):
            for method, result in catalogue[evaluation].items():
                print(evaluation, method, f"new recall {result['by_target']['new']['recall']:.2%}",
                      f"overall recall {result['overall_basket_recall']:.2%}")


if __name__ == "__main__":
    main()
