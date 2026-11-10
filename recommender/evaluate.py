"""Phase 10: Comprehensive Evaluation, Baselines & Conditional Test-Gate Runner.

Evaluates and compares four next-basket recommendation systems:
1. Global Popularity Baseline (ranked by training split order frequency < 2025-10-01)
2. Personal Repeat-Purchase + Popularity Fallback Baseline (customer's historical SKU frequency
   and recency strictly prior to as_of_date, padded with global popularity)
3. Stage 1: Modular Late-Fusion Two-Tower Retriever (CustomerTower + ProductTower)
4. Stage 2: Unified Profile-Sequence Candidate-Aware Neural Ranker (UPS-CANR)

Metrics computed across all transitions and stratified by Cold/Light History (<= 2 prior baskets)
vs Warm/Heavy History (> 2 prior baskets):
- Recall@5, Recall@10, Recall@20, Recall@50, Recall@100
- HitRate@10
- NDCG@10
- MRR
- Cross-Attention Weight Breakdown (Token 0 Customer Profile vs Tokens 1..32 Purchase History)

Implements the Conditional Testing Gate:
- Evaluates on Validation split (2025-10-01 to 2026-02-18) first.
- Scores the held-out test split only after explicit opt-in and a passing validation gate.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.candidates import (
    NUM_PAIRWISE_FEATURES,
    PAIRWISE_FEATURE_NAMES,
    CandidateRetriever,
    rank_personal_repeat,
    rank_unseen_candidates,
)
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.ranker import UPSCANRRanker
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_ranker import (
    RankerDataset,
    SLATE_SCHEMA_VERSION,
    build_slate_cache,
    checkpoint_digest,
    collate_ranker_batch,
    load_ranker_retriever,
    load_slate_cache,
    prediction_day_offsets,
    prepare_slate_inputs,
    slate_cache_identity,
    slate_cache_path,
)
from recommender.train_retriever import build_retriever_transitions, get_device


def compute_single_transition_metrics(
    ranked_pids: Sequence[str],
    target_pids: Set[str],
    k_values: Sequence[int] = (5, 10, 20, 50, 100),
) -> Dict[str, float]:
    """Compute Recall@K, HitRate@10, NDCG@10, and MRR for a single transition's ranked list."""
    if not target_pids:
        return {f"recall@{k}": 0.0 for k in k_values} | {
            "hitrate@10": 0.0,
            "ndcg@10": 0.0,
            "mrr": 0.0,
        }

    n_targets = len(target_pids)
    metrics: Dict[str, float] = {}

    for k in k_values:
        hits_k = sum(1 for p in ranked_pids[:k] if p in target_pids)
        metrics[f"recall@{k}"] = hits_k / n_targets

    hits_10 = sum(1 for p in ranked_pids[:10] if p in target_pids)
    metrics["hitrate@10"] = 1.0 if hits_10 > 0 else 0.0

    dcg_10 = sum(
        1.0 / math.log2(r + 2)
        for r, p in enumerate(ranked_pids[:10])
        if p in target_pids
    )
    idcg_10 = sum(1.0 / math.log2(r + 2) for r in range(min(10, n_targets)))
    metrics["ndcg@10"] = (dcg_10 / idcg_10) if idcg_10 > 0 else 0.0

    rr = 0.0
    for r, p in enumerate(ranked_pids[:100]):
        if p in target_pids:
            rr = 1.0 / (r + 1)
            break
    metrics["mrr"] = rr
    return metrics


def aggregate_transition_metrics(
    per_transition: List[Dict[str, float]],
    prior_basket_counts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Aggregate per-transition metrics overall and stratified by Cold/Light (<=2) vs Warm (>2)."""
    if not per_transition:
        return {"overall": {}, "cold_light_lte_2_baskets": {}, "warm_gt_2_baskets": {}}

    keys = list(per_transition[0].keys())
    overall = {k: float(np.mean([m[k] for m in per_transition])) for k in keys}
    overall["num_transitions"] = len(per_transition)

    if prior_basket_counts is None or len(prior_basket_counts) != len(per_transition):
        return {"overall": overall}

    cold_Finalize = [
        m for m, c in zip(per_transition, prior_basket_counts) if c <= 2
    ]
    warm_Finalize = [
        m for m, c in zip(per_transition, prior_basket_counts) if c > 2
    ]

    cold_agg = (
        {k: float(np.mean([m[k] for m in cold_Finalize])) for k in keys}
        if cold_Finalize
        else {k: 0.0 for k in keys}
    )
    cold_agg["num_transitions"] = len(cold_Finalize)

    warm_agg = (
        {k: float(np.mean([m[k] for m in warm_Finalize])) for k in keys}
        if warm_Finalize
        else {k: 0.0 for k in keys}
    )
    warm_agg["num_transitions"] = len(warm_Finalize)

    return {
        "overall": overall,
        "cold_light_lte_2_baskets": cold_agg,
        "warm_gt_2_baskets": warm_agg,
    }


def build_global_popularity_ranking(
    purchases_df: pd.DataFrame,
    eligible_pids: Sequence[str],
    cutoff: Union[str, pd.Timestamp] = TRAIN_CUTOFF,
) -> List[str]:
    """Rank all eligible products by purchase frequency strictly prior to cutoff."""
    cutoff_ts = pd.Timestamp(cutoff)
    df = purchases_df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed")
    train_df = df.loc[df["purchase_date"] < cutoff_ts]

    eligible_set = set(str(p) for p in eligible_pids)
    counts = (
        train_df["product_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    ranked = sorted(
        [str(p) for p in eligible_pids],
        key=lambda pid: (-counts.get(pid, 0), pid),
    )
    return [p for p in ranked if p in eligible_set]


def build_personal_repeat_ranking(
    cust_history_df: pd.DataFrame,
    as_of_date: Union[str, pd.Timestamp],
    global_popularity: Sequence[str],
    max_k: int = 100,
) -> List[str]:
    """Rank customer's previously purchased products by order frequency & recency, padded with global popularity."""
    if cust_history_df is None or cust_history_df.empty:
        return list(global_popularity[:max_k])

    as_of_ts = pd.Timestamp(as_of_date)
    hist = cust_history_df.loc[cust_history_df["purchase_date"] < as_of_ts]
    if hist.empty:
        return list(global_popularity[:max_k])

    # Group by product_id: count distinct orders/events and most recent purchase date
    stats: Dict[str, Tuple[int, pd.Timestamp]] = {}
    for _, row in hist.iterrows():
        pid = str(row["product_id"])
        dt = row["purchase_date"]
        if pid not in stats:
            stats[pid] = (1, dt)
        else:
            cnt, last_dt = stats[pid]
            stats[pid] = (cnt + 1, max(last_dt, dt))

    return rank_personal_repeat(stats, global_popularity, max_k)


def evaluate_split(
    split_name: str,
    transitions: List[Dict[str, Any]],
    slates_data: Dict[str, Any],
    retriever: CandidateRetriever,
    ranker: UPSCANRRanker,
    catalog_product_vectors: torch.Tensor,
    eligible_pids: List[str],
    cust_groups: Dict[str, pd.DataFrame],
    global_pop_ranking: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Evaluate all 4 methods on a given split (Validation or Test)."""
    if ranker.statistics_attention_weights is not None:
        slates_data = {**slates_data, "prediction_days": prediction_day_offsets(transitions)}
    dataset = RankerDataset(slates_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranker_batch,
        num_workers=0,
    )

    # Compute prior basket count for each transition for Cold/Light vs Warm breakdown
    prior_basket_counts: List[int] = []
    prior_products: List[Set[str]] = []
    for t in transitions:
        cid = str(t["customer_id"])
        as_of = pd.Timestamp(t["as_of_date"])
        grp = cust_groups.get(cid)
        if grp is None or grp.empty:
            prior_basket_counts.append(0)
            prior_products.append(set())
        else:
            prior = grp.loc[grp["purchase_date"] < as_of]
            prior_basket_counts.append(int(prior["purchase_date"].nunique()))
            prior_products.append(set(prior["product_id"].astype(str)))

    # 1. Global Popularity & 2. Personal Repeat-Purchase Baselines
    pop_metrics_list: List[Dict[str, float]] = []
    repeat_metrics_list: List[Dict[str, float]] = []
    rolling_pool = retriever.rolling_pool
    eligible_by_date = {}
    popularity_by_date = {}
    if rolling_pool is not None:
        catalogue_ids = set(eligible_pids)
        for date in {t["as_of_date"] for t in transitions}:
            eligible_by_date[date] = rolling_pool.eligible_on(date) & catalogue_ids
            popularity_by_date[date] = [sku for sku in global_pop_ranking if sku in eligible_by_date[date]]
    heuristic_new_top10_hits = 0
    heuristic_new_top50_hits = 0
    for t, prior in zip(transitions, prior_products):
        cid = str(t["customer_id"])
        as_of = t["as_of_date"]
        targets = set(str(p) for p in t["positive_product_ids"])
        grp = cust_groups.get(cid, pd.DataFrame())

        popularity = popularity_by_date[as_of] if rolling_pool is not None else global_pop_ranking
        pop_metrics_list.append(compute_single_transition_metrics(popularity[:100], targets))
        rep_ranking = build_personal_repeat_ranking(grp, as_of, popularity, max_k=100)
        novel = targets - prior
        heuristic_new_top10_hits += len(novel.intersection(rep_ranking[:10]))
        heuristic_new_top50_hits += len(novel.intersection(rep_ranking[:50]))
        repeat_metrics_list.append(
            compute_single_transition_metrics(rep_ranking, targets)
        )

    # 3. Stage 1 Retriever & 4. Stage 2 UPS-CANR Ranker
    stage1_metrics_list: List[Dict[str, float]] = []
    mixed_metrics_list: List[Dict[str, float]] = []
    stage2_metrics_list: List[Dict[str, float]] = []
    slate_recalls: List[float] = []
    target_counts = {"repeat": 0, "new_to_customer": 0}
    tower_hits = dict(target_counts)
    slate_hits = dict(target_counts)
    reranked_top10_hits = dict(target_counts)
    discovery_top25_hits = 0
    discovery_top50_hits = 0
    token0_weights_all: List[float] = []
    history_weights_all: List[float] = []
    token0_weights_cold: List[float] = []
    token0_weights_warm: List[float] = []

    retriever.model.eval()
    ranker.eval()
    cat_vectors_dev = catalog_product_vectors.to(device)
    retriever_sequence = slates_data.get("retriever_sequence")

    trans_cursor = 0
    with torch.no_grad():
        for batch in loader:
            b_size = batch["text_embedding"].size(0)
            t_emb = batch["text_embedding"].to(device)
            city_id = batch["city_id"].to(device)
            reg_id = batch["region_id"].to(device)
            ind_id = batch["industry_id"].to(device)
            tab_f = batch["tabular_features"].to(device)
            retriever_tab_f = batch["retriever_tabular_features"].to(device)
            seq = batch["sequence"].to(device)
            mask = batch["attention_mask"].to(device)
            seq_pids = batch["seq_product_ids"].to(device)
            seq_cids = batch["seq_category_ids"].to(device)
            seq_sub_ids = batch["seq_subcategory_ids"].to(device)
            seq_line_ids = batch["seq_business_line_ids"].to(device)

            cand_idx = batch["candidate_indices"].to(device)
            pw_f = batch["pairwise_features"].to(device)

            # Stage 1 full-catalog top-100 ranking
            purchase_dates = batch["seq_purchase_dates"]
            tower_inputs = {
                name: tensor[trans_cursor:trans_cursor + b_size].to(device)
                for name, tensor in retriever_sequence.items()
            } if retriever_sequence else {
                "sequence": seq, "attention_mask": mask, "seq_product_ids": seq_pids,
                "seq_category_ids": seq_cids,
                "seq_purchase_dates": purchase_dates.to(device) if purchase_dates is not None else None,
            }
            u_c = retriever.model.customer_tower(
                text_embedding=t_emb,
                city_id=city_id,
                region_id=reg_id,
                industry_id=ind_id,
                tabular_features=retriever_tab_f,
                catalog_product_vectors=cat_vectors_dev,
                **tower_inputs,
            )
            rows = transitions[trans_cursor:trans_cursor + b_size]
            dates = [t["as_of_date"] for t in rows]
            full_logits = retriever.compute_logits(
                u_c, cat_vectors_dev, customer_ids=batch["customer_ids"], as_of_dates=dates,
            )
            if rolling_pool is None:
                top100_idx = full_logits.topk(k=min(100, len(eligible_pids)), dim=-1).indices
                top100 = [[eligible_pids[i] for i in row] for row in top100_idx.tolist()]
                eligible = [set(eligible_pids)] * b_size
            else:
                eligible = [eligible_by_date[date] for date in dates]
                top100 = rank_unseen_candidates(full_logits, eligible_pids, eligible, [set()] * b_size, 100)
            unseen = rank_unseen_candidates(
                full_logits, eligible_pids, eligible, prior_products[trans_cursor:trans_cursor + b_size], 50,
            )

            # Stage 2 UPS-CANR reranking over top-50 slate
            cand_vectors = batch["candidate_vectors"]
            cand_vectors = cat_vectors_dev[cand_idx] if cand_vectors is None else cand_vectors.to(device)
            out = ranker(
                text_embedding=t_emb,
                city_id=city_id,
                region_id=reg_id,
                industry_id=ind_id,
                tabular_features=tab_f,
                sequence=seq,
                attention_mask=mask,
                seq_product_ids=seq_pids,
                seq_category_ids=seq_cids,
                seq_subcategory_ids=seq_sub_ids,
                seq_business_line_ids=seq_line_ids,
                candidate_vectors=cand_vectors,
                pairwise_features=pw_f,
                catalog_product_vectors=cat_vectors_dev,
                seq_purchase_dates=purchase_dates.to(device) if purchase_dates is not None else None,
                prediction_day=batch["prediction_day"].to(device) if batch["prediction_day"] is not None else None,
            )

            ranker_logits_np = out.logits.cpu().numpy()
            t0_batch = out.token_0_weight.mean(dim=1).cpu().numpy()
            th_batch = out.history_weights.mean(dim=1).cpu().numpy()

            for b in range(b_size):
                targets = set(str(p) for p in batch["positive_pids"][b])
                prior_cnt = prior_basket_counts[trans_cursor + b]

                token0_weights_all.append(float(t0_batch[b]))
                history_weights_all.append(float(th_batch[b]))
                if prior_cnt <= 2:
                    token0_weights_cold.append(float(t0_batch[b]))
                else:
                    token0_weights_warm.append(float(t0_batch[b]))

                # Stage 1 top-100 list
                s1_pids = top100[b]
                stage1_metrics_list.append(
                    compute_single_transition_metrics(s1_pids, targets)
                )

                # Measure candidate coverage before scoring, then rerank.
                c_pids = batch["candidate_pids"][b]
                mixed_metrics_list.append(compute_single_transition_metrics(c_pids, targets))
                slate_top50 = set(c_pids[:50])
                tower_top50 = set(s1_pids[:50])
                prior = prior_products[trans_cursor + b]
                novel = targets - prior
                discovery_top25_hits += len(novel.intersection(unseen[b][:25]))
                discovery_top50_hits += len(novel.intersection(unseen[b]))
                slate_recalls.append(len(targets & slate_top50) / len(targets) if targets else 0.0)
                order = np.argsort(-ranker_logits_np[b])
                s2_top50 = [c_pids[i] for i in order]
                s2_seen = set(s2_top50)
                s2_full100 = s2_top50 + [p for p in s1_pids if p not in s2_seen]
                final_top10 = set(s2_full100[:10])
                for group, subset in (("repeat", targets & prior), ("new_to_customer", targets - prior)):
                    target_counts[group] += len(subset)
                    tower_hits[group] += len(subset & tower_top50)
                    slate_hits[group] += len(subset & slate_top50)
                    reranked_top10_hits[group] += len(subset & final_top10)
                stage2_metrics_list.append(
                    compute_single_transition_metrics(s2_full100[:100], targets)
                )

            trans_cursor += b_size

    return {
        "split": split_name,
        "num_transitions": len(transitions),
        "models": {
            "global_popularity": aggregate_transition_metrics(
                pop_metrics_list, prior_basket_counts
            ),
            "personal_repeat_baseline": aggregate_transition_metrics(
                repeat_metrics_list, prior_basket_counts
            ),
            "stage1_modular_two_tower_retriever": aggregate_transition_metrics(
                stage1_metrics_list, prior_basket_counts
            ),
            "stage1_mixed_slate": aggregate_transition_metrics(mixed_metrics_list, prior_basket_counts),
            "stage2_ups_canr_ranker": aggregate_transition_metrics(
                stage2_metrics_list, prior_basket_counts
            ),
        },
        "stage2_candidate_coverage": {
            "recall@50": float(np.mean(slate_recalls)) if slate_recalls else 0.0,
            "by_target": {
                group: {
                    "count": count,
                    "stage1_recall@50": tower_hits[group] / count if count else 0.0,
                    "stage2_slate_recall@50": slate_hits[group] / count if count else 0.0,
                    "stage2_reranked_recall@10": reranked_top10_hits[group] / count if count else 0.0,
                }
                for group, count in target_counts.items()
            },
        },
        "stage1_discovery": {
            "new_targets": target_counts["new_to_customer"],
            "new_to_customer_recall@25": discovery_top25_hits / max(target_counts["new_to_customer"], 1),
            "new_to_customer_recall@50": discovery_top50_hits / max(target_counts["new_to_customer"], 1),
            "personal_repeat_plus_popularity_new_recall@10": (
                heuristic_new_top10_hits / max(target_counts["new_to_customer"], 1)
            ),
            "personal_repeat_plus_popularity_new_recall@50": (
                heuristic_new_top50_hits / max(target_counts["new_to_customer"], 1)
            ),
        },
        "attention_diagnostics": {
            "overall_token0_profile_weight": float(np.mean(token0_weights_all)) if token0_weights_all else 0.0,
            "overall_history_tokens_weight": float(np.mean(history_weights_all)) if history_weights_all else 0.0,
            "cold_light_token0_profile_weight": float(np.mean(token0_weights_cold)) if token0_weights_cold else 0.0,
            "warm_token0_profile_weight": float(np.mean(token0_weights_warm)) if token0_weights_warm else 0.0,
        },
    }


def run_full_evaluation(
    retriever_checkpoint_path: Union[str, Path] = "checkpoints/retriever-discovery-reference/best_retriever.pt",
    ranker_checkpoint_path: Union[str, Path] = "checkpoints/ranker/best_ranker.pt",
    output_report_path: Union[str, Path] = "checkpoints/evaluation_report.json",
    device: Optional[str] = None,
    *,
    allow_test: bool = False,
) -> Dict[str, Any]:
    """Evaluate validation, and score test only with opt-in and a passing gate."""
    dev = get_device(device)
    print(f"Running evaluation on compute device: {dev}")

    retriever_checkpoint_path = Path(retriever_checkpoint_path)
    retriever_digest = checkpoint_digest(retriever_checkpoint_path)
    ranker_ckpt = torch.load(ranker_checkpoint_path, map_location="cpu")
    if (
        ranker_ckpt.get("retriever_digest") != retriever_digest
        or ranker_ckpt.get("slate_schema_version") != SLATE_SCHEMA_VERSION
    ):
        raise ValueError(
            "The ranker checkpoint does not match this retriever and slate version. "
            "Train the ranker with the selected retriever before running evaluation."
        )
    heuristic_fraction = ranker_ckpt.get("heuristic_fraction", 0.0)

    purchases_df = pd.read_csv(
        "data/interim/cleaned_purchases.csv",
        dtype={"customer_id": str, "product_id": str},
        low_memory=False,
    )
    purchases_df["purchase_date"] = pd.to_datetime(purchases_df["purchase_date"], format="mixed")
    all_purchases = purchases_df if allow_test else None
    purchases_df = purchases_df.loc[purchases_df["purchase_date"] < TEST_START]
    catalogue_df = load_product_catalogue(get_datasets_locs("items"))
    companies_df = load_companies(purchases=purchases_df, minimum_purchases=1)

    retriever, vocabs = load_ranker_retriever(
        retriever_checkpoint_path, purchases_df, catalogue_df, companies_df, dev,
    )
    rolling = retriever.rolling_pool is not None
    if ranker_ckpt.get("rolling_candidate_pool", False) != rolling:
        raise ValueError("The ranker checkpoint does not match the rolling candidate policy.")
    eligible_pids = retriever.eligible_product_ids
    product_vectors = retriever.get_product_vectors()
    ranker = UPSCANRRanker.from_retriever_checkpoint(
        checkpoint_path=retriever_checkpoint_path,
        device=dev,
        num_pairwise_features=NUM_PAIRWISE_FEATURES,
        statistics_attention=ranker_ckpt.get("statistics_attention", False),
        excluded_statistics=tuple(ranker_ckpt.get("excluded_statistics", ())),
    )
    ranker.load_state_dict(ranker_ckpt["model_state_dict"])
    ranker.eval()

    train_transitions, val_transitions, _ = build_retriever_transitions(purchases_df)
    cust_groups = {str(cid): group for cid, group in purchases_df.groupby("customer_id")}
    global_pop = build_global_popularity_ranking(purchases_df, eligible_pids, cutoff=TRAIN_CUTOFF)
    val_identity = slate_cache_identity(
        retriever_digest, retriever_checkpoint_path, purchases_df, catalogue_df, companies_df,
        include_subcategory=retriever.model.product_tower.subcat_embedding is not None,
        heuristic_fraction=heuristic_fraction,
        company_cache_name=retriever.company_cache_name,
        rolling=rolling,
        product_inputs=retriever.product_inputs,
    )
    val_path = slate_cache_path("val", heuristic_fraction, rolling=rolling)
    val_slates = load_slate_cache(val_path, f"{val_identity}:val")
    if val_slates is None:
        slate_inputs = prepare_slate_inputs(
            retriever, retriever_checkpoint_path, purchases_df, catalogue_df, companies_df,
            train_transitions, vocabs,
        )
        slate_inputs["heuristic_fraction"] = heuristic_fraction
        val_slates = build_slate_cache(val_path, f"{val_identity}:val", val_transitions, slate_inputs)

    print(f"Evaluating on Validation split ({len(val_transitions)} transitions)...")
    val_report = evaluate_split(
        split_name="validation",
        transitions=val_transitions,
        slates_data=val_slates,
        retriever=retriever,
        ranker=ranker,
        catalog_product_vectors=product_vectors,
        eligible_pids=eligible_pids,
        cust_groups=cust_groups,
        global_pop_ranking=global_pop,
        device=dev,
    )

    s1_val_ndcg = val_report["models"]["stage1_modular_two_tower_retriever"]["overall"]["ndcg@10"]
    s2_val_ndcg = val_report["models"]["stage2_ups_canr_ranker"]["overall"]["ndcg@10"]
    rep_val_ndcg = val_report["models"]["personal_repeat_baseline"]["overall"]["ndcg@10"]
    s2_val_rec10 = val_report["models"]["stage2_ups_canr_ranker"]["overall"]["recall@10"]

    validation_gate_passed = bool(
        s2_val_ndcg > s1_val_ndcg
        and s2_val_ndcg >= rep_val_ndcg
        and s2_val_rec10 >= 0.50
    )

    report: Dict[str, Any] = {
        "architecture_name": "Modular Two-Tower Retriever + Unified Profile-Sequence Candidate-Aware Neural Ranker (UPS-CANR)",
        "heuristic_fraction": heuristic_fraction,
        "statistics_attention": ranker.statistics_attention_weights is not None,
        "excluded_statistics": list(ranker.excluded_statistics),
        "ranker_training": {
            "epoch": ranker_ckpt.get("epoch"),
            "new_positive_weight": ranker_ckpt.get("config", {}).get("new_positive_weight", 1.0),
            "selection_metric": ranker_ckpt.get("best_metric_name", "ndcg@10"),
            "discovery_min_overall_recall": ranker_ckpt.get("discovery_min_overall_recall"),
            "seed": ranker_ckpt.get("config", {}).get("seed", 42),
        },
        "rolling_candidate_pool": rolling,
        "ranker_pairwise_features": list(PAIRWISE_FEATURE_NAMES),
        "validation_gate": {
            "passed": validation_gate_passed,
            "stage1_val_ndcg@10": s1_val_ndcg,
            "stage2_val_ndcg@10": s2_val_ndcg,
            "personal_repeat_val_ndcg@10": rep_val_ndcg,
            "stage2_val_recall@10": s2_val_rec10,
        },
        "validation": val_report,
    }

    if validation_gate_passed and all_purchases is not None:
        _, _, test_transitions = build_retriever_transitions(all_purchases)
        print(f"Validation gate passed. Evaluating {len(test_transitions)} test transitions.")
        test_companies = load_companies(purchases=all_purchases, minimum_purchases=1)
        test_retriever, test_vocabs = load_ranker_retriever(
            retriever_checkpoint_path, all_purchases, catalogue_df, test_companies, dev,
        )
        test_identity = slate_cache_identity(
            retriever_digest, retriever_checkpoint_path, all_purchases, catalogue_df, test_companies,
            include_subcategory=test_retriever.model.product_tower.subcat_embedding is not None,
            heuristic_fraction=heuristic_fraction,
            company_cache_name=test_retriever.company_cache_name,
            rolling=rolling,
            product_inputs=test_retriever.product_inputs,
        )
        test_path = slate_cache_path("test", heuristic_fraction, rolling=rolling)
        test_slates = load_slate_cache(test_path, f"{test_identity}:test")
        if test_slates is None:
            slate_inputs = prepare_slate_inputs(
                test_retriever, retriever_checkpoint_path, all_purchases, catalogue_df,
                test_companies, train_transitions, test_vocabs,
            )
            slate_inputs["heuristic_fraction"] = heuristic_fraction
            test_slates = build_slate_cache(
                test_path, f"{test_identity}:test", test_transitions, slate_inputs,
            )
        test_groups = {str(cid): group for cid, group in all_purchases.groupby("customer_id")}
        test_report = evaluate_split(
            split_name="test",
            transitions=test_transitions,
            slates_data=test_slates,
            retriever=test_retriever,
            ranker=ranker,
            catalog_product_vectors=product_vectors,
            eligible_pids=eligible_pids,
            cust_groups=test_groups,
            global_pop_ranking=global_pop,
            device=dev,
        )
        report["test"] = test_report
    elif not validation_gate_passed:
        print("Validation gate did not pass.")
    else:
        print("Validation gate passed. Held-out test evaluation requires explicit opt-in.")

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {out_path}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
