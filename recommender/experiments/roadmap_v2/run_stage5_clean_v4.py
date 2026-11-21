"""Validation-only Stage 5 v4 Clean Architecture run on modest_8h_ffn384_mlp384 (seed=42, zero Test access).

Implements ALL user-specified architectural & representation upgrades together:
  1. Pure pack-free BGE-M3 text (585 distinct base_names; packaging & volume excluded from text).
  2. Full 65D structured multi-hot specs in ProductTower:
       12D container_type + 8D volume tiers + 11D application_domain + 34D fixed multi-value viscosity_grade.
  3. Fresh 19D point-in-time Customer State (8D profile + 11D cadence/overdue/30d/90d/financial/manager-change)
     MOVED OUT of candidate rows and INTO BOTH:
       - Stage 1 CustomerTower (live u_bn updated with fresh customer state!)
       - Stage 2 Token 0 Customer Profile Embedding (live in Token 0 value + statistics_attention bias!)
  4. Strictly 38D Candidate / Customer-Candidate features:
       - Zero month_sin/month_cos, zero history_month
       - Zero broadcast customer constants
       - Zero exact binary/formula duplicates (category_never_seen, family_never_seen,
         same_viscosity_grade_bought, base_name_hierarchical_backoff removed)
       - Keeps base_name_orders_90d & fixed multi-value user_viscosity_order_share
       - All customer/spec shares computed natively on collapsed (customer_id, purchase_date, base_name) events
  5. Zero dead weights in Stage 2 (CleanStage5BaseNameRanker deletes dead product_text_proj & unused base.mlp).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json
from recommender.catalogue import RollingCandidatePool, load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.experiments.static_enrichment.metadata import encode_categorical_values, load_enriched_companies
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.features import LEGACY_PROFILE_FEATURES, PointInTimeNormalizer, compute_point_in_time_features
from recommender.ranker import RankerLoss
from recommender.scope import get_datasets_locs
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.text import load_cached_embeddings
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata, get_device

from .augmentation import build_strict_60d_aux_labels
from .clean_ranker import CleanStage5BaseNameRanker
from .commercial_features import COMMERCIAL_PARQUET_PATH
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_WEIGHT_DECAY,
)
from .dated_baskets import (
    ENRICHMENT,
    VOCABS,
    aligned_commercial_events,
    basket_cases,
    first_observed_pack_indices,
    validate_dated_split,
)
from .native_base_name import (
    APPLICATION_BUCKETS,
    CANDIDATE_38_FEATURE_NAMES,
    CONTAINER_BUCKETS,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
    VISCOSITY_BUCKETS,
    VOLUME_BUCKETS,
    NativeBaseNameFeatureEngine,
    build_base_name_static_representations,
)
from .run_stage5 import _collapse_bool_matrix_to_bn, _verify_reference_files, train_stage5_two_tower
from .run_stage5_ablations import _evaluate_with_cohorts_and_sku
from .stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
)


OUT_ROOT = ARTIFACT_ROOT / "stage5_clean_v4"
SUMMARY_PATH = EVIDENCE_ROOT / "stage5_clean_v4_seed42.json"
REF_3STAGE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"


def _prepare_clean_v4_data():
    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_enriched_companies(purchases=purchases, minimum_purchases=0)
    enrichment = torch.load(ENRICHMENT, map_location="cpu", weights_only=False)
    product_ids = list(enrichment["product_ids"])
    vocabs = torch.load(VOCABS, map_location="cpu", weights_only=False)["vocabs"]
    config = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["config"]
    text_file = company_embeddings_path(DEFAULT_RETRIEVER, config["company_cache_name"])
    texts = load_cached_embeddings(text_file, "customer_id")
    normalizer = PointInTimeNormalizer.from_state(config["profile_normalizer"])
    profiles = build_customer_metadata(companies, vocabs)
    sales_branch = companies.set_index("customer_id").sales_branch.to_dict()

    pool = RollingCandidatePool(catalogue, purchases)
    universe = build_base_name_universe(product_ids, catalogue, purchases)
    reps = first_observed_pack_indices(product_ids, universe["bn_to_member_pid_indices"], pool.first_sales)
    representatives = catalogue.set_index("product_id").loc[np.asarray(product_ids)[reps]].reset_index()

    commercial = aligned_commercial_events(
        pd.read_parquet(COMMERCIAL_PARQUET_PATH, filters=[("purchase_date", "<", TEST_START)]),
        purchases,
    )
    engine = NativeBaseNameFeatureEngine(
        purchases, commercial, catalogue, companies, product_ids, universe, reps
    )
    familiarity = FamiliarityStore(
        purchases, catalogue, np.asarray(product_ids)[reps], tuple(ATTRIBUTE_COLUMNS)
    )
    histories = dict(tuple(purchases.groupby("customer_id", sort=False)))
    cases, _ = basket_cases(purchases, product_ids, universe, pool)

    splits, sku_data = {}, {}
    for split, rows in cases.items():
        cids, dates, first, targets, new, allowed, unseen, sku_targets, sku_allowed = zip(*rows, strict=True)
        days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]
        sequences = build_base_name_sequences(
            purchases, cids, days, universe, representatives,
            vocabs["category_vocab"], vocabs["subcategory_vocab"], vocabs["business_line_vocab"],
        )
        # Permanently zero calendar month (cols 3:5) in sequence
        sequences["sequence"][..., 3:5] = 0.0

        cand_feats = np.empty((len(rows), len(reps), len(CANDIDATE_38_FEATURE_NAMES)), np.float16)
        cust_19 = np.empty((len(rows), len(LEGACY_PROFILE_FEATURES) + len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES)), np.float32)
        t_feat = time.perf_counter()
        for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
            c38, cu11 = engine.build_split(cid, date)
            cand_feats[i] = c38
            stats = compute_point_in_time_features(
                histories[cid], cid, date, normalizer=normalizer, include_median_purchase_quantity=True
            )
            p8 = np.array([stats[k] for k in LEGACY_PROFILE_FEATURES], dtype=np.float32)
            cust_19[i, :8] = p8
            cust_19[i, 8:] = cu11
            if (i + 1) % 5000 == 0:
                print(f"[clean-v4] {split} {i + 1}/{len(rows)} ({time.perf_counter() - t_feat:.1f}s)", flush=True)

        cust_19 = np.nan_to_num(cust_19, nan=0.0, posinf=1.0, neginf=-1.0)
        splits[split] = {
            "customer_ids": list(cids),
            "prediction_days": torch.tensor(days, dtype=torch.float32),
            "first_observed": torch.tensor(first, dtype=torch.bool),
            "target_bn": torch.from_numpy(np.stack(targets)),
            "new_target_bn": torch.from_numpy(np.stack(new)),
            "allowed_bn": torch.from_numpy(np.stack(allowed)),
            "unseen_bn": torch.from_numpy(np.stack(unseen)),
            "features_bn": torch.from_numpy(cand_feats),
            "feature_names": CANDIDATE_38_FEATURE_NAMES,
            "tabular_features": torch.from_numpy(cust_19),  # 19D fresh customer state (8 profile + 11 dynamic)!
            "customer_feature_names": tuple(LEGACY_PROFILE_FEATURES) + CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
            "familiarity_bn": torch.cat(
                [familiarity.batch_features(cids[i : i + 256], dates[i : i + 256]) for i in range(0, len(rows), 256)]
            ),
            "text_embedding": torch.from_numpy(
                np.stack([texts.get(cid, np.zeros(1024, np.float32)) for cid in cids])
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
            first_sales = purchases.groupby(["customer_id", "product_id"]).purchase_date.min().unstack()
            known_at = first_sales.reindex(index=cids, columns=product_ids).to_numpy(dtype="datetime64[D]")
            target_np = np.stack(sku_targets)
            sku_data[split] = {
                "target": torch.from_numpy(target_np),
                "allowed": torch.from_numpy(np.stack(sku_allowed)),
                "new_target": torch.from_numpy(
                    target_np & ~(known_at < np.asarray(dates, dtype="datetime64[D]")[:, None])
                ),
            }

    bn_static = build_base_name_static_representations(universe, product_ids, catalogue, reps)
    return splits, bn_static, universe, enrichment, vocabs, purchases, catalogue, product_ids, sku_data, reps, engine


def _build_clean_batch(split_d: Mapping[str, Any], v_bn_dev: torch.Tensor, rows: torch.Tensor, dev: torch.device) -> Dict[str, Any]:
    b = len(rows)
    return {
        "text_embedding": split_d["text_embedding"][rows].to(dev),
        "city_id": split_d["city_id"][rows].to(dev),
        "region_id": split_d["region_id"][rows].to(dev),
        "industry_id": split_d["industry_id"][rows].to(dev),
        "sales_branch": split_d["sales_branch"][rows].to(dev),
        "customer_features": split_d["tabular_features"][rows].to(dev, dtype=torch.float32),
        "sequence": split_d["sequence"][rows].to(dev),
        "attention_mask": split_d["attention_mask"][rows].to(dev),
        "seq_product_ids": split_d["seq_product_ids"][rows].to(dev),
        "seq_category_ids": split_d["seq_category_ids"][rows].to(dev),
        "seq_subcategory_ids": split_d["seq_subcategory_ids"][rows].to(dev),
        "seq_business_line_ids": split_d["seq_business_line_ids"][rows].to(dev),
        "seq_purchase_dates": split_d["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_d["prediction_days"][rows].to(dev),
        "candidate_features": split_d["features_bn"][rows].to(dev, dtype=torch.float32),
        "candidate_vectors": v_bn_dev.unsqueeze(0).expand(b, -1, -1),
        "candidate_mask": split_d["allowed_bn"][rows].to(dev),
    }


@torch.no_grad()
def _score_clean_ranker(model: CleanStage5BaseNameRanker, split_d: Mapping[str, Any], v_bn_dev: torch.Tensor, dev: torch.device, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    chunks = []
    for rows in torch.arange(len(split_d["customer_ids"])).split(batch_size):
        inp = _build_clean_batch(split_d, v_bn_dev, rows, dev)
        out = model(**inp, return_aux=False)
        chunks.append(out.logits.cpu())
    return torch.cat(chunks, dim=0)


def run_clean_v4(*, seed: int = SEED, device: Optional[str] = None) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    ref3 = json.loads(REF_3STAGE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(ref3)

    print("[clean-v4] Building native 38D candidate features + 19D fresh customer state + 65D multi-hot ProductTower specs...", flush=True)
    (
        bn_data,
        bn_static,
        universe,
        enrichment,
        vocabs,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        reps,
        _,
    ) = _prepare_clean_v4_data()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_bn = universe["num_base_names"]
    bn_to_members = universe["bn_to_member_pid_indices"]
    tr_bn = bn_data["train"]
    va_bn = bn_data["val"]
    sku_val = sku_data["val"]
    orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]

    aux_60d_sku = build_strict_60d_aux_labels(
        purchases, tr_bn["customer_ids"], tr_bn["prediction_days"].tolist(), product_ids
    )
    aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members).float()
    full_window = tr_bn["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

    attr_ids = {k: v[reps] for k, v in enrichment["product_attr_ids"].items()}
    run_dir = OUT_ROOT / f"clean_v4_fullspec_cust19-seed{seed}"
    tt_dir = run_dir / "two_tower"
    s2_dir = run_dir / "ranker_clean_8h"
    s2_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[clean-v4] Stage 1 Two-Tower: pack-free BGE-M3 + 65D multi-hot [12 container + 8 vol + 11 app + 34 visc] "
        f"+ CustomerTower 19D fresh state...",
        flush=True,
    )
    tt_res = train_stage5_two_tower(
        bn_data=bn_data,
        prod_inputs={
            "text_embedding": bn_static["text_pack_free"].float(),
            "demand_features": bn_static["full_specs_multihot"].float(),
            "attr_ids": attr_ids,
        },
        universe=universe,
        enrichment=enrichment,
        out_dir=tt_dir,
        seed=seed,
        epochs=30,
        device=str(dev),
        num_customer_tabular_features=19,
    )
    v_bn = torch.load(tt_dir / "train_vectors.pt", map_location="cpu", weights_only=False)["v_bn"]
    v_dev = v_bn.to(dev)

    num_sales_branches = len(enrichment["customer_vocabs"]["sales_branch"])

    def _make_ranker(aux: bool) -> CleanStage5BaseNameRanker:
        return CleanStage5BaseNameRanker(
            num_base_names=n_bn,
            num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
            num_customer_features=19,
            num_pairwise_bias_features=13,
            num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
            num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
            num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
            num_sales_branches=num_sales_branches,
            num_categories=len(vocabs["category_vocab"]) + 1,
            num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
            num_business_lines=len(vocabs["business_line_vocab"]) + 1,
            d_model=128,
            num_heads=8,
            memory_ffn_dim=384,
            mlp_widths=(384, 192, 96),
            dropout=0.12,
            aux_60d_enabled=aux,
            seed=seed,
        ).to(dev)

    print("[clean-v4] Stage 2: Fitting CleanStage5BaseNameRanker (zero dead weights, 19D Token-0 Customer State, 38D Cand Features)...", flush=True)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _make_ranker(aux=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[clean-v4] Trainable params: {n_params:,} (zero dead weights!)", flush=True)
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=STAGE2_WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STAGE2_EPOCHS, eta_min=STAGE2_ETA_MIN)
    loss_fn = RankerLoss().to(dev)
    aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    best_ndcg_key = (-1.0, -1, -1.0)
    best_disc_key = (-1, -1.0, -1.0)
    t_s2 = time.perf_counter()

    for ep in range(1, STAGE2_EPOCHS + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses = []
        perm = torch.randperm(len(tr_bn["customer_ids"]), generator=gen)
        for rows in perm.split(64):
            opt.zero_grad(set_to_none=True)
            inp = _build_clean_batch(tr_bn, v_dev, rows, dev)
            cmask = inp["candidate_mask"]
            y = tr_bn["target_bn"][rows].to(dev, dtype=torch.float32) * cmask.float()
            out, a_lg = model(**inp, return_aux=True)
            l_main = loss_fn(out.logits, y, candidate_mask=cmask)["loss"]
            aux_rows = full_window[rows].to(dev)
            loss = l_main
            if aux_rows.any():
                y_aux = aux_60d_bn[rows].to(dev)[aux_rows] * cmask[aux_rows].float()
                l_aux = aux_loss_fn(a_lg[aux_rows], y_aux, candidate_mask=cmask[aux_rows])["loss"]
                loss = loss + 0.3 * l_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, STAGE2_GRAD_CLIP_NORM)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sch.step()

        va_lg = _score_clean_ranker(model, va_bn, v_dev, dev, batch_size=64)
        ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        m_ep = evaluate_base_name_predictions(ranked, va_bn["target_bn"], va_bn["new_target_bn"], va_bn["customer_ids"])
        inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}

        nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
        dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, s2_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, s2_dir / "best_discovery.pt")
        print(
            f"[clean-v4] ep={ep:02d}/{STAGE2_EPOCHS} loss={np.mean(losses):.4f} "
            f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
            f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
            f"({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    # Replay verification + paired bootstrap comparisons + CPU 1-request latency benchmark
    prev_8h_disc_npz = Path("checkpoints/roadmap-v2/stage5_bigger_ranker_v1/modest_8h_ffn384_mlp384-seed42/predictions_discovery.npz")
    prev_mlp384_npz = Path("checkpoints/roadmap-v2/stage5_bigger_ranker_v1/modest_wider_mlp_384-seed42/predictions_ndcg.npz")
    arm1_ndcg_npz = Path("checkpoints/roadmap-v2/stage5_pure_bn_v1/pure_bn_pack_free_multihot_pv-seed42/ranker_8h/predictions_ndcg.npz")
    arm1_disc_npz = Path("checkpoints/roadmap-v2/stage5_pure_bn_v1/pure_bn_pack_free_multihot_pv-seed42/ranker_8h/predictions_discovery.npz")

    s2_res: Dict[str, Any] = {
        "trainable_params": int(n_params),
        "ranker_training_seconds": float(time.perf_counter() - t_s2),
    }
    cpu_bench_ms = {}
    for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
        ck = torch.load(s2_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_ranker(aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_lg = _score_clean_ranker(eval_m, va_bn, v_dev, dev, batch_size=64)
        ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        npz_p = s2_dir / f"predictions_{sel_name}.npz"
        ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
        if ev["bn_metrics"] != ck["metrics"]:
            raise AssertionError(f"clean-v4 {sel_name} replay mismatch!")
        ev["epoch"] = int(ck["epoch"])
        for tag, ref_npz in (
            ("paired_vs_prev_8h_disc", prev_8h_disc_npz),
            ("paired_vs_prev_mlp384", prev_mlp384_npz),
            ("paired_vs_arm1_ndcg", arm1_ndcg_npz),
            ("paired_vs_arm1_disc", arm1_disc_npz),
        ):
            if ref_npz.is_file():
                ev[tag] = paired_base_name_bootstrap_ci(ref_npz, npz_p)
        s2_res[sel_name] = ev

        if sel_name == "ndcg":
            eval_cpu = eval_m.to("cpu").eval()
            v_cpu = v_bn.to("cpu")
            one_inp = _build_clean_batch(va_bn, v_cpu, torch.tensor([0]), torch.device("cpu"))
            with torch.no_grad():
                for _ in range(10):
                    eval_cpu(**one_inp)
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    eval_cpu(**one_inp)
                    ts.append((time.perf_counter() - t0) * 1000.0)
            cpu_bench_ms = {
                "median_ms": float(np.median(ts)),
                "p95_ms": float(np.percentile(ts, 95)),
                "mean_ms": float(np.mean(ts)),
            }

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    _verify_reference_files(ref3)

    results = {
        "experiment": "stage5_clean_v4_fullspec_cust19_cand38",
        "architecture": {
            "stage1_product_tower": {
                "text": "585 pure pack-free base_name BGE-M3 (no packaging/volume in text)",
                "structured_multihot_dim": 65,
                "breakdown": {
                    "container_type": len(CONTAINER_BUCKETS),
                    "volume_tiers": len(VOLUME_BUCKETS),
                    "application_domain": len(APPLICATION_BUCKETS),
                    "viscosity_grade": len(VISCOSITY_BUCKETS),
                },
            },
            "stage1_customer_tower": {
                "customer_tabular_features": 19,
                "breakdown": "8 legacy profile + 11 fresh dynamic cadence/financial/manager state",
            },
            "stage2_ranker": {
                "class": "CleanStage5BaseNameRanker",
                "d_model": 128,
                "num_heads": 8,
                "memory_layers": 1,
                "memory_ffn_dim": 384,
                "mlp_widths": [384, 192, 96],
                "dead_weights_removed": ["product_text_proj", "habit_proj", "base.mlp"],
                "token0_customer_features": 19,
                "candidate_features": 38,
                "active_seq_stats": 6,
                "month_removed": True,
                "trainable_params": int(n_params),
            },
        },
        "candidate_feature_names": list(CANDIDATE_38_FEATURE_NAMES),
        "customer_dynamic_feature_names": list(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
        "seed": int(seed),
        "heldout_access": False,
        "two_tower": tt_res,
        "ranker": s2_res,
        "cpu_single_request_ranker_ms": cpu_bench_ms,
        "preserved_hashes_verified": True,
        "elapsed_seconds": float(time.perf_counter() - t_all),
    }
    atomic_json(results, SUMMARY_PATH, overwrite=True)
    del model
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    res = run_clean_v4(seed=args.seed, device=args.device)
    print(f"\nSaved clean v4 summary to {SUMMARY_PATH} ({res['elapsed_seconds']:.1f}s)", flush=True)
    print(f"CPU 1-req CleanStage5BaseNameRanker latency: {res['cpu_single_request_ranker_ms']}", flush=True)


if __name__ == "__main__":
    main()
