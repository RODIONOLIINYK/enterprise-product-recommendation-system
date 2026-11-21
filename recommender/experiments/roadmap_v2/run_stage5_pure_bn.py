"""Validation-only comparison on modest_8h_ffn384_mlp384 with ALL pure-base_name & no-month upgrades:
  1. Native 585-base_name feature engine (53 features: month_sin & month_cos deleted; cat/subcat/brand & parsed_specs
     counted on collapsed base_name events; zero 841-SKU intermediate arrays).
  2. `history_month` excluded from Stage 2 sequence attention statistics.
  3. Head-to-head ProductTower representation test (refitting Two-Tower + modest_8h_ffn384_mlp384 per arm):
       - Arm A (`pure_bn_pack_free_multihot_pv`): Pure pack-free BGE-M3 text (packaging & volume excluded from text)
         + 20D structured multi-hot [12D container_type + 8D volume tiers] in ProductTower!
       - Arm B (`pure_bn_pack_free`): Pure pack-free BGE-M3 text + zero container/volume features (12D zeros).
       - Arm C (`pure_bn_all_packs_multihot_pv`): BGE-M3 text incl. all containers/volumes + 20D multi-hot [container + volume].
"""

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
from recommender.experiments.static_enrichment.config import TRANSFERRED_CUSTOMER_STAT_NAMES
from recommender.experiments.static_enrichment.data import quantize_transferred_stats
from recommender.experiments.static_enrichment.metadata import encode_categorical_values, load_enriched_companies
from recommender.familiarity import ATTRIBUTE_COLUMNS, FamiliarityStore
from recommender.features import LEGACY_PROFILE_FEATURES, PointInTimeNormalizer, compute_point_in_time_features
from recommender.ranker import RankerLoss
from recommender.scope import get_datasets_locs
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TEST_START, TRAIN_CUTOFF
from recommender.train_ranker import company_embeddings_path
from recommender.train_retriever import build_customer_metadata, get_device
from recommender.text import load_cached_embeddings

from .augmentation import build_strict_60d_aux_labels
from .commercial_features import COMMERCIAL_PARQUET_PATH
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_LR,
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
    NATIVE_53_FEATURE_NAMES,
    NativeBaseNameFeatureEngine,
    build_base_name_static_representations,
)
from .run_stage5 import _collapse_bool_matrix_to_bn, _verify_reference_files, train_stage5_two_tower
from .run_stage5_ablations import _build_fullpool_batch, _evaluate_with_cohorts_and_sku, _score_fullpool_ranker
from .run_stage5_bigger_ranker import MODEST_VARIANTS, BiggerStage5BaseNameRanker
from .stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
)


OUT_ROOT = ARTIFACT_ROOT / "stage5_pure_bn_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "stage5_pure_bn_seed42.json"
REF_3STAGE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"


def _prepare_native_dated_baskets():
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
    cases, counts = basket_cases(purchases, product_ids, universe, pool)

    splits, sku_data = {}, {}
    for split, rows in cases.items():
        cids, dates, first, targets, new, allowed, unseen, sku_targets, sku_allowed = zip(*rows, strict=True)
        days = [(dt - PURCHASE_DATE_ORIGIN).days for dt in dates]
        sequences = build_base_name_sequences(
            purchases, cids, days, universe, representatives,
            vocabs["category_vocab"], vocabs["subcategory_vocab"], vocabs["business_line_vocab"],
        )
        # Zero out history_month (cols 3:5) in sequence so neither Two-Tower nor Stage 2 sees calendar month!
        sequences["sequence"][..., 3:5] = 0.0

        features = np.empty((len(rows), len(reps), len(NATIVE_53_FEATURE_NAMES)), np.float16)
        profile8, profile6 = [], []
        t_feat = time.perf_counter()
        for i, (cid, date) in enumerate(zip(cids, dates, strict=True)):
            features[i] = engine.build(cid, date)
            stats = compute_point_in_time_features(
                histories[cid], cid, date, normalizer=normalizer, include_median_purchase_quantity=True
            )
            profile8.append([stats[k] for k in LEGACY_PROFILE_FEATURES])
            profile6.append([stats[k] for k in TRANSFERRED_CUSTOMER_STAT_NAMES])
            if (i + 1) % 5000 == 0:
                print(f"[native-53] {split} {i + 1}/{len(rows)} ({time.perf_counter() - t_feat:.1f}s)", flush=True)

        splits[split] = {
            "customer_ids": list(cids),
            "prediction_days": torch.tensor(days, dtype=torch.float32),
            "first_observed": torch.tensor(first, dtype=torch.bool),
            "target_bn": torch.from_numpy(np.stack(targets)),
            "new_target_bn": torch.from_numpy(np.stack(new)),
            "allowed_bn": torch.from_numpy(np.stack(allowed)),
            "unseen_bn": torch.from_numpy(np.stack(unseen)),
            "features_bn": torch.from_numpy(features),
            "feature_names": NATIVE_53_FEATURE_NAMES,
            "familiarity_bn": torch.cat(
                [familiarity.batch_features(cids[i : i + 256], dates[i : i + 256]) for i in range(0, len(rows), 256)]
            ),
            "text_embedding": torch.from_numpy(
                np.stack([texts.get(cid, np.zeros(1024, np.float32)) for cid in cids])
            ),
            "tabular_features": torch.tensor(profile8, dtype=torch.float32),
            "customer_stats_6": quantize_transferred_stats(np.asarray(profile6, np.float32)),
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
    return splits, bn_static, universe, enrichment, vocabs, purchases, catalogue, product_ids, sku_data, reps


def run_pure_bn_experiment(*, seed: int = SEED, device: Optional[str] = None) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    ref3 = json.loads(REF_3STAGE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(ref3)

    print("[pure-bn] Building native 585-base_name 53D features (no month_sin/cos) + pure BGE-M3 texts...", flush=True)
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
    ) = _prepare_native_dated_baskets()

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

    cfg = MODEST_VARIANTS["modest_8h_ffn384_mlp384"]
    prev_8h_disc_npz = Path("checkpoints/roadmap-v2/stage5_bigger_ranker_v1/modest_8h_ffn384_mlp384-seed42/predictions_discovery.npz")
    prev_mlp384_npz = Path("checkpoints/roadmap-v2/stage5_bigger_ranker_v1/modest_wider_mlp_384-seed42/predictions_ndcg.npz")

    attr_ids = {k: v[reps] for k, v in enrichment["product_attr_ids"].items()}
    arms = {
        "pure_bn_pack_free_multihot_pv": {
            "text_embedding": bn_static["text_pack_free"].float(),
            "demand_features": bn_static["packaging_volume_multihot"].float(),
            "attr_ids": attr_ids,
            "description": "585 pure pack-free base_name BGE-M3 texts (no packaging/volume in text) + 20D multi-hot [12D container + 8D volume] in ProductTower + 53 native base_name features (no month)",
        },
        "pure_bn_pack_free": {
            "text_embedding": bn_static["text_pack_free"].float(),
            "demand_features": torch.zeros((n_bn, 12), dtype=torch.float32),
            "attr_ids": attr_ids,
            "description": "585 pure pack-free base_name BGE-M3 texts + zero container/volume + 53 native base_name features (no month)",
        },
        "pure_bn_all_packs_multihot_pv": {
            "text_embedding": bn_static["text_all_packs"].float(),
            "demand_features": bn_static["packaging_volume_multihot"].float(),
            "attr_ids": attr_ids,
            "description": "585 base_name BGE-M3 texts incl. all available containers/volumes + 20D multi-hot [12D container + 8D volume] in ProductTower + 53 native base_name features (no month)",
        },
    }

    results: Dict[str, Any] = {
        "experiment": "stage5_pure_bn_nomonth_modest_8h",
        "ranker_architecture": {"variant": "modest_8h_ffn384_mlp384", **cfg},
        "num_features": len(NATIVE_53_FEATURE_NAMES),
        "feature_names": list(NATIVE_53_FEATURE_NAMES),
        "month_features_removed": True,
        "history_month_excluded": True,
        "seed": int(seed),
        "heldout_access": False,
        "arms": {},
    }

    for arm_name, p_in in arms.items():
        arm_dir = OUT_ROOT / f"{arm_name}-seed{seed}"
        tt_dir = arm_dir / "two_tower"
        s2_dir = arm_dir / "ranker_8h"
        s2_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[pure-bn/{arm_name}] Stage 1: Fitting Two-Tower over 585 distinct base_names...", flush=True)
        tt_res = train_stage5_two_tower(
            bn_data=bn_data,
            prod_inputs={
                "text_embedding": p_in["text_embedding"],
                "demand_features": p_in["demand_features"],
                "attr_ids": p_in["attr_ids"],
            },
            universe=universe,
            enrichment=enrichment,
            out_dir=tt_dir,
            seed=seed,
            epochs=30,
            device=str(dev),
        )
        v_bn = torch.load(tt_dir / "train_vectors.pt", map_location="cpu", weights_only=False)["v_bn"]
        v_dev = v_bn.to(dev)

        def _make_ranker(aux: bool) -> BiggerStage5BaseNameRanker:
            m = BiggerStage5BaseNameRanker(
                num_base_names=n_bn,
                num_features=len(NATIVE_53_FEATURE_NAMES),
                num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
                num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
                num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
                num_categories=len(vocabs["category_vocab"]) + 1,
                num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
                num_business_lines=len(vocabs["business_line_vocab"]) + 1,
                d_model=cfg["d_model"],
                num_heads=cfg["num_heads"],
                memory_layers=cfg["memory_layers"],
                memory_ffn_mult=cfg["memory_ffn_mult"],
                memory_ffn_dim=cfg["memory_ffn_dim"],
                cross_attn_layers=cfg["cross_attn_layers"],
                mlp_widths=cfg["mlp_widths"],
                dropout=cfg["dropout"],
                aux_60d_enabled=aux,
                seed=seed,
            ).to(dev)
            m.base.excluded_statistics = ("history_repeat", "history_month")
            return m

        print(f"[pure-bn/{arm_name}] Stage 2: Fitting No-MLP Full-Pool modest_8h_ffn384_mlp384 (53D, no month)...", flush=True)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = _make_ranker(aux=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=STAGE2_WEIGHT_DECAY)
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
            for rows in perm.split(cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                inp = _build_fullpool_batch(tr_bn, v_dev, rows, dev)
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

            va_lg = _score_fullpool_ranker(model, va_bn, v_dev, dev, batch_size=64)
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
                f"[pure-bn/{arm_name}] ep={ep:02d}/{STAGE2_EPOCHS} loss={np.mean(losses):.4f} "
                f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
                f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
                f"({time.perf_counter() - ep_t0:.1f}s)",
                flush=True,
            )

        arm_res: Dict[str, Any] = {
            "description": p_in["description"],
            "two_tower": tt_res,
            "trainable_params": int(n_params),
            "ranker_training_seconds": float(time.perf_counter() - t_s2),
        }
        for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
            ck = torch.load(s2_dir / fname, map_location="cpu", weights_only=False)
            eval_m = _make_ranker(aux=False)
            eval_m.load_state_dict(ck["model_state_dict"], strict=True)
            va_lg = _score_fullpool_ranker(eval_m, va_bn, v_dev, dev, batch_size=64)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            npz_p = s2_dir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"{arm_name} {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            if prev_8h_disc_npz.is_file():
                ev["paired_vs_prev_8h_disc"] = paired_base_name_bootstrap_ci(prev_8h_disc_npz, npz_p)
            if prev_mlp384_npz.is_file():
                ev["paired_vs_prev_mlp384"] = paired_base_name_bootstrap_ci(prev_mlp384_npz, npz_p)
            arm_res[sel_name] = ev

        results["arms"][arm_name] = arm_res
        del model
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    _verify_reference_files(ref3)
    results["preserved_hashes_verified"] = True
    results["elapsed_seconds"] = float(time.perf_counter() - t_all)
    atomic_json(results, SUMMARY_PATH, overwrite=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    res = run_pure_bn_experiment(seed=args.seed, device=args.device)
    print(f"\nSaved pure-base_name experiment summary to {SUMMARY_PATH} ({res['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
