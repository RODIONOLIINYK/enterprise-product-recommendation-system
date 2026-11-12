"""Static enrichment campaign runner.

Run sequentially on one MPS device at a time:
  PYTHONPATH=. ./venv/bin/python experiments/static_enrichment/run.py --phase all
"""

import argparse
import json
from pathlib import Path
import torch

from recommender.artifacts import atomic_json
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import make_slates
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from recommender.experiments.static_enrichment.config import (
    ARTIFACT_ROOT,
    CLEAN_V2_ROOT,
    EVIDENCE_ROOT,
    build_clean_v2_variant_specs,
    build_variant_specs,
)
from recommender.experiments.static_enrichment.data import prepare_static_enrichment_bundle
from recommender.experiments.static_enrichment.train import (
    _load_stage1_campaign_data,
    fit_compact_stage1_selector,
    measure_v3_warm_latency,
    run_clean_v2_downstream,
    run_downstream_for_stage1,
    run_v3_stage2_ranker,
    slate_coverage_summary,
    train_stage1_variant,
)
from recommender.train_retriever import get_device


def _winner(results: dict, names: tuple[str, ...]) -> str:
    return max(
        names,
        key=lambda n: (
            results[n]["metrics"]["new_hits@25"],
            results[n]["metrics"]["new_hits@50"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "prep", "c0", "screen", "repeat", "downstream", "clean_v2", "stage1_mlp_v3"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--ranker-epochs", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = get_device(args.device)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    enrichment = prepare_static_enrichment_bundle(device=str(device))
    if args.phase == "prep":
        print("Prepared static enrichment bundle and text caches.", flush=True)
        return

    train, val, info, train_source, val_source = prepare_retrieval_data()
    if args.phase == "clean_v2":
        CLEAN_V2_ROOT.mkdir(parents=True, exist_ok=True)
        v2_path = EVIDENCE_ROOT / "clean_v2_results.json"
        if v2_path.is_file():
            v2_results = json.loads(v2_path.read_text(encoding="utf-8"))
        else:
            v2_results = {"stage1": {}, "downstream": {}, "decisions": {}}
        v2_specs = build_clean_v2_variant_specs()
        seed = 42
        for vname in ("C0_clean", "P_t1_fam_clean", "U0_t1_fam_clean", "U0_t1_fam_nosubcat"):
            res = train_stage1_variant(
                v2_specs[vname],
                seed,
                train,
                val,
                info,
                train_source,
                val_source,
                enrichment,
                CLEAN_V2_ROOT,
                epochs=args.stage1_epochs,
                device=str(device),
                save_train_cache=True,
            )
            v2_results["stage1"][f"{vname}-seed{seed}"] = res
            atomic_json(v2_results, v2_path)

        clean_candidates = ("P_t1_fam_clean", "U0_t1_fam_clean", "U0_t1_fam_nosubcat")
        best_clean = _winner(
            {n: v2_results["stage1"][f"{n}-seed{seed}"] for n in clean_candidates},
            clean_candidates,
        )
        v2_results["decisions"] = {"best_clean": best_clean, "seed": seed}
        atomic_json(v2_results, v2_path)

        ds_runs = [
            ("C0_clean", v2_specs["C0_clean"], "dot100_41"),
            (best_clean, v2_specs[best_clean], "dot100_41"),
            (best_clean, v2_specs[best_clean], "dot100_47"),
            (best_clean, v2_specs[best_clean], "precomputed_mlp_47"),
        ]
        ds_res = run_clean_v2_downstream(
            ds_runs,
            seed,
            train,
            val,
            info,
            train_source,
            val_source,
            enrichment,
            CLEAN_V2_ROOT,
            epochs=args.ranker_epochs,
            device=str(device),
        )
        for cfg_key, res in ds_res.items():
            v2_results["downstream"][f"{cfg_key}-seed{seed}"] = res
        atomic_json(v2_results, v2_path)
        print(f"Clean v2 campaign complete (best_clean={best_clean}).", flush=True)
        return

    if args.phase == "stage1_mlp_v3":
        v3_root = ARTIFACT_ROOT / "stage1_mlp_v3"
        v3_root.mkdir(parents=True, exist_ok=True)
        v3_path = EVIDENCE_ROOT / "stage1_mlp_v3_results.json"
        if v3_path.is_file():
            v3_results = json.loads(v3_path.read_text(encoding="utf-8"))
        else:
            v3_results = {"stage1": {}, "dot_baseline": {}, "selectors": {}, "downstream": {}, "latency": {}}

        v2_specs = build_clean_v2_variant_specs()
        seed = 42
        stage1_dirs = {}
        for vname in ("C0_clean", "P_t1_fam_clean", "U0_t1_fam_clean"):
            res = train_stage1_variant(
                v2_specs[vname],
                seed,
                train,
                val,
                info,
                train_source,
                val_source,
                enrichment,
                CLEAN_V2_ROOT,
                epochs=args.stage1_epochs,
                device=str(device),
                save_train_cache=True,
            )
            v3_results["stage1"][vname] = res
            s1_dir = CLEAN_V2_ROOT / "runs" / f"stage1-{vname}-seed{seed}"
            stage1_dirs[vname] = s1_dir
            cdata, _, _, _ = _load_stage1_campaign_data(
                v2_specs[vname], s1_dir, train, val, info, train_source, val_source, enrichment, device
            )
            va = cdata["val"]
            dot_pred = v3_root / f"dot_baseline_{vname}_val_predictions.npz"
            dot_metrics = ranking_metrics(va["dot_scores"], va, None, dot_pred)
            dot_slate = make_slates(va, va["dot_scores"], info["product_ids"], "quota100_50")
            dot_cov = slate_coverage_summary(dot_slate["candidate_indices"], va["target"], va["unseen"])
            v3_results["dot_baseline"][vname] = {
                "full_pool": dot_metrics,
                "quota100_50_coverage": dot_cov,
                "predictions": str(dot_pred),
            }
            atomic_json(v3_results, v3_path)

        u0_spec = v2_specs["U0_t1_fam_clean"]
        u0_dir = stage1_dirs["U0_t1_fam_clean"]
        c0_ckpt = stage1_dirs["C0_clean"] / "best.pt"
        c0_dir = stage1_dirs["C0_clean"]
        p_spec = v2_specs["P_t1_fam_clean"]
        p_dir = stage1_dirs["P_t1_fam_clean"]

        sel_results = {}
        for recipe in ("uv_47", "uv_47_ref"):
            sel_res = fit_compact_stage1_selector(
                "U0_t1_fam_clean",
                u0_spec,
                recipe,
                seed,
                train,
                val,
                info,
                train_source,
                val_source,
                enrichment,
                u0_dir,
                v3_root,
                epochs=args.ranker_epochs,
                device=str(device),
            )
            sel_results[recipe] = sel_res
            v3_results["selectors"][f"U0_t1_fam_clean-{recipe}"] = sel_res
            atomic_json(v3_results, v3_path)

        ranker_specs = [
            ("U0-dot100_41", "U0_t1_fam_clean", u0_spec, u0_dir, "dot", None, "stage1_copied", None, None, None),
            (
                "U0-mlp303_100_41",
                "U0_t1_fam_clean",
                u0_spec,
                u0_dir,
                "uv_47",
                Path(sel_results["uv_47"]["selected_checkpoint"]).parent,
                "stage1_copied",
                None,
                None,
                None,
            ),
            (
                "U0-mlp304_100_41",
                "U0_t1_fam_clean",
                u0_spec,
                u0_dir,
                "uv_47_ref",
                Path(sel_results["uv_47_ref"]["selected_checkpoint"]).parent,
                "stage1_copied",
                None,
                None,
                None,
            ),
            ("P-dot100_41", "P_t1_fam_clean", p_spec, p_dir, "dot", None, "stage1_copied", None, None, None),
            ("P-dot100_41-init_c0", "P_t1_fam_clean", p_spec, p_dir, "dot", None, "common_checkpoint", c0_ckpt, None, None),
            (
                "P-c0_slate_init_c0-41",
                "P_t1_fam_clean",
                p_spec,
                p_dir,
                "dot",
                None,
                "common_checkpoint",
                c0_ckpt,
                c0_dir,
                v2_specs["C0_clean"],
            ),
        ]
        u0_rankers = {}
        for cfg_name, vname, sp, s1d, sel_t, sel_d, init_m, init_c, sl_over, sl_sp in ranker_specs:
            res = run_v3_stage2_ranker(
                f"ranker-{cfg_name}-seed{seed}",
                vname,
                sp,
                s1d,
                v3_root,
                seed,
                train,
                val,
                info,
                train_source,
                val_source,
                enrichment,
                selector_type=sel_t,
                selector_dir=sel_d,
                init_mode=init_m,
                init_checkpoint_path=init_c,
                slate_override_stage1_dir=sl_over,
                slate_override_spec=sl_sp,
                epochs=args.ranker_epochs,
                device=str(device),
            )
            v3_results["downstream"][cfg_name] = res
            if cfg_name in ("U0-dot100_41", "U0-mlp303_100_41", "U0-mlp304_100_41"):
                u0_rankers[cfg_name] = res
            atomic_json(v3_results, v3_path)

        v3_results["latency"] = measure_v3_warm_latency(
            u0_spec,
            u0_dir,
            u0_rankers,
            sel_results,
            val,
            info,
            val_source,
            enrichment,
        )
        atomic_json(v3_results, v3_path)
        print("Stage 1 MLP v3 campaign complete.", flush=True)
        return

    results_path = EVIDENCE_ROOT / "raw_results.json"
    if results_path.is_file():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        all_results = {"stage1": {}, "downstream": {}, "decisions": {}}

    def run_s1(spec, seed: int, *, save_train_cache: bool = False):
        key = f"{spec.name}-seed{seed}"
        res = train_stage1_variant(
            spec,
            seed,
            train,
            val,
            info,
            train_source,
            val_source,
            enrichment,
            ARTIFACT_ROOT,
            epochs=args.stage1_epochs,
            device=str(device),
            save_train_cache=save_train_cache,
        )
        all_results["stage1"][key] = res
        atomic_json(all_results, results_path)
        return res

    base_specs = build_variant_specs()

    if args.phase in ("all", "c0"):
        c0_42 = run_s1(base_specs["C0"], 42, save_train_cache=True)
        print(
            f"C0 seed=42 selected epoch={c0_42['selected_epoch']} "
            f"hits@25={c0_42['metrics']['new_hits@25']} hits@50={c0_42['metrics']['new_hits@50']}",
            flush=True,
        )
        if args.phase == "c0":
            return

    if args.phase in ("all", "screen"):
        run_s1(base_specs["C0"], 42, save_train_cache=True)
        t1 = run_s1(base_specs["T1"], 42)
        t2 = run_s1(base_specs["T2"], 42)
        best_t = _winner({"T1": t1, "T2": t2}, ("T1", "T2"))
        specs = build_variant_specs(best_t=best_t)

        a_res = run_s1(specs["A"], 42)
        f_emb = run_s1(specs["F_emb"], 42)
        f_txt = run_s1(specs["F_txt"], 42)
        best_f = _winner({"F_emb": f_emb, "F_txt": f_txt}, ("F_emb", "F_txt"))
        specs = build_variant_specs(best_t=best_t, best_f=best_f)

        p_res = run_s1(specs["P"], 42)
        p_nu = run_s1(specs["P_no_unit"], 42)
        best_p = _winner({"P": p_res, "P_no_unit": p_nu}, ("P", "P_no_unit"))
        specs = build_variant_specs(best_t=best_t, best_f=best_f, best_p=best_p)

        u0 = run_s1(specs["U0"], 42)
        u1 = run_s1(specs["U1"], 42)

        screen_pool = {
            "C0": all_results["stage1"]["C0-seed42"],
            "T1": t1,
            "T2": t2,
            "A": a_res,
            "F_emb": f_emb,
            "F_txt": f_txt,
            "P": p_res,
            "P_no_unit": p_nu,
            "U0": u0,
            "U1": u1,
        }
        best_screen_overall = _winner(screen_pool, tuple(screen_pool.keys()))
        best_screen_product = _winner(
            screen_pool, ("T1", "T2", "A", "F_emb", "F_txt", "P", "P_no_unit")
        )
        all_results["decisions"] = {
            "best_t": best_t,
            "best_f": best_f,
            "best_p": best_p,
            "best_screen_product": best_screen_product,
            "best_screen_overall": best_screen_overall,
        }
        atomic_json(all_results, results_path)
        atomic_json(all_results["decisions"], EVIDENCE_ROOT / "decisions.json")
        if args.phase == "screen":
            return

    dec = all_results["decisions"]
    specs = build_variant_specs(
        best_t=dec["best_t"],
        best_f=dec["best_f"],
        best_p=dec["best_p"],
    )

    if args.phase in ("all", "repeat"):
        repeat_variants = ["C0", "U0", "U1"]
        for cand in (dec["best_screen_product"], dec["best_screen_overall"]):
            if cand not in repeat_variants and cand != "C0":
                repeat_variants.append(cand)
        for vname in repeat_variants:
            for seed in (42, 17, 2026):
                run_s1(specs[vname], seed, save_train_cache=(vname in ("C0", dec["best_screen_overall"], "U0", "U1")))
        # Pick 3-seed winner among non-C0 repeated variants by mean (new_hits@25, new_hits@50)
        non_c0 = [v for v in repeat_variants if v != "C0"]
        best_3seed = max(
            non_c0,
            key=lambda v: (
                sum(all_results["stage1"][f"{v}-seed{s}"]["metrics"]["new_hits@25"] for s in (42, 17, 2026)) / 3.0,
                sum(all_results["stage1"][f"{v}-seed{s}"]["metrics"]["new_hits@50"] for s in (42, 17, 2026)) / 3.0,
            ),
        )
        all_results["decisions"]["repeated_variants"] = repeat_variants
        all_results["decisions"]["best_stage1_3seed"] = best_3seed
        atomic_json(all_results, results_path)
        atomic_json(all_results["decisions"], EVIDENCE_ROOT / "decisions.json")
        if args.phase == "repeat":
            return

    if args.phase in ("all", "downstream"):
        best_stage1 = all_results["decisions"]["best_stage1_3seed"]
        downstream_targets = ["C0"] if best_stage1 == "C0" else ["C0", best_stage1]
        for vname in downstream_targets:
            for seed in (42, 17, 2026):
                res = run_downstream_for_stage1(
                    vname,
                    specs[vname],
                    seed,
                    train,
                    val,
                    info,
                    train_source,
                    val_source,
                    enrichment,
                    ARTIFACT_ROOT,
                    modes=("control", "dynamic_features"),
                    epochs=args.ranker_epochs,
                    device=str(device),
                )
                for mode, r in res.items():
                    all_results["downstream"][f"{vname}-dot100-{mode}-seed{seed}"] = r
                    atomic_json(all_results, results_path)

    print("Static enrichment campaign runs complete.", flush=True)


if __name__ == "__main__":
    main()
