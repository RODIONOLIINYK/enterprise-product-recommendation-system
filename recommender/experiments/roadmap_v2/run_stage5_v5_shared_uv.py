"""Validation-only Stage 5 v5 (`v5_shared_uv`) experiment over saved v4b_fullspec65_nosb Stage 1 u_bn & v_bn.

Architecture (`SharedUVStage5Ranker`, modest_8h_ffn384_mlp384 topology, zero scratch ID/text tables):
  - Token 0 (Customer): residual MLP over [Stage 1 u_bn (128), current dynamic cust_11 (11)] -> 128D
  - Tokens 1..32 (History): direct precomputed Stage 1 v_bn[seq_product_ids] (128D) + pos_emb + type_emb[1] (no extra MLP!)
  - Candidate Queries Q: residual MLP over [Stage 1 v_bn (128), cand_38 (38)] -> 128D (injects dynamic candidate context before cross-attention!)
  - Scores ALL 585 eligible formulations directly (zero Stage 1 MLP, zero top-50/top-100 truncation).
  - Strictly Train (29,095 < 2025-10-01) -> Validation (6,480 2025-10-01..2026-02-18), ZERO held-out Test access!
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from recommender.artifacts import atomic_json
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TRAIN_CUTOFF
from recommender.train_retriever import get_device

from .augmentation import build_strict_60d_aux_labels
from .clean_ranker import SharedUVStage5Ranker
from .config import (
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_WEIGHT_DECAY,
)
from .native_base_name import CANDIDATE_38_FEATURE_NAMES, CUSTOMER_DYNAMIC_11_FEATURE_NAMES
from .run_stage5 import _collapse_bool_matrix_to_bn, _verify_reference_files
from .run_stage5_ablations import _evaluate_with_cohorts_and_sku
from .run_stage5_clean_v4b import _prepare_clean_v4b_data
from .stage5_base_entity import evaluate_base_name_predictions, paired_base_name_bootstrap_ci


OUT_ROOT = ARTIFACT_ROOT / "stage5_v5_shared_uv"
SUMMARY_PATH = EVIDENCE_ROOT / "stage5_v5_shared_uv_seed42.json"
REF_PATH = EVIDENCE_ROOT / "stage5_reference.json"
REF_3STAGE_PATH = EVIDENCE_ROOT / "stage5_3stage_reference.json"
V4B_TT_DIR = ARTIFACT_ROOT / "stage5_clean_v4b" / "v4b_fullspec65_nosb-seed42" / "two_tower"


def _build_v5_batch(
    split_d: Mapping[str, Any],
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    return {
        "customer_vector": u_bn_split[rows].to(dev, dtype=torch.float32),
        "customer_features": split_d["tabular_features"][rows].to(dev, dtype=torch.float32),
        "catalog_product_vectors": v_bn_dev,
        "sequence": split_d["sequence"][rows].to(dev),
        "attention_mask": split_d["attention_mask"][rows].to(dev),
        "seq_product_ids": split_d["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_d["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_d["prediction_days"][rows].to(dev),
        "candidate_features": split_d["features_bn"][rows].to(dev, dtype=torch.float32),
        "candidate_mask": split_d["allowed_bn"][rows].to(dev),
    }


@torch.no_grad()
def _score_v5(
    model: SharedUVStage5Ranker,
    split_d: Mapping[str, Any],
    u_bn_split: torch.Tensor,
    v_bn_dev: torch.Tensor,
    dev: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for rows in torch.arange(len(split_d["customer_ids"])).split(batch_size):
        out = model(**_build_v5_batch(split_d, u_bn_split, v_bn_dev, rows, dev), return_aux=False)
        chunks.append(out.logits.cpu())
    return torch.cat(chunks, dim=0)


def run_v5_shared_uv(*, seed: int = SEED, device: Optional[str] = None) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    ref_active = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ref3 = json.loads(REF_3STAGE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(ref_active)
    _verify_reference_files(ref3)

    print("[v5-shared-uv] Preparing native 38D candidate + 11D current customer features & loading frozen v4b u_bn/v_bn...", flush=True)
    (
        bn_data,
        _,
        universe,
        _,
        _,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        _,
    ) = _prepare_clean_v4b_data()

    tr_vec = torch.load(V4B_TT_DIR / "train_vectors.pt", map_location="cpu", weights_only=False)
    va_vec = torch.load(V4B_TT_DIR / "val_vectors.pt", map_location="cpu", weights_only=False)
    v_bn = tr_vec["v_bn"].float()
    u_tr = tr_vec["u_bn"].float()
    u_va = va_vec["u_bn"].float()
    v_dev = v_bn.to(dev)

    tr_bn = bn_data["train"]
    va_bn = bn_data["val"]
    sku_val = sku_data["val"]
    bn_to_members = universe["bn_to_member_pid_indices"]

    aux_60d_sku = build_strict_60d_aux_labels(
        purchases, tr_bn["customer_ids"], tr_bn["prediction_days"].tolist(), product_ids
    )
    aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members).float()
    full_window = tr_bn["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

    s2_dir = OUT_ROOT / f"v5_shared_uv-seed{seed}"
    s2_dir.mkdir(parents=True, exist_ok=True)

    def _make_ranker(aux: bool) -> SharedUVStage5Ranker:
        return SharedUVStage5Ranker(
            num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
            num_customer_features=len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
            num_pairwise_bias_features=13,
            d_model=128,
            num_heads=8,
            memory_ffn_dim=384,
            mlp_widths=(384, 192, 96),
            dropout=0.12,
            aux_60d_enabled=aux,
            seed=seed,
        ).to(dev)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _make_ranker(aux=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[v5-shared-uv] Fitting SharedUVStage5Ranker ({n_params:,} params; zero scratch text/ID tables)...", flush=True)

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
            inp = _build_v5_batch(tr_bn, u_tr, v_dev, rows, dev)
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

        va_lg = _score_v5(model, va_bn, u_va, v_dev, dev, batch_size=64)
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
            f"[v5-shared-uv] ep={ep:02d}/{STAGE2_EPOCHS} loss={np.mean(losses):.4f} "
            f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
            f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
            f"({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    v4b_ndcg_npz = Path("checkpoints/roadmap-v2/stage5_clean_v4b/v4b_fullspec65_nosb-seed42/ranker_clean_8h/predictions_ndcg.npz")
    v4b_disc_npz = Path("checkpoints/roadmap-v2/stage5_clean_v4b/v4b_fullspec65_nosb-seed42/ranker_clean_8h/predictions_discovery.npz")

    s2_res: Dict[str, Any] = {
        "trainable_params": int(n_params),
        "ranker_training_seconds": float(time.perf_counter() - t_s2),
    }
    cpu_bench_ms = {}
    for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
        ck = torch.load(s2_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_ranker(aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_lg = _score_v5(eval_m, va_bn, u_va, v_dev, dev, batch_size=64)
        ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        npz_p = s2_dir / f"predictions_{sel_name}.npz"
        ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
        if ev["bn_metrics"] != ck["metrics"]:
            raise AssertionError(f"v5_shared_uv {sel_name} replay mismatch!")
        ev["epoch"] = int(ck["epoch"])
        for tag, ref_npz in (("paired_vs_v4b_ndcg", v4b_ndcg_npz), ("paired_vs_v4b_disc", v4b_disc_npz)):
            if ref_npz.is_file():
                ev[tag] = paired_base_name_bootstrap_ci(ref_npz, npz_p)
        s2_res[sel_name] = ev

        if sel_name == "ndcg":
            eval_cpu = eval_m.to("cpu").eval()
            one_inp = _build_v5_batch(va_bn, u_va, v_bn, torch.tensor([0]), torch.device("cpu"))
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
    _verify_reference_files(ref_active)
    _verify_reference_files(ref3)

    results = {
        "experiment": "stage5_v5_shared_uv_cand_mlp",
        "architecture": {
            "class": "SharedUVStage5Ranker",
            "token0": "LayerNorm(u_bn + MLP([u_bn (128), cust_11 (11)])) + type_emb[0]",
            "tokens_1_32": "Direct Stage-1 v_bn[seq_product_ids] (128) + pos_emb + type_emb[1] (no extra MLP)",
            "candidate_queries": "LayerNorm(v_bn + MLP([v_bn (128), cand_38 (38)]))",
            "d_model": 128,
            "num_heads": 8,
            "memory_ffn_dim": 384,
            "mlp_widths": [384, 192, 96],
            "trainable_params": int(n_params),
        },
        "two_tower_source": str(V4B_TT_DIR / "best.pt"),
        "seed": int(seed),
        "heldout_access": False,
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
    res = run_v5_shared_uv(seed=args.seed, device=args.device)
    print(f"\nSaved v5_shared_uv summary to {SUMMARY_PATH} ({res['elapsed_seconds']:.1f}s)", flush=True)
    print(f"CPU 1-req SharedUVStage5Ranker latency: {res['cpu_single_request_ranker_ms']}", flush=True)


if __name__ == "__main__":
    main()
