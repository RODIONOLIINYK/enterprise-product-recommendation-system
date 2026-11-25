"""Run Wave 1 Stage 2 experiments on `v2_full` (validation-only, zero held-out Test access):

  Pretrained Stage 1 (`v5_shared_uv`) arms:
    - Arm 1 (`arm1_2stage_unseen_attn_aux`): Frozen Stage 1 `u_bn`/`v_bn` + training-only
      unseen-discovery attention aux head `Head_attn([v_bn, h_att, v_bn * h_att])` (`beta = 0.25`,
      ZERO `cand_38` shortcuts, discarded at inference).
    - Arm 2 (`arm2_2stage_unfreeze_vbn_unseen_attn`): Frozen Stage 1 `u_bn` + unfrozen pretrained
      Stage 1 `ProductTower` (`v_bn`) fine-tuned end-to-end inside Stage 2 (Ep 1 frozen warmup,
      Ep 2..10 `LR = 1e-4`) + unseen-discovery attention aux head (`beta = 0.25`).

  Single-Stage End-to-End (trained ONLY during Stage 2 — zero separate Stage 1 pretraining!) arms:
    - Arm 3 (`arm3_e2e_stage2_only`): Live `ProductTower` + live Token 0 customer profile trained
      from scratch ONLY in Stage 2 (`L_next + 0.3 * L_60d`; cached `v_bn` at inference -> 0 ms).
    - Arm 4 (`arm4_e2e_unseen_attn_aux`): Arm 3 + unseen-discovery attention aux head (`beta = 0.25`).

  (Historical Arm 5 `qty_time` attention bias was rejected after Wave 1 regression and removed.)

Uses fast block-shuffled sequential NVMe memmap reading (< 2.5 GB RAM) and compares every arm
against `baseline_2stage_v5` with paired 95% customer-clustered bootstrap CIs over 16,623 val baskets.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.roadmap_v2.clean_ranker import SharedUVStage5Ranker
from recommender.experiments.roadmap_v2.config import EVIDENCE_ROOT as V1_EVIDENCE_ROOT
from recommender.experiments.roadmap_v2.native_base_name import (
    CANDIDATE_38_FEATURE_NAMES,
    CUSTOMER_DYNAMIC_11_FEATURE_NAMES,
)
from recommender.experiments.roadmap_v2.run_stage5 import _verify_reference_files
from recommender.experiments.roadmap_v2.run_stage5_ablations import _evaluate_with_cohorts_and_sku
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from .provenance import input_digest, prepared_dataset_digest
from .prepare_v2_full import load_prepared_metadata
from .config import (
    ARTIFACT_ROOT,
    load_v2_development_purchases,
    EVIDENCE_ROOT,
    PREPARED_ROOT,
    RAW_ITEMS_PATH,
    SEED,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_WEIGHT_DECAY,
    TEST_START_V2,
    TRAIN_CUTOFF_V2,
)
from .e2e_ranker import EndToEndStage2Ranker, TwoStageAdaptiveRanker
from .train_v2_full_baseline import (
    SELECTED_OUTPUTS,
    completed_arm,
    amp_autocast_context,
    build_v2_stage1_tower,
    iter_training_batches,
    load_v2_stage1_vectors,
    rank_eligible_formulations,
    sync_outputs_to_dir,
)


ARMS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "arm1_2stage_unseen_attn_aux",
        "mode": "two_stage",
        "unfreeze_product_tower": False,
        "unseen_attn_aux_enabled": True,
        "unseen_attn_beta": 0.25,
    },
    {
        "name": "arm2_2stage_unfreeze_vbn_unseen_attn",
        "mode": "two_stage",
        "unfreeze_product_tower": True,
        "product_tower_lr": 1e-4,
        "product_tower_warmup_epochs": 1,
        "unseen_attn_aux_enabled": True,
        "unseen_attn_beta": 0.25,
    },
    {
        "name": "arm3_e2e_stage2_only",
        "mode": "e2e",
        "unseen_attn_aux_enabled": False,
        "unseen_attn_beta": 0.0,
    },
    {
        "name": "arm4_e2e_unseen_attn_aux",
        "mode": "e2e",
        "unseen_attn_aux_enabled": True,
        "unseen_attn_beta": 0.25,
    },
)


def _to_dev_prod_inputs(prod_inputs: Mapping[str, Any], dev: torch.device) -> Dict[str, Any]:
    return {
        "text_embedding": prod_inputs["text_embedding"].to(dev, dtype=torch.float32),
        "demand_features": prod_inputs["demand_features"].to(dev, dtype=torch.float32),
        "attr_ids": {k: v.to(dev) for k, v in prod_inputs["attr_ids"].items()},
    }


def _build_two_stage_batch(
    split_meta: Mapping[str, Any],
    cand_np: np.ndarray,
    u_bn_split: torch.Tensor,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    return {
        "customer_vector": u_bn_split[rows].to(dev, dtype=torch.float32),
        "customer_features": split_meta["tabular_features"][rows].to(dev, dtype=torch.float32),
        "sequence": split_meta["sequence"][rows].to(dev),
        "attention_mask": split_meta["attention_mask"][rows].to(dev),
        "seq_product_ids": split_meta["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_meta["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_meta["prediction_days"][rows].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand_np, dtype=np.float32)).to(dev),
        "candidate_mask": split_meta["allowed_bn"][rows].to(dev),
    }


def _build_e2e_batch(
    split_meta: Mapping[str, Any],
    cand_np: np.ndarray,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    return {
        "company_text_embedding": split_meta["text_embedding"][rows].to(dev, dtype=torch.float32),
        "city_id": split_meta["city_id"][rows].to(dev),
        "region_id": split_meta["region_id"][rows].to(dev),
        "industry_id": split_meta["industry_id"][rows].to(dev),
        "customer_features": split_meta["tabular_features"][rows].to(dev, dtype=torch.float32),
        "sequence": split_meta["sequence"][rows].to(dev),
        "attention_mask": split_meta["attention_mask"][rows].to(dev),
        "seq_product_ids": split_meta["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_meta["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_meta["prediction_days"][rows].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand_np, dtype=np.float32)).to(dev),
        "candidate_mask": split_meta["allowed_bn"][rows].to(dev),
    }


def _strip_training_aux(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("aux_60d_head.")
        and not k.startswith("unseen_attn_head.")
        and not k.startswith("ranker.aux_60d_head.")
    }


def _load_pretrained_product_tower(
    shared_meta: Mapping[str, Any],
    s1_best_pt: Path,
    dev: torch.device,
) -> torch.nn.Module:
    n_demand = int(shared_meta["prod_inputs"]["demand_features"].shape[1])
    full_s1 = build_v2_stage1_tower(shared_meta, num_demand_features=n_demand)
    ck = torch.load(s1_best_pt, map_location="cpu", weights_only=True)
    full_s1.load_state_dict(ck["model_state_dict"], strict=True)
    return full_s1.product_tower.to(dev)


@torch.no_grad()
def _score_val_memmap(
    model: torch.nn.Module,
    *,
    mode: str,
    va_meta: Mapping[str, Any],
    va_cand_mm: np.ndarray,
    prod_inputs_dev: Mapping[str, Any],
    u_va: Optional[torch.Tensor],
    frozen_v_dev: Optional[torch.Tensor],
    dev: torch.device,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    if mode == "e2e" or getattr(model, "product_tower", None) is not None:
        v_bn = model.precompute_product_vectors(prod_inputs_dev)
    else:
        assert frozen_v_dev is not None
        v_bn = frozen_v_dev

    n_rows = len(va_meta["customer_ids"])
    chunks = []
    for s0 in range(0, n_rows, 2048):
        s1 = min(n_rows, s0 + 2048)
        buf_cand = np.asarray(va_cand_mm[s0:s1], dtype=np.float32)
        for off in range(0, s1 - s0, batch_size):
            rows = torch.arange(s0 + off, min(s1, s0 + off + batch_size), dtype=torch.long)
            c_slice = buf_cand[off : off + len(rows)]
            if mode == "e2e":
                inp = _build_e2e_batch(va_meta, c_slice, rows, dev)
            else:
                assert u_va is not None
                inp = _build_two_stage_batch(va_meta, c_slice, u_va, rows, dev)
            out = model(**inp, precomputed_v_bn=v_bn, return_aux=False)
            chunks.append(out.logits.cpu())
        del buf_cand
    return torch.cat(chunks, dim=0), v_bn.cpu()


def train_and_eval_arm(
    arm_cfg: Mapping[str, Any],
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    tr_cand_mm: np.ndarray,
    va_cand_mm: np.ndarray,
    u_tr: torch.Tensor,
    u_va: torch.Tensor,
    frozen_v_bn: torch.Tensor,
    s1_best_pt: Path,
    dev_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    dataset_identity: Optional[str] = None,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    dev: torch.device,
    amp: bool = False,
    in_memory: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    arm_name = str(arm_cfg["name"])
    mode = str(arm_cfg["mode"])
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_summary_p = out_dir / "arm_result.json"
    if dataset_identity is None:
        dataset_identity = prepared_dataset_digest(
            shared_meta, tr_meta, va_meta, tr_cand_mm, va_cand_mm, dev_purchases, catalogue
        )
    run_settings = {
        "dataset_identity": dataset_identity,
        "stage1_sha256": file_digest(s1_best_pt),
        "vectors_sha256": input_digest((u_tr, u_va, frozen_v_bn)),
        "config": json.loads(json.dumps(arm_cfg)), "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "in_memory": in_memory,
        "bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()),
    }
    cached = completed_arm(out_dir, run_settings)
    if cached is not None:
        sync_outputs_to_dir(sync_dir, (out_dir,))
        return cached

    vocabs = shared_meta["vocabs"]
    enrichment = shared_meta["enrichment"]
    prod_inputs_dev = _to_dev_prod_inputs(shared_meta["prod_inputs"], dev)
    frozen_v_dev = frozen_v_bn.to(dev)
    n_train_bn = int(shared_meta["universe"]["trained_base_names"])
    n_demand = int(shared_meta["prod_inputs"]["demand_features"].shape[1])
    unfreeze_pt = bool(arm_cfg.get("unfreeze_product_tower", False))
    pt_warmup = int(arm_cfg.get("product_tower_warmup_epochs", 0))
    pt_lr = float(arm_cfg.get("product_tower_lr", 1e-4))
    unseen_beta = float(arm_cfg["unseen_attn_beta"])

    def _make_model(*, training_aux: bool) -> torch.nn.Module:
        if mode == "e2e":
            return EndToEndStage2Ranker(
                product_attr_vocab_sizes={k: len(v) for k, v in enrichment["product_vocabs"].items()},
                num_cities=len(vocabs["city_vocab"]) + 1,
                num_regions=len(vocabs["region_vocab"]) + 1,
                num_industries=len(vocabs["industry_vocab"]) + 1,
                num_demand_features=n_demand,
                num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
                num_customer_features=len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
                num_pairwise_bias_features=13,
                d_model=128,
                num_heads=8,
                memory_ffn_dim=384,
                mlp_widths=(384, 192, 96),
                dropout=0.12,
                aux_60d_enabled=training_aux,
                unseen_attn_aux_enabled=bool(arm_cfg["unseen_attn_aux_enabled"]) and training_aux,
                seed=seed,
            ).to(dev)

        base_ranker = SharedUVStage5Ranker(
            num_candidate_features=len(CANDIDATE_38_FEATURE_NAMES),
            num_customer_features=len(CUSTOMER_DYNAMIC_11_FEATURE_NAMES),
            num_pairwise_bias_features=13,
            d_model=128,
            num_heads=8,
            memory_ffn_dim=384,
            mlp_widths=(384, 192, 96),
            dropout=0.12,
            aux_60d_enabled=training_aux,
            use_memory_encoder=True,
            self_attention_bias_mode="none",
            seed=seed,
        ).to(dev)
        pt = _load_pretrained_product_tower(shared_meta, s1_best_pt, dev) if unfreeze_pt else None
        return TwoStageAdaptiveRanker(
            ranker=base_ranker,
            product_tower=pt,
            d_model=128,
            unseen_attn_aux_enabled=bool(arm_cfg["unseen_attn_aux_enabled"]) and training_aux,
        ).to(dev)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _make_model(training_aux=True)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_inf_params = sum(
        p.numel()
        for k, p in model.named_parameters()
        if p.requires_grad
        and not k.startswith("aux_60d_head.")
        and not k.startswith("unseen_attn_head.")
        and not k.startswith("ranker.aux_60d_head.")
    )
    n_req_params = sum(
        p.numel()
        for k, p in model.named_parameters()
        if p.requires_grad
        and not k.startswith("aux_60d_head.")
        and not k.startswith("unseen_attn_head.")
        and not k.startswith("ranker.aux_60d_head.")
        and not k.startswith("product_tower.")
    )

    if mode == "e2e":
        param_groups = model.optimizer_param_groups(STAGE2_WEIGHT_DECAY)
    elif unfreeze_pt:
        pt_params = list(model.product_tower.parameters())
        pt_ids = {id(p) for p in pt_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in pt_ids]
        param_groups = [
            {"params": other_params, "lr": 1e-3, "weight_decay": STAGE2_WEIGHT_DECAY},
            {"params": pt_params, "lr": pt_lr, "weight_decay": STAGE2_WEIGHT_DECAY},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-3, "weight_decay": STAGE2_WEIGHT_DECAY}]

    opt = torch.optim.AdamW(param_groups, lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(epochs), eta_min=STAGE2_ETA_MIN)
    loss_fn = RankerLoss().to(dev)
    aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    unseen_attn_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    n_tr = len(tr_meta["customer_ids"])
    tot_va_new = int(va_meta["new_target_bn"].sum().item())
    aux_60d_bn = tr_meta["aux_60d_bn"]
    full_window = tr_meta["full_window"]

    use_bf16 = bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported())
    print(
        f"\n[v2-exp] Training {arm_name} (mode={mode}, device={dev}, bf16={use_bf16}, in_memory={in_memory}, "
        f"train_params={n_train_params:,}, inf_params={n_inf_params:,}, per_req_params={n_req_params:,}, "
        f"beta_unseen_attn={unseen_beta})...",
        flush=True,
    )

    best_ndcg_key = (-1.0, -1, -1.0)
    best_disc_key = (-1, -1.0, -1.0)
    epoch_history = []
    t_train = time.perf_counter()

    for ep in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        # Safe warmup: freeze pretrained ProductTower during Ep 1 when unfreezing inside 2-stage
        if unfreeze_pt and model.product_tower is not None:
            pt_active = ep > pt_warmup
            for p in model.product_tower.parameters():
                p.requires_grad_(pt_active)
            if not pt_active:
                model.product_tower.eval()

        losses, main_losses, aux_losses, uattn_losses = [], [], [], []
        for rows, cand_np in iter_training_batches(tr_cand_mm, n_tr, batch_size, gen, in_memory=in_memory):
            opt.zero_grad(set_to_none=True)
            if mode == "e2e":
                inp = _build_e2e_batch(tr_meta, cand_np, rows, dev)
                with amp_autocast_context(dev, use_bf16):
                    res = model(**inp, prod_inputs=prod_inputs_dev, train_bn_limit=n_train_bn, return_aux=True)
            else:
                inp = _build_two_stage_batch(tr_meta, cand_np, u_tr, rows, dev)
                with amp_autocast_context(dev, use_bf16):
                    if unfreeze_pt and ep > pt_warmup:
                        res = model(**inp, prod_inputs=prod_inputs_dev, train_bn_limit=n_train_bn, return_aux=True)
                    else:
                        res = model(**inp, precomputed_v_bn=frozen_v_dev, return_aux=True)

            cmask = inp["candidate_mask"]
            y = tr_meta["target_bn"][rows].to(dev, dtype=torch.float32) * cmask.float()
            l_main = loss_fn(res["output"].logits.float(), y, candidate_mask=cmask)["loss"]
            loss = l_main
            main_losses.append(float(l_main.detach().cpu()))

            aux_rows = full_window[rows].to(dev)
            if aux_rows.any() and res["aux_60d_logits"] is not None:
                y_aux = aux_60d_bn[rows].to(dev, dtype=torch.float32)[aux_rows] * cmask[aux_rows].float()
                l_aux = aux_loss_fn(res["aux_60d_logits"][aux_rows].float(), y_aux, candidate_mask=cmask[aux_rows])["loss"]
                loss = loss + 0.3 * l_aux
                aux_losses.append(float(l_aux.detach().cpu()))

            if unseen_beta > 0.0 and aux_rows.any() and res["unseen_attn_logits"] is not None:
                unseen_m = tr_meta["unseen_bn"][rows].to(dev) & cmask
                y_u60 = (aux_60d_bn[rows].to(dev) & unseen_m).float()
                valid_u = aux_rows & (y_u60.sum(dim=-1) > 0)
                if valid_u.any():
                    u_logits = res["unseen_attn_logits"][valid_u].float().masked_fill(~unseen_m[valid_u], -torch.inf)
                    l_uattn = unseen_attn_loss_fn(
                        u_logits, y_u60[valid_u], candidate_mask=unseen_m[valid_u]
                    )["loss"]
                    loss = loss + unseen_beta * l_uattn
                    uattn_losses.append(float(l_uattn.detach().cpu()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], STAGE2_GRAD_CLIP_NORM
            )
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sch.step()

        va_lg, _ = _score_val_memmap(
            model,
            mode=mode,
            va_meta=va_meta,
            va_cand_mm=va_cand_mm,
            prod_inputs_dev=prod_inputs_dev,
            u_va=u_va,
            frozen_v_dev=frozen_v_dev,
            dev=dev,
            batch_size=batch_size,
        )
        ranked = rank_eligible_formulations(va_lg, va_meta["allowed_bn"])
        m_ep = evaluate_base_name_predictions(
            ranked, va_meta["target_bn"], va_meta["new_target_bn"], va_meta["customer_ids"]
        )
        inf_state = _strip_training_aux(model.state_dict())

        nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
        dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, out_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, out_dir / "best_discovery.pt")

        ep_rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "main_loss": float(np.mean(main_losses)),
            "aux_60d_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "unseen_attn_loss": float(np.mean(uattn_losses)) if uattn_losses else 0.0,
            "bn_recall@5": float(m_ep["bn_recall@5"]),
            "bn_recall@10": float(m_ep["bn_recall@10"]),
            "bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
            "true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
            "true_new_bn_hits@25": int(m_ep["true_new_bn_hits@25"]),
        }
        epoch_history.append(ep_rec)
        print(
            f"[v2-exp:{arm_name}] ep={ep:02d}/{epochs} loss={ep_rec['loss']:.4f} "
            f"R@5={ep_rec['bn_recall@5']:.5f} R@10={ep_rec['bn_recall@10']:.5f} "
            f"NDCG@10={ep_rec['bn_ndcg@10']:.5f} New@10={ep_rec['true_new_bn_hits@10']}/{tot_va_new} "
            f"New@25={ep_rec['true_new_bn_hits@25']} ({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    # Replay + cohort + resolved SKU + 1-req CPU latency
    product_ids = shared_meta["product_ids"]
    universe = shared_meta["universe"]
    sku_val = va_meta["sku_val"]

    arm_res: Dict[str, Any] = {
        "arm": arm_name,
        "config": dict(arm_cfg),
        "train_params": int(n_train_params),
        "inference_params": int(n_inf_params),
        "per_request_params": int(n_req_params),
        "training_seconds": float(time.perf_counter() - t_train),
        "epoch_history": epoch_history,
    }
    cpu_bench_ms = {}

    for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
        ck = torch.load(out_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_model(training_aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_lg, v_bn_cpu = _score_val_memmap(
            eval_m,
            mode=mode,
            va_meta=va_meta,
            va_cand_mm=va_cand_mm,
            prod_inputs_dev=prod_inputs_dev,
            u_va=u_va,
            frozen_v_dev=frozen_v_dev,
            dev=dev,
            batch_size=batch_size,
        )
        ranked = rank_eligible_formulations(va_lg, va_meta["allowed_bn"])
        npz_p = out_dir / f"predictions_{sel_name}.npz"
        ev = _evaluate_with_cohorts_and_sku(
            ranked, va_meta, sku_val, dev_purchases, catalogue, product_ids, universe, npz_p
        )
        if ev["bn_metrics"] != ck["metrics"]:
            raise AssertionError(f"{arm_name} {sel_name} replay mismatch!")
        ev["epoch"] = int(ck["epoch"])
        arm_res[sel_name] = ev

        if sel_name == "ndcg":
            eval_cpu = eval_m.to("cpu").eval()
            one_cand = np.asarray(va_cand_mm[0:1], dtype=np.float32)
            if mode == "e2e":
                one_inp = _build_e2e_batch(va_meta, one_cand, torch.tensor([0]), torch.device("cpu"))
            else:
                one_inp = _build_two_stage_batch(va_meta, one_cand, u_va, torch.tensor([0]), torch.device("cpu"))
            with torch.no_grad():
                for _ in range(10):
                    eval_cpu(**one_inp, precomputed_v_bn=v_bn_cpu)
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    eval_cpu(**one_inp, precomputed_v_bn=v_bn_cpu)
                    ts.append((time.perf_counter() - t0) * 1000.0)
            cpu_bench_ms = {
                "median_ms": float(np.median(ts)),
                "p95_ms": float(np.percentile(ts, 95)),
                "mean_ms": float(np.mean(ts)),
            }

    arm_res["cpu_single_request_ranker_ms"] = cpu_bench_ms

    arm_res["run_settings"] = run_settings
    arm_res["output_sha256"] = {name: file_digest(out_dir / name) for name in SELECTED_OUTPUTS}
    atomic_json(arm_res, arm_summary_p, overwrite=True)
    sync_outputs_to_dir(sync_dir, (out_dir,))
    del model, opt, sch
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    return arm_res


def run_all_v2_full_experiments(
    *,
    seed: int = SEED,
    epochs: int = STAGE2_EPOCHS,
    batch_size: int = STAGE2_BATCH_SIZE,
    arms: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    amp: bool = False,
    in_memory: bool = False,
    skip_v1_verify: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    before_hashes = None
    ref_v1 = ref_v1_3s = None
    if not skip_v1_verify:
        before_hashes = snapshot_preserved_hashes()
        ref_v1 = json.loads((V1_EVIDENCE_ROOT / "stage5_reference.json").read_text(encoding="utf-8"))
        ref_v1_3s = json.loads((V1_EVIDENCE_ROOT / "stage5_3stage_reference.json").read_text(encoding="utf-8"))
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    baseline_json_p = EVIDENCE_ROOT / f"stage5_v5_baseline_seed{seed}.json"
    if not baseline_json_p.is_file():
        raise FileNotFoundError(f"Run baseline first: {baseline_json_p} not found.")
    baseline = json.loads(baseline_json_p.read_text(encoding="utf-8"))

    s1_dir = ARTIFACT_ROOT / f"baseline_v5_shared_uv-seed{seed}" / "two_tower"
    s1_best_pt = s1_dir / "best.pt"
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)
    frozen_v_bn, u_tr, u_va = load_v2_stage1_vectors(s1_dir, shared_meta, tr_meta, va_meta)
    mmap_mode = None if in_memory else "r"
    if in_memory:
        print("[v2-exp] Loading float16 candidate features into RAM (--in-memory)...", flush=True)
    tr_cand_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)

    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)

    if arms is not None and (not arms or set(arms) - {a["name"] for a in ARMS}):
        raise ValueError(f"Unsupported Wave 1 arm selection: {arms!r}")
    selected_arms = [a for a in ARMS if arms is None or a["name"] in arms]
    dataset_identity = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand_mm, va_cand_mm, dev_purchases, catalogue
    )
    exp_root = ARTIFACT_ROOT / f"experiments_all-seed{seed}"
    arm_results: Dict[str, Any] = {}
    for arm_cfg in selected_arms:
        arm_results[arm_cfg["name"]] = train_and_eval_arm(
            arm_cfg,
            dataset_identity=dataset_identity,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand_mm=tr_cand_mm,
            va_cand_mm=va_cand_mm,
            u_tr=u_tr,
            u_va=u_va,
            frozen_v_bn=frozen_v_bn,
            s1_best_pt=s1_best_pt,
            dev_purchases=dev_purchases,
            catalogue=catalogue,
            out_dir=exp_root / arm_cfg["name"],
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            dev=dev,
            amp=amp,
            in_memory=in_memory,
            sync_dir=sync_dir,
        )

    comparisons: Dict[str, Any] = {}
    base_ndcg_npz = Path(baseline["stage2_ranker"]["ndcg"]["predictions_npz"])
    base_disc_npz = Path(baseline["stage2_ranker"]["discovery"]["predictions_npz"])
    for name, res in arm_results.items():
        comparisons[f"{name}_vs_baseline_ndcg"] = paired_base_name_bootstrap_ci(
            base_ndcg_npz, Path(res["ndcg"]["predictions_npz"]), seed=seed
        )
        comparisons[f"{name}_vs_baseline_discovery"] = paired_base_name_bootstrap_ci(
            base_disc_npz, Path(res["discovery"]["predictions_npz"]), seed=seed
        )
    stepwise_pairs = (
        ("arm1_2stage_unseen_attn_aux", "arm2_2stage_unfreeze_vbn_unseen_attn"),
        ("arm3_e2e_stage2_only", "arm4_e2e_unseen_attn_aux"),
        ("arm2_2stage_unfreeze_vbn_unseen_attn", "arm4_e2e_unseen_attn_aux"),
    )
    for a, b in stepwise_pairs:
        if a in arm_results and b in arm_results:
            comparisons[f"{b}_vs_{a}_ndcg"] = paired_base_name_bootstrap_ci(
                Path(arm_results[a]["ndcg"]["predictions_npz"]),
                Path(arm_results[b]["ndcg"]["predictions_npz"]),
                seed=seed,
            )

    if not skip_v1_verify:
        if snapshot_preserved_hashes() != before_hashes:
            raise AssertionError("v1 preserved reference hashes changed!")
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_ROOT / f"v2_full_experiments_seed{seed}.json"
    if len(selected_arms) != len(ARMS):
        summary_path = summary_path.with_stem(summary_path.stem + "_" + "_".join(a["name"].split("_")[0] for a in selected_arms))
    summary = {
        "selection_version": 2,
        "experiment": "v2_full_all_5_arms_stage2_experiments",
        "dataset": "v2_full",
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "amp_bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()),
        "baseline_2stage_v5": {
            "stage1_two_tower": baseline["stage1_two_tower"],
            "stage2_ndcg": baseline["stage2_ranker"]["ndcg"],
            "stage2_discovery": baseline["stage2_ranker"]["discovery"],
            "cpu_single_request_ranker_ms": baseline["cpu_single_request_ranker_ms"],
        },
        "arms": arm_results,
        "paired_bootstrap_95ci": comparisons,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(summary, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (summary_path,))
    print(f"\nSaved v2_full 5-arm summary -> {summary_path} ({summary['elapsed_seconds']:.1f}s)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=STAGE2_BATCH_SIZE)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--skip-v1-verify", action="store_true")
    parser.add_argument("--sync-dir", type=Path, default=None)
    args = parser.parse_args()
    run_all_v2_full_experiments(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        arms=args.arms,
        device=args.device,
        amp=args.amp,
        in_memory=args.in_memory,
        skip_v1_verify=args.skip_v1_verify,
        sync_dir=args.sync_dir,
    )


if __name__ == "__main__":
    main()
