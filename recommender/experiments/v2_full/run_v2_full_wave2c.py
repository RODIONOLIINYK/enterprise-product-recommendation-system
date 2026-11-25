"""Run Wave 2C clean separated Option A loss-weighting experiments (`v2_full`).

Evaluates Option A loss-weighting arms (`1.0 * L_optA` and `2.0 * L_optA`) exclusively on
single-stage E2E `Wave2BE2ERanker` (`arm4_e2e_unseen_attn_aux`, `cand_42 + CUST_7 + SEQ_5`,
Pure-Q `Q = v_bn`, Full-42 cross-attention bias, `OptionAUnseenHead` strictly on the 37 non-repeat
cols `5..41` with **0% dropout**, conflict-free separated loss masks, AdamW, native `bf16` AMP):

  - Reference (`v2_full_reference.json`): `w2c_exp1_clean_sep_w05` (`0.5 * L_optA`, `cal-Discovery` Ep 10:
    `R@10 = 82.07%`, `NDCG@10 = 0.65439`, `New@10 = 1,193 [691 Ret / 502 FO]`, `PureU@10 = 2,235`).
  - `w2c_exp1_clean_sep_w10`: Clean separated Option A with **`1.0 * L_optA`**.
  - `w2c_exp1_clean_sep_w20`: Clean separated Option A with **`2.0 * L_optA`** (imbalance loss weighting).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.experiments.roadmap_v2.config import EVIDENCE_ROOT as V1_EVIDENCE_ROOT
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

from .config import (
    ACTIVE_V2_FULL_BASE_MODEL,
    ACTIVE_V2_FULL_REFERENCE_ID,
    ARTIFACT_ROOT,
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
    load_v2_development_purchases,
    resolve_v2_full_reference_predictions,
)
from .extra_features import (
    NUM_CAND_42_FEATURES,
    NUM_CUST_7_FEATURES,
    SEQ_5_ACTIVE_INDICES,
    assemble_cand_42,
    build_or_load_extra_cand_5,
)
from .prepare_v2_full import load_prepared_metadata
from .provenance import input_digest, prepared_dataset_digest
from .run_v2_full_experiments import _to_dev_prod_inputs
from .run_v2_full_wave2b import (
    SUPPORTED_WAVE2B_BASES,
    _build_w2b_batch,
    _completed_w2b_arm,
    _expected_outputs_for_arm,
    _score_val_wave2b,
    iter_paired_training_batches,
)
from .train_v2_full_baseline import (
    amp_autocast_context,
    rank_eligible_formulations,
    sync_outputs_to_dir,
)
from .wave2b_models import (
    Wave2BE2ERanker,
    evaluate_pure_unseen_predictions,
    fit_oof_affine_unseen_blend,
    strip_wave2b_training_aux,
)

SUPPORTED_WAVE2C_BASES: Tuple[str, ...] = SUPPORTED_WAVE2B_BASES


def _expected_wave2c_outputs() -> Tuple[str, ...]:
    """Expected SHA-256-verified output files for one Wave 2C arm."""
    return (
        *_expected_outputs_for_arm(option_a=True),
        "predictions_optionA_oof_discovery.npz",
    )


def build_wave2c_arm_specs(base_name: str = ACTIVE_V2_FULL_BASE_MODEL) -> Tuple[Dict[str, Any], ...]:
    """Build active Wave 2C E2E loss-weighting arm specs (`w=1.0`, `w=2.0`) over the `w=0.5` reference."""
    if base_name not in SUPPORTED_WAVE2C_BASES:
        raise ValueError(f"Unsupported Wave 2C base={base_name!r}; expected one of {SUPPORTED_WAVE2C_BASES}")

    common = {
        "base_model": base_name,
        "mode": "e2e",
        "num_candidate_features": NUM_CAND_42_FEATURES,
        "num_customer_features": NUM_CUST_7_FEATURES,
        "active_seq_stat_indices": list(SEQ_5_ACTIVE_INDICES),
        "d_model": 128,
        "num_heads": 8,
        "memory_ffn_dim": 384,
        "memory_layers": 1,
        "mlp_widths": [384, 192, 96],
        "dropout": 0.12,
        "pure_v_query": True,
        "num_pairwise_bias_features": NUM_CAND_42_FEATURES,
        "option_a_unseen_head": True,
        "unseen_dropout_cold_prob": 0.0,
        "unseen_attn_aux_enabled": False,
        "unseen_attn_beta": 0.0,
        "separated_main_targets": True,
    }
    return (
        {
            **common,
            "name": "w2c_exp1_clean_sep_w10",
            "unseen_head_weight": 1.0,
        },
        {
            **common,
            "name": "w2c_exp1_clean_sep_w20",
            "unseen_head_weight": 2.0,
        },
    )


def compute_wave2c_losses(
    *,
    main_logits: torch.Tensor,
    main_aux_60d_logits: Optional[torch.Tensor],
    gen_unseen_logits: torch.Tensor,
    gen_unseen_aux_60d_logits: Optional[torch.Tensor],
    allowed_bn: torch.Tensor,
    unseen_bn: torch.Tensor,
    target_bn: torch.Tensor,
    aux_60d_bn: torch.Tensor,
    full_window: torch.Tensor,
    first_observed: torch.Tensor,
    main_loss_fn: RankerLoss,
    main_aux_loss_fn: RankerLoss,
    unseen_next_loss_fn: RankerLoss,
    unseen_60d_loss_fn: RankerLoss,
    unseen_head_weight: float,
) -> Dict[str, Any]:
    """Compute conflict-free separated Wave 2C losses for Main Head and Option A Head.

    - Main Head: returning baskets only (`~first_observed`), `candidate_mask_main = allowed & ~(target & unseen)`,
      `y_main = target & ~unseen` (0-repeat returning baskets still receive BCE as all-zero negatives).
      Same conflict-free mask for `0.3 * L_aux60d`.
    - Option A Head: `candidate_mask_unseen = allowed & unseen`, real unseen positives only
      (`L_unseen_next + 0.3 * L_unseen_60d`), weighted by `unseen_head_weight`.
    """
    dev = main_logits.device
    allowed = allowed_bn.bool()
    unseen = unseen_bn.bool()
    target = target_bn.bool() & allowed
    returning = ~first_observed.bool()

    l_main = None
    l_aux = None
    total_loss: Optional[torch.Tensor] = None

    if bool(returning.any()):
        a_ret = allowed[returning]
        u_ret = unseen[returning]
        t_ret = target[returning]
        main_mask = a_ret & (~(t_ret & u_ret))
        y_main = (t_ret & (~u_ret)).float() * main_mask.float()
        main_lg = main_logits[returning].float().masked_fill(~main_mask, -torch.inf)
        l_main = main_loss_fn(main_lg, y_main, candidate_mask=main_mask)["loss"]
        total_loss = l_main

    aux_ret = full_window.bool() & returning
    if bool(aux_ret.any()) and main_aux_60d_logits is not None:
        a_aux = allowed[aux_ret]
        u_aux = unseen[aux_ret]
        t60 = aux_60d_bn[aux_ret].bool() & a_aux
        main_aux_mask = a_aux & (~(t60 & u_aux))
        y_main_aux = (t60 & (~u_aux)).float() * main_aux_mask.float()
        aux_lg = main_aux_60d_logits[aux_ret].float().masked_fill(~main_aux_mask, -torch.inf)
        l_aux = main_aux_loss_fn(aux_lg, y_main_aux, candidate_mask=main_aux_mask)["loss"]
        total_loss = (0.3 * l_aux) if total_loss is None else (total_loss + 0.3 * l_aux)

    unseen_m = unseen & allowed
    y_u_gen = (target & unseen_m).float()
    valid_gen = y_u_gen.sum(dim=-1) > 0

    l_opt_a: Optional[torch.Tensor] = None
    l_u_next: Optional[torch.Tensor] = None
    l_u_60d: Optional[torch.Tensor] = None

    if bool(valid_gen.any()):
        u_lg = gen_unseen_logits[valid_gen].float().masked_fill(~unseen_m[valid_gen], -torch.inf)
        l_u_next = unseen_next_loss_fn(u_lg, y_u_gen[valid_gen], candidate_mask=unseen_m[valid_gen])["loss"]
        l_opt_a = l_u_next

    if bool(full_window.bool().any()) and gen_unseen_aux_60d_logits is not None:
        y_u60 = (aux_60d_bn.bool() & unseen_m).float()
        valid_u60 = full_window.bool() & (y_u60.sum(dim=-1) > 0)
        if bool(valid_u60.any()):
            u60_lg = gen_unseen_aux_60d_logits[valid_u60].float().masked_fill(~unseen_m[valid_u60], -torch.inf)
            l_u_60d = unseen_60d_loss_fn(u60_lg, y_u60[valid_u60], candidate_mask=unseen_m[valid_u60])["loss"]
            l_opt_a = (0.3 * l_u_60d) if l_opt_a is None else (l_opt_a + 0.3 * l_u_60d)

    if l_opt_a is not None:
        weighted_opt_a = float(unseen_head_weight) * l_opt_a
        total_loss = weighted_opt_a if total_loss is None else (total_loss + weighted_opt_a)

    if total_loss is None:
        total_loss = torch.zeros((), dtype=torch.float32, device=dev, requires_grad=True)

    return {
        "loss": total_loss,
        "main_loss": l_main,
        "aux_60d_loss": l_aux,
        "option_a_loss": l_opt_a,
        "unseen_next_loss": l_u_next,
        "unseen_60d_loss": l_u_60d,
    }


def train_and_eval_wave2c_arm(
    arm_cfg: Mapping[str, Any],
    *,
    shared_meta: Mapping[str, Any],
    tr_meta: Mapping[str, Any],
    va_meta: Mapping[str, Any],
    tr_cand38_mm: np.ndarray,
    va_cand38_mm: np.ndarray,
    tr_extra5_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
    extra5_identity: str,
    dataset_identity: Optional[str] = None,
    dev_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    dev: torch.device,
    amp: bool = False,
    in_memory: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train, replay-verify, evaluate OOF affine blend, and CPU-benchmark one clean Wave 2C arm."""
    arm_name = str(arm_cfg["name"])
    expected_files = _expected_wave2c_outputs()

    out_dir.mkdir(parents=True, exist_ok=True)
    arm_summary_p = out_dir / "arm_result.json"
    if dataset_identity is None:
        dataset_identity = prepared_dataset_digest(
            shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
        )
        extra5_identity = input_digest((extra5_identity, tr_extra5_mm, va_extra5_mm))
    use_bf16 = bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False))
    run_settings = {
        "version": 3,
        "wave": "wave2c",
        "dataset_identity": dataset_identity,
        "config": json.loads(json.dumps(arm_cfg)),
        "extra5_identity": str(extra5_identity),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "in_memory": bool(in_memory),
        "bf16": use_bf16,
    }
    cached = _completed_w2b_arm(out_dir, run_settings, expected_files)
    if cached is None and sync_dir is not None and not out_dir.is_absolute():
        stored = sync_dir / out_dir
        cached = _completed_w2b_arm(stored, run_settings, expected_files)
        if cached is not None:
            shutil.copytree(stored, out_dir, dirs_exist_ok=True)
    if cached is not None:
        sync_outputs_to_dir(sync_dir, (out_dir,))
        return cached

    vocabs = shared_meta["vocabs"]
    enrichment = shared_meta["enrichment"]
    prod_inputs_dev = _to_dev_prod_inputs(shared_meta["prod_inputs"], dev)
    n_train_bn = int(shared_meta["universe"]["trained_base_names"])
    n_demand = int(shared_meta["prod_inputs"]["demand_features"].shape[1])
    unseen_head_weight = float(arm_cfg["unseen_head_weight"])

    def _make_model(*, training_aux: bool) -> Wave2BE2ERanker:
        return Wave2BE2ERanker(
            product_attr_vocab_sizes={k: len(v) for k, v in enrichment["product_vocabs"].items()},
            num_cities=len(vocabs["city_vocab"]) + 1,
            num_regions=len(vocabs["region_vocab"]) + 1,
            num_industries=len(vocabs["industry_vocab"]) + 1,
            num_demand_features=n_demand,
            num_candidate_features=int(arm_cfg["num_candidate_features"]),
            num_customer_features=int(arm_cfg["num_customer_features"]),
            num_pairwise_bias_features=int(arm_cfg["num_pairwise_bias_features"]),
            active_seq_stat_indices=tuple(arm_cfg["active_seq_stat_indices"]),
            pure_v_query=bool(arm_cfg["pure_v_query"]),
            option_a_unseen_head=True,
            unseen_dropout_cold_prob=0.0,
            unseen_attn_aux_enabled=False,
            d_model=int(arm_cfg["d_model"]),
            num_heads=int(arm_cfg["num_heads"]),
            memory_ffn_dim=int(arm_cfg["memory_ffn_dim"]),
            memory_layers=int(arm_cfg["memory_layers"]),
            mlp_widths=tuple(int(w) for w in arm_cfg["mlp_widths"]),
            dropout=float(arm_cfg["dropout"]),
            aux_60d_enabled=training_aux,
            seed=seed,
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
        and not k.startswith("option_a_head.aux_60d_head.")
    )
    n_req_params = sum(
        p.numel()
        for k, p in model.named_parameters()
        if p.requires_grad
        and not k.startswith("aux_60d_head.")
        and not k.startswith("unseen_attn_head.")
        and not k.startswith("option_a_head.aux_60d_head.")
        and not k.startswith("product_tower.")
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(epochs), eta_min=STAGE2_ETA_MIN)
    main_loss_fn = RankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    main_aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    unseen_next_loss_fn = RankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    unseen_60d_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    n_tr = len(tr_meta["customer_ids"])
    tot_va_new = int(va_meta["new_target_bn"].sum().item())
    aux_60d_bn = tr_meta["aux_60d_bn"]
    full_window = tr_meta["full_window"]

    print(
        f"\n[v2-w2c] Training {arm_cfg['base_model']}::{arm_name} "
        f"(sep_targets=True, dropout=0%, w_optA={unseen_head_weight}, bf16={use_bf16}, "
        f"train_params={n_train_params:,}, per_req_params={n_req_params:,})...",
        flush=True,
    )

    best_ndcg_key = (-1.0, -1, -1.0)
    best_disc_key = (-1, -1.0, -1.0)
    best_unseen_key = (-1, -1.0, -1)
    epoch_history = []
    t_train = time.perf_counter()

    for ep in range(1, int(epochs) + 1):
        ep_t0 = time.perf_counter()
        model.train()
        losses, main_losses, aux_losses, opt_a_losses = [], [], [], []

        for rows, cand42_np in iter_paired_training_batches(
            tr_cand38_mm, tr_extra5_mm, n_tr, batch_size, gen, in_memory=in_memory
        ):
            opt.zero_grad(set_to_none=True)
            inp = _build_w2b_batch(split_meta=tr_meta, cand42_np=cand42_np, rows=rows, dev=dev)
            with amp_autocast_context(dev, use_bf16):
                res = model(**inp, prod_inputs=prod_inputs_dev, train_bn_limit=n_train_bn, return_aux=True)

            loss_dict = compute_wave2c_losses(
                main_logits=res["output"].logits,
                main_aux_60d_logits=res["aux_60d_logits"],
                gen_unseen_logits=res["unseen_logits"],
                gen_unseen_aux_60d_logits=res["unseen_aux_60d_logits"],
                allowed_bn=inp["candidate_mask"],
                unseen_bn=tr_meta["unseen_bn"][rows].to(dev),
                target_bn=tr_meta["target_bn"][rows].to(dev),
                aux_60d_bn=aux_60d_bn[rows].to(dev),
                full_window=full_window[rows].to(dev),
                first_observed=tr_meta["first_observed"][rows].to(dev),
                main_loss_fn=main_loss_fn,
                main_aux_loss_fn=main_aux_loss_fn,
                unseen_next_loss_fn=unseen_next_loss_fn,
                unseen_60d_loss_fn=unseen_60d_loss_fn,
                unseen_head_weight=unseen_head_weight,
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], STAGE2_GRAD_CLIP_NORM
            )
            opt.step()

            losses.append(float(loss.detach().cpu()))
            if loss_dict["main_loss"] is not None:
                main_losses.append(float(loss_dict["main_loss"].detach().cpu()))
            if loss_dict["aux_60d_loss"] is not None:
                aux_losses.append(float(loss_dict["aux_60d_loss"].detach().cpu()))
            if loss_dict["option_a_loss"] is not None:
                opt_a_losses.append(float(loss_dict["option_a_loss"].detach().cpu()))
        sch.step()

        va_main_lg, va_unseen_lg, _ = _score_val_wave2b(
            model,
            va_meta=va_meta,
            va_cand38_mm=va_cand38_mm,
            va_extra5_mm=va_extra5_mm,
            prod_inputs_dev=prod_inputs_dev,
            dev=dev,
            batch_size=batch_size,
        )
        assert va_unseen_lg is not None
        ranked_main = rank_eligible_formulations(va_main_lg, va_meta["allowed_bn"])
        m_ep = evaluate_base_name_predictions(
            ranked_main, va_meta["target_bn"], va_meta["new_target_bn"], va_meta["customer_ids"]
        )
        pure_opt_a_ep = evaluate_pure_unseen_predictions(
            va_unseen_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            new_target_bn=va_meta["new_target_bn"],
            first_observed=va_meta["first_observed"],
        )
        oof_scores_ep, oof_meta_ep = fit_oof_affine_unseen_blend(
            va_main_lg,
            va_unseen_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            target_bn=va_meta["target_bn"],
            new_target_bn=va_meta["new_target_bn"],
            customer_ids=va_meta["customer_ids"],
            n_folds=2,
        )
        ranked_oof_ep = rank_eligible_formulations(oof_scores_ep, va_meta["allowed_bn"])
        m_oof_ep = evaluate_base_name_predictions(
            ranked_oof_ep, va_meta["target_bn"], va_meta["new_target_bn"], va_meta["customer_ids"]
        )

        inf_state = strip_wave2b_training_aux(model.state_dict())
        nk = (
            float(m_oof_ep["bn_ndcg@10"]),
            int(m_oof_ep["true_new_bn_hits@10"]),
            float(m_oof_ep["bn_recall@10"]),
        )
        dk = (
            int(m_oof_ep["true_new_bn_hits@10"]),
            float(m_oof_ep["bn_ndcg@10"]),
            float(m_oof_ep["bn_recall@10"]),
        )
        ov = pure_opt_a_ep["overall"]
        uk = (
            int(ov["pure_unseen_hits@10"]),
            float(ov["pure_unseen_ndcg@10_new_baskets"]),
            int(ov["pure_unseen_hits@25"]),
        )
        ckpt_payload = {
            "epoch": ep,
            "metrics": m_ep,
            "oof_metrics": m_oof_ep,
            "pure_unseen": pure_opt_a_ep,
            "model_state_dict": inf_state,
        }
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save(ckpt_payload, out_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save(ckpt_payload, out_dir / "best_discovery.pt")
        if uk > best_unseen_key:
            best_unseen_key = uk
            atomic_torch_save(ckpt_payload, out_dir / "best_unseen.pt")

        full_ab = oof_meta_ep["deployment_full_fit"]
        ep_rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "main_loss": float(np.mean(main_losses)) if main_losses else 0.0,
            "aux_60d_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "option_a_loss": float(np.mean(opt_a_losses)) if opt_a_losses else 0.0,
            "bn_recall@5": float(m_oof_ep["bn_recall@5"]),
            "bn_recall@10": float(m_oof_ep["bn_recall@10"]),
            "bn_ndcg@10": float(m_oof_ep["bn_ndcg@10"]),
            "true_new_bn_hits@10": int(m_oof_ep["true_new_bn_hits@10"]),
            "true_new_bn_hits@25": int(m_oof_ep["true_new_bn_hits@25"]),
            "main_only_bn_recall@10": float(m_ep["bn_recall@10"]),
            "main_only_bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
            "main_only_true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
            "oof_affine_full_a": float(full_ab["a"]),
            "oof_affine_full_b": float(full_ab["b"]),
            "pure_unseen_hits@10": int(ov["pure_unseen_hits@10"]),
            "pure_unseen_hits@25": int(ov["pure_unseen_hits@25"]),
            "pure_unseen_ndcg@10": float(ov["pure_unseen_ndcg@10_new_baskets"]),
        }
        epoch_history.append(ep_rec)
        print(
            f"[v2-w2c:{arm_name}] ep={ep:02d}/{epochs} loss={ep_rec['loss']:.4f} "
            f"main={ep_rec['main_loss']:.4f} optA={ep_rec['option_a_loss']:.4f} "
            f"CalR@10={ep_rec['bn_recall@10']:.5f} CalNDCG={ep_rec['bn_ndcg@10']:.5f} "
            f"CalNew@10={ep_rec['true_new_bn_hits@10']}/{tot_va_new} "
            f"PureU@10={ep_rec['pure_unseen_hits@10']}/{tot_va_new} "
            f"PureU@25={ep_rec['pure_unseen_hits@25']} ({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    product_ids = shared_meta["product_ids"]
    universe = shared_meta["universe"]
    sku_val = va_meta["sku_val"]

    arm_res: Dict[str, Any] = {
        "arm": arm_name,
        "base_model": str(arm_cfg["base_model"]),
        "config": dict(arm_cfg),
        "train_params": int(n_train_params),
        "inference_params": int(n_inf_params),
        "per_request_params": int(n_req_params),
        "training_seconds": float(time.perf_counter() - t_train),
        "epoch_history": epoch_history,
    }
    cpu_bench_ms: Dict[str, float] = {}

    for sel_name, fname in (
        ("ndcg", "best_ndcg.pt"),
        ("discovery", "best_discovery.pt"),
        ("unseen", "best_unseen.pt"),
    ):
        ck = torch.load(out_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_model(training_aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_main_lg, va_unseen_lg, v_bn_cpu = _score_val_wave2b(
            eval_m,
            va_meta=va_meta,
            va_cand38_mm=va_cand38_mm,
            va_extra5_mm=va_extra5_mm,
            prod_inputs_dev=prod_inputs_dev,
            dev=dev,
            batch_size=batch_size,
        )
        assert va_unseen_lg is not None
        ranked_main = rank_eligible_formulations(va_main_lg, va_meta["allowed_bn"])
        pure_main = evaluate_pure_unseen_predictions(
            va_main_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            new_target_bn=va_meta["new_target_bn"],
            first_observed=va_meta["first_observed"],
        )

        if sel_name in ("ndcg", "discovery"):
            npz_p = out_dir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(
                ranked_main, va_meta, sku_val, dev_purchases, catalogue, product_ids, universe, npz_p
            )
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"{arm_name} {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            ev["pure_unseen_main"] = pure_main
            arm_res[sel_name] = ev

        pure_unseen = evaluate_pure_unseen_predictions(
            va_unseen_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            new_target_bn=va_meta["new_target_bn"],
            first_observed=va_meta["first_observed"],
        )
        if pure_unseen != ck["pure_unseen"]:
            raise AssertionError(f"{arm_name} {sel_name} pure unseen replay mismatch!")

        oof_scores, oof_meta = fit_oof_affine_unseen_blend(
            va_main_lg,
            va_unseen_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            target_bn=va_meta["target_bn"],
            new_target_bn=va_meta["new_target_bn"],
            customer_ids=va_meta["customer_ids"],
            n_folds=2,
        )
        ranked_oof = rank_eligible_formulations(oof_scores, va_meta["allowed_bn"])
        oof_npz = out_dir / f"predictions_optionA_oof_{sel_name}.npz"
        ev_oof = _evaluate_with_cohorts_and_sku(
            ranked_oof, va_meta, sku_val, dev_purchases, catalogue, product_ids, universe, oof_npz
        )
        if "oof_metrics" in ck and ev_oof["bn_metrics"] != ck["oof_metrics"]:
            raise AssertionError(f"{arm_name} {sel_name} OOF blend replay mismatch!")
        ev_oof["epoch"] = int(ck["epoch"])
        ev_oof["pure_unseen_head"] = pure_unseen
        ev_oof["affine_oof_calibration"] = oof_meta
        arm_res[f"optionA_oof_{sel_name}"] = ev_oof
        if sel_name in ("ndcg", "discovery"):
            arm_res[sel_name]["pure_unseen_head"] = pure_unseen

        if sel_name == "ndcg":
            np.save(out_dir / "val_logits_ndcg.npy", va_main_lg.float().numpy())
            np.save(out_dir / "val_unseen_logits_ndcg.npy", va_unseen_lg.float().numpy())

            eval_cpu = eval_m.to("cpu").eval()
            one_cand = assemble_cand_42(va_cand38_mm[0:1], va_extra5_mm[0:1])
            one_inp = _build_w2b_batch(
                split_meta=va_meta, cand42_np=one_cand, rows=torch.tensor([0]), dev=torch.device("cpu")
            )
            with torch.no_grad():
                for _ in range(10):
                    eval_cpu(**one_inp, precomputed_v_bn=v_bn_cpu, return_aux=False)
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    eval_cpu(**one_inp, precomputed_v_bn=v_bn_cpu, return_aux=False)
                    ts.append((time.perf_counter() - t0) * 1000.0)
            cpu_bench_ms = {
                "median_ms": float(np.median(ts)),
                "p95_ms": float(np.percentile(ts, 95)),
                "mean_ms": float(np.mean(ts)),
            }

    arm_res["cpu_single_request_ranker_ms"] = cpu_bench_ms
    arm_res["run_settings"] = run_settings
    arm_res["output_sha256"] = {name: file_digest(out_dir / name) for name in expected_files}
    atomic_json(arm_res, arm_summary_p, overwrite=True)
    sync_outputs_to_dir(sync_dir, (out_dir,))

    del model, opt, sch
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    return arm_res


def run_v2_full_wave2c(
    *,
    base_model: str = ACTIVE_V2_FULL_BASE_MODEL,
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
    """Run clean Wave 2C loss-weighting arms (`w=1.0`, `w=2.0`) with paired 95% CIs vs. `v2_full_reference.json`."""
    t_all = time.perf_counter()
    if base_model not in SUPPORTED_WAVE2C_BASES:
        raise ValueError(f"Unsupported Wave 2C base={base_model!r}; expected one of {SUPPORTED_WAVE2C_BASES}")

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

    all_specs = build_wave2c_arm_specs(base_model)
    available = {a["name"] for a in all_specs}
    if arms is not None and (not arms or set(arms) - available):
        raise ValueError(f"Unsupported Wave 2C arms={arms!r}. Available: {sorted(available)}")
    selected_specs = [a for a in all_specs if arms is None or a["name"] in arms]

    tr_e5_p, va_e5_p, e5_manifest = build_or_load_extra_cand_5()
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)

    mmap_mode = None if in_memory else "r"
    if in_memory:
        print("[v2-w2c] Loading float16 feature caches into RAM (--in-memory)...", flush=True)
    tr_cand38_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand38_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)
    tr_extra5_mm = np.load(tr_e5_p, mmap_mode=mmap_mode)
    va_extra5_mm = np.load(va_e5_p, mmap_mode=mmap_mode)

    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)

    dataset_identity = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
    )
    e5_id = input_digest((e5_manifest["identity"], e5_manifest["output_sha256"]))

    exp_root = ARTIFACT_ROOT / f"wave2c_{base_model}-seed{seed}"
    arm_results: Dict[str, Any] = {}
    for arm_cfg in selected_specs:
        arm_results[arm_cfg["name"]] = train_and_eval_wave2c_arm(
            arm_cfg,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand38_mm=tr_cand38_mm,
            va_cand38_mm=va_cand38_mm,
            tr_extra5_mm=tr_extra5_mm,
            va_extra5_mm=va_extra5_mm,
            extra5_identity=e5_id,
            dataset_identity=dataset_identity,
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
    refs = resolve_v2_full_reference_predictions(sync_dir)
    expected_refs = {"v2_full_ref_primary", "v2_full_ref_ndcg", "v2_full_ref_cal_ndcg"}
    missing_refs = sorted(expected_refs - refs.keys())
    if missing_refs:
        print(f"Reference prediction files unavailable for paired comparisons: {missing_refs}", flush=True)
    for rname, rpath in refs.items():
        for aname, ares in arm_results.items():
            comparisons[f"{aname}_cal_unseen_vs_{rname}"] = paired_base_name_bootstrap_ci(
                rpath, Path(ares["optionA_oof_unseen"]["predictions_npz"]), seed=seed
            )
            comparisons[f"{aname}_cal_ndcg_vs_{rname}"] = paired_base_name_bootstrap_ci(
                rpath, Path(ares["optionA_oof_ndcg"]["predictions_npz"]), seed=seed
            )
            comparisons[f"{aname}_cal_discovery_vs_{rname}"] = paired_base_name_bootstrap_ci(
                rpath, Path(ares["optionA_oof_discovery"]["predictions_npz"]), seed=seed
            )

    arm_names = list(arm_results.keys())
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            a, b = arm_names[i], arm_names[j]
            for key, tag in (
                ("optionA_oof_unseen", "cal_unseen"),
                ("optionA_oof_ndcg", "cal_ndcg"),
                ("optionA_oof_discovery", "cal_discovery"),
            ):
                comparisons[f"{b}_{tag}_vs_{a}_{tag}"] = paired_base_name_bootstrap_ci(
                    Path(arm_results[a][key]["predictions_npz"]),
                    Path(arm_results[b][key]["predictions_npz"]),
                    seed=seed,
                )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_ROOT / f"v2_full_wave2c_{base_model}_seed{seed}.json"
    if len(selected_specs) != len(all_specs):
        summary_path = summary_path.with_stem(
            summary_path.stem + "_" + "_".join(a["name"] for a in selected_specs)
        )
    summary = {
        "experiment": "v2_full_wave2c_clean_optionA_loss_weight",
        "dataset": "v2_full",
        "reference_id": ACTIVE_V2_FULL_REFERENCE_ID,
        "feature_schema": "cand_42+CUST_7+SEQ_5 (OptionA=cand_37, dropout=0%)",
        "base_model": base_model,
        "extra_cand_5_identity": str(e5_manifest["identity"]),
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "amp_bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)),
        "arms": arm_results,
        "missing_comparison_references": missing_refs,
        "paired_bootstrap_95ci": comparisons,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(summary, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (summary_path,))
    print(f"\nSaved Wave 2C summary ({base_model}) -> {summary_path}", flush=True)

    if not skip_v1_verify:
        if snapshot_preserved_hashes() != before_hashes:
            raise AssertionError("v1 preserved reference hashes changed!")
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=str,
        default=ACTIVE_V2_FULL_BASE_MODEL,
        choices=SUPPORTED_WAVE2C_BASES,
        help="Active Wave 2C E2E base model (arm4_e2e_unseen_attn_aux).",
    )
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
    run_v2_full_wave2c(
        base_model=args.base,
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
