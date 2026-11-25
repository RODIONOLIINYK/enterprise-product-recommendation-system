"""Run Wave 2B experiments on `v2_full` (`arm4_e2e_unseen_attn_aux`, strictly `cand_42 + CUST_7 + SEQ_5`).

Runs exclusively on the active single-stage E2E architecture (`Wave2BE2ERanker`, zero Stage 1
pretraining) with the canonical 42-feature pruned schema (`cand_42 + CUST_7 + SEQ_5`, zero held-out
Test >= 2026-07-01 access) and computes paired 95% customer-clustered bootstrap CIs against the
canonical `v2_full` reference (`v2_full_reference.json`):

  1. `w2b_cand42_pruned`:
       `cand_42` (`cand_38` minus dead col 22 + `extra_cand_5`), `CUST_7` (drops cols 1,7,9,10),
       `SEQ_5` (`(0, 1, 5, 6, 8)` keeping `live_age`, dropping `gap_days`). Standard 13-feature bias,
       standard `candidate_mlp`, `arm4` training-only `Head_attn`.
  2. `w2b_cand42_optionA_70_30`:
       Replaces training-only `Head_attn` with inference-capable Option A Unseen-Only Discovery Head
       (`OptionAUnseenHead`) trained strictly on `unseen_bn & allowed_bn` with per-basket 70/30
       unscaled feature-group dropout (70% all 37 non-repeat features / 30% 7 global product stats
       only; 100% on at inference) + 2-fold customer-grouped OOF affine Top-10 calibration blend.
  3. `w2b_cand42_pureQ_full42bias`:
       Removes `candidate_mlp` before cross-attention (`Q = v_bn` directly) and expands
       `statistics_attention_weights` from 13 to all 42 candidate features (+1 constant intercept)
       x `(SEQ_5 + CUST_7)`.
  4. `w2b_cand42_optionA_pureQ_full42bias`:
       Combines Option A (70/30 + OOF affine blend) with Pure-Q (`Q = v_bn`) + Full-42 attention bias.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    prune_cust_7,
)
from .prepare_v2_full import load_prepared_metadata
from .provenance import input_digest, prepared_dataset_digest
from .run_v2_full_experiments import _to_dev_prod_inputs
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

SUPPORTED_WAVE2B_BASES: Tuple[str, ...] = (ACTIVE_V2_FULL_BASE_MODEL,)


def build_wave2b_arm_specs(base_name: str = ACTIVE_V2_FULL_BASE_MODEL) -> Tuple[Dict[str, Any], ...]:
    """Build the 4 Wave 2B E2E arm specs on `arm4_e2e_unseen_attn_aux` (`cand_42 + CUST_7 + SEQ_5`)."""
    if base_name not in SUPPORTED_WAVE2B_BASES:
        raise ValueError(f"Unsupported Wave 2B base={base_name!r}; expected one of {SUPPORTED_WAVE2B_BASES}")

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
        "unseen_dropout_cold_prob": 0.30,
        "unseen_head_weight": 0.5,
    }
    return (
        {
            **common,
            "name": "w2b_cand42_pruned",
            "pure_v_query": False,
            "num_pairwise_bias_features": 13,
            "option_a_unseen_head": False,
            "unseen_attn_aux_enabled": True,
            "unseen_attn_beta": 0.25,
        },
        {
            **common,
            "name": "w2b_cand42_optionA_70_30",
            "pure_v_query": False,
            "num_pairwise_bias_features": 13,
            "option_a_unseen_head": True,
            "unseen_attn_aux_enabled": False,
            "unseen_attn_beta": 0.0,
        },
        {
            **common,
            "name": "w2b_cand42_pureQ_full42bias",
            "pure_v_query": True,
            "num_pairwise_bias_features": NUM_CAND_42_FEATURES,
            "option_a_unseen_head": False,
            "unseen_attn_aux_enabled": True,
            "unseen_attn_beta": 0.25,
        },
        {
            **common,
            "name": "w2b_cand42_optionA_pureQ_full42bias",
            "pure_v_query": True,
            "num_pairwise_bias_features": NUM_CAND_42_FEATURES,
            "option_a_unseen_head": True,
            "unseen_attn_aux_enabled": False,
            "unseen_attn_beta": 0.0,
        },
    )


def iter_paired_training_batches(
    cand38_arr: np.ndarray,
    extra5_arr: np.ndarray,
    n_rows: int,
    batch_size: int,
    gen: torch.Generator,
    *,
    in_memory: bool = False,
    block_rows: int = 512,
    blocks_per_buffer: int = 8,
):
    """Yield `(rows_tensor, cand42_float32_np)` cleanly from RAM or synchronized NVMe memmaps."""
    if in_memory:
        perm = torch.randperm(n_rows, generator=gen)
        for rows in perm.split(int(batch_size)):
            r_np = rows.numpy()
            yield rows, assemble_cand_42(cand38_arr[r_np], extra5_arr[r_np])
        return

    starts = list(range(0, n_rows, int(block_rows)))
    block_order = torch.randperm(len(starts), generator=gen).tolist()
    for b_off in range(0, len(block_order), int(blocks_per_buffer)):
        chosen = block_order[b_off : b_off + int(blocks_per_buffer)]
        row_chunks = []
        cand_chunks = []
        for blk_i in chosen:
            s0 = starts[blk_i]
            s1 = min(n_rows, s0 + int(block_rows))
            row_chunks.append(np.arange(s0, s1, dtype=np.int64))
            cand_chunks.append(assemble_cand_42(cand38_arr[s0:s1], extra5_arr[s0:s1]))
        buf_rows = np.concatenate(row_chunks, axis=0)
        buf_cand = np.concatenate(cand_chunks, axis=0)
        perm = torch.randperm(len(buf_rows), generator=gen).numpy()
        for off in range(0, len(buf_rows), int(batch_size)):
            idx = perm[off : off + int(batch_size)]
            yield torch.from_numpy(buf_rows[idx].copy()), buf_cand[idx]
        del buf_rows, buf_cand, row_chunks, cand_chunks


def _build_w2b_batch(
    *,
    split_meta: Mapping[str, Any],
    cand42_np: np.ndarray,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    cust_7 = prune_cust_7(split_meta["tabular_features"][rows]).to(dev, dtype=torch.float32)
    return {
        "company_text_embedding": split_meta["text_embedding"][rows].to(dev, dtype=torch.float32),
        "city_id": split_meta["city_id"][rows].to(dev),
        "region_id": split_meta["region_id"][rows].to(dev),
        "industry_id": split_meta["industry_id"][rows].to(dev),
        "customer_features": cust_7,
        "sequence": split_meta["sequence"][rows].to(dev),
        "attention_mask": split_meta["attention_mask"][rows].to(dev),
        "seq_product_ids": split_meta["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_meta["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_meta["prediction_days"][rows].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand42_np, dtype=np.float32)).to(dev),
        "candidate_mask": split_meta["allowed_bn"][rows].to(dev),
    }


@torch.no_grad()
def _score_val_wave2b(
    model: Wave2BE2ERanker,
    *,
    va_meta: Mapping[str, Any],
    va_cand38_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
    prod_inputs_dev: Mapping[str, Any],
    dev: torch.device,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    model.eval()
    v_bn = model.precompute_product_vectors(prod_inputs_dev)

    n_rows = len(va_meta["customer_ids"])
    main_chunks: List[torch.Tensor] = []
    unseen_chunks: List[torch.Tensor] = []
    has_unseen = False

    for s0 in range(0, n_rows, 2048):
        s1 = min(n_rows, s0 + 2048)
        buf_cand = assemble_cand_42(va_cand38_mm[s0:s1], va_extra5_mm[s0:s1])
        for off in range(0, s1 - s0, batch_size):
            rows = torch.arange(s0 + off, min(s1, s0 + off + batch_size), dtype=torch.long)
            inp = _build_w2b_batch(
                split_meta=va_meta,
                cand42_np=buf_cand[off : off + len(rows)],
                rows=rows,
                dev=dev,
            )
            res = model(**inp, precomputed_v_bn=v_bn, return_aux=False)
            main_chunks.append(res["output"].logits.cpu())
            if res["unseen_logits"] is not None:
                has_unseen = True
                unseen_chunks.append(res["unseen_logits"].cpu())
        del buf_cand

    main_all = torch.cat(main_chunks, dim=0)
    unseen_all = torch.cat(unseen_chunks, dim=0) if has_unseen else None
    return main_all, unseen_all, v_bn.cpu()


def _expected_outputs_for_arm(option_a: bool) -> Tuple[str, ...]:
    base = [
        "best_ndcg.pt",
        "best_discovery.pt",
        "predictions_ndcg.npz",
        "predictions_discovery.npz",
        "val_logits_ndcg.npy",
    ]
    if option_a:
        base.extend(
            [
                "best_unseen.pt",
                "predictions_optionA_oof_ndcg.npz",
                "predictions_optionA_oof_unseen.npz",
                "val_unseen_logits_ndcg.npy",
            ]
        )
    return tuple(base)


def _completed_w2b_arm(
    directory: Path, run_settings: Mapping[str, Any], expected_files: Sequence[str]
) -> Optional[Dict[str, Any]]:
    summary_path = directory / "arm_result.json"
    if not summary_path.is_file():
        return None
    try:
        result = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if result.get("run_settings") != run_settings:
        return None
    for name in expected_files:
        p = directory / name
        if not p.is_file() or file_digest(p) != result.get("output_sha256", {}).get(name):
            return None
    return result


def train_and_eval_wave2b_arm(
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
    """Train, replay-verify, evaluate Mixed + Pure Unseen + OOF affine blend, and CPU-benchmark one Wave 2B E2E arm."""
    arm_name = str(arm_cfg["name"])
    option_a = bool(arm_cfg["option_a_unseen_head"])
    expected_files = _expected_outputs_for_arm(option_a)

    out_dir.mkdir(parents=True, exist_ok=True)
    arm_summary_p = out_dir / "arm_result.json"
    if dataset_identity is None:
        dataset_identity = prepared_dataset_digest(
            shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
        )
        extra5_identity = input_digest((extra5_identity, tr_extra5_mm, va_extra5_mm))
    run_settings = {
        "version": 3,
        "dataset_identity": dataset_identity,
        "config": json.loads(json.dumps(arm_cfg)),
        "extra5_identity": str(extra5_identity),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "in_memory": bool(in_memory),
        "bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()),
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

    unseen_attn_beta = float(arm_cfg.get("unseen_attn_beta", 0.0))
    unseen_head_weight = float(arm_cfg.get("unseen_head_weight", 0.5))

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
            option_a_unseen_head=option_a,
            unseen_dropout_cold_prob=float(arm_cfg["unseen_dropout_cold_prob"]),
            unseen_attn_aux_enabled=bool(arm_cfg["unseen_attn_aux_enabled"]) and training_aux,
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
    loss_fn = RankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    unseen_next_loss_fn = RankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    unseen_60d_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    n_tr = len(tr_meta["customer_ids"])
    tot_va_new = int(va_meta["new_target_bn"].sum().item())
    aux_60d_bn = tr_meta["aux_60d_bn"]
    full_window = tr_meta["full_window"]
    use_bf16 = bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported())

    print(
        f"\n[v2-w2b] Training {arm_cfg['base_model']}::{arm_name} (cand=42, pure_Q={arm_cfg['pure_v_query']}, "
        f"bias_feats={arm_cfg['num_pairwise_bias_features']}, optionA={option_a}, bf16={use_bf16}, "
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
        losses, main_losses, aux_losses, opt_a_losses, uattn_losses = [], [], [], [], []
        for rows, cand42_np in iter_paired_training_batches(
            tr_cand38_mm, tr_extra5_mm, n_tr, batch_size, gen, in_memory=in_memory
        ):
            opt.zero_grad(set_to_none=True)
            inp = _build_w2b_batch(split_meta=tr_meta, cand42_np=cand42_np, rows=rows, dev=dev)
            with amp_autocast_context(dev, use_bf16):
                res = model(**inp, prod_inputs=prod_inputs_dev, train_bn_limit=n_train_bn, return_aux=True)

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

            if option_a and res["unseen_logits"] is not None:
                unseen_m = tr_meta["unseen_bn"][rows].to(dev) & cmask
                y_u_next = (tr_meta["target_bn"][rows].to(dev) & unseen_m).float()
                valid_un = y_u_next.sum(dim=-1) > 0
                l_opt_a = None
                if valid_un.any():
                    u_lg = res["unseen_logits"][valid_un].float().masked_fill(~unseen_m[valid_un], -torch.inf)
                    l_opt_a = unseen_next_loss_fn(u_lg, y_u_next[valid_un], candidate_mask=unseen_m[valid_un])["loss"]

                if aux_rows.any() and res["unseen_aux_60d_logits"] is not None:
                    y_u60 = (aux_60d_bn[rows].to(dev) & unseen_m).float()
                    valid_u60 = aux_rows & (y_u60.sum(dim=-1) > 0)
                    if valid_u60.any():
                        u60_lg = res["unseen_aux_60d_logits"][valid_u60].float().masked_fill(
                            ~unseen_m[valid_u60], -torch.inf
                        )
                        l_u60 = unseen_60d_loss_fn(u60_lg, y_u60[valid_u60], candidate_mask=unseen_m[valid_u60])["loss"]
                        l_opt_a = (0.3 * l_u60) if l_opt_a is None else (l_opt_a + 0.3 * l_u60)

                if l_opt_a is not None:
                    loss = loss + unseen_head_weight * l_opt_a
                    opt_a_losses.append(float(l_opt_a.detach().cpu()))

            if unseen_attn_beta > 0.0 and aux_rows.any() and res["unseen_attn_logits"] is not None:
                unseen_m = tr_meta["unseen_bn"][rows].to(dev) & cmask
                y_u60 = (aux_60d_bn[rows].to(dev) & unseen_m).float()
                valid_u = aux_rows & (y_u60.sum(dim=-1) > 0)
                if valid_u.any():
                    u_logits = res["unseen_attn_logits"][valid_u].float().masked_fill(~unseen_m[valid_u], -torch.inf)
                    l_uattn = unseen_60d_loss_fn(u_logits, y_u60[valid_u], candidate_mask=unseen_m[valid_u])["loss"]
                    loss = loss + unseen_attn_beta * l_uattn
                    uattn_losses.append(float(l_uattn.detach().cpu()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], STAGE2_GRAD_CLIP_NORM
            )
            opt.step()
            losses.append(float(loss.detach().cpu()))
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
        ranked_main = rank_eligible_formulations(va_main_lg, va_meta["allowed_bn"])
        m_ep = evaluate_base_name_predictions(
            ranked_main, va_meta["target_bn"], va_meta["new_target_bn"], va_meta["customer_ids"]
        )
        pure_main_ep = evaluate_pure_unseen_predictions(
            va_main_lg,
            allowed_bn=va_meta["allowed_bn"],
            unseen_bn=va_meta["unseen_bn"],
            new_target_bn=va_meta["new_target_bn"],
            first_observed=va_meta["first_observed"],
        )
        pure_opt_a_ep = None
        if va_unseen_lg is not None:
            pure_opt_a_ep = evaluate_pure_unseen_predictions(
                va_unseen_lg,
                allowed_bn=va_meta["allowed_bn"],
                unseen_bn=va_meta["unseen_bn"],
                new_target_bn=va_meta["new_target_bn"],
                first_observed=va_meta["first_observed"],
            )

        inf_state = strip_wave2b_training_aux(model.state_dict())
        nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
        dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save(
                {"epoch": ep, "metrics": m_ep, "pure_unseen": pure_opt_a_ep or pure_main_ep, "model_state_dict": inf_state},
                out_dir / "best_ndcg.pt",
            )
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save(
                {"epoch": ep, "metrics": m_ep, "pure_unseen": pure_opt_a_ep or pure_main_ep, "model_state_dict": inf_state},
                out_dir / "best_discovery.pt",
            )
        if option_a and pure_opt_a_ep is not None:
            ov = pure_opt_a_ep["overall"]
            uk = (
                int(ov["pure_unseen_hits@10"]),
                float(ov["pure_unseen_ndcg@10_new_baskets"]),
                int(ov["pure_unseen_hits@25"]),
            )
            if uk > best_unseen_key:
                best_unseen_key = uk
                atomic_torch_save(
                    {"epoch": ep, "metrics": m_ep, "pure_unseen": pure_opt_a_ep, "model_state_dict": inf_state},
                    out_dir / "best_unseen.pt",
                )

        pu_log = (pure_opt_a_ep or pure_main_ep)["overall"]
        ep_rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "main_loss": float(np.mean(main_losses)),
            "aux_60d_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "option_a_loss": float(np.mean(opt_a_losses)) if opt_a_losses else 0.0,
            "unseen_attn_loss": float(np.mean(uattn_losses)) if uattn_losses else 0.0,
            "bn_recall@5": float(m_ep["bn_recall@5"]),
            "bn_recall@10": float(m_ep["bn_recall@10"]),
            "bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
            "true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
            "true_new_bn_hits@25": int(m_ep["true_new_bn_hits@25"]),
            "pure_unseen_hits@10": int(pu_log["pure_unseen_hits@10"]),
            "pure_unseen_hits@25": int(pu_log["pure_unseen_hits@25"]),
        }
        epoch_history.append(ep_rec)
        print(
            f"[v2-w2b:{arm_name}] ep={ep:02d}/{epochs} loss={ep_rec['loss']:.4f} "
            f"R@10={ep_rec['bn_recall@10']:.5f} NDCG@10={ep_rec['bn_ndcg@10']:.5f} "
            f"New@10={ep_rec['true_new_bn_hits@10']}/{tot_va_new} "
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

    selections = [("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")]
    if option_a:
        selections.append(("unseen", "best_unseen.pt"))

    for sel_name, fname in selections:
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

        if option_a and va_unseen_lg is not None and sel_name in ("ndcg", "unseen"):
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
            ev_oof["epoch"] = int(ck["epoch"])
            ev_oof["pure_unseen_head"] = pure_unseen
            ev_oof["affine_oof_calibration"] = oof_meta
            arm_res[f"optionA_oof_{sel_name}"] = ev_oof
            if sel_name == "ndcg":
                arm_res["ndcg"]["pure_unseen_head"] = pure_unseen

        if sel_name == "ndcg":
            np.save(out_dir / "val_logits_ndcg.npy", va_main_lg.float().numpy())
            if option_a and va_unseen_lg is not None:
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


def _resolve_comparison_references(
    base_name: str, seed: int, sync_dir: Optional[Path]
) -> Dict[str, Path]:
    refs: Dict[str, Path] = resolve_v2_full_reference_predictions(sync_dir)
    w1_path = EVIDENCE_ROOT / f"v2_full_experiments_seed{seed}.json"
    if sync_dir is not None and not w1_path.is_file() and (sync_dir / w1_path).is_file():
        w1_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sync_dir / w1_path, w1_path)
    if not w1_path.is_file():
        return refs
    w1 = json.loads(w1_path.read_text(encoding="utf-8"))
    p_str = w1.get("arms", {}).get(base_name, {}).get("ndcg", {}).get("predictions_npz")
    if p_str:
        p = Path(p_str)
        if not p.is_file() and sync_dir is not None and (sync_dir / p).is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sync_dir / p, p)
        if p.is_file():
            refs[base_name] = p
    return refs


def run_v2_full_wave2b(
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
    t_all = time.perf_counter()
    if base_model not in SUPPORTED_WAVE2B_BASES:
        raise ValueError(f"Unsupported Wave 2B base={base_model!r}; expected one of {SUPPORTED_WAVE2B_BASES}")

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

    tr_e5_p, va_e5_p, e5_manifest = build_or_load_extra_cand_5()
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)

    mmap_mode = None if in_memory else "r"
    if in_memory:
        print("[v2-w2b] Loading float16 cand_38 + extra_cand_5 into RAM (--in-memory)...", flush=True)
    tr_cand38_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand38_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)
    tr_extra5_mm = np.load(tr_e5_p, mmap_mode=mmap_mode)
    va_extra5_mm = np.load(va_e5_p, mmap_mode=mmap_mode)

    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)

    dataset_identity = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
    )
    all_specs = build_wave2b_arm_specs(base_model)
    available = {a["name"] for a in all_specs}
    if arms is not None and (not arms or set(arms) - available):
        raise ValueError(f"Unsupported Wave 2B arms={arms!r}. Available: {sorted(available)}")
    selected_specs = [a for a in all_specs if arms is None or a["name"] in arms]

    exp_root = ARTIFACT_ROOT / f"wave2b_{base_model}-seed{seed}"
    arm_results: Dict[str, Any] = {}
    for arm_cfg in selected_specs:
        arm_results[arm_cfg["name"]] = train_and_eval_wave2b_arm(
            arm_cfg,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            tr_cand38_mm=tr_cand38_mm,
            va_cand38_mm=va_cand38_mm,
            tr_extra5_mm=tr_extra5_mm,
            va_extra5_mm=va_extra5_mm,
            extra5_identity=input_digest((e5_manifest["identity"], e5_manifest["output_sha256"])),
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
    refs = _resolve_comparison_references(base_model, seed, sync_dir)
    expected_refs = {"v2_full_ref_primary", base_model}
    missing_refs = sorted(expected_refs - refs.keys())
    if missing_refs:
        print(f"Reference prediction files unavailable for paired comparisons: {missing_refs}", flush=True)
    for rname, rpath in refs.items():
        for aname, ares in arm_results.items():
            if rname == "v2_full_ref_primary":
                comparisons[f"{aname}_disc_vs_{rname}"] = paired_base_name_bootstrap_ci(
                    rpath, Path(ares["discovery"]["predictions_npz"]), seed=seed
                )
                comparisons[f"{aname}_ndcg_vs_{rname}"] = paired_base_name_bootstrap_ci(
                    rpath, Path(ares["ndcg"]["predictions_npz"]), seed=seed
                )
                if "optionA_oof_ndcg" in ares:
                    comparisons[f"{aname}_optionA_oof_ndcg_vs_{rname}"] = paired_base_name_bootstrap_ci(
                        rpath, Path(ares["optionA_oof_ndcg"]["predictions_npz"]), seed=seed
                    )
                if "optionA_oof_unseen" in ares:
                    comparisons[f"{aname}_optionA_oof_unseen_vs_{rname}"] = paired_base_name_bootstrap_ci(
                        rpath, Path(ares["optionA_oof_unseen"]["predictions_npz"]), seed=seed
                    )
            else:
                suffix = "" if rname.endswith("_ndcg") else "_ndcg"
                comparisons[f"{aname}_main_vs_{rname}{suffix}"] = paired_base_name_bootstrap_ci(
                    rpath, Path(ares["ndcg"]["predictions_npz"]), seed=seed
                )
                if "optionA_oof_ndcg" in ares:
                    comparisons[f"{aname}_optionA_oof_vs_{rname}{suffix}"] = paired_base_name_bootstrap_ci(
                        rpath, Path(ares["optionA_oof_ndcg"]["predictions_npz"]), seed=seed
                    )
    for aname, ares in arm_results.items():
        if "optionA_oof_ndcg" in ares:
            comparisons[f"{aname}_optionA_oof_vs_own_main"] = paired_base_name_bootstrap_ci(
                Path(ares["ndcg"]["predictions_npz"]),
                Path(ares["optionA_oof_ndcg"]["predictions_npz"]),
                seed=seed,
            )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_ROOT / f"v2_full_wave2b_{base_model}_seed{seed}.json"
    if len(selected_specs) != len(all_specs):
        summary_path = summary_path.with_stem(
            summary_path.stem + "_" + "_".join(a["name"] for a in selected_specs)
        )
    summary = {
        "experiment": "v2_full_wave2b_feature_optionA_pureQ",
        "dataset": "v2_full",
        "reference_id": ACTIVE_V2_FULL_REFERENCE_ID,
        "feature_schema": "cand_42+CUST_7+SEQ_5",
        "base_model": base_model,
        "extra_cand_5_identity": str(e5_manifest["identity"]),
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "amp_bf16": bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()),
        "arms": arm_results,
        "missing_comparison_references": missing_refs,
        "paired_bootstrap_95ci": comparisons,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(summary, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (summary_path,))
    print(f"\nSaved Wave 2B summary ({base_model}) -> {summary_path}", flush=True)

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
        choices=SUPPORTED_WAVE2B_BASES,
        help="Active Wave 2B E2E base model (arm4_e2e_unseen_attn_aux).",
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
    run_v2_full_wave2b(
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
