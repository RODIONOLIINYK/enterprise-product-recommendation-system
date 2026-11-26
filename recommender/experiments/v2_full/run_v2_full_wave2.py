"""Run controlled single-factor Wave 2A experiments on `v2_full` built on the canonical reference.

Built exclusively on the promoted canonical `v2_full` reference architecture:
`arm4_e2e_unseen_attn_aux / w2c_exp1_clean_sep_w10` (`Wave2BE2ERanker`, single-stage E2E,
`cand_42 + CUST_7 + SEQ_5`, Pure-Q `Q = v_bn`, Full-42 cross-attention bias, clean conflict-free
separated Main/Option A targets, `OptionAUnseenHead` on `cand_37` with **0% tabular dropout**,
`unseen_head_weight = 1.0`, and 2-param customer-grouped OOF affine blend, zero held-out Test
>= 2026-07-01 access, paired 95% customer-clustered bootstrap CIs over 16,623 validation baskets
vs. `v2_full_reference.json`):

  Model-Size Scaling Ladder (`--preset size` runs these 8 arms):
    1. `w2_2mem_blocks`:
         2x stacked `[Self-Attention -> FFN(384)]` memory blocks (`memory_layers=2`), keeping
         `d_model=128` and `(384, 192, 96)` scoring MLPs fixed.
    2. `w2_wider_mlp`:
         Wider MLPs (`memory_ffn_dim=512` + `(512, 256, 128)` scoring MLPs on Main & Option A),
         keeping `memory_layers=1` and `d_model=128` fixed.
    3. `w2_2mem_wider_mlp`:
         Both 2x `[Self-Attention -> FFN(512)]` memory blocks AND wider `(512, 256, 128)` scoring
         MLPs (`d_model=128`).
    4. `w2_wider_towers`:
         Wider Product & Customer towers (`product_tower_hidden_dim=256`, `cust_tower_hidden_dim=256`,
         `attr_emb_dim=32`) while keeping `d_model=128`, `memory_layers=1`, and `(384, 192, 96)` MLPs
         fixed (adds ~0 ms per-request CPU latency since `ProductTower` is cached offline).
    5. `w2_dim192`:
         Bigger product & customer representation dimensionality (`d_model=192`, `num_heads=8`
         [24D/head], `memory_ffn_dim=576`), keeping `memory_layers=1` and `(384, 192, 96)` MLPs.
    6. `w2_dim192_2mem_wider_mlp`:
         Compound `d_model=192` scaling: `d_model=192`, `num_heads=8` [24D/head], `memory_layers=2`,
         `memory_ffn_dim=576`, `mlp_widths=(512, 256, 128)`, `attr_emb_dim=24`.
    7. `w2_dim256`:
         Isolated `d_model=256` width scaling: `d_model=256`, `num_heads=8` [32D/head],
         `memory_layers=1`, `memory_ffn_dim=768`, `mlp_widths=(512, 256, 128)`, `attr_emb_dim=32`.
    8. `w2_dim256_2mem_4mlp`:
         Large flagship capacity arm: `d_model=256`, `num_heads=8` [32D/head], `memory_layers=2`,
         `memory_ffn_dim=1024`, 4-block `mlp_widths=(768, 512, 256, 128)`,
         `product_tower_hidden_dim=384`, `cust_tower_hidden_dim=384`, `attr_emb_dim=32`.

  Longer Purchase History & Recency Sample Weighting:
    9. `w2_hist_len64`:
         `seq_len=64` (all other architecture/loss settings identical to the promoted `w10` reference).
   10. `w2_recency_hl365`:
         Exponential recency basket weighting (`half_life_days=365.0`, `min_weight=0.1`,
         mean-normalized to 1.0 across train baskets) applied to separated `L_main`, `L_60d`,
         and Option A unseen losses (`L_unseen_next`, `L_unseen_60d`).

  Separate Bigger-Model Synthetic Unseen Augmentation Experiment:
   11. `w2_2mem_wider_mlp_syn7030`:
         Exact `w2_2mem_wider_mlp` architecture trained with offline exact 70/30 masked-b* + FO
         synthetic unseen augmentation (`synthetic_unseen.py`, `w_optA=1.0`, `skip_main_head=True`),
         compared directly against both `w10` and `w2_2mem_wider_mlp`.

  Conditional Wave 3 (OFF by default):
   12. `w2_hist_len96` (`--include-hist96`):
         `seq_len=96` — deferred to Wave 3 and run ONLY if `w2_hist_len64` improves over the
         32-event `w2c_exp1_clean_sep_w10` reference control.
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
from recommender.experiments.roadmap_v2.dated_baskets import validate_dated_split
from recommender.experiments.roadmap_v2.run_stage5 import _verify_reference_files
from recommender.experiments.roadmap_v2.run_stage5_ablations import _evaluate_with_cohorts_and_sku
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    evaluate_base_name_predictions,
    paired_base_name_bootstrap_ci,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.experiments.static_enrichment.metadata import load_enriched_catalogue
from recommender.shared_data import atomic_torch_save

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
    V2_FULL_REFERENCE_PATH,
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
from .run_v2_full_wave2b import (
    _completed_w2b_arm,
    _expected_outputs_for_arm,
    iter_paired_training_batches,
)
from .synthetic_unseen import (
    CUST_7_FO_DEFAULT,
    SYNTHETIC_RETURNING_FRACTION,
    assemble_synthetic_returning_cand_42,
    build_or_load_synthetic_unseen_cache,
    build_synthetic_fo_cand_42,
)
from .train_v2_full_baseline import (
    rank_eligible_formulations,
    sync_outputs_to_dir,
)
from .wave2_utils import (
    WeightedRankerLoss,
    compute_recency_sample_weights,
    ensure_v2_sequences_for_length,
    is_tpu_device,
    resolve_wave2_device,
    supports_native_bf16,
    tpu_mark_step,
    tpu_optimizer_step,
    wave2_autocast_context,
)
from .wave2b_models import (
    Wave2BE2ERanker,
    evaluate_pure_unseen_predictions,
    fit_oof_affine_unseen_blend,
    strip_wave2b_training_aux,
)

ACTIVE_E2E_BASE_CONFIG: Dict[str, Any] = {
    "name": ACTIVE_V2_FULL_BASE_MODEL,
    "mode": "e2e",
    "pure_v_query": True,
    "num_pairwise_bias_features": NUM_CAND_42_FEATURES,
    "option_a_unseen_head": True,
    "unseen_dropout_cold_prob": 0.0,
    "separated_main_targets": True,
    "unseen_head_weight": 1.0,
    "unseen_attn_aux_enabled": False,
    "unseen_attn_beta": 0.0,
}

WAVE2A_SCHEMA_VERSION = "cand42_clean_sep_w10_optionA_pureQ_full42bias_cust7_seq5_v4"
REQUIRED_WAVE2A_REF_KEYS: Tuple[str, ...] = ("v2_full_ref_primary", "v2_full_ref_cal_ndcg")
SIZE_SCALING_ARM_NAMES: Tuple[str, ...] = (
    "w2_2mem_blocks",
    "w2_wider_mlp",
    "w2_2mem_wider_mlp",
    "w2_wider_towers",
    "w2_dim192",
    "w2_dim192_2mem_wider_mlp",
    "w2_dim256",
    "w2_dim256_2mem_4mlp",
)
FINAL_2MEM_ARM_NAMES: Tuple[str, ...] = (
    "w2_2mem_hist64",
    "w2_2mem_recency_hl365",
    "w2_2mem_hist64_recency_hl365",
)


def _wave2a_code_digest() -> str:
    """Digest of Wave 2A runner, weighted-loss/sequence utils, synthetic builder, and model definitions."""
    here = Path(__file__).resolve().parent
    return input_digest(
        tuple(
            (name, file_digest(here / name))
            for name in ("run_v2_full_wave2.py", "wave2_utils.py", "wave2b_models.py", "synthetic_unseen.py")
        )
    )


def _expected_wave2a_outputs(option_a: bool) -> Tuple[str, ...]:
    base = _expected_outputs_for_arm(option_a)
    if option_a:
        return (*base, "predictions_optionA_oof_discovery.npz")
    return base


def _selection_key(selection: Mapping[str, Any]) -> Tuple[float, int, float]:
    metrics = selection["bn_metrics"]
    return (float(metrics["bn_ndcg@10"]), int(metrics["true_new_bn_hits@10"]), float(metrics["bn_recall@10"]))


def _discovery_key(selection: Mapping[str, Any]) -> Tuple[int, float, float]:
    metrics = selection["bn_metrics"]
    return (int(metrics["true_new_bn_hits@10"]), float(metrics["bn_ndcg@10"]), float(metrics["bn_recall@10"]))


def build_wave2_arm_specs(
    base_cfg: Mapping[str, Any] = ACTIVE_E2E_BASE_CONFIG,
    *,
    include_hist96: bool = False,
) -> Tuple[Dict[str, Any], ...]:
    """Construct the controlled single-factor Wave 2A E2E arm matrix on top of the promoted `w10` reference."""
    if base_cfg.get("mode", "e2e") != "e2e":
        raise ValueError(f"Wave 2A runs exclusively on E2E arm4; got mode={base_cfg.get('mode')!r}")
    defaults = {
        **dict(base_cfg),
        "num_candidate_features": NUM_CAND_42_FEATURES,
        "num_customer_features": NUM_CUST_7_FEATURES,
        "active_seq_stat_indices": list(SEQ_5_ACTIVE_INDICES),
        "num_pairwise_bias_features": int(base_cfg.get("num_pairwise_bias_features", NUM_CAND_42_FEATURES)),
        "pure_v_query": bool(base_cfg.get("pure_v_query", True)),
        "option_a_unseen_head": bool(base_cfg.get("option_a_unseen_head", True)),
        "unseen_dropout_cold_prob": float(base_cfg.get("unseen_dropout_cold_prob", 0.0)),
        "separated_main_targets": bool(base_cfg.get("separated_main_targets", True)),
        "unseen_head_weight": float(base_cfg.get("unseen_head_weight", 1.0)),
        "unseen_attn_aux_enabled": bool(base_cfg.get("unseen_attn_aux_enabled", False)),
        "unseen_attn_beta": float(base_cfg.get("unseen_attn_beta", 0.0)),
        "d_model": 128,
        "num_heads": 8,
        "memory_ffn_dim": 384,
        "memory_layers": 1,
        "mlp_widths": (384, 192, 96),
        "product_tower_hidden_dim": None,
        "cust_tower_hidden_dim": None,
        "attr_emb_dim": 16,
        "dropout": 0.12,
        "seq_len": 32,
        "recency_half_life_days": None,
        "recency_min_weight": 0.1,
        "synthetic_unseen_aug": False,
        "synthetic_returning_fraction": SYNTHETIC_RETURNING_FRACTION,
    }
    specs: List[Dict[str, Any]] = [
        {
            **defaults,
            "name": "w2_2mem_blocks",
            "memory_layers": 2,
            "memory_ffn_dim": 384,
            "mlp_widths": (384, 192, 96),
        },
        {
            **defaults,
            "name": "w2_wider_mlp",
            "memory_layers": 1,
            "memory_ffn_dim": 512,
            "mlp_widths": (512, 256, 128),
        },
        {
            **defaults,
            "name": "w2_2mem_wider_mlp",
            "memory_layers": 2,
            "memory_ffn_dim": 512,
            "mlp_widths": (512, 256, 128),
        },
        {
            **defaults,
            "name": "w2_wider_towers",
            "d_model": 128,
            "num_heads": 8,
            "memory_layers": 1,
            "memory_ffn_dim": 384,
            "mlp_widths": (384, 192, 96),
            "product_tower_hidden_dim": 256,
            "cust_tower_hidden_dim": 256,
            "attr_emb_dim": 32,
        },
        {
            **defaults,
            "name": "w2_dim192",
            "d_model": 192,
            "num_heads": 8,
            "memory_ffn_dim": 576,
            "memory_layers": 1,
            "mlp_widths": (384, 192, 96),
        },
        {
            **defaults,
            "name": "w2_dim192_2mem_wider_mlp",
            "d_model": 192,
            "num_heads": 8,
            "memory_ffn_dim": 576,
            "memory_layers": 2,
            "mlp_widths": (512, 256, 128),
            "attr_emb_dim": 24,
        },
        {
            **defaults,
            "name": "w2_dim256",
            "d_model": 256,
            "num_heads": 8,
            "memory_ffn_dim": 768,
            "memory_layers": 1,
            "mlp_widths": (512, 256, 128),
            "attr_emb_dim": 32,
        },
        {
            **defaults,
            "name": "w2_dim256_2mem_4mlp",
            "d_model": 256,
            "num_heads": 8,
            "memory_ffn_dim": 1024,
            "memory_layers": 2,
            "mlp_widths": (768, 512, 256, 128),
            "product_tower_hidden_dim": 384,
            "cust_tower_hidden_dim": 384,
            "attr_emb_dim": 32,
        },
        {
            **defaults,
            "name": "w2_hist_len64",
            "seq_len": 64,
        },
        {
            **defaults,
            "name": "w2_recency_hl365",
            "recency_half_life_days": 365.0,
            "recency_min_weight": 0.1,
        },
        {
            **defaults,
            "name": "w2_2mem_wider_mlp_syn7030",
            "memory_layers": 2,
            "memory_ffn_dim": 512,
            "mlp_widths": (512, 256, 128),
            "synthetic_unseen_aug": True,
        },
        {
            **defaults,
            "name": "w2_2mem_hist64",
            "memory_layers": 2,
            "memory_ffn_dim": 384,
            "mlp_widths": (384, 192, 96),
            "seq_len": 64,
        },
        {
            **defaults,
            "name": "w2_2mem_recency_hl365",
            "memory_layers": 2,
            "memory_ffn_dim": 384,
            "mlp_widths": (384, 192, 96),
            "recency_half_life_days": 365.0,
            "recency_min_weight": 0.1,
        },
        {
            **defaults,
            "name": "w2_2mem_hist64_recency_hl365",
            "memory_layers": 2,
            "memory_ffn_dim": 384,
            "mlp_widths": (384, 192, 96),
            "seq_len": 64,
            "recency_half_life_days": 365.0,
            "recency_min_weight": 0.1,
        },
    ]
    if include_hist96:
        specs.append(
            {
                **defaults,
                "name": "w2_hist_len96",
                "seq_len": 96,
            }
        )
    return tuple(specs)


def compute_wave2_weighted_separated_losses(
    *,
    main_logits: torch.Tensor,
    main_aux_60d_logits: Optional[torch.Tensor],
    gen_unseen_logits: Optional[torch.Tensor],
    gen_unseen_aux_60d_logits: Optional[torch.Tensor],
    allowed_bn: torch.Tensor,
    unseen_bn: torch.Tensor,
    target_bn: torch.Tensor,
    aux_60d_bn: torch.Tensor,
    full_window: torch.Tensor,
    first_observed: torch.Tensor,
    main_loss_fn: WeightedRankerLoss,
    main_aux_loss_fn: WeightedRankerLoss,
    unseen_next_loss_fn: WeightedRankerLoss,
    unseen_60d_loss_fn: WeightedRankerLoss,
    unseen_head_weight: float = 1.0,
    separated_main_targets: bool = True,
    sample_weight: Optional[torch.Tensor] = None,
    static_shapes: Optional[bool] = None,
    valid_batch_mask: Optional[torch.Tensor] = None,
    syn_unseen_logits: Optional[torch.Tensor] = None,
    syn_allowed_bn: Optional[torch.Tensor] = None,
    syn_unseen_bn: Optional[torch.Tensor] = None,
    syn_target_bn: Optional[torch.Tensor] = None,
    syn_valid_batch_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute per-basket weighted Main + Option A losses (conflict-free separated when enabled).

    When `separated_main_targets=True` (default, matching promoted `w2c_exp1_clean_sep_w10`):
      - Main Head: returning baskets only (`~first_observed`), `main_mask = allowed & ~(target & unseen)`,
        `y_main = target & ~unseen` (0-repeat returning baskets still receive BCE as all-zero negatives).
        Same conflict-free mask for `0.3 * L_aux60d`.
      - Option A Head: `unseen_m = allowed & unseen`, real unseen positives only
        (`L_unseen_next + 0.3 * L_unseen_60d`), weighted by `unseen_head_weight`.
      - When `sample_weight` is provided (`w2_recency_hl365`), the exact per-basket weight slice is
        passed into every `WeightedRankerLoss` call.
      - When `static_shapes=True` (automatic on TPU/XLA), avoids all dynamic boolean row slicing
        and device-to-host `.any()` syncs by zeroing excluded rows via `sample_weight` and
        `candidate_mask` — algebraically identical to row slicing while compiling a single XLA graph.
    """
    dev = main_logits.device
    use_static = is_tpu_device(dev) if static_shapes is None else bool(static_shapes)
    allowed = allowed_bn.bool()
    unseen = unseen_bn.bool()
    aux_all = full_window.bool()
    returning = ~first_observed.bool()

    if valid_batch_mask is not None:
        vbm = valid_batch_mask.bool()
        allowed = allowed & vbm[:, None]
        returning = returning & vbm
        aux_all = aux_all & vbm
    target = target_bn.bool() & allowed
    aux60 = aux_60d_bn.bool() & allowed

    if use_static:
        b_size = main_logits.size(0)
        base_w = (
            torch.ones((b_size,), dtype=torch.float32, device=dev)
            if sample_weight is None
            else sample_weight.to(device=dev, dtype=torch.float32)
        )
        if valid_batch_mask is not None:
            base_w = torch.where(valid_batch_mask.bool(), base_w, torch.zeros_like(base_w))
        zeros_w = torch.zeros_like(base_w)

        if separated_main_targets:
            main_mask = allowed & (~(target & unseen)) & returning[:, None]
            y_main = (target & (~unseen)).float() * main_mask.float()
            w_main = torch.where(returning, base_w, zeros_w)
            l_main = main_loss_fn(
                main_logits.float(), y_main, candidate_mask=main_mask, sample_weight=w_main
            )["loss"]
            total_loss = l_main

            l_aux: Optional[torch.Tensor] = None
            if main_aux_60d_logits is not None:
                aux_ret = aux_all & returning
                main_aux_mask = allowed & (~(aux60 & unseen)) & aux_ret[:, None]
                y_main_aux = (aux60 & (~unseen)).float() * main_aux_mask.float()
                w_aux = torch.where(aux_ret, base_w, zeros_w)
                l_aux = main_aux_loss_fn(
                    main_aux_60d_logits.float(), y_main_aux, candidate_mask=main_aux_mask, sample_weight=w_aux
                )["loss"]
                total_loss = total_loss + 0.3 * l_aux
        else:
            y = target.float() * allowed.float()
            l_main = main_loss_fn(
                main_logits.float(), y, candidate_mask=allowed, sample_weight=base_w
            )["loss"]
            total_loss = l_main
            l_aux = None
            if main_aux_60d_logits is not None:
                aux_mask = allowed & aux_all[:, None]
                y_aux = aux60.float() * aux_mask.float()
                w_aux = torch.where(aux_all, base_w, zeros_w)
                l_aux = main_aux_loss_fn(
                    main_aux_60d_logits.float(), y_aux, candidate_mask=aux_mask, sample_weight=w_aux
                )["loss"]
                total_loss = total_loss + 0.3 * l_aux

        l_opt_a: Optional[torch.Tensor] = None
        l_u_next: Optional[torch.Tensor] = None
        l_u_60d: Optional[torch.Tensor] = None
        if gen_unseen_logits is not None:
            unseen_m = unseen & allowed
            y_u_next = (target & unseen_m).float()
            valid_un = y_u_next.sum(dim=-1) > 0
            u_next_mask = unseen_m & valid_un[:, None]
            w_un = torch.where(valid_un, base_w, zeros_w)
            u_lg_in = gen_unseen_logits.float()
            u_y_in = y_u_next * u_next_mask.float()
            u_m_in = u_next_mask
            u_w_in = w_un
            if syn_unseen_logits is not None:
                assert syn_allowed_bn is not None and syn_unseen_bn is not None and syn_target_bn is not None
                s_alw = syn_allowed_bn.bool()
                if syn_valid_batch_mask is not None:
                    s_alw = s_alw & syn_valid_batch_mask.bool()[:, None]
                s_m = syn_unseen_bn.bool() & s_alw
                s_y = (syn_target_bn.bool() & s_m).float()
                s_valid = s_y.sum(dim=-1) > 0
                if syn_valid_batch_mask is not None:
                    s_valid = s_valid & syn_valid_batch_mask.bool()
                s_mask_in = s_m & s_valid[:, None]
                s_w_in = s_valid.to(dtype=torch.float32)
                u_lg_in = torch.cat([u_lg_in, syn_unseen_logits.float()], dim=0)
                u_y_in = torch.cat([u_y_in, s_y * s_mask_in.float()], dim=0)
                u_m_in = torch.cat([u_m_in, s_mask_in], dim=0)
                u_w_in = torch.cat([u_w_in, s_w_in], dim=0)
            l_u_next = unseen_next_loss_fn(
                u_lg_in, u_y_in, candidate_mask=u_m_in, sample_weight=u_w_in
            )["loss"]
            l_opt_a = l_u_next

            if gen_unseen_aux_60d_logits is not None:
                y_u60 = (aux60 & unseen_m).float()
                valid_u60 = aux_all & (y_u60.sum(dim=-1) > 0)
                u60_mask = unseen_m & valid_u60[:, None]
                w_u60 = torch.where(valid_u60, base_w, zeros_w)
                l_u_60d = unseen_60d_loss_fn(
                    gen_unseen_aux_60d_logits.float(),
                    y_u60 * u60_mask.float(),
                    candidate_mask=u60_mask,
                    sample_weight=w_u60,
                )["loss"]
                l_opt_a = l_opt_a + 0.3 * l_u_60d

            total_loss = total_loss + float(unseen_head_weight) * l_opt_a

        return {
            "loss": total_loss,
            "main_loss": l_main,
            "aux_60d_loss": l_aux,
            "option_a_loss": l_opt_a,
            "unseen_next_loss": l_u_next,
            "unseen_60d_loss": l_u_60d,
        }

    l_main = None
    l_aux = None
    total_loss: Optional[torch.Tensor] = None

    if separated_main_targets:
        if bool(returning.any()):
            a_ret = allowed[returning]
            u_ret = unseen[returning]
            t_ret = target[returning]
            main_mask = a_ret & (~(t_ret & u_ret))
            y_main = (t_ret & (~u_ret)).float() * main_mask.float()
            main_lg = main_logits[returning].float().masked_fill(~main_mask, -torch.inf)
            w_ret = sample_weight[returning] if sample_weight is not None else None
            l_main = main_loss_fn(main_lg, y_main, candidate_mask=main_mask, sample_weight=w_ret)["loss"]
            total_loss = l_main

        aux_ret = aux_all & returning
        if bool(aux_ret.any()) and main_aux_60d_logits is not None:
            a_aux = allowed[aux_ret]
            u_aux = unseen[aux_ret]
            t60 = aux60[aux_ret]
            main_aux_mask = a_aux & (~(t60 & u_aux))
            y_main_aux = (t60 & (~u_aux)).float() * main_aux_mask.float()
            aux_lg = main_aux_60d_logits[aux_ret].float().masked_fill(~main_aux_mask, -torch.inf)
            w_aux = sample_weight[aux_ret] if sample_weight is not None else None
            l_aux = main_aux_loss_fn(aux_lg, y_main_aux, candidate_mask=main_aux_mask, sample_weight=w_aux)["loss"]
            total_loss = (0.3 * l_aux) if total_loss is None else (total_loss + 0.3 * l_aux)
    else:
        y = target.float() * allowed.float()
        w_eff = (
            torch.where(valid_batch_mask.bool(), sample_weight, torch.zeros_like(sample_weight))
            if (valid_batch_mask is not None and sample_weight is not None)
            else (valid_batch_mask.float() if valid_batch_mask is not None else sample_weight)
        )
        l_main = main_loss_fn(main_logits.float(), y, candidate_mask=allowed, sample_weight=w_eff)["loss"]
        total_loss = l_main
        if bool(aux_all.any()) and main_aux_60d_logits is not None:
            y_aux = aux60[aux_all].float() * allowed[aux_all].float()
            w_aux = sample_weight[aux_all] if sample_weight is not None else None
            l_aux = main_aux_loss_fn(
                main_aux_60d_logits[aux_all].float(), y_aux, candidate_mask=allowed[aux_all], sample_weight=w_aux
            )["loss"]
            total_loss = total_loss + 0.3 * l_aux

    l_opt_a = None
    l_u_next = None
    l_u_60d = None

    if gen_unseen_logits is not None:
        unseen_m = unseen & allowed
        y_u_next = (target & unseen_m).float()
        valid_un = y_u_next.sum(dim=-1) > 0
        u_lg_parts: List[torch.Tensor] = []
        u_y_parts: List[torch.Tensor] = []
        u_m_parts: List[torch.Tensor] = []
        u_w_parts: List[torch.Tensor] = []
        if bool(valid_un.any()):
            u_lg_parts.append(gen_unseen_logits[valid_un].float().masked_fill(~unseen_m[valid_un], -torch.inf))
            u_y_parts.append(y_u_next[valid_un])
            u_m_parts.append(unseen_m[valid_un])
            if sample_weight is not None:
                u_w_parts.append(sample_weight[valid_un].to(dtype=torch.float32))
            else:
                u_w_parts.append(torch.ones((int(valid_un.sum().item()),), dtype=torch.float32, device=dev))

        if syn_unseen_logits is not None and syn_unseen_logits.size(0) > 0:
            assert syn_allowed_bn is not None and syn_unseen_bn is not None and syn_target_bn is not None
            s_alw = syn_allowed_bn.bool()
            if syn_valid_batch_mask is not None:
                s_alw = s_alw & syn_valid_batch_mask.bool()[:, None]
            s_m = syn_unseen_bn.bool() & s_alw
            s_y = (syn_target_bn.bool() & s_m).float()
            s_valid = s_y.sum(dim=-1) > 0
            if bool(s_valid.any()):
                u_lg_parts.append(syn_unseen_logits[s_valid].float().masked_fill(~s_m[s_valid], -torch.inf))
                u_y_parts.append(s_y[s_valid])
                u_m_parts.append(s_m[s_valid])
                u_w_parts.append(torch.ones((int(s_valid.sum().item()),), dtype=torch.float32, device=dev))

        if u_lg_parts:
            comb_w = torch.cat(u_w_parts, dim=0) if (sample_weight is not None or len(u_w_parts) > 1) else None
            l_u_next = unseen_next_loss_fn(
                torch.cat(u_lg_parts, dim=0),
                torch.cat(u_y_parts, dim=0),
                candidate_mask=torch.cat(u_m_parts, dim=0),
                sample_weight=comb_w,
            )["loss"]
            l_opt_a = l_u_next

        if bool(aux_all.any()) and gen_unseen_aux_60d_logits is not None:
            y_u60 = (aux60 & unseen_m).float()
            valid_u60 = aux_all & (y_u60.sum(dim=-1) > 0)
            if bool(valid_u60.any()):
                u60_lg = gen_unseen_aux_60d_logits[valid_u60].float().masked_fill(~unseen_m[valid_u60], -torch.inf)
                w_u60 = sample_weight[valid_u60] if sample_weight is not None else None
                l_u_60d = unseen_60d_loss_fn(
                    u60_lg, y_u60[valid_u60], candidate_mask=unseen_m[valid_u60], sample_weight=w_u60
                )["loss"]
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


def _build_wave2_batch(
    *,
    split_meta: Mapping[str, Any],
    cand42_np: np.ndarray,
    rows: torch.Tensor,
    dev: torch.device,
) -> Dict[str, Any]:
    cust_raw = split_meta["tabular_features"][rows]
    cust_7 = prune_cust_7(cust_raw) if cust_raw.shape[-1] == 11 else cust_raw
    return {
        "company_text_embedding": split_meta["text_embedding"][rows].to(dev, dtype=torch.float32),
        "city_id": split_meta["city_id"][rows].to(dev),
        "region_id": split_meta["region_id"][rows].to(dev),
        "industry_id": split_meta["industry_id"][rows].to(dev),
        "customer_features": cust_7.to(dev, dtype=torch.float32),
        "sequence": split_meta["sequence"][rows].to(dev),
        "attention_mask": split_meta["attention_mask"][rows].to(dev),
        "seq_product_ids": split_meta["seq_product_ids"][rows].to(dev),
        "seq_purchase_dates": split_meta["seq_purchase_dates"][rows].to(dev),
        "prediction_day": split_meta["prediction_days"][rows].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand42_np, dtype=np.float32)).to(dev),
        "candidate_mask": split_meta["allowed_bn"][rows].to(dev),
    }


def _pad_batch_for_static_tpu(
    rows: torch.Tensor,
    cand42_np: np.ndarray,
    batch_size: int,
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """Pad a short tail batch up to `batch_size` so TPU/XLA compiles one fixed graph per arm."""
    real_b = int(len(rows))
    if real_b >= int(batch_size):
        return rows, cand42_np, real_b
    pad_n = int(batch_size) - real_b
    pad_rows = rows[:1].expand(pad_n)
    rows_padded = torch.cat([rows, pad_rows], dim=0)
    cand_padded = np.concatenate([cand42_np, np.repeat(cand42_np[:1], pad_n, axis=0)], axis=0)
    return rows_padded, cand_padded, real_b


@torch.no_grad()
def _score_val_wave2(
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
    on_tpu = is_tpu_device(dev)
    v_bn = model.precompute_product_vectors(prod_inputs_dev)
    if on_tpu:
        tpu_mark_step(dev)

    n_rows = len(va_meta["customer_ids"])
    main_chunks: List[torch.Tensor] = []
    unseen_chunks: List[torch.Tensor] = []
    has_unseen = False
    for s0 in range(0, n_rows, 2048):
        s1 = min(n_rows, s0 + 2048)
        buf_cand = assemble_cand_42(va_cand38_mm[s0:s1], va_extra5_mm[s0:s1])
        dev_main_buf: List[Tuple[torch.Tensor, int]] = []
        dev_unseen_buf: List[Tuple[torch.Tensor, int]] = []
        for off in range(0, s1 - s0, batch_size):
            rows = torch.arange(s0 + off, min(s1, s0 + off + batch_size), dtype=torch.long)
            cand_slice = buf_cand[off : off + len(rows)]
            real_b = int(len(rows))
            if on_tpu and real_b < int(batch_size):
                rows, cand_slice, real_b = _pad_batch_for_static_tpu(rows, cand_slice, int(batch_size))
            inp = _build_wave2_batch(
                split_meta=va_meta,
                cand42_np=cand_slice,
                rows=rows,
                dev=dev,
            )
            res = model(**inp, precomputed_v_bn=v_bn, return_aux=False)
            if on_tpu:
                dev_main_buf.append((res["output"].logits, real_b))
                if res["unseen_logits"] is not None:
                    has_unseen = True
                    dev_unseen_buf.append((res["unseen_logits"], real_b))
            else:
                main_chunks.append(res["output"].logits[:real_b].cpu())
                if res["unseen_logits"] is not None:
                    has_unseen = True
                    unseen_chunks.append(res["unseen_logits"][:real_b].cpu())
        if on_tpu:
            tpu_mark_step(dev)
            for lg, rb in dev_main_buf:
                main_chunks.append(lg[:rb].cpu())
            for lg, rb in dev_unseen_buf:
                unseen_chunks.append(lg[:rb].cpu())
        del buf_cand
    main_all = torch.cat(main_chunks, dim=0)
    unseen_all = torch.cat(unseen_chunks, dim=0) if has_unseen else None
    return main_all, unseen_all, v_bn.cpu()


def build_synthetic_unseen_batch(
    syn_indices: np.ndarray,
    *,
    tr_meta: Mapping[str, Any],
    tr_cand38_mm: np.ndarray,
    tr_extra5_mm: np.ndarray,
    syn_meta: Mapping[str, Any],
    syn_ret_cand27_mm: np.ndarray,
    dev: torch.device,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble a mixed batch of 70% returning masked-b* + 30% wiped-history FO synthetic rows."""
    n_ret = int(len(syn_meta["ret_source_rows"]))
    n_b = len(syn_indices)
    seq_len = int(syn_meta["ret_sequence"].shape[1])
    n_bn = int(tr_meta["allowed_bn"].shape[1])

    is_ret = syn_indices < n_ret
    ret_pos = np.where(is_ret)[0]
    fo_pos = np.where(~is_ret)[0]
    ret_idx = syn_indices[ret_pos]
    fo_idx = syn_indices[fo_pos] - n_ret

    src_rows = np.empty(n_b, dtype=np.int64)
    if len(ret_pos):
        src_rows[ret_pos] = syn_meta["ret_source_rows"][ret_idx].numpy()
    if len(fo_pos):
        src_rows[fo_pos] = syn_meta["fo_source_rows"][fo_idx].numpy()

    base_cand42 = assemble_cand_42(tr_cand38_mm[src_rows], tr_extra5_mm[src_rows])
    cand42_out = np.empty((n_b, n_bn, NUM_CAND_42_FEATURES), dtype=np.float32)
    cust7_out = torch.zeros((n_b, NUM_CUST_7_FEATURES), dtype=torch.float32)
    seq_out = torch.zeros((n_b, seq_len, 9), dtype=torch.float32)
    am_out = torch.zeros((n_b, seq_len), dtype=torch.long)
    spid_out = torch.zeros((n_b, seq_len), dtype=torch.long)
    sdt_out = torch.zeros((n_b, seq_len), dtype=torch.float32)
    unseen_out = torch.zeros((n_b, n_bn), dtype=torch.bool)
    target_out = torch.zeros((n_b, n_bn), dtype=torch.bool)

    src_t = torch.from_numpy(src_rows)
    allowed_out = tr_meta["allowed_bn"][src_t].bool()

    if len(ret_pos):
        rp_t = torch.from_numpy(ret_pos)
        ri_t = torch.from_numpy(ret_idx)
        cand42_out[ret_pos] = assemble_synthetic_returning_cand_42(
            base_cand42[ret_pos], syn_ret_cand27_mm[ret_idx]
        )
        cust7_out[rp_t] = syn_meta["ret_cust_7"][ri_t]
        seq_out[rp_t] = syn_meta["ret_sequence"][ri_t]
        am_out[rp_t] = syn_meta["ret_attention_mask"][ri_t]
        spid_out[rp_t] = syn_meta["ret_seq_product_ids"][ri_t]
        sdt_out[rp_t] = syn_meta["ret_seq_purchase_dates"][ri_t]
        unseen_out[rp_t] = syn_meta["ret_unseen_bn"][ri_t].bool()
        target_out[rp_t] = syn_meta["ret_target_bn"][ri_t].bool()

    if len(fo_pos):
        fp_t = torch.from_numpy(fo_pos)
        f_src_t = src_t[fp_t]
        cand42_out[fo_pos] = build_synthetic_fo_cand_42(base_cand42[fo_pos])
        cust7_out[fp_t] = torch.tensor(CUST_7_FO_DEFAULT, dtype=torch.float32)
        unseen_out[fp_t] = allowed_out[fp_t]
        target_out[fp_t] = tr_meta["target_bn"][f_src_t].bool() & allowed_out[fp_t]

    inp = {
        "company_text_embedding": tr_meta["text_embedding"][src_t].to(dev, dtype=torch.float32),
        "city_id": tr_meta["city_id"][src_t].to(dev),
        "region_id": tr_meta["region_id"][src_t].to(dev),
        "industry_id": tr_meta["industry_id"][src_t].to(dev),
        "customer_features": cust7_out.to(dev),
        "sequence": seq_out.to(dev),
        "attention_mask": am_out.to(dev),
        "seq_product_ids": spid_out.to(dev),
        "seq_purchase_dates": sdt_out.to(dev),
        "prediction_day": tr_meta["prediction_days"][src_t].to(dev),
        "candidate_features": torch.from_numpy(np.ascontiguousarray(cand42_out, dtype=np.float32)).to(dev),
        "candidate_mask": allowed_out.to(dev),
    }
    return inp, allowed_out.to(dev), unseen_out.to(dev), target_out.to(dev)


def train_and_eval_wave2_arm(
    arm_cfg: Mapping[str, Any],
    *,
    shared_meta: Mapping[str, Any],
    tr_meta_arm: Mapping[str, Any],
    va_meta_arm: Mapping[str, Any],
    tr_cand38_mm: np.ndarray,
    va_cand38_mm: np.ndarray,
    tr_extra5_mm: np.ndarray,
    va_extra5_mm: np.ndarray,
    extra5_identity: str,
    dev_purchases: pd.DataFrame,
    catalogue: pd.DataFrame,
    dataset_identity: Optional[str] = None,
    comparison_settings: Optional[Mapping[str, Any]] = None,
    syn_meta: Optional[Mapping[str, Any]] = None,
    syn_ret_cand27_mm: Optional[np.ndarray] = None,
    syn_identity: Optional[str] = None,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    dev: torch.device,
    amp: bool = False,
    in_memory: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train, replay-verify, cohort/SKU/Pure-Unseen/OOF-evaluate, and CPU-benchmark one Wave 2A E2E arm."""
    validate_dated_split(tr_meta_arm, "train", train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2)
    validate_dated_split(va_meta_arm, "val", train_cutoff=TRAIN_CUTOFF_V2, test_start=TEST_START_V2)
    arm_name = str(arm_cfg["name"])
    num_cand_feats = int(arm_cfg.get("num_candidate_features", NUM_CAND_42_FEATURES))
    num_cust_feats = int(arm_cfg.get("num_customer_features", NUM_CUST_7_FEATURES))
    num_bias_feats = int(arm_cfg.get("num_pairwise_bias_features", NUM_CAND_42_FEATURES))
    active_seq_stats = tuple(int(i) for i in arm_cfg.get("active_seq_stat_indices", SEQ_5_ACTIVE_INDICES))
    pure_v_query = bool(arm_cfg.get("pure_v_query", True))
    option_a = bool(arm_cfg.get("option_a_unseen_head", True))
    unseen_cold_p = float(arm_cfg.get("unseen_dropout_cold_prob", 0.0))
    separated_main = bool(arm_cfg.get("separated_main_targets", True))
    unseen_head_weight = float(arm_cfg.get("unseen_head_weight", 1.0))
    unseen_attn_aux = bool(arm_cfg.get("unseen_attn_aux_enabled", False))
    unseen_beta = float(arm_cfg.get("unseen_attn_beta", 0.0))
    use_syn = bool(arm_cfg.get("synthetic_unseen_aug", False))
    if use_syn and (syn_meta is None or syn_ret_cand27_mm is None):
        raise ValueError(f"Arm {arm_name} requires syn_meta and syn_ret_cand27_mm when synthetic_unseen_aug=True.")
    d_model = int(arm_cfg["d_model"])
    num_heads = int(arm_cfg["num_heads"])
    memory_ffn_dim = int(arm_cfg["memory_ffn_dim"])
    memory_layers = int(arm_cfg["memory_layers"])
    mlp_widths = tuple(int(w) for w in arm_cfg["mlp_widths"])
    raw_p_hid = arm_cfg.get("product_tower_hidden_dim")
    product_tower_hidden_dim = int(raw_p_hid) if raw_p_hid is not None else None
    raw_c_hid = arm_cfg.get("cust_tower_hidden_dim")
    cust_tower_hidden_dim = int(raw_c_hid) if raw_c_hid is not None else None
    attr_emb_dim = int(arm_cfg.get("attr_emb_dim", 16))
    dropout = float(arm_cfg.get("dropout", 0.12))
    seq_len = int(arm_cfg["seq_len"])
    recency_hl = arm_cfg.get("recency_half_life_days")
    recency_min_w = float(arm_cfg.get("recency_min_weight", 0.1))
    expected_files = _expected_wave2a_outputs(option_a)

    out_dir.mkdir(parents=True, exist_ok=True)
    arm_summary_p = out_dir / "arm_result.json"
    if dataset_identity is None:
        dataset_identity = prepared_dataset_digest(
            shared_meta, tr_meta_arm, va_meta_arm, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
        )
        extra5_identity = input_digest((extra5_identity, tr_extra5_mm, va_extra5_mm))
    on_tpu = is_tpu_device(dev)
    use_bf16 = bool(amp and supports_native_bf16(dev))
    run_settings = {
        "schema_version": WAVE2A_SCHEMA_VERSION,
        "code_digest": _wave2a_code_digest(),
        "dataset_identity": dataset_identity,
        "extra5_identity": str(extra5_identity),
        "syn_identity": str(syn_identity) if use_syn else None,
        "comparison_settings": comparison_settings,
        "config": json.loads(json.dumps(arm_cfg)),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "in_memory": bool(in_memory),
        "bf16": use_bf16,
        "accelerator": "tpu" if on_tpu else dev.type,
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

    def _make_model(*, training_aux: bool) -> Wave2BE2ERanker:
        return Wave2BE2ERanker(
            product_attr_vocab_sizes={k: len(v) for k, v in enrichment["product_vocabs"].items()},
            num_cities=len(vocabs["city_vocab"]) + 1,
            num_regions=len(vocabs["region_vocab"]) + 1,
            num_industries=len(vocabs["industry_vocab"]) + 1,
            num_demand_features=n_demand,
            num_candidate_features=num_cand_feats,
            num_customer_features=num_cust_feats,
            num_pairwise_bias_features=num_bias_feats,
            active_seq_stat_indices=active_seq_stats,
            pure_v_query=pure_v_query,
            option_a_unseen_head=option_a,
            unseen_dropout_cold_prob=unseen_cold_p,
            unseen_attn_aux_enabled=unseen_attn_aux and training_aux,
            d_model=d_model,
            num_heads=num_heads,
            memory_ffn_dim=memory_ffn_dim,
            memory_layers=memory_layers,
            max_seq_len=seq_len,
            mlp_widths=mlp_widths,
            product_tower_hidden_dim=product_tower_hidden_dim,
            cust_tower_hidden_dim=cust_tower_hidden_dim,
            attr_emb_dim=attr_emb_dim,
            dropout=dropout,
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
    loss_fn = WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    aux_loss_fn = WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    unseen_next_loss_fn = WeightedRankerLoss(pos_weight=10.0, listwise_weight=1.0).to(dev)
    unseen_60d_loss_fn = WeightedRankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    train_sample_weights: Optional[torch.Tensor] = None
    weight_summary: Optional[Dict[str, float]] = None
    if recency_hl is not None:
        train_sample_weights = compute_recency_sample_weights(
            tr_meta_arm["prediction_days"],
            train_cutoff=TRAIN_CUTOFF_V2,
            half_life_days=float(recency_hl),
            min_weight=recency_min_w,
        )
        weight_summary = {
            "half_life_days": float(recency_hl),
            "min_weight_floor": float(recency_min_w),
            "min_norm_weight": float(train_sample_weights.min().item()),
            "median_norm_weight": float(train_sample_weights.median().item()),
            "max_norm_weight": float(train_sample_weights.max().item()),
            "mean_norm_weight": float(train_sample_weights.mean().item()),
        }

    n_tr = len(tr_meta_arm["customer_ids"])
    tot_va_new = int(va_meta_arm["new_target_bn"].sum().item())
    aux_60d_bn = tr_meta_arm["aux_60d_bn"]
    full_window = tr_meta_arm["full_window"]
    n_syn_total = (
        int(len(syn_meta["ret_source_rows"]) + len(syn_meta["fo_source_rows"]))
        if (use_syn and syn_meta is not None)
        else 0
    )
    n_steps_ep = (n_tr + int(batch_size) - 1) // int(batch_size)
    syn_batch_size = (n_syn_total + n_steps_ep - 1) // n_steps_ep if n_syn_total > 0 else 0
    syn_gen = torch.Generator().manual_seed(int(seed) + 1009) if use_syn else None

    print(
        f"\n[v2-w2] Training {arm_name} (e2e cand={num_cand_feats}, sep_targets={separated_main}, "
        f"optA_drop={unseen_cold_p:.0%}, w_optA={unseen_head_weight}, syn7030={use_syn} [{n_syn_total:,}], "
        f"d={d_model}, h={num_heads}, mem_layers={memory_layers}, ffn={memory_ffn_dim}, mlp={mlp_widths}, "
        f"p_hid={product_tower_hidden_dim or d_model}, c_hid={cust_tower_hidden_dim or d_model}, emb={attr_emb_dim}, "
        f"seq={seq_len}, recency_hl={recency_hl}, dev={dev}, bf16={use_bf16}, "
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
        tpu_loss_sum = torch.zeros((), dtype=torch.float32, device=dev) if on_tpu else None
        tpu_main_sum = torch.zeros((), dtype=torch.float32, device=dev) if on_tpu else None
        tpu_aux_sum = torch.zeros((), dtype=torch.float32, device=dev) if on_tpu else None
        tpu_opta_sum = torch.zeros((), dtype=torch.float32, device=dev) if on_tpu else None
        tpu_steps = 0
        syn_perm = (
            torch.randperm(n_syn_total, generator=syn_gen).numpy()
            if (use_syn and n_syn_total > 0 and syn_gen is not None)
            else None
        )
        for step_idx, (rows, cand42_np) in enumerate(
            iter_paired_training_batches(
                tr_cand38_mm, tr_extra5_mm, n_tr, batch_size, gen, in_memory=in_memory
            ),
            start=1,
        ):
            opt.zero_grad(set_to_none=True)
            valid_batch_mask: Optional[torch.Tensor] = None
            if on_tpu and len(rows) < int(batch_size):
                rows, cand42_np, real_b = _pad_batch_for_static_tpu(rows, cand42_np, int(batch_size))
                valid_batch_mask = torch.cat(
                    [
                        torch.ones(real_b, dtype=torch.bool),
                        torch.zeros(int(batch_size) - real_b, dtype=torch.bool),
                    ]
                ).to(dev)
            row_w = train_sample_weights[rows].to(dev) if train_sample_weights is not None else None
            inp = _build_wave2_batch(split_meta=tr_meta_arm, cand42_np=cand42_np, rows=rows, dev=dev)
            with wave2_autocast_context(dev, use_bf16):
                res = model(**inp, prod_inputs=prod_inputs_dev, train_bn_limit=n_train_bn, return_aux=True)

            syn_unseen_lg = syn_alw = syn_uns = syn_tgt = syn_vbm = None
            if use_syn and syn_perm is not None and syn_meta is not None and syn_ret_cand27_mm is not None:
                s_lo = (step_idx - 1) * syn_batch_size
                s_hi = min(n_syn_total, step_idx * syn_batch_size)
                syn_idx = syn_perm[s_lo:s_hi]
                real_syn = int(len(syn_idx))
                if on_tpu and real_syn < syn_batch_size:
                    if real_syn == 0:
                        syn_idx = syn_perm[:syn_batch_size]
                    else:
                        syn_idx = np.concatenate([syn_idx, np.repeat(syn_idx[:1], syn_batch_size - real_syn)])
                    syn_vbm = torch.cat(
                        [
                            torch.ones(real_syn, dtype=torch.bool),
                            torch.zeros(syn_batch_size - real_syn, dtype=torch.bool),
                        ]
                    ).to(dev)
                if len(syn_idx) > 0:
                    syn_inp, syn_alw, syn_uns, syn_tgt = build_synthetic_unseen_batch(
                        syn_idx,
                        tr_meta=tr_meta_arm,
                        tr_cand38_mm=tr_cand38_mm,
                        tr_extra5_mm=tr_extra5_mm,
                        syn_meta=syn_meta,
                        syn_ret_cand27_mm=syn_ret_cand27_mm,
                        dev=dev,
                    )
                    with wave2_autocast_context(dev, use_bf16):
                        res_syn = model(
                            **syn_inp,
                            precomputed_v_bn=res["v_bn"],
                            skip_main_head=True,
                            return_aux=False,
                        )
                    syn_unseen_lg = res_syn["unseen_logits"]

            cmask = inp["candidate_mask"]
            loss_dict = compute_wave2_weighted_separated_losses(
                main_logits=res["output"].logits,
                main_aux_60d_logits=res["aux_60d_logits"],
                gen_unseen_logits=res["unseen_logits"] if option_a else None,
                gen_unseen_aux_60d_logits=res["unseen_aux_60d_logits"] if option_a else None,
                allowed_bn=cmask,
                unseen_bn=tr_meta_arm["unseen_bn"][rows].to(dev),
                target_bn=tr_meta_arm["target_bn"][rows].to(dev),
                aux_60d_bn=aux_60d_bn[rows].to(dev),
                full_window=full_window[rows].to(dev),
                first_observed=tr_meta_arm["first_observed"][rows].to(dev),
                main_loss_fn=loss_fn,
                main_aux_loss_fn=aux_loss_fn,
                unseen_next_loss_fn=unseen_next_loss_fn,
                unseen_60d_loss_fn=unseen_60d_loss_fn,
                unseen_head_weight=unseen_head_weight,
                separated_main_targets=separated_main,
                sample_weight=row_w,
                static_shapes=on_tpu,
                valid_batch_mask=valid_batch_mask,
                syn_unseen_logits=syn_unseen_lg,
                syn_allowed_bn=syn_alw,
                syn_unseen_bn=syn_uns,
                syn_target_bn=syn_tgt,
                syn_valid_batch_mask=syn_vbm,
            )
            loss = loss_dict["loss"]
            assert loss is not None

            if unseen_beta > 0.0 and res["unseen_attn_logits"] is not None:
                aux_rows = full_window[rows].to(dev)
                if aux_rows.any():
                    unseen_m = tr_meta_arm["unseen_bn"][rows].to(dev) & cmask
                    y_u60 = (aux_60d_bn[rows].to(dev) & unseen_m).float()
                    valid_u = aux_rows & (y_u60.sum(dim=-1) > 0)
                    if valid_u.any():
                        u_logits = res["unseen_attn_logits"][valid_u].float().masked_fill(~unseen_m[valid_u], -torch.inf)
                        w_u = row_w[valid_u] if row_w is not None else None
                        l_uattn = unseen_60d_loss_fn(
                            u_logits,
                            y_u60[valid_u],
                            candidate_mask=unseen_m[valid_u],
                            sample_weight=w_u,
                        )["loss"]
                        loss = loss + unseen_beta * l_uattn
                        uattn_losses.append(float(l_uattn.detach().cpu()))

            if on_tpu:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], STAGE2_GRAD_CLIP_NORM
                )
                tpu_optimizer_step(opt, dev)
                assert tpu_loss_sum is not None and tpu_main_sum is not None
                assert tpu_aux_sum is not None and tpu_opta_sum is not None
                tpu_loss_sum = tpu_loss_sum + loss.detach()
                if loss_dict["main_loss"] is not None:
                    tpu_main_sum = tpu_main_sum + loss_dict["main_loss"].detach()
                if loss_dict["aux_60d_loss"] is not None:
                    tpu_aux_sum = tpu_aux_sum + loss_dict["aux_60d_loss"].detach()
                if loss_dict["option_a_loss"] is not None:
                    tpu_opta_sum = tpu_opta_sum + loss_dict["option_a_loss"].detach()
                tpu_steps += 1
                if step_idx % 200 == 0:
                    chk = float(tpu_loss_sum.cpu().item())
                    if not np.isfinite(chk):
                        raise FloatingPointError(f"Non-finite TPU loss in {arm_name} epoch {ep} step {step_idx}")
            else:
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss encountered in {arm_name} epoch {ep}: {loss.item()}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], STAGE2_GRAD_CLIP_NORM
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError(f"Non-finite grad norm encountered in {arm_name} epoch {ep}: {grad_norm}")
                opt.step()
                losses.append(float(loss.detach().cpu()))
                if loss_dict["main_loss"] is not None:
                    main_losses.append(float(loss_dict["main_loss"].detach().cpu()))
                if loss_dict["aux_60d_loss"] is not None:
                    aux_losses.append(float(loss_dict["aux_60d_loss"].detach().cpu()))
                if loss_dict["option_a_loss"] is not None:
                    opt_a_losses.append(float(loss_dict["option_a_loss"].detach().cpu()))
        if on_tpu and tpu_steps > 0:
            assert tpu_loss_sum is not None and tpu_main_sum is not None
            assert tpu_aux_sum is not None and tpu_opta_sum is not None
            mean_l = float(tpu_loss_sum.cpu().item()) / tpu_steps
            if not np.isfinite(mean_l):
                raise FloatingPointError(f"Non-finite TPU epoch loss in {arm_name} epoch {ep}")
            losses = [mean_l]
            main_losses = [float(tpu_main_sum.cpu().item()) / tpu_steps]
            aux_losses = [float(tpu_aux_sum.cpu().item()) / tpu_steps]
            opt_a_losses = [float(tpu_opta_sum.cpu().item()) / tpu_steps]
        sch.step()

        va_main_lg, va_unseen_lg, _ = _score_val_wave2(
            model,
            va_meta=va_meta_arm,
            va_cand38_mm=va_cand38_mm,
            va_extra5_mm=va_extra5_mm,
            prod_inputs_dev=prod_inputs_dev,
            dev=dev,
            batch_size=batch_size,
        )
        ranked_main = rank_eligible_formulations(va_main_lg, va_meta_arm["allowed_bn"])
        m_ep = evaluate_base_name_predictions(
            ranked_main, va_meta_arm["target_bn"], va_meta_arm["new_target_bn"], va_meta_arm["customer_ids"]
        )
        pure_main_ep = evaluate_pure_unseen_predictions(
            va_main_lg,
            allowed_bn=va_meta_arm["allowed_bn"],
            unseen_bn=va_meta_arm["unseen_bn"],
            new_target_bn=va_meta_arm["new_target_bn"],
            first_observed=va_meta_arm["first_observed"],
        )
        pure_opt_a_ep = None
        m_oof_ep = None
        oof_meta_ep = None
        if option_a and va_unseen_lg is not None:
            pure_opt_a_ep = evaluate_pure_unseen_predictions(
                va_unseen_lg,
                allowed_bn=va_meta_arm["allowed_bn"],
                unseen_bn=va_meta_arm["unseen_bn"],
                new_target_bn=va_meta_arm["new_target_bn"],
                first_observed=va_meta_arm["first_observed"],
            )
            oof_scores_ep, oof_meta_ep = fit_oof_affine_unseen_blend(
                va_main_lg,
                va_unseen_lg,
                allowed_bn=va_meta_arm["allowed_bn"],
                unseen_bn=va_meta_arm["unseen_bn"],
                target_bn=va_meta_arm["target_bn"],
                new_target_bn=va_meta_arm["new_target_bn"],
                customer_ids=va_meta_arm["customer_ids"],
                n_folds=2,
            )
            ranked_oof_ep = rank_eligible_formulations(oof_scores_ep, va_meta_arm["allowed_bn"])
            m_oof_ep = evaluate_base_name_predictions(
                ranked_oof_ep, va_meta_arm["target_bn"], va_meta_arm["new_target_bn"], va_meta_arm["customer_ids"]
            )

        inf_state = {k: v.detach().cpu() for k, v in strip_wave2b_training_aux(model.state_dict()).items()}
        sel_m = m_oof_ep if m_oof_ep is not None else m_ep
        nk = (float(sel_m["bn_ndcg@10"]), int(sel_m["true_new_bn_hits@10"]), float(sel_m["bn_recall@10"]))
        dk = (int(sel_m["true_new_bn_hits@10"]), float(sel_m["bn_ndcg@10"]), float(sel_m["bn_recall@10"]))
        ckpt_payload: Dict[str, Any] = {
            "epoch": ep,
            "metrics": m_ep,
            "pure_unseen": pure_opt_a_ep or pure_main_ep,
            "model_state_dict": inf_state,
        }
        if m_oof_ep is not None:
            ckpt_payload["oof_metrics"] = m_oof_ep

        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save(ckpt_payload, out_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save(ckpt_payload, out_dir / "best_discovery.pt")
        if option_a and pure_opt_a_ep is not None:
            ov = pure_opt_a_ep["overall"]
            uk = (
                int(ov["pure_unseen_hits@10"]),
                float(ov["pure_unseen_ndcg@10_new_baskets"]),
                int(ov["pure_unseen_hits@25"]),
            )
            if uk > best_unseen_key:
                best_unseen_key = uk
                atomic_torch_save(ckpt_payload, out_dir / "best_unseen.pt")

        pu_log = (pure_opt_a_ep or pure_main_ep)["overall"]
        ep_rec: Dict[str, Any] = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "main_loss": float(np.mean(main_losses)) if main_losses else 0.0,
            "aux_60d_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "option_a_loss": float(np.mean(opt_a_losses)) if opt_a_losses else 0.0,
            "unseen_attn_loss": float(np.mean(uattn_losses)) if uattn_losses else 0.0,
            "bn_recall@5": float(sel_m["bn_recall@5"]),
            "bn_recall@10": float(sel_m["bn_recall@10"]),
            "bn_ndcg@10": float(sel_m["bn_ndcg@10"]),
            "true_new_bn_hits@10": int(sel_m["true_new_bn_hits@10"]),
            "true_new_bn_hits@25": int(sel_m["true_new_bn_hits@25"]),
            "main_only_bn_recall@10": float(m_ep["bn_recall@10"]),
            "main_only_bn_ndcg@10": float(m_ep["bn_ndcg@10"]),
            "main_only_true_new_bn_hits@10": int(m_ep["true_new_bn_hits@10"]),
            "pure_unseen_hits@10": int(pu_log["pure_unseen_hits@10"]),
            "pure_unseen_hits@25": int(pu_log["pure_unseen_hits@25"]),
            "pure_unseen_ndcg@10": float(pu_log["pure_unseen_ndcg@10_new_baskets"]),
        }
        if oof_meta_ep is not None:
            full_ab = oof_meta_ep["deployment_full_fit"]
            ep_rec["oof_affine_full_a"] = float(full_ab["a"])
            ep_rec["oof_affine_full_b"] = float(full_ab["b"])
        epoch_history.append(ep_rec)
        print(
            f"[v2-w2:{arm_name}] ep={ep:02d}/{epochs} loss={ep_rec['loss']:.4f} "
            f"main={ep_rec['main_loss']:.4f} optA={ep_rec['option_a_loss']:.4f} "
            f"CalR@10={ep_rec['bn_recall@10']:.5f} CalNDCG={ep_rec['bn_ndcg@10']:.5f} "
            f"CalNew@10={ep_rec['true_new_bn_hits@10']}/{tot_va_new} "
            f"PureU@10={ep_rec['pure_unseen_hits@10']}/{tot_va_new} "
            f"PureU@25={ep_rec['pure_unseen_hits@25']} ({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    product_ids = shared_meta["product_ids"]
    universe = shared_meta["universe"]
    sku_val = va_meta_arm["sku_val"]

    arm_res: Dict[str, Any] = {
        "arm": arm_name,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in arm_cfg.items()},
        "train_params": int(n_train_params),
        "inference_params": int(n_inf_params),
        "per_request_params": int(n_req_params),
        "training_seconds": float(time.perf_counter() - t_train),
        "epoch_history": epoch_history,
    }
    if weight_summary is not None:
        arm_res["recency_weights"] = weight_summary

    cpu_bench_ms: Dict[str, float] = {}
    selections = [("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")]
    if option_a:
        selections.append(("unseen", "best_unseen.pt"))

    for sel_name, fname in selections:
        ck = torch.load(out_dir / fname, map_location="cpu", weights_only=False)
        eval_m = _make_model(training_aux=False)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_main_lg, va_unseen_lg, v_bn_cpu = _score_val_wave2(
            eval_m,
            va_meta=va_meta_arm,
            va_cand38_mm=va_cand38_mm,
            va_extra5_mm=va_extra5_mm,
            prod_inputs_dev=prod_inputs_dev,
            dev=dev,
            batch_size=batch_size,
        )
        ranked_main = rank_eligible_formulations(va_main_lg, va_meta_arm["allowed_bn"])
        pure_main = evaluate_pure_unseen_predictions(
            va_main_lg,
            allowed_bn=va_meta_arm["allowed_bn"],
            unseen_bn=va_meta_arm["unseen_bn"],
            new_target_bn=va_meta_arm["new_target_bn"],
            first_observed=va_meta_arm["first_observed"],
        )

        if sel_name in ("ndcg", "discovery"):
            npz_p = out_dir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(
                ranked_main, va_meta_arm, sku_val, dev_purchases, catalogue, product_ids, universe, npz_p
            )
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"{arm_name} {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            ev["pure_unseen_main"] = pure_main
            arm_res[sel_name] = ev

        if option_a and va_unseen_lg is not None:
            pure_unseen = evaluate_pure_unseen_predictions(
                va_unseen_lg,
                allowed_bn=va_meta_arm["allowed_bn"],
                unseen_bn=va_meta_arm["unseen_bn"],
                new_target_bn=va_meta_arm["new_target_bn"],
                first_observed=va_meta_arm["first_observed"],
            )
            if pure_unseen != ck["pure_unseen"]:
                raise AssertionError(f"{arm_name} {sel_name} pure unseen replay mismatch!")

            oof_scores, oof_meta = fit_oof_affine_unseen_blend(
                va_main_lg,
                va_unseen_lg,
                allowed_bn=va_meta_arm["allowed_bn"],
                unseen_bn=va_meta_arm["unseen_bn"],
                target_bn=va_meta_arm["target_bn"],
                new_target_bn=va_meta_arm["new_target_bn"],
                customer_ids=va_meta_arm["customer_ids"],
                n_folds=2,
            )
            ranked_oof = rank_eligible_formulations(oof_scores, va_meta_arm["allowed_bn"])
            oof_npz = out_dir / f"predictions_optionA_oof_{sel_name}.npz"
            ev_oof = _evaluate_with_cohorts_and_sku(
                ranked_oof, va_meta_arm, sku_val, dev_purchases, catalogue, product_ids, universe, oof_npz
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
            if option_a and va_unseen_lg is not None:
                np.save(out_dir / "val_unseen_logits_ndcg.npy", va_unseen_lg.float().numpy())

            eval_cpu = eval_m.to("cpu").eval()
            one_cand = assemble_cand_42(va_cand38_mm[0:1], va_extra5_mm[0:1])
            one_inp = _build_wave2_batch(
                split_meta=va_meta_arm,
                cand42_np=one_cand,
                rows=torch.tensor([0]),
                dev=torch.device("cpu"),
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


def run_v2_full_wave2(
    *,
    seed: int = SEED,
    epochs: int = STAGE2_EPOCHS,
    batch_size: int = STAGE2_BATCH_SIZE,
    preset: str = "all",
    arms: Optional[Sequence[str]] = None,
    include_hist96: bool = False,
    device: Optional[str] = None,
    amp: bool = False,
    in_memory: bool = False,
    skip_v1_verify: bool = False,
    sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if preset not in ("all", "size", "final_2mem"):
        raise ValueError(f"Unsupported Wave 2 preset {preset!r}. Expected 'all', 'size', or 'final_2mem'.")
    if preset != "all" and arms is not None:
        raise ValueError(f"Specify either --preset {preset} or --arms, not both.")
    t_all = time.perf_counter()
    dev = resolve_wave2_device(device)
    on_tpu = is_tpu_device(dev)
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

    base_name = ACTIVE_V2_FULL_BASE_MODEL
    print(
        f"[v2-w2] Wave 2A base: {base_name} / w2c_exp1_clean_sep_w10 "
        "(mode=e2e, schema=cand_42+CUST_7+SEQ_5, pure_Q=True, bias=42, sep_targets=True, dropout=0%, w_optA=1.0)",
        flush=True,
    )

    ref_preds = resolve_v2_full_reference_predictions(sync_dir)
    missing_refs = [k for k in REQUIRED_WAVE2A_REF_KEYS if k not in ref_preds]
    if missing_refs:
        raise ValueError(
            "Wave 2A comparisons require verified calibrated `v2_full_reference.json` predictions; "
            f"missing keys: {missing_refs}"
        )

    all_specs = build_wave2_arm_specs(ACTIVE_E2E_BASE_CONFIG, include_hist96=include_hist96)
    available = {a["name"] for a in all_specs}
    if preset == "size":
        effective_arms: Optional[Sequence[str]] = SIZE_SCALING_ARM_NAMES
    elif preset == "final_2mem":
        effective_arms = FINAL_2MEM_ARM_NAMES
    else:
        effective_arms = arms
    if effective_arms is not None and (not effective_arms or set(effective_arms) - available):
        raise ValueError(f"Unsupported Wave 2 arm selection {effective_arms!r}. Available: {sorted(available)}")
    selected_specs = [a for a in all_specs if effective_arms is None or a["name"] in effective_arms]

    tr_e5_p, va_e5_p, e5_manifest = build_or_load_extra_cand_5()
    shared_meta, tr_meta, va_meta = load_prepared_metadata(PREPARED_ROOT)
    mmap_mode = None if in_memory else "r"
    if in_memory:
        print("[v2-w2] Loading float16 cand_38 + extra_cand_5 into RAM (--in-memory)...", flush=True)
    tr_cand38_mm = np.load(tr_meta["cand_38_path"], mmap_mode=mmap_mode)
    va_cand38_mm = np.load(va_meta["cand_38_path"], mmap_mode=mmap_mode)
    tr_extra5_mm = np.load(tr_e5_p, mmap_mode=mmap_mode)
    va_extra5_mm = np.load(va_e5_p, mmap_mode=mmap_mode)
    extra5_identity = input_digest((e5_manifest["identity"], e5_manifest["output_sha256"]))

    dev_purchases = load_v2_development_purchases()
    catalogue = load_product_catalogue(RAW_ITEMS_PATH)
    enriched_catalogue = load_enriched_catalogue(RAW_ITEMS_PATH, descriptions_path=None)

    syn_meta: Optional[Mapping[str, Any]] = None
    syn_ret_cand27_mm: Optional[np.ndarray] = None
    syn_identity: Optional[str] = None
    if any(bool(a.get("synthetic_unseen_aug", False)) for a in selected_specs):
        syn_c27_p, syn_meta_p, syn_manifest = build_or_load_synthetic_unseen_cache(
            returning_fraction=SYNTHETIC_RETURNING_FRACTION,
            seed=seed,
        )
        syn_meta = torch.load(syn_meta_p, map_location="cpu", weights_only=False)
        syn_ret_cand27_mm = np.load(syn_c27_p, mmap_mode=mmap_mode)
        syn_identity = input_digest((syn_manifest["identity"], syn_manifest["output_sha256"]))

    dataset_identity = prepared_dataset_digest(
        shared_meta, tr_meta, va_meta, tr_cand38_mm, va_cand38_mm, dev_purchases, catalogue
    )
    use_bf16 = bool(amp and supports_native_bf16(dev))
    exp_root = ARTIFACT_ROOT / f"wave2a_{base_name}_clean_sep_w10-seed{seed}"
    arm_results: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    code_digest = _wave2a_code_digest()
    comparison_settings = {
        "schema_version": WAVE2A_SCHEMA_VERSION,
        "code_digest": code_digest,
        "dataset": dataset_identity,
        "extra5_identity": extra5_identity,
        "base_model": base_name,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "in_memory": bool(in_memory),
        "bf16": use_bf16,
        "accelerator": "tpu" if on_tpu else dev.type,
    }

    def _arm_dataset_identity(tr_a: Mapping[str, Any], va_a: Mapping[str, Any]) -> str:
        return input_digest(
            (
                dataset_identity,
                tr_a["sequence"],
                va_a["sequence"],
                tr_a["seq_product_ids"],
                va_a["seq_product_ids"],
                tr_a["attention_mask"],
                va_a["attention_mask"],
                tr_a["seq_purchase_dates"],
                va_a["seq_purchase_dates"],
            )
        )

    specs_by_name = {a["name"]: a for a in all_specs}

    for arm_cfg in selected_specs:
        if arm_cfg["name"] == "w2_hist_len96":
            hist64 = arm_results.get("w2_hist_len64")
            if hist64 is None:
                h64_cfg = specs_by_name["w2_hist_len64"]
                tr_h64, va_h64 = ensure_v2_sequences_for_length(
                    64,
                    shared_meta=shared_meta,
                    tr_meta=tr_meta,
                    va_meta=va_meta,
                    dev_purchases=dev_purchases,
                    catalogue=enriched_catalogue,
                    cache_dir=PREPARED_ROOT,
                )
                expected_h64_settings = {
                    "schema_version": WAVE2A_SCHEMA_VERSION,
                    "code_digest": code_digest,
                    "dataset_identity": _arm_dataset_identity(tr_h64, va_h64),
                    "extra5_identity": str(extra5_identity),
                    "syn_identity": None,
                    "comparison_settings": comparison_settings,
                    "config": json.loads(json.dumps(h64_cfg)),
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "in_memory": bool(in_memory),
                    "bf16": use_bf16,
                    "accelerator": "tpu" if on_tpu else dev.type,
                }
                del tr_h64, va_h64
                directory = exp_root / "w2_hist_len64"
                locations = [directory] if sync_dir is None else [directory, sync_dir / directory]
                expected_h64 = _expected_wave2a_outputs(bool(h64_cfg.get("option_a_unseen_head", True)))
                for location in locations:
                    hist64 = _completed_w2b_arm(location, expected_h64_settings, expected_h64)
                    if hist64 is not None:
                        break
            ctrl_disc: Optional[Mapping[str, Any]] = None
            ctrl_ndcg: Optional[Mapping[str, Any]] = None
            if V2_FULL_REFERENCE_PATH.is_file():
                v2_ref_doc = json.loads(V2_FULL_REFERENCE_PATH.read_text(encoding="utf-8"))
                if v2_ref_doc.get("reference_id") == ACTIVE_V2_FULL_REFERENCE_ID:
                    if "validation_metrics_primary_cal_discovery" in v2_ref_doc:
                        ctrl_disc = {"bn_metrics": v2_ref_doc["validation_metrics_primary_cal_discovery"]}
                    if "validation_metrics_companion_cal_ndcg" in v2_ref_doc:
                        ctrl_ndcg = {"bn_metrics": v2_ref_doc["validation_metrics_companion_cal_ndcg"]}
            improved = False
            if hist64 is not None:
                if ctrl_disc is not None and "optionA_oof_discovery" in hist64:
                    improved = improved or (_discovery_key(hist64["optionA_oof_discovery"]) > _discovery_key(ctrl_disc))
                if ctrl_ndcg is not None and "optionA_oof_ndcg" in hist64:
                    improved = improved or (_selection_key(hist64["optionA_oof_ndcg"]) > _selection_key(ctrl_ndcg))
            if not improved:
                skipped[arm_cfg["name"]] = "No verified 64-event improvement over the 32-event w10 reference control."
                continue

        slen = int(arm_cfg["seq_len"])
        tr_meta_arm, va_meta_arm = ensure_v2_sequences_for_length(
            slen,
            shared_meta=shared_meta,
            tr_meta=tr_meta,
            va_meta=va_meta,
            dev_purchases=dev_purchases,
            catalogue=enriched_catalogue,
            cache_dir=PREPARED_ROOT,
        )
        use_syn_arm = bool(arm_cfg.get("synthetic_unseen_aug", False))
        arm_results[arm_cfg["name"]] = train_and_eval_wave2_arm(
            arm_cfg,
            comparison_settings=comparison_settings,
            dataset_identity=_arm_dataset_identity(tr_meta_arm, va_meta_arm),
            shared_meta=shared_meta,
            tr_meta_arm=tr_meta_arm,
            va_meta_arm=va_meta_arm,
            tr_cand38_mm=tr_cand38_mm,
            va_cand38_mm=va_cand38_mm,
            tr_extra5_mm=tr_extra5_mm,
            va_extra5_mm=va_extra5_mm,
            extra5_identity=extra5_identity,
            syn_meta=syn_meta if use_syn_arm else None,
            syn_ret_cand27_mm=syn_ret_cand27_mm if use_syn_arm else None,
            syn_identity=syn_identity if use_syn_arm else None,
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
        del tr_meta_arm, va_meta_arm

    comparisons: Dict[str, Any] = {}
    for arm_name, result in arm_results.items():
        if "v2_full_ref_primary" in ref_preds:
            r_prim = ref_preds["v2_full_ref_primary"]
            for key, tag in (
                ("optionA_oof_discovery", "cal_discovery"),
                ("optionA_oof_ndcg", "cal_ndcg"),
                ("optionA_oof_unseen", "cal_unseen"),
            ):
                if key in result:
                    comparisons[f"{arm_name}_{tag}_vs_v2_full_ref_primary"] = paired_base_name_bootstrap_ci(
                        r_prim, Path(result[key]["predictions_npz"]), seed=seed
                    )
        if "v2_full_ref_cal_ndcg" in ref_preds:
            r_cal_ndcg = ref_preds["v2_full_ref_cal_ndcg"]
            for key, tag in (
                ("optionA_oof_discovery", "cal_discovery"),
                ("optionA_oof_ndcg", "cal_ndcg"),
                ("optionA_oof_unseen", "cal_unseen"),
            ):
                if key in result:
                    comparisons[f"{arm_name}_{tag}_vs_v2_full_ref_cal_ndcg"] = paired_base_name_bootstrap_ci(
                        r_cal_ndcg, Path(result[key]["predictions_npz"]), seed=seed
                    )

    for base_arm, cand_arm in (
        ("w2_2mem_blocks", "w2_2mem_wider_mlp"),
        ("w2_wider_mlp", "w2_2mem_wider_mlp"),
        ("w2_dim192", "w2_dim192_2mem_wider_mlp"),
        ("w2_2mem_wider_mlp", "w2_dim192_2mem_wider_mlp"),
        ("w2_dim192", "w2_dim256"),
        ("w2_dim256", "w2_dim256_2mem_4mlp"),
        ("w2_dim192_2mem_wider_mlp", "w2_dim256_2mem_4mlp"),
        ("w2_2mem_wider_mlp", "w2_2mem_wider_mlp_syn7030"),
        ("w2_2mem_hist64", "w2_2mem_hist64_recency_hl365"),
        ("w2_2mem_recency_hl365", "w2_2mem_hist64_recency_hl365"),
        ("w2_2mem_hist64", "w2_2mem_recency_hl365"),
    ):
        if base_arm in arm_results and cand_arm in arm_results:
            b_res = arm_results[base_arm]
            c_res = arm_results[cand_arm]
            for key, tag in (
                ("optionA_oof_discovery", "cal_discovery"),
                ("optionA_oof_ndcg", "cal_ndcg"),
                ("optionA_oof_unseen", "cal_unseen"),
            ):
                if key in b_res and key in c_res:
                    comparisons[f"{cand_arm}_{tag}_vs_{base_arm}_{tag}"] = paired_base_name_bootstrap_ci(
                        Path(b_res[key]["predictions_npz"]),
                        Path(c_res[key]["predictions_npz"]),
                        seed=seed,
                    )

    if not skip_v1_verify:
        if snapshot_preserved_hashes() != before_hashes:
            raise AssertionError("v1 preserved reference hashes changed!")
        _verify_reference_files(ref_v1)
        _verify_reference_files(ref_v1_3s)

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if preset in ("size", "final_2mem"):
        summary_path = EVIDENCE_ROOT / f"v2_full_wave2a_{base_name}_clean_sep_w10_{preset}_seed{seed}.json"
    else:
        summary_path = EVIDENCE_ROOT / f"v2_full_wave2a_{base_name}_clean_sep_w10_seed{seed}.json"
        if len(selected_specs) != len(all_specs):
            summary_path = summary_path.with_stem(summary_path.stem + "_" + "_".join(a["name"] for a in selected_specs))
    summary = {
        "experiment": "v2_full_wave2a_controlled_ablations_clean_sep_w10",
        "preset": preset,
        "dataset": "v2_full",
        "reference_id": ACTIVE_V2_FULL_REFERENCE_ID,
        "feature_schema": "cand_42+CUST_7+SEQ_5 (OptionA=cand_37, dropout=0%, sep_targets=True, w_optA=1.0)",
        "extra_cand_5_identity": str(e5_manifest["identity"]),
        "base_model": base_name,
        "base_config": ACTIVE_E2E_BASE_CONFIG,
        "train_cutoff": str(TRAIN_CUTOFF_V2),
        "test_start": str(TEST_START_V2),
        "heldout_access": False,
        "seed": int(seed),
        "amp_bf16": use_bf16,
        "arms": arm_results,
        "skipped": skipped,
        "paired_bootstrap_95ci": comparisons,
        "v1_preserved_hashes_verified": not skip_v1_verify,
        "elapsed_seconds": float(time.perf_counter() - t_all),
        "summary_path": str(summary_path),
    }
    atomic_json(summary, summary_path, overwrite=True)
    sync_outputs_to_dir(sync_dir, (summary_path,))
    print(f"\nSaved v2_full Wave 2A summary -> {summary_path} ({summary['elapsed_seconds']:.1f}s)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=STAGE2_BATCH_SIZE)
    parser.add_argument("--preset", choices=("all", "size", "final_2mem"), default="all")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--include-hist96", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--skip-v1-verify", action="store_true")
    parser.add_argument("--sync-dir", type=Path, default=None)
    args = parser.parse_args()
    run_v2_full_wave2(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        preset=args.preset,
        arms=args.arms,
        include_hist96=args.include_hist96,
        device=args.device,
        amp=args.amp,
        in_memory=args.in_memory,
        skip_v1_verify=args.skip_v1_verify,
        sync_dir=args.sync_dir,
    )


if __name__ == "__main__":
    main()
