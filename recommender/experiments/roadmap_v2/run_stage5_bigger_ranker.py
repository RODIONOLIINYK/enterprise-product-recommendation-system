"""Validation-only Stage 5 capacity scaling (No-MLP Full-Pool UPS-CANR):
Tests two bigger Stage 2 architectures directly over ALL 585 eligible formulations (no Stage 1 MLP):
  1. `bigger_128d_8h_2L_4mlp`: 8 heads, 2-layer memory Transformer (FFN 512), 2-layer cross-attention, 4-layer MLP (512->256->128->64)
  2. `wider_256d_8h_2L_4mlp`: d_model=256 (projected candidates), 8 heads, 2-layer memory Transformer (FFN 512), 2-layer cross-attention, 4-layer MLP (512->256->128->64)
"""

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.artifacts import atomic_json
from recommender.experiments.rolling_candidate_v2.config import DEFAULT_RETRIEVER
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.sequences import PURCHASE_DATE_ORIGIN
from recommender.shared_data import atomic_torch_save
from recommender.split import TRAIN_CUTOFF
from recommender.train_retriever import get_device

from .augmentation import build_strict_60d_aux_labels
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
from .dated_baskets import FEATURE_NAMES, prepare_dated_baskets
from .run_stage5 import _collapse_bool_matrix_to_bn, _verify_reference_files
from .run_stage5_ablations import (
    REFERENCE_PATH,
    _build_fullpool_batch,
    _evaluate_with_cohorts_and_sku,
    _score_fullpool_ranker,
)
from .stage5_base_entity import evaluate_base_name_predictions, paired_base_name_bootstrap_ci


OUT_ROOT = ARTIFACT_ROOT / "stage5_bigger_ranker_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "stage5_bigger_ranker_seed42.json"


class BiggerStage5BaseNameRanker(nn.Module):
    """Scaled Stage 2 UPS-CANR with configurable d_model, attention heads, memory/cross-attn depth, and MLP trunk."""

    def __init__(
        self,
        *,
        num_base_names: int,
        num_features: int,
        num_cities: int,
        num_regions: int,
        num_industries: int,
        num_categories: int,
        num_subcategories: int,
        num_business_lines: int,
        d_model: int = 128,
        num_heads: int = 8,
        memory_layers: int = 2,
        memory_ffn_mult: int = 4,
        memory_ffn_dim: Optional[int] = None,
        cross_attn_layers: int = 2,
        mlp_widths: Sequence[int] = (512, 256, 128, 64),
        dropout: float = 0.15,
        aux_60d_enabled: bool = True,
        seed: int = SEED,
    ):
        super().__init__()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_features = int(num_features)
        self.aux_60d_enabled = bool(aux_60d_enabled)

        self.base = UPSCANRRanker(
            text_dim=1024,
            memory_dim=self.d_model,
            num_cities=int(num_cities),
            num_regions=int(num_regions),
            num_industries=int(num_industries),
            num_products=int(num_base_names) + 1,
            num_categories=int(num_categories),
            num_subcategories=int(num_subcategories),
            num_business_lines=int(num_business_lines),
            num_tabular_features=8,
            num_pairwise_features=13,
            seq_feature_dim=9,
            hidden_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            statistics_attention=True,
            excluded_statistics=("history_repeat",),
            cached_only=True,
        )
        # Replace 1-layer memory encoder with configurable TransformerEncoder
        ffn_dim = int(memory_ffn_dim) if memory_ffn_dim is not None else self.d_model * int(memory_ffn_mult)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=ffn_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=(int(memory_layers) > 1),
        )
        self.base.memory_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=int(memory_layers), enable_nested_tensor=False
        )
        self.base.mlp.requires_grad_(False)

        # Project 128D Two-Tower candidate vectors to d_model when d_model != 128
        self.cand_proj: Optional[nn.Module] = (
            nn.Sequential(nn.Linear(128, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
            if self.d_model != 128
            else None
        )

        # Optional extra cross-attention refinement layers (layer 1 lives in self.base.cross_attention)
        self.extra_cross_layers = nn.ModuleList()
        self.extra_cross_norms = nn.ModuleList()
        self.extra_ffns = nn.ModuleList()
        self.extra_ffn_norms = nn.ModuleList()
        for _ in range(max(0, int(cross_attn_layers) - 1)):
            self.extra_cross_layers.append(
                nn.MultiheadAttention(self.d_model, self.num_heads, dropout=float(dropout), batch_first=True)
            )
            self.extra_cross_norms.append(nn.LayerNorm(self.d_model))
            self.extra_ffns.append(
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 2),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Dropout(float(dropout)),
                )
            )
            self.extra_ffn_norms.append(nn.LayerNorm(self.d_model))

        # Deeper & wider scoring trunk (with residual skips across consecutive equal-width blocks)
        total_in = self.d_model * 3 + self.num_features
        self.mlp_blocks = nn.ModuleList()
        self.mlp_residual_flags: list[bool] = []
        prev = total_in
        for idx, width in enumerate(mlp_widths):
            w = int(width)
            dp = 0.25 if idx == 0 else (0.1 if idx == len(mlp_widths) - 1 else 0.2)
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(prev, w),
                    nn.LayerNorm(w),
                    nn.GELU(),
                    nn.Dropout(dp),
                )
            )
            self.mlp_residual_flags.append(prev == w)
            prev = w
        self.main_head = nn.Linear(prev, 1)
        self.aux_60d_head: Optional[nn.Linear] = nn.Linear(prev, 1) if self.aux_60d_enabled else None

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        **kwargs,
    ) -> Any:
        z_cand = self.cand_proj(candidate_vectors) if self.cand_proj is not None else candidate_vectors
        encoded = self.base.encode_candidate_features(
            pairwise_features=pairwise_features[..., :13],
            candidate_vectors=z_cand,
            memory_value_statistics=False,
            **kwargs,
        )
        h_att = encoded.h_attended
        if len(self.extra_cross_layers) > 0:
            kpm = self.base.build_key_padding_mask(kwargs["attention_mask"])
            mem = encoded.memory
            for attn, norm1, ffn, norm2 in zip(
                self.extra_cross_layers, self.extra_cross_norms, self.extra_ffns, self.extra_ffn_norms
            ):
                delta, _ = attn(
                    query=norm1(z_cand + h_att),
                    key=mem,
                    value=mem,
                    key_padding_mask=kpm,
                    need_weights=False,
                )
                h_att = h_att + delta
                h_att = h_att + ffn(norm2(h_att))

        hidden = torch.cat([z_cand, h_att, z_cand * h_att, pairwise_features], dim=-1)
        for block, use_res in zip(self.mlp_blocks, self.mlp_residual_flags):
            out_h = block(hidden)
            hidden = hidden + out_h if use_res else out_h
        logits = self.main_head(hidden).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -torch.inf)
        out = encoded.to_output(logits)
        if return_aux and self.aux_60d_head is not None:
            aux_logits = self.aux_60d_head(hidden).squeeze(-1)
            if candidate_mask is not None:
                aux_logits = aux_logits.masked_fill(~candidate_mask, -torch.inf)
            return out, aux_logits
        return out


VARIANTS: Dict[str, Dict[str, Any]] = {
    "bigger_128d_8h_2L_4mlp": {
        "d_model": 128,
        "num_heads": 8,
        "memory_layers": 2,
        "memory_ffn_mult": 4,
        "cross_attn_layers": 2,
        "mlp_widths": (512, 256, 128, 64),
        "dropout": 0.15,
        "batch_size": 64,
        "lr": 8e-4,
    },
    "deep_128d_8h_3mem_3xattn_5mlp": {
        "d_model": 128,
        "num_heads": 8,
        "memory_layers": 3,
        "memory_ffn_mult": 4,
        "cross_attn_layers": 3,
        "mlp_widths": (512, 512, 256, 128, 64),
        "dropout": 0.15,
        "batch_size": 48,
        "lr": 6e-4,
    },
    "wider_256d_8h_2L_4mlp": {
        "d_model": 256,
        "num_heads": 8,
        "memory_layers": 2,
        "memory_ffn_mult": 2,
        "cross_attn_layers": 2,
        "mlp_widths": (512, 256, 128, 64),
        "dropout": 0.15,
        "batch_size": 48,
        "lr": 6e-4,
    },
}


MODEST_VARIANTS: Dict[str, Dict[str, Any]] = {
    "modest_wider_mlp_384": {
        "d_model": 128,
        "num_heads": 4,
        "memory_layers": 1,
        "memory_ffn_mult": 2,
        "memory_ffn_dim": 256,
        "cross_attn_layers": 1,
        "mlp_widths": (384, 192, 96),
        "dropout": 0.1,
        "batch_size": 64,
        "lr": STAGE2_LR,
    },
    "modest_8h_ffn384_mlp384": {
        "d_model": 128,
        "num_heads": 8,
        "memory_layers": 1,
        "memory_ffn_mult": 3,
        "memory_ffn_dim": 384,
        "cross_attn_layers": 1,
        "mlp_widths": (384, 192, 96),
        "dropout": 0.1,
        "batch_size": 64,
        "lr": STAGE2_LR,
    },
}


def run_bigger_ranker(
    *,
    seed: int = SEED,
    device: Optional[str] = None,
    mode: str = "modest",
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    _verify_reference_files(reference)
    saved_val = json.loads(Path(reference["checkpoints"]["summary"]["path"]).read_text(encoding="utf-8"))

    print(f"[bigger-ranker/{mode}] Preparing canonical dated first/returning baskets...", flush=True)
    prepared = prepare_dated_baskets()
    (
        bn_data,
        _,
        universe,
        _,
        vocabs,
        purchases,
        catalogue,
        product_ids,
        sku_data,
        manifest,
    ) = prepared
    if manifest != saved_val["data_manifest"]:
        raise ValueError("Dated manifest mismatch vs pinned reference!")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_bn = universe["num_base_names"]
    bn_to_members = universe["bn_to_member_pid_indices"]
    tr_bn = bn_data["train"]
    va_bn = bn_data["val"]
    sku_val = sku_data["val"]
    orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]

    tt_dir = Path(reference["checkpoints"]["two_tower"]["path"]).parent
    v_bn = torch.load(tt_dir / "train_vectors.pt", map_location="cpu", weights_only=False)["v_bn"]
    v_dev = v_bn.to(dev)

    aux_60d_sku = build_strict_60d_aux_labels(
        purchases, tr_bn["customer_ids"], tr_bn["prediction_days"].tolist(), product_ids
    )
    aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members).float()
    full_window = tr_bn["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

    ref_ndcg_npz = Path(saved_val["stage2_ranker"]["ndcg"]["bn_predictions_npz"])
    selected_variants = MODEST_VARIANTS if mode == "modest" else VARIANTS
    summary_path = (
        EVIDENCE_ROOT / f"stage5_modest_ranker_seed{seed}.json"
        if mode == "modest"
        else SUMMARY_PATH
    )
    results: Dict[str, Any] = {
        "experiment": f"stage5_{mode}_fullpool_ranker",
        "reference_id": reference["reference_id"],
        "seed": int(seed),
        "heldout_access": False,
        "variants": {},
    }

    for vname, cfg in selected_variants.items():
        vdir = OUT_ROOT / f"{vname}-seed{seed}"
        vdir.mkdir(parents=True, exist_ok=True)

        def _make(aux: bool) -> BiggerStage5BaseNameRanker:
            return BiggerStage5BaseNameRanker(
                num_base_names=n_bn,
                num_features=len(FEATURE_NAMES),
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
                memory_ffn_dim=cfg.get("memory_ffn_dim"),
                cross_attn_layers=cfg["cross_attn_layers"],
                mlp_widths=cfg["mlp_widths"],
                dropout=cfg["dropout"],
                aux_60d_enabled=aux,
                seed=seed,
            ).to(dev)

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = _make(aux=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"[bigger-ranker/{vname}] Trainable params: {n_params:,} (config={cfg})", flush=True)

        opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=STAGE2_WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STAGE2_EPOCHS, eta_min=STAGE2_ETA_MIN)
        loss_fn = RankerLoss().to(dev)
        aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)

        gen = torch.Generator().manual_seed(int(seed))
        best_ndcg_key = (-1.0, -1, -1.0)
        best_disc_key = (-1, -1.0, -1.0)
        t_v = time.perf_counter()

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

            va_lg = _score_fullpool_ranker(model, va_bn, v_dev, dev, batch_size=48)
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            m_ep = evaluate_base_name_predictions(ranked, va_bn["target_bn"], va_bn["new_target_bn"], va_bn["customer_ids"])
            inf_state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux_60d_head.")}

            nk = (float(m_ep["bn_ndcg@10"]), int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_recall@10"]))
            dk = (int(m_ep["true_new_bn_hits@10"]), float(m_ep["bn_ndcg@10"]), float(m_ep["bn_recall@10"]))
            if nk > best_ndcg_key:
                best_ndcg_key = nk
                atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, vdir / "best_ndcg.pt")
            if dk > best_disc_key:
                best_disc_key = dk
                atomic_torch_save({"epoch": ep, "metrics": m_ep, "model_state_dict": inf_state}, vdir / "best_discovery.pt")
            print(
                f"[bigger-ranker/{vname}] ep={ep:02d}/{STAGE2_EPOCHS} loss={np.mean(losses):.4f} "
                f"R@5={m_ep['bn_recall@5']:.5f} R@10={m_ep['bn_recall@10']:.5f} "
                f"NDCG@10={m_ep['bn_ndcg@10']:.5f} New@10={m_ep['true_new_bn_hits@10']}/1721 "
                f"({time.perf_counter() - ep_t0:.1f}s)",
                flush=True,
            )

        v_res: Dict[str, Any] = {
            "config": cfg,
            "trainable_params": int(n_params),
            "training_seconds": float(time.perf_counter() - t_v),
        }
        for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
            ck = torch.load(vdir / fname, map_location="cpu", weights_only=False)
            eval_m = _make(aux=False)
            eval_m.load_state_dict(ck["model_state_dict"], strict=True)
            t_inf = time.perf_counter()
            va_lg = _score_fullpool_ranker(eval_m, va_bn, v_dev, dev, batch_size=48)
            inf_s = time.perf_counter() - t_inf
            ranked = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
            npz_p = vdir / f"predictions_{sel_name}.npz"
            ev = _evaluate_with_cohorts_and_sku(ranked, va_bn, sku_val, purchases, catalogue, product_ids, universe, npz_p)
            if ev["bn_metrics"] != ck["metrics"]:
                raise AssertionError(f"{vname} {sel_name} replay mismatch!")
            ev["epoch"] = int(ck["epoch"])
            ev["inference_seconds_6480"] = float(inf_s)
            ev["paired_vs_reference_ndcg"] = paired_base_name_bootstrap_ci(ref_ndcg_npz, npz_p)
            v_res[sel_name] = ev
        results["variants"][vname] = v_res
        del model
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    _verify_reference_files(reference)
    results["preserved_hashes_verified"] = True
    results["elapsed_seconds"] = float(time.perf_counter() - t_all)
    atomic_json(results, summary_path, overwrite=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", choices=("modest", "large"), default="modest")
    args = parser.parse_args()
    res = run_bigger_ranker(seed=args.seed, device=args.device, mode=args.mode)
    print(f"\nSaved {args.mode} Stage 2 ranker summary ({res['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
