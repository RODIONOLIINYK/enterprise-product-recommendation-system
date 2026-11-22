"""Fit the canonical first/returning formulation reference or replay a versioned Stage 5 experiment."""

import argparse
from collections import Counter
from tempfile import mkdtemp
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recommender.artifacts import atomic_json, file_digest
from recommender.catalogue import load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.embedding_ablation.config import build_ablation_specs
from recommender.experiments.rolling_candidate_v2.config import (
    DEFAULT_RETRIEVER,
    SLATE_POLICY,
)
from recommender.experiments.rolling_candidate_v2.train import snapshot_preserved_hashes
from recommender.experiments.static_enrichment.model import StaticEnrichedRetriever
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.train_retriever import get_device

from recommender.split import TRAIN_CUTOFF
from recommender.sequences import PURCHASE_DATE_ORIGIN

from .augmentation import build_strict_60d_aux_labels
from .base_name_features import (
    assemble_arm_features_numpy,
    prepare_or_load_base_name_feature_bundle,
)
from .commercial_features import (
    COMMERCIAL_FEATURE_NAMES,
    prepare_or_load_commercial_feature_bundle,
)
from .config import (
    ARM_BASE_NAME_ONLY,
    ARTIFACT_ROOT,
    EVIDENCE_ROOT,
    SEED,
    STAGE1_BATCH_SIZE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE1_POS_WEIGHT,
    STAGE1_WEIGHT_DECAY,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_ETA_MIN,
    STAGE2_GRAD_CLIP_NORM,
    STAGE2_LR,
    STAGE2_WEIGHT_DECAY,
    feature_names_for_arm,
)
from .first_baskets import FIRST_ORDER_GROUPS, append_first_baskets, prepare_first_baskets
from .parsed_spec_features import PARSED_SPEC_FEATURE_NAMES, prepare_or_load_parsed_spec_feature_bundle
from .stage5_artifacts import product_inputs_digest, vector_identity
from .stage1_mlp import RoadmapStage1Selector
from .stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    evaluate_base_name_predictions,
    resolve_base_names_to_skus,
)


# Pack-agnostic Stage 3 feature column subsets (excluding exact-SKU / container-pack columns)
PACK_AGNOSTIC_COMM_INDICES: Dict[str, Tuple[int, ...]] = {
    "3a_financial": tuple(range(0, 6)),
    "3b_pricing": tuple(range(6, 13)),
    "3c_manager": (14, 15, 16, 17),  # mgr_base_name_share_90d, mgr_brand_share_90d, mgr_cat_share_90d, mgr_changed_recently
    "3d_warehouse": (19, 20),        # wh_sku_shipped_90d (max over packs), wh_bn_shipped_30d
}
PACK_AGNOSTIC_SPEC_INDICES: Tuple[int, ...] = (0, 1, 6, 7)  # viscosity bought/share, equipment match, app share


def _pool_extra_stage3_to_base_name(
    comm_arr: np.ndarray,
    spec_arr: np.ndarray,
    rows: Any,
    groups: Sequence[str],
    bn_to_members: Sequence[Sequence[int]],
) -> Tuple[Optional[np.ndarray], Tuple[str, ...]]:
    """Pool selected Stage 3 groups to pack-agnostic (B_rows, 585, D_extra) float32 tensor."""
    if not groups:
        return None, ()
    parts: List[np.ndarray] = []
    names: List[str] = []
    n_bn = len(bn_to_members)

    # Pre-fetch slices once
    need_comm = any(g in PACK_AGNOSTIC_COMM_INDICES for g in groups)
    need_spec = "3e_parsed_specs" in groups
    c_sl = np.array(comm_arr[rows], dtype=np.float32) if need_comm else None
    s_sl = np.array(spec_arr[rows], dtype=np.float32) if need_spec else None

    for g in groups:
        if g in PACK_AGNOSTIC_COMM_INDICES:
            cols = list(PACK_AGNOSTIC_COMM_INDICES[g])
            names.extend(COMMERCIAL_FEATURE_NAMES[i] for i in cols)
            sub = c_sl[..., cols]  # (R, P, C)
            pooled = np.zeros((sub.shape[0], n_bn, len(cols)), dtype=np.float32)
            for b_i, members in enumerate(bn_to_members):
                if len(members) == 1:
                    pooled[:, b_i, :] = sub[:, members[0], :]
                elif g == "3d_warehouse":
                    pooled[:, b_i, :] = sub[:, members, :].max(axis=1)
                else:
                    pooled[:, b_i, :] = sub[:, members, :].mean(axis=1)
            parts.append(pooled)
        elif g == "3e_parsed_specs":
            cols = list(PACK_AGNOSTIC_SPEC_INDICES)
            names.extend(PARSED_SPEC_FEATURE_NAMES[i] for i in cols)
            sub = s_sl[..., cols]
            pooled = np.zeros((sub.shape[0], n_bn, len(cols)), dtype=np.float32)
            for b_i, members in enumerate(bn_to_members):
                if len(members) == 1:
                    pooled[:, b_i, :] = sub[:, members[0], :]
                else:
                    pooled[:, b_i, :] = sub[:, members, :].max(axis=1)
            parts.append(pooled)
        else:
            raise ValueError(f"Unknown Stage 3 group for Stage 5: {g!r}")
    return np.concatenate(parts, axis=-1), tuple(names)


class Stage5BaseNameRanker(nn.Module):
    """UPS-CANR pointwise ranker operating natively on canonical base_name entities (585 bodies)."""

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
        aux_60d_enabled: bool = True,
        seed: int = SEED,
    ):
        super().__init__()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.num_features = int(num_features)
        self.extra_dim = self.num_features - 13
        self.aux_60d_enabled = bool(aux_60d_enabled)

        self.base = UPSCANRRanker(
            text_dim=1024,
            memory_dim=128,
            num_cities=int(num_cities),
            num_regions=int(num_regions),
            num_industries=int(num_industries),
            num_products=int(num_base_names) + 1,  # 0 = padding, 1..585 = canonical base_names
            num_categories=int(num_categories),
            num_subcategories=int(num_subcategories),
            num_business_lines=int(num_business_lines),
            num_tabular_features=8,
            num_pairwise_features=13,
            seq_feature_dim=9,
            hidden_dim=128,
            num_heads=4,
            statistics_attention=True,
            excluded_statistics=("history_repeat",),
            cached_only=True,
        )
        self.base.mlp.requires_grad_(False)
        total_in = 128 * 3 + 13 + self.extra_dim
        self.head_trunk = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.main_head = nn.Linear(64, 1)
        self.aux_60d_head: Optional[nn.Linear] = nn.Linear(64, 1) if self.aux_60d_enabled else None

    def forward(
        self,
        *,
        pairwise_features: torch.Tensor,
        cached_customer_vectors: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        **kwargs,
    ) -> Any:
        encoded = self.base.encode_candidate_features(
            pairwise_features=pairwise_features[..., :13],
            memory_value_statistics=False,
            **kwargs,
        )
        fused = torch.cat([encoded.features, pairwise_features[..., 13:]], dim=-1)
        hidden = self.head_trunk(fused)
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


def _collapse_bool_matrix_to_bn(mat: torch.Tensor, bn_to_members: Sequence[Sequence[int]]) -> torch.Tensor:
    out = torch.zeros((mat.shape[0], len(bn_to_members)), dtype=torch.bool)
    for b_i, members in enumerate(bn_to_members):
        if len(members) == 1:
            out[:, b_i] = mat[:, members[0]]
        else:
            out[:, b_i] = mat[:, members].any(dim=1)
    return out


def _stage5_tower(
    universe,
    enrichment,
    num_demand_features: int = 12,
    num_customer_tabular_features: Optional[int] = None,
    drop_sales_branch: bool = False,
):
    from dataclasses import replace

    spec = build_ablation_specs()["E_no_brand"].variant_spec
    updates = {}
    if int(num_demand_features) != int(spec.product_demand_features):
        updates["product_demand_features"] = int(num_demand_features)
    if num_customer_tabular_features is not None and int(num_customer_tabular_features) != int(spec.customer_tabular_features):
        updates["customer_tabular_features"] = int(num_customer_tabular_features)
    if drop_sales_branch:
        updates["customer_embeddings"] = tuple(c for c in spec.customer_embeddings if c != "sales_branch")
    if updates:
        spec = replace(spec, **updates)
    state = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]
    return StaticEnrichedRetriever(
        spec,
        num_cities=state["customer_tower.city_embedding.weight"].shape[0],
        num_regions=state["customer_tower.region_embedding.weight"].shape[0],
        num_industries=state["customer_tower.industry_embedding.weight"].shape[0],
        num_catalog_products=universe["num_base_names"],
        num_demand_features=int(num_demand_features),
        product_attr_vocab_sizes={key: len(vocab) for key, vocab in enrichment["product_vocabs"].items()},
        customer_attr_vocab_sizes={key: len(vocab) for key, vocab in enrichment["customer_vocabs"].items()},
    )


@torch.no_grad()
def _encode_first_baskets(data, tower, product_vectors, device):
    inputs = ("text_embedding", "city_id", "region_id", "industry_id",
              "tabular_features", "sequence", "attention_mask", "seq_product_ids", "seq_purchase_dates")
    chunks = []
    for rows in torch.arange(len(data["customer_ids"])).split(256):
        values = {key: data[key][rows].to(device) for key in inputs}
        values["extra_attr_ids"] = {"sales_branch": data["sales_branch"][rows].to(device)}
        chunks.append(tower.encode_purchase(values, product_vectors).squeeze(1).cpu())
    data["u_bn"] = torch.cat(chunks)
    if not torch.isfinite(data["u_bn"]).all():
        raise ValueError("Non-finite vector for an empty history.")


def _verify_reference_files(reference):
    for artifact in reference["checkpoints"].values():
        if file_digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Stage 5 reference checksum mismatch: {artifact['path']}")


def train_stage5_two_tower(
    *,
    bn_data: Mapping[str, Dict[str, Any]],
    prod_inputs: Mapping[str, Any],
    universe: Mapping[str, Any],
    enrichment: Mapping[str, Any],
    out_dir: Path,
    seed: int = SEED,
    epochs: int = 30,
    device: Optional[str] = None,
    replay_checkpoint: Optional[Path] = None,
    num_customer_tabular_features: Optional[int] = None,
    drop_sales_branch: bool = False,
) -> Dict[str, Any]:
    """Train pack-agnostic Two-Tower retriever (E_no_brand recipe) natively over 585 canonical base_names."""
    dev = get_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").is_file() and (out_dir / "best.pt").is_file():
        return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    demand_dim = int(prod_inputs["demand_features"].shape[1])
    model = _stage5_tower(
        universe,
        enrichment,
        num_demand_features=demand_dim,
        num_customer_tabular_features=num_customer_tabular_features,
        drop_sales_branch=drop_sales_branch,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    tr = bn_data["train"]
    va = bn_data["val"]
    n_train_bn = int(universe["trained_base_names"])
    useful = torch.where((tr["target_bn"][:, :n_train_bn] & tr["unseen_bn"][:, :n_train_bn]).any(-1))[0]
    gen = torch.Generator().manual_seed(int(seed))

    def _p_inputs(limit: Optional[int] = None) -> Dict[str, Any]:
        end = limit if limit is not None else universe["num_base_names"]
        return {
            "text_embedding": prod_inputs["text_embedding"][:end].to(dev),
            "demand_features": prod_inputs["demand_features"][:end].to(dev),
            "attr_ids": {k: v[:end].to(dev) for k, v in prod_inputs["attr_ids"].items()},
        }

    def _c_inputs(split_d: Mapping[str, Any], rows: torch.Tensor) -> Dict[str, Any]:
        return {
            "text_embedding": split_d["text_embedding"][rows].to(dev),
            "city_id": split_d["city_id"][rows].to(dev),
            "region_id": split_d["region_id"][rows].to(dev),
            "industry_id": split_d["industry_id"][rows].to(dev),
            "tabular_features": split_d["tabular_features"][rows].to(dev),
            "sequence": split_d["sequence"][rows].to(dev),
            "attention_mask": split_d["attention_mask"][rows].to(dev),
            "seq_purchase_dates": split_d["seq_purchase_dates"][rows].to(dev),
            "seq_product_ids": split_d["seq_product_ids"][rows].to(dev),
            "extra_attr_ids": {"sales_branch": split_d["sales_branch"][rows].to(dev)},
        }

    @torch.no_grad()
    def _encode_split(split_d: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.eval()
        vecs = model.product_vectors(_p_inputs())
        idxs = torch.arange(len(vecs), device=dev)
        u_chunks, sc_chunks = [], []
        for rows in torch.arange(len(split_d["target_bn"])).split(256):
            u_b = model.encode_purchase(_c_inputs(split_d, rows), vecs)
            sc_b = model.score_cached(u_b, vecs, idxs, split_d["familiarity_bn"][rows].to(dev))
            u_chunks.append(u_b.squeeze(1).cpu())
            sc_chunks.append(sc_b.cpu())
        return vecs.cpu(), torch.cat(u_chunks, dim=0), torch.cat(sc_chunks, dim=0)

    train_p = _p_inputs(limit=n_train_bn)
    train_idxs = torch.arange(n_train_bn, device=dev)
    pw_t = torch.tensor(20.0, device=dev)

    best_key = (-1, -1)
    t0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1 if replay_checkpoint is None else 1):
        model.train()
        losses = []
        for rows in useful[torch.randperm(len(useful), generator=gen)].split(256):
            optimizer.zero_grad(set_to_none=True)
            vecs = model.product_vectors(train_p)
            u_b = model.encode_purchase(_c_inputs(tr, rows), vecs)
            logits = model.score_cached(u_b, vecs, train_idxs, tr["familiarity_bn"][rows, :n_train_bn].to(dev))
            allowed = tr["unseen_bn"][rows, :n_train_bn].to(dev)
            target = tr["target_bn"][rows, :n_train_bn].to(dev).float()
            loss = F.binary_cross_entropy_with_logits(logits[allowed], target[allowed], pos_weight=pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        _, _, va_logits = _encode_split(va)
        unseen_ord = va_logits.masked_fill(~va["unseen_bn"], -torch.inf).argsort(dim=-1, descending=True, stable=True)
        available_new = va["new_target_bn"] & va["unseen_bn"]
        h25 = int(available_new.gather(1, unseen_ord[:, :25]).sum().item())
        h50 = int(available_new.gather(1, unseen_ord[:, :50]).sum().item())
        if (h25, h50) > best_key:
            best_key = (h25, h50)
            atomic_torch_save({"epoch": ep, "h25": h25, "h50": h50, "model_state_dict": model.state_dict()}, out_dir / "best.pt")
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(
                f"[stage5/two-tower] ep={ep:02d}/{epochs} loss={np.mean(losses):.5f} "
                f"true_new_bn@25={h25}/{int(va['new_target_bn'].sum())} true_new_bn@50={h50}",
                flush=True,
            )

    ckpt = torch.load(replay_checkpoint or out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    v_bn, u_tr, tr_dot = _encode_split(tr)
    _, u_va, va_dot = _encode_split(va)

    order = va_dot.masked_fill(~va["unseen_bn"], -torch.inf).argsort(descending=True, stable=True, dim=-1)
    available_new = va["new_target_bn"] & va["unseen_bn"]
    for k in (25, 50):
        if int(available_new.gather(1, order[:, :k]).sum()) != ckpt[f"h{k}"]:
            raise ValueError(f"Two-Tower validation replay mismatch at depth {k}.")
    tower_hash = file_digest(replay_checkpoint or out_dir / "best.pt")
    inputs_hash = product_inputs_digest(prod_inputs)
    for name, split, u, scores in (("train", tr, u_tr, tr_dot), ("val", va, u_va, va_dot)):
        identity = vector_identity(split, universe["base_names"], tower_hash, inputs_hash)
        atomic_torch_save({**identity, "v_bn": v_bn, "u_bn": u, "dot_scores": scores},
                          out_dir / f"{name}_vectors.pt")
    res = {
        "selected_epoch": int(ckpt["epoch"]),
        "true_new_bn_hits@25": int(ckpt["h25"]),
        "true_new_bn_hits@50": int(ckpt["h50"]),
        "true_new_bn_targets": int(va["new_target_bn"].sum().item()),
        "training_seconds": float(time.perf_counter() - t0),
    }
    atomic_json(res, out_dir / "result.json", overwrite=True)
    return res


def run_stage5(
    *,
    stage3_groups: Optional[Sequence[str]] = None,
    aux_lambda: float = 0.3,
    sku_reference_dir: Optional[Path] = None,
    seed: int = SEED,
    device: Optional[str] = None,
    first_order_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if stage3_groups is None:
        if first_order_mode is None:
            if aux_lambda != 0.3:
                raise ValueError("The dated reference uses the 60-day objective at weight 0.3.")
            return run_dated_reference(seed=seed, device=device)
        stage3_groups = FIRST_ORDER_GROUPS
    dev = get_device(device)
    before_hashes = snapshot_preserved_hashes()
    reference = None
    if first_order_mode is not None:
        if first_order_mode not in ("control", "mixed"):
            raise ValueError("First-order mode must be control or mixed.")
        reference = json.loads((EVIDENCE_ROOT / "stage5_returning_reference.json").read_text())
        if (
            tuple(stage3_groups) != FIRST_ORDER_GROUPS
            or list(stage3_groups) != reference["feature_groups"]
            or seed != reference["seed"]
            or aux_lambda != reference["aux_lambda"]
            or reference["candidate_policy"] != SLATE_POLICY
            or reference["heldout_access"] is not False
        ):
            raise ValueError("First-order comparison must use the pinned Stage 5 recipe.")
        _verify_reference_files(reference)

    campaign_data, info, _, meta = prepare_or_load_base_name_feature_bundle(build=False)
    comm_mmaps, _ = prepare_or_load_commercial_feature_bundle(build=False)
    spec_mmaps, _ = prepare_or_load_parsed_spec_feature_bundle(build=False)
    enrichment = torch.load("data/interim/static_enrichment_v1/enrichment_bundle.pt", map_location="cpu", weights_only=False)

    purchases = load_development_purchases("data/interim/cleaned_purchases.csv")
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    product_ids = list(info["product_ids"])
    universe = build_base_name_universe(product_ids, catalogue, purchases)
    n_bn = universe["num_base_names"]
    bn_to_members = universe["bn_to_member_pid_indices"]
    bn_rep = universe["bn_rep_pid_idx"]

    print(
        f"[stage5] Canonical base_name universe: {n_bn} unique product bodies "
        f"({universe['trained_base_names']} trained < TRAIN_CUTOFF) from {len(product_ids)} SKUs.",
        flush=True,
    )

    # 1. Build pack-agnostic ProductTower inputs for 585 base_names (averaged t1_fam text, mode attrs, ZERO demand/container!)
    sku_text = enrichment["text_embeddings"]["t1_fam"].float()  # (841, 1024)
    bn_text = torch.zeros((n_bn, sku_text.shape[1]), dtype=torch.float32)
    for b_i, members in enumerate(bn_to_members):
        mean_v = sku_text[members].mean(dim=0)
        bn_text[b_i] = F.normalize(mean_v, dim=-1)

    spec_e = build_ablation_specs()["E_no_brand"].variant_spec
    bn_attrs: Dict[str, torch.Tensor] = {}
    for attr_name in spec_e.product_embeddings:
        sku_a = enrichment["product_attr_ids"][attr_name]
        vals = []
        for members in bn_to_members:
            # Mode across member SKUs
            cnt = Counter(sku_a[members].tolist())
            vals.append(cnt.most_common(1)[0][0])
        bn_attrs[attr_name] = torch.tensor(vals, dtype=torch.long)

    prod_inputs = {
        "text_embedding": bn_text,
        "demand_features": torch.zeros((n_bn, 12), dtype=torch.float32),  # zero demand AND zero container one-hot!
        "attr_ids": bn_attrs,
    }

    # 2. Build base_name-level train/val datasets (exact same 26,253 train & 6,191 val baskets!)
    # Retrieve Stage 2 vocabularies from shared cache manifest
    shared_manifest = torch.load("data/interim/shared_stage1_v1/manifest.pt", map_location="cpu", weights_only=False)
    vocabs = shared_manifest["vocabs"]

    # Familiarity at base_name level (4 flags: brand, category, business_line, family) — already precomputed strictly < date!
    bn_data: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val"):
        d = campaign_data[split]
        src = d["source"]
        cids = src["customer_ids"]
        days = src["prediction_days"].tolist()
        cohort_rows = d["cohort_rows"]

        target_bn = _collapse_bool_matrix_to_bn(d["target"], bn_to_members)
        allowed_bn = _collapse_bool_matrix_to_bn(d["allowed"], bn_to_members)
        # Strictly causal seen_bn from base_name_features col 0 (has_bought_base_sku < date)
        has_bought_bn = torch.from_numpy(
            np.array(d["base_name_features"][:, bn_rep, 0], dtype=np.float32) > 0.5
        )
        unseen_bn = allowed_bn & (~has_bought_bn)
        new_target_bn = target_bn & unseen_bn

        seqs = build_base_name_sequences(
            purchases, cids, days, universe, catalogue,
            vocabs["category_vocab"], vocabs["subcategory_vocab"], vocabs["business_line_vocab"],
            seq_len=32,
        )
        fam_bn = d["familiarity"][:, bn_rep].bool()
        sales_branch = enrichment[f"{split}_customer_attrs"]["sales_branch"][cohort_rows]

        # Build (N, 585, 41 + D_extra) strictly-causal base_name features
        f41_rep = np.array(d["features"][:, bn_rep, :], dtype=np.float32)
        f19_rep = np.array(d["base_name_features"][:, bn_rep, :], dtype=np.float32)
        feats_41 = assemble_arm_features_numpy(f41_rep, f19_rep, ARM_BASE_NAME_ONLY)
        extra_arr, extra_names = _pool_extra_stage3_to_base_name(
            comm_mmaps[split], spec_mmaps[split], slice(None), stage3_groups, bn_to_members
        )
        feats_all = np.concatenate([feats_41, extra_arr], axis=-1) if extra_arr is not None else feats_41

        bn_data[split] = {
            "customer_ids": cids,
            "prediction_days": src["prediction_days"],
            "text_embedding": src["text_embedding"],
            "city_id": src["city_id"],
            "region_id": src["region_id"],
            "industry_id": src["industry_id"],
            "tabular_features": src["tabular_features"],  # 8D for Stage 2
            "customer_stats_6": d["customer_stats_6"],    # 6D for Stage 1 MLP
            "sales_branch": sales_branch,
            "target_bn": target_bn,
            "allowed_bn": allowed_bn,
            "unseen_bn": unseen_bn,
            "new_target_bn": new_target_bn,
            "familiarity_bn": fam_bn,
            "features_bn": torch.from_numpy(feats_all).contiguous(),
            "feature_names": feature_names_for_arm(ARM_BASE_NAME_ONLY) + extra_names,
            **seqs,
        }

    # Verify exact validation true_new_bn count
    val_bn = bn_data["val"]
    print(
        f"[stage5] Val base_name targets: total={int(val_bn['target_bn'].sum())} | "
        f"TRULY NEW base_name targets={int(val_bn['new_target_bn'].sum())} (0 pack switches by construction!)",
        flush=True,
    )

    # 3. Stage 5.1 — Train pack-agnostic base_name Two-Tower
    tag = "base_only" if not stage3_groups else "plus_" + "_".join(stage3_groups)
    stage5_root = ARTIFACT_ROOT / "stage5_base_entity"
    tt_dir = stage5_root / f"two_tower-seed{seed}"
    if reference is not None:
        tt_dir = Path(reference["checkpoints"]["two_tower"]["path"]).parent
        stage5_root = ARTIFACT_ROOT / "stage5_first_order_v1" / first_order_mode
        ref_summary = json.loads(Path(reference["checkpoints"]["summary"]["path"]).read_text())
        tt_res = ref_summary["two_tower"]
    else:
        tt_res = train_stage5_two_tower(
            bn_data=bn_data, prod_inputs=prod_inputs, universe=universe, enrichment=enrichment,
            out_dir=tt_dir, seed=seed, epochs=30, device=str(dev),
        )
    tr_vecs = torch.load(tt_dir / "train_vectors.pt", map_location="cpu", weights_only=False)
    va_vecs = torch.load(tt_dir / "val_vectors.pt", map_location="cpu", weights_only=False)
    v_bn = tr_vecs["v_bn"]
    bn_data["train"]["u_bn"] = tr_vecs["u_bn"]
    bn_data["val"]["u_bn"] = va_vecs["u_bn"]

    first_val = None
    first_order_counts = None
    returning_train_count = len(bn_data["train"]["customer_ids"])
    if reference is not None:
        first, first_order_counts = prepare_first_baskets(
            purchases, catalogue, product_ids, universe, vocabs,
            enrichment["customer_vocabs"]["sales_branch"], _pool_extra_stage3_to_base_name,
        )
        print("[stage5/first-order] Cohorts:", json.dumps(first_order_counts), flush=True)
        tower = _stage5_tower(universe, enrichment)
        tower_checkpoint = torch.load(tt_dir / "best.pt", map_location="cpu", weights_only=False)
        tower.load_state_dict(tower_checkpoint["model_state_dict"], strict=True)
        tower.to(dev).eval()
        first_val = first["val"]
        _encode_first_baskets(first_val, tower, v_bn.to(dev), dev)
        if first_order_mode == "mixed":
            first_train = first["train"]
            _encode_first_baskets(first_train, tower, v_bn.to(dev), dev)
            familiarity = bn_data["train"]["familiarity_bn"]
            first_train["familiarity_bn"] = familiarity.new_zeros(
                (len(first_train["customer_ids"]), *familiarity.shape[1:])
            )
            bn_data["train"] = append_first_baskets(bn_data["train"], first_train)
            del first_train
        del first, tower, tower_checkpoint

    return _fit_and_evaluate(
        bn_data, v_bn, universe, purchases, catalogue, product_ids, vocabs, campaign_data,
        stage5_root, tag, stage3_groups, aux_lambda, seed, dev, tt_res, before_hashes,
        first_order_mode=first_order_mode, first_val=first_val, first_order_counts=first_order_counts,
        returning_train_count=returning_train_count, reference=reference,
    )


def run_dated_reference(*, seed=SEED, device=None, replay=False):
    from .dated_baskets import prepare_dated_baskets

    return _run_dated(prepare_dated_baskets(historical_recipe=replay), seed, device, replay)


def _run_dated(prepared, seed, device, replay):
    (bn_data, inputs, universe, enrichment, vocabs, purchases, catalogue,
     product_ids, sku_data, manifest) = prepared
    dev = get_device(device)
    root = ARTIFACT_ROOT / "stage5_dated_first_baskets"
    root.mkdir(parents=True, exist_ok=True)
    run = root / Path(mkdtemp(dir=root, prefix=f"seed{seed}-")).name
    tt_dir = run / "two_tower"
    before_hashes = snapshot_preserved_hashes()
    reference = None
    if replay:
        reference = json.loads((EVIDENCE_ROOT / "stage5_3stage_reference.json").read_text())
        _verify_reference_files(reference)
        saved = json.loads(Path(reference["checkpoints"]["summary"]["path"]).read_text())
        if (manifest != saved["data_manifest"] or seed != reference["seed"] or seed != saved["seed"]
                or reference["feature_groups"] != list(FIRST_ORDER_GROUPS)
                or reference["candidate_policy"] != SLATE_POLICY or reference["aux_lambda"] != 0.3
                or reference["heldout_access"] is not False):
            raise ValueError("Dated reference data or scoring recipe mismatch.")
    tt_res = train_stage5_two_tower(
        bn_data=bn_data, prod_inputs=inputs, universe=universe, enrichment=enrichment,
        out_dir=tt_dir, seed=seed, device=str(dev),
        replay_checkpoint=Path(reference["checkpoints"]["two_tower"]["path"]) if replay else None,
    )
    for split in ("train", "val"):
        vectors = torch.load(tt_dir / f"{split}_vectors.pt", map_location="cpu", weights_only=False)
        bn_data[split]["u_bn"] = vectors["u_bn"]
        if not torch.isfinite(vectors["u_bn"]).all():
            raise ValueError(f"Invalid {split} customer embeddings.")
    result = _fit_and_evaluate(
        bn_data, vectors["v_bn"], universe, purchases, catalogue, product_ids, vocabs, sku_data,
        run, "dated", FIRST_ORDER_GROUPS, 0.3, seed, dev, tt_res, before_hashes,
        reference=reference, first_order_mode="control" if replay else "dated", data_manifest=manifest,
    )
    if replay:
        for selection in ("ndcg", "discovery"):
            for key in ("cohorts", "bn_metrics", "resolved_sku_metrics"):
                if result["stage2_ranker"][selection][key] != saved["stage2_ranker"][selection][key]:
                    raise ValueError(f"Dated reference {selection} replay mismatch in {key}.")
        if result["stage2_slate_coverage"] != saved["stage2_slate_coverage"]:
            raise ValueError("Dated reference candidate coverage mismatch.")
    return result


def _fit_and_evaluate(
    bn_data, v_bn, universe, purchases, catalogue, product_ids, vocabs, campaign_data,
    stage5_root, tag, stage3_groups, aux_lambda, seed, dev, tt_res, before_hashes, *,
    first_order_mode=None, first_val=None, first_order_counts=None, returning_train_count=None,
    reference=None, data_manifest=None,
):
    val_bn = bn_data["val"]
    n_bn = universe["num_base_names"]
    bn_to_members = universe["bn_to_member_pid_indices"]
    # Build strictly leak-free 60-day base_name auxiliary labels (< TRAIN_CUTOFF)
    aux_60d_sku = build_strict_60d_aux_labels(
        purchases, bn_data["train"]["customer_ids"], bn_data["train"]["prediction_days"].tolist(), product_ids
    )
    aux_60d_bn = _collapse_bool_matrix_to_bn(aux_60d_sku, bn_to_members)

    # A truncated 60-day window cannot establish a negative for the remaining days.
    full_window = bn_data["train"]["prediction_days"] <= (TRAIN_CUTOFF - PURCHASE_DATE_ORIGIN).days - 60

    # 4. Stage 5.2 — Train Stage 1 MLP natively over 585 base_names (with 60d aux head stripped at inference)
    s1_dir = stage5_root / f"stage1_mlp-{tag}-seed{seed}"
    s1_dir.mkdir(parents=True, exist_ok=True)
    fnames = bn_data["train"]["feature_names"]
    use_aux = aux_lambda > 0.0

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    s1_model = RoadmapStage1Selector(fnames, aux_60d_enabled=use_aux).to(dev)
    opt1 = torch.optim.AdamW(s1_model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS, eta_min=1e-5)
    v_dev = v_bn.to(dev)
    pw_t = torch.tensor(float(STAGE1_POS_WEIGHT), device=dev)
    aux_pw_t = torch.tensor(10.0, device=dev)
    gen1 = torch.Generator().manual_seed(int(seed))

    @torch.no_grad()
    def _score_s1(sp_d: Mapping[str, Any]) -> torch.Tensor:
        s1_model.eval()
        chunks = []
        for st in range(0, len(sp_d["u_bn"]), 128):
            en = min(st + 128, len(sp_d["u_bn"]))
            lg = s1_model(
                u=sp_d["u_bn"][st:en].to(dev),
                v=v_dev,
                features=sp_d["features_bn"][st:en].to(dev),
                customer_stats_6=sp_d["customer_stats_6"][st:en].to(dev),
            )
            chunks.append(lg.masked_fill(~sp_d["allowed_bn"][st:en].to(dev), -1e9).cpu())
        return torch.cat(chunks, dim=0)

    best_s1_key = (-1, -1.0, -1.0)
    best_s1_ep = 0
    for ep in range(1, STAGE1_EPOCHS + 1 if first_order_mode != "control" else 1):
        s1_model.train()
        perm = torch.randperm(len(bn_data["train"]["u_bn"]), generator=gen1)
        for rows in perm.split(STAGE1_BATCH_SIZE):
            opt1.zero_grad(set_to_none=True)
            allowed = bn_data["train"]["allowed_bn"][rows].to(dev)
            target = bn_data["train"]["target_bn"][rows].to(dev).float()
            if use_aux:
                m_lg, a_lg = s1_model(
                    u=bn_data["train"]["u_bn"][rows].to(dev),
                    v=v_dev,
                    features=bn_data["train"]["features_bn"][rows].to(dev),
                    customer_stats_6=bn_data["train"]["customer_stats_6"][rows].to(dev),
                    return_aux=True,
                )
                loss = F.binary_cross_entropy_with_logits(m_lg[allowed], target[allowed], pos_weight=pw_t)
                aux_mask = allowed if data_manifest is None else allowed & full_window[rows, None].to(dev)
                if aux_mask.any():
                    loss = loss + float(aux_lambda) * F.binary_cross_entropy_with_logits(
                        a_lg[aux_mask], aux_60d_bn[rows].to(dev).float()[aux_mask], pos_weight=aux_pw_t)
            else:
                m_lg = s1_model(
                    u=bn_data["train"]["u_bn"][rows].to(dev),
                    v=v_dev,
                    features=bn_data["train"]["features_bn"][rows].to(dev),
                    customer_stats_6=bn_data["train"]["customer_stats_6"][rows].to(dev),
                )
                loss = F.binary_cross_entropy_with_logits(m_lg[allowed], target[allowed], pos_weight=pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s1_model.parameters(), 1.0)
            opt1.step()
        sch1.step()
        va_sc = _score_s1(val_bn)
        ranked = va_sc.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        m1 = evaluate_base_name_predictions(ranked, val_bn["target_bn"], val_bn["new_target_bn"], val_bn["customer_ids"])
        key = (int(m1["true_new_bn_hits@50"]), float(m1["bn_ndcg@10"]), float(m1["bn_recall@10"]))
        if key > best_s1_key:
            best_s1_key = key
            best_s1_ep = ep
            inf_state = {k: v for k, v in s1_model.state_dict().items() if not k.startswith("aux_60d_head.")}
            atomic_torch_save({"epoch": ep, "metrics": m1, "model_state_dict": inf_state}, s1_dir / "best.pt")
        print(
            f"[stage5/S1/{tag}] ep={ep:02d} bnR@5={m1['bn_recall@5']:.5f} bnR@10={m1['bn_recall@10']:.5f} "
            f"bnNDCG@10={m1['bn_ndcg@10']:.5f} trueNew@10={m1['true_new_bn_hits@10']} "
            f"trueNew@25={m1['true_new_bn_hits@25']} trueNew@50={m1['true_new_bn_hits@50']}",
            flush=True,
        )

    s1_path = Path(reference["checkpoints"]["selector"]["path"]) if first_order_mode == "control" else s1_dir / "best.pt"
    s1_ckpt = torch.load(s1_path, map_location="cpu", weights_only=False)
    best_s1_ep = int(s1_ckpt["epoch"])
    s1_inf = RoadmapStage1Selector(fnames, aux_60d_enabled=False).to(dev)
    s1_inf.load_state_dict(s1_ckpt["model_state_dict"], strict=True)
    s1_model = s1_inf
    s1_tr_scores = _score_s1(bn_data["train"])
    s1_va_scores = _score_s1(val_bn)
    s1_ranked = s1_va_scores.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
    s1_metrics = evaluate_base_name_predictions(
        s1_ranked, val_bn["target_bn"], val_bn["new_target_bn"], val_bn["customer_ids"],
        save_path=s1_dir / "bn_predictions.npz",
    )
    if data_manifest is not None and s1_metrics != s1_ckpt["metrics"]:
        raise ValueError("Dated selector validation replay mismatch.")
    first_order_results = None
    if first_val is not None:
        first_scores = _score_s1(first_val)
        first_idx = select_candidates(first_scores, first_val["allowed_bn"], first_val["unseen_bn"], SLATE_POLICY)
        first_mask = first_val["allowed_bn"].gather(1, first_idx)
        first_ranked = first_scores.argsort(dim=-1, descending=True, stable=True)[:, :50].numpy()
        first_order_results = {
            "cohorts": first_order_counts,
            "returning_train_rows": returning_train_count,
            "total_train_rows": len(bn_data["train"]["customer_ids"]),
            "stage1_mlp": evaluate_base_name_predictions(
                first_ranked, first_val["target_bn"], first_val["new_target_bn"], first_val["customer_ids"],
            ),
            "covered_targets": int(first_val["target_bn"].gather(1, first_idx).sum()),
            "stage2_ranker": {},
        }

    # 5. Build Stage 5 quota100_50 slates over 585 base_names!
    tr_idx = select_candidates(s1_tr_scores, bn_data["train"]["allowed_bn"], bn_data["train"]["unseen_bn"], SLATE_POLICY)
    va_idx = select_candidates(s1_va_scores, val_bn["allowed_bn"], val_bn["unseen_bn"], SLATE_POLICY)
    va_slate_tot = int(val_bn["target_bn"].gather(1, va_idx).sum().item())
    va_slate_new = int(val_bn["new_target_bn"].gather(1, va_idx).sum().item())
    print(
        f"[stage5] base_name quota100_50 val slate coverage: total={va_slate_tot}/{int(val_bn['target_bn'].sum())} "
        f"({100*va_slate_tot/int(val_bn['target_bn'].sum()):.2f}%), "
        f"true_new={va_slate_new}/{int(val_bn['new_target_bn'].sum())} ({100*va_slate_new/int(val_bn['new_target_bn'].sum()):.2f}%)",
        flush=True,
    )

    # 6. Stage 5.3 — Train Stage 2 UPS-CANR Transformer natively over 585 base_name entities!
    s2_dir = stage5_root / f"stage2_ranker-{tag}-seed{seed}"
    s2_dir.mkdir(parents=True, exist_ok=True)
    orig_s = torch.load(DEFAULT_RETRIEVER, map_location="cpu", weights_only=False)["model_state_dict"]
    s2_model = Stage5BaseNameRanker(
        num_base_names=n_bn,
        num_features=len(fnames),
        num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
        num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
        num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
        num_categories=len(vocabs["category_vocab"]) + 1,
        num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
        num_business_lines=len(vocabs["business_line_vocab"]) + 1,
        aux_60d_enabled=use_aux,
        seed=seed,
    ).to(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    trainable = [p for p in s2_model.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(trainable, lr=STAGE2_LR, weight_decay=STAGE2_WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, STAGE2_EPOCHS, eta_min=STAGE2_ETA_MIN)
    loss_fn = RankerLoss().to(dev)
    aux_loss_fn = RankerLoss(pos_weight=5.0, listwise_weight=1.0).to(dev)

    tr_bn_aux_slate = aux_60d_bn.gather(1, tr_idx).float()
    tr_bn_labels = bn_data["train"]["target_bn"].gather(1, tr_idx).float()
    tr_bn_mask = bn_data["train"]["allowed_bn"].gather(1, tr_idx)
    va_bn_mask = val_bn["allowed_bn"].gather(1, va_idx)

    def _s2_batch_inputs(sp_d: Mapping[str, Any], cand_idx: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor) -> Dict[str, Any]:
        c_b = cand_idx[rows]
        row_ar = rows[:, None]
        pf = sp_d["features_bn"][row_ar, c_b].to(dev, dtype=torch.float32)
        c_dev = c_b.to(dev)
        return {
            "text_embedding": sp_d["text_embedding"][rows].to(dev),
            "city_id": sp_d["city_id"][rows].to(dev),
            "region_id": sp_d["region_id"][rows].to(dev),
            "industry_id": sp_d["industry_id"][rows].to(dev),
            "tabular_features": sp_d["tabular_features"][rows].to(dev),
            "sequence": sp_d["sequence"][rows].to(dev),
            "attention_mask": sp_d["attention_mask"][rows].to(dev),
            "seq_product_ids": sp_d["seq_product_ids"][rows].to(dev),
            "seq_category_ids": sp_d["seq_category_ids"][rows].to(dev),
            "seq_subcategory_ids": sp_d["seq_subcategory_ids"][rows].to(dev),
            "seq_business_line_ids": sp_d["seq_business_line_ids"][rows].to(dev),
            "seq_purchase_dates": sp_d["seq_purchase_dates"][rows].to(dev),
            "prediction_day": sp_d["prediction_days"][rows].to(dev),
            "pairwise_features": pf,
            "candidate_vectors": v_dev[c_dev],
            "candidate_mask": mask[rows].to(dev),
            "candidate_indices": c_dev,
        }

    @torch.no_grad()
    def _predict_s2_val(
        model_eval: Stage5BaseNameRanker,
        data: Mapping[str, Any] = val_bn,
        candidate_idx: torch.Tensor = va_idx,
        candidate_mask: torch.Tensor = va_bn_mask,
    ) -> torch.Tensor:
        model_eval.eval()
        chunks = []
        for rows in torch.arange(len(data["customer_ids"])).split(128):
            inp = _s2_batch_inputs(data, candidate_idx, candidate_mask, rows)
            chunks.append(model_eval(**inp).logits.cpu())
        return torch.cat(chunks, dim=0)

    gen2 = torch.Generator().manual_seed(int(seed))
    best_ndcg_key = (-1.0, -1, -1.0)
    best_disc_key = (-1, -1.0, -1.0)
    for ep in range(1, STAGE2_EPOCHS + 1 if first_order_mode != "control" else 1):
        ep_t0 = time.perf_counter()
        s2_model.train()
        losses = []
        perm = torch.randperm(len(bn_data["train"]["customer_ids"]), generator=gen2)
        for rows in perm.split(STAGE2_BATCH_SIZE):
            opt2.zero_grad(set_to_none=True)
            inp = _s2_batch_inputs(bn_data["train"], tr_idx, tr_bn_mask, rows)
            y = tr_bn_labels[rows].to(dev)
            if use_aux:
                out, a_lg = s2_model(**inp, return_aux=True)
                l_main = loss_fn(out.logits, y, candidate_mask=inp["candidate_mask"])["loss"]
                aux_rows = full_window[rows].to(dev) if data_manifest is not None else torch.ones_like(y[:, 0], dtype=torch.bool)
                loss = l_main
                if aux_rows.any():
                    l_aux = aux_loss_fn(a_lg[aux_rows], tr_bn_aux_slate[rows].to(dev)[aux_rows],
                                        candidate_mask=inp["candidate_mask"][aux_rows])["loss"]
                    loss = loss + float(aux_lambda) * l_aux
            else:
                out = s2_model(**inp)
                loss = loss_fn(out.logits, y, candidate_mask=inp["candidate_mask"])["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, STAGE2_GRAD_CLIP_NORM)
            opt2.step()
            losses.append(float(loss.detach().cpu()))
        sch2.step()

        va_lg = _predict_s2_val(s2_model)
        top_pos = va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50]
        ranked_bn = va_idx.gather(1, top_pos).numpy()
        m2 = evaluate_base_name_predictions(ranked_bn, val_bn["target_bn"], val_bn["new_target_bn"], val_bn["customer_ids"])
        inf_state = {k: v for k, v in s2_model.state_dict().items() if not k.startswith("aux_60d_head.")}

        nk = (float(m2["bn_ndcg@10"]), int(m2["true_new_bn_hits@10"]), float(m2["bn_recall@10"]))
        dk = (int(m2["true_new_bn_hits@10"]), float(m2["bn_ndcg@10"]), float(m2["bn_recall@10"]))
        if nk > best_ndcg_key:
            best_ndcg_key = nk
            atomic_torch_save({"epoch": ep, "metrics": m2, "model_state_dict": inf_state}, s2_dir / "best_ndcg.pt")
        if dk > best_disc_key:
            best_disc_key = dk
            atomic_torch_save({"epoch": ep, "metrics": m2, "model_state_dict": inf_state}, s2_dir / "best_discovery.pt")
        print(
            f"[stage5/S2/{tag}] ep={ep:02d} loss={np.mean(losses):.4f} "
            f"bnR@5={m2['bn_recall@5']:.5f} bnR@10={m2['bn_recall@10']:.5f} "
            f"bnNDCG@10={m2['bn_ndcg@10']:.5f} trueNew@10={m2['true_new_bn_hits@10']}/{m2['true_new_bn_targets']} "
            f"({time.perf_counter() - ep_t0:.1f}s)",
            flush=True,
        )

    # Replay best NDCG & Discovery checkpoints and also evaluate causal pack-size resolution to concrete SKUs!
    s2_selections: Dict[str, Any] = {}
    for sel_name, fname in (("ndcg", "best_ndcg.pt"), ("discovery", "best_discovery.pt")):
        reference_key = "ranker" if sel_name == "ndcg" else "discovery_ranker"
        s2_path = Path(reference["checkpoints"][reference_key]["path"]) if first_order_mode == "control" else s2_dir / fname
        ck = torch.load(s2_path, map_location="cpu", weights_only=False)
        eval_m = Stage5BaseNameRanker(
            num_base_names=n_bn,
            num_features=len(fnames),
            num_cities=orig_s["customer_tower.city_embedding.weight"].shape[0],
            num_regions=orig_s["customer_tower.region_embedding.weight"].shape[0],
            num_industries=orig_s["customer_tower.industry_embedding.weight"].shape[0],
            num_categories=len(vocabs["category_vocab"]) + 1,
            num_subcategories=len(vocabs["subcategory_vocab"]) + 1,
            num_business_lines=len(vocabs["business_line_vocab"]) + 1,
            aux_60d_enabled=False,
            seed=seed,
        ).to(dev)
        eval_m.load_state_dict(ck["model_state_dict"], strict=True)
        va_lg = _predict_s2_val(eval_m)
        ranked_bn = va_idx.gather(1, va_lg.argsort(dim=-1, descending=True, stable=True)[:, :50]).numpy()
        npz_path = s2_dir / f"bn_predictions_{sel_name}.npz"
        replayed = evaluate_base_name_predictions(
            ranked_bn, val_bn["target_bn"], val_bn["new_target_bn"], val_bn["customer_ids"], save_path=npz_path
        )
        cohorts = {}
        if data_manifest is not None:
            for name, mask in (("first_observed", val_bn["first_observed"]), ("returning", ~val_bn["first_observed"])):
                ids = [cid for cid, keep in zip(val_bn["customer_ids"], mask.tolist()) if keep]
                cohorts[name] = evaluate_base_name_predictions(
                    ranked_bn[mask.numpy()], val_bn["target_bn"][mask], val_bn["new_target_bn"][mask], ids)
        if replayed != ck["metrics"]:
            raise AssertionError(f"Stage 5 S2 {sel_name} replay mismatch!")
        if first_order_results is not None:
            scores = _predict_s2_val(eval_m, first_val, first_idx, first_mask)
            ranked = first_idx.gather(1, scores.argsort(dim=-1, descending=True, stable=True)[:, :50]).numpy()
            first_predictions = s2_dir / f"first_basket_predictions_{sel_name}.npz"
            first_order_results["stage2_ranker"][sel_name] = {
                "metrics": evaluate_base_name_predictions(
                    ranked, first_val["target_bn"], first_val["new_target_bn"], first_val["customer_ids"],
                    save_path=first_predictions,
                ),
                "predictions_npz": str(first_predictions),
            }

        # Optional pack-size resolution to concrete SKUs (top-10)
        resolved_skus = resolve_base_names_to_skus(
            ranked_bn, val_bn["customer_ids"], val_bn["prediction_days"].tolist(),
            campaign_data["val"]["allowed"], purchases, catalogue, product_ids, universe, top_k=10,
        )
        t_sku = campaign_data["val"]["target"].numpy().astype(bool)
        nt_sku = campaign_data["val"]["new_target"].numpy().astype(bool)
        disc_arr = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
        sku_r10, sku_ndcg10 = [], []
        sku_new10 = 0
        for r_i in range(len(resolved_skus)):
            row_p = resolved_skus[r_i]
            valid_p = row_p[row_p >= 0]
            rel = t_sku[r_i, valid_p].astype(np.float64)
            n_pos = int(t_sku[r_i].sum())
            if n_pos > 0:
                sku_r10.append(float(rel.sum()) / n_pos)
                dcg = float((rel * disc_arr[: len(rel)]).sum())
                idcg = float(disc_arr[: min(n_pos, 10)].sum())
                sku_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
            sku_new10 += int(nt_sku[r_i, valid_p].sum())

        s2_selections[sel_name] = {
            "epoch": int(ck["epoch"]),
            "bn_metrics": replayed,
            **({"cohorts": cohorts} if cohorts else {}),
            "bn_predictions_npz": str(npz_path),
            "resolved_sku_metrics": {
                "sku_recall@10": float(np.mean(sku_r10)),
                "sku_ndcg@10": float(np.mean(sku_ndcg10)),
                "sku_new_hits@10": int(sku_new10),
            },
        }

    if snapshot_preserved_hashes() != before_hashes:
        raise AssertionError("Preserved reference hashes changed!")
    if reference is not None:
        _verify_reference_files(reference)

    summary = {
        "stage": "stage5_base_name_canonical_entity",
        "tag": tag,
        "stage3_groups": list(stage3_groups),
        "aux_lambda": float(aux_lambda),
        "seed": int(seed),
        "heldout_access": False,
        "num_base_names": n_bn,
        "trained_base_names": int(universe["trained_base_names"]),
        "two_tower": tt_res,
        "stage1_mlp": {"selected_epoch": best_s1_ep, "bn_metrics": s1_metrics, "bn_predictions_npz": str(s1_dir / "bn_predictions.npz")},
        "stage2_slate_coverage": {"total": va_slate_tot, "true_new": va_slate_new},
        "stage2_ranker": s2_selections,
        "preserved_hashes_verified": True,
    }
    if data_manifest is not None:
        summary.update(data_manifest=data_manifest, mode="dated_mixed", run_directory=str(stage5_root))
        atomic_json(summary, stage5_root / "summary.json")
        return summary
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if first_order_results is not None:
        summary.update(first_order=first_order_results, reference_id=reference["reference_id"], mode=first_order_mode)
        filename = f"stage5_first_order_{first_order_mode}_seed{seed}.json"
    else:
        filename = f"stage5_{tag}_summary.json"
    atomic_json(summary, EVIDENCE_ROOT / filename, overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="*", default=None, help="Run a historical Stage 5 feature ablation.")
    parser.add_argument("--aux-lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--first-order-mode", choices=("control", "mixed"), help="Reproduce the earlier frozen-tower comparison.")
    parser.add_argument("--replay-reference", action="store_true", help="Verify and evaluate the pinned dated reference.")
    args = parser.parse_args()
    if args.groups is None and args.first_order_mode is None:
        if args.aux_lambda != 0.3:
            parser.error("The canonical reference uses --aux-lambda 0.3.")
        result = run_dated_reference(seed=args.seed, device=args.device, replay=args.replay_reference)
        print(result["run_directory"], json.dumps(result["stage2_ranker"]["ndcg"]["bn_metrics"], indent=2))
        return
    if args.replay_reference:
        parser.error("Reference replay uses its pinned feature groups and cohort.")
    summary = run_stage5(
        stage3_groups=args.groups or (FIRST_ORDER_GROUPS if args.first_order_mode else ()),
        aux_lambda=args.aux_lambda, seed=args.seed, device=args.device, first_order_mode=args.first_order_mode,
    )
    print("\n================ STAGE 5 BASE_NAME-CANONICAL SUMMARY ================", flush=True)
    print("Stage 1 MLP:", json.dumps(summary["stage1_mlp"]["bn_metrics"], indent=2))
    for sel in ("ndcg", "discovery"):
        print(f"Stage 2 [{sel}]:", json.dumps(summary["stage2_ranker"][sel]["bn_metrics"], indent=2))
        print(f"Stage 2 [{sel}] resolved SKU:", json.dumps(summary["stage2_ranker"][sel]["resolved_sku_metrics"], indent=2))
        if args.first_order_mode:
            print(f"First basket [{sel}]:", json.dumps(summary["first_order"]["stage2_ranker"][sel]["metrics"], indent=2))


if __name__ == "__main__":
    main()
