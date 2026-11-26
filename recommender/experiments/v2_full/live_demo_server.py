#!/usr/bin/env python3
"""Standalone Live Python PyTorch Inference Server for B2B AI Product Recommendations.

Zero external repository imports — requires only Python 3.9+, PyTorch (`torch`), and NumPy (`numpy`).
Loads the promoted `Wave2BE2ERanker` checkpoint (`best_discovery.pt`, Ep 10 `w2_2mem_recency_hl365`),
runs `ProductTower` once at startup to cache all 1,141 product vectors, and executes live single-order
PyTorch CPU inference (`~2–3 ms` per order) over real held-out Test orders (`2026-07-01 .. 2026-11-20`).

Works in two modes automatically:
  1. Portable ZIP mode (default): loads `./model/best_discovery.pt` + `./model/test_live_bundle.pt.gz`
     (1,000 real held-out Test orders: 700 Trained C&P + 300 Unlearned).
  2. Full-dataset repo mode (`--data-dir data/v2_full/interim/test_live_cache`): loads ALL 13,184
     real held-out Test orders (9,839 Trained C&P + 3,345 Unlearned) via zero-copy memmap.

Usage:
    python live_demo_server.py
    python live_demo_server.py --port 8799 --no-browser
    python live_demo_server.py --data-dir data/v2_full/interim/test_live_cache --port 8799
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date as dt_date, timedelta
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import platform
import random
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Exact Wave2BE2ERanker + StaticEnrichedProductTower + OptionAUnseenHead
# (standalone mirror of recommender/experiments/v2_full/wave2b_models.py)
# ---------------------------------------------------------------------------

DAYS_PER_YEAR: float = 365.25
HISTORY_AGE_INDEX: int = 1
PURCHASE_ORIGIN = dt_date(2020, 1, 1)
CAND_42_NON_REPEAT_INDICES: Tuple[int, ...] = tuple(range(5, 42))
UNSEEN_37_GLOBAL_LOCAL_INDICES: Tuple[int, ...] = (3, 13, 14, 15, 16, 17, 18)

DRIVER_IDX: Dict[str, int] = {
    "has_bought_base_sku": 0,
    "user_base_sku_order_count_norm": 1,
    "days_since_last_base_sku_buy_norm": 3,
    "base_sku_urgency": 4,
    "base_name_industry_affinity": 9,
    "base_name_recent3_lift": 25,
    "mgr_base_name_share_90d": 31,
    "user_viscosity_order_share": 34,
    "app_matches_company_equipment": 35,
    "mgr_base_name_share_30d": 37,
    "mgr_industry_base_name_share_90d": 38,
    "region_base_name_share_90d": 39,
    "full_cf_cosine": 40,
    "full_cf_recency_weighted": 41,
}


class StaticEnrichedProductTower(nn.Module):
    def __init__(
        self,
        attribute_vocab_sizes: Mapping[str, int],
        *,
        text_dim: int = 1024,
        output_dim: int = 128,
        hidden_dim: int = 128,
        num_demand_features: int = 65,
        attr_emb_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attr_order = tuple(attribute_vocab_sizes.keys())
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attr_embeddings = nn.ModuleDict(
            {name: nn.Embedding(int(size), attr_emb_dim) for name, size in attribute_vocab_sizes.items()}
        )
        total_in = hidden_dim + attr_emb_dim * len(self.attr_order) + num_demand_features
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        demand_features: torch.Tensor,
        attr_ids: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = [self.text_proj(text_embedding)]
        for name in self.attr_order:
            parts.append(self.attr_embeddings[name](attr_ids[name]))
        parts.append(demand_features)
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), p=2, dim=-1)


class OptionAUnseenHead(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        non_repeat_indices: Sequence[int] = CAND_42_NON_REPEAT_INDICES,
        global_local_indices: Sequence[int] = UNSEEN_37_GLOBAL_LOCAL_INDICES,
        mlp_widths: Sequence[int] = (384, 192, 96),
    ):
        super().__init__()
        self.non_repeat_indices = tuple(int(i) for i in non_repeat_indices)
        global_mask = torch.zeros(len(self.non_repeat_indices), dtype=torch.float32)
        global_mask[list(global_local_indices)] = 1.0
        self.register_buffer("global_only_mask", global_mask, persistent=False)

        self.mlp_blocks = nn.ModuleList()
        prev = d_model * 3 + len(self.non_repeat_indices)
        for idx, width in enumerate(mlp_widths):
            w = int(width)
            dp = 0.25 if idx == 0 else (0.1 if idx == len(mlp_widths) - 1 else 0.2)
            self.mlp_blocks.append(
                nn.Sequential(nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(dp))
            )
            prev = w
        self.main_head = nn.Linear(prev, 1)

    def forward(self, v_exp: torch.Tensor, h_att: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        unseen_tab = candidate_features[..., list(self.non_repeat_indices)]
        hidden = torch.cat([v_exp, h_att, v_exp * h_att, unseen_tab], dim=-1)
        for block in self.mlp_blocks:
            hidden = block(hidden)
        return self.main_head(hidden).squeeze(-1)


class StandaloneWave2BE2ERanker(nn.Module):
    """Exact inference mirror of `Wave2BE2ERanker` (`strict=True` checkpoint compatible)."""

    def __init__(self, arch: Mapping[str, Any]):
        super().__init__()
        self.d_model = int(arch["d_model"])
        self.num_heads = int(arch["num_heads"])
        self.memory_layers = int(arch.get("memory_layers", 2))
        self.max_seq_len = int(arch.get("max_seq_len", 32))
        self.num_candidate_features = int(arch.get("num_candidate_features", 42))
        self.num_customer_features = int(arch.get("num_customer_features", 7))
        self.num_pairwise_bias_features = int(arch.get("num_pairwise_bias_features", 42))
        self.active_seq_stat_indices = tuple(int(i) for i in arch.get("active_seq_stat_indices", (0, 1, 5, 6, 8)))
        self.num_hist_key_stats = len(self.active_seq_stat_indices)
        self.pure_v_query = bool(arch.get("pure_v_query", True))
        dropout = float(arch.get("dropout", 0.12))
        mlp_widths = tuple(int(w) for w in arch.get("mlp_widths", (384, 192, 96)))

        self.product_tower = StaticEnrichedProductTower(
            attribute_vocab_sizes=arch["product_attr_vocab_sizes"],
            text_dim=1024,
            output_dim=self.d_model,
            hidden_dim=self.d_model,
            num_demand_features=int(arch.get("num_demand_features", 65)),
            attr_emb_dim=16,
            dropout=0.1,
        )
        self.cust_text_proj = nn.Sequential(
            nn.Linear(1024, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.city_embedding = nn.Embedding(int(arch["num_cities"]), 16, padding_idx=0)
        self.region_embedding = nn.Embedding(int(arch["num_regions"]), 16, padding_idx=0)
        self.industry_embedding = nn.Embedding(int(arch["num_industries"]), 16, padding_idx=0)
        tab_in = 16 * 3 + self.num_customer_features
        self.cust_tab_proj = nn.Sequential(
            nn.Linear(tab_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.token0_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(2, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=int(arch.get("memory_ffn_dim", 384)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.memory_layers, enable_nested_tensor=False
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.statistics_attention_weights = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.num_pairwise_bias_features + 1,
                self.num_hist_key_stats + self.num_customer_features,
            )
        )

        self.mlp_blocks = nn.ModuleList()
        prev = self.d_model * 3 + self.num_candidate_features
        for idx, width in enumerate(mlp_widths):
            dp = 0.25 if idx == 0 else (0.1 if idx == len(mlp_widths) - 1 else 0.2)
            self.mlp_blocks.append(
                nn.Sequential(nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dp))
            )
            prev = width
        self.main_head = nn.Linear(prev, 1)
        self.option_a_head = OptionAUnseenHead(d_model=self.d_model, mlp_widths=mlp_widths)

    @torch.no_grad()
    def precompute_product_vectors(self, prod_inputs: Mapping[str, Any]) -> torch.Tensor:
        return self.product_tower(
            prod_inputs["text_embedding"].float(),
            prod_inputs["demand_features"].float(),
            prod_inputs["attr_ids"],
        )

    @torch.no_grad()
    def score_single(
        self,
        *,
        company_text_embedding: torch.Tensor,
        city_id: torch.Tensor,
        region_id: torch.Tensor,
        industry_id: torch.Tensor,
        customer_features: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_product_ids: torch.Tensor,
        seq_purchase_dates: torch.Tensor,
        prediction_day: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        v_bn: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        b_size = company_text_embedding.size(0)
        seq_len = attention_mask.size(1)
        c_slate = candidate_features.size(1)

        purchase_days = (seq_purchase_dates * DAYS_PER_YEAR).round()
        age_days = (prediction_day[:, None] - purchase_days).clamp_min(0)
        age = age_days / (age_days + 90.0)

        z_text = self.cust_text_proj(company_text_embedding)
        z_tab = self.cust_tab_proj(
            torch.cat(
                [
                    self.city_embedding(city_id),
                    self.region_embedding(region_id),
                    self.industry_embedding(industry_id),
                    customer_features,
                ],
                dim=-1,
            )
        )
        t_0 = (self.token0_fusion(torch.cat([z_text, z_tab], dim=-1)) + self.type_embedding.weight[0]).unsqueeze(1)

        v_padded = F.pad(v_bn, (0, 0, 1, 0))
        t_hist = F.embedding(seq_product_ids, v_padded, padding_idx=0)
        t_hist = t_hist + self.pos_embedding[:, :seq_len, :] + self.type_embedding.weight[1]

        m_raw = torch.cat([t_0, t_hist], dim=1)
        k_mask = torch.zeros((b_size, seq_len + 1), dtype=torch.bool, device=attention_mask.device)
        k_mask[:, 1:] = attention_mask == 0
        memory = torch.nan_to_num(self.memory_encoder(m_raw, src_key_padding_mask=k_mask), nan=0.0)

        v_exp = v_bn.unsqueeze(0).expand(b_size, -1, -1)
        seq_clean = sequence.clone()
        seq_clean[..., HISTORY_AGE_INDEX] = age
        active_hist_stats = seq_clean[..., list(self.active_seq_stat_indices)]

        key_statistics = torch.cat(
            (
                F.pad(customer_features[:, None, :], (self.num_hist_key_stats, 0)),
                F.pad(active_hist_stats, (0, self.num_customer_features)),
            ),
            dim=1,
        )

        cand_bias_feats = candidate_features[..., : self.num_pairwise_bias_features]
        candidate_statistics = F.pad(cand_bias_feats, (1, 0), value=1.0)
        coefficients = candidate_statistics[:, None] @ self.statistics_attention_weights
        bias = coefficients @ key_statistics.transpose(-1, -2)[:, None]
        bias = bias.masked_fill(k_mask[:, None, None, :], -torch.inf)
        attention_bias = bias.reshape(b_size * self.num_heads, c_slate, memory.size(1)).to(v_exp.dtype)

        h_att, _ = self.cross_attention(
            query=v_exp,
            key=memory,
            value=memory,
            attn_mask=attention_bias,
            need_weights=False,
        )
        h_att = torch.nan_to_num(h_att, nan=0.0)

        hidden = torch.cat([v_exp, h_att, v_exp * h_att, candidate_features], dim=-1)
        for block in self.mlp_blocks:
            hidden = block(hidden)
        main_logits = self.main_head(hidden).squeeze(-1).masked_fill(~candidate_mask, -torch.inf)
        unseen_logits = self.option_a_head(v_exp, h_att, candidate_features).masked_fill(~candidate_mask, -torch.inf)
        return main_logits.squeeze(0).cpu().numpy(), unseen_logits.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Live Inference Engine
# ---------------------------------------------------------------------------

def _extract_drivers_uk(f_row: np.ndarray, *, is_new_candidate: bool) -> List[str]:
    out: List[str] = []
    if f_row[DRIVER_IDX["has_bought_base_sku"]] > 0.5:
        n_ord = max(1, int(round(math.expm1(float(f_row[DRIVER_IDX["user_base_sku_order_count_norm"]]) * math.log1p(50.0)))))
        days = int(round(math.expm1(float(f_row[DRIVER_IDX["days_since_last_base_sku_buy_norm"]]) * math.log1p(365.0))))
        out.append(f"Купував {n_ord}× · {days}д тому")
        urg = float(f_row[DRIVER_IDX["base_sku_urgency"]]) * 5.0
        if urg >= 0.6:
            out.append(f"Час повтору {urg:.1f}×")
    else:
        cf_rec = float(f_row[DRIVER_IDX["full_cf_recency_weighted"]])
        cf_cos = float(f_row[DRIVER_IDX["full_cf_cosine"]])
        if cf_rec > 0.08:
            out.append(f"Схожі клієнти {cf_rec:.2f}")
        elif cf_cos > 0.08:
            out.append(f"CF схожість {cf_cos:.2f}")

    if f_row[DRIVER_IDX["base_name_recent3_lift"]] > 0.15:
        out.append(f"Зв'язка +{float(f_row[DRIVER_IDX['base_name_recent3_lift']]):.2f}")
    if f_row[DRIVER_IDX["mgr_base_name_share_30d"]] > 0.02:
        out.append(f"Менеджер 30д {100.0 * float(f_row[DRIVER_IDX['mgr_base_name_share_30d']]):.1f}%")
    elif f_row[DRIVER_IDX["mgr_industry_base_name_share_90d"]] > 0.02:
        out.append(f"Мен×Галузь {100.0 * float(f_row[DRIVER_IDX['mgr_industry_base_name_share_90d']]):.1f}%")
    elif f_row[DRIVER_IDX["mgr_base_name_share_90d"]] > 0.02:
        out.append(f"Менеджер 90д {100.0 * float(f_row[DRIVER_IDX['mgr_base_name_share_90d']]):.1f}%")
    if f_row[DRIVER_IDX["region_base_name_share_90d"]] > 0.025 and is_new_candidate:
        out.append(f"Регіон 90д {100.0 * float(f_row[DRIVER_IDX['region_base_name_share_90d']]):.1f}%")
    if f_row[DRIVER_IDX["base_name_industry_affinity"]] > 0.03:
        out.append(f"Галузь {100.0 * float(f_row[DRIVER_IDX['base_name_industry_affinity']]):.1f}%")
    if f_row[DRIVER_IDX["app_matches_company_equipment"]] > 0.5:
        out.append("Збіг техніки")
    if f_row[DRIVER_IDX["user_viscosity_order_share"]] > 0.0 and is_new_candidate:
        out.append(f"В'язкість {100.0 * float(f_row[DRIVER_IDX['user_viscosity_order_share']]):.0f}%")
    return out[:4]


class LiveTestInferenceEngine:
    def __init__(self, *, checkpoint_path: Path, data_dir: Optional[Path] = None, bundle_path: Optional[Path] = None):
        t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.dev = torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path)

        if data_dir is not None and (Path(data_dir) / "test_live_meta.pt").is_file():
            dd = Path(data_dir)
            print(f"[live-demo] Loading full 13,184-order held-out Test cache from {dd}...", flush=True)
            self.payload = torch.load(dd / "test_live_meta.pt", map_location="cpu", weights_only=False)
            self.cand42_source = np.load(dd / "test_cand_42.npy", mmap_mode="r")
            self.storage_mode = "full_memmap"
        elif bundle_path is not None and Path(bundle_path).is_file():
            bp = Path(bundle_path)
            print(f"[live-demo] Loading compressed portable Test bundle from {bp}...", flush=True)
            with gzip.open(bp, "rb") as gz:
                self.payload = torch.load(gz, map_location="cpu", weights_only=False)
            self.cand42_source = self.payload["cand_42_embedded"].numpy()
            self.storage_mode = "portable_bundle"
        else:
            raise FileNotFoundError(
                f"Neither full live cache ({data_dir}) nor portable bundle ({bundle_path}) found."
            )

        self.meta: Dict[str, Any] = dict(self.payload["meta"])
        self.n_cases = int(self.meta["num_cases"])
        self.n_bn = int(self.meta["num_formulations"])
        self.n_skus = int(self.meta["num_skus"])
        self.cal_a = float(self.meta["calibration"]["a"])
        self.cal_b = float(self.meta["calibration"]["b"])

        # Instantiate exact standalone Wave2BE2ERanker and load promoted weights strictly
        self.model = StandaloneWave2BE2ERanker(self.payload["arch"]).to(self.dev)
        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.model.eval()

        # Precompute all 1,141 product vectors via live ProductTower once at startup
        t_pt0 = time.perf_counter()
        self.v_bn = self.model.precompute_product_vectors(self.payload["prod_inputs"]).detach()
        self.product_tower_startup_ms = round(1000.0 * (time.perf_counter() - t_pt0), 2)

        # Build fast cohort index pools & search corpus
        flags = self.payload["flags"]
        tr_cp = flags["trained_cp"].astype(bool)
        fo = flags["first_observed"].astype(bool)
        hn = flags["has_new_ret"].astype(bool)
        self.cohorts: Dict[str, List[int]] = {
            "trained_cp": np.flatnonzero(tr_cp).tolist(),
            "unlearned": np.flatnonzero(~tr_cp).tolist(),
            "all": list(range(self.n_cases)),
            "first_observed": np.flatnonzero(fo).tolist(),
            "has_new_ret": np.flatnonzero(hn).tolist(),
        }
        # Prebuild lowercase search strings per case for <5ms search across all 13,184 orders
        bn_meta = self.payload["bn_meta"]
        companies = self.payload["companies"]
        cids = self.payload["customer_ids"]
        dates_s = self.payload["dates_str"]
        tgt_lists = self.payload["target_bn_lists"]
        self.search_blobs: List[str] = []
        for i in range(self.n_cases):
            cid = cids[i]
            comp = companies.get(cid, {})
            t_names = " ".join(bn_meta[b]["nm"] for b in tgt_lists[i][:4])
            blob = f"#{i} {cid} {dates_s[i]} {comp.get('nm', '')} {comp.get('ct', '')} {comp.get('rg', '')} {comp.get('ind', '')} {t_names}".lower()
            self.search_blobs.append(blob)

        # Warm up PyTorch CPU kernels so live timings reflect steady-state latency
        for w_i in range(min(3, self.n_cases)):
            self.evaluate_case(w_i, mode="blended")

        print(
            f"[live-demo] Ready in {time.perf_counter() - t0:.2f}s | {self.n_cases:,} real held-out Test orders "
            f"(Trained C&P: {len(self.cohorts['trained_cp']):,}, Unlearned: {len(self.cohorts['unlearned']):,}) | "
            f"ProductTower startup: {self.product_tower_startup_ms:.1f} ms | PyTorch {torch.__version__} CPU",
            flush=True,
        )

    def _unpack_bool(self, packed_row: np.ndarray, length: int) -> np.ndarray:
        return np.unpackbits(packed_row)[:length].astype(bool)

    def _resolve_skus_causal(
        self, top_bns: Sequence[int], cid: str, query_day: int, sku_allowed: np.ndarray
    ) -> List[int]:
        sku_cnt: Counter[int] = Counter()
        sku_last: Dict[int, int] = {}
        cont_cnt: Counter[str] = Counter()
        sku_cont = self.payload["sku_cont"]
        entry = self.payload["cust_sku_hist"].get(cid)
        if entry is not None:
            days_arr, s_arr = entry
            stop = int(days_arr.searchsorted(query_day, side="left"))
            for pos in range(stop):
                s_i = int(s_arr[pos])
                if s_i >= 0:
                    sku_cnt[s_i] += 1
                    sku_last[s_i] = pos
                    if sku_cont[s_i]:
                        cont_cnt[sku_cont[s_i]] += 1
        dom_cont = cont_cnt.most_common(1)[0][0] if cont_cnt else ""
        bn_to_skus = self.payload["bn_to_sku_idxs"]
        sale_days = self.payload["sku_sale_days"]
        out: List[int] = []
        for b_i in top_bns:
            elig = [s_i for s_i in bn_to_skus[b_i] if sku_allowed[s_i]]
            if not elig:
                out.append(-1)
                continue
            if len(elig) == 1:
                out.append(elig[0])
                continue
            prior_m = [s_i for s_i in elig if sku_cnt[s_i] > 0]
            if prior_m:
                out.append(max(prior_m, key=lambda s_i: (sku_cnt[s_i], sku_last[s_i])))
                continue
            if dom_cont:
                cont_m = [s_i for s_i in elig if sku_cont[s_i] == dom_cont]
                if cont_m:
                    out.append(
                        max(cont_m, key=lambda s_i: int(sale_days[s_i].searchsorted(query_day, side="left")) if s_i in sale_days else 0)
                    )
                    continue
            out.append(
                max(elig, key=lambda s_i: int(sale_days[s_i].searchsorted(query_day, side="left")) if s_i in sale_days else 0)
            )
        return out

    def metadata_payload(self) -> Dict[str, Any]:
        return {
            **self.meta,
            "storage_mode": self.storage_mode,
            "product_tower_startup_ms": self.product_tower_startup_ms,
            "cohort_counts": {k: len(v) for k, v in self.cohorts.items()},
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "CPU (live PyTorch)",
                "cpu_threads": int(torch.get_num_threads()),
                "platform": platform.machine() or platform.system(),
            },
        }

    def search_cases(self, cohort: str = "trained_cp", query: str = "", limit: int = 2000) -> Dict[str, Any]:
        pool = self.cohorts.get(cohort, self.cohorts["all"])
        q = (query or "").strip().lower()
        if not q:
            return {"total": len(pool), "indices": pool[:limit]}
        if q.startswith("#") and q[1:].isdigit():
            target_id = int(q[1:])
            if 0 <= target_id < self.n_cases:
                return {"total": 1, "indices": [target_id]}
        if q.isdigit():
            target_id = int(q)
            if 0 <= target_id < self.n_cases:
                return {"total": 1, "indices": [target_id]}
        hits = [i for i in pool if q in self.search_blobs[i]]
        return {"total": len(hits), "indices": hits[:limit]}

    def evaluate_case(self, idx: int, *, mode: str = "blended") -> Dict[str, Any]:
        mode_clean = "pure_unseen" if str(mode).strip().lower() == "pure_unseen" else "blended"
        i = max(0, min(self.n_cases - 1, int(idx)))

        with self.lock:
            p = self.payload
            cid = p["customer_ids"][i]
            dt_str = p["dates_str"][i]
            q_day = int(p["prediction_days"][i].item())

            # 1. Prepare tensors strictly < query_day
            t_f0 = time.perf_counter()
            cand_42_np = np.asarray(self.cand42_source[i], dtype=np.float32)
            allowed = self._unpack_bool(p["allowed_bn_packed"][i], self.n_bn)
            unseen = self._unpack_bool(p["unseen_bn_packed"][i], self.n_bn)
            t_idx = int(p["cust_text_idx"][i].item())
            text_emb = p["cust_text_table"][t_idx : t_idx + 1].float()
            city_id = p["city_id"][i : i + 1].long()
            region_id = p["region_id"][i : i + 1].long()
            industry_id = p["industry_id"][i : i + 1].long()
            cust_7 = p["cust_7"][i : i + 1].float()
            seq = p["sequence"][i : i + 1].float()
            att_mask = p["attention_mask"][i : i + 1].long()
            seq_pids = p["seq_product_ids"][i : i + 1].long()
            seq_dates = p["seq_purchase_dates"][i : i + 1].float()
            pred_day_t = p["prediction_days"][i : i + 1].float()
            cand_t = torch.from_numpy(cand_42_np).unsqueeze(0)
            allowed_t = torch.from_numpy(allowed).unsqueeze(0)
            feat_ms = 1000.0 * (time.perf_counter() - t_f0)

            # 2. LIVE PyTorch forward pass over ALL 1,141 products
            t_m0 = time.perf_counter()
            s_main, s_unseen = self.model.score_single(
                company_text_embedding=text_emb,
                city_id=city_id,
                region_id=region_id,
                industry_id=industry_id,
                customer_features=cust_7,
                sequence=seq,
                attention_mask=att_mask,
                seq_product_ids=seq_pids,
                seq_purchase_dates=seq_dates,
                prediction_day=pred_day_t,
                candidate_features=cand_t,
                candidate_mask=allowed_t,
                v_bn=self.v_bn,
            )
            model_ms = 1000.0 * (time.perf_counter() - t_m0)

            elig_unseen = allowed & unseen
            if mode_clean == "pure_unseen":
                active_mask = elig_unseen
                scores = np.where(elig_unseen, s_unseen, -np.inf)
            else:
                active_mask = allowed
                s_u_safe = np.where(elig_unseen, s_unseen, 0.0)
                s_m_safe = np.where(allowed, s_main, 0.0)
                scores = np.where(allowed, np.where(unseen, self.cal_a * s_u_safe + self.cal_b, s_m_safe), -np.inf)

            order = np.argsort(-scores, kind="mergesort")
            valid_count = int(active_mask.sum())
            top10 = order[: min(10, valid_count)].tolist()
            rank_of_bn = {int(bn_i): pos + 1 for pos, bn_i in enumerate(order[:valid_count].tolist())}

            # 3. Live causal SKU pack-size resolution
            t_s0 = time.perf_counter()
            sku_allowed = self._unpack_bool(p["sku_allowed_packed"][i], self.n_skus)
            resolved_skus = self._resolve_skus_causal(top10, cid, q_day, sku_allowed)
            sku_ms = 1000.0 * (time.perf_counter() - t_s0)

            # 4. Assemble Top-10 predictions, ground truth, profile & strictly-prior history
            tgt_set = set(p["target_bn_lists"][i])
            new_set = set(p["new_target_bn_lists"][i])
            sku_tgt_set = set(p["sku_target_lists"][i])
            unmod_list = p["unmodeled_targets"][i]
            bn_meta = p["bn_meta"]
            sku_labels = p["sku_labels"]

            predictions = []
            rel_flags = []
            new_hits_10 = 0
            for r0, bn_i in enumerate(top10):
                meta_b = bn_meta[bn_i]
                is_new_cand = bool(unseen[bn_i])
                is_hit = (bn_i in new_set) if mode_clean == "pure_unseen" else (bn_i in tgt_set)
                is_new_hit = bn_i in new_set
                rel_flags.append(1.0 if is_hit else 0.0)
                if is_new_hit:
                    new_hits_10 += 1
                lv = float(scores[bn_i])
                prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, lv))))
                r_sku = resolved_skus[r0]
                predictions.append(
                    {
                        "rank": r0 + 1,
                        "bn": meta_b["nm"],
                        "br": meta_b["br"],
                        "cat": meta_b["cat"],
                        "sku": sku_labels[r_sku] if r_sku >= 0 else "—",
                        "sku_hit": r_sku in sku_tgt_set,
                        "prob": round(prob, 4),
                        "logit": round(lv, 3),
                        "is_repeat": not is_new_cand,
                        "is_hit": is_hit,
                        "is_new_hit": is_new_hit,
                        "drivers": _extract_drivers_uk(cand_42_np[bn_i], is_new_candidate=is_new_cand),
                    }
                )

            gt_items = []
            for bn_i in p["target_bn_lists"][i]:
                meta_b = bn_meta[bn_i]
                rk = rank_of_bn.get(bn_i) if active_mask[bn_i] else None
                gt_items.append(
                    {
                        "bn": meta_b["nm"],
                        "br": meta_b["br"],
                        "cat": meta_b["cat"],
                        "is_new": bn_i in new_set,
                        "is_modeled": True,
                        "rank": rk,
                        "hit10": rk is not None and rk <= 10,
                    }
                )
            for unmod_bn, is_n in unmod_list:
                gt_items.append(
                    {
                        "bn": unmod_bn,
                        "br": "Новий запуск (>07.2026)",
                        "cat": "Поза офлайн-таблицею 1 141",
                        "is_new": bool(is_n),
                        "is_modeled": False,
                        "rank": None,
                        "hit10": False,
                    }
                )
            gt_items.sort(key=lambda x: (x["rank"] is None, x["rank"] or 9999))

            extra_bn = len(unmod_list)
            extra_new = sum(1 for _, is_n in unmod_list if is_n)
            n_pos = (len(new_set) + extra_new) if mode_clean == "pure_unseen" else (len(tgt_set) + extra_bn)
            n_new_tot = len(new_set) + extra_new
            hits_10 = int(sum(rel_flags))
            disc = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
            dcg = float(np.sum(np.array(rel_flags, dtype=np.float64) * disc[: len(rel_flags)]))
            idcg = float(np.sum(disc[: min(n_pos, 10)])) if n_pos > 0 else 1.0
            ndcg_10 = dcg / idcg if (n_pos > 0 and idcg > 0) else 0.0
            r10 = (hits_10 / n_pos) if n_pos > 0 else 0.0

            # Strictly-prior history stats (< query_day) via binary search
            prior_hist = []
            n_prior_ord = 0
            n_prior_bns = 0
            med_gap = None
            days_last = None
            h_entry = p["cust_bn_hist"].get(cid)
            if h_entry is not None:
                h_days, h_bns, h_qtys = h_entry
                stop = int(h_days.searchsorted(q_day, side="left"))
                if stop > 0:
                    sub_d = h_days[:stop]
                    sub_b = h_bns[:stop]
                    sub_q = h_qtys[:stop]
                    uniq_d = np.unique(sub_d)
                    n_prior_ord = int(len(uniq_d))
                    n_prior_bns = int(len(np.unique(sub_b)))
                    days_last = int(q_day - int(uniq_d[-1]))
                    if len(uniq_d) >= 2:
                        med_gap = round(float(np.median(np.diff(uniq_d))), 1)
                    start_show = max(0, stop - 15)
                    for pos in range(stop - 1, start_show - 1, -1):
                        dv = int(sub_d[pos])
                        bi = int(sub_b[pos])
                        prior_hist.append(
                            {
                                "dt": (PURCHASE_ORIGIN + timedelta(days=dv)).isoformat(),
                                "ago": int(q_day - dv),
                                "bn": bn_meta[bi]["nm"],
                                "br": bn_meta[bi]["br"],
                                "q": round(float(sub_q[pos]), 2),
                            }
                        )

            comp = p["companies"].get(cid, {"nm": cid, "ct": "—", "rg": "—", "ind": "—", "desc": "—"})
            fin = p["cust_fin"][i]
            flags = p["flags"]
            return {
                "id": i,
                "dt": dt_str,
                "mode": mode_clean,
                "el": int(allowed.sum()),
                "elu": int(elig_unseen.sum()),
                "tr_c": int(flags["trained_c"][i]),
                "tr_p": int(flags["trained_p"][i]),
                "tr_cp": int(flags["trained_cp"][i]),
                "fo": int(flags["first_observed"][i]),
                "timings_ms": {
                    "feat": round(feat_ms, 2),
                    "model": round(model_ms, 2),
                    "sku": round(sku_ms, 2),
                    "total": round(feat_ms + model_ms + sku_ms, 2),
                },
                "metrics": {
                    "r10": round(r10, 4),
                    "ndcg10": round(ndcg_10, 4),
                    "hits10": hits_10,
                    "targets": n_pos,
                    "new_hits10": new_hits_10,
                    "new_targets": n_new_tot,
                    "unmodeled_count": extra_bn,
                },
                "customer": {
                    "cid": cid,
                    "nm": comp["nm"],
                    "ct": comp["ct"],
                    "rg": comp["rg"],
                    "ind": comp["ind"],
                    "desc": comp["desc"],
                    "ord": n_prior_ord,
                    "bns": n_prior_bns,
                    "gap": med_gap,
                    "last": days_last,
                    "prep": int(fin[0].item()),
                    "def": int(fin[1].item()),
                    "ovd": int(fin[2].item()),
                },
                "hist": prior_hist,
                "preds": predictions,
                "gt": gt_items,
            }

    def run_live_benchmark(self, *, n_orders: int = 100, cohort: str = "trained_cp") -> Dict[str, Any]:
        pool = self.cohorts.get(cohort, self.cohorts["all"])
        n = max(10, min(500, int(n_orders), len(pool)))
        rng = random.Random()
        sample_ids = rng.sample(pool, n)

        model_ms_list: List[float] = []
        total_ms_list: List[float] = []
        r10_list: List[float] = []
        ndcg_list: List[float] = []
        new_hits = 0
        new_tgts = 0

        t_wall0 = time.perf_counter()
        for idx in sample_ids:
            res = self.evaluate_case(idx, mode="blended")
            model_ms_list.append(res["timings_ms"]["model"])
            total_ms_list.append(res["timings_ms"]["total"])
            if res["metrics"]["targets"] > 0:
                r10_list.append(res["metrics"]["r10"])
                ndcg_list.append(res["metrics"]["ndcg10"])
            new_hits += res["metrics"]["new_hits10"]
            new_tgts += res["metrics"]["new_targets"]
        wall_ms = 1000.0 * (time.perf_counter() - t_wall0)

        m_arr = np.array(model_ms_list, dtype=np.float64)
        t_arr = np.array(total_ms_list, dtype=np.float64)
        mean_m = float(m_arr.mean())
        mean_t = float(t_arr.mean())
        return {
            "cohort": cohort,
            "n_orders": n,
            "median_model_ms": round(float(np.median(m_arr)), 2),
            "p95_model_ms": round(float(np.percentile(m_arr, 95)), 2),
            "mean_model_ms": round(mean_m, 2),
            "median_total_ms": round(float(np.median(t_arr)), 2),
            "p95_total_ms": round(float(np.percentile(t_arr, 95)), 2),
            "throughput_model_rps": round(1000.0 / max(mean_m, 0.01), 1),
            "throughput_e2e_rps": round(1000.0 / max(mean_t, 0.01), 1),
            "wall_elapsed_ms": round(wall_ms, 1),
            "sample_recall_at_10": round(100.0 * float(np.mean(r10_list)) if r10_list else 0.0, 2),
            "sample_ndcg_at_10": round(100.0 * float(np.mean(ndcg_list)) if ndcg_list else 0.0, 2),
            "sample_new_hits": int(new_hits),
            "sample_new_targets": int(new_tgts),
            "torch_version": torch.__version__,
            "cpu_threads": int(torch.get_num_threads()),
        }


# ---------------------------------------------------------------------------
# Clean Light-Theme Ukrainian Live UI (matches OPEN_DEMO.html design system)
# ---------------------------------------------------------------------------

LIVE_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Система AI-рекомендацій B2B — Live Python Демо (w2_2mem_recency_hl365)</title>
<style>
  :root {
    --bg: #f6f8fc;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.10);
    --green: #059669;
    --green-bg: rgba(16, 185, 129, 0.11);
    --green-border: #10b981;
    --amber: #b45309;
    --amber-bg: rgba(245, 158, 11, 0.13);
    --amber-border: #f59e0b;
    --purple: #7c3aed;
    --purple-bg: rgba(124, 58, 237, 0.10);
    --shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    --radius: 14px;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); line-height: 1.5; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 18px 22px 42px; display: flex; flex-direction: column; gap: 16px; }
  .topbar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 20px; box-shadow: var(--shadow);
  }
  .brand-title { font-size: 19px; font-weight: 800; letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .badge-live {
    font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em;
    padding: 3px 10px; border-radius: 999px; background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border);
  }
  .brand-sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .top-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .nav-pill {
    text-decoration: none; font-size: 13px; font-weight: 600; color: var(--muted);
    padding: 7px 13px; border-radius: 9px; border: 1px solid var(--border); background: var(--surface-2);
    transition: all 0.15s;
  }
  .nav-pill:hover { border-color: var(--primary); color: var(--primary); }
  .nav-pill.active { background: var(--primary); color: #fff; border-color: var(--primary); }

  .controls-bar {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; box-shadow: var(--shadow);
  }
  select, input[type="text"], button.btn {
    font-family: var(--sans); font-size: 13px; border-radius: 9px;
    border: 1px solid var(--border); background: var(--surface-2); color: var(--text); padding: 8px 12px;
  }
  input[type="text"] { min-width: 210px; flex: 1; }
  button.btn { cursor: pointer; font-weight: 600; transition: all 0.12s; }
  button.btn:hover:not(:disabled) { border-color: var(--primary); }
  button.btn:disabled { opacity: 0.55; cursor: wait; }
  button.btn.primary { background: var(--primary); color: #fff; border-color: var(--primary); }
  button.btn.primary:hover:not(:disabled) { opacity: 0.92; }
  button.btn.bench { background: var(--green-bg); color: var(--green); border-color: var(--green-border); }
  button.btn.bench:hover:not(:disabled) { background: #d1fae5; }

  .bench-card {
    display: none; background: var(--surface); border: 1.5px solid var(--green-border);
    border-radius: var(--radius); padding: 14px 18px; box-shadow: var(--shadow);
  }
  .bench-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 10px; }
  .bench-item { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 9px 12px; }
  .bench-item .k { font-size: 11px; color: var(--muted); font-weight: 700; text-transform: uppercase; }
  .bench-item .v { font-size: 19px; font-weight: 800; font-family: var(--mono); margin-top: 2px; color: var(--text); }
  .bench-item .s { font-size: 11.5px; color: var(--muted); margin-top: 2px; }

  .kpi {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 15px 18px; box-shadow: var(--shadow);
  }
  .kpi .lbl { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 700; }
  .kpi .num { font-size: 26px; font-weight: 800; font-family: var(--mono); margin-top: 5px; letter-spacing: -0.02em; }
  .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 3px; }
  .num.green { color: var(--green); }
  .num.amber { color: var(--amber); }

  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px 20px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 14px;
  }
  .card-hdr { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; }
  .card-title { font-size: 14px; font-weight: 800; letter-spacing: -0.01em; }
  .explorer-grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; align-items: start; }

  .pill {
    display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 700;
    padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface-2); color: var(--muted);
    white-space: nowrap;
  }
  .pill.hit { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
  .pill.newhit { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber); }
  .pill.cold { background: var(--purple-bg); border-color: var(--purple); color: var(--purple); }
  .pill.ret { background: var(--primary-soft); border-color: var(--primary); color: var(--primary); }
  .pill.newtag { background: var(--amber-bg); border-color: rgba(245, 158, 11, 0.45); color: var(--amber); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); padding: 8px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 10px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr.row-hit { background: var(--green-bg); }
  tr.row-hit td:first-child { border-left: 4px solid var(--green-border); }
  tr.row-newhit { background: var(--amber-bg); }
  tr.row-newhit td:first-child { border-left: 4px solid var(--amber-border); }
  .rank-col { font-family: var(--mono); font-weight: 800; width: 36px; }
  .prod-nm { font-weight: 700; }
  .prod-sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .drivers { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
  .drv { font-size: 11px; padding: 2px 7px; border-radius: 5px; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted); font-family: var(--mono); }
  .score-box { width: 86px; }
  .score-bar { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; margin-top: 4px; }
  .score-fill { height: 100%; background: var(--primary); border-radius: 999px; }
  tr.row-hit .score-fill { background: var(--green-border); }
  tr.row-newhit .score-fill { background: var(--amber-border); }

  .gt-list { display: flex; flex-direction: column; gap: 7px; }
  .gt-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 9px 12px; border-radius: 10px; border: 1px solid var(--border); background: var(--surface-2); }
  .gt-row.hit { background: var(--green-bg); border-color: var(--green-border); }
  .gt-row.newhit { background: var(--amber-bg); border-color: var(--amber-border); }

  .mini-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .mstat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 9px 11px; }
  .mstat .k { font-size: 11px; color: var(--muted); }
  .mstat .v { font-size: 14px; font-weight: 800; font-family: var(--mono); margin-top: 2px; }
  .case-kpi-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .hist-scroll { max-height: 290px; overflow-y: auto; }
  .mono { font-family: var(--mono); }
  .footer {
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; font-size: 12px;
    color: var(--muted); padding: 12px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  }
  @media (max-width: 1020px) {
    .explorer-grid, .case-kpi-strip, .bench-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <div class="brand-title">
        <span>Система AI-рекомендацій B2B</span>
        <span class="badge-live" id="live-badge">⚡ LIVE PYTHON · PYTORCH CPU</span>
      </div>
      <div class="brand-sub" id="top-subtitle">Завантаження параметрів живої нейромережі...</div>
    </div>
    <div class="top-actions">
      <a class="nav-pill active" href="/">⚡ Live Python Демо</a>
      <a class="nav-pill" id="nav-pres" href="PRESENTATION.html" style="display:none;">Презентація</a>
      <a class="nav-pill" id="nav-doc" href="DOCUMENTATION.html" style="display:none;">Документація та 3D</a>
    </div>
  </div>

  <div class="controls-bar">
    <select id="sel-cohort"></select>
    <select id="sel-mode">
      <option value="blended">Змішаний прогноз (Повтори + Новинки)</option>
      <option value="pure_unseen">Лише нові товари (Крос-продажі)</option>
    </select>
    <input type="text" id="inp-search" placeholder="Пошук компанії, ID, міста, товару або #номера кейсу..." />
    <button class="btn" id="btn-prev" title="Попередній кейс">←</button>
    <button class="btn" id="btn-next" title="Наступний кейс">→</button>
    <button class="btn primary" id="btn-random">Випадковий кейс</button>
    <button class="btn bench" id="btn-bench" title="Запустити 100 реальних прогнозів PyTorch на CPU">⚡ Бенчмарк (100 замовлень)</button>
    <span class="pill mono" id="pool-counter">—</span>
  </div>

  <div class="bench-card" id="bench-card">
    <div class="card-hdr">
      <div class="card-title" style="color:var(--green);" id="bench-title">⚡ Результати живого бенчмарку PyTorch CPU (100 реальних замовлень по одному)</div>
      <button class="btn" id="btn-bench-close" style="padding:4px 10px;font-size:12px;">Сховати</button>
    </div>
    <div class="bench-grid" id="bench-grid"></div>
  </div>

  <div id="unlearned-warn" style="display:none;background:var(--amber-bg);border:1.5px solid var(--amber-border);border-radius:12px;padding:11px 16px;font-size:13px;color:#92400e;font-weight:600;"></div>

  <div class="case-kpi-strip">
    <div class="kpi">
      <div class="lbl" id="lbl-c-r10">Точність прогнозу (Recall@10)</div>
      <div class="num green" id="c-r10">—</div>
      <div class="sub" id="c-hits">—</div>
    </div>
    <div class="kpi">
      <div class="lbl">Знайдено нових товарів у Топ-10</div>
      <div class="num amber" id="c-new">—</div>
      <div class="sub">Товари, які клієнт ніколи не купував раніше</div>
    </div>
    <div class="kpi">
      <div class="lbl">Жива швидкість PyTorch (CPU)</div>
      <div class="num" id="c-ms" style="font-size:22px;padding-top:3px;color:var(--primary);">—</div>
      <div class="sub" id="c-date-sub">—</div>
    </div>
  </div>

  <div class="explorer-grid">
    <div class="card">
      <div class="card-hdr">
        <div class="card-title">Топ-10 рекомендацій нейромережі (живий розрахунок PyTorch)</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span class="pill hit">✓ Вгадано (Повтор)</span>
          <span class="pill newhit">★ Вгадано (Новий товар!)</span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Рекомендований продукт і причини</th>
            <th>Фасування (SKU)</th>
            <th>Оцінка</th>
            <th>Факт</th>
          </tr>
        </thead>
        <tbody id="pred-tbody"></tbody>
      </table>

      <div style="margin-top:6px;">
        <div class="card-hdr" style="margin-bottom:8px;">
          <div class="card-title">Що клієнт реально купив <span class="mono" id="gt-date"></span></div>
          <span class="pill" id="gt-badge">—</span>
        </div>
        <div class="gt-list" id="gt-list"></div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Профіль клієнта (строго до дати замовлення)</div>
          <span class="pill" id="cust-cohort">—</span>
        </div>
        <div>
          <div style="font-size:16px;font-weight:800;" id="cust-name">—</div>
          <div style="font-size:12.5px;color:var(--muted);margin-top:2px;" id="cust-loc">—</div>
        </div>
        <div id="cust-desc" style="font-size:12.5px;color:var(--muted);background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;">—</div>
        <div class="mini-stats">
          <div class="mstat"><div class="k">Минулих замовлень</div><div class="v" id="st-ord">—</div></div>
          <div class="mstat"><div class="k">Унікальних товарів</div><div class="v" id="st-bns">—</div></div>
          <div class="mstat"><div class="k">Періодичність</div><div class="v" id="st-gap">—</div></div>
          <div class="mstat"><div class="k">Днів з останнього</div><div class="v" id="st-last">—</div></div>
          <div class="mstat"><div class="k">Передплата / готівка</div><div class="v" id="st-prep">—</div></div>
          <div class="mstat"><div class="k">Відстрочка / простр.</div><div class="v" id="st-def">—</div></div>
        </div>
      </div>

      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Останні покупки клієнта (&lt; дати прогнозу)</div>
          <span class="pill" id="hist-badge">—</span>
        </div>
        <div class="hist-scroll">
          <table>
            <thead><tr><th>Дата</th><th>Продукт</th><th>К-сть</th></tr></thead>
            <tbody id="hist-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div id="footer-left"></div>
    <div id="footer-right">Кожен клік запускає реальний forward-pass нейромережі Wave2BE2ERanker у PyTorch на CPU.</div>
  </div>
</div>

<script>
let META = null;
let matchedIds = [];
let totalMatched = 0;
let curPos = 0;
let curCaseId = 0;
let searchTimer = null;

const esc = s => String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

async function init() {
  const res = await fetch('/api/meta');
  META = await res.json();
  if (META.has_presentation) document.getElementById('nav-pres').style.display = '';
  if (META.has_documentation) document.getElementById('nav-doc').style.display = '';
  const nCases = META.num_cases.toLocaleString('uk-UA');
  document.getElementById('live-badge').textContent = `⚡ LIVE PYTHON · PYTORCH ${META.runtime.torch} CPU · ${nCases} ЗАМОВЛЕНЬ`;
  document.getElementById('top-subtitle').textContent =
    `Реальний запуск ${META.arm} (Епоха ${META.epoch}) · ${META.date_range[0]} → ${META.date_range[1]} · ProductTower старт: ${META.product_tower_startup_ms} мс · ${META.num_formulations.toLocaleString('uk-UA')} продуктів`;
  document.getElementById('footer-left').textContent =
    `Реліз: ${META.reference_id} · PyTorch ${META.runtime.torch} · Python ${META.runtime.python} (${META.runtime.platform}, ${META.runtime.cpu_threads} потоків CPU)`;

  const cc = META.cohort_counts;
  const opts = [
    ['trained_cp', `Навчені клієнти й товари · тест (${cc.trained_cp.toLocaleString('uk-UA')})`],
    ['unlearned', `Ненавчені клієнти або товари ⚠️ · тест (${cc.unlearned.toLocaleString('uk-UA')})`],
    ['all', `Усі замовлення тесту (${cc.all.toLocaleString('uk-UA')})`],
    ['first_observed', `Нові клієнти / холодний старт (${cc.first_observed.toLocaleString('uk-UA')})`],
    ['has_new_ret', `Повернені клієнти з новинками (${cc.has_new_ret.toLocaleString('uk-UA')})`],
  ];
  document.getElementById('sel-cohort').innerHTML = opts
    .filter(([k]) => (cc[k] || 0) > 0)
    .map(([k, lbl]) => `<option value="${k}">${esc(lbl)}</option>`).join('');

  await refreshPool(true);
}

async function refreshPool(pickRandom = false) {
  const cohort = document.getElementById('sel-cohort').value;
  const q = document.getElementById('inp-search').value.trim();
  const res = await fetch(`/api/search?cohort=${encodeURIComponent(cohort)}&q=${encodeURIComponent(q)}`);
  const d = await res.json();
  matchedIds = d.indices || [];
  totalMatched = d.total || 0;
  if (!matchedIds.length) {
    document.getElementById('pool-counter').textContent = '0 збігів';
    return;
  }
  curPos = pickRandom ? Math.floor(Math.random() * matchedIds.length) : 0;
  await loadCase(matchedIds[curPos]);
}

async function loadCase(caseId) {
  curCaseId = caseId;
  const mode = document.getElementById('sel-mode').value;
  const tClient0 = performance.now();
  const res = await fetch(`/api/case?idx=${caseId}&mode=${encodeURIComponent(mode)}`);
  const d = await res.json();
  const roundtripMs = (performance.now() - tClient0).toFixed(1);
  renderCase(d, roundtripMs);
}

function renderCase(d, roundtripMs) {
  const isPure = d.mode === 'pure_unseen';
  document.getElementById('pool-counter').textContent =
    `${(curPos + 1).toLocaleString('uk-UA')} / ${totalMatched.toLocaleString('uk-UA')} (Кейс #${d.id})`;

  const warnEl = document.getElementById('unlearned-warn');
  if (d.tr_cp === 0) {
    const reasons = [];
    if (d.tr_c === 0) reasons.push('клієнт не зустрічався в навчанні (< 2026-02-01)');
    if (d.tr_p === 0) reasons.push('у кошику є новий товар, якого не було в навчанні');
    warnEl.textContent = `⚠️ Увага (Варіант 2 впровадження): ${reasons.join(' та ')} — прогноз орієнтовний, рекомендовано перевірити вручну.`;
    warnEl.style.display = 'block';
  } else {
    warnEl.style.display = 'none';
  }

  const m = d.metrics;
  document.getElementById('lbl-c-r10').textContent = isPure ? 'Точність новинок (Cross-Sell@10)' : 'Точність прогнозу (Recall@10)';
  document.getElementById('c-r10').textContent = m.targets > 0 ? `${(100 * m.r10).toFixed(1)}%` : '—';
  document.getElementById('c-hits').textContent = isPure
    ? `${m.hits10} із ${m.targets} нових продуктів у Топ-10`
    : `${m.hits10} із ${m.targets} куплених продуктів у Топ-10` + (m.unmodeled_count > 0 ? ` (+${m.unmodeled_count} поза 1 141)` : '');
  document.getElementById('c-new').textContent = `${m.new_hits10} / ${m.new_targets}`;

  const tm = d.timings_ms;
  document.getElementById('c-ms').textContent = `${tm.model.toFixed(2)} мс`;
  document.getElementById('c-date-sub').textContent =
    `Ознаки ${tm.feat.toFixed(2)} + Модель ${tm.model.toFixed(2)} + SKU ${tm.sku.toFixed(2)} = ${tm.total.toFixed(2)} мс · HTTP ${roundtripMs} мс · ${d.dt} (${d.el.toLocaleString('uk-UA')} товарів)`;

  document.getElementById('pred-tbody').innerHTML = d.preds.map(p => {
    const rowCls = p.is_new_hit ? 'row-newhit' : (p.is_hit ? 'row-hit' : '');
    const badge = p.is_new_hit
      ? '<span class="pill newhit">★ Вгадано · Новий!</span>'
      : (p.is_hit ? '<span class="pill hit">✓ Вгадано · Повтор</span>'
        : (p.is_repeat ? '<span class="pill">Повтор</span>' : '<span class="pill newtag">Новий товар</span>'));
    const drvs = p.drivers.map(x => `<span class="drv">${esc(x)}</span>`).join('');
    const skuMark = p.sku_hit ? ' <span style="color:var(--green);font-weight:800;">✓</span>' : '';
    const pct = Math.min(100, Math.round(100 * p.prob));
    return `<tr class="${rowCls}">
      <td class="rank-col">#${p.rank}</td>
      <td>
        <div class="prod-nm">${esc(p.bn)}</div>
        <div class="prod-sub">${esc(p.br)} · ${esc(p.cat)}</div>
        <div class="drivers">${drvs}</div>
      </td>
      <td class="mono" style="font-size:11.5px;color:var(--text);">${esc(p.sku)}${skuMark}</td>
      <td class="score-box">
        <div class="mono" style="font-weight:800;font-size:12px;">${(100 * p.prob).toFixed(1)}%</div>
        <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
      </td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  document.getElementById('gt-date').textContent = `(${d.dt})`;
  document.getElementById('gt-badge').textContent = `${d.gt.length} товарів (${m.new_targets} нових)`;
  document.getElementById('gt-list').innerHTML = d.gt.map(g => {
    const cls = g.hit10 ? (g.is_new ? 'newhit' : 'hit') : '';
    let rkTxt = g.rank !== null ? (g.hit10 ? `Місце #${g.rank} ✓` : `Місце #${g.rank}`)
      : (!g.is_modeled ? 'Поза табл. 1 141' : (isPure && !g.is_new ? 'Приховано (Повтор)' : 'Недоступний на дату'));
    const tag = g.is_new ? '<span class="pill newtag">Новий товар</span>' : '<span class="pill">Повтор</span>';
    return `<div class="gt-row ${cls}">
      <div><div class="prod-nm">${esc(g.bn)}</div><div class="prod-sub">${esc(g.br)} · ${esc(g.cat)}</div></div>
      <div style="display:flex;gap:6px;align-items:center;">${tag}<span class="pill mono ${cls}">${esc(rkTxt)}</span></div>
    </div>`;
  }).join('');

  const c = d.customer;
  document.getElementById('cust-cohort').className = 'pill ' + (d.fo ? 'cold' : 'ret');
  document.getElementById('cust-cohort').textContent = d.fo ? 'Новий клієнт (0 покупок)' : 'Постійний клієнт';
  document.getElementById('cust-name').textContent = c.nm;
  document.getElementById('cust-loc').textContent = `${c.ct} (${c.rg}) · ${c.ind} · ID: ${c.cid}`;
  document.getElementById('cust-desc').textContent = c.desc || '—';
  document.getElementById('st-ord').textContent = c.ord;
  document.getElementById('st-bns').textContent = c.bns;
  document.getElementById('st-gap').textContent = c.gap !== null ? `${c.gap}д` : '—';
  document.getElementById('st-last').textContent = c.last !== null ? `${c.last}д` : '—';
  document.getElementById('st-prep').textContent = `${c.prep}%`;
  document.getElementById('st-def').textContent = `${c.def}д / ${c.ovd}%`;

  document.getElementById('hist-badge').textContent = `${d.hist.length}`;
  document.getElementById('hist-tbody').innerHTML = d.hist.length
    ? d.hist.map(h => `<tr>
        <td class="mono" style="font-size:11.5px;white-space:nowrap;">${esc(h.dt)}<div style="color:var(--muted);font-size:10.5px;">-${h.ago}д</div></td>
        <td><div style="font-weight:600;">${esc(h.bn)}</div><div style="color:var(--muted);font-size:11px;">${esc(h.br)}</div></td>
        <td class="mono">${h.q}</td>
      </tr>`).join('')
    : `<tr><td colspan="3" style="color:var(--muted);padding:14px;">Перше замовлення клієнта — історії покупок ще немає. Прогноз базується на галузі, регіоні, профілі компанії та попиті.</td></tr>`;
}

async function runBenchmark() {
  const btn = document.getElementById('btn-bench');
  const cohort = document.getElementById('sel-cohort').value;
  btn.disabled = true;
  btn.textContent = '⏳ Виконую 100 замовлень...';
  try {
    const res = await fetch(`/api/benchmark?n=100&cohort=${encodeURIComponent(cohort)}`);
    const b = await res.json();
    const card = document.getElementById('bench-card');
    card.style.display = 'block';
    document.getElementById('bench-title').textContent =
      `⚡ Живий бенчмарк PyTorch ${b.torch_version} CPU (${b.n_orders} реальних замовлень · ${b.cpu_threads} потоків · за ${b.wall_elapsed_ms} мс)`;
    document.getElementById('bench-grid').innerHTML = `
      <div class="bench-item"><div class="k">Медіана нейромережі</div><div class="v" style="color:var(--green);">${b.median_model_ms} мс</div><div class="s">P95: ${b.p95_model_ms} мс · 1 141 товар</div></div>
      <div class="bench-item"><div class="k">Повний цикл (E2E)</div><div class="v">${b.median_total_ms} мс</div><div class="s">Ознаки + PyTorch + SKU</div></div>
      <div class="bench-item"><div class="k">Пропускна здатність</div><div class="v" style="color:var(--primary);">${b.throughput_model_rps} /с</div><div class="s">E2E: ${b.throughput_e2e_rps} замовл./сек</div></div>
      <div class="bench-item"><div class="k">Recall@10 на цих ${b.n_orders}</div><div class="v" style="color:var(--green);">${b.sample_recall_at_10}%</div><div class="s">NDCG@10: ${b.sample_ndcg_at_10}%</div></div>
      <div class="bench-item"><div class="k">Нові товари у Топ-10</div><div class="v" style="color:var(--amber);">${b.sample_new_hits} / ${b.sample_new_targets}</div><div class="s">Реальний тест без GPU</div></div>
    `;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Бенчмарк (100 замовлень)';
  }
}

document.getElementById('sel-cohort').addEventListener('change', () => refreshPool(true));
document.getElementById('sel-mode').addEventListener('change', () => loadCase(curCaseId));
document.getElementById('inp-search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => refreshPool(false), 180);
});
document.getElementById('btn-prev').addEventListener('click', () => {
  if (!matchedIds.length) return;
  curPos = (curPos - 1 + matchedIds.length) % matchedIds.length;
  loadCase(matchedIds[curPos]);
});
document.getElementById('btn-next').addEventListener('click', () => {
  if (!matchedIds.length) return;
  curPos = (curPos + 1) % matchedIds.length;
  loadCase(matchedIds[curPos]);
});
document.getElementById('btn-random').addEventListener('click', () => {
  if (!matchedIds.length) return;
  curPos = Math.floor(Math.random() * matchedIds.length);
  loadCase(matchedIds[curPos]);
});
document.getElementById('btn-bench').addEventListener('click', runBenchmark);
document.getElementById('btn-bench-close').addEventListener('click', () => {
  document.getElementById('bench-card').style.display = 'none';
});

init();
</script>
</body>
</html>
"""


def create_live_server(engine: LiveTestInferenceEngine, docs_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    docs_root = Path(docs_dir).resolve()
    allowed_static = {
        "/PRESENTATION.html": ("PRESENTATION.html", "text/html; charset=utf-8"),
        "/DOCUMENTATION.html": ("DOCUMENTATION.html", "text/html; charset=utf-8"),
        "/OPEN_DEMO_OFFLINE.html": ("OPEN_DEMO.html", "text/html; charset=utf-8"),
        "/PRESENTATION.pdf": ("PRESENTATION.pdf", "application/pdf"),
    }

    class LiveHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html", "/LIVE_DEMO.html", "/OPEN_DEMO.html"):
                self._send_bytes(LIVE_HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path in allowed_static:
                fn, ctype = allowed_static[path]
                target = docs_root / fn
                if target.is_file():
                    self._send_bytes(target.read_bytes(), ctype)
                    return
            if path == "/api/meta":
                mp = dict(engine.metadata_payload())
                mp["has_presentation"] = (docs_root / "PRESENTATION.html").is_file()
                mp["has_documentation"] = (docs_root / "DOCUMENTATION.html").is_file()
                self._send_json(mp)
                return
            if path == "/api/search":
                qs = parse_qs(parsed.query)
                cohort = qs.get("cohort", ["trained_cp"])[0]
                q = qs.get("q", [""])[0]
                self._send_json(engine.search_cases(cohort=cohort, query=q))
                return
            if path == "/api/case":
                qs = parse_qs(parsed.query)
                idx = int(qs.get("idx", ["0"])[0])
                mode = qs.get("mode", ["blended"])[0]
                self._send_json(engine.evaluate_case(idx, mode=mode))
                return
            if path == "/api/benchmark":
                qs = parse_qs(parsed.query)
                n = int(qs.get("n", ["100"])[0])
                cohort = qs.get("cohort", ["trained_cp"])[0]
                self._send_json(engine.run_live_benchmark(n_orders=n, cohort=cohort))
                return
            self._send_json({"error": f"Not found: {path}"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), LiveHandler)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best_discovery.pt")
    parser.add_argument("--data-dir", type=Path, default=None, help="Full 13,184-order cache dir (test_live_cache)")
    parser.add_argument("--bundle", type=Path, default=None, help="Portable compressed bundle (test_live_bundle.pt.gz)")
    parser.add_argument("--docs-dir", type=Path, default=None, help="Directory with PRESENTATION.html & DOCUMENTATION.html")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window on startup.")
    args = parser.parse_args()

    # Auto-resolve paths for both standalone unzipped folder and repo invocation
    ckpt = args.checkpoint
    if ckpt is None:
        for cand in [
            here / "model" / "best_discovery.pt",
            Path("exports/demo_preview/B2B_Recommender_Demo/model/best_discovery.pt"),
            Path("checkpoints/v2_full/wave2b/arm4_w2a_2mem_recency_hl365/best_discovery.pt"),
        ]:
            if cand.is_file():
                ckpt = cand
                break
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError("Could not locate best_discovery.pt; pass --checkpoint PATH.")

    data_dir = args.data_dir
    bundle = args.bundle
    if data_dir is None and bundle is None:
        if (here / "data" / "test_live_meta.pt").is_file():
            data_dir = here / "data"
        elif (here / "model" / "test_live_bundle.pt.gz").is_file():
            bundle = here / "model" / "test_live_bundle.pt.gz"
        elif Path("data/v2_full/interim/test_live_cache/test_live_meta.pt").is_file():
            data_dir = Path("data/v2_full/interim/test_live_cache")
        elif Path("data/v2_full/interim/test_live_bundle_1000.pt.gz").is_file():
            bundle = Path("data/v2_full/interim/test_live_bundle_1000.pt.gz")

    docs_dir = args.docs_dir
    if docs_dir is None:
        docs_dir = here if (here / "DOCUMENTATION.html").is_file() else Path("exports/demo_preview/B2B_Recommender_Demo")

    engine = LiveTestInferenceEngine(checkpoint_path=ckpt, data_dir=data_dir, bundle_path=bundle)
    server = create_live_server(engine, docs_dir=docs_dir, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"[live-demo] Server running at {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[live-demo] Stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
