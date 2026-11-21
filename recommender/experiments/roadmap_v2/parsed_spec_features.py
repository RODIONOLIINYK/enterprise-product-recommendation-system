"""Stage 3E: Deterministic parsed product specs, company equipment tags, and causal pairwise match features."""

from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiments.rolling_candidate_v2.data import (
    compute_rolling_source_digest,
    prepare_or_load_rolling_feature_bundle,
)
from recommender.scope import get_datasets_locs
from recommender.split import TEST_START

from .config import CACHE_ROOT

PARSED_SPEC_CACHE_VERSION = 1
PARSED_SPEC_CACHE_NAMESPACE = "roadmap_v2_parsed_specs_8d_v1"

PARSED_SPEC_FEATURE_NAMES: Tuple[str, ...] = (
    "same_viscosity_grade_bought",
    "user_viscosity_order_share",
    "matches_user_dominant_container",
    "user_container_order_share",
    "pack_size_log_norm",
    "pack_size_vs_user_median_pack",
    "app_matches_company_equipment",
    "user_application_order_share",
)
NUM_PARSED_SPEC_FEATURES = len(PARSED_SPEC_FEATURE_NAMES)

# Regex patterns for lubricant/industrial product specs
_SAE_RE = re.compile(r"(?<![0-9.,])(\d{1,2}W)[-\s]?(\d{2})(?![0-9.,])", re.IGNORECASE)
_ISO_VG_RE = re.compile(
    r"(?<![0-9.,])(?:ISO\s*)?(?:VG\s*|S\d\s*[A-Z]{1,3}\s*|[MGRBV]\s*|GX\s*|MX\s*)"
    r"(10|15|22|32|46|68|100|150|220|320|460|680|1000)(?![0-9.,\w])"
)
_NLGI_RE = re.compile(r"\b(?:NLGI\s*|Gadus[^\n]{0,25}\s)([0-3](?:\.5)?|00|000)\b", re.IGNORECASE)
_NLGI_EXPLICIT_RE = re.compile(r"\bNLGI\s*(000|00|[0-3](?:\.5)?)(?![0-9.,])", re.IGNORECASE)
_GADUS_NLGI_RE = re.compile(
    r"\bGadus\s+S\d\s+[A-Z0-9]{1,8}(?:\s+[A-Z]{1,4})?\s+(000|00|[0-3](?:\.5)?)(?![0-9.,\w])",
    re.IGNORECASE,
)
_GREASE_END_NLGI_RE = re.compile(
    r"(?:grease|смазк\w*|змазк\w*|ep|complex|plex|lithium)\s+(000|00|[0-3](?:\.5)?)(?![0-9.,\w])",
    re.IGNORECASE,
)
_PACK_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(л|l|кг|kg|мл|ml|г|g)\b", re.IGNORECASE)


def parse_viscosity_grades(text: str) -> Set[str]:
    """Extract ALL canonical viscosity / NLGI grades from text (e.g. {'ISO220', 'NLGI2'}).

    Fixes decimal pack false-positives (such as '0.4кг' matching NLGI0) and preserves both
    base-oil ISO VG and NLGI consistency grades on greases.
    """
    if not text:
        return set()
    out: Set[str] = set()
    for m in _SAE_RE.finditer(text):
        out.add(f"{m.group(1).upper()}-{m.group(2)}")
    for m in _ISO_VG_RE.finditer(text):
        out.add(f"ISO{m.group(1)}")
    low = text.lower()
    if any(k in low for k in ("gadus", "grease", "смазк", "змазк", "nlgi", "cassida grease")):
        for rx in (_NLGI_EXPLICIT_RE, _GADUS_NLGI_RE, _GREASE_END_NLGI_RE):
            for m in rx.finditer(text):
                out.add(f"NLGI{m.group(1)}")
    return out


def parse_viscosity_grade(text: str) -> str:
    """Extract canonical viscosity / NLGI grade string (e.g. '15W-40', 'ISO46', 'NLGI2') or ''."""
    if not text:
        return ""
    m = _SAE_RE.search(text)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    if any(k in text.lower() for k in ("gadus", "grease", "смазк", "змазк", "nlgi", "cassida grease")):
        m = _NLGI_RE.search(text)
        if m:
            return f"NLGI{m.group(1)}"
    m = _ISO_VG_RE.search(text)
    if m:
        return f"ISO{m.group(1)}"
    return ""


def parse_pack_size_liters_kg(text: str, container_type: str = "") -> float:
    """Extract numeric pack volume/mass in liters or kg (converting ml/g -> L/kg), or NaN if unknown."""
    if text:
        matches = _PACK_RE.findall(text)
        if matches:
            # Prefer the last explicit pack mention (often at end of SKU name like '209л' or '20л')
            val_str, unit = matches[-1]
            val = float(val_str.replace(",", "."))
            u = unit.lower()
            if u in ("мл", "ml", "г", "g"):
                val /= 1000.0
            if val > 0:
                return val
    ct = (container_type or "").strip().upper()
    fallback = {
        "БОЧКА": 209.0,
        "ВЕДРО": 20.0,
        "КАНИСТРА": 4.0,
        "ТУБА": 0.4,
        "СПРЕЙ": 0.4,
        "BULK": 1000.0,
    }
    return fallback.get(ct, float("nan"))


def parse_product_application(
    product_name: str,
    category: str = "",
    subcategory: str = "",
    business_line: str = "",
) -> str:
    """Map a product into a canonical B2B lubricant/chemical application domain."""
    blob = f"{product_name} {category} {subcategory} {business_line}".lower()
    cat_u = (category or "").upper()
    if "ПИЩЕВ" in cat_u or "cassida" in blob or "food" in blob or "пищев" in blob or "харчов" in blob:
        return "food_grade"
    if "ТЕХНОЛ" in cat_u or any(k in blob for k in ("сож", "металлообр", "macron", "dromus", "metalina", "cutting", "формов", "закал", "prokat", "прокат")):
        return "metalworking"
    if "АВИА" in cat_u or "СУДОВ" in cat_u or "ЖД-" in cat_u or any(k in blob for k in ("argina", "gadinia", "alexia", "aeroshell", "marine", "судов")):
        return "marine_rail_aviation"
    if any(k in blob for k in ("adblue", "антифриз", "antifreeze", "coolant", "охлажд", "омыват", "тормозн", "brake")):
        return "coolant_adblue"
    if any(k in blob for k in ("gadus", "grease", "смазк", "змазк", "литол", "солидол")):
        return "grease"
    if any(k in blob for k in ("tellus", "гидравл", "гідравл", "hydraulic", "hv ", "hvlp", "hlp")):
        return "hydraulic"
    if any(k in blob for k in ("corena", "компрессор", "компресор", "compressor", "vacuum", "вакуум", "refriger", "холодиль")):
        return "compressor"
    if any(k in blob for k in ("omala", "редуктор", "gear", "tonna", "направляющ", "heat transfer", "термомас", "теплонос", "morlina", "турбин", "turbo", "diala", "трансформ")):
        return "industrial_gear_process"
    if any(k in blob for k in ("rimula", "spirax", "rotella", "uttr", "утто", "stou", "трактор", "грузов", "дизел", "heavy duty", "transm", "трансмис", "atf")):
        return "heavy_diesel_agro"
    if "АВТО" in cat_u or any(k in blob for k in ("helix", "advance", "моторн", "легков")):
        return "passenger_auto"
    if "ИНДУСТР" in cat_u:
        return "industrial_gear_process"
    return "other"


_COMPANY_DOMAIN_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "food_grade": (
        "пищев", "харчов", "молоч", "мяс", "м'яс", "хлеб", "хліб", "кондитер", "конфет",
        "напитк", "пив", "масложир", "сыр", "сир", "птицефабр", "птиц", "колбас", "сахар",
        "цукор", "мук", "борошн", "консерв", "шоколад", "пекар", "food", "агропрод",
    ),
    "metalworking": (
        "металлообр", "металообр", "машиностро", "машинобуд", "металлург", "металург",
        "литье", "литьё", "литт", "штампов", "токарн", "фрезер", "чпу", "cnc", "станок",
        "станк", "верстат", "алюмин", "сталь", "прокат", "трубн", "завод", "мех. обраб",
    ),
    "heavy_diesel_agro": (
        "грузов", "вантаж", "самосвал", "тягач", "автопарк", "логист", "логіст", "перевоз",
        "перевез", "транспорт", "трактор", "комбайн", "агро", "сельхоз", "сільгосп", "ферм",
        "зерн", "элеватор", "елеватор", "карьер", "кар'єр", "щебн", "бетон", "дор строй",
        "дорож", "строит", "будів", "man ", "daf", "volvo", "scania", "mercedes", "kamaz",
        "caterpillar", "john deere", "claas", "komatsu", "jcb", "экскаватор", "погрузчик",
    ),
    "hydraulic": (
        "гидравл", "гідравл", "пресс", "прес", "экскаватор", "екскаватор", "погрузчик",
        "навантажувач", "кран", "термопласт", "литье под давл", "литьё под давл", "caterpillar",
        "komatsu", "jcb", "карьер", "кар'єр", "бетон", "дорож", "строит", "будів", "комбайн",
    ),
    "industrial_gear_process": (
        "цемент", "редуктор", "конвейер", "конвеєр", "дробил", "мельниц", "млин", "завод",
        "фабрик", "комбинат", "комбінат", "производств", "виробництв", "котельн", "тец",
        "энерг", "енерг", "бумаг", "папір", "картон", "деревообр", "стекл", "скло", "кирпич",
        "цегл", "пластик", "полимер", "полімер", "упаков", "горно", "гірнич", "шахт",
    ),
    "compressor": (
        "компрессор", "компресор", "холодиль", "пневмат", "завод", "фабрик", "производств",
        "виробництв", "пищев", "харчов", "стекл", "цемент",
    ),
    "grease": (
        "подшипник", "підшипник", "карьер", "кар'єр", "цемент", "трактор", "комбайн",
        "агро", "сельхоз", "сільгосп", "самосвал", "грузов", "вантаж", "металлург", "завод",
        "дорож", "строит", "будів", "горно", "гірнич",
    ),
    "passenger_auto": (
        "сто ", "автосервис", "автосервіс", "автомагазин", "легков", "розниц", "роздріб",
        "дилер", "автосалон", "шиномонтаж", "запчаст", "азс", "масломаркет",
    ),
    "coolant_adblue": (
        "euro 5", "euro 6", "євро 5", "євро 6", "самосвал", "тягач", "грузов", "вантаж",
        "автопарк", "перевоз", "перевез", "транспорт", "трактор", "комбайн", "сто ", "автосервис",
    ),
    "marine_rail_aviation": (
        "судо", "порт", "морск", "морськ", "речн", "річк", "теплоход", "барж", "авиа",
        "авіа", "железнодорож", "залізнич", "локомотив", "тепловоз", "депо",
    ),
}


def parse_company_equipment_domains(description: str, industry_branch: str = "") -> Set[str]:
    """Extract canonical set of matching product application domains from company text + industry."""
    blob = f" {description or ''} {industry_branch or ''} ".lower()
    matched: Set[str] = set()
    for domain, keywords in _COMPANY_DOMAIN_PATTERNS.items():
        if any(kw in blob for kw in keywords):
            matched.add(domain)
    return matched


class ParsedSpecRollingFeatureBuilder:
    """Computes 8 strictly-causal (< as_of_date) parsed spec & company equipment match features."""

    def __init__(
        self,
        purchases: pd.DataFrame,
        catalogue: pd.DataFrame,
        companies: pd.DataFrame,
        product_ids: Sequence[str],
        *,
        allow_heldout_test: bool = False,
    ):
        self.allow_heldout_test = bool(allow_heldout_test)
        self.product_ids = [str(p).strip() for p in product_ids]
        self.num_products = len(self.product_ids)

        cat_df = catalogue.copy()
        cat_df["product_id"] = cat_df["product_id"].astype(str).str.strip()
        cat_dedup = cat_df.drop_duplicates("product_id").set_index("product_id")

        # Parse specs for ALL catalogue items (so prior purchases outside 841 still count!)
        all_visc: Dict[str, str] = {}
        all_cont: Dict[str, str] = {}
        all_pack: Dict[str, float] = {}
        all_app: Dict[str, str] = {}
        for pid, row in cat_dedup.iterrows():
            text = " ".join(
                str(row.get(c, "")) for c in ("product_name", "base_name", "short_name") if pd.notna(row.get(c))
            )
            cont = str(row.get("container_type", "")).strip().upper() if pd.notna(row.get("container_type")) else ""
            if cont in ("", "NONE", "NAN"):
                cont = ""
            all_visc[pid] = parse_viscosity_grade(text)
            all_cont[pid] = cont
            all_pack[pid] = parse_pack_size_liters_kg(text, cont)
            all_app[pid] = parse_product_application(
                text,
                category=str(row.get("product_category", "")) if pd.notna(row.get("product_category")) else "",
                subcategory=str(row.get("product_subcategory", "")) if pd.notna(row.get("product_subcategory")) else "",
                business_line=str(row.get("business_line", "")) if pd.notna(row.get("business_line")) else "",
            )

        self.prod_visc = [all_visc.get(p, "") for p in self.product_ids]
        self.prod_cont = [all_cont.get(p, "") for p in self.product_ids]
        self.prod_pack = np.array([all_pack.get(p, float("nan")) for p in self.product_ids], dtype=np.float32)
        self.prod_app = [all_app.get(p, "other") for p in self.product_ids]

        # Static pack_size_log_norm for all products
        self.prod_pack_norm = np.where(
            np.isfinite(self.prod_pack) & (self.prod_pack > 0),
            np.clip(np.log1p(np.maximum(self.prod_pack, 0.0)) / math.log1p(1000.0), 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        # Precompute company equipment domain match vectors (P,) per customer
        self.company_app_match: Dict[str, np.ndarray] = {}
        if companies is not None and not companies.empty:
            for _, row in companies.iterrows():
                cid = str(row["customer_id"]).strip()
                desc = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
                ind = str(row.get("industry_branch", "")) if pd.notna(row.get("industry_branch")) else ""
                domains = parse_company_equipment_domains(desc, ind)
                if domains:
                    vec = np.array([1.0 if a in domains else 0.0 for a in self.prod_app], dtype=np.float32)
                    self.company_app_match[cid] = vec

        # Precompute per-customer chronological purchase specs strictly < TEST_START
        self._cust_dates: Dict[str, np.ndarray] = {}
        self._cust_visc: Dict[str, List[str]] = {}
        self._cust_cont: Dict[str, List[str]] = {}
        self._cust_pack: Dict[str, np.ndarray] = {}
        self._cust_app: Dict[str, List[str]] = {}

        if not purchases.empty:
            df = purchases.copy()
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed").dt.normalize()
            if bool(df["purchase_date"].isna().any()) or (
                not self.allow_heldout_test and not bool(df["purchase_date"].lt(TEST_START).all())
            ):
                raise ValueError(f"Development purchases only (< {TEST_START.date()}).")
            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["product_id"] = df["product_id"].astype(str).str.strip()
            df = df.sort_values(["purchase_date", "product_id"], kind="stable")
            for cid, grp in df.groupby("customer_id", sort=False):
                pids_list = grp["product_id"].tolist()
                self._cust_dates[cid] = grp["purchase_date"].to_numpy(dtype="datetime64[ns]")
                self._cust_visc[cid] = [all_visc.get(p, "") for p in pids_list]
                self._cust_cont[cid] = [all_cont.get(p, "") for p in pids_list]
                self._cust_pack[cid] = np.array([all_pack.get(p, float("nan")) for p in pids_list], dtype=np.float32)
                self._cust_app[cid] = [all_app.get(p, "other") for p in pids_list]

        # Fast lookup index lists per unique viscosity / container / application
        self._visc_positions: Dict[str, np.ndarray] = {}
        for i, v in enumerate(self.prod_visc):
            if v:
                self._visc_positions.setdefault(v, []).append(i)
        self._visc_positions = {k: np.array(v, dtype=np.int64) for k, v in self._visc_positions.items()}

        self._cont_positions: Dict[str, np.ndarray] = {}
        for i, c in enumerate(self.prod_cont):
            if c:
                self._cont_positions.setdefault(c, []).append(i)
        self._cont_positions = {k: np.array(v, dtype=np.int64) for k, v in self._cont_positions.items()}

        self._app_positions: Dict[str, np.ndarray] = {}
        for i, a in enumerate(self.prod_app):
            if a and a != "other":
                self._app_positions.setdefault(a, []).append(i)
        self._app_positions = {k: np.array(v, dtype=np.int64) for k, v in self._app_positions.items()}

    def build(self, customer_id: str, as_of_date: Union[str, pd.Timestamp]) -> np.ndarray:
        """Build (P, 8) float32 strictly-causal parsed spec & equipment match features."""
        date_ts = pd.Timestamp(as_of_date)
        if pd.isna(date_ts) or (not self.allow_heldout_test and date_ts >= TEST_START):
            raise ValueError("Development predictions only")

        cid = str(customer_id).strip()
        out = np.zeros((self.num_products, NUM_PARSED_SPEC_FEATURES), dtype=np.float32)
        out[:, 4] = self.prod_pack_norm
        app_vec = self.company_app_match.get(cid)
        if app_vec is not None:
            out[:, 6] = app_vec

        dts = self._cust_dates.get(cid)
        if dts is not None:
            stop = int(dts.searchsorted(date_ts.to_datetime64(), side="left"))
            if stop > 0:
                inv_stop = 1.0 / float(stop)
                # Viscosity shares
                v_cnt = Counter(v for v in self._cust_visc[cid][:stop] if v)
                for v, cnt in v_cnt.items():
                    pos = self._visc_positions.get(v)
                    if pos is not None:
                        out[pos, 0] = 1.0
                        out[pos, 1] = min(1.0, cnt * inv_stop)

                # Container shares & dominant container
                c_cnt = Counter(c for c in self._cust_cont[cid][:stop] if c)
                if c_cnt:
                    dom_c = max(c_cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
                    dom_pos = self._cont_positions.get(dom_c)
                    if dom_pos is not None:
                        out[dom_pos, 2] = 1.0
                    for c, cnt in c_cnt.items():
                        pos = self._cont_positions.get(c)
                        if pos is not None:
                            out[pos, 3] = min(1.0, cnt * inv_stop)

                # Median pack size ratio
                p_slice = self._cust_pack[cid][:stop]
                p_valid = p_slice[np.isfinite(p_slice) & (p_slice > 0)]
                if len(p_valid):
                    med_pack = float(np.median(p_valid))
                    has_p = np.isfinite(self.prod_pack) & (self.prod_pack > 0)
                    out[has_p, 5] = np.clip(
                        np.log((self.prod_pack[has_p] + 0.5) / (med_pack + 0.5)) / 4.0,
                        -1.0,
                        1.0,
                    )

                # Application order share
                a_cnt = Counter(a for a in self._cust_app[cid][:stop] if a and a != "other")
                for a, cnt in a_cnt.items():
                    pos = self._app_positions.get(a)
                    if pos is not None:
                        out[pos, 7] = min(1.0, cnt * inv_stop)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def prepare_or_load_parsed_spec_feature_bundle(
    *,
    cache_root: Path = CACHE_ROOT,
    build: bool = True,
    purchases_path: Union[str, Path] = "data/interim/cleaned_purchases.csv",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build or validate sealed (N, 841, 8) float16 Stage 3E parsed-spec rolling feature arrays."""
    campaign_data, info, _, meta = prepare_or_load_rolling_feature_bundle(build=False, purchases_path=purchases_path)
    product_ids = list(info["product_ids"])
    purchases = load_development_purchases(str(purchases_path))
    catalogue = load_product_catalogue(get_datasets_locs("items"))
    companies = load_companies(get_datasets_locs("companies"), purchases=purchases, minimum_purchases=1)
    src_digest = compute_rolling_source_digest(purchases, catalogue, companies)

    contract = {
        "version": PARSED_SPEC_CACHE_VERSION,
        "namespace": PARSED_SPEC_CACHE_NAMESPACE,
        "names": list(PARSED_SPEC_FEATURE_NAMES),
        "dtype_on_disk": "float16",
        "compute_dtype": "float32",
    }
    identity = json_digest(
        {
            "contract": contract,
            "rolling_features_identity": meta["rolling_features_identity"],
            "observed_source_digest": src_digest["observed_source_digest"],
        }
    )

    root = Path(cache_root) / "parsed_specs_8d"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("parsed_specs_8d cache manifest identity mismatch.")
        for split in ("train", "val"):
            p = root / f"{split}_parsed_specs_8d.npy"
            if file_digest(p) != manifest["feature_sha256"][split]:
                raise ValueError(f"Checksum mismatch for {p}.")
    else:
        if not build:
            raise FileNotFoundError(f"parsed_specs_8d cache missing in {root}; build=True required.")
        builder = ParsedSpecRollingFeatureBuilder(purchases, catalogue, companies, product_ids)
        splits = ("train", "val")
        arrays = [
            np.lib.format.open_memmap(
                root / f"{split}_parsed_specs_8d.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(campaign_data[split]["source"]["customer_ids"]), len(product_ids), NUM_PARSED_SPEC_FEATURES),
            )
            for split in splits
        ]
        started = time.perf_counter()
        origin = pd.Timestamp("2025-01-01")
        for split_idx, split in enumerate(splits):
            cids = campaign_data[split]["source"]["customer_ids"]
            days = campaign_data[split]["source"]["prediction_days"].tolist()
            for row_idx, (cid, day) in enumerate(zip(cids, days)):
                arrays[split_idx][row_idx] = builder.build(cid, origin + pd.Timedelta(days=int(day)))
            arrays[split_idx].flush()
        prep_sec = time.perf_counter() - started
        manifest = {
            "version": PARSED_SPEC_CACHE_VERSION,
            "namespace": PARSED_SPEC_CACHE_NAMESPACE,
            "identity": identity,
            "contract": contract,
            "rolling_features_identity": meta["rolling_features_identity"],
            "feature_sha256": {
                split: file_digest(root / f"{split}_parsed_specs_8d.npy") for split in splits
            },
            "coverage": {
                "viscosity_parsed": sum(1 for v in builder.prod_visc if v),
                "pack_parsed": int(np.isfinite(builder.prod_pack).sum()),
                "application_parsed_non_other": sum(1 for a in builder.prod_app if a != "other"),
                "companies_with_equipment_tags": len(builder.company_app_match),
            },
            "preparation_seconds": float(prep_sec),
            "heldout_access": False,
        }
        atomic_json(manifest, manifest_path, overwrite=False)

    mmaps = {
        split: np.load(root / f"{split}_parsed_specs_8d.npy", mmap_mode="r")
        for split in ("train", "val")
    }
    return mmaps, manifest
