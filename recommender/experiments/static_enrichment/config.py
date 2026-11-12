"""Configuration and variant specifications for the static enrichment campaign."""

from dataclasses import dataclass
import os
from pathlib import Path


CACHE_DIR = Path(os.environ.get("RECSYS_STATIC_CACHE", "data/interim/static_enrichment_v1"))
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_STATIC_ROOT", "checkpoints/static-enrichment-v1"))
CLEAN_V2_ROOT = ARTIFACT_ROOT / "clean_v2"
EVIDENCE_ROOT = Path(os.environ.get("RECSYS_STATIC_EVIDENCE", "experiments/static_enrichment"))

DEFAULT_RETRIEVER = Path("checkpoints/retriever-discovery-reference/best_retriever.pt")
DEFAULT_DISCOVERY_RANKER = Path("checkpoints/ranker/best_discovery_ranker.pt")
DEFAULT_DESCRIPTION_PATH = Path("data/raw/описание.csv")
DYNAMIC_CACHE_DIR = Path("data/interim/dynamic_campaign_v1")

MIN_TRAIN_CATEGORY_COUNT = 3

TRANSFERRED_CUSTOMER_STAT_NAMES = (
    "distinct_orders",
    "total_quantity",
    "distinct_products",
    "distinct_categories",
    "median_order_gap",
    "median_purchase_quantity",
)

PRODUCT_ATTR_COLUMNS = {
    "brand": "brand",
    "category": "product_category",
    "subcategory": "product_subcategory",
    "business_line": "business_line",
    "product_family": "product_family",
    "tier": "tier",
    "unit": "unit",
    "country_of_origin": "country_of_origin",
}

CUSTOMER_ATTR_COLUMNS = {
    "sales_branch": "sales_branch",
    "vat_status": "vat_status",
}


@dataclass(frozen=True)
class VariantSpec:
    """Architecture and text-recipe definition for a Stage 1 static enrichment variant."""

    name: str
    text_recipe: str
    product_embeddings: tuple[str, ...] = ()
    customer_embeddings: tuple[str, ...] = ()
    is_control: bool = False
    customer_tabular_features: int = 6
    product_demand_features: int = 18


def build_variant_specs(*, best_t: str = "T1", best_f: str = "F_emb", best_p: str = "P") -> dict[str, VariantSpec]:
    """Return the variant registry, parameterized by screening winners for chained stages."""
    t_recipe = "t2" if best_t == "T2" else "t1"
    f_recipe = f"{t_recipe}_fam" if best_f == "F_txt" else t_recipe
    base_a = ("category", "subcategory", "business_line", "brand")
    f_embs = (*base_a, "product_family") if best_f == "F_emb" else base_a
    p_embs_full = (*f_embs, "tier", "unit", "country_of_origin")
    p_embs_no_unit = (*f_embs, "tier", "country_of_origin")
    best_p_embs = p_embs_no_unit if best_p == "P_no_unit" else p_embs_full
    return {
        "C0": VariantSpec(name="C0", text_recipe="c0_s0", is_control=True),
        "T1": VariantSpec(name="T1", text_recipe="t1"),
        "T2": VariantSpec(name="T2", text_recipe="t2"),
        "A": VariantSpec(name="A", text_recipe=t_recipe, product_embeddings=base_a),
        "F_emb": VariantSpec(name="F_emb", text_recipe=t_recipe, product_embeddings=(*base_a, "product_family")),
        "F_txt": VariantSpec(name="F_txt", text_recipe=f"{t_recipe}_fam", product_embeddings=base_a),
        "P": VariantSpec(name="P", text_recipe=f_recipe, product_embeddings=p_embs_full),
        "P_no_unit": VariantSpec(name="P_no_unit", text_recipe=f_recipe, product_embeddings=p_embs_no_unit),
        "U0": VariantSpec(
            name="U0",
            text_recipe=f_recipe,
            product_embeddings=best_p_embs,
            customer_embeddings=("sales_branch",),
        ),
        "U1": VariantSpec(
            name="U1",
            text_recipe=f_recipe,
            product_embeddings=best_p_embs,
            customer_embeddings=("sales_branch", "vat_status"),
        ),
    }


def build_clean_v2_variant_specs() -> dict[str, VariantSpec]:
    """Return clean-tower Stage 1 variants (0 customer tabular stats, 12D container_type only)."""
    p_full = (
        "category",
        "subcategory",
        "business_line",
        "brand",
        "tier",
        "unit",
        "country_of_origin",
    )
    p_nosubcat = (
        "category",
        "business_line",
        "brand",
        "tier",
        "unit",
        "country_of_origin",
    )
    return {
        "C0_clean": VariantSpec(
            name="C0_clean",
            text_recipe="c0_s0",
            is_control=True,
            customer_tabular_features=0,
            product_demand_features=12,
        ),
        "P_t1_fam_clean": VariantSpec(
            name="P_t1_fam_clean",
            text_recipe="t1_fam",
            product_embeddings=p_full,
            customer_tabular_features=0,
            product_demand_features=12,
        ),
        "U0_t1_fam_clean": VariantSpec(
            name="U0_t1_fam_clean",
            text_recipe="t1_fam",
            product_embeddings=p_full,
            customer_embeddings=("sales_branch",),
            customer_tabular_features=0,
            product_demand_features=12,
        ),
        "U0_t1_fam_nosubcat": VariantSpec(
            name="U0_t1_fam_nosubcat",
            text_recipe="t1_fam",
            product_embeddings=p_nosubcat,
            customer_embeddings=("sales_branch",),
            customer_tabular_features=0,
            product_demand_features=12,
        ),
    }
