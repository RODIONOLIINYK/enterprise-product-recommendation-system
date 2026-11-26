"""Configuration and variant definitions for isolated cleaned-product-description experiments."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple

from recommender.experiments.static_enrichment.config import VariantSpec


CACHE_DIR = Path(os.environ.get("RECSYS_DESC_CLEAN_CACHE", "data/interim/description_cleaning_v1"))
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_DESC_CLEAN_ROOT", "checkpoints/description-cleaning-v1"))
EVIDENCE_ROOT = Path(os.environ.get("RECSYS_DESC_CLEAN_EVIDENCE", "experiments/description_cleaning_v1"))

BASELINE_REFERENCE_JSON = Path("experiments/embedding_ablation_v1/reference.json")
BASELINE_REFERENCE_DIR = Path("checkpoints/stage1-mlp-reference/no-brand-v1")
BASELINE_TEXT_CACHE_DIR = Path("data/interim/embedding_ablation_v1/text_embeddings")

NO_BRAND_PRODUCT_EMBEDDINGS: Tuple[str, ...] = (
    "category",
    "subcategory",
    "business_line",
    "tier",
    "unit",
    "country_of_origin",
)
NO_BRAND_CUSTOMER_EMBEDDINGS: Tuple[str, ...] = ("sales_branch",)

DESCRIPTION_TEXT_RECIPES: Tuple[str, ...] = (
    "t1_fam",
    "t1_fam_desc_html_v1",
    "t1_fam_desc_clean_v1",
)


@dataclass(frozen=True)
class DescriptionVariantSpec:
    """Variant definition matching active no-brand-v1 reference except product text description recipe."""

    name: str
    description_mode: str  # "none" | "html_basic" | "clean_v1"
    variant_spec: VariantSpec
    description: str


def build_description_variant_specs() -> dict[str, DescriptionVariantSpec]:
    """Return baseline reference descriptor plus D_html and D_clean Stage 1 + uv_47 MLP variants."""
    return {
        "no_brand_v1": DescriptionVariantSpec(
            name="no_brand_v1",
            description_mode="none",
            variant_spec=VariantSpec(
                name="E_no_brand",
                text_recipe="t1_fam",
                product_embeddings=NO_BRAND_PRODUCT_EMBEDDINGS,
                customer_embeddings=NO_BRAND_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description="Active reference baseline (E_no_brand + uv_47, t1_fam text without descriptions).",
        ),
        "D_html": DescriptionVariantSpec(
            name="D_html",
            description_mode="html_basic",
            variant_spec=VariantSpec(
                name="D_html",
                text_recipe="t1_fam_desc_html_v1",
                product_embeddings=NO_BRAND_PRODUCT_EMBEDDINGS,
                customer_embeddings=NO_BRAND_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description="Appends legacy clean_html_description(raw_description) to exact t1_fam prefix.",
        ),
        "D_clean": DescriptionVariantSpec(
            name="D_clean",
            description_mode="clean_v1",
            variant_spec=VariantSpec(
                name="D_clean",
                text_recipe="t1_fam_desc_clean_v1",
                product_embeddings=NO_BRAND_PRODUCT_EMBEDDINGS,
                customer_embeddings=NO_BRAND_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description="Appends deterministic desc_clean_v1 cleaned description to exact t1_fam prefix.",
        ),
    }
