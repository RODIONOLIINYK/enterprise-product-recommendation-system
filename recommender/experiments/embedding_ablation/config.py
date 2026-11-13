"""Configuration and variant definitions for isolated embedding and text ablations."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Tuple

from recommender.experiments.static_enrichment.config import VariantSpec


CACHE_DIR = Path(os.environ.get("RECSYS_EMB_ABLATION_CACHE", "data/interim/embedding_ablation_v1"))
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_EMB_ABLATION_ROOT", "checkpoints/embedding-ablation-v1"))
EVIDENCE_ROOT = Path(os.environ.get("RECSYS_EMB_ABLATION_EVIDENCE", "experiments/embedding_ablation_v1"))

CLEAN_V2_U0_DIR = Path("checkpoints/static-enrichment-v1/clean_v2/runs/stage1-U0_t1_fam_clean-seed42")
STATIC_ENRICHMENT_V1_MANIFEST = Path("data/interim/static_enrichment_v1/enrichment_manifest.json")

BASELINE_PRODUCT_EMBEDDINGS: Tuple[str, ...] = (
    "category",
    "subcategory",
    "business_line",
    "brand",
    "tier",
    "unit",
    "country_of_origin",
)
BASELINE_CUSTOMER_EMBEDDINGS: Tuple[str, ...] = ("sales_branch",)

ABLATED_TEXT_FIELDS: Tuple[str, ...] = (
    "base_name",
    "product_subcategory",
    "product_family",
    "product_category",
    "brand",
    "business_line",
)
INAPPLICABLE_TEXT_ABLATIONS: Dict[str, str] = {
    "unit": "unit is not an explicit field in t1_fam text rendering; do not invent a text-unit ablation.",
}

ABLATED_PRODUCT_EMBEDDINGS: Tuple[str, ...] = (
    "category",
    "subcategory",
    "brand",
    "unit",
    "business_line",
)
INAPPLICABLE_EMBEDDING_ABLATIONS: Dict[str, str] = {
    "product_family": "There is no product_family categorical embedding in the U0_t1_fam_clean baseline.",
}

TEXT_ABLATION_RECIPE_NAMES: Dict[str, str] = {
    "base_name": "t1_fam_no_base_name",
    "product_subcategory": "t1_fam_no_subcategory",
    "product_family": "t1_fam_no_family",
    "product_category": "t1_fam_no_category",
    "brand": "t1_fam_no_brand",
    "business_line": "t1_fam_no_business_line",
}

TEXT_ABLATION_VARIANT_NAMES: Dict[str, str] = {
    "base_name": "T_no_base_name",
    "product_subcategory": "T_no_subcategory",
    "product_family": "T_no_family",
    "product_category": "T_no_category",
    "brand": "T_no_brand",
    "business_line": "T_no_business_line",
}

EMBEDDING_ABLATION_VARIANT_NAMES: Dict[str, str] = {
    "category": "E_no_category",
    "subcategory": "E_no_subcategory",
    "brand": "E_no_brand",
    "unit": "E_no_unit",
    "business_line": "E_no_business_line",
}


@dataclass(frozen=True)
class AblationSpec:
    """Metadata and VariantSpec for an isolated U0_t1_fam_clean ablation."""

    name: str
    group: str  # "baseline" | "text_deletion" | "embedding_deletion" | "exploratory_combined"
    ablated_target: str
    omitted_text_fields: Tuple[str, ...]
    omitted_product_embeddings: Tuple[str, ...]
    variant_spec: VariantSpec
    description: str


def build_ablation_specs() -> Dict[str, AblationSpec]:
    """Return baseline + 6 text-deletion + 5 product-embedding-deletion ablation specs."""
    specs: Dict[str, AblationSpec] = {
        "U0_t1_fam_clean": AblationSpec(
            name="U0_t1_fam_clean",
            group="baseline",
            ablated_target="none",
            omitted_text_fields=(),
            omitted_product_embeddings=(),
            variant_spec=VariantSpec(
                name="U0_t1_fam_clean",
                text_recipe="t1_fam",
                product_embeddings=BASELINE_PRODUCT_EMBEDDINGS,
                customer_embeddings=BASELINE_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description="Baseline U0_t1_fam_clean (t1_fam text, 7 product embeddings, sales_branch customer embedding, 0 tabular stats, 12D container_type).",
        )
    }

    for field in ABLATED_TEXT_FIELDS:
        vname = TEXT_ABLATION_VARIANT_NAMES[field]
        recipe = TEXT_ABLATION_RECIPE_NAMES[field]
        specs[vname] = AblationSpec(
            name=vname,
            group="text_deletion",
            ablated_target=field,
            omitted_text_fields=(field,),
            omitted_product_embeddings=(),
            variant_spec=VariantSpec(
                name=vname,
                text_recipe=recipe,
                product_embeddings=BASELINE_PRODUCT_EMBEDDINGS,
                customer_embeddings=BASELINE_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description=f"Omit structured text field {field!r} from t1_fam rendering; keep all 7 product embeddings fixed.",
        )

    for attr in ABLATED_PRODUCT_EMBEDDINGS:
        vname = EMBEDDING_ABLATION_VARIANT_NAMES[attr]
        remaining = tuple(p for p in BASELINE_PRODUCT_EMBEDDINGS if p != attr)
        specs[vname] = AblationSpec(
            name=vname,
            group="embedding_deletion",
            ablated_target=attr,
            omitted_text_fields=(),
            omitted_product_embeddings=(attr,),
            variant_spec=VariantSpec(
                name=vname,
                text_recipe="t1_fam",
                product_embeddings=remaining,
                customer_embeddings=BASELINE_CUSTOMER_EMBEDDINGS,
                is_control=False,
                customer_tabular_features=0,
                product_demand_features=12,
            ),
            description=f"Remove learned 16D product embedding {attr!r}; keep t1_fam text and remaining static inputs fixed.",
        )

    return specs


def make_exploratory_combined_spec(
    name: str,
    *,
    omitted_text_fields: Tuple[str, ...] = (),
    omitted_product_embeddings: Tuple[str, ...] = (),
    recipe_name: str,
    description: str = "",
) -> AblationSpec:
    """Create an explicitly labeled exploratory cumulative combination spec."""
    for field in omitted_text_fields:
        if field not in ABLATED_TEXT_FIELDS:
            raise ValueError(f"Unsupported text field for omission: {field!r}")
    for attr in omitted_product_embeddings:
        if attr not in BASELINE_PRODUCT_EMBEDDINGS:
            raise ValueError(f"Unsupported product embedding for omission: {attr!r}")
    remaining = tuple(p for p in BASELINE_PRODUCT_EMBEDDINGS if p not in set(omitted_product_embeddings))
    return AblationSpec(
        name=name,
        group="exploratory_combined",
        ablated_target="+".join([*omitted_text_fields, *omitted_product_embeddings]) or "none",
        omitted_text_fields=tuple(omitted_text_fields),
        omitted_product_embeddings=tuple(omitted_product_embeddings),
        variant_spec=VariantSpec(
            name=name,
            text_recipe=recipe_name,
            product_embeddings=remaining,
            customer_embeddings=BASELINE_CUSTOMER_EMBEDDINGS,
            is_control=False,
            customer_tabular_features=0,
            product_demand_features=12,
        ),
        description=description or f"Exploratory cumulative ablation: text_omit={omitted_text_fields}, emb_omit={omitted_product_embeddings}.",
    )
