"""Unit tests for Stage 3E parsed product specs, company equipment matching, and strict causality."""

import numpy as np
import pandas as pd
import pytest

from recommender.experiments.roadmap_v2.parsed_spec_features import (
    PARSED_SPEC_FEATURE_NAMES,
    ParsedSpecRollingFeatureBuilder,
    parse_company_equipment_domains,
    parse_pack_size_liters_kg,
    parse_product_application,
    parse_viscosity_grade,
)


def test_parsers():
    assert parse_viscosity_grade("SHELL Rimula R4 L 15W-40 209л") == "15W-40"
    assert parse_viscosity_grade("SHELL Tellus S2 MX 46 20л") == "ISO46"
    assert parse_viscosity_grade("SHELL Gadus S2 V220 2 18кг") == "NLGI2"
    assert np.isclose(parse_pack_size_liters_kg("SHELL Rimula R4 L 15W-40 209л"), 209.0)
    assert np.isclose(parse_pack_size_liters_kg("Смазка 400г"), 0.4)
    assert parse_product_application("Cassida Fluid GL 220", category="ПИЩЕВЫЕ-СМ") == "food_grade"
    assert parse_product_application("SHELL Tellus S2 MX 46", category="ИНДУСТР-СМ") == "hydraulic"
    assert "food_grade" in parse_company_equipment_domains("Производство конфет и шоколада")
    assert "heavy_diesel_agro" in parse_company_equipment_domains("В парке 30 самосвалов MAN и Caterpillar")


def test_parsed_specs_strict_causality():
    catalogue = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "product_name": [
                "SHELL Rimula R4 L 15W-40 209л",
                "SHELL Rimula R5 E 15W-40 20л",
                "Cassida Fluid GL 220 20л",
            ],
            "base_name": ["Rimula R4 L 15W-40", "Rimula R5 E 15W-40", "Cassida Fluid GL 220"],
            "short_name": ["Rimula 209л", "Rimula 20л", "Cassida 20л"],
            "container_type": ["БОЧКА", "ВЕДРО", "ВЕДРО"],
            "product_category": ["АВТО-СМ", "АВТО-СМ", "ПИЩЕВЫЕ-СМ"],
            "product_subcategory": ["", "", ""],
            "business_line": ["", "", ""],
        }
    )
    companies = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "description": ["Парк грузовиков MAN, самосвалы"],
            "industry_branch": ["Транспорт"],
        }
    )
    purchases = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "purchase_date": pd.to_datetime(["2025-01-10", "2025-02-15"]),
            "product_id": ["P1", "P3"],
            "quantity": [1.0, 1.0],
        }
    )
    b_full = ParsedSpecRollingFeatureBuilder(purchases, catalogue, companies, ["P1", "P2", "P3"])
    b_trunc = ParsedSpecRollingFeatureBuilder(
        purchases.loc[purchases["purchase_date"] < pd.Timestamp("2025-02-15")],
        catalogue,
        companies,
        ["P1", "P2", "P3"],
    )
    f_full = b_full.build("C1", "2025-02-15")
    f_trunc = b_trunc.build("C1", "2025-02-15")
    np.testing.assert_allclose(f_full, f_trunc, rtol=0.0, atol=0.0)

    idx = {n: i for i, n in enumerate(PARSED_SPEC_FEATURE_NAMES)}
    # P2 is unseen SKU, DIFFERENT base_name, but SAME viscosity 15W-40 as P1!
    assert f_full[1, idx["same_viscosity_grade_bought"]] == 1.0
    assert f_full[2, idx["same_viscosity_grade_bought"]] == 0.0
    # Company is a trucking fleet -> heavy_diesel_agro matches P1 & P2, NOT food_grade P3!
    assert f_full[0, idx["app_matches_company_equipment"]] == 1.0
    assert f_full[1, idx["app_matches_company_equipment"]] == 1.0
    assert f_full[2, idx["app_matches_company_equipment"]] == 0.0

    with pytest.raises(ValueError, match="Development predictions only"):
        b_full.build("C1", "2026-02-19")
