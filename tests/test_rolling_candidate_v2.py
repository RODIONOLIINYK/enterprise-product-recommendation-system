"""Synthetic and artifact-gated regression tests for rolling_observed_v2 and candidate self-attention."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.candidates import (
    PAIRWISE_FEATURE_NAMES,
    PointInTimeFeatureStore,
    STATISTICS_POLICY_FROZEN,
    STATISTICS_POLICY_ROLLING,
)
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import (
    DEFAULT_NEW_STATISTICS_POLICY,
    NAMES as DYNAMIC_FEATURE_NAMES,
    contract as feature_contract_for_policy,
    quantize_features,
    rolling_contract,
    rolling_schema_digest,
    schema_digest as feature_schema_digest_for_policy,
)
from recommender.experiments.mlp_feature_ablation.config import (
    REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
    REPEAT_RESTORED_RECIPE_NAME,
    REPEAT_RESTORED_VARIANT_NAME,
)
from recommender.experiments.mlp_feature_ablation.model import FeatureAblatedStage1Selector
from recommender.experiments.mlp_feature_ablation.schema import build_feature_ablation_spec
from recommender.experiments.rolling_candidate_v2.config import (
    CACHE_ROOT,
    CANDIDATE_SOURCE_ROLLING_SELECTOR,
    EVIDENCE_ROOT,
    LEGACY_V2_CACHE_ROOT,
)
from recommender.experiments.rolling_candidate_v2.data import (
    FEATURE_CACHE_VERSION,
    ROLLING_CACHE_NAMESPACE,
    SLATE_CACHE_VERSION,
    RollingDynamicFeatureAdapter,
    RollingSlateDataset,
    compute_rolling_source_digest,
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_slate_cache,
    validate_rolling_feature_manifest,
)
from recommender.experiments.rolling_candidate_v2.evaluate import (
    evaluate_rolling_slate_metrics,
    paired_rolling_intervals,
)
from recommender.experiments.rolling_candidate_v2.model import (
    RollingCandidateAttentionRanker,
    load_rolling_ranker_checkpoint,
    save_rolling_ranker_checkpoint,
)
from recommender.ranker import RankerLoss, UPSCANRRanker


def _idx(name: str) -> int:
    return DYNAMIC_FEATURE_NAMES.index(name)


def _synthetic_market_tables():
    # Train < 2025-10-01, Val in [2025-10-01, 2026-02-18].
    catalogue = pd.DataFrame(
        [
            {
                "product_id": "SKU_HOT_A",
                "product_category": "CAT_X",
                "product_subcategory": "SUB_X1",
                "business_line": "LINE_1",
                "brand": "BRAND_1",
                "product_family": "FAM_A",
            },
            {
                "product_id": "SKU_VAL_B",
                "product_category": "CAT_X",
                "product_subcategory": "SUB_X1",
                "business_line": "LINE_1",
                "brand": "BRAND_1",
                "product_family": "FAM_B",
            },
            {
                "product_id": "SKU_COMP_C",
                "product_category": "CAT_Y",
                "product_subcategory": "SUB_Y1",
                "business_line": "LINE_2",
                "brand": "BRAND_2",
                "product_family": "FAM_C",
            },
        ]
    )
    companies = pd.DataFrame(
        [
            {"customer_id": "CUST_TARGET", "city": "City1", "region_code": "Reg1", "industry_branch": "IND_1"},
            {"customer_id": "CUST_MKT1", "city": "City1", "region_code": "Reg1", "industry_branch": "IND_1"},
            {"customer_id": "CUST_MKT2", "city": "City2", "region_code": "Reg1", "industry_branch": "IND_1"},
            {"customer_id": "CUST_OTHER", "city": "City3", "region_code": "Reg2", "industry_branch": "IND_2"},
        ]
    )
    rows = [
        # Training history (< 2025-10-01): CUST_TARGET has two baskets so sequential transitions exist.
        {"customer_id": "CUST_TARGET", "product_id": "SKU_HOT_A", "purchase_date": "2025-07-01", "quantity": 5.0},
        {"customer_id": "CUST_TARGET", "product_id": "SKU_HOT_A", "purchase_date": "2025-09-10", "quantity": 5.0},
        {"customer_id": "CUST_MKT1", "product_id": "SKU_HOT_A", "purchase_date": "2025-09-15", "quantity": 4.0},
        {"customer_id": "CUST_MKT1", "product_id": "SKU_VAL_B", "purchase_date": "2025-09-15", "quantity": 2.0},
        # Earlier validation purchases (2025-10-05 & 2025-10-06) before val query 2025-10-20:
        # Market buys SKU_VAL_B cleanly across IND_1 buyers and triggers sequential transition SKU_HOT_A -> SKU_VAL_B + same-basket co-occurrence SKU_VAL_B <-> SKU_COMP_C.
        {"customer_id": "CUST_MKT1", "product_id": "SKU_HOT_A", "purchase_date": "2025-10-02", "quantity": 3.0},
        {"customer_id": "CUST_MKT1", "product_id": "SKU_VAL_B", "purchase_date": "2025-10-05", "quantity": 6.0},
        {"customer_id": "CUST_MKT1", "product_id": "SKU_COMP_C", "purchase_date": "2025-10-05", "quantity": 2.0},
        {"customer_id": "CUST_MKT2", "product_id": "SKU_VAL_B", "purchase_date": "2025-10-06", "quantity": 4.0},
        # Same-date addition on 2025-10-20 (must NEVER affect 2025-10-20 query) & future addition on 2025-10-25.
        {"customer_id": "CUST_OTHER", "product_id": "SKU_VAL_B", "purchase_date": "2025-10-20", "quantity": 10.0},
        {"customer_id": "CUST_OTHER", "product_id": "SKU_COMP_C", "purchase_date": "2025-10-25", "quantity": 10.0},
    ]
    purchases = pd.DataFrame(rows)
    purchases["purchase_date"] = pd.to_datetime(purchases["purchase_date"])
    return purchases, catalogue, companies


def test_rolling_vs_frozen_policy_causality_windows_and_relationships():
    base_purchases, catalogue, companies = _synthetic_market_tables()
    product_ids = ["SKU_HOT_A", "SKU_VAL_B", "SKU_COMP_C"]

    # 1. Policy defaults & schema digests are strictly separated.
    assert DEFAULT_NEW_STATISTICS_POLICY == STATISTICS_POLICY_ROLLING
    assert feature_schema_digest_for_policy(STATISTICS_POLICY_FROZEN) != rolling_schema_digest()
    assert rolling_contract()["version"] == 2
    assert rolling_contract()["statistics_policy_id"] == STATISTICS_POLICY_ROLLING

    # 2. Rolling builder reflects earlier validation purchases (< t) in globals, PointInTimeFeatureStore
    #    original-13 fields, and relationship tables; frozen builder freezes them at TRAIN_CUTOFF.
    rolling_builder = DynamicFeatureBuilder(
        base_purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING, allow_non_monotonic=True
    )
    frozen_builder = DynamicFeatureBuilder(
        base_purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_FROZEN, allow_non_monotonic=True
    )

    f_roll_early = rolling_builder.build("CUST_TARGET", "2025-10-01")
    f_roll_late = rolling_builder.build("CUST_TARGET", "2025-10-20")
    f_froz_early = frozen_builder.build("CUST_TARGET", "2025-10-01")
    f_froz_late = frozen_builder.build("CUST_TARGET", "2025-10-20")

    b = 1  # SKU_VAL_B
    c = 2  # SKU_COMP_C
    # Frozen globals stay identical on 2025-10-01 vs 2025-10-20 despite validation buys on Oct 5-6:
    for col_name in (
        "global_sku_popularity_norm",
        "industry_sku_affinity",
        "sku_orders_7d",
        "sku_orders_30d",
        "sku_orders_90d",
        "sku_buyers_30d",
        "sku_growth_30d",
        "category_relative_share",
        "global_sale_age",
        "sku_last_lift",
        "sku_recent3_lift",
        "category_last_lift",
        "family_last_lift",
        "line_last_lift",
        "sku_support_confidence",
        "hierarchical_backoff",
        "same_basket_lift",
    ):
        col = _idx(col_name)
        assert f_froz_late[b, col] == pytest.approx(f_froz_early[b, col], abs=1e-7), col_name

    # Rolling globals MUST increase for SKU_VAL_B and SKU_COMP_C between 2025-10-01 and 2025-10-20:
    assert f_roll_late[b, _idx("global_sku_popularity_norm")] > f_roll_early[b, _idx("global_sku_popularity_norm")]
    assert f_roll_late[b, _idx("industry_sku_affinity")] > f_roll_early[b, _idx("industry_sku_affinity")]
    assert f_roll_late[b, _idx("sku_orders_30d")] > f_roll_early[b, _idx("sku_orders_30d")]
    assert f_roll_late[b, _idx("sku_orders_90d")] > f_roll_early[b, _idx("sku_orders_90d")]
    assert f_roll_late[b, _idx("sku_buyers_30d")] > f_roll_early[b, _idx("sku_buyers_30d")]
    assert f_roll_late[b, _idx("category_relative_share")] > f_roll_early[b, _idx("category_relative_share")]
    assert f_roll_late[b, _idx("global_sale_age")] < f_roll_early[b, _idx("global_sale_age")]
    assert f_roll_late[b, _idx("sku_last_lift")] > f_roll_early[b, _idx("sku_last_lift")]
    assert f_roll_late[b, _idx("sku_support_confidence")] > f_roll_early[b, _idx("sku_support_confidence")]
    assert f_roll_late[c, _idx("sku_orders_30d")] > f_roll_early[c, _idx("sku_orders_30d")]

    # PointInTimeFeatureStore.popularity_candidates also rolls under rolling_observed_v2.
    pop_roll = rolling_builder.store.popularity_candidates(pd.Timestamp("2025-10-20"), set(product_ids))
    pop_froz = frozen_builder.store.popularity_candidates(pd.Timestamp("2025-10-20"), set(product_ids))
    assert "SKU_COMP_C" in pop_roll and "SKU_COMP_C" not in pop_froz

    # 3. Adding extra same-date (2025-10-20) and future (2025-10-26) purchases cannot alter 2025-10-20!
    augmented = pd.concat(
        [
            base_purchases,
            pd.DataFrame(
                [
                    {
                        "customer_id": "CUST_MKT1",
                        "product_id": "SKU_COMP_C",
                        "purchase_date": pd.Timestamp("2025-10-20"),
                        "quantity": 99.0,
                    },
                    {
                        "customer_id": "CUST_MKT2",
                        "product_id": "SKU_VAL_B",
                        "purchase_date": pd.Timestamp("2025-10-26"),
                        "quantity": 99.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    aug_builder = DynamicFeatureBuilder(
        augmented, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING, allow_non_monotonic=True
    )
    f_roll_aug_oct20 = aug_builder.build("CUST_TARGET", "2025-10-20")
    np.testing.assert_allclose(f_roll_aug_oct20, f_roll_late, atol=0.0, rtol=0.0)

    # 4. Window expiry without any new purchase:
    #    Oct 5 & Oct 6 purchases fall out of 7d by Oct 15 and out of 30d by Nov 10 (leaving Oct 20),
    #    Oct 20 falls out of 30d by Nov 25, and all Oct purchases fall out of 90d by Jan 25.
    f_oct10 = rolling_builder.build("CUST_TARGET", "2025-10-10")  # Oct 5/6 within 7d
    f_oct15 = rolling_builder.build("CUST_TARGET", "2025-10-15")  # Oct 5/6 expired from 7d, still in 30d
    f_oct21 = rolling_builder.build("CUST_TARGET", "2025-10-21")  # Oct 5, 6, and 20 within 30d
    f_nov10 = rolling_builder.build("CUST_TARGET", "2025-11-10")  # Oct 5/6 expired from 30d, Oct 20 still in 30d
    f_nov25 = rolling_builder.build("CUST_TARGET", "2025-11-25")  # Oct 20 expired from 30d, still in 90d
    f_jan25 = rolling_builder.build("CUST_TARGET", "2026-01-25")  # All Oct purchases expired from 90d!

    assert f_oct10[b, _idx("sku_orders_7d")] > f_oct15[b, _idx("sku_orders_7d")] == 0.0
    assert f_oct21[b, _idx("sku_orders_30d")] > f_nov10[b, _idx("sku_orders_30d")] > f_nov25[b, _idx("sku_orders_30d")] == 0.0
    assert f_nov25[b, _idx("sku_orders_90d")] > f_jan25[b, _idx("sku_orders_90d")] == 0.0
    assert f_jan25[b, _idx("global_sale_age")] > f_oct10[b, _idx("global_sale_age")]


def test_adapter_cache_parity_and_legacy_rejection(tmp_path):
    purchases, catalogue, companies = _synthetic_market_tables()
    product_ids = ["SKU_HOT_A", "SKU_VAL_B", "SKU_COMP_C"]

    # Reject legacy frozen policy by default in RollingDynamicFeatureAdapter.
    with pytest.raises(ValueError, match="legacy-only and prohibited"):
        RollingDynamicFeatureAdapter(
            purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_FROZEN
        )

    # Monotonic cache builder vs non-monotonic (reversed) adapter queries have exact 0.0 quantized parity.
    cache_builder = DynamicFeatureBuilder(
        purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING, allow_non_monotonic=False
    )
    dates = ["2025-10-01", "2025-10-12", "2025-10-20"]
    cached_quantized = {d: quantize_features(cache_builder.build("CUST_TARGET", d)) for d in dates}

    adapter = RollingDynamicFeatureAdapter(purchases, catalogue, companies, product_ids)
    assert adapter.statistics_policy == STATISTICS_POLICY_ROLLING
    for d in reversed(dates):
        live = adapter.build_quantized("CUST_TARGET", d)
        np.testing.assert_allclose(live, cached_quantized[d], atol=0.0, rtol=0.0)

    # Active selector scoring through RollingDynamicFeatureAdapter masks disallowed items to -inf
    # and is invariant to candidate ordering.
    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    torch.manual_seed(7)
    selector = FeatureAblatedStage1Selector(spec)
    u = torch.randn(128)
    v = torch.randn(len(product_ids), 128)
    stats6 = torch.rand(6)
    mask = torch.tensor([True, True, False])

    s_fwd = adapter.score_active_selector(
        selector,
        u=u,
        v=v,
        customer_stats_6=stats6,
        customer_id="CUST_TARGET",
        as_of_date="2025-10-20",
        candidate_ids=product_ids,
        allowed_mask=mask,
    )
    s_rev = adapter.score_active_selector(
        selector,
        u=u,
        v=v,
        customer_stats_6=stats6,
        customer_id="CUST_TARGET",
        as_of_date="2025-10-20",
        candidate_ids=list(reversed(product_ids)),
        allowed_mask=torch.flip(mask, dims=[0]),
    )
    assert float(s_fwd[0, 2].item()) == -torch.inf
    torch.testing.assert_close(torch.flip(s_rev, dims=[1])[:, :2], s_fwd[:, :2], atol=1e-6, rtol=1e-6)

    # Validate rolling manifest rejects legacy frozen manifests/digests and RollingSlateDataset rejects mlp_score.
    frozen_manifest = {
        "version": 1,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "identity": "abc",
        "statistics_policy": STATISTICS_POLICY_FROZEN,
        "feature_contract": feature_contract_for_policy(STATISTICS_POLICY_FROZEN),
        "schema_digest": feature_schema_digest_for_policy(STATISTICS_POLICY_FROZEN),
        "observed_source_digest": "d" * 64,
        "test_access": False,
        "heldout_access": False,
    }
    with pytest.raises(ValueError, match="legacy frozen"):
        validate_rolling_feature_manifest(frozen_manifest, expected_identity="abc")

    v2_unsealed_rolling = {
        "version": 2,
        "namespace": "rolling_candidate_v2",
        "identity": "abc",
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_contract": feature_contract_for_policy(STATISTICS_POLICY_ROLLING),
        "schema_digest": feature_schema_digest_for_policy(STATISTICS_POLICY_ROLLING),
        "test_access": False,
        "heldout_access": False,
    }
    with pytest.raises(ValueError, match="legacy frozen or unsealed v2"):
        validate_rolling_feature_manifest(v2_unsealed_rolling, expected_identity="abc")

    current_version_frozen_contract = {
        **frozen_manifest,
        "version": FEATURE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
    }
    with pytest.raises(ValueError, match="statistics_policy mismatch"):
        validate_rolling_feature_manifest(current_version_frozen_contract, expected_identity="abc")

    valid_v3_manifest = {
        "version": FEATURE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "identity": "abc",
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_contract": feature_contract_for_policy(STATISTICS_POLICY_ROLLING),
        "schema_digest": feature_schema_digest_for_policy(STATISTICS_POLICY_ROLLING),
        "observed_source_digest": "d" * 64,
        "test_access": False,
        "heldout_access": False,
    }
    validate_rolling_feature_manifest(
        valid_v3_manifest,
        expected_identity="abc",
        expected_observed_source_digest="d" * 64,
    )
    with pytest.raises(ValueError, match="observed_source_digest mismatch"):
        validate_rolling_feature_manifest(
            valid_v3_manifest,
            expected_identity="abc",
            expected_observed_source_digest="e" * 64,
        )

    with pytest.raises(ValueError, match="forbids mlp_score"):
        RollingSlateDataset({"mlp_score": torch.zeros(2, 5), "candidate_indices": torch.zeros(2, 5, dtype=torch.long)})


def _make_synthetic_base_and_dyn(seed: int = 42) -> tuple[UPSCANRRanker, CandidateContextRanker]:
    torch.manual_seed(seed)
    base = UPSCANRRanker(
        text_dim=1024,
        memory_dim=128,
        num_cities=20,
        num_regions=10,
        num_industries=12,
        num_products=50,
        num_categories=10,
        num_subcategories=16,
        num_business_lines=8,
        num_tabular_features=8,
        num_pairwise_features=13,
        seq_feature_dim=9,
        hidden_dim=128,
        num_heads=4,
        statistics_attention=True,
        excluded_statistics=("history_repeat",),
        cached_only=True,
    )
    dyn = CandidateContextRanker(copy.deepcopy(base), "dynamic_features")
    return base, dyn


def _make_synthetic_ranker_inputs(batch_size: int = 3, slate_size: int = 6, seed: int = 11) -> dict:
    torch.manual_seed(seed)
    return {
        "text_embedding": torch.randn(batch_size, 1024),
        "city_id": torch.randint(0, 20, (batch_size,)),
        "region_id": torch.randint(0, 10, (batch_size,)),
        "industry_id": torch.randint(0, 12, (batch_size,)),
        "tabular_features": torch.rand(batch_size, 8),
        "sequence": torch.rand(batch_size, 32, 9),
        "attention_mask": torch.ones(batch_size, 32, dtype=torch.long),
        "seq_product_ids": torch.randint(0, 50, (batch_size, 32)),
        "seq_category_ids": torch.randint(0, 10, (batch_size, 32)),
        "seq_subcategory_ids": torch.randint(0, 16, (batch_size, 32)),
        "seq_business_line_ids": torch.randint(0, 8, (batch_size, 32)),
        "seq_purchase_dates": torch.rand(batch_size, 32) * 0.5,
        "prediction_day": torch.full((batch_size,), 300.0),
        "pairwise_features": torch.rand(batch_size, slate_size, 41),
        "candidate_vectors": torch.randn(batch_size, slate_size, 128),
        "cached_customer_vectors": torch.randn(batch_size, 128),
    }


def test_rolling_candidate_attention_init_parity_permutation_and_masking():
    _, dyn = _make_synthetic_base_and_dyn(42)
    dyn.eval()

    ctrl = RollingCandidateAttentionRanker.from_dynamic_context_ranker(dyn, candidate_attention_enabled=False)
    treat = ctrl.clone_with_candidate_attention(True)
    ctrl.eval()
    treat.eval()

    inputs = _make_synthetic_ranker_inputs(batch_size=3, slate_size=8, seed=13)
    with torch.no_grad():
        old_out = dyn(**inputs)
        ctrl_out = ctrl(**inputs)
        treat_out = treat(**inputs)
        treat_no_attn = treat(**inputs, disable_candidate_attention=True)

    # Exact zero-initialized residual branch preserves 425D pointwise scorer verbatim at init.
    torch.testing.assert_close(ctrl_out.logits, old_out.logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(treat_out.logits, old_out.logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(treat_no_attn.logits, old_out.logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(treat_out.attn_weights, old_out.attn_weights, atol=0.0, rtol=0.0)

    # Strictly forbid Stage 1 mlp_score input and wrong feature widths.
    with pytest.raises(ValueError, match="forbids Stage 1 mlp_score"):
        treat(**inputs, mlp_score=torch.zeros(3, 8))
    bad_pf = dict(inputs, pairwise_features=torch.rand(3, 8, 40))
    with pytest.raises(ValueError, match="Expected pairwise_features shape"):
        treat(**bad_pf)

    # Activate non-zero residual head weights to test candidate mixing, permutation equivariance, and NaN masking.
    with torch.no_grad():
        treat.residual_head[-1].weight.fill_(0.05)
        treat.residual_head[-1].bias.fill_(0.01)
        out_orig = treat(**inputs)

        # 1) Unordered candidate set: reversing candidates permutes logits identically.
        rev_inputs = {
            k: torch.flip(v, dims=[1]) if k in ("pairwise_features", "candidate_vectors") else v
            for k, v in inputs.items()
        }
        out_rev = treat(**rev_inputs)
        torch.testing.assert_close(torch.flip(out_rev.logits, dims=[1]), out_orig.logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(treat.forward_scores(**inputs), out_orig.logits, atol=0.0, rtol=0.0)

        # 2) Candidate-to-candidate attention actually mixes context in treatment, while control ignores peer changes.
        perturbed = copy.deepcopy(inputs)
        perturbed["pairwise_features"][:, -1, 12] += 1.0
        perturbed["candidate_vectors"][:, -1] += 2.0
        ctrl_orig = ctrl(**inputs).logits
        ctrl_pert = ctrl(**perturbed).logits
        treat_pert = treat(**perturbed).logits
        torch.testing.assert_close(ctrl_pert[:, :-1], ctrl_orig[:, :-1], atol=0.0, rtol=0.0)
        assert float((treat_pert[:, :-1] - out_orig.logits[:, :-1]).abs().max().item()) > 1e-4

        # 3) NaN/Inf padded candidates cannot contaminate valid candidates; all-masked row stays finite except -inf logits.
        pad_inputs = copy.deepcopy(inputs)
        pad_inputs["pairwise_features"] = torch.cat(
            [inputs["pairwise_features"], torch.full((3, 2, 41), float("nan"))], dim=1
        )
        pad_inputs["candidate_vectors"] = torch.cat(
            [inputs["candidate_vectors"], torch.full((3, 2, 128), float("inf"))], dim=1
        )
        pad_mask = torch.cat([torch.ones(3, 8, dtype=torch.bool), torch.zeros(3, 2, dtype=torch.bool)], dim=1)
        pad_out = treat(**pad_inputs, candidate_mask=pad_mask)
        torch.testing.assert_close(pad_out.logits[:, :8], out_orig.logits, atol=1e-5, rtol=1e-5)
        assert bool(torch.all(pad_out.logits[:, 8:] == -torch.inf))

        all_false_mask = torch.zeros(3, 10, dtype=torch.bool)
        all_false_out = treat(**pad_inputs, candidate_mask=all_false_mask)
        assert bool(torch.all(all_false_out.logits == -torch.inf))
        assert bool(torch.isfinite(all_false_out.h_attended).all())


def test_rolling_candidate_attention_synthetic_optimizer_step_and_checkpoint_strictness(tmp_path):
    _, dyn = _make_synthetic_base_and_dyn(42)
    ctrl = RollingCandidateAttentionRanker.from_dynamic_context_ranker(dyn, candidate_attention_enabled=False)
    treat = ctrl.clone_with_candidate_attention(True)

    # Control leaves residual branch frozen (grad None); treatment trains residual attention branch cleanly.
    inputs = _make_synthetic_ranker_inputs(batch_size=4, slate_size=10, seed=31)
    labels = torch.zeros(4, 10)
    labels[:, 0] = 1.0
    labels[:, 4] = 1.0
    loss_fn = RankerLoss()

    ctrl.train()
    ctrl.zero_grad(set_to_none=True)
    loss_fn(ctrl(**inputs).logits, labels)["loss"].backward()
    assert all(p.grad is None for p in ctrl.candidate_attention.parameters())

    treat.train()
    opt = torch.optim.AdamW([p for p in treat.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss_fn(treat(**inputs).logits, labels)["loss"].backward()
    assert float(sum(p.grad.abs().sum().item() for p in treat.residual_head.parameters() if p.grad is not None)) > 0.0
    # After step 1 with residual_head[-1].weight previously zero, grads hit residual_head[-1]; step once and step twice
    # to confirm self-attention layers receive nonzero gradients too.
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss_fn(treat(**inputs).logits, labels)["loss"].backward()
    assert float(sum(p.grad.abs().sum().item() for p in treat.candidate_attention.parameters() if p.grad is not None)) > 0.0

    # Strict checkpoint save/load round-trip and tamper rejection.
    treat.eval()
    prov = {
        "reference_id": "repeat-no-month-v1",
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "teacher_score_included": False,
        "upstream_retriever_sha256": "a" * 64,
    }
    schema_dig = rolling_schema_digest()
    ckpt_path = save_rolling_ranker_checkpoint(
        tmp_path / "treat.pt",
        treat,
        epoch=1,
        metrics={"recall@10": 0.78, "ndcg@10": 0.60},
        selection_criterion="val_ndcg_at_10",
        provenance=prov,
        feature_contract_digest=schema_dig,
    )
    reloaded, payload = load_rolling_ranker_checkpoint(
        ckpt_path,
        expected_provenance=prov,
        expected_feature_contract_digest=schema_dig,
        expected_candidate_attention_enabled=True,
    )
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).logits, treat(**inputs).logits, atol=0.0, rtol=0.0)

    # Reject frozen policy, teacher score inclusion, candidate_attention mismatch, feature digest mismatch, bad width.
    with pytest.raises(ValueError, match="candidate_attention_enabled mismatch"):
        load_rolling_ranker_checkpoint(ckpt_path, expected_candidate_attention_enabled=False)

    with pytest.raises(ValueError, match="input_contract mismatch"):
        load_rolling_ranker_checkpoint(ckpt_path, expected_feature_contract_digest="bad_digest")

    bad_policy = copy.deepcopy(payload)
    bad_policy["model_config"]["statistics_policy"] = STATISTICS_POLICY_FROZEN
    torch.save(bad_policy, tmp_path / "bad_policy.pt")
    with pytest.raises(ValueError, match="statistics_policy"):
        load_rolling_ranker_checkpoint(tmp_path / "bad_policy.pt")

    bad_teacher = copy.deepcopy(payload)
    bad_teacher["model_config"]["teacher_score_included"] = True
    torch.save(bad_teacher, tmp_path / "bad_teacher.pt")
    with pytest.raises(ValueError, match="teacher_score_included"):
        load_rolling_ranker_checkpoint(tmp_path / "bad_teacher.pt")

    bad_width = copy.deepcopy(payload)
    bad_width["model_state_dict"]["base_head.0.weight"] = torch.zeros(256, 426)
    torch.save(bad_width, tmp_path / "bad_width.pt")
    with pytest.raises(ValueError, match="base_head.0.weight width mismatch"):
        load_rolling_ranker_checkpoint(tmp_path / "bad_width.pt")


def test_rolling_slate_evaluation_and_paired_intervals(tmp_path):
    n_rows, n_prods, slate_size = 4, 110, 100
    indices = torch.arange(slate_size).unsqueeze(0).expand(n_rows, -1).clone()
    data = {
        "allowed": torch.ones(n_rows, n_prods, dtype=torch.bool),
        "target": torch.zeros(n_rows, n_prods, dtype=torch.bool),
        "new_target": torch.zeros(n_rows, n_prods, dtype=torch.bool),
        "source": {
            "customer_ids": [f"C{i % 2}" for i in range(n_rows)],
            "positive_pids": [["P0", "P20"] for _ in range(n_rows)],
            "new_positive_pids": [["P20"] for _ in range(n_rows)],
            "attention_mask": torch.ones(n_rows, 32, dtype=torch.long),
        },
    }
    data["target"][:, 0] = True
    data["target"][:, 20] = True
    data["new_target"][:, 20] = True

    ctrl_scores = torch.linspace(1.0, 0.0, slate_size).unsqueeze(0).expand(n_rows, -1).clone()
    trt_scores = ctrl_scores.clone()
    trt_scores[:, 20] = 2.0

    m_ctrl = evaluate_rolling_slate_metrics(ctrl_scores, data, indices, tmp_path / "ctrl.npz")
    m_trt = evaluate_rolling_slate_metrics(trt_scores, data, indices, tmp_path / "trt.npz")
    assert m_ctrl["new_hits@10"] == 0
    assert m_trt["new_hits@10"] == n_rows
    ci = paired_rolling_intervals(tmp_path / "ctrl.npz", tmp_path / "trt.npz", repeats=100, seed=42)
    assert ci["new_recall@10_delta"]["point_delta"] == pytest.approx(1.0)


def test_nat_and_missing_date_rejection():
    purchases, catalogue, companies = _synthetic_market_tables()
    product_ids = ["SKU_HOT_A", "SKU_VAL_B", "SKU_COMP_C"]

    bad_purchases = purchases.copy()
    bad_purchases.loc[0, "purchase_date"] = pd.NaT
    with pytest.raises(ValueError, match="Development purchases only"):
        DynamicFeatureBuilder(
            bad_purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING
        )

    builder = DynamicFeatureBuilder(
        purchases, catalogue, companies, product_ids, statistics_policy=STATISTICS_POLICY_ROLLING, allow_non_monotonic=True
    )
    adapter = RollingDynamicFeatureAdapter(purchases, catalogue, companies, product_ids)
    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    selector = FeatureAblatedStage1Selector(spec)
    u = torch.randn(128)
    v = torch.randn(len(product_ids), 128)
    stats6 = torch.rand(6)

    for bad_date in (pd.NaT, None, "NaT", "2026-02-19"):
        with pytest.raises(ValueError, match="Development predictions only"):
            builder.global_features(bad_date)
        with pytest.raises(ValueError, match="Development predictions only"):
            builder.build("CUST_TARGET", bad_date)
        with pytest.raises(ValueError, match="Development predictions only"):
            adapter.build_raw("CUST_TARGET", bad_date)
        with pytest.raises(ValueError, match="Development predictions only"):
            adapter.build_quantized("CUST_TARGET", bad_date)
        with pytest.raises(ValueError, match="Development predictions only"):
            adapter.score_active_selector(
                selector, u=u, v=v, customer_stats_6=stats6, customer_id="CUST_TARGET", as_of_date=bad_date
            )


def test_source_digest_invalidation_and_score_active_selector_validation(tmp_path, monkeypatch):
    purchases, catalogue, companies = _synthetic_market_tables()
    product_ids = ["SKU_HOT_A", "SKU_VAL_B", "SKU_COMP_C"]

    d_base = compute_rolling_source_digest(purchases, catalogue, companies)
    # Order-invariant to shuffling rows
    d_shuf = compute_rolling_source_digest(
        purchases.sample(frac=1.0, random_state=7),
        catalogue.sample(frac=1.0, random_state=7),
        companies.sample(frac=1.0, random_state=7),
    )
    assert d_base["observed_source_digest"] == d_shuf["observed_source_digest"]

    # Rejects held-out purchase dates (>= 2026-02-19)
    heldout_p = pd.concat(
        [
            purchases,
            pd.DataFrame(
                [
                    {
                        "customer_id": "CUST_TARGET",
                        "product_id": "SKU_VAL_B",
                        "purchase_date": pd.Timestamp("2026-02-19"),
                        "quantity": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="refuses held-out or invalid purchase dates"):
        compute_rolling_source_digest(heldout_p, catalogue, companies)

    # Changes when a dev purchase, catalogue field, or company field changes
    p_mod = purchases.copy()
    p_mod.loc[0, "quantity"] = 999.0
    assert compute_rolling_source_digest(p_mod, catalogue, companies)["observed_source_digest"] != d_base["observed_source_digest"]

    c_mod = catalogue.copy()
    c_mod.loc[0, "product_category"] = "CAT_NEW"
    assert compute_rolling_source_digest(purchases, c_mod, companies)["observed_source_digest"] != d_base["observed_source_digest"]

    co_mod = companies.copy()
    co_mod.loc[0, "industry_branch"] = "IND_NEW"
    assert compute_rolling_source_digest(purchases, catalogue, co_mod)["observed_source_digest"] != d_base["observed_source_digest"]

    # Manifest validation detects source change (simulating prepare_or_load_rolling_feature_bundle invalidation)
    manifest = {
        "version": FEATURE_CACHE_VERSION,
        "namespace": ROLLING_CACHE_NAMESPACE,
        "identity": "id1",
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "feature_contract": feature_contract_for_policy(STATISTICS_POLICY_ROLLING),
        "schema_digest": feature_schema_digest_for_policy(STATISTICS_POLICY_ROLLING),
        "observed_source_digest": d_base["observed_source_digest"],
        "test_access": False,
        "heldout_access": False,
    }
    validate_rolling_feature_manifest(
        manifest,
        expected_identity="id1",
        expected_observed_source_digest=d_base["observed_source_digest"],
    )
    for changed_digest in (
        compute_rolling_source_digest(p_mod, catalogue, companies)["observed_source_digest"],
        compute_rolling_source_digest(purchases, c_mod, companies)["observed_source_digest"],
        compute_rolling_source_digest(purchases, catalogue, co_mod)["observed_source_digest"],
    ):
        with pytest.raises(ValueError, match="observed_source_digest mismatch"):
            validate_rolling_feature_manifest(
                manifest,
                expected_identity="id1",
                expected_observed_source_digest=changed_digest,
            )

    # End-to-end synthetic cache build + invalidation on dev source edit (without touching data/ or checkpoints/)
    import recommender.experiments.rolling_candidate_v2.data as rdata

    monkeypatch.setattr(rdata, "EXPECTED_PRODUCT_UNIVERSE", 3)
    monkeypatch.setattr(rdata, "MIN_ELIGIBLE_SKUS", 1)
    monkeypatch.setattr(rdata, "EXPECTED_TRAIN_ROWS", 1)
    monkeypatch.setattr(rdata, "EXPECTED_VAL_ROWS", 1)
    monkeypatch.setattr(rdata, "EXPECTED_VAL_TOTAL_TARGETS", 1)
    monkeypatch.setattr(rdata, "EXPECTED_VAL_NEW_TARGETS", 1)
    monkeypatch.setattr(rdata, "EXPECTED_RETRIEVER_SHA256", "r" * 64)
    monkeypatch.setattr(rdata, "EXPECTED_SELECTOR_SHA256", "s" * 64)

    e1_dir = tmp_path / "e1_e5"
    e1_dir.mkdir()
    for name in ("train.pt", "val.pt", "manifest.pt"):
        (e1_dir / name).write_bytes(b"stub")
    monkeypatch.setattr(rdata, "E1_E5_RETRIEVAL_CACHE_DIR", e1_dir)

    up_dir = tmp_path / "upstream"
    sel_dir = tmp_path / "selector"
    up_dir.mkdir()
    sel_dir.mkdir()
    monkeypatch.setattr(rdata, "UPSTREAM_STAGE1_RUN_DIR", up_dir)
    monkeypatch.setattr(rdata, "SELECTOR_SOURCE_RUN_DIR", sel_dir)

    spec = build_feature_ablation_spec(
        REPEAT_RESTORED_OMITTED_CONTEXT_FEATURES,
        name=REPEAT_RESTORED_VARIANT_NAME,
        recipe=REPEAT_RESTORED_RECIPE_NAME,
        expected_omitted_indices=REPEAT_RESTORED_EXPECTED_OMITTED_SOURCE_INDICES,
    )
    selector = FeatureAblatedStage1Selector(spec)
    contract = selector.input_contract()
    v_mat = torch.randn(3, 128)
    for split, day_int in (("train", 100), ("val", 300)):
        torch.save(
            {
                "model_digest": "r" * 64,
                "data_identity": "data-id-1",
                "enrichment_identity": "enr-id-1",
                "customer_vectors": torch.randn(1, 1, 128),
                "product_vectors": v_mat,
                "scores": torch.zeros(1, 3),
            },
            up_dir / f"{split}_purchase_cache.pt",
        )
        torch.save(
            {
                "selector_digest": "s" * 64,
                "upstream_digest": "r" * 64,
                "data_identity": "data-id-1",
                "input_contract": contract,
                "scores": torch.zeros(1, 3),
            },
            sel_dir / f"{split}_scores.pt",
        )

    def _fake_prepare_retrieval_data():
        tr_ret = {
            "allowed": torch.ones(1, 3, dtype=torch.bool),
            "unseen": torch.ones(1, 3, dtype=torch.bool),
            "target": torch.tensor([[True, False, False]]),
            "new_target": torch.tensor([[False, False, False]]),
        }
        va_ret = {
            "allowed": torch.ones(1, 3, dtype=torch.bool),
            "unseen": torch.ones(1, 3, dtype=torch.bool),
            "target": torch.tensor([[False, True, False]]),
            "new_target": torch.tensor([[False, True, False]]),
        }
        tr_src = {
            "customer_ids": ["CUST_TARGET"],
            "prediction_days": torch.tensor([100]),
            "transition_keys": [("CUST_TARGET", 100, ("SKU_HOT_A",))],
            "positive_pids": [["SKU_HOT_A"]],
            "new_positive_pids": [[]],
            "retriever_tabular_features": torch.rand(1, 6),
        }
        va_src = {
            "customer_ids": ["CUST_TARGET"],
            "prediction_days": torch.tensor([292]),  # 2025-10-20
            "transition_keys": [("CUST_TARGET", 292, ("SKU_VAL_B",))],
            "positive_pids": [["SKU_VAL_B"]],
            "new_positive_pids": [["SKU_VAL_B"]],
            "retriever_tabular_features": torch.rand(1, 6),
        }
        info = {"identity": "data-id-1", "product_ids": product_ids}
        return tr_ret, va_ret, info, tr_src, va_src

    monkeypatch.setattr(rdata, "prepare_retrieval_data", _fake_prepare_retrieval_data)
    monkeypatch.setattr(
        rdata,
        "verify_active_reference",
        lambda load_selector=True: {
            "manifest": {"source_identities": {"enrichment_identity": "enr-id-1"}},
            "selector": selector,
            "selector_input_contract": contract,
            "hashes": {
                "best_retriever": "r" * 64,
                "best_selector": "s" * 64,
                "train_purchase_cache": "p" * 64,
                "val_purchase_cache": "q" * 64,
                "train_scores": "t" * 64,
                "val_scores": "u" * 64,
            },
        },
    )

    state = {"purchases": purchases.copy(), "catalogue": catalogue.copy(), "companies": companies.copy()}
    monkeypatch.setattr(rdata, "load_development_purchases", lambda _: state["purchases"].copy())
    monkeypatch.setattr(rdata, "load_product_catalogue", lambda _: state["catalogue"].copy())
    monkeypatch.setattr(rdata, "load_companies", lambda *a, **kw: state["companies"].copy())

    cache_dir = tmp_path / "cache_v3"
    rdata.prepare_or_load_rolling_feature_bundle(
        cache_root=cache_dir,
        build=True,
        purchases_path="ignored.csv",
        catalogue_path=tmp_path / "cat.csv",
        companies_path=tmp_path / "comp.csv",
    )
    # Reloading with unchanged sources succeeds
    rdata.prepare_or_load_rolling_feature_bundle(
        cache_root=cache_dir,
        build=False,
        purchases_path="ignored.csv",
        catalogue_path=tmp_path / "cat.csv",
        companies_path=tmp_path / "comp.csv",
    )
    # Mutating observed dev purchase invalidates existing cache immediately
    state["purchases"] = p_mod.copy()
    with pytest.raises(ValueError, match="observed_source_digest mismatch"):
        rdata.prepare_or_load_rolling_feature_bundle(
            cache_root=cache_dir,
            build=True,
            purchases_path="ignored.csv",
            catalogue_path=tmp_path / "cat.csv",
            companies_path=tmp_path / "comp.csv",
        )

    # Restore and also test score_active_selector input validation
    adapter = RollingDynamicFeatureAdapter(purchases, catalogue, companies, product_ids)
    u = torch.randn(128)
    v = torch.randn(len(product_ids), 128)
    stats6 = torch.rand(6)
    with pytest.raises(TypeError, match="FeatureAblatedStage1Selector"):
        adapter.score_active_selector(
            torch.nn.Linear(4, 4), u=u, v=v, customer_stats_6=stats6, customer_id="CUST_TARGET", as_of_date="2025-10-20"
        )
    with pytest.raises(ValueError, match="v_product_ids"):
        adapter.score_active_selector(
            selector,
            u=u,
            v=v,
            customer_stats_6=stats6,
            customer_id="CUST_TARGET",
            as_of_date="2025-10-20",
            v_product_ids=list(reversed(product_ids)),
        )
    with pytest.raises(ValueError, match="Duplicate candidate_ids"):
        adapter.score_active_selector(
            selector,
            u=u,
            v=v,
            customer_stats_6=stats6,
            customer_id="CUST_TARGET",
            as_of_date="2025-10-20",
            candidate_ids=["SKU_HOT_A", "SKU_HOT_A"],
        )
    with pytest.raises(ValueError, match="Unknown candidate_ids"):
        adapter.score_active_selector(
            selector,
            u=u,
            v=v,
            customer_stats_6=stats6,
            customer_id="CUST_TARGET",
            as_of_date="2025-10-20",
            candidate_ids=["UNKNOWN_SKU"],
        )
    with pytest.raises(ValueError, match="allowed_mask"):
        adapter.score_active_selector(
            selector,
            u=u,
            v=v,
            customer_stats_6=stats6,
            customer_id="CUST_TARGET",
            as_of_date="2025-10-20",
            allowed_mask=torch.tensor([True, False]),
        )
    bad_u = u.clone()
    bad_u[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        adapter.score_active_selector(
            selector, u=bad_u, v=v, customer_stats_6=stats6, customer_id="CUST_TARGET", as_of_date="2025-10-20"
        )


@pytest.mark.local_artifact
def test_saved_rolling_v2_benchmark_and_cache_provenance():
    bench_path = EVIDENCE_ROOT / "benchmark.json"
    summary_path = EVIDENCE_ROOT / "summary.json"
    report_path = EVIDENCE_ROOT / "REPORT.md"
    if not bench_path.is_file():
        pytest.skip("rolling_candidate_v2 benchmark artifact not present.")

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    assert bench["training_executed"] is False
    assert bench["heldout_access"] is False
    assert bench["promoted"] is False
    assert bench["statistics_policy"] == STATISTICS_POLICY_ROLLING
    assert bench["universe_counts"]["products"] == 841
    assert bench["universe_counts"]["dev_unique_customers"] == 2571
    assert "cpu" in bench["devices"]

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["training_executed"] is False
        assert summary["heldout_access"] is False
        assert summary["promoted"] is False
        assert summary["preserved_hashes_before"] == summary["preserved_hashes_after"]
    if report_path.is_file():
        report_txt = report_path.read_text(encoding="utf-8")
        assert "rolling_observed_v2" in report_txt
        assert "shape_only" in report_txt.lower() or "untrained" in report_txt.lower()

    assert LEGACY_V2_CACHE_ROOT.is_dir()
    feat_manifest_path = CACHE_ROOT / "features_manifest.json"
    scores_manifest_path = CACHE_ROOT / "selector_scores_manifest.json"
    slate_manifest_path = CACHE_ROOT / "slates_manifest.json"
    if feat_manifest_path.is_file():
        fm = json.loads(feat_manifest_path.read_text(encoding="utf-8"))
        assert fm["version"] == FEATURE_CACHE_VERSION
        assert fm["namespace"] == ROLLING_CACHE_NAMESPACE
        assert fm["statistics_policy"] == STATISTICS_POLICY_ROLLING
        assert isinstance(fm.get("observed_source_digest"), str) and len(fm["observed_source_digest"]) == 64
        assert fm["heldout_access"] is False
        assert fm["test_access"] is False
    if scores_manifest_path.is_file():
        scm = json.loads(scores_manifest_path.read_text(encoding="utf-8"))
        assert scm["statistics_policy"] == STATISTICS_POLICY_ROLLING
        assert scm["training_executed"] is False
        assert scm["heldout_access"] is False
        assert scm["train_parity_max_abs_diff_allowed"] < 1e-4
    if slate_manifest_path.is_file():
        sm = json.loads(slate_manifest_path.read_text(encoding="utf-8"))
        assert sm["cache_version"] == SLATE_CACHE_VERSION
        assert sm["namespace"] == ROLLING_CACHE_NAMESPACE
        assert sm["candidate_source"] == CANDIDATE_SOURCE_ROLLING_SELECTOR
        assert sm["statistics_policy"] == STATISTICS_POLICY_ROLLING
        assert sm["teacher_score_included"] is False
        assert sm["heldout_access"] is False
        assert isinstance(sm.get("observed_source_digest"), str) and len(sm["observed_source_digest"]) == 64

    replay_path = EVIDENCE_ROOT / "rolling_selector_replay.json"
    if replay_path.is_file():
        rep = json.loads(replay_path.read_text(encoding="utf-8"))
        assert rep["training_executed"] is False
        assert rep["heldout_access"] is False
        assert rep["promoted"] is False
        assert rep["same_weights_changed_validation_inputs"] is True
        assert rep["preserved_hashes_before"] == rep["preserved_hashes_after"]
