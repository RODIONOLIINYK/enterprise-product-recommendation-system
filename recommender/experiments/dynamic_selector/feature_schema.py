"""Explicit ordered, versioned feature lifecycle contract."""
import numpy as np
from recommender.candidates import (
    PAIRWISE_FEATURE_NAMES,
    STATISTICS_POLICY_FROZEN,
    STATISTICS_POLICY_ROLLING,
    VALID_STATISTICS_POLICIES,
)
from recommender.artifacts import json_digest

DEFAULT_NEW_STATISTICS_POLICY = STATISTICS_POLICY_ROLLING

DYNAMIC_NAMES = (
    'customer_age', 'customer_overdue_ratio', 'customer_30d_activity', 'customer_90d_activity',
    'category_age', 'category_overdue_ratio', 'category_never_seen',
    'family_age', 'family_overdue_ratio', 'family_never_seen', 'family_history_share',
    'sku_orders_7d', 'sku_orders_30d', 'sku_orders_90d', 'sku_buyers_30d',
    'sku_growth_30d', 'category_relative_share', 'global_sale_age', 'month_sin', 'month_cos',
)
RELATION_NAMES = ('sku_last_lift', 'sku_recent3_lift', 'category_last_lift', 'family_last_lift',
                  'line_last_lift', 'sku_support_confidence', 'hierarchical_backoff', 'same_basket_lift')
NAMES = (*PAIRWISE_FEATURE_NAMES, *DYNAMIC_NAMES, *RELATION_NAMES)
STATIC_COLUMNS = (0, 1, 2, 5, 6, 7, 10, 11, 12, NAMES.index('family_history_share'))
DYNAMIC_COLUMNS = (*range(13), *range(13, 13 + len(DYNAMIC_NAMES)))
RELATION_COLUMNS = tuple(range(13 + len(DYNAMIC_NAMES), len(NAMES)))


def contract(statistics_policy: str = STATISTICS_POLICY_FROZEN):
    if statistics_policy not in VALID_STATISTICS_POLICIES:
        raise ValueError(
            f'Unknown statistics_policy {statistics_policy!r}; expected one of {VALID_STATISTICS_POLICIES}.'
        )
    if statistics_policy == STATISTICS_POLICY_FROZEN:
        return {'version': 1, 'names': list(NAMES), 'dtype_on_disk': 'float16', 'compute_dtype': 'float32',
                'count_scale': 'log1p(x)/log1p(1000), clipped [0,1]', 'age_scale': 'days/365, clipped [0,1]',
                'ratio_scale': 'days/max(median_basket_gap,1)/4, clipped [0,1]',
                'lift_scale': 'log_lift/5 clipped [-1,1]', 'missing': 'never flags; missing metadata yields zero',
                'statistics_policy': 'strictly earlier train; frozen train-cutoff validation globals',
                'metadata_policy': 'snapshot_unverified', 'customer_embedding_policy': 'last-purchase snapshot only',
                'product_embedding_policy': 'demand disabled, explicit model/catalogue version refresh'}
    return {
        'version': 2,
        'names': list(NAMES),
        'dtype_on_disk': 'float16',
        'compute_dtype': 'float32',
        'count_scale': 'log1p(x)/log1p(1000), clipped [0,1]',
        'age_scale': 'days/365, clipped [0,1]',
        'ratio_scale': 'days/max(median_basket_gap,1)/4, clipped [0,1]',
        'lift_scale': 'log_lift/5 clipped [-1,1]',
        'missing': 'never flags; missing metadata yields zero',
        'statistics_policy_id': STATISTICS_POLICY_ROLLING,
        'statistics_policy': (
            'strictly earlier observed purchases (< as_of_date) across train and prior validation months; '
            'fixed train-fitted popularity normalization; never same-date/future or held-out test'
        ),
        'relationship_tables_policy': 'strictly_prior_observed_baskets_rolling',
        'metadata_policy': 'snapshot_unverified',
        'customer_embedding_policy': 'last-purchase snapshot only',
        'product_embedding_policy': 'demand disabled, explicit model/catalogue version refresh',
    }


def rolling_contract():
    return contract(STATISTICS_POLICY_ROLLING)


def schema_digest(statistics_policy: str = STATISTICS_POLICY_FROZEN):
    return json_digest(contract(statistics_policy))


def rolling_schema_digest():
    return schema_digest(STATISTICS_POLICY_ROLLING)


def quantize_features(values):
    """Serving reproduces the float16 cache boundary before float32 computation."""
    return np.asarray(values, dtype=np.float16).astype(np.float32)
