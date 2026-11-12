"""Explicit ordered, versioned feature lifecycle contract."""
import numpy as np
from recommender.candidates import PAIRWISE_FEATURE_NAMES
from recommender.artifacts import json_digest

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


def contract():
    return {'version': 1, 'names': list(NAMES), 'dtype_on_disk': 'float16', 'compute_dtype': 'float32',
            'count_scale': 'log1p(x)/log1p(1000), clipped [0,1]', 'age_scale': 'days/365, clipped [0,1]',
            'ratio_scale': 'days/max(median_basket_gap,1)/4, clipped [0,1]',
            'lift_scale': 'log_lift/5 clipped [-1,1]', 'missing': 'never flags; missing metadata yields zero',
            'statistics_policy': 'strictly earlier train; frozen train-cutoff validation globals',
            'metadata_policy': 'snapshot_unverified', 'customer_embedding_policy': 'last-purchase snapshot only',
            'product_embedding_policy': 'demand disabled, explicit model/catalogue version refresh'}


def schema_digest():
    return json_digest(contract())


def quantize_features(values):
    """Serving reproduces the float16 cache boundary before float32 computation."""
    return np.asarray(values, dtype=np.float16).astype(np.float32)
