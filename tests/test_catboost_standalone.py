"""Standalone trees must not use neural scores, retrieval, or invalid products."""

import numpy as np
import pandas as pd
import pytest

from recommender.experiments.catboost_comparison.standalone import (
    FEATURE_NAMES, sample_training_pairs, statistical_frame,
)


def test_uniform_training_sample_preserves_positives_and_full_pool_weight():
    allowed = np.ones((3, 20), dtype=bool)
    allowed[0, :3] = False
    allowed[1, 5:] = False
    target = np.zeros_like(allowed)
    target[0, [0, 4, 7]] = True  # Invalid target zero must never be inserted.
    target[1, 2] = True
    rows, columns, weights = sample_training_pairs(allowed, target, negatives=4)
    assert allowed[rows, columns].all()
    assert len(set(zip(rows, columns))) == len(rows)
    assert int(target[rows, columns].sum()) == int((allowed & target).sum())
    for row in range(len(allowed)):
        np.testing.assert_allclose(weights[rows == row].sum(), allowed[row].sum(), rtol=1e-6)
    again = sample_training_pairs(allowed, target, negatives=4)
    for first, second in zip((rows, columns, weights), again):
        np.testing.assert_array_equal(first, second)


def test_standalone_feature_contract_contains_only_categories_and_statistics():
    catalogue = pd.DataFrame({'product_id': ['a', 'b'], 'product_category': ['c', 'd'],
                              'business_line': ['line', 'line']})
    features = np.arange(2 * 2 * 41, dtype=np.float32).reshape(2, 2, 41)
    profile = np.arange(16, dtype=np.float32).reshape(2, 8)
    rows, columns = np.array([1, 0, 1]), np.array([0, 1, 1])
    frame = statistical_frame(features, profile, ['a', 'b'], catalogue, rows, columns)
    assert tuple(frame.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 52
    assert not any('dot' in f or 'embedding' in f or 'mlp' in f for f in FEATURE_NAMES)
    np.testing.assert_array_equal(frame.iloc[:, 3:44].to_numpy(), features[rows, columns])
    np.testing.assert_array_equal(frame.iloc[:, 44:].to_numpy(), profile[rows])
    assert frame.product_id.tolist() == ['a', 'b', 'b']
    with pytest.raises(ValueError, match='dimensions'):
        statistical_frame(features, profile[:, :6], ['a', 'b'], catalogue, rows, columns)
