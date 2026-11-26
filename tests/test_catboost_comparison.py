"""Current CatBoost data must be causal, correctly ordered, and target-independent."""

import numpy as np
import pandas as pd
import pytest

from recommender.experiments.catboost_comparison.features import (
    LEGACY_FEATURES, LEGACY_NUMERIC, build_legacy_numeric, categorical_frame, legacy_frame,
)


def fixture():
    catalogue = pd.DataFrame({'product_id': ['a', 'b'], 'product_category': ['cat_a', 'cat_b'], 'business_line': ['line_a', 'line_b']})
    rows = pd.DataFrame({
        'customer_id': ['c', 'c', 'c', 'other', 'other'], 'product_id': ['a', 'a', 'a', 'a', 'b'],
        'purchase_date': pd.to_datetime(['2025-08-01', '2025-08-11', '2025-08-31', '2025-08-15', '2025-08-20']),
        'quantity': [1., 3., 2., 4., 1.],
    })
    return rows, catalogue


def test_legacy_feature_values_and_nullable_cadence():
    rows, cat = fixture()
    idx = np.array([[0, 1]])
    out = build_legacy_numeric(rows, cat, ['a', 'b'], ['c'], ['2025-09-01'], idx)
    np.testing.assert_allclose(out[0, 0], [3, 6, 2, 1, 15, 5, 14, 3, 1, 3, 1, 4, 2, 3])
    assert np.isnan(out[0, 1, 4:7]).all()
    np.testing.assert_equal(out[0, 1, :4], 0)
    np.testing.assert_equal(out[0, 1, 11:], [1, 1, 1])
    frame = legacy_frame(out, categorical_frame(['a', 'b'], cat, idx))
    assert tuple(frame.columns) == LEGACY_FEATURES
    assert len(LEGACY_NUMERIC) == 14


def test_same_day_and_future_purchases_do_not_change_earlier_features():
    rows, cat = fixture()
    args = (cat, ['a', 'b'], ['c', 'c'], ['2025-08-31', '2025-09-01'], np.array([[0, 1], [0, 1]]))
    reference = build_legacy_numeric(rows, *args)
    changed = pd.concat([rows, pd.DataFrame({
        'customer_id': ['c', 'other'], 'product_id': ['b', 'a'],
        'purchase_date': pd.to_datetime(['2025-09-01', '2025-09-05']), 'quantity': [1e6, 1e6],
    })], ignore_index=True)
    np.testing.assert_equal(reference, build_legacy_numeric(changed, *args))
    assert reference[0, 0, 0] == 2  # 2025-08-31 purchase must not be visible that day
    assert reference[1, 0, 0] == 3


def test_shuffled_queries_and_candidate_permutations_keep_semantics():
    rows, cat = fixture()
    dates = ['2025-09-01', '2025-08-10']
    idx = np.array([[0, 1], [1, 0]])
    actual = build_legacy_numeric(rows, cat, ['a', 'b'], ['c', 'c'], dates, idx)
    permuted = build_legacy_numeric(rows, cat, ['a', 'b'], ['c', 'c'], dates[::-1], idx[::-1, ::-1])
    np.testing.assert_equal(actual, permuted[::-1, ::-1])


def test_heldout_and_missing_query_dates_are_rejected():
    rows, cat = fixture()
    for dates in (['2026-02-19'], [pd.NaT]):
        with pytest.raises(ValueError, match='development'):
            build_legacy_numeric(rows, cat, ['a', 'b'], ['c'], dates, np.array([[0]]))


def test_catboost_categorical_and_nan_contract_reloads(tmp_path):
    cb = pytest.importorskip('catboost')
    rows, cat = fixture()
    idx = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    features = build_legacy_numeric(rows, cat, ['a', 'b'], ['c'] * 4,
                                    ['2025-09-01', '2025-08-10', '2025-08-12', '2025-08-21'], idx)
    frame = legacy_frame(features, categorical_frame(['a', 'b'], cat, idx))
    pool = cb.Pool(frame, label=[1, 0, 0, 1, 1, 0, 0, 1], cat_features=list(LEGACY_FEATURES[:3]))
    m = cb.CatBoostClassifier(iterations=4, depth=2, verbose=False, thread_count=1, allow_writing_files=False)
    m.fit(pool)
    path = tmp_path / 'test.cbm'; m.save_model(str(path))
    loaded = cb.CatBoostClassifier(); loaded.load_model(str(path))
    np.testing.assert_equal(m.predict(pool, prediction_type='RawFormulaVal'), loaded.predict(pool, prediction_type='RawFormulaVal'))
