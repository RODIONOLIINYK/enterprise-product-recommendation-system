import copy
import numpy as np
import pandas as pd
import pytest
import torch
from recommender.experimental_signals import SignalRanker, TransitionStatistics
from recommender.ranker import UPSCANRRanker


def ranker_inputs():
    return dict(text_embedding=torch.randn(2, 16), city_id=torch.zeros(2, dtype=torch.long),
                region_id=torch.zeros(2, dtype=torch.long), industry_id=torch.zeros(2, dtype=torch.long),
                tabular_features=torch.rand(2, 8), sequence=torch.rand(2, 4, 9), attention_mask=torch.ones(2, 4),
                seq_purchase_dates=torch.zeros(2, 4), prediction_day=torch.ones(2),
                candidate_vectors=torch.randn(2, 3, 16), pairwise_features=torch.zeros(2, 3, 13))


@pytest.mark.parametrize("mode", ["unseen", "transition_residual", "transition_mlp"])
def test_zero_initialization_and_signal_gradients(mode):
    torch.manual_seed(7)
    base = UPSCANRRanker(text_dim=16, hidden_dim=16, dropout=0., statistics_attention=True)
    base.eval()
    inputs = ranker_inputs()
    inputs["pairwise_features"][:, 1, 0] = 1
    expected = base(**inputs).logits
    wrapped = SignalRanker(copy.deepcopy(base), mode).eval()
    if mode.startswith("transition"):
        inputs["pairwise_features"] = torch.cat([inputs["pairwise_features"], torch.ones(2, 3, 1)], -1)
    output = wrapped(**inputs).logits
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    parameter = wrapped.head[0].weight if mode == "transition_mlp" else wrapped.residual_weight
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0
    if mode == "unseen":
        with torch.no_grad():
            wrapped.residual_weight.fill_(2.)
        changed = wrapped(**inputs).logits - expected
        torch.testing.assert_close(changed, 2 * (inputs["pairwise_features"][..., 0] == 0).float())
    restored = SignalRanker(copy.deepcopy(base), mode)
    restored.load_state_dict(wrapped.state_dict(), strict=True)
    restored.eval()
    torch.testing.assert_close(wrapped(**inputs).logits, restored(**inputs).logits)


def purchases():
    return pd.DataFrame({"customer_id": ["x", "x", "y", "y", "z", "z"],
                         "product_id": ["A", "B", "A", "C", "A", "B"],
                         "purchase_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-03-01", "2025-01-01", "2025-10-01"])})


def test_transition_counts_strict_destination_date_and_training_cutoff():
    frame = purchases()
    stats = TransitionStatistics(frame, alpha=2.)
    np.testing.assert_array_equal(stats.score("x", "2025-02-01", ["B", "C"]), [0, 0])
    stats.score("x", "2025-02-02", ["B", "C"])
    assert stats.total == 1
    assert stats.a[stats.index["A"]] == 1
    assert stats.ab[stats.index["A"], stats.index["B"]] == 1
    stats.score("x", "2025-10-02", ["B", "C"])
    assert stats.total == 2  # Oct 1 destination never enters training counts
    with pytest.raises(ValueError, match="chronologically"):
        stats.advance("2025-01-01")


def test_future_baskets_and_global_validation_outcomes_do_not_change_prior_score():
    frame = purchases()
    later = frame.copy()
    later.loc[3, "product_id"] = "B"
    for date in ("2025-02-01", "2025-02-02"):
        np.testing.assert_array_equal(TransitionStatistics(frame).score("x", date, ["B", "C"]),
                                      TransitionStatistics(later).score("x", date, ["B", "C"]))
    changed = frame.copy()
    changed.loc[5, "product_id"] = "C"
    np.testing.assert_array_equal(TransitionStatistics(frame).score("x", "2026-01-01", ["B", "C"]),
                                  TransitionStatistics(changed).score("x", "2026-01-01", ["B", "C"]))


def test_smoothing_matches_formula_and_cold_customers():
    stats = TransitionStatistics(purchases(), alpha=2.)
    actual = stats.score("z", "2025-02-02", ["B", "C"])
    probability = np.array([.75, .25])
    expected = np.log((np.array([1, 0]) + 2 * probability) / 3 / probability)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert np.isfinite(stats.score("x", "2025-02-02", ["unknown"])).all()
    assert (stats.score("cold", "2025-02-02", ["B"]) == 0).all()
    invalid = purchases()
    invalid.loc[0, "purchase_date"] = pd.Timestamp("2026-02-19")
    with pytest.raises(ValueError, match="Development"):
        TransitionStatistics(invalid)
