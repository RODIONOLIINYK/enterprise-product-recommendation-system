"""Padding must not change the ranking objective or contaminate its gradients."""

import pytest
import torch
import torch.nn.functional as F

from recommender.ranker import RankerLoss


@pytest.mark.parametrize('new_weight', [1.0, 3.0])
def test_masked_loss_matches_physically_removed_candidates(new_weight):
    logits = torch.tensor([[1.0, -float('inf'), 0.0, float('nan')]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0, float('nan')]])
    mask = torch.tensor([[True, False, True, False]])
    unseen = torch.tensor([[True, False, False, False]])
    criterion = RankerLoss(new_positive_weight=new_weight)
    actual = criterion(logits, labels, unseen_mask=unseen, candidate_mask=mask)
    trimmed = logits.detach()[:, [0, 2]].clone().requires_grad_(True)
    expected = criterion(trimmed, labels[:, [0, 2]], unseen_mask=unseen[:, [0, 2]])
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    actual['loss'].backward()
    expected['loss'].backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, [0, 2]], trimmed.grad)
    assert torch.equal(logits.grad[:, [1, 3]], torch.zeros(1, 2))


def test_all_masked_rows_are_differentiable_zero_and_do_not_dilute_other_rows():
    criterion = RankerLoss()
    z = torch.tensor([[1., 0., -float('inf')], [float('nan'), -float('inf'), 5.]], requires_grad=True)
    y = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    actual = criterion(z, y, candidate_mask=mask)
    expected = criterion(z[:1, :2], y[:1, :2])
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    actual['loss'].backward()
    assert torch.isfinite(z.grad).all()
    assert torch.equal(z.grad[1], torch.zeros(3))
    empty = torch.full((2, 3), -float('inf'), requires_grad=True)
    result = criterion(empty, torch.zeros_like(empty), candidate_mask=torch.zeros_like(empty, dtype=torch.bool))
    assert all(t.item() == 0. for t in result.values())
    result['loss'].backward()
    assert torch.equal(empty.grad, torch.zeros_like(empty))


def test_masked_rows_without_positives_have_only_valid_negative_bce():
    logits = torch.tensor([[.3, -float('inf'), -.2]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    result = RankerLoss()(logits, torch.zeros_like(logits), candidate_mask=mask)
    torch.testing.assert_close(result['loss'], F.binary_cross_entropy_with_logits(logits[:, [0, 2]], torch.zeros(1, 2)))
    assert result['list_loss'].item() == 0.
    result['loss'].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0.


def test_variable_lengths_use_valid_entry_bce_and_positive_row_listwise_reductions():
    logits = torch.tensor([[1., .3, -float('inf')], [-.2, -float('inf'), -float('inf')]], requires_grad=True)
    labels = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    result = RankerLoss()(logits, labels, candidate_mask=mask)
    expected_bce = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=torch.tensor([10.]))
    expected_lw = -F.log_softmax(logits[0, :2], dim=-1)[0] / 2
    torch.testing.assert_close(result['bce_loss'], expected_bce)
    torch.testing.assert_close(result['list_loss'], expected_lw)


def test_all_valid_mask_preserves_legacy_values_and_gradients_exactly():
    g = torch.Generator().manual_seed(42)
    z = torch.randn(4, 7, generator=g, requires_grad=True)
    z_ref = z.detach().clone().requires_grad_(True)
    y = torch.randint(0, 2, (4, 7), generator=g).float()
    unseen = torch.rand(4, 7, generator=g) > .5
    criterion = RankerLoss(new_positive_weight=2.)
    actual = criterion(z, y, unseen_mask=unseen, candidate_mask=torch.ones_like(y, dtype=torch.bool))
    expected = criterion(z_ref, y, unseen_mask=unseen)
    for k in actual:
        assert torch.equal(actual[k], expected[k])
    actual['loss'].backward()
    expected['loss'].backward()
    assert torch.equal(z.grad, z_ref.grad)


@pytest.mark.parametrize('mask,match', [
    (torch.ones(1, 3), 'boolean'),
    (torch.ones(1, 2, dtype=torch.bool), 'shape'),
])
def test_invalid_mask_contract_is_rejected(mask, match):
    with pytest.raises(ValueError, match=match):
        RankerLoss()(torch.zeros(1, 3), torch.zeros(1, 3), candidate_mask=mask)


def test_positive_target_in_padding_and_nonfinite_valid_values_are_rejected():
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match='Positive targets'):
        RankerLoss()(torch.tensor([[1., -float('inf')]]), torch.ones(1, 2), candidate_mask=mask)
    with pytest.raises(ValueError, match='logits must be finite'):
        RankerLoss()(torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 2), candidate_mask=mask)
