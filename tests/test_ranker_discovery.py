import math

import pytest
import torch
from torch import nn

from recommender.ranker import RankerLoss, RankerOutput
from recommender.train_ranker import evaluate_ranker


def test_new_positive_weight_one_matches_existing_objective():
    torch.manual_seed(7)
    logits = torch.randn(2, 4)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    unseen = torch.tensor([[True, True, False, False], [False, False, True, False]])
    baseline = RankerLoss(pos_weight=3.0, listwise_weight=0.7)(logits, targets)
    explicitly_unweighted = RankerLoss(
        pos_weight=3.0, listwise_weight=0.7, new_positive_weight=1.0,
    )(logits, targets, unseen_mask=unseen)
    for name in ("loss", "bce_loss", "list_loss"):
        torch.testing.assert_close(baseline[name], explicitly_unweighted[name])


def test_one_new_positive_has_twice_bce_gradient_and_listwise_contribution():
    logits = torch.zeros((1, 3), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    unseen = torch.tensor([[True, False, False]])

    base_bce = RankerLoss(pos_weight=1.0, listwise_weight=0.0)(logits, targets)
    weighted_bce = RankerLoss(
        pos_weight=1.0, listwise_weight=0.0, new_positive_weight=2.0,
    )(logits, targets, unseen_mask=unseen)
    base_grad = torch.autograd.grad(base_bce["loss"], logits, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted_bce["loss"], logits)[0]
    torch.testing.assert_close(weighted_grad[0, 0], 2 * base_grad[0, 0])
    torch.testing.assert_close(weighted_grad[0, 1:], base_grad[0, 1:])

    listwise_base = RankerLoss(pos_weight=1.0, listwise_weight=1.0)(logits, targets)
    listwise_weighted = RankerLoss(
        pos_weight=1.0, listwise_weight=1.0, new_positive_weight=2.0,
    )(logits, targets, unseen_mask=unseen)
    torch.testing.assert_close(listwise_weighted["list_loss"], 2 * listwise_base["list_loss"])


def test_repeat_positive_and_negative_loss_gradients_are_unchanged():
    logits = torch.tensor([[0.2, -0.3, 0.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0, 0.0]])
    unseen = torch.tensor([[True, False, True]])
    baseline = RankerLoss(pos_weight=2.0, listwise_weight=0.0)(logits, targets)
    weighted = RankerLoss(
        pos_weight=2.0, listwise_weight=0.0, new_positive_weight=2.0,
    )(logits, targets, unseen_mask=unseen)
    grad_base = torch.autograd.grad(baseline["loss"], logits, retain_graph=True)[0]
    grad_weighted = torch.autograd.grad(weighted["loss"], logits)[0]
    torch.testing.assert_close(grad_weighted[0, 1], grad_base[0, 1])
    torch.testing.assert_close(grad_weighted[0, 2], grad_base[0, 2])


class _FixedLogitRanker(nn.Module):
    def forward(self, **kwargs):
        logits = torch.tensor([[3.0, 2.0]], device=kwargs["candidate_vectors"].device)
        batch, candidates = logits.shape
        zeros = torch.zeros((batch, candidates, 1), device=logits.device)
        return RankerOutput(
            scores=logits,
            raw_scores=logits,
            attn_weights=torch.zeros((batch, candidates, 33), device=logits.device),
            h_attended=zeros,
            z_candidate=zeros,
            token_0_weight=torch.zeros((batch, candidates), device=logits.device),
            history_weights=torch.zeros((batch, candidates), device=logits.device),
        )


def test_new_recall_denominator_includes_positive_missing_from_slate():
    batch = {
        "text_embedding": torch.zeros((1, 1)),
        "city_id": torch.zeros(1, dtype=torch.long),
        "region_id": torch.zeros(1, dtype=torch.long),
        "industry_id": torch.zeros(1, dtype=torch.long),
        "tabular_features": torch.zeros((1, 1)),
        "sequence": torch.zeros((1, 1, 1)),
        "attention_mask": torch.zeros((1, 1), dtype=torch.long),
        "seq_product_ids": torch.zeros((1, 1), dtype=torch.long),
        "seq_category_ids": torch.zeros((1, 1), dtype=torch.long),
        "seq_subcategory_ids": torch.zeros((1, 1), dtype=torch.long),
        "seq_business_line_ids": torch.zeros((1, 1), dtype=torch.long),
        "candidate_indices": torch.zeros((1, 2), dtype=torch.long),
        "candidate_vectors": torch.zeros((1, 2, 1)),
        "pairwise_features": torch.zeros((1, 2, 13)),
        "labels": torch.tensor([[1.0, 0.0]]),
        "candidate_pids": [["seen-in-slate", "other"]],
        "positive_pids": [["seen-in-slate", "omitted-from-slate"]],
        "new_positive_pids": [["seen-in-slate", "omitted-from-slate"]],
        "seq_purchase_dates": None,
    }
    result = evaluate_ranker(
        model=_FixedLogitRanker(),
        val_loader=[batch],
        catalog_product_vectors=torch.zeros((2, 1)),
        device=torch.device("cpu"),
        loss_module=RankerLoss(),
        k_list=(1, 2),
    )
    assert result["new_target_count"] == 2
    assert result["new_hits@1"] == 1
    assert result["new_recall@1"] == pytest.approx(0.5)
    assert result["new_recall@2"] == pytest.approx(0.5)


def test_discovery_loss_requires_point_in_time_repeat_mask():
    with pytest.raises(ValueError, match="unseen_mask"):
        RankerLoss(new_positive_weight=2.0)(torch.zeros((1, 2)), torch.tensor([[1.0, 0.0]]))

