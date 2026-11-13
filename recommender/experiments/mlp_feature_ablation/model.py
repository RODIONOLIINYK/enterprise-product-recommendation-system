"""Versioned Stage 1 MLP selector with physical feature-omission contract."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .schema import (
    FeatureAblationSpec,
    build_feature_ablation_spec,
    build_input_contract,
    spec_from_input_contract,
    validate_input_contract,
)


class FeatureAblatedStage1Selector(nn.Module):
    """Pointwise Stage 1 MLP selector over [u, v, retained_context, customer_stats_6].

    Supports arbitrary name-resolved feature-removal specs, e.g.:
    - no_repeat_no_month (uv_44): 38 retained context features -> 300D input (44 scalars).
    - repeat_no_month (uv_45): 39 retained context features -> 301D input (45 scalars).
    """

    def __init__(
        self,
        spec: Optional[FeatureAblationSpec] = None,
        *,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spec = spec if spec is not None else build_feature_ablation_spec()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width.")

        self.embedding_dim = int(self.spec.embedding_dim)
        self.source_num_features = int(self.spec.source_num_features)
        self.num_retained_features = int(self.spec.num_retained_features)
        self.num_customer_stats = int(self.spec.num_customer_stats)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.recipe = self.spec.recipe
        self.ordered_input_blocks = self.spec.ordered_input_blocks
        self.input_dim = int(self.spec.input_dim)

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for width in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

        self.register_buffer(
            "retained_indices_tensor",
            torch.tensor(self.spec.retained_indices, dtype=torch.long),
            persistent=False,
        )

    def input_contract(self) -> Dict[str, Any]:
        return build_input_contract(self.spec)

    def select_context(self, features_41: torch.Tensor) -> torch.Tensor:
        """Select retained context columns by resolved schema index before concatenation."""
        if features_41.ndim != 3 or features_41.shape[-1] != self.source_num_features:
            raise ValueError(
                f"Expected features_{self.source_num_features} shape (B, P, {self.source_num_features}), "
                f"got {tuple(features_41.shape)}."
            )
        idx = self.retained_indices_tensor
        if idx.device != features_41.device:
            idx = idx.to(features_41.device)
        return torch.index_select(features_41, -1, idx)

    def _score_chunk(
        self,
        u_expanded: torch.Tensor,
        v_chunk: torch.Tensor,
        f_chunk: torch.Tensor,
        stats_expanded: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([u_expanded, v_chunk, f_chunk, stats_expanded], dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Concatenated input width {x.shape[-1]} != expected model input_dim {self.input_dim}."
            )
        return self.head(x).squeeze(-1)

    def forward(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        customer_stats_6: torch.Tensor,
        features_41: Optional[torch.Tensor] = None,
        retained_context: Optional[torch.Tensor] = None,
        context_38: Optional[torch.Tensor] = None,
        product_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if retained_context is not None and context_38 is not None:
            raise ValueError("Provide at most one of retained_context or context_38.")
        ctx_tensor = retained_context if retained_context is not None else context_38
        if (features_41 is None) == (ctx_tensor is None):
            raise ValueError("Provide exactly one of features_41 or retained_context (or context_38).")
        if u.ndim != 2 or u.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected u shape (B, {self.embedding_dim}), got {tuple(u.shape)}."
            )
        if (
            customer_stats_6.ndim != 2
            or customer_stats_6.shape[-1] != self.num_customer_stats
        ):
            raise ValueError(
                f"Expected customer_stats_6 shape (B, {self.num_customer_stats}), "
                f"got {tuple(customer_stats_6.shape)}."
            )

        if features_41 is not None:
            retained = self.select_context(features_41)
        else:
            if (
                ctx_tensor.ndim != 3
                or ctx_tensor.shape[-1] != self.num_retained_features
            ):
                raise ValueError(
                    f"Expected retained context shape (B, P, {self.num_retained_features}), "
                    f"got {tuple(ctx_tensor.shape)}."
                )
            retained = ctx_tensor

        batch_size, num_products, _ = retained.shape
        if u.shape[0] != batch_size or customer_stats_6.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: u={u.shape[0]}, stats={customer_stats_6.shape[0]}, context={batch_size}."
            )

        if v.ndim == 2:
            if v.shape != (num_products, self.embedding_dim):
                raise ValueError(
                    f"Expected v shape ({num_products}, {self.embedding_dim}), got {tuple(v.shape)}."
                )
        elif v.ndim == 3:
            if v.shape != (batch_size, num_products, self.embedding_dim):
                raise ValueError(
                    f"Expected v shape ({batch_size}, {num_products}, {self.embedding_dim}), "
                    f"got {tuple(v.shape)}."
                )
        else:
            raise ValueError(f"Expected 2D or 3D v tensor, got shape {tuple(v.shape)}.")

        if (
            product_chunk_size is not None
            and int(product_chunk_size) > 0
            and int(product_chunk_size) < num_products
        ):
            chunk = int(product_chunk_size)
            outputs = []
            for start in range(0, num_products, chunk):
                end = min(start + chunk, num_products)
                span = end - start
                u_chunk = u.unsqueeze(1).expand(batch_size, span, self.embedding_dim)
                stats_chunk = customer_stats_6.unsqueeze(1).expand(
                    batch_size, span, self.num_customer_stats
                )
                v_chunk = (
                    v[start:end].unsqueeze(0).expand(batch_size, span, self.embedding_dim)
                    if v.ndim == 2
                    else v[:, start:end, :]
                )
                f_chunk = retained[:, start:end, :]
                outputs.append(self._score_chunk(u_chunk, v_chunk, f_chunk, stats_chunk))
            return torch.cat(outputs, dim=1)

        u_exp = u.unsqueeze(1).expand(batch_size, num_products, self.embedding_dim)
        stats_exp = customer_stats_6.unsqueeze(1).expand(
            batch_size, num_products, self.num_customer_stats
        )
        v_exp = (
            v.unsqueeze(0).expand(batch_size, num_products, self.embedding_dim)
            if v.ndim == 2
            else v
        )
        return self._score_chunk(u_exp, v_exp, retained, stats_exp)


def load_ablated_selector_checkpoint(
    checkpoint_path: Union[Path, str],
    *,
    expected_spec: Optional[FeatureAblationSpec] = None,
    map_location: Union[str, torch.device] = "cpu",
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.1,
) -> Tuple[FeatureAblatedStage1Selector, Dict[str, Any]]:
    """Strictly load a FeatureAblatedStage1Selector checkpoint, rejecting any contract/width mismatch."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}.")
    if "input_contract" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required persisted 'input_contract'."
        )
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required 'model_state_dict'."
        )

    persisted_contract = ckpt["input_contract"]
    if expected_spec is not None:
        validate_input_contract(persisted_contract, build_input_contract(expected_spec))
        spec = expected_spec
    else:
        spec = spec_from_input_contract(persisted_contract)

    state = ckpt["model_state_dict"]
    if "head.0.weight" not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing 'head.0.weight'.")
    first_weight = state["head.0.weight"]
    if first_weight.ndim != 2 or int(first_weight.shape[1]) != spec.input_dim:
        raise ValueError(
            f"Checkpoint input weight width mismatch in {checkpoint_path}: "
            f"head.0.weight has in_features={first_weight.shape[1] if first_weight.ndim == 2 else None}, "
            f"expected {spec.input_dim}."
        )

    model = FeatureAblatedStage1Selector(
        spec,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(state, strict=True)
    return model, dict(ckpt)
