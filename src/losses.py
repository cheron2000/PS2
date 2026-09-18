"""Losses for Sentinel-2 super-resolution training.

T2 implementation:
- configurable L1 or L2 reconstruction loss;
- heteroscedastic Gaussian NLL for the model's per-pixel log-variance output;
- weighted combination with optional validity mask.

The uncertainty head predicts log-variance, so the NLL is computed as
0.5 * (exp(-log_var) * residual^2 + log_var).  Clamping is used only inside
the loss to prevent numerical overflow/underflow during early training.

Expected shapes:
    prediction: (B, C, H, W)
    target:     (B, C, H, W)
    log_var:    (B, C, H, W)
    valid_mask: broadcastable to the tensors, typically (B, 1, H, W).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _masked_mean(value: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over valid elements; returns zero for an entirely invalid mask."""
    if valid_mask is None:
        return value.mean()

    mask = valid_mask.to(dtype=value.dtype)
    mask = torch.broadcast_to(mask, value.shape)
    denom = mask.sum()
    if denom.item() == 0:
        # Keep the returned scalar connected to autograd without producing NaNs.
        return value.sum() * 0.0
    return (value * mask).sum() / denom


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: str = "l1",
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute masked L1 or L2 reconstruction loss."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    kind = kind.lower()
    if kind == "l1":
        error = (prediction - target).abs()
    elif kind == "l2":
        error = (prediction - target).pow(2)
    else:
        raise ValueError("kind must be 'l1' or 'l2'")

    return _masked_mean(error, valid_mask)


def heteroscedastic_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    min_log_var: float = -20.0,
    max_log_var: float = 10.0,
) -> torch.Tensor:
    """Gaussian heteroscedastic NLL for a predicted log-variance map.

    The constant 0.5*log(2*pi) is omitted because it does not affect
    optimization. The model therefore learns both the reconstruction mean and
    a calibrated relative uncertainty field.
    """
    if prediction.shape != target.shape or prediction.shape != log_var.shape:
        raise ValueError(
            "prediction, target, and log_var must have identical shapes; got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(log_var.shape)}"
        )
    if min_log_var >= max_log_var:
        raise ValueError("min_log_var must be smaller than max_log_var")

    safe_log_var = log_var.clamp(min=min_log_var, max=max_log_var)
    inv_var = torch.exp(-safe_log_var)
    squared_error = (prediction - target).pow(2)
    nll = 0.5 * (inv_var * squared_error + safe_log_var)
    return _masked_mean(nll, valid_mask)


class SRLoss(nn.Module):
    """Combined reconstruction + uncertainty-aware training objective.

    Args:
        reconstruction: 'l1' or 'l2'.
        reconstruction_weight: weight of the direct image reconstruction term.
        nll_weight: weight of the heteroscedastic uncertainty NLL term.
        min_log_var/max_log_var: numerical safety bounds used by the NLL.
    """

    def __init__(
        self,
        reconstruction: str = "l1",
        reconstruction_weight: float = 1.0,
        nll_weight: float = 0.1,
        min_log_var: float = -20.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        if reconstruction_weight < 0 or nll_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if reconstruction_weight == 0 and nll_weight == 0:
            raise ValueError("at least one loss weight must be positive")

        self.reconstruction = reconstruction.lower()
        if self.reconstruction not in {"l1", "l2"}:
            raise ValueError("reconstruction must be 'l1' or 'l2'")
        self.reconstruction_weight = float(reconstruction_weight)
        self.nll_weight = float(nll_weight)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        return_components: bool = False,
    ):
        """Return total loss, optionally alongside detached component values."""
        recon = reconstruction_loss(
            prediction, target, kind=self.reconstruction, valid_mask=valid_mask
        )
        nll = heteroscedastic_nll(
            prediction,
            target,
            log_var,
            valid_mask=valid_mask,
            min_log_var=self.min_log_var,
            max_log_var=self.max_log_var,
        )
        total = self.reconstruction_weight * recon + self.nll_weight * nll

        if return_components:
            return total, {
                "reconstruction": recon.detach(),
                "uncertainty_nll": nll.detach(),
                "total": total.detach(),
            }
        return total


def smoke_test() -> None:
    """Minimal CPU test for shapes, masking, gradients, and finite losses."""
    torch.manual_seed(0)
    prediction = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randn(2, 4, 16, 16)
    log_var = torch.zeros_like(prediction, requires_grad=True)

    # Ignore the lower-right quadrant to exercise broadcastable (B,1,H,W) masks.
    mask = torch.ones(2, 1, 16, 16)
    mask[:, :, 8:, 8:] = 0

    criterion = SRLoss(reconstruction="l1", reconstruction_weight=1.0, nll_weight=0.1)
    total, parts = criterion(
        prediction, target, log_var, valid_mask=mask, return_components=True
    )
    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in parts.values())

    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert log_var.grad is not None and torch.isfinite(log_var.grad).all()

    # Fully invalid masks should be safe and produce a zero objective.
    zero_mask = torch.zeros_like(mask)
    zero = criterion(prediction, target, log_var, valid_mask=zero_mask)
    assert torch.equal(zero, torch.zeros_like(zero))

    print(
        "[losses smoke_test OK] "
        f"total={total.item():.6f}, recon={parts['reconstruction'].item():.6f}, "
        f"nll={parts['uncertainty_nll'].item():.6f}"
    )


if __name__ == "__main__":
    smoke_test()
