"""Saliency-prediction losses and metrics.

Ports of the losses/metrics from Kroner et al.'s reference implementation
(https://github.com/alexanderkroner/saliency/blob/master/loss.py, MIT) plus
the correlation-coefficient (CC) validation metric used across the saliency
literature.

Both functions operate on `(N, 1, H, W)` tensors — the shape `MSINet.forward`
returns and the shape `UEyesDataset` produces for ground-truth maps. No
channel-order assumptions; they treat saliency as a single-channel scalar
field.

Kroner's formulation converts both maps to probability distributions (sum
to 1 per image) before computing KL divergence. That's standard for
saliency: absolute scale is uninformative — what matters is where
attention goes relative to other pixels.
"""

from __future__ import annotations

import torch


def kl_divergence_saliency(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """KL divergence between predicted and target saliency maps.

    Each map is divided by its per-image sum so both sides sum to 1, then
    KL(target || pred) is computed over all pixels and averaged across the
    batch. Matches Kroner's `loss.kld` bit-for-bit (ported from TF to
    PyTorch with identical eps handling).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted saliency map, shape `(N, 1, H, W)`. Must be non-negative.
    target : torch.Tensor
        Ground-truth saliency map, same shape, non-negative. Absolute scale
        of either argument does not matter — both are sum-normalised
        internally.
    eps : float
        Small constant to avoid division-by-zero and log(0). Matches the
        reference's default.

    Returns
    -------
    torch.Tensor
        Scalar loss (batch-averaged KL divergence).
    """
    # why: match Kroner's exact reduction axes — (1, 2, 3) in NCHW = channel
    # + H + W = everything but the batch dim. keepdim=True so the
    # broadcasting-divide gives a per-image normalisation.
    #
    # Degenerate-input note: if `target` is all-zero (extremely rare but
    # possible on a blank ground-truth map), this divide makes `target` a
    # tensor of zeros — the subsequent `target * log(...)` is then
    # zero-everywhere and the loss is 0, not a uniform distribution as
    # one might expect from a "normalise by sum" description. Phase 5's
    # Dataset currently can't produce such a map (saliency values are
    # [0, 1] by construction with at least one non-zero pixel per UEyes
    # ground truth), so we don't guard against it. If that ever changes,
    # the failure mode is "loss = 0 silently" rather than NaN.
    target_sum = target.sum(dim=(1, 2, 3), keepdim=True)
    target = target / (eps + target_sum)

    pred_sum = pred.sum(dim=(1, 2, 3), keepdim=True)
    pred = pred / (eps + pred_sum)

    # why: eps inside the log argument *and* on each of the two eps-shifted
    # terms follows Kroner's reference. The double-eps looks redundant but
    # protects against both (a) target close to zero (the outer eps) and
    # (b) pred close to zero (the inner eps).
    loss = target * torch.log(eps + target / (eps + pred))
    return loss.sum(dim=(1, 2, 3)).mean()


def correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Pearson correlation coefficient between predicted and target saliency
    maps, computed per image and averaged over the batch.

    The standard saliency-prediction validation metric. Range `[-1, 1]`;
    higher is better. Unlike KL divergence, CC is symmetric in its
    arguments and treats the maps as random variables rather than
    distributions.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted saliency map, shape `(N, 1, H, W)`.
    target : torch.Tensor
        Ground-truth saliency map, same shape.
    eps : float
        Numerical guard against division by zero when a map is constant.

    Returns
    -------
    torch.Tensor
        Scalar in `[-1, 1]`.
    """
    # Flatten all dims except batch so each image contributes one
    # scalar correlation; then average.
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)
    # why: sqrt of product-of-variances; eps inside the sqrt avoids NaN
    # gradient when either map has zero variance (rare but possible on
    # degenerate inputs).
    denominator = torch.sqrt(
        eps + (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1)
    )
    return (numerator / denominator).mean()
