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


def normalized_scanpath_saliency(
    pred: torch.Tensor,
    fixmap: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Normalized Scanpath Saliency (NSS) metric.

    Standard saliency-prediction metric alongside CC and KLD. Defined as
    the mean of the z-normalised prediction sampled at fixation locations.
    A random uniform prediction gives NSS ≈ 0; a perfect prediction
    (spike at every fixation, zero elsewhere) gives NSS equal to the
    per-image standard deviation over the fixation set.

    Reference: Bylinskii et al., "What Do Different Evaluation Metrics
    Tell Us About Saliency Models?", TPAMI 2019.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted saliency map, shape `(N, 1, H, W)`.
    fixmap : torch.Tensor
        Binary fixation map, same shape. Values in [0, 1]; any pixel
        above 0.5 counts as a fixation (threshold handles both strictly-
        binary {0, 1} UEyes fixmaps and lightly-smoothed variants).
    eps : float
        Numerical guard when std or fixation count is zero.

    Returns
    -------
    torch.Tensor
        Scalar NSS, averaged across the batch.
    """
    # Z-normalise pred per image.
    pred_flat = pred.flatten(start_dim=1)
    mu = pred_flat.mean(dim=1, keepdim=True)
    std = pred_flat.std(dim=1, keepdim=True)
    pred_z = (pred_flat - mu) / (eps + std)

    # Sample at fixation locations.
    fixmap_flat = fixmap.flatten(start_dim=1)
    fixmap_bin = (fixmap_flat > 0.5).to(pred_z.dtype)
    # why: sum/count rather than masked mean because fixmap_bin is a float
    # tensor on possibly-MPS device; boolean advanced indexing moves the
    # tensor through an unsupported codepath.
    n_fix = fixmap_bin.sum(dim=1)
    nss_per_image = (pred_z * fixmap_bin).sum(dim=1) / (eps + n_fix)
    return nss_per_image.mean()


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
