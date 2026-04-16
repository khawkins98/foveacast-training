"""Unit tests for KL divergence and correlation-coefficient metrics."""

from __future__ import annotations

import pytest
import torch

from foveacast_training.losses import correlation_coefficient, kl_divergence_saliency


@pytest.fixture
def random_map() -> torch.Tensor:
    """A fixed-seed random saliency map for KL / CC sanity tests."""
    torch.manual_seed(0)
    return torch.rand(2, 1, 32, 40)


# ---------- KL divergence ------------------------------------------------


def test_kl_is_near_zero_for_identical_maps(random_map):
    """KL(p || p) = 0 by definition. The eps shifts in Kroner's formulation
    produce a small residual (~1e-4 for a 32×40 map) but it's bounded and
    not meaningfully non-zero compared to realistic training losses.
    """
    loss = kl_divergence_saliency(random_map, random_map)
    assert abs(loss.item()) < 1e-3


def test_kl_is_positive_for_different_maps(random_map):
    torch.manual_seed(1)
    other = torch.rand_like(random_map)
    loss = kl_divergence_saliency(other, random_map)
    assert loss.item() > 0


def test_kl_is_scale_invariant():
    """Both arguments get sum-normalised internally, so multiplying either
    by a positive constant must not change the loss.
    """
    torch.manual_seed(2)
    target = torch.rand(1, 1, 16, 16) + 0.01
    pred = torch.rand(1, 1, 16, 16) + 0.01
    base = kl_divergence_saliency(pred, target)
    scaled = kl_divergence_saliency(pred * 17.0, target * 0.3)
    assert torch.isclose(base, scaled, rtol=1e-4)


def test_kl_handles_all_zero_target():
    """Degenerate input (all-zero ground truth) shouldn't NaN. The eps
    shifts produce a uniform distribution in the limit, so the loss is
    bounded.
    """
    target = torch.zeros(1, 1, 8, 8)
    pred = torch.rand(1, 1, 8, 8) + 0.01
    loss = kl_divergence_saliency(pred, target)
    assert torch.isfinite(loss)


def test_kl_output_is_scalar(random_map):
    loss = kl_divergence_saliency(random_map, random_map)
    assert loss.ndim == 0


# ---------- Correlation coefficient --------------------------------------


def test_cc_is_one_for_identical_maps(random_map):
    cc = correlation_coefficient(random_map, random_map)
    assert torch.isclose(cc, torch.tensor(1.0), atol=1e-5)


def test_cc_is_between_minus_one_and_one(random_map):
    torch.manual_seed(3)
    other = torch.rand_like(random_map)
    cc = correlation_coefficient(other, random_map)
    assert -1.0 <= cc.item() <= 1.0


def test_cc_is_symmetric(random_map):
    """cc(a, b) == cc(b, a) — property of Pearson correlation."""
    torch.manual_seed(4)
    other = torch.rand_like(random_map)
    cc_ab = correlation_coefficient(random_map, other)
    cc_ba = correlation_coefficient(other, random_map)
    assert torch.isclose(cc_ab, cc_ba)


def test_cc_is_negative_for_anti_correlated_maps(random_map):
    anti = -random_map + random_map.mean()
    cc = correlation_coefficient(anti, random_map)
    assert cc.item() < 0


def test_cc_handles_constant_map():
    """A constant map has zero variance; eps should prevent NaN."""
    target = torch.rand(1, 1, 8, 8)
    pred = torch.ones_like(target) * 0.5
    cc = correlation_coefficient(pred, target)
    assert torch.isfinite(cc)
