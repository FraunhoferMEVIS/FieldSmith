# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains smoothed minimax homogeneity objectives for magnet optimization.
#
#  objectives.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch


def relative_deviation(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Dimensionless relative deviation (|B| - mean|B|) / mean|B|, shape (P,)."""
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError("field must have shape (P, 3).")
    field_abs = torch.linalg.norm(field, dim=-1)
    mean_abs = field_abs.mean()
    return (field_abs - mean_abs) / mean_abs.abs().clamp_min(eps)


def soft_peak_to_peak_ppm(field: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the log-sum-exp smoothed peak-to-peak inhomogeneity of |B|.

    Args:
        field: Magnetic field in tesla, shape (P, 3).
        alpha: Smoothing sharpness; larger values approach the exact extrema.
        eps: Lower bound on the mean field and on alpha.

    Returns:
        Smoothed peak-to-peak in ppm, scalar. The log(P) bias is subtracted, so a uniform
        field scores exactly zero and the value lower-bounds the exact peak-to-peak.
    """
    rel = relative_deviation(field, eps)
    sharpness = rel.new_tensor(float(alpha)).clamp_min(eps)
    log_n = torch.log(rel.new_tensor(float(rel.numel())))
    soft_hi = (torch.logsumexp(sharpness * rel, dim=0) - log_n) / sharpness
    soft_lo = -(torch.logsumexp(-sharpness * rel, dim=0) - log_n) / sharpness
    return (soft_hi - soft_lo) * 1e6


def peak_to_peak_ppm(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact peak-to-peak inhomogeneity of |B| in ppm, for logging and gating."""
    rel = relative_deviation(field, eps)
    return (rel.max() - rel.min()) * 1e6


def rms_deviation_ppm(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Root-mean-square relative deviation of |B| in ppm."""
    return torch.sqrt((relative_deviation(field, eps) ** 2).mean()) * 1e6


def field_band_loss(field: torch.Tensor, lower_t: float, upper_t: float) -> torch.Tensor:
    """Dimensionless one-sided quadratic penalty, zero while mean|B| stays in the band."""
    if not upper_t >= lower_t > 0.0:
        raise ValueError("field_band_loss requires 0 < lower_t <= upper_t.")
    mean_abs = torch.linalg.norm(field, dim=-1).mean()
    below = torch.clamp((lower_t - mean_abs) / lower_t, min=0.0)
    above = torch.clamp((mean_abs - upper_t) / upper_t, min=0.0)
    return below * below + above * above


class SmoothedMinimaxObjective:
    """Weighted smoothed-minimax loss in ppm with a geometric alpha ramp."""

    def __init__(
        self, alpha: float = 2.0e3, alpha_max: float = 2.0e4, alpha_growth: float = 1.01,
        rms_weight: float = 0.0, band_weight: float = 0.0, band_t: tuple[float, float] | None = None,
    ) -> None:
        self.alpha: float = alpha
        self.alpha_max: float = alpha_max
        self.alpha_growth: float = alpha_growth
        self.rms_weight: float = rms_weight
        self.band_weight: float = band_weight
        self.band_t: tuple[float, float] | None = band_t

    def step(self) -> float:
        """Ramp alpha geometrically toward alpha_max and return the new value."""
        self.alpha = min(self.alpha * self.alpha_growth, self.alpha_max)
        return self.alpha

    def __call__(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the weighted loss in ppm on a model output holding "B" as (P, 3)."""
        field = output["B"]
        loss = soft_peak_to_peak_ppm(field, self.alpha)
        if self.rms_weight != 0.0:
            loss = loss + self.rms_weight * rms_deviation_ppm(field)
        if self.band_weight != 0.0 and self.band_t is not None:
            loss = loss + self.band_weight * field_band_loss(field, *self.band_t)
        return loss


class FieldFloor:
    """
    Hold the mean field above a floor, by an augmented Lagrangian.

    Homogeneity is bought by letting magnets oppose one another, which costs
    field strength: an unconstrained descent trades away as much as it is given.
    A specification that says "at least this many millitesla" therefore has to be
    enforced, and a plain quadratic penalty cannot do it -- it balances the
    objective against a finite gradient and settles slightly below the floor.

    The multiplier estimate accumulates the violation instead, so the constraint
    is met rather than approached. Update it on a slow cadence, not every step:
    it is a dual variable and reacts to the average violation, not the current
    one.
    """

    def __init__(self, floor_t: float, rho: float = 1.0e4, multiplier: float = 0.0) -> None:
        if floor_t <= 0.0:
            raise ValueError("floor_t must be positive.")
        self.floor_t: float = floor_t
        self.rho: float = rho
        self.multiplier: float = multiplier

    def violation(self, field: torch.Tensor) -> torch.Tensor:
        """Shortfall of the mean field below the floor, in tesla; zero when met."""
        return torch.clamp(self.floor_t - torch.linalg.norm(field, dim=-1).mean(), min=0.0)

    def update_multiplier(self, field: torch.Tensor) -> float:
        """Accumulate the current violation into the multiplier estimate."""
        with torch.no_grad():
            self.multiplier = max(0.0, self.multiplier + self.rho * float(self.violation(field)))
        return self.multiplier

    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        shortfall = self.violation(field)
        return 0.5 * self.rho * shortfall * shortfall + self.multiplier * shortfall


class CollisionPenalty:
    """
    Penalise magnet bodies that intersect.

    Only meaningful once positions or orientations can move bodies into one
    another. The penalty is quadratic in the penetration beyond a margin, so it
    is zero and gradient-free for a design that is comfortably apart and grows
    smoothly as one closes. The margin is deliberate slack: driving penetration
    to exactly zero fights the objective for no manufacturing benefit, since the
    acceptance gate allows a small tolerance anyway.

    The pair list is passed in rather than rebuilt: it is valid only for the
    geometry it was built from, so refreshing it is the caller's decision, taken
    at the same points where the demagnetisation coupling is refreshed.
    """

    def __init__(self, half_extent_m: float, margin_m: float = 3.0e-4, weight: float = 1.0) -> None:
        self.half_extent_m: float = half_extent_m
        self.margin_m: float = margin_m
        self.weight: float = weight

    def __call__(self, positions: torch.Tensor, axes: torch.Tensor,
                 index_i: torch.Tensor, index_j: torch.Tensor) -> torch.Tensor:
        from fieldsmith.geometry.collision import box_penetration

        depth = box_penetration(positions[index_i], positions[index_j],
                                axes[index_i], axes[index_j], self.half_extent_m)
        beyond = torch.clamp(depth - self.margin_m, min=0.0)
        return self.weight * (beyond * beyond).sum()
