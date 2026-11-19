# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains smoothed minimax homogeneity objectives for magnet optimization.
#
#  objectives.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

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


def batched_peak_to_peak_ppm(readings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact peak-to-peak of scalar readings (..., P) in ppm, one value per leading index."""
    mean = readings.mean(dim=-1, keepdim=True)
    rel = (readings - mean) / mean.abs().clamp_min(eps)
    return (rel.amax(dim=-1) - rel.amin(dim=-1)) * 1e6


def batched_soft_peak_to_peak_ppm(readings: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Log-sum-exp smoothed peak-to-peak of scalar readings (..., P) in ppm.

    The batched counterpart of ``soft_peak_to_peak_ppm`` for many children at once;
    the readings are |B| or a simulated probe reading, not field vectors.
    """
    mean = readings.mean(dim=-1, keepdim=True)
    rel = (readings - mean) / mean.abs().clamp_min(eps)
    sharpness = rel.new_tensor(float(alpha)).clamp_min(eps)
    log_n = torch.log(rel.new_tensor(float(rel.shape[-1])))
    soft_hi = (torch.logsumexp(sharpness * rel, dim=-1) - log_n) / sharpness
    soft_lo = -(torch.logsumexp(-sharpness * rel, dim=-1) - log_n) / sharpness
    return (soft_hi - soft_lo) * 1e6


def conditional_value_at_risk(values: torch.Tensor, level: float = 0.95) -> torch.Tensor:
    """
    Mean of the worst (1 - level) fraction of a sample, at least one value.

    CVaR at 0.95 of a homogeneity distribution is the average of its worst 5% of
    builds. It bounds the 95th percentile from above and, unlike a quantile, passes
    gradient to every sample in the tail rather than to one or two.
    """
    if not 0.0 <= level < 1.0:
        raise ValueError("level must lie in [0, 1).")
    count = max(1, math.ceil((1.0 - level) * values.numel() - 1e-9))
    return torch.topk(values.reshape(-1), count).values.mean()


class RobustObjective:
    """
    Tolerance-aware loss: mean, CVaR, and spread of the child homogeneity distribution.

    Each child of the nominal design is scored by its smoothed peak-to-peak in
    ppm. The mean term uses every child and moves the whole distribution; the
    CVaR term uses only the worst tail and pushes down the bad builds without
    penalising lucky ones. The standard-deviation term narrows the distribution
    explicitly; it also penalises lucky builds, so it is best weighted below the
    mean. The optional band acts on the nominal field, as in
    ``SmoothedMinimaxObjective``.
    """

    def __init__(
        self, alpha: float = 2.0e3, alpha_max: float = 2.0e4, alpha_growth: float = 1.01,
        cvar_level: float = 0.95, mean_weight: float = 1.0, cvar_weight: float = 1.0,
        std_weight: float = 0.0, band_weight: float = 0.0, band_t: tuple[float, float] | None = None,
    ) -> None:
        self.alpha: float = alpha
        self.alpha_max: float = alpha_max
        self.alpha_growth: float = alpha_growth
        self.cvar_level: float = cvar_level
        self.mean_weight: float = mean_weight
        self.cvar_weight: float = cvar_weight
        self.std_weight: float = std_weight
        self.band_weight: float = band_weight
        self.band_t: tuple[float, float] | None = band_t

    def step(self) -> float:
        """Ramp alpha geometrically toward alpha_max and return the new value."""
        self.alpha = min(self.alpha * self.alpha_growth, self.alpha_max)
        return self.alpha

    def child_homogeneity(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Smoothed peak-to-peak in ppm of every child, shape (N,)."""
        return batched_soft_peak_to_peak_ppm(output["child_readings"], self.alpha)

    def __call__(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the loss in ppm on an output holding "child_readings" as (N, P)."""
        homogeneity = self.child_homogeneity(output)
        loss = (self.mean_weight * homogeneity.mean()
                + self.cvar_weight * conditional_value_at_risk(homogeneity, self.cvar_level))
        if self.std_weight != 0.0:
            loss = loss + self.std_weight * homogeneity.std(unbiased=False)
        if self.band_weight != 0.0 and self.band_t is not None:
            loss = loss + self.band_weight * field_band_loss(output["B"], *self.band_t)
        return loss


class DiracWassersteinObjective:
    """
    Tolerance-aware loss: transport distance of the child homogeneity distribution to a Dirac.

    The children's smoothed peak-to-peak values form an empirical distribution that
    is pulled towards a point mass at ``target_ppm`` (0 by default) with the
    Sinkhorn divergence of ``geomloss.SamplesLoss``. Against a Dirac the transport
    plan is forced -- all mass goes to the one point -- so the divergence is
    E[|h - t|^p] / p, less an entropic debiasing term that vanishes with the blur.
    For p = 2 that is ((mean - t)^2 + std^2) / 2: the location and the breadth of
    the distribution in a single term. p = 1 reduces to the mean distance alone.

    With ``target="parent"`` the Dirac sits at the smoothed peak-to-peak of the
    unperturbed design itself, so the distance is sqrt(gap^2 + std^2): how far and
    how widely the built children fall from the design value. The target is
    detached (stop-gradient): the gradient pulls the children towards the design
    and never the design towards the children, which would otherwise be the
    cheapest way to close the gap. Pair it with ``mean_weight`` > 0 so the overall
    homogeneity cannot drift.

    The loss is ``mean_weight * mean(h) + distance_weight * W``, where W is reported
    as (p * S)^(1/p), the Wasserstein-p distance in ppm, so its scale matches the
    other objectives and the optional band on the nominal field keeps its meaning.
    Homogeneities are divided by ``scale_ppm`` inside the transport problem only,
    to keep the blur and the solver well conditioned.
    """

    def __init__(
        self, alpha: float = 2.0e3, alpha_max: float = 2.0e4, alpha_growth: float = 1.01,
        p: int = 2, blur_ppm: float = 50.0, target_ppm: float = 0.0, scale_ppm: float = 1.0e3,
        band_weight: float = 0.0, band_t: tuple[float, float] | None = None,
        target: str = "fixed", mean_weight: float = 0.0, distance_weight: float = 1.0,
    ) -> None:
        from geomloss import SamplesLoss

        if p not in (1, 2):
            raise ValueError("p must be 1 or 2.")
        if target not in ("fixed", "parent"):
            raise ValueError('target must be "fixed" or "parent".')
        if blur_ppm <= 0.0 or scale_ppm <= 0.0:
            raise ValueError("blur_ppm and scale_ppm must be positive.")
        self.alpha: float = alpha
        self.alpha_max: float = alpha_max
        self.alpha_growth: float = alpha_growth
        self.p: int = p
        self.blur_ppm: float = blur_ppm
        self.target_ppm: float = target_ppm
        self.scale_ppm: float = scale_ppm
        self.target: str = target
        self.mean_weight: float = mean_weight
        self.distance_weight: float = distance_weight
        self.band_weight: float = band_weight
        self.band_t: tuple[float, float] | None = band_t
        self.samples_loss = SamplesLoss("sinkhorn", p=p, blur=blur_ppm / scale_ppm, debias=True)

    def step(self) -> float:
        """Ramp alpha geometrically toward alpha_max and return the new value."""
        self.alpha = min(self.alpha * self.alpha_growth, self.alpha_max)
        return self.alpha

    def child_homogeneity(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Smoothed peak-to-peak in ppm of every child, shape (N,)."""
        return batched_soft_peak_to_peak_ppm(output["child_readings"], self.alpha)

    def target_homogeneity(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Position of the Dirac in ppm: fixed, or the detached smoothed p2p of the nominal field."""
        if self.target == "fixed":
            return output["child_readings"].new_tensor(self.target_ppm)
        readings = torch.linalg.norm(output["B"], dim=-1)[None, :]
        return batched_soft_peak_to_peak_ppm(readings, self.alpha)[0].detach()

    def distance_ppm(self, homogeneity: torch.Tensor, target_ppm: torch.Tensor | float | None = None) -> torch.Tensor:
        """Wasserstein-p distance in ppm between child homogeneities (N,) and the Dirac."""
        target_ppm = self.target_ppm if target_ppm is None else target_ppm
        samples = (homogeneity / self.scale_ppm)[:, None]
        target = (torch.as_tensor(target_ppm, dtype=samples.dtype, device=samples.device) / self.scale_ppm).reshape(1, 1)
        divergence = self.samples_loss(samples, target).clamp_min(1e-12)
        return self.scale_ppm * (self.p * divergence) ** (1.0 / self.p)

    def __call__(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the loss in ppm on an output holding "child_readings" as (N, P)."""
        homogeneity = self.child_homogeneity(output)
        loss = self.distance_weight * self.distance_ppm(homogeneity, self.target_homogeneity(output))
        if self.mean_weight != 0.0:
            loss = loss + self.mean_weight * homogeneity.mean()
        if self.band_weight != 0.0 and self.band_t is not None:
            loss = loss + self.band_weight * field_band_loss(output["B"], *self.band_t)
        return loss


class SensitivityRegularized:
    """
    Add the linearized error sensitivity of the design to another objective.

    ``ToleranceAwareModel(compute_sensitivity=True)`` puts the closed-form RMS
    inhomogeneity error in ppm into the output as "sensitivity_ppm". Unlike the
    spread of a finite set of children it has no sampling noise, so it gives the
    optimizer a clean gradient towards designs that tolerate their errors.
    """

    def __init__(self, objective, weight: float) -> None:
        self.objective = objective
        self.weight: float = weight

    @property
    def alpha(self) -> float:
        return self.objective.alpha

    def step(self) -> float:
        return self.objective.step()

    def __call__(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.objective(output) + self.weight * output["sensitivity_ppm"]


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
