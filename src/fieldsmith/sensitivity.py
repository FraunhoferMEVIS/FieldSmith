# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the linearized sensitivity of a design to manufacturing errors.
#
#  sensitivity.py
#  Kostiantyn Lavronenko
#  19.10.2026
# -----------------------------------------------------------------------------

"""
Closed-form, differentiable error variance of a dipole design under small tolerances.

For small errors the field responds linearly. Each magnet's position, tilt, and
remanence errors are independent, so the variance of the field-strength error at
every evaluation point is a sum of per-magnet terms with closed-form dipole
derivatives -- no sampling, hence an exact and noise-free gradient towards
designs whose field is less sensitive to the errors. Only the part of the error
that varies across the field of view is counted: a common offset of |B| leaves
the peak-to-peak homogeneity unchanged.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint

from fieldsmith.field_simulations.dipole import MU_0

_K = MU_0 / (4.0 * torch.pi)


def _variance_block(
    positions: torch.Tensor,
    directions: torch.Tensor,
    moments: torch.Tensor,
    points: torch.Tensor,
    field_direction: torch.Tensor,
    position_variance: float,
    tilt_variance: float,
    remanence_variance_ratio: float,
) -> torch.Tensor:
    """Summed variance contribution (P,) of a block of magnets, in tesla squared."""
    d = points[:, None, :] - positions[None, :, :]
    r2 = (d * d).sum(dim=-1, keepdim=True)
    inv_r = torch.rsqrt(r2)
    inv_r3 = inv_r / r2
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2
    b = field_direction[:, None, :]
    d_dot_b = (d * b).sum(dim=-1, keepdim=True)
    moment = moments[None, :, :]
    m_dot_d = (moment * d).sum(dim=-1, keepdim=True)
    m_dot_b = (moment * b).sum(dim=-1, keepdim=True)

    # w = K G(d) b: the field-strength response at a point per unit moment.
    response = _K * (3.0 * d * d_dot_b * inv_r5 - b * inv_r3)
    # Gradient of b . B_i with respect to d, i.e. minus the position response.
    position_response = _K * (
        3.0 * (moment * d_dot_b + b * m_dot_d) * inv_r5
        - 15.0 * m_dot_d * d_dot_b * d * inv_r7
        + 3.0 * m_dot_b * d * inv_r5
    )
    # Remove the field-of-view mean per magnet: a uniform shift is not inhomogeneity.
    response = response - response.mean(dim=0, keepdim=True)
    position_response = position_response - position_response.mean(dim=0, keepdim=True)

    moment_norm2 = (moments * moments).sum(dim=-1)[None, :]
    along = (response * directions[None, :, :]).sum(dim=-1)
    remanence_term = remanence_variance_ratio * moment_norm2 * along * along
    tilt_term = 0.5 * tilt_variance * moment_norm2 * ((response * response).sum(dim=-1) - along * along)
    position_term = position_variance * (position_response * position_response).sum(dim=-1)
    return (remanence_term + tilt_term + position_term).sum(dim=1)


def linearized_error_variance(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    remanence: torch.Tensor,
    volumes: torch.Tensor,
    points: torch.Tensor,
    nominal_field: torch.Tensor,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    remanence_tolerance_t: float,
    chunk_magnets: int | None = 256,
) -> torch.Tensor:
    """
    Variance of the field-strength error at every point, first order in the errors.

    The tolerances are uniform plus-or-minus bounds, as in ``TolerancePerturbation``,
    so each contributes a variance of tolerance^2 / 3: per axis for positions, for
    the tilt angle (in a uniformly random direction perpendicular to the nominal
    magnetization), and for the remanence magnitude.

    Args:
        positions: Magnet centres in meters, shape (M, 3).
        orientations: Magnetization directions, shape (M, 3).
        remanence: Remanence magnitudes in tesla, shape (M,).
        volumes: Magnet volumes in m^3, shape (M,).
        points: Evaluation points in meters, shape (P, 3).
        nominal_field: Field of the nominal design at the points, shape (P, 3); only
            its direction is used, to project the errors on |B|.
        chunk_magnets: Magnets per checkpointed block, to bound memory.

    Returns:
        Variance in tesla squared of the field-of-view-varying part of the
        field-strength error, shape (P,).
    """
    directions = orientations / torch.linalg.norm(orientations, dim=-1, keepdim=True)
    moments = directions * (remanence * volumes / MU_0)[:, None]
    field_direction = (nominal_field / torch.linalg.norm(nominal_field, dim=-1, keepdim=True)).detach()
    position_variance = position_tolerance_m**2 / 3.0
    tilt_variance = rotation_tolerance_rad**2 / 3.0
    remanence_variance_ratio = (remanence_tolerance_t**2 / 3.0) / remanence.detach().mean().item() ** 2
    chunk = chunk_magnets or positions.shape[0]
    total = points.new_zeros(points.shape[0])
    grad = torch.is_grad_enabled() and (positions.requires_grad or orientations.requires_grad)
    for start in range(0, positions.shape[0], chunk):
        arguments = (
            positions[start:start + chunk], directions[start:start + chunk], moments[start:start + chunk],
            points, field_direction, position_variance, tilt_variance, remanence_variance_ratio,
        )
        if grad:
            total = total + torch.utils.checkpoint.checkpoint(_variance_block, *arguments, use_reentrant=False)
        else:
            total = total + _variance_block(*arguments)
    return total


def linearized_sensitivity_ppm(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    remanence: torch.Tensor,
    volumes: torch.Tensor,
    points: torch.Tensor,
    nominal_field: torch.Tensor,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    remanence_tolerance_t: float,
    chunk_magnets: int | None = 256,
) -> torch.Tensor:
    """
    RMS over the points of the expected inhomogeneity error, in ppm of the mean field.

    A scalar summary of ``linearized_error_variance`` that falls when the design
    becomes less sensitive to its tolerances, and also when the mean field rises,
    since the errors are absolute and the homogeneity is relative.
    """
    variance = linearized_error_variance(
        positions, orientations, remanence, volumes, points, nominal_field,
        position_tolerance_m, rotation_tolerance_rad, remanence_tolerance_t, chunk_magnets,
    )
    mean_field = torch.linalg.norm(nominal_field, dim=-1).mean()
    return variance.mean().clamp_min(0.0).sqrt() / mean_field * 1e6


__all__ = ["linearized_error_variance", "linearized_sensitivity_ppm"]
