# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains least-squares seeding of magnet orientations.
#
#  lsq_seed.py
#  Marian Frei
#  22.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import warnings

import torch


def dipole_design_operator(
    positions: torch.Tensor,
    roi_points: torch.Tensor,
    volumes: torch.Tensor,
) -> torch.Tensor:
    """
    Build the linear map from per-magnet remanence vectors to the ROI field.

    Row block p, column block i is the 3x3 dipole kernel of magnet i at point p,
    volume_i / (4 pi d^3) (3 u u^T - I). Stacking the remanence vectors into a
    (3M,) vector and applying this matrix reproduces the field that
    calc_b_at_points_fast computes at every ROI point.

    Args:
        positions: Magnet positions in meters, shape (M, 3).
        roi_points: Field evaluation points in meters, shape (P, 3).
        volumes: Magnet volumes in m^3, shape (M,).

    Returns:
        Design operator, shape (3P, 3M).
    """
    difference = roi_points[:, None, :] - positions[None, :, :]
    distance = difference.norm(dim=-1)
    unit = difference / distance[..., None]
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    blocks = (volumes[None, :] / (4.0 * math.pi * distance**3))[..., None, None] * (
        3.0 * unit[..., :, None] * unit[..., None, :] - identity
    )
    n_points, n_magnets = distance.shape
    return blocks.permute(0, 2, 1, 3).reshape(3 * n_points, 3 * n_magnets)


def _mirror_pairing(positions: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Return the permutation pairing each magnet with its mirror image."""
    distances = torch.cdist(positions * signs, positions)
    closest, pairing = distances.min(dim=1)
    # A micron: tighter than any build tolerance, above the round-off of cdist.
    if float(closest.max()) > 1e-6:
        raise ValueError("positions are not symmetric under the requested mirror plane.")
    if pairing.sort().values.ne(torch.arange(len(pairing), device=pairing.device)).any():
        raise ValueError("mirror pairing of the positions is not one-to-one.")
    return pairing


def lsq_seed_orientations(
    positions: torch.Tensor,
    roi_points: torch.Tensor,
    target_direction: torch.Tensor,
    volumes: torch.Tensor,
    reg: float = 1e-3,
    iterations: int = 40,
    rho_start: float = 1e-4,
    rho_end: float = 1e4,
    mirror: str | None = None,
) -> torch.Tensor:
    """
    Solve for unit orientations producing the most uniform field along a target.

    The field is linear in the remanence vectors, so a uniform target is a
    least-squares problem. Its unconstrained solution gives every magnet its own
    moment magnitude, which an array of identical magnets cannot build, and
    normalizing that solution afterwards discards most of its quality. This solve
    carries the constraint instead: a ridge solve alternates with a proximal pull
    toward the nearest unit-magnitude design while the pull ramps up, so the
    iterate already has uniform magnitudes by the time it is normalized. The field
    level the array can reach is recomputed at every step rather than prescribed,
    so the magnitude of the target never enters.

    Any position set and any target direction are admissible; no shell geometry is
    assumed.

    Args:
        positions: Magnet positions in meters, shape (M, 3).
        roi_points: ROI points the field should be uniform over, shape (P, 3).
        target_direction: Desired field direction, shape (3,), need not be normalized.
        volumes: Magnet volumes in m^3, shape (M,).
        reg: Relative ridge regularization of the design operator.
        iterations: Proximal iterations over which the magnitude spread closes.
        rho_start: Initial proximal weight, relative to the operator scale.
        rho_end: Final proximal weight, large enough to make magnitudes uniform.
        mirror: Enforce mirror symmetry: "x", "y" or "z" names the coordinate that
            flips, so mirror="y" reflects across the y = 0 plane. The positions
            must be symmetric under that reflection, the ROI is symmetrized
            internally and every iterate is averaged with its reflected partner,
            so the result is symmetric to the last bit. A mirror-symmetric design
            produces no mean field along the mirror normal, so such targets are
            rejected.

    Returns:
        Unit remanence directions, shape (M, 3).
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (M, 3).")
    if roi_points.ndim != 2 or roi_points.shape[1] != 3:
        raise ValueError("roi_points must have shape (P, 3).")
    if target_direction.shape != (3,):
        raise ValueError("target_direction must have shape (3,).")

    # The solve runs in float64 on whatever device it is given, and only the
    # returned directions are cast back. The ridge solve divides the null-space
    # component of its right-hand side by the ridge, which amplifies any loss of
    # orthogonality in the factorization past what float32 can absorb.
    out_dtype = positions.dtype
    positions = positions.double()
    roi_points = roi_points.double()
    target_direction = target_direction.double()
    volumes = volumes.double()

    n_magnets = positions.shape[0]
    pairing = None
    if mirror is not None:
        axis = {"x": 0, "y": 1, "z": 2}.get(mirror)
        if axis is None:
            raise ValueError('mirror must be "x", "y" or "z".')
        if float(target_direction[axis].abs()) > 1e-12 * float(target_direction.norm()):
            raise ValueError(
                f"a {mirror}-mirror-symmetric design cannot produce a mean field "
                f"with a component along {mirror}; adjust the target or drop the mirror."
            )
        signs = torch.ones(3, dtype=positions.dtype, device=positions.device)
        signs[axis] = -1.0
        pairing = _mirror_pairing(positions, signs)
        roi_points = torch.cat((roi_points, roi_points * signs))

    def symmetrized(moments: torch.Tensor) -> torch.Tensor:
        if pairing is None:
            return moments
        vectors = moments.reshape(n_magnets, 3)
        return (0.5 * (vectors + vectors[pairing] * signs)).reshape(moments.shape)

    design = dipole_design_operator(positions, roi_points, volumes)
    target = target_direction / target_direction.norm()
    target_stack = target.repeat(roi_points.shape[0])

    # Every ridge solve goes through the (3P, 3P) Gram matrix,
    #     (A'A + mu I)^-1 r = [r - A' (A A' + mu I)^-1 A r] / mu,
    # so one eigendecomposition serves the whole ramp of mu. The design operator
    # is wide, 3P << 3M, so the Gram is its small dimension. Forming it squares
    # the condition number, which the ridge, never below reg * scale, keeps
    # harmless.
    gram = design @ design.T
    evals, evecs = torch.linalg.eigh(gram)
    evals = evals.clamp_min(0.0)
    scale = evals.sum() / (3 * n_magnets)
    design_t_target = design.T @ target_stack

    def ridge_solve(rhs: torch.Tensor, ridge: torch.Tensor) -> torch.Tensor:
        inner = evecs @ ((evecs.T @ (design @ rhs)) / (evals + ridge))
        return (rhs - design.T @ inner) / ridge

    moments = symmetrized(design.T @ (evecs @ ((evecs.T @ target_stack) / (evals + reg * scale))))
    # Held on the host so the loop never synchronizes with the device.
    rhos = torch.logspace(math.log10(rho_start), math.log10(rho_end), iterations).tolist()
    for rho in rhos:
        unit = moments.reshape(n_magnets, 3)
        unit = (unit / unit.norm(dim=1, keepdim=True).clamp_min(1e-30)).reshape(-1)
        level = (target_stack @ (design @ unit)) / roi_points.shape[0]
        # Flip the array if it points against the target. Done with torch.where
        # rather than a branch, which would read the sign back from the device.
        sign = torch.where(level < 0.0, -1.0, 1.0)
        unit, level = unit * sign, level * sign
        moments = symmetrized(
            ridge_solve(level * design_t_target + rho * scale * unit, (reg + rho) * scale)
        )

    moments = moments.reshape(n_magnets, 3)
    magnitudes = moments.norm(dim=1)
    if float(magnitudes.min() / magnitudes.max()) < 0.99:
        warnings.warn(
            "lsq_seed_orientations did not reach uniform magnitudes; "
            "the final normalization is lossy. Raise rho_end or iterations.",
            stacklevel=2,
        )
    return (moments / magnitudes[:, None].clamp_min(1e-30)).to(out_dtype)
