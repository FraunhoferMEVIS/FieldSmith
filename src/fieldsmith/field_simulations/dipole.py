# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains fast dipole magnetic-field calculation utilities.
#
#  dipole.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

MU_0 = 4.0e-7 * torch.pi


def calc_b_at_points_fast(
    dipole_br: torch.Tensor,
    vol: torch.Tensor | float,
    dipole_pos: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-12,
    chunk_points: int | None = 65536,
    chunk_magnets: int | None = None,
) -> torch.Tensor:
    """
    Compute the magnetic field at arbitrary points for magnetic dipoles.

    Args:
        dipole_br: Remanence vectors in tesla, shape (M, 3).
        vol: Magnet volumes in m^3, shape (M,) or a scalar.
        dipole_pos: Dipole positions in meters, shape (M, 3).
        points: Evaluation points in meters, shape (P, 3).
        eps: Minimum distance used for numerical stability.
        chunk_points: Number of evaluation points per chunk.
        chunk_magnets: Number of dipoles per chunk.

    Returns:
        Magnetic field in tesla, shape (P, 3).
    """
    device = points.device
    dtype = points.dtype

    if not isinstance(dipole_br, torch.Tensor):
        dipole_br = torch.tensor(dipole_br, dtype=dtype, device=device)
    else:
        dipole_br = dipole_br.to(device=device, dtype=dtype)

    if not isinstance(dipole_pos, torch.Tensor):
        dipole_pos = torch.tensor(dipole_pos, dtype=dtype, device=device)
    else:
        dipole_pos = dipole_pos.to(device=device, dtype=dtype)

    if not isinstance(vol, torch.Tensor):
        if isinstance(vol, (int, float)):
            vol = torch.full((dipole_br.shape[0],), vol, dtype=dtype, device=device)
        else:
            vol = torch.tensor(vol, dtype=dtype, device=device)
    else:
        vol = vol.to(device=device, dtype=dtype)
    vol = vol.reshape(-1)

    if vol.numel() == 1:
        vol = vol.expand(dipole_br.shape[0])

    if dipole_br.shape != dipole_pos.shape:
        raise ValueError("dipole_br and dipole_pos must both have shape (M, 3).")
    if vol.shape[0] != dipole_br.shape[0]:
        raise ValueError("vol must be scalar or have shape (M,).")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (P, 3).")

    dipole_moments = dipole_br * (vol[:, None] / MU_0)
    n_points = points.shape[0]
    n_magnets = dipole_br.shape[0]
    b_out = torch.empty((n_points, 3), dtype=dtype, device=device)

    if chunk_points is None:
        chunk_points = n_points
    if chunk_magnets is None:
        chunk_magnets = n_magnets

    k = MU_0 / (4.0 * torch.pi)
    eps2 = eps * eps

    for point_start in range(0, n_points, chunk_points):
        point_end = min(point_start + chunk_points, n_points)
        point_chunk = points[point_start:point_end]
        b_chunk = torch.zeros((point_chunk.shape[0], 3), dtype=dtype, device=device)

        for magnet_start in range(0, n_magnets, chunk_magnets):
            magnet_end = min(magnet_start + chunk_magnets, n_magnets)
            positions = dipole_pos[magnet_start:magnet_end]
            moments = dipole_moments[magnet_start:magnet_end]

            r_vec = point_chunk[:, None, :] - positions[None, :, :]
            r2 = torch.sum(r_vec * r_vec, dim=-1, keepdim=True).clamp_min(eps2)
            inv_r = torch.rsqrt(r2)
            inv_r3 = inv_r / r2
            inv_r5 = inv_r3 / r2
            m_dot_r = torch.sum(moments[None, :, :] * r_vec, dim=-1, keepdim=True)
            b = 3.0 * r_vec * m_dot_r * inv_r5 - moments[None, :, :] * inv_r3
            b_chunk = b_chunk + b.sum(dim=1)

        b_out[point_start:point_end] = k * b_chunk

    return b_out


def calc_b_at_points(
    dipole_br: torch.Tensor,
    vol: torch.Tensor | float,
    dipole_pos: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-12,
    chunk_points: int | None = 65536,
) -> torch.Tensor:
    """
    Compatibility wrapper for arbitrary-point B-field calculation.

    Returns:
        Magnetic field in tesla, shape (P, 3).
    """
    return calc_b_at_points_fast(
        dipole_br=dipole_br,
        vol=vol,
        dipole_pos=dipole_pos,
        points=points,
        eps=eps,
        chunk_points=chunk_points,
    )
