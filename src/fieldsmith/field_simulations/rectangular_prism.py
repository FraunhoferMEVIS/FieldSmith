# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains fast dipole magnetic-field calculation utilities.
# Implementation follows the results of the publication:
#   "The Magnetic Scalar Potential for a Rectangular Prism", 
#   James B. et al., IEEE Transactions on Magnetics, vol. 62, no. 1, Jan. 2026.
#   https://ieeexplore.ieee.org/document/11245567
#
#  rectangular_prism.py
#  Kostiantyn Lavronenko
#  24.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

MU_0 = 4.0e-7 * torch.pi


def _as_tensor(x, *, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def _safe_log(x: torch.Tensor, eps: float) -> torch.Tensor:
    # The analytical expression has removable singularities.
    # This clamp is a practical numerical regularization.
    return torch.log(torch.clamp(x, min=eps))


def _F_prism(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Primitive used in the rectangular-prism scalar potential.

    F(i,j,k) = -i atan(jk / (i r)) + j log(k+r) + k log(j+r)

    Implemented with atan2 for better quadrant handling.
    """
    r = torch.sqrt(torch.clamp(i * i + j * j + k * k, min=eps * eps))

    atan_term = torch.atan2(j * k, i * r)
    return (
        -i * atan_term
        + j * _safe_log(k + r, eps)
        + k * _safe_log(j + r, eps)
    )


def _rect_face_integral(
    normal_coord: torch.Tensor,
    tangential_1: torch.Tensor,
    tangential_2: torch.Tensor,
    half_1: torch.Tensor,
    half_2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Integral over one rectangular face.

    Computes:
        sum_{s1,s2 = ±1} s1*s2 * F(normal_coord,
                                    tangential_1 + s1*half_1,
                                    tangential_2 + s2*half_2)
    """
    out = torch.zeros_like(normal_coord)

    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            out = out + (s1 * s2) * _F_prism(
                normal_coord,
                tangential_1 + s1 * half_1,
                tangential_2 + s2 * half_2,
                eps,
            )

    return out


def _scalar_potential_rectangular_prisms(
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Magnetic scalar potential phi_M from uniformly magnetized rectangular prisms.

    Args:
        magnet_br: remanence vectors in tesla, shape (M, 3).
                   This is converted to magnetization M = Br / mu0.
        magnet_size: full prism side lengths in meters, shape (M, 3).
                     Columns are [sx, sy, sz].
        magnet_pos: prism centers in meters, shape (M, 3).
        points: evaluation points in meters, shape (P, 3).

    Returns:
        phi_M at each point, shape (P,).
    """
    magnetization = magnet_br / MU_0  # A/m
    half = 0.5 * magnet_size

    # Relative coordinates from magnet center to evaluation point.
    rel = points[:, None, :] - magnet_pos[None, :, :]

    x = rel[..., 0]
    y = rel[..., 1]
    z = rel[..., 2]

    a = half[:, 0][None, :]
    b = half[:, 1][None, :]
    c = half[:, 2][None, :]

    mx = magnetization[:, 0][None, :]
    my = magnetization[:, 1][None, :]
    mz = magnetization[:, 2][None, :]

    # x-faces: +a face minus -a face
    nx = (
        _rect_face_integral(x - a, y, z, b, c, eps)
        - _rect_face_integral(x + a, y, z, b, c, eps)
    )

    # y-faces
    ny = (
        _rect_face_integral(y - b, z, x, c, a, eps)
        - _rect_face_integral(y + b, z, x, c, a, eps)
    )

    # z-faces
    nz = (
        _rect_face_integral(z - c, x, y, a, b, eps)
        - _rect_face_integral(z + c, x, y, a, b, eps)
    )

    phi_per_magnet = (mx * nx + my * ny + mz * nz) / (4.0 * torch.pi)

    return phi_per_magnet.sum(dim=1)


def calc_b_at_points_rect_prism_potential(
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-12,
    chunk_points: int | None = 16384,
) -> torch.Tensor:
    """
    Compute B-field from uniformly magnetized rectangular prisms using the
    scalar-potential approximation from James et al.

    Args:
        magnet_br:
            Remanence vectors in tesla, shape (M, 3).
            Same role as dipole_br in your current code.

        magnet_size:
            Full side lengths of each rectangular prism in meters, shape (M, 3).
            Example: cube of side length L -> [L, L, L].

        magnet_pos:
            Prism centers in meters, shape (M, 3).

        points:
            Evaluation points in meters, shape (P, 3).

        eps:
            Numerical regularization for removable log/atan singularities.

        chunk_points:
            Number of evaluation points per chunk.

    Returns:
        B-field in tesla, shape (P, 3).

    Notes:
        This returns B = mu0 * H, which is the correct flux density in air/outside
        the magnets. For points inside a magnet, the material relation is
        B = mu0 * (H + M), so you would need to add Br for the containing prism.
    """
    device = points.device
    dtype = points.dtype

    magnet_br = _as_tensor(magnet_br, dtype=dtype, device=device)
    magnet_size = _as_tensor(magnet_size, dtype=dtype, device=device)
    magnet_pos = _as_tensor(magnet_pos, dtype=dtype, device=device)
    points = _as_tensor(points, dtype=dtype, device=device)

    if magnet_br.ndim != 2 or magnet_br.shape[1] != 3:
        raise ValueError("magnet_br must have shape (M, 3).")
    if magnet_size.ndim != 2 or magnet_size.shape[1] != 3:
        raise ValueError("magnet_size must have shape (M, 3).")
    if magnet_pos.shape != magnet_br.shape:
        raise ValueError("magnet_pos and magnet_br must both have shape (M, 3).")
    if magnet_size.shape[0] != magnet_br.shape[0]:
        raise ValueError("magnet_size must have shape (M, 3).")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (P, 3).")

    n_points = points.shape[0]
    if chunk_points is None:
        chunk_points = n_points

    outer_grad_enabled = torch.is_grad_enabled()
    chunks: list[torch.Tensor] = []

    for point_start in range(0, n_points, chunk_points):
        point_end = min(point_start + chunk_points, n_points)

        # We need gradients w.r.t. the evaluation coordinates to get H = -grad(phi).
        point_chunk = points[point_start:point_end].detach().clone().requires_grad_(True)

        with torch.enable_grad():
            phi = _scalar_potential_rectangular_prisms(
                magnet_br =magnet_br,
                magnet_size=magnet_size,
                magnet_pos=magnet_pos,
                points=point_chunk,
                eps=eps,
            )

            grad_phi = torch.autograd.grad(
                phi.sum(),
                point_chunk,
                create_graph=outer_grad_enabled,
                retain_graph=outer_grad_enabled,
            )[0]

            h_chunk = -grad_phi
            b_chunk = MU_0 * h_chunk

        chunks.append(b_chunk)

    return torch.cat(chunks, dim=0)

def calc_b_at_points_rect_prism_from_cube_volume(
    magnet_br: torch.Tensor,
    vol: torch.Tensor | float,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-12,
    chunk_points: int | None = 16384,
) -> torch.Tensor:
    """
    Convenience wrapper assuming every magnet is a cube with volume vol.
    """
    device = points.device
    dtype = points.dtype

    magnet_br = _as_tensor(magnet_br, dtype=dtype, device=device)

    if not isinstance(vol, torch.Tensor):
        vol = torch.tensor(vol, dtype=dtype, device=device)
    else:
        vol = vol.to(dtype=dtype, device=device)

    vol = vol.reshape(-1)

    if vol.numel() == 1:
        vol = vol.expand(magnet_br.shape[0])

    side = torch.pow(vol, 1.0 / 3.0)
    magnet_size = side[:, None].expand(-1, 3)

    return calc_b_at_points_rect_prism_potential(
        magnet_br=magnet_br,
        magnet_size=magnet_size,
        magnet_pos=magnet_pos,
        points=points,
        eps=eps,
        chunk_points=chunk_points,
    )
