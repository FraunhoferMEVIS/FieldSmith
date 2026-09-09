# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the exact cuboid magnetic-field kernel with body rotation.
#
#  cuboid.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

# Sign pattern of the eight cuboid vertices, shared by all three cyclic terms.
_SIGNS = (
    (-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
)


def _as_tensor(x, *, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def _field_z_magnetised(
    rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    br_z: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Closed-form field of cuboids magnetized along their local z-axis.

    Args:
        rx, ry, rz: Displacements from magnet centers to points, each (P, M).
        a, b, c: Half side lengths along x, y and z, each (M,).
        br_z: Remanence along the local z-axis in tesla, shape (M,).
        eps: Regularization of the removable log/atan singularities.

    Returns:
        The local x, y and z field components in tesla, each of shape (P, M).
    """
    sa, sb, sc = torch.tensor(_SIGNS, dtype=rx.dtype, device=rx.device)

    # (P, M, 1) against (M, 8) -> (P, M, 8), one slot per cuboid vertex.
    dx = rx[..., None] - sa * a[:, None]
    dy = ry[..., None] - sb * b[:, None]
    dz = rz[..., None] - sc * c[:, None]
    r = torch.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
    pre = (-sa * sb * sc / (4.0 * torch.pi)) * br_z[:, None]
    return (
        (pre * torch.log(dy + r + eps)).sum(-1),
        (pre * torch.log(dx + r + eps)).sum(-1),
        (-pre * torch.atan2(dx * dy, dz * r + eps)).sum(-1),
    )


def _field_per_magnet(
    magnet_br: torch.Tensor, half: torch.Tensor, rel: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Per-magnet field (P, M, 3) in the frame rel and magnet_br are given in, as the
    z-magnetized solution plus its two cyclic permutations.
    """
    x, y, z = rel[..., 0], rel[..., 1], rel[..., 2]
    a, b, c = half[:, 0], half[:, 1], half[:, 2]
    bxz, byz, bzz = _field_z_magnetised(x, y, z, a, b, c, magnet_br[:, 2], eps)
    byx, bzx, bxx = _field_z_magnetised(y, z, x, b, c, a, magnet_br[:, 0], eps)
    bzy, bxy, byy = _field_z_magnetised(z, x, y, c, a, b, magnet_br[:, 1], eps)

    return torch.stack((bxx + bxy + bxz, byx + byy + byz, bzx + bzy + bzz), dim=-1)


def calc_b_at_points_cuboid(
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    axes: torch.Tensor | None = None,
    eps: float = 1e-20,
    chunk_points: int | None = 4096,
) -> torch.Tensor:
    """
    Compute the exact B-field of uniformly magnetized cuboids, optionally rotated.

    Args:
        magnet_br: World-frame remanence vectors in tesla, shape (M, 3); with axes
            given they are projected onto the body frame internally.
        magnet_size: Full side lengths in meters, shape (M, 3), measured along the
            body axes when axes is given.
        magnet_pos: Cuboid centers in meters, shape (M, 3).
        points: Evaluation points in meters, shape (P, 3).
        axes: Optional body frames, shape (M, 3, 3), as returned by
            cube_axes_from_directions; row k of a block is local axis k in world
            coordinates. None means world-axis-aligned cuboids, matching the
            convention of calc_b_at_points_rect_prism_potential.
        eps: Numerical regularization for removable log/atan singularities. Keep it
            far below the field scale: it is added inside log(dy + r), where the
            argument cancels to near zero for points roughly on a face normal, so a
            larger value such as 1e-12 costs four digits of accuracy there.
        chunk_points: Number of evaluation points per chunk.

    Returns:
        B-field in tesla, shape (P, 3).

    Notes:
        Evaluated directly rather than by differentiating a scalar potential, so it
        is cheap and builds no autograd graph over the evaluation coordinates. It
        returns B outside the magnets; inside one, B = mu0 * (H + M) adds Br.
    """
    device, dtype = points.device, points.dtype

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
    if axes is not None:
        axes = _as_tensor(axes, dtype=dtype, device=device)
        if axes.shape != (magnet_br.shape[0], 3, 3):
            raise ValueError("axes must have shape (M, 3, 3).")
        magnet_br = torch.einsum("mkj,mj->mk", axes, magnet_br)

    half = 0.5 * magnet_size
    n_points = points.shape[0]
    if chunk_points is None:
        chunk_points = n_points

    chunks: list[torch.Tensor] = []
    for start in range(0, n_points, chunk_points):
        rel = points[start:start + chunk_points, None, :] - magnet_pos[None, :, :]

        if axes is None:
            chunks.append(_field_per_magnet(magnet_br, half, rel, eps).sum(dim=1))
        else:
            rel = torch.einsum("mkj,pmj->pmk", axes, rel)
            local = _field_per_magnet(magnet_br, half, rel, eps)
            chunks.append(torch.einsum("mkj,pmk->pj", axes, local))

    return torch.cat(chunks, dim=0)
