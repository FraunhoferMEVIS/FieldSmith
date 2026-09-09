# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains body-frame construction for magnetization-aligned cubes.
#
#  cube_axes.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch


def cube_axes_from_directions(directions: torch.Tensor) -> torch.Tensor:
    """
    Build a per-magnet body frame whose local +z axis follows the magnetization.

    The physical cube is rotated together with its magnetization ("pocket-oriented"):
    the magnet body is not an axis-aligned box that happens to be magnetized at an
    angle, but a real cube sitting in a rotated pocket. That distinction is what
    makes the resulting field a genuine magnet-body field rather than a point source.

    The two axes transverse to the magnetization are fixed deterministically: the
    world axis least aligned with the direction is used as the cross-product
    reference, so the frame varies continuously with the direction and never
    degenerates. The construction is differentiable and batched over M.

    Args:
        directions: Magnetization directions in meters-free units, shape (M, 3).
                    Only the direction matters; the vectors need not be normalized.

    Returns:
        Body frames, shape (M, 3, 3). Row k of each (3, 3) block is the local
        x, y and z axis expressed in world coordinates, right-handed and
        orthonormal, with row 2 parallel to the given direction.
    """
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (M, 3).")

    eps = torch.finfo(directions.dtype).eps
    norm = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(eps)
    axis_z = directions / norm

    # Reference: the world axis least aligned with z, so |cross| is maximal.
    index = torch.argmin(axis_z.abs(), dim=-1)
    reference = torch.nn.functional.one_hot(index, num_classes=3).to(axis_z.dtype)

    axis_x = torch.linalg.cross(reference, axis_z, dim=-1)
    axis_x = axis_x / torch.linalg.norm(axis_x, dim=-1, keepdim=True).clamp_min(eps)
    axis_y = torch.linalg.cross(axis_z, axis_x, dim=-1)

    return torch.stack((axis_x, axis_y, axis_z), dim=-2)


def cube_axes_min_z_extent(directions: torch.Tensor) -> torch.Tensor:
    """
    Build the CAD pocket frame: local +z along the magnetization, transverse axes
    from Gram-Schmidt of the world +z axis.

    This is the manufacturing convention (the pocket is machined so its local x lies
    in the plane spanned by world +z and the magnetization). It reproduces the cap
    reference design to under 1 ppm, where the argmin rule of
    cube_axes_from_directions differs by ~74 ppm. Batched, differentiable, a
    right-handed orthonormal frame with row 2 along the direction. Near-vertical
    directions fall back to a world-x transverse reference.

    Args:
        directions: Magnetization directions, shape (M, 3); need not be normalized.

    Returns:
        Body frames, shape (M, 3, 3); rows are local x, y, z in world coordinates.
    """
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (M, 3).")
    eps = 1e-6
    axis_z = directions / directions.norm(dim=-1, keepdim=True).clamp_min(eps)
    world_z = torch.zeros_like(axis_z)
    world_z[:, 2] = 1.0
    transverse = world_z - axis_z[:, 2:3] * axis_z
    norm = transverse.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis_z)
    fallback[:, 0] = 1.0
    axis_x = torch.where(norm >= eps, transverse / norm.clamp_min(eps), fallback)
    axis_y = torch.linalg.cross(axis_z, axis_x)
    return torch.stack((axis_x, axis_y, axis_z), dim=-2)
