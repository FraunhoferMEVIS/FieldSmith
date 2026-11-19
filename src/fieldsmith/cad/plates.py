# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the discrete-shell plate geometry: which magnets share a
# level, how tall each one stands, and whether a stack leaves room to build.
#
#  plates.py
#  Marian Frei
#  31.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

SQRT3 = 3.0 ** 0.5


def assign_planes(positions: torch.Tensor, tolerance: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Group magnets into the z levels a design actually occupies.

    Args:
        positions: Magnet centres in meters, shape (M, 3).
        tolerance: Centres closer than this in z belong to the same level.

    Returns:
        Level index per magnet, shape (M,), and the level heights, shape (L,).
    """
    z = positions[:, 2]
    order = torch.argsort(z)
    levels = [z[order[0]]]
    for index in order[1:]:
        if float(z[index] - levels[-1]) > tolerance:
            levels.append(z[index])
    heights = torch.stack(levels)
    return torch.argmin((z[:, None] - heights[None, :]).abs(), dim=1), heights


def z_extent(orientations: torch.Tensor, magnet_size: float) -> torch.Tensor:
    """
    Return how tall each cube stands in world z, in meters.

    In the pocket frame the height is size * (|cos t| + |sin t|) for a tilt t
    from vertical, which peaks at size * sqrt(2) at 45 degrees.

    Args:
        orientations: Magnetization directions, shape (M, 3); need not be normalized.
        magnet_size: Cube edge length in meters.

    Returns:
        World-z height per magnet in meters, shape (M,).
    """
    cosine = (orientations[:, 2] / orientations.norm(dim=1).clamp_min(1e-12)).abs()
    return magnet_size * (cosine + (1.0 - cosine * cosine).clamp_min(0.0).sqrt())


def plate_bounds(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    magnet_size: float,
    roof: float,
    floor: float,
    slab_gap: float = 0.0,
    max_floor: float = 6.0e-3,
) -> list[dict[str, float]]:
    """
    Return the plate each level would be printed as, after clamping.

    A plate reaches a roof above its tallest magnet and a floor below its
    lowest. The roof is served first. Where levels are tight the floor is
    clamped against the plate below and comes out thinner than asked.

    Args:
        positions: Magnet centres in meters, shape (M, 3).
        orientations: Magnetization directions, shape (M, 3).
        magnet_size: Cube edge length in meters.
        roof: Desired material above a pocket, in meters.
        floor: Desired material below a pocket, in meters.
        slab_gap: Air left between neighbouring plates, in meters.

    Returns:
        One dict per level with its height, z bounds, radii, magnet count, the
        roof and floor actually achieved, and any spacer still needed beneath
        it, all in meters.
    """
    labels, heights = assign_planes(positions)
    extent = z_extent(orientations, magnet_size)
    z = positions[:, 2]
    tops = torch.stack([(z + extent / 2)[labels == i].max() for i in range(len(heights))])
    bottoms = torch.stack([(z - extent / 2)[labels == i].min() for i in range(len(heights))])
    radius = positions[:, :2].norm(dim=1)
    plates: list[dict[str, float]] = []
    for level in range(len(heights)):
        top = float(tops[level]) + roof
        bottom = float(bottoms[level]) - floor
        spacer = 0.0
        if level > 0:
            # Reach down to the plate below so the stack touches, up to
            # max_floor. Any remaining gap is reported as a spacer.
            reachable = plates[-1]["z_max"] + slab_gap
            bottom = max(reachable, float(bottoms[level]) - max_floor)
            spacer = bottom - reachable
        here = labels == level
        plates.append({
            "height": float(heights[level]), "z_min": bottom, "z_max": top,
            "r_min": float(radius[here].min()) - magnet_size * SQRT3 / 2,
            "r_max": float(radius[here].max()) + magnet_size * SQRT3 / 2,
            "count": int(here.sum()), "roof": top - float(tops[level]),
            "floor": float(bottoms[level]) - bottom, "spacer_below": spacer,
        })
    return plates
