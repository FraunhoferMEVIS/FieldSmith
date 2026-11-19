# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains region-of-interest helpers for field evaluation.
#
#  roi.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


def half_sphere_boundary_points(
    radius_m: float,
    z_min_m: float,
    n_points: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Even points on the boundary of a half sphere cut at z = z_min.

    Golden-angle spirals on the spherical cap and on the floor disk, split by
    area. The field extremes lie on the boundary, so training samples it.
    """
    if not (0.0 <= z_min_m < radius_m):
        raise ValueError("requires 0 <= z_min_m < radius_m.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    floor_radius = math.sqrt(radius_m**2 - z_min_m**2)
    cap_area = 2.0 * math.pi * radius_m * (radius_m - z_min_m)
    floor_area = math.pi * floor_radius**2
    n_cap = max(1, round(n_points * cap_area / (cap_area + floor_area)))
    n_floor = max(1, n_points - n_cap)

    index_cap = torch.arange(n_cap, dtype=dtype, device=device)
    z = z_min_m + (radius_m - z_min_m) * (index_cap + 0.5) / n_cap
    ring = torch.sqrt((radius_m**2 - z**2).clamp_min(0.0))
    phi = index_cap * golden
    cap = torch.stack((ring * torch.cos(phi), ring * torch.sin(phi), z), dim=1)

    index_floor = torch.arange(n_floor, dtype=dtype, device=device)
    r = floor_radius * torch.sqrt((index_floor + 0.5) / n_floor)
    phi = index_floor * golden
    floor = torch.stack((r * torch.cos(phi), r * torch.sin(phi),
                         torch.full_like(r, z_min_m)), dim=1)
    return torch.cat((cap, floor))


def load_evaluation_points(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Load a stored evaluation point set.

    Peak-to-peak depends on the exact points it is measured on, so reported
    numbers use a stored set rather than a regenerated one. The points are kept
    as float32 and returned in the requested precision.
    """
    with np.load(Path(path)) as archive:
        points = archive["points"]
    return torch.tensor(np.asarray(points, dtype=np.float64), dtype=dtype, device=device)


def roi_boundary_points(
    shape: str,
    a_m: float,
    z_min_m: float = 0.0,
    n_points: int = 3000,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Even, deterministic points on the boundary of a region.

    Shapes: a sphere of radius ``a_m`` and the half sphere above ``z_min_m``.
    """
    if shape == "half_sphere":
        return half_sphere_boundary_points(a_m, z_min_m, n_points, device, dtype)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    index = torch.arange(n_points, dtype=dtype, device=device)
    phi = index * golden
    if shape == "sphere":
        z = a_m * (2.0 * (index + 0.5) / n_points - 1.0)
        ring = torch.sqrt((a_m**2 - z**2).clamp_min(0.0))
        return torch.stack((ring * torch.cos(phi), ring * torch.sin(phi), z), dim=1)
    raise ValueError(f"unknown roi shape: {shape}")


def roi_mask_fn(shape: str, a_m: float, z_min_m: float = 0.0):
    """Boolean mask over (P, 3) points for the same regions."""
    def mask(points: torch.Tensor) -> torch.Tensor:
        radial2 = points[:, 0] ** 2 + points[:, 1] ** 2
        z = points[:, 2]
        if shape == "sphere":
            return radial2 + z**2 <= a_m**2
        if shape == "half_sphere":
            return (radial2 + z**2 <= a_m**2) & (z >= z_min_m)
        raise ValueError(f"unknown roi shape: {shape}")
    return mask
