# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Tests for region-of-interest helpers.
#
#  test_roi.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fieldsmith.geometry.roi import (half_sphere_boundary_points, load_evaluation_points, roi_boundary_points,
                                     roi_mask_fn)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_dome_region_is_the_documented_shape() -> None:
    # The pinned set every dome number is read on: a sphere of 75 mm radius cut
    # at z = 10 mm, sampled at 1.25 mm, 367,685 points.
    points = load_evaluation_points(DATA / "head_roi.npz")
    assert points.shape == (367685, 3)
    assert points.dtype == torch.float64
    assert bool((points[:, 2] >= 0.010 - 1e-6).all())
    assert bool((points.norm(dim=1) <= 0.075 + 1e-6).all())


def test_boundary_points_lie_on_the_roi_boundary() -> None:
    points = half_sphere_boundary_points(0.075, 0.010, 3000)
    assert points.shape == (3000, 3)
    radius = points.norm(dim=1)
    on_cap = torch.isclose(radius, torch.full_like(radius, 0.075), atol=1e-9)
    on_floor = torch.isclose(points[:, 2], torch.full_like(radius, 0.010), atol=1e-9)
    assert bool((on_cap | on_floor).all())
    assert bool((points[:, 2] >= 0.010 - 1e-12).all()) and bool((radius <= 0.075 + 1e-12).all())
    # Deterministic: the same call gives the same points.
    assert torch.equal(points, half_sphere_boundary_points(0.075, 0.010, 3000))


def test_sphere_region_and_its_boundary() -> None:
    points = roi_boundary_points("sphere", a_m=0.1, n_points=500)
    assert torch.allclose(points.norm(dim=1), torch.full((500,), 0.1, dtype=torch.float64))
    probe = torch.tensor([[0.0, 0.0, 0.09], [0.0, 0.0, 0.11], [0.0, 0.0, -0.09]], dtype=torch.float64)
    assert roi_mask_fn("sphere", a_m=0.1)(probe).tolist() == [True, False, True]
    assert roi_mask_fn("half_sphere", a_m=0.1, z_min_m=0.0)(probe).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        roi_boundary_points("cube", a_m=0.1)
    with pytest.raises(ValueError):
        roi_mask_fn("cube", a_m=0.1)(probe)
