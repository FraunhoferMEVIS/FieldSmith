# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Discrete-shell plate geometry: levels, cube heights and buildability.
#
#  test_plates.py
#  Marian Frei
#  31.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry.cube_axes import cube_axes_min_z_extent
from fieldsmith.cad.plates import assign_planes, plate_bounds, z_extent

DATA = Path(__file__).resolve().parents[1] / "data"
CUBE = 0.012


@pytest.fixture(scope="module")
def cap():
    """The dome design reported in the paper, eleven plates of 12 mm cubes."""
    configuration = MagnetConfiguration.from_file(DATA / "paper" / "dome_980ppm.pt")
    return configuration.positions.double(), configuration.orientations.double()


def test_z_extent_matches_the_corners_of_the_rotated_body() -> None:
    # The closed form is what the compliance term differentiates; it has to agree
    # with the frame the pocket is actually machined to, not approximately.
    torch.manual_seed(0)
    directions = torch.randn(500, 3, dtype=torch.float64)
    axes = cube_axes_min_z_extent(directions)
    corners = torch.tensor(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=torch.float64)
    world = torch.einsum("cj,mjk->mck", corners * CUBE, axes)
    measured = world[..., 2].amax(dim=1) - world[..., 2].amin(dim=1)
    assert torch.allclose(measured, z_extent(directions, CUBE), atol=1e-12)


def test_a_tilted_cube_is_taller_than_its_edge() -> None:
    # 41 percent taller at 45 degrees, which is the whole reason plates run out
    # of room; a vertical or horizontal magnet costs only its edge length.
    angles = torch.tensor([0.0, torch.pi / 4, torch.pi / 2], dtype=torch.float64)
    directions = torch.stack((torch.sin(angles), torch.zeros(3, dtype=torch.float64), torch.cos(angles)), dim=1)
    extents = z_extent(directions, CUBE)
    assert extents[0] == pytest.approx(CUBE)
    assert extents[1] == pytest.approx(CUBE * 2 ** 0.5)
    assert extents[2] == pytest.approx(CUBE)
    assert extents[1] == extents.max()


def test_the_reference_design_sits_on_eleven_levels(cap) -> None:
    positions, _ = cap
    labels, heights = assign_planes(positions)
    assert len(heights) == 11
    assert torch.all(torch.diff(heights) > 0.015)               # levels, not a smear
    assert torch.bincount(labels).sum() == len(positions)


def test_plates_serve_the_roof_first_and_leave_no_dead_air(cap) -> None:
    # Splitting the difference between two levels would reserve room above that
    # nothing uses while starving the floor below it by about a millimetre.
    positions, orientations = cap
    plates = plate_bounds(positions, orientations, CUBE, roof=3.0e-4, floor=2.0e-3)
    assert len(plates) == 11
    assert all(plate["roof"] == pytest.approx(3.0e-4) for plate in plates)
    # A floor never falls short of the request and grows only into space that
    # would otherwise be dead air between two plates, up to its 6 mm cap.
    assert all(2.0e-3 - 1e-12 <= plate["floor"] <= 6.0e-3 + 1e-12 for plate in plates)
    assert all(lower["z_max"] <= upper["z_min"] + 1e-12 for lower, upper in zip(plates, plates[1:]))


def test_a_tight_design_thins_its_floor_rather_than_failing(cap) -> None:
    # The exporter clamps instead of refusing, so an over-ambitious floor is not
    # an error: it silently comes out thinner, and that has to be reported.
    positions, orientations = cap
    plates = plate_bounds(positions, orientations, CUBE, roof=3.0e-4, floor=4.0e-3)
    floors = [plate["floor"] for plate in plates]
    assert min(floors) < 4.0e-3
    assert min(floors) > 0.0                                    # never a through-hole
    assert all(lower["z_max"] <= upper["z_min"] + 1e-12 for lower, upper in zip(plates, plates[1:]))


def test_a_slab_gap_pushes_the_plates_apart(cap) -> None:
    positions, orientations = cap
    plates = plate_bounds(positions, orientations, CUBE, roof=3.0e-4, floor=2.0e-3, slab_gap=1.0e-3)
    assert all(upper["z_min"] - lower["z_max"] >= 1.0e-3 - 1e-12
               for lower, upper in zip(plates, plates[1:]))
    assert min(plate["floor"] for plate in plates) < 2.0e-3      # the gap is paid for by the floor
