# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Tests for the head-scanner envelope rule and its lattice.
#
#  test_head_lattice.py
#  Marian Frei
#  06.08.2026
# -----------------------------------------------------------------------------

import math

import torch

from fieldsmith.geometry.cap_lattice import CapLatticeGeometry, head_envelope_mm
from fieldsmith.geometry.collision import box_penetration, candidate_pairs

BOX_HALF_MM = 155.0
MARGIN_MM = 10.4  # full half space-diagonal of a 12 mm cube, tilt-safe in 3-D
PITCH_MM = 16.5

# The design point of the 2026-08-06 head-scanner spec, pinned.
EXPECTED_ENVELOPE = (
    (10.0, 125.4, 144.2),
    (30.0, 123.4, 141.4),
    (50.0, 117.0, 135.6),
    (70.0, 105.1, 126.5),
    (90.0, 85.4, 113.1),
    (110.0, 46.9, 93.8),
    (130.0, 11.0, 63.3),
)


def build_head_lattice() -> CapLatticeGeometry:
    return CapLatticeGeometry(envelope_mm=head_envelope_mm(), ring_pitch_mm=PITCH_MM,
                              magnet_size=0.012, remanence=1.31)


def test_envelope_matches_the_pinned_design_point() -> None:
    assert head_envelope_mm() == EXPECTED_ENVELOPE


def test_lattice_size_and_outline_stay_inside_the_budget_box() -> None:
    config = build_head_lattice().build()
    assert config.num_magnets == 582
    positions_mm = config.positions * 1e3
    # Worst case: a cube tilted so its half-diagonal points along an axis.
    assert float(positions_mm[:, 0].abs().max()) + MARGIN_MM <= BOX_HALF_MM
    assert float(positions_mm[:, 1].abs().max()) + MARGIN_MM <= BOX_HALF_MM
    assert float(positions_mm[:, 2].max()) + MARGIN_MM <= BOX_HALF_MM
    assert float(positions_mm[:, 2].min()) - MARGIN_MM >= 0.0 - MARGIN_MM


def test_every_plate_holds_at_least_two_radial_rings() -> None:
    for _, r_lo, r_hi in head_envelope_mm():
        assert int(math.floor((r_hi - r_lo) / PITCH_MM + 1e-9)) + 1 >= 2


def test_no_cube_corner_enters_the_head_keep_out() -> None:
    config = build_head_lattice().build()
    positions_mm = config.positions * 1e3
    radial = positions_mm[:, :2].norm(dim=1)
    # The corner nearest the bore, for an axis-aligned or tilted cube, lies at
    # most the half-diagonal closer in both the radial and z direction.
    corner_r = (radial - MARGIN_MM).clamp_min(0.0)
    corner_z = (positions_mm[:, 2] - MARGIN_MM).clamp_min(0.0)
    outside = (corner_r / 115.0) ** 2 + (corner_z / 105.0) ** 2
    assert float(outside.min()) >= 1.0 - 1e-6


def test_packing_stays_in_the_legalisable_regime() -> None:
    """
    As built, with placeholder +z orientations, diagonal neighbours may
    interpenetrate slightly -- exactly as the proven reference lattice does at
    the same pitch, where the campaign optimised that to zero. What must hold
    is the packing rule itself and that no pair starts deeper than the
    legalisation the reference campaign demonstrated (~1 mm).
    """
    config = build_head_lattice().build()
    distances = torch.cdist(config.positions, config.positions)
    distances += torch.eye(config.num_magnets, dtype=config.positions.dtype) * 1e3
    assert float(distances.min()) >= 0.0135
    axes = torch.eye(3, dtype=config.positions.dtype).expand(config.num_magnets, 3, 3)
    index_i, index_j = candidate_pairs(config.positions, reach=2.0 * 0.012 * math.sqrt(3.0))
    depth = box_penetration(config.positions[index_i], config.positions[index_j],
                            axes[index_i], axes[index_j], half_extent=0.006)
    assert float(depth.max()) <= 1.05e-3
