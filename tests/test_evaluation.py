# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Tests for evaluating a configuration under an explicitly chosen physics.
#
#  test_evaluation.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import copy

import pytest
import torch

from fieldsmith.evaluation import evaluate_configuration, field_at_points, truth_eval_report
from fieldsmith.geometry.dome import DomeGeometry

DTYPE = torch.float64


def _dome():
    return DomeGeometry(0.06, 0.17, 0.19, 6, 0.012, 1.3, 0.0135, "cpu", DTYPE).build()


def _points(n: int = 120) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(n, 3, generator=generator, dtype=DTYPE)
    points = points / points.norm(dim=1, keepdim=True)
    radii = 0.04 * torch.rand(n, 1, generator=generator, dtype=DTYPE) ** (1.0 / 3.0)
    return points * radii + torch.tensor([0.0, 0.0, 0.05], dtype=DTYPE)


def test_demagnetisation_changes_the_verdict() -> None:
    configuration, points = _dome(), _points()
    ideal = evaluate_configuration(configuration, points, "cuboid", 0.0, 0.0)
    built = evaluate_configuration(configuration, points, "cuboid", 0.05, 0.10)
    # The correction is real: it lowers the field of the array it is applied to.
    assert built["fieldstrength_mT"] < ideal["fieldstrength_mT"]
    assert built["peak_to_peak_ppm"] != ideal["peak_to_peak_ppm"]


def test_dipole_and_cuboid_agree_in_the_far_field() -> None:
    configuration = _dome()
    far = torch.tensor([[0.0, 0.0, 1.5]], dtype=DTYPE)
    assert torch.allclose(field_at_points(configuration, far, "dipole"),
                          field_at_points(configuration, far, "cuboid"), rtol=1e-4)


def test_frame_rule_selects_different_physics() -> None:
    # A rotated cube's field depends on its body frame, so the two conventions
    # are not interchangeable and the caller has to choose deliberately.
    configuration, points = _dome(), _points()
    argmin = evaluate_configuration(configuration, points, "cuboid", 0.05, 0.10, frame_rule="argmin")
    pocket = evaluate_configuration(configuration, points, "cuboid", 0.05, 0.10, frame_rule="gs")
    assert argmin["peak_to_peak_ppm"] != pocket["peak_to_peak_ppm"]


def test_truth_report_covers_both_physics_and_the_overlap_gate() -> None:
    configuration, points = _dome(), _points()
    report = truth_eval_report(configuration, points)
    assert set(report) == {"chi0", "demag", "overlaps", "contacts", "tilt_deg"}
    assert report["overlaps"] == 0                       # the builder spaces magnets apart
    assert report["demag"]["fieldstrength_mT"] < report["chi0"]["fieldstrength_mT"]


def test_overlap_gate_notices_a_broken_arrangement() -> None:
    configuration = _dome()
    moved = copy.copy(configuration)
    moved.positions = configuration.positions.clone()
    moved.positions[1] = moved.positions[0]              # two magnets in the same place
    assert truth_eval_report(moved, _points())["overlaps"] > 0


def test_truth_eval_reports_contacts_beside_the_gate() -> None:
    # Two cubes 0.2 mm interpenetrated: inside the 0.4 mm acceptance, so no
    # overlap -- but they touch, and the printable plates need to know.
    moved = _dome().detached_cpu()
    moved.positions = torch.tensor([[0.0, 0.0, 0.0], [0.0118, 0.0, 0.0]], dtype=DTYPE)
    moved.orientations = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
    moved.volumes = torch.full((2,), 0.012 ** 3, dtype=DTYPE)
    moved.metadata["magnet_size_m"] = 0.012
    report = truth_eval_report(moved, _points())
    assert report["overlaps"] == 0 and report["contacts"] == 1


def _roma_slice():
    """Six cubes of ROMA's first ring and six 12 x 12 x 50 mm bars of its end ring."""
    from pathlib import Path

    from fieldsmith.configurations import MagnetConfiguration
    from fieldsmith.geometry.roma import RomaGeometry

    rings = Path(__file__).resolve().parents[1] / "data" / "osii_rc2.1"
    built = RomaGeometry(rings, 1.31, device="cpu", dtype=DTYPE).build()
    index = built.metadata["ring_indices"]
    keep = torch.cat((torch.nonzero(index == 0).flatten()[:6], torch.nonzero(index == 8).flatten()[:6]))
    return MagnetConfiguration(positions=built.positions[keep], orientations=built.orientations[keep],
                               volumes=built.volumes[keep], remanence=1.31,
                               metadata={"magnet_size_m": 0.012,
                                         "magnet_dimensions_m": built.metadata["magnet_dimensions_m"][keep]})


def test_mixed_model_is_the_cuboid_model_with_cubes_as_dipoles() -> None:
    configuration, points = _roma_slice(), _points()
    mixed = field_at_points(configuration, points, "mixed", frame_rule="world")
    cuboid = field_at_points(configuration, points, "cuboid", frame_rule="world")
    assert not torch.equal(mixed, cuboid)
    assert float(((mixed - cuboid).norm(dim=1) / cuboid.norm(dim=1)).max()) < 1e-3


def test_seated_frame_turns_bars_only_about_the_bore() -> None:
    configuration, points = _roma_slice(), _points()
    world = field_at_points(configuration, points, "cuboid", frame_rule="world")
    seated = field_at_points(configuration, points, "cuboid", frame_rule="seated")
    assert float((world - seated).abs().max()) > 0.0      # small at this distance, but a different body
    for frame in ("world", "seated"):
        report = truth_eval_report(configuration, points, 0.0, 0.0, frame_rule=frame)
        assert report["overlaps"] == 0 and report["contacts"] == 0 and report["tilt_deg"] == 0.0


def test_truth_report_flags_a_tilted_bar() -> None:
    configuration, points = _roma_slice(), _points()
    tilt = torch.deg2rad(torch.tensor(10.0, dtype=DTYPE))
    direction = configuration.orientations[-1]
    configuration.orientations[-1] = torch.cat((direction[:2] * torch.cos(tilt), torch.sin(tilt).reshape(1)))
    report = truth_eval_report(configuration, points, 0.0, 0.0, frame_rule="world")
    assert report["tilt_deg"] == pytest.approx(10.0)
