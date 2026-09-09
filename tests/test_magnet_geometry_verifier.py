from argparse import Namespace
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from fieldsmith import magnet_geometry_verifier as mgv
from fieldsmith.configurations import MagnetConfiguration


@pytest.fixture
def configuration() -> MagnetConfiguration:
    return MagnetConfiguration(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        orientations=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        volumes=torch.ones(2, dtype=torch.float64),
        remanence=1.3,
        metadata={"magnet_size_m": 0.012},
    )


def test_parse_args_reads_all_command_line_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_magnet_geometry",
            "configuration.pt",
            "--dimensions",
            "0.01",
            "0.02",
            "0.03",
            "--frame-rule",
            "gs",
            "--touching-counts-as-intersection",
            "--max-allowed-intersection-mm",
            "0.5",
            "--log-file",
            "report.json",
            "--visualization-directory",
            "images",
        ],
    )

    args = mgv.parse_args()

    assert args.config == Path("configuration.pt")
    assert args.dimensions == [0.01, 0.02, 0.03]
    assert args.frame_rule == "gs"
    assert args.touching_counts_as_intersection
    assert args.max_allowed_intersection_mm == 0.5
    assert args.log_file == Path("report.json")
    assert args.visualization_directory == Path("images")


def test_parse_args_prompts_for_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["check_magnet_geometry"])
    monkeypatch.setattr("builtins.input", lambda _: "prompted.npy")

    args = mgv.parse_args()

    assert args.config == Path("prompted.npy")


@pytest.mark.parametrize(
    "arguments",
    [
        ["configuration.pt", "--dimensions", "1", "1", "1", "--cube-size-mm", "12"],
        ["configuration.pt", "--max-allowed-intersection-mm", "-1"],
        ["configuration.pt", "--cube-size-mm", "0"],
    ],
)
def test_parse_args_rejects_invalid_option_combinations(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["check_magnet_geometry", *arguments])

    with pytest.raises(SystemExit):
        mgv.parse_args()


def test_load_configuration_selects_format_loader(
    monkeypatch: pytest.MonkeyPatch,
    configuration: MagnetConfiguration,
) -> None:
    npy_loader = Mock(return_value=configuration)
    checkpoint_loader = Mock(return_value=configuration)
    monkeypatch.setattr(mgv, "load_halbach_setup_npy", npy_loader)
    monkeypatch.setattr(mgv.MagnetConfiguration, "from_file", checkpoint_loader)

    assert mgv._load_configuration(Path("setup.NPY")) is configuration
    assert mgv._load_configuration(Path("setup.pt")) is configuration

    npy_loader.assert_called_once_with(Path("setup.NPY"), dtype=torch.float64)
    checkpoint_loader.assert_called_once_with(Path("setup.pt"), dtype=torch.float64)


def test_dimensions_from_configuration_uses_override(
    configuration: MagnetConfiguration,
) -> None:
    dimensions = mgv._dimensions_from_configuration(
        configuration,
        [0.01, 0.02, 0.03],
    )

    assert torch.equal(
        dimensions,
        torch.tensor(
            [[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]],
            dtype=torch.float64,
        ),
    )


def test_dimensions_from_configuration_uses_per_magnet_metadata(
    configuration: MagnetConfiguration,
) -> None:
    configuration.metadata = {
        "magnet_dimensions_m": [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]
    }

    dimensions = mgv._dimensions_from_configuration(configuration, None)

    assert dimensions.dtype == configuration.positions.dtype
    assert torch.equal(
        dimensions,
        torch.tensor(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
            dtype=torch.float64,
        ),
    )


def test_dimensions_from_configuration_uses_scalar_metadata(
    configuration: MagnetConfiguration,
) -> None:
    dimensions = mgv._dimensions_from_configuration(configuration, None)

    assert torch.equal(
        dimensions,
        torch.full((2, 3), 0.012, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, 'configuration metadata needs "magnet_dimensions_m"'),
        ({"magnet_dimensions_m": [0.01, 0.02, 0.03]}, r"must have shape \(M, 3\)"),
        (
            {"magnet_dimensions_m": [[0.01, 0.02, 0.03], [0.01, 0.0, 0.03]]},
            "magnet dimensions must be positive",
        ),
    ],
)
def test_dimensions_from_configuration_rejects_invalid_metadata(
    configuration: MagnetConfiguration,
    metadata: dict[str, object],
    message: str,
) -> None:
    configuration.metadata = metadata

    with pytest.raises(ValueError, match=message):
        mgv._dimensions_from_configuration(configuration, None)


def test_load_config_returns_stable_dictionary(
    monkeypatch: pytest.MonkeyPatch,
    configuration: MagnetConfiguration,
) -> None:
    monkeypatch.setattr(mgv, "_load_configuration", Mock(return_value=configuration))

    result = mgv.load_config(Path("configuration.pt"), [0.01, 0.02, 0.03])

    assert result["configuration"] is configuration
    assert result["magnet_positions_m"] is configuration.positions
    assert result["magnet_orientations"] is configuration.orientations
    assert result["magnet_dimensions_m"].shape == (2, 3)


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros(3),
        np.zeros((2, 2)),
    ],
)
def test_center_distance_stats_rejects_invalid_shapes(positions: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"positions_m must have shape \(M, 3\)"):
        mgv.center_distance_stats(positions)


def test_center_distance_stats_handles_single_magnet() -> None:
    result = mgv.center_distance_stats(np.array([[1.0, 2.0, 3.0]]))

    assert result["min_center_distance_m"] is None
    assert result["pair_for_min_center_distance"] is None
    for axis in "xyz":
        assert result[f"min_abs_d{axis}_m"] is None
        assert result[f"min_nonzero_abs_d{axis}_m"] is None
        assert result[f"zero_d{axis}_pair_count"] == 0
        assert result[f"example_zero_d{axis}_pair"] is None


def test_center_distance_stats_reports_zero_and_nonzero_axis_pairs() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 5.0],
        ]
    )

    result = mgv.center_distance_stats(positions)

    assert result["min_center_distance_m"] == pytest.approx(5.0**0.5)
    assert result["zero_dx_pair_count"] == 3
    assert result["min_nonzero_abs_dx_m"] is None
    assert result["pair_for_min_nonzero_abs_dx"] is None
    assert result["example_zero_dx_pair"] == (0, 1)
    assert result["min_nonzero_abs_dy_m"] == pytest.approx(1.0)


def test_find_intersections_handles_gs_frame_and_no_intersections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64)
    orientations = torch.ones((2, 3), dtype=torch.float64)
    dimensions = torch.ones((2, 3), dtype=torch.float64)
    axes = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    axes_builder = Mock(return_value=axes)
    intersection_finder = Mock(
        return_value=(
            torch.empty((0, 2), dtype=torch.long),
            torch.empty(0, dtype=torch.float64),
        )
    )
    monkeypatch.setattr(mgv, "cube_axes_min_z_extent", axes_builder)
    monkeypatch.setattr(mgv, "find_box_intersections", intersection_finder)

    result = mgv.find_intersections(
        positions,
        orientations,
        dimensions,
        frame_rule="gs",
        touching_counts=True,
    )

    axes_builder.assert_called_once_with(orientations)
    intersection_finder.assert_called_once_with(
        positions,
        axes,
        dimensions,
        touching_counts=True,
    )
    assert result == {
        "intersection_count": 0,
        "has_intersections": False,
        "intersections": [],
        "max_intersection_depth_m": 0.0,
        "max_intersection_depth_mm": 0.0,
        "max_intersection_pair": None,
    }


def test_min_center_axis_distances_reports_meters_and_millimeters() -> None:
    positions = np.array([
        [0.000, 0.000, 0.000],
        [0.013, 0.004, 0.008],
        [0.020, 0.001, 0.002],
    ])

    result = mgv.min_center_axis_distances(positions)

    assert result["min_abs_dx_m"] == pytest.approx(0.007)
    assert result["min_abs_dy_mm"] == pytest.approx(1.0)
    assert result["min_abs_dz_mm"] == pytest.approx(2.0)


def test_find_intersections_uses_rotations_and_cuboid_dimensions() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.011, 0.0, 0.0]], dtype=torch.float64)
    orientations = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    dimensions = torch.full((2, 3), 0.012, dtype=torch.float64)

    result = mgv.find_intersections(positions, orientations, dimensions, "argmin", False)

    assert result["intersection_count"] == 1
    assert result["max_intersection_depth_mm"] == pytest.approx(1.0)


def test_build_report_classifies_minor_intersection_as_warning() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0115, 0.0, 0.0]], dtype=torch.float64)
    config = {
        "magnet_positions_m": positions,
        "magnet_orientations": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ),
        "magnet_dimensions_m": torch.full((2, 3), 0.012, dtype=torch.float64),
    }
    args = Namespace(
        config=Path("configuration.pt"), cube_size_mm=None, frame_rule="argmin",
        touching_counts_as_intersection=False, max_allowed_intersection_mm=1.0,
    )

    report = mgv.build_report(args, config)

    assert report["geometry_status"] == "warning"


@pytest.mark.parametrize(
    ("report", "maximum", "status", "should_fail", "warning_only"),
    [
        (
            {"has_intersections": False, "max_intersection_depth_mm": 0.0},
            1.0,
            "ok",
            False,
            False,
        ),
        (
            {"has_intersections": True, "max_intersection_depth_mm": 0.5},
            1.0,
            "warning",
            False,
            True,
        ),
        (
            {"has_intersections": True, "max_intersection_depth_mm": 1.5},
            1.0,
            "error",
            True,
            False,
        ),
    ],
)
def test_classify_intersection_severity(
    report: dict[str, object],
    maximum: float,
    status: str,
    should_fail: bool,
    warning_only: bool,
) -> None:
    result = mgv.classify_intersection_severity(report, maximum)

    assert result["geometry_status"] == status
    assert result["should_fail"] is should_fail
    assert result["warning_only"] is warning_only
    assert result["message"]


def test_build_report_applies_cube_size_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    config = {
        "magnet_positions_m": positions,
        "magnet_orientations": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        "magnet_dimensions_m": torch.full((1, 3), 0.02, dtype=torch.float64),
    }
    intersection_report = {
        "intersection_count": 0,
        "has_intersections": False,
        "intersections": [],
        "max_intersection_depth_m": 0.0,
        "max_intersection_depth_mm": 0.0,
        "max_intersection_pair": None,
    }
    find_intersections = Mock(return_value=intersection_report)
    monkeypatch.setattr(mgv, "find_intersections", find_intersections)
    args = Namespace(
        config=Path("configuration.pt"),
        cube_size_mm=12.0,
        frame_rule="gs",
        touching_counts_as_intersection=True,
        max_allowed_intersection_mm=1.0,
    )

    report = mgv.build_report(args, config)

    expected_dimensions = torch.full((1, 3), 0.012, dtype=torch.float64)
    assert np.array_equal(report["magnet_dimensions_m"], expected_dimensions.numpy())
    find_intersections.assert_called_once()
    assert torch.equal(find_intersections.call_args.args[2], expected_dimensions)
    assert report["geometry_status"] == "ok"


def test_format_distance_handles_missing_and_available_values() -> None:
    assert mgv._format_distance("distance", None, None) == "  distance: not available"
    assert (
        mgv._format_distance("distance", 1.2345678, (1, 2))
        == "  distance = 1.234568 mm, pair = (1, 2)"
    )


def test_print_report_outputs_intersections_and_maximum_pair(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "config_file": "configuration.pt",
        "magnet_count": 2,
        "geometry_mode": "oriented_cuboids_from_magnetization_argmin",
        "touching_counts_as_intersection": False,
        "min_center_distance_mm": 11.0,
        "pair_for_min_center_distance": (0, 1),
        "min_nonzero_abs_dx_mm": 11.0,
        "pair_for_min_nonzero_abs_dx": (0, 1),
        "min_nonzero_abs_dy_mm": None,
        "pair_for_min_nonzero_abs_dy": None,
        "min_nonzero_abs_dz_mm": None,
        "pair_for_min_nonzero_abs_dz": None,
        "zero_dx_pair_count": 0,
        "zero_dy_pair_count": 1,
        "zero_dz_pair_count": 1,
        "intersection_count": 1,
        "intersections": [
            {
                "pair": [0, 1],
                "center_distance_mm": 11.0,
                "penetration_depth_mm": 1.0,
            }
        ],
        "geometry_status": "warning",
        "message": "Minor intersection.",
        "max_intersection_pair": [0, 1],
        "max_intersection_depth_mm": 1.0,
    }

    mgv.print_report(report)

    output = capsys.readouterr().out
    assert "=== Magnet Geometry Check ===" in output
    assert "min non-zero |dy|: not available" in output
    assert "pair (0, 1): center distance 11.000000 mm" in output
    assert "Geometry status: WARNING" in output
    assert "Maximum penetration: 1.000000 mm, pair = (0, 1)" in output


def test_box_vertices_and_faces_describe_axis_aligned_box() -> None:
    vertices = mgv._box_vertices(
        centre=np.array([1.0, 2.0, 3.0]),
        axes=np.eye(3),
        dimensions=np.array([2.0, 4.0, 6.0]),
    )
    faces = mgv._box_faces(vertices)

    assert vertices.shape == (8, 3)
    assert np.array_equal(vertices.min(axis=0), np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(vertices.max(axis=0), np.array([2.0, 4.0, 6.0]))
    assert len(faces) == 6
    assert all(face.shape == (4, 3) for face in faces)


def test_set_equal_3d_limits_sets_symmetric_ranges() -> None:
    axis = Mock()
    vertices = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])

    mgv._set_equal_3d_limits(axis, vertices, padding=0.0)

    axis.set_xlim.assert_called_once_with(-3.0, 3.0)
    axis.set_ylim.assert_called_once_with(-3.0, 3.0)
    axis.set_zlim.assert_called_once_with(-3.0, 3.0)
    axis.set_box_aspect.assert_called_once_with((1, 1, 1))


def test_set_equal_3d_limits_handles_zero_extent() -> None:
    axis = Mock()
    vertices = np.zeros((8, 3))

    mgv._set_equal_3d_limits(axis, vertices)

    lower, upper = axis.set_xlim.call_args.args
    assert lower < 0.0 < upper


def test_save_intersection_visualizations_creates_pair_image() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.011, 0.0, 0.0]], dtype=torch.float64)
    config = {
        "magnet_positions_m": positions,
        "magnet_orientations": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ),
        "magnet_dimensions_m": torch.full((2, 3), 0.012, dtype=torch.float64),
    }
    report = mgv.find_intersections(
        positions, config["magnet_orientations"], config["magnet_dimensions_m"], "argmin", False
    )

    output_directory = Path("tests") / "generated_intersection_visualization"
    try:
        paths = mgv.save_intersection_visualizations(report, config, output_directory, "argmin")

        assert len(paths) == 1
        assert Path(paths[0]).is_file()
        assert Path(paths[0]).stat().st_size > 0
    finally:
        for output_path in output_directory.glob("*.png"):
            output_path.unlink()
        output_directory.rmdir()


def test_save_intersection_visualizations_skips_empty_report(tmp_path: Path) -> None:
    result = mgv.save_intersection_visualizations(
        {"has_intersections": False},
        {},
        tmp_path / "unused",
        "gs",
    )

    assert result == []
    assert not (tmp_path / "unused").exists()


def test_save_intersection_visualizations_supports_gs_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orientations = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    axes_builder = Mock(
        return_value=torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    )
    monkeypatch.setattr(mgv, "cube_axes_min_z_extent", axes_builder)
    config = {
        "magnet_positions_m": torch.tensor(
            [[0.0, 0.0, 0.0], [0.011, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        "magnet_orientations": orientations,
        "magnet_dimensions_m": torch.full((2, 3), 0.012, dtype=torch.float64),
    }
    report = {
        "has_intersections": True,
        "intersections": [{"pair": [0, 1], "penetration_depth_mm": 1.0}],
    }

    paths = mgv.save_intersection_visualizations(
        report,
        config,
        tmp_path,
        "gs",
    )

    axes_builder.assert_called_once_with(orientations)
    assert len(paths) == 1
    assert Path(paths[0]).is_file()


def test_numpy_json_encoder_converts_numpy_values() -> None:
    encoder = mgv.NumpyJSONEncoder()
    encoded = json.dumps(
        {
            "array": np.array([1, 2]),
            "integer": np.int64(3),
        },
        cls=mgv.NumpyJSONEncoder,
    )

    assert json.loads(encoded) == {
        "array": [1, 2],
        "integer": 3,
    }
    assert encoder.default(np.float64(4.5)) == 4.5


def test_numpy_json_encoder_delegates_unsupported_values() -> None:
    encoder = mgv.NumpyJSONEncoder()

    with pytest.raises(TypeError):
        encoder.default(object())
