import numpy as np
import plotly.graph_objects as go
import pytest

from fieldsmith.visualizations.plotly_3d import (
    _cube_face_indices,
    _cube_vertices,
    _make_arc,
    _normalize,
    _rotation_from_x_to_dir,
    compute_surface_mask,
    plot_magnets_and_field_3d_plotly,
)


def sample_plot_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-0.1, 0.1, 3)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    field = np.ones_like(grid)
    positions = np.array([[0.2, 0.0, 0.0]])
    orientations = np.array([[1.0, 0.0, 0.0]])
    return grid, field, positions, orientations


def test_compute_surface_mask_removes_only_enclosed_voxels() -> None:
    mask = np.ones((3, 3, 3), dtype=bool)

    surface = compute_surface_mask(mask)

    assert surface.dtype == bool
    assert surface.sum() == 26
    assert not surface[1, 1, 1]
    assert surface[0, 0, 0]


def test_rotation_and_cube_helpers_create_right_handed_geometry() -> None:
    direction = np.array([0.0, 0.0, 1.0])
    rotation = _rotation_from_x_to_dir(direction)
    vertices = _cube_vertices(
        center=np.array([1.0, 2.0, 3.0]),
        size=2.0,
        rotation=rotation,
    )

    assert np.array_equal(_normalize(np.zeros(3)), np.zeros(3))
    assert np.allclose(rotation[:, 0], direction)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert vertices.shape == (8, 3)
    assert np.allclose(vertices.mean(axis=0), [1.0, 2.0, 3.0])
    assert _cube_face_indices().shape == (12, 3)


def test_make_arc_supports_all_planes_and_rejects_unknown_plane() -> None:
    for plane in ("xy", "xz", "yz"):
        x, y, z = _make_arc([0.0, 0.0, 0.0], radius=1.0, plane=plane, n=5)
        assert x.shape == y.shape == z.shape == (5,)

    with pytest.raises(ValueError, match="plane must be one of"):
        _make_arc([0.0, 0.0, 0.0], radius=1.0, plane="ab")


def test_plotly_figure_contains_field_magnet_and_orientation_traces() -> None:
    grid, field, positions, orientations = sample_plot_inputs()

    figure = plot_magnets_and_field_3d_plotly(
        grid,
        field,
        positions,
        orientations,
        symmetry_type=None,
        cube_size=np.array([0.02, 0.03, 0.04]),
        title="Plotly test",
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 5
    assert isinstance(figure.data[0], go.Scatter3d)
    assert len(figure.data[0].x) == 27
    assert isinstance(figure.data[1], go.Mesh3d)
    assert np.ptp(np.asarray(figure.data[1].x)) == pytest.approx(0.02)
    assert figure.layout.title.text == "Plotly test"
    assert figure.layout.scene.aspectmode == "cube"


@pytest.mark.parametrize(
    ("symmetry_type", "expected_traces"),
    [
        ("octant", 9),
        ("quadrant", 9),
    ],
)
def test_plotly_figure_adds_symmetry_boundary(
    symmetry_type: str,
    expected_traces: int,
) -> None:
    grid, field, positions, orientations = sample_plot_inputs()

    figure = plot_magnets_and_field_3d_plotly(
        grid,
        field,
        positions,
        orientations,
        symmetry_type=symmetry_type,
    )

    assert len(figure.data) == expected_traces
    assert figure.data[0].name == "Optimization boundary"


def test_plotly_surface_only_mask_excludes_interior_points() -> None:
    grid, field, positions, orientations = sample_plot_inputs()

    figure = plot_magnets_and_field_3d_plotly(
        grid,
        field,
        positions,
        orientations,
        mask=np.ones((3, 3, 3), dtype=bool),
        symmetry_type=None,
        surface_only=True,
    )

    assert len(figure.data[0].x) == 26


@pytest.mark.parametrize(
    ("cube_size", "message"),
    [
        (np.ones((2, 2)), "cube_size must be scalar"),
        (0.0, "cube_size entries must be positive"),
        (np.array([0.01, -0.01, 0.01]), "cube_size entries must be positive"),
    ],
)
def test_plotly_figure_rejects_invalid_magnet_sizes(
    cube_size: np.ndarray | float,
    message: str,
) -> None:
    grid, field, positions, orientations = sample_plot_inputs()

    with pytest.raises(ValueError, match=message):
        plot_magnets_and_field_3d_plotly(
            grid,
            field,
            positions,
            orientations,
            symmetry_type=None,
            cube_size=cube_size,
        )


def test_plotly_figure_rejects_unknown_symmetry() -> None:
    grid, field, positions, orientations = sample_plot_inputs()

    with pytest.raises(ValueError, match="symmetry_type must be"):
        plot_magnets_and_field_3d_plotly(
            grid,
            field,
            positions,
            orientations,
            symmetry_type="half",
        )
