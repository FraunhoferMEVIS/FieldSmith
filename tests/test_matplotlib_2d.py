import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.quiver import Quiver

from fieldsmith.visualizations.matplotlib_2d import (
    _cube_vertices,
    _normalize,
    _project_cube_outline,
    _rotation_from_x_to_dir,
    plot_angles,
    plot_field_slice_xy,
    plot_ring_magnets,
)


def test_normalize_handles_zero_and_nonzero_vectors() -> None:
    assert np.array_equal(_normalize(np.zeros(3)), np.zeros(3))
    assert np.allclose(_normalize(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0])


@pytest.mark.parametrize(
    "direction",
    [
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
def test_rotation_maps_local_x_to_direction_and_is_orthonormal(
    direction: np.ndarray,
) -> None:
    rotation = _rotation_from_x_to_dir(direction)

    assert np.allclose(rotation[:, 0], direction)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_zero_direction_returns_identity_rotation() -> None:
    assert np.array_equal(_rotation_from_x_to_dir(np.zeros(3)), np.eye(3))


def test_cube_vertices_and_projected_outline_have_expected_geometry() -> None:
    center = np.array([1.0, 2.0, 3.0])
    vertices = _cube_vertices(center, size=2.0, rotation=np.eye(3))
    outline = _project_cube_outline(center, np.array([1.0, 0.0, 0.0]), 2.0)

    assert vertices.shape == (8, 3)
    assert np.allclose(vertices.mean(axis=0), center)
    assert np.allclose(vertices.min(axis=0), center - 1.0)
    assert np.allclose(vertices.max(axis=0), center + 1.0)
    assert outline.shape == (4, 2)


def test_plot_ring_magnets_creates_patches_arrows_and_unique_legend() -> None:
    positions = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    orientations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    figure = plot_ring_magnets(
        positions,
        orientations,
        cube_size=0.02,
        title="Ring test",
    )
    axis = figure.axes[0]
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert len(axis.patches) == 4
    assert legend_labels == ["Magnets", "Magnet orientation"]
    assert axis.get_xlabel() == "x [m]"
    assert axis.get_ylabel() == "y [m]"
    assert axis.get_title() == "Ring test"
    assert axis.get_aspect() == pytest.approx(1.0)
    plt.close(figure)


def test_plot_field_slice_subsamples_field_and_adds_colorbar() -> None:
    points = torch.tensor(
        [[-0.02, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
    )
    field = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    )
    positions = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    orientations = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    figure = plot_field_slice_xy(
        points,
        field,
        positions,
        orientations,
        cube_size=0.02,
        field_subsample=2,
    )
    axis = figure.axes[0]
    quiver = next(collection for collection in axis.collections if isinstance(collection, Quiver))

    assert quiver.get_offsets().shape == (2, 2)
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "|B| [T]"
    plt.close(figure)


def test_plot_field_slice_reuses_supplied_axis() -> None:
    figure, axis = plt.subplots()

    returned = plot_field_slice_xy(
        points=torch.empty((0, 3)),
        field=torch.empty((0, 3)),
        positions=torch.tensor([[0.1, 0.0, 0.0]]),
        orientations=torch.tensor([[1.0, 0.0, 0.0]]),
        cube_size=0.02,
        axis=axis,
        title="Existing axis",
    )

    assert returned is figure
    assert len(figure.axes) == 1
    assert axis.get_title() == "Existing axis"
    plt.close(figure)


def test_plot_angles_creates_current_and_reference_series() -> None:
    position_angles = torch.linspace(0.0, 2.0 * torch.pi, 18 + 1)[:-1]
    positions = torch.stack(
        (
            torch.cos(position_angles),
            torch.sin(position_angles),
            torch.zeros_like(position_angles),
        ),
        dim=1,
    )
    magnet_angles = 2.0 * position_angles

    figure = plot_angles(positions, magnet_angles, title="Angle test")
    axis = figure.axes[0]

    assert len(axis.collections) == 1
    assert len(axis.lines) == 2
    assert axis.get_xlabel() == r"$\phi$ [deg]"
    assert axis.get_ylabel() == r"$\theta_M$ [deg]"
    assert axis.get_title() == "Angle test"
    plt.close(figure)
