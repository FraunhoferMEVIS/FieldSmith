# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains shared 3D system visualizations.
#
#  system_3d.py
#  Kostiantyn Lavronenko
#  01.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch


def _to_numpy(value: torch.Tensor | Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _rotation_from_x(direction: np.ndarray) -> np.ndarray:
    """Return a body frame whose x-axis follows ``direction``."""
    x_axis = direction / max(float(np.linalg.norm(direction)), 1e-12)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(x_axis, up_hint))) > 0.95:
        up_hint = np.asarray([0.0, 1.0, 0.0])
    y_axis = np.cross(up_hint, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _cuboid_vertices(
    center: np.ndarray,
    dimensions: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Return the eight world-space corners of an oriented cuboid."""
    signs = np.asarray([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ])
    local_vertices = signs * dimensions[None, :] / 2.0
    return local_vertices @ _rotation_from_x(direction).T + center


def _cuboid_faces(vertices: np.ndarray) -> list[np.ndarray]:
    """Return the six quadrilateral faces of a cuboid."""
    face_indices = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (2, 6, 7, 3), (0, 3, 7, 4), (1, 5, 6, 2),
    )
    return [vertices[list(indices)] for indices in face_indices]


def plot_magnets_and_field_3d_matplotlib(
    voxel_grid: torch.Tensor,
    field: torch.Tensor,
    magnet_positions: torch.Tensor,
    magnet_orientations: torch.Tensor,
    cube_size: float | torch.Tensor | np.ndarray,
    title: str = "Magnet system and field of view",
    elevation_deg: float = 62.0,
    azimuth_deg: float = 135.0,
    roll_deg: float = 0.0,
    arrow_length_scale: float = 1.55,
    show_colorbar: bool = True,
    show_legend: bool = True,
    view_zoom: float = 1.0,
    figure_size: tuple[float, float] = (7.0, 5.4),
) -> plt.Figure:
    """Create a static 3D view with oriented cuboids, arrows, and FOV field strength."""
    points = np.asarray(_to_numpy(voxel_grid), dtype=float).reshape(-1, 3)
    field_vectors = np.asarray(_to_numpy(field), dtype=float).reshape(-1, 3)
    positions = np.asarray(_to_numpy(magnet_positions), dtype=float).reshape(-1, 3)
    orientations = np.asarray(_to_numpy(magnet_orientations), dtype=float).reshape(-1, 3)
    dimensions = np.asarray(_to_numpy(cube_size), dtype=float)
    if dimensions.ndim == 0:
        dimensions = np.full((len(positions), 3), float(dimensions))
    elif dimensions.shape == (3,):
        dimensions = np.repeat(dimensions[None, :], len(positions), axis=0)
    elif dimensions.shape == (len(positions),):
        dimensions = np.repeat(dimensions[:, None], 3, axis=1)
    elif dimensions.shape != positions.shape:
        raise ValueError("cube_size must be scalar, shape (3,), (M,), or (M, 3).")

    figure = plt.figure(figsize=figure_size)
    axis = figure.add_subplot(111, projection="3d")
    all_vertices = []
    for position, orientation, magnet_dimensions in zip(positions, orientations, dimensions):
        vertices = _cuboid_vertices(position, magnet_dimensions, orientation)
        all_vertices.append(vertices)
        axis.add_collection3d(Poly3DCollection(
            _cuboid_faces(vertices), facecolor="#b9d3e6", edgecolor="#355d7a",
            linewidth=0.35, alpha=0.62,
        ))

    field_strength_mt = np.linalg.norm(field_vectors, axis=1) * 1e3
    normalization = colors.Normalize(
        vmin=float(field_strength_mt.min()), vmax=float(field_strength_mt.max()),
    )
    field_points = axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=field_strength_mt,
        cmap="coolwarm", norm=normalization, s=8, alpha=0.95, depthshade=False,
    )
    if show_colorbar:
        colorbar = figure.colorbar(field_points, ax=axis, shrink=0.62, pad=0.01)
        colorbar.set_label(r"Field strength $|B|$ (mT)")

    direction_norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    directions = orientations / np.maximum(direction_norms, 1e-12)
    arrow_length = float(np.mean(dimensions[:, 0])) * arrow_length_scale
    # axis.quiver(
    #     positions[:, 0], positions[:, 1], positions[:, 2],
    #     directions[:, 0], directions[:, 1], directions[:, 2],
    #     length=arrow_length, normalize=True, color="#0b2538", linewidth=0.9,
    #     arrow_length_ratio=0.46,
    # )

    extent_points = np.concatenate((*all_vertices, points), axis=0)
    minimum = extent_points.min(axis=0)
    maximum = extent_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = float((maximum - minimum).max()) * 0.54
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=view_zoom)
    axis.view_init(elev=elevation_deg, azim=azimuth_deg, roll=roll_deg)
    # axis.set_title(title, pad=3)
    axis.set_axis_off()
    if show_legend:
        axis.legend(
            handles=[
                Patch(facecolor="#b9d3e6", edgecolor="#355d7a", alpha=0.8, label="Magnets"),
                Line2D([0], [0], color="#0b2538", marker=">", label="Magnetization"),
            ],
            loc="upper left", frameon=False,
        )
    figure.subplots_adjust(
        left=0.0, right=0.94 if show_colorbar else 1.0, bottom=0.0, top=0.91,
    )
    return figure


def _plot_magnets_matplotlib(
    magnet_positions: torch.Tensor,
    magnet_orientations: torch.Tensor,
    cube_size: float | torch.Tensor,
    title: str,
) -> plt.Figure:
    positions = magnet_positions.detach().cpu().numpy()
    orientations = magnet_orientations.detach().cpu().numpy()
    step = max(1, positions.shape[0] // 500)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=12, alpha=0.6)
    maximum_magnet_size = float(torch.as_tensor(cube_size).max().item())
    ax.quiver(
        positions[::step, 0],
        positions[::step, 1],
        positions[::step, 2],
        orientations[::step, 0],
        orientations[::step, 1],
        orientations[::step, 2],
        length=maximum_magnet_size * 1.2,
        normalize=True,
        color="tab:green",
        linewidth=0.8,
    )
    max_radius = float(torch.linalg.norm(magnet_positions, dim=-1).max().detach().cpu().item())
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(min(0.0, float(positions[:, 2].min())), max_radius)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_magnets_and_field_3d(
    voxel_grid: torch.Tensor,
    field: torch.Tensor,
    magnet_positions: torch.Tensor,
    magnet_orientations: torch.Tensor,
    cube_size: float | torch.Tensor,
    mask: torch.Tensor | None = None,
    symmetry_type: str | None = None,
    title: str = "3D Magnetic Field, Magnets, and Optimization Region",
) -> Any:
    """Plot magnets and field points with Plotly when available, otherwise matplotlib."""
    try:
        from fieldsmith.visualizations.plotly_3d import plot_magnets_and_field_3d_plotly
    except ModuleNotFoundError as exc:
        if exc.name != "plotly":
            raise
        return _plot_magnets_matplotlib(
            magnet_positions=magnet_positions,
            magnet_orientations=magnet_orientations,
            cube_size=cube_size,
            title=title,
        )

    return plot_magnets_and_field_3d_plotly(
        voxel_grid=_to_numpy(voxel_grid),
        B=_to_numpy(field) * 1e3,
        magnet_positions=_to_numpy(magnet_positions),
        magnet_orientations=_to_numpy(magnet_orientations),
        mask=None if mask is None else _to_numpy(mask),
        symmetry_type=symmetry_type,
        cube_size=_to_numpy(cube_size),
        center=[0.0, 0.0, 0.0],
        title=title,
    )


def plot_model_output_3d(
    output: dict[str, torch.Tensor],
    cube_size: float | torch.Tensor,
    mask: torch.Tensor | None = None,
    symmetry_type: str | None = None,
    title: str = "3D Magnetic Field, Magnets, and Optimization Region",
) -> Any:
    """Plot a standard model output dictionary."""
    points = output.get("positions", output.get("points"))
    if points is None:
        raise ValueError("output must contain 'positions' or 'points'.")
    return plot_magnets_and_field_3d(
        voxel_grid=points,
        field=output["B"],
        magnet_positions=output["magnet_positions"],
        magnet_orientations=output.get("magnet_vectors", output.get("magnet_orientations")),
        cube_size=cube_size,
        mask=mask,
        symmetry_type=symmetry_type,
        title=title,
    )
