# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains 2D visualization utilities for magnet optimization.
#
#  matplotlib_2d.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Polygon
import torch

_AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def _rotation_from_x_to_dir(direction: np.ndarray) -> np.ndarray:
    x_axis = _normalize(direction)
    if np.linalg.norm(x_axis) < 1e-12:
        return np.eye(3)

    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])

    y_axis = _normalize(np.cross(up_hint, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _cube_vertices(center: np.ndarray, size: float, rotation: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    half_size = size / 2.0
    vertices_local = np.array(
        [
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size],
        ]
    )
    return (rotation @ vertices_local.T).T + center


def _project_points(points: np.ndarray, plane: str = "xy") -> np.ndarray:
    first_axis = _AXIS_TO_ID[plane[0]]
    second_axis = _AXIS_TO_ID[plane[1]]
    return points[:, [first_axis, second_axis]]


def _slice_axis_from_plane(plane: str = "xy") -> int:
    return ({0, 1, 2} - {_AXIS_TO_ID[plane[0]], _AXIS_TO_ID[plane[1]]}).pop()


def _project_cube_outline(center: np.ndarray, direction: np.ndarray, size: float, plane: str = "xy") -> np.ndarray:
    rotation = _rotation_from_x_to_dir(direction)
    vertices_3d = _cube_vertices(center=center, size=size, rotation=rotation)
    vertices_2d = _project_points(vertices_3d, plane=plane)
    centroid = vertices_2d.mean(axis=0)
    angles = np.arctan2(vertices_2d[:, 1] - centroid[1], vertices_2d[:, 0] - centroid[0])
    polygon = vertices_2d[np.argsort(angles)]

    cleaned = []
    for point in polygon:
        if len(cleaned) == 0 or np.linalg.norm(point - cleaned[-1]) > 1e-12:
            cleaned.append(point)
    return np.array(cleaned)


def _deduplicate_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    if unique_labels:
        ax.legend(unique_handles, unique_labels, loc="upper right")


def plot_ring_magnets(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    cube_size: float,
    title: str = "Halbach ring",
) -> plt.Figure:
    """Create a 2D magnet visualization using projected cube outlines."""
    positions_np = positions.detach().cpu().numpy()
    orientations_np = orientations.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    first_patch = True
    first_arrow = True

    for position, orientation in zip(positions_np, orientations_np):
        polygon = _project_cube_outline(position, orientation, size=cube_size, plane="xy")
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="tab:red",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.35,
            label="Magnets" if first_patch else None,
        )
        ax.add_patch(patch)
        first_patch = False

        direction = _normalize(orientation)
        start = position[:2]
        arrow_len = cube_size * 0.9
        ax.arrow(
            start[0],
            start[1],
            direction[0] * arrow_len,
            direction[1] * arrow_len,
            width=cube_size * 0.06,
            head_width=cube_size * 0.22,
            head_length=cube_size * 0.22,
            length_includes_head=True,
            color="darkgreen",
            alpha=0.95,
            label="Magnet orientation" if first_arrow else None,
        )
        first_arrow = False

    max_radius = (positions[:, :2].norm(dim=-1).max().detach().cpu().item()) * 1.2
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _deduplicate_legend(ax)
    fig.tight_layout()
    return fig


def plot_field_slice_xy(
    points: torch.Tensor,
    field: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    cube_size: float,
    field_subsample: int = 1,
    title: str = "2D magnetic field",
    axis: plt.Axes | None = None,
) -> plt.Figure:
    """Create a 2D field-arrow slice with projected magnet cube outlines."""
    points_np = points.detach().cpu().numpy()
    field_np = field.detach().cpu().numpy()
    positions_np = positions.detach().cpu().numpy()
    orientations_np = orientations.detach().cpu().numpy()

    if field_subsample > 1 and len(points_np) > 0:
        indices = np.arange(0, len(points_np), field_subsample)
        points_np = points_np[indices]
        field_np = field_np[indices]

    field_abs = np.linalg.norm(field_np, axis=-1) if len(field_np) > 0 else np.array([0.0])
    projected_points = _project_points(points_np, plane="xy") if len(points_np) > 0 else np.zeros((0, 2))
    projected_field = _project_points(field_np, plane="xy") if len(field_np) > 0 else np.zeros((0, 2))

    owns_figure = axis is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = axis.figure
        ax = axis
    norm = colors.Normalize(vmin=float(field_abs.min()), vmax=float(field_abs.max()) if field_abs.max() > 0 else 1.0)
    cmap = cm.jet

    if len(projected_points) > 0:
        inplane_mag = np.linalg.norm(projected_field, axis=-1)
        scale_ref = np.mean(inplane_mag[inplane_mag > 0]) if np.any(inplane_mag > 0) else 1.0
        quiver = ax.quiver(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_field[:, 0],
            projected_field[:, 1],
            field_abs,
            cmap=cmap,
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=max(scale_ref, 1e-12) / 0.015,
            width=0.004,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
            pivot="middle",
        )
        colorbar = fig.colorbar(quiver, ax=ax, pad=0.02)
        colorbar.set_label("|B| [T]")

    first_patch = True
    first_arrow = True
    for position, orientation in zip(positions_np, orientations_np):
        polygon = _project_cube_outline(position, orientation, size=cube_size, plane="xy")
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="tab:red",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.35,
            label="Magnets" if first_patch else None,
        )
        ax.add_patch(patch)
        first_patch = False

        direction = _normalize(orientation)
        start = position[:2]
        arrow_len = cube_size * 0.9
        ax.arrow(
            start[0],
            start[1],
            direction[0] * arrow_len,
            direction[1] * arrow_len,
            width=cube_size * 0.06,
            head_width=cube_size * 0.22,
            head_length=cube_size * 0.22,
            length_includes_head=True,
            color="darkgreen",
            alpha=0.95,
            label="Magnet orientation" if first_arrow else None,
        )
        first_arrow = False

    max_radius = (positions[:, :2].norm(dim=-1).max().detach().cpu().item()) * 1.15
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _deduplicate_legend(ax)
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_angles(
    positions: torch.Tensor,
    magnet_angles: torch.Tensor,
    title: str = "Magnet angles",
) -> plt.Figure:
    """Compare current magnet angles with ideal Halbach and Tewari/Webb orientations."""
    positions_np = positions.detach().cpu().numpy()
    magnet_angles_np = magnet_angles.detach().cpu().numpy()
    magnet_angles_np = magnet_angles_np
    

    position_angle = np.rad2deg(np.arctan2(positions_np[:, 1], positions_np[:, 0]))
    position_angle = (position_angle - position_angle[0]) % 360.0

    current_angle = np.rad2deg(magnet_angles_np)
    current_angle = np.roll(current_angle, 9)  # Align first magnet with position angle 0
    current_angle_offset = (current_angle - 2.0 * position_angle).mean()
    current_angle = (current_angle - current_angle_offset) % 360.0
    angle_diff = np.diff(current_angle)
    angle_diff = np.where(angle_diff > 180.0, angle_diff - 360.0, angle_diff)
    angle_diff = np.where(angle_diff < -180.0, angle_diff + 360.0, angle_diff)
    current_angle_unwrapped = np.concatenate(([current_angle[0]], np.cumsum(angle_diff) + current_angle[0]))

    for index in range(len(current_angle_unwrapped)):
        derivation = np.floor((current_angle_unwrapped[index] - 2.0 * position_angle[index]) / 180.0)
        # current_angle_unwrapped[index] -= derivation * 180.0

    oreilly_angle = (
        np.rad2deg(
            2.0 * np.deg2rad(position_angle)
            + np.pi / 8.0 * np.cos(np.deg2rad(2.0 * position_angle - 90.0))
        )
        + 90.0
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(position_angle, current_angle_unwrapped - 90, label="Current rotations", marker="o", s=45)
    ax.plot(position_angle, 2.0 * position_angle + 90.0, "--", color="black", label=r"$\theta_M = 2\phi$")
    ax.plot(
        position_angle,
        oreilly_angle,
        "-",
        color="tab:orange",
        label=r"Tewari/Webb orientation",
    )
    ax.set_xlim(0.0, 350.0)
    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta_M$ [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    fig.tight_layout()
    return fig
