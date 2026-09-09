# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains interactive 3D Plotly visualization utilities.
#
#  plotly_3d.py
#  Kostiantyn Lavronenko
#  01.07.2026
# -----------------------------------------------------------------------------

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


def compute_surface_mask(mask):
    """
    Extract surface voxels from a 3D boolean mask.
    A voxel is on the surface if at least one of its 6 neighbors is outside.
    """
    mask = mask.astype(bool)

    surface = np.zeros_like(mask, dtype=bool)

    shifts = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for dx, dy, dz in shifts:
        shifted = np.roll(mask, shift=(dx, dy, dz), axis=(0, 1, 2))

        if dx == 1:
            shifted[0, :, :] = False
        if dx == -1:
            shifted[-1, :, :] = False
        if dy == 1:
            shifted[:, 0, :] = False
        if dy == -1:
            shifted[:, -1, :] = False
        if dz == 1:
            shifted[:, :, 0] = False
        if dz == -1:
            shifted[:, :, -1] = False

        surface |= mask & (~shifted)

    return surface


def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _rotation_from_x_to_dir(direction, up_hint=(0.0, 0.0, 1.0)):
    """
    Build a rotation matrix whose local +x axis points along `direction`.
    """
    x_axis = _normalize(direction)
    if np.linalg.norm(x_axis) < 1e-12:
        return np.eye(3)

    up_hint = _normalize(up_hint)
    if abs(np.dot(x_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])

    y_axis = _normalize(np.cross(up_hint, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))

    return np.column_stack([x_axis, y_axis, z_axis])


def _cube_vertices(center, size=0.012, rotation=None):
    """
    Returns 8 rotated cube vertices.
    """
    center = np.asarray(center, dtype=float)
    h = size / 2.0

    vertex_signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    v_local = vertex_signs * h

    if rotation is None:
        v = v_local + center
    else:
        v = (rotation @ v_local.T).T + center

    return v


def _cube_face_indices():
    """
    Returns the 12 triangles for a cube surface mesh.
    Vertex order assumes the indexing from _cube_vertices.
    """
    triangles = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 3, 7],
        [0, 7, 4],
    ]
    return np.array(triangles, dtype=int)


def _make_single_cube_mesh_trace(
    center,
    size=0.012,
    rotation=None,
    color="red",
    opacity=0.35,
    name=None,
    showscale=False,
    showlegend=False,
    legendgroup=None,
):
    """
    Create one Plotly Mesh3d trace for one cube.
    """
    v = _cube_vertices(center=center, size=size, rotation=rotation)
    tri = _cube_face_indices()

    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showscale=showscale,
        showlegend=showlegend,
        legendgroup=legendgroup,
        hoverinfo="skip",
    )


def _make_octant_surface(center, radius, n_phi=30, n_theta=30):
    phi = np.linspace(0, np.pi / 2, n_phi)
    theta = np.linspace(0, np.pi / 2, n_theta)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    return x, y, z


def _make_arc(center, radius, plane="xy", n=100):
    t = np.linspace(0, np.pi / 2, n)

    if plane == "xy":
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = np.full_like(t, center[2])
    elif plane == "xz":
        x = center[0] + radius * np.cos(t)
        y = np.full_like(t, center[1])
        z = center[2] + radius * np.sin(t)
    elif plane == "yz":
        x = np.full_like(t, center[0])
        y = center[1] + radius * np.cos(t)
        z = center[2] + radius * np.sin(t)
    else:
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    return x, y, z


def _make_quadrant_surface(center, radius, n_phi=30, n_theta=60):
    center = np.asarray(center, dtype=float)

    phi = np.linspace(0, np.pi / 2, n_phi)
    theta = np.linspace(0, np.pi, n_theta)

    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    return x, y, z


def _make_quadrant_arcs(center, radius, n=100, **kwargs):
    center = np.asarray(center, dtype=float)
    t = np.linspace(0, np.pi, n)

    x1 = center[0] + radius * np.sin(t)
    y1 = np.full_like(t, center[1])
    z1 = center[2] + radius * np.cos(t)

    x2 = np.full_like(t, center[0])
    y2 = center[1] + radius * np.sin(t)
    z2 = center[2] + radius * np.cos(t)

    t2 = np.linspace(0, np.pi / 2, n)
    x3 = center[0] + radius * np.cos(t2)
    y3 = center[1] + radius * np.sin(t2)
    z3 = np.full_like(t2, center[2])

    return (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)


def plot_magnets_and_field_3d_plotly(
    voxel_grid,
    B,
    magnet_positions,
    magnet_orientations,
    mask=None,
    symmetry_type="octant",
    cube_size=0.012,
    field_marker_size=2.5,
    show_field_points=True,
    magnet_color="gainsboro",
    magnet_opacity=0.8,
    optimization_color="aqua",
    optimization_opacity=0.5,
    field_colorscale="Jet",
    force_equal_axes=True,
    surface_only=True,
    center=None,
    title="3D Magnetic Field, Magnets, and Optimization Region",
):
    voxel_grid = np.asarray(voxel_grid)
    B = np.asarray(B)
    magnet_positions = np.asarray(magnet_positions)
    magnet_orientations = np.asarray(magnet_orientations)
    magnet_sizes = np.asarray(cube_size, dtype=float)
    if magnet_sizes.ndim == 0:
        magnet_sizes = np.full((len(magnet_positions), 3), float(magnet_sizes))
    elif magnet_sizes.shape == (3,):
        magnet_sizes = np.repeat(magnet_sizes[None, :], len(magnet_positions), axis=0)
    elif magnet_sizes.shape != magnet_positions.shape:
        raise ValueError("cube_size must be scalar, shape (3,), or shape (M, 3).")
    if np.any(magnet_sizes <= 0.0):
        raise ValueError("cube_size entries must be positive.")

    voxel_grid_flat = voxel_grid.reshape(-1, 3)
    B_flat = B.reshape(-1, 3)

    if mask is not None:
        mask_flat = np.asarray(mask).astype(bool)
        if surface_only:
            mask_flat = compute_surface_mask(mask_flat).reshape(-1)
        else:
            mask_flat = mask_flat.reshape(-1)
        voxel_grid_flat = voxel_grid_flat[mask_flat]
        B_flat = B_flat[mask_flat]

    B_abs = np.linalg.norm(B_flat, axis=-1)

    fig = go.Figure()

    # ------------------------------------------------------------
    # Optimization region
    # ------------------------------------------------------------
    if symmetry_type is not None:
        if len(voxel_grid_flat) > 0:
            if center is None:
                center = voxel_grid_flat.mean(axis=0)
            else:
                center = np.asarray(center, dtype=float)

            radius = np.linalg.norm(voxel_grid_flat - center, axis=1).max() * 1.05
            optimization_group = "optimization_region"

            if symmetry_type == "octant":
                xs, ys, zs = _make_octant_surface(center, radius, n_phi=28, n_theta=28)

                fig.add_trace(
                    go.Surface(
                        x=xs,
                        y=ys,
                        z=zs,
                        opacity=optimization_opacity,
                        showscale=False,
                        colorscale=[[0, optimization_color], [1, optimization_color]],
                        name="Optimization boundary",
                        legendgroup=optimization_group,
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

                for plane in ("xy", "xz", "yz"):
                    xa, ya, za = _make_arc(center, radius, plane=plane, n=120)
                    fig.add_trace(
                        go.Scatter3d(
                            x=xa,
                            y=ya,
                            z=za,
                            mode="lines",
                            line=dict(color=optimization_color, width=5),
                            name="Optimization boundary",
                            legendgroup=optimization_group,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            elif symmetry_type == "quadrant":
                xs, ys, zs = _make_quadrant_surface(center, radius, n_phi=28, n_theta=56)

                fig.add_trace(
                    go.Surface(
                        x=xs,
                        y=ys,
                        z=zs,
                        opacity=optimization_opacity,
                        showscale=False,
                        colorscale=[[0, optimization_color], [1, optimization_color]],
                        name="Optimization boundary",
                        legendgroup=optimization_group,
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

                arc1, arc2, arc3 = _make_quadrant_arcs(center, radius, n=120)

                for xa, ya, za in (arc1, arc2, arc3):
                    fig.add_trace(
                        go.Scatter3d(
                            x=xa,
                            y=ya,
                            z=za,
                            mode="lines",
                            line=dict(color=optimization_color, width=5),
                            name="Optimization boundary",
                            legendgroup=optimization_group,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            else:
                raise ValueError("symmetry_type must be 'octant' or 'quadrant'")

    # ------------------------------------------------------------
    # Field points
    # ------------------------------------------------------------
    if show_field_points:
        fig.add_trace(
            go.Scatter3d(
                x=voxel_grid_flat[:, 0],
                y=voxel_grid_flat[:, 1],
                z=voxel_grid_flat[:, 2],
                mode="markers",
                marker=dict(
                    size=field_marker_size,
                    color=B_abs,
                    colorscale=field_colorscale,
                    opacity=1,
                    colorbar=dict(
                        title=dict(text="|B| [mT]", font=dict(size=34)),
                        tickfont=dict(size=28),
                        thickness=35,  # very wide
                        len=0.9,  # almost full height
                        x=0.76,
                        y=0.5,
                        outlinewidth=1,
                    ),
                ),
                name="Field points",
            )
        )

    # ------------------------------------------------------------
    # Magnets
    # ------------------------------------------------------------
    magnet_group = "magnets"

    for i, (pos, mdir, size) in enumerate(zip(magnet_positions, magnet_orientations, magnet_sizes)):
        R = _rotation_from_x_to_dir(mdir)
        fig.add_trace(
            _make_single_cube_mesh_trace(
                center=pos,
                size=size,
                rotation=R,
                color=magnet_color,
                opacity=magnet_opacity,
                name="Magnets",
                showlegend=(i == 0),
                legendgroup=magnet_group,
            )
        )

    # ------------------------------------------------------------
    # Magnet centers + orientation
    # ------------------------------------------------------------
    mag_norm = np.linalg.norm(magnet_orientations, axis=-1, keepdims=True)
    mag_dir = magnet_orientations / (mag_norm + 1e-12)
    magnet_arrow_lengths = magnet_sizes[:, 0] * 1.2

    line_x, line_y, line_z = [], [], []
    tip_x, tip_y, tip_z = [], [], []

    for pos, d, arrow_length in zip(magnet_positions, mag_dir, magnet_arrow_lengths):
        start = pos
        end = pos + d * arrow_length

        line_x.extend([start[0], end[0], None])
        line_y.extend([start[1], end[1], None])
        line_z.extend([start[2], end[2], None])

        tip_x.append(end[0])
        tip_y.append(end[1])
        tip_z.append(end[2])

    magnet_orientation_group = "magnet_orientation"

    fig.add_trace(
        go.Scatter3d(
            x=magnet_positions[:, 0],
            y=magnet_positions[:, 1],
            z=magnet_positions[:, 2],
            mode="markers",
            marker=dict(size=3, color="black"),
            name="Magnet orientation",
            legendgroup=magnet_orientation_group,
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color="darkgreen", width=6),
            name="Magnet orientation",
            legendgroup=magnet_orientation_group,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=tip_x,
            y=tip_y,
            z=tip_z,
            mode="markers",
            marker=dict(size=4, color="darkgreen"),
            name="Magnet orientation",
            legendgroup=magnet_orientation_group,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ------------------------------------------------------------
    # Layout / axis scaling
    # ------------------------------------------------------------
    all_pts = np.vstack([voxel_grid_flat, magnet_positions]) if len(voxel_grid_flat) > 0 else magnet_positions

    if len(all_pts) > 0:
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        ranges = maxs - mins
        max_range = ranges.max()
        maximum_magnet_size = float(magnet_sizes.max())
        pad = max(
            maximum_magnet_size * 2.0,
            0.05 * max_range if max_range > 0 else maximum_magnet_size * 2.0,
        )

        if force_equal_axes:
            half = 0.5 * max_range + pad
            x_range = [center[0] - half, center[0] + half]
            y_range = [center[1] - half, center[1] + half]
            z_range = [center[2] - half, center[2] + half]
        else:
            x_range = [mins[0] - pad, maxs[0] + pad]
            y_range = [mins[1] - pad, maxs[1] + pad]
            z_range = [mins[2] - pad, maxs[2] + pad]
    else:
        x_range = y_range = z_range = [-0.1, 0.1]

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="top",
            font=dict(size=48),
        ),
        scene=dict(
            domain=dict(x=[0.0, 0.84], y=[0.0, 1.0]),  # leaves room for legend/colorbar, keeps them close
            xaxis=dict(
                title="x [m]",
                range=x_range,
                backgroundcolor="rgba(240,240,240,1)",
                title_font=dict(size=16),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="y [m]",
                range=y_range,
                backgroundcolor="rgba(240,240,240,1)",
                title_font=dict(size=16),
                tickfont=dict(size=12),
            ),
            zaxis=dict(
                title="z [m]",
                range=z_range,
                backgroundcolor="rgba(240,240,240,1)",
                title_font=dict(size=16),
                tickfont=dict(size=12),
            ),
            aspectmode="cube" if force_equal_axes else "data",
        ),
        legend=dict(
            x=-0.06,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=18),  # ← legend items BIG
            title=dict(text="Legend", font=dict(size=20)),  # ← set your title here  # ← legend title BIGGER
            itemsizing="constant",
            itemwidth=40,  # ← more spacing per item
            tracegroupgap=8,  # ← spacing between groups
            groupclick="togglegroup",
        ),
        margin=dict(l=20, r=20, b=20, t=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig
