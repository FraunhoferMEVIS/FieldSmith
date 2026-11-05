# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file cuts a discrete-shell design into printable plates: a hull body over
# a smooth bore, a seat and insertion channel per magnet, an ejection hole under
# each seat, and a recess for the lid.
#
#  slabs.py
#  Marian Frei
#  31.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
import torch
import trimesh
from scipy.spatial import ConvexHull

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry.cube_axes import cube_axes_min_z_extent
from fieldsmith.cad.plates import assign_planes, plate_bounds

SQRT3 = 3.0 ** 0.5
# Cuts reach this far past a plate surface so no face is coplanar with it.
OVERSHOOT_M = 5.0e-3
# Below this upward component the ejector hole runs vertically instead of along the seat axis.
UPRIGHT_MIN = 0.3


def _robust_difference(body: trimesh.Trimesh, cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Subtract every cutter, falling back to one at a time.

    The manifold backend rejects a whole batch if one cutter is degenerate, so
    the fast batched difference is tried first.
    """
    if not cutters:
        return body
    try:
        # A difference against the list, not a concatenation: the cutters
        # overlap, and a merged solid would self-intersect.
        return trimesh.boolean.difference([body, *cutters])
    except Exception:
        pass
    for cutter in cutters:
        try:
            body = trimesh.boolean.difference([body, cutter])
        except Exception:
            continue
    return body


def _hull_body(points_xy: np.ndarray, margin: float, z_low: float, z_high: float,
               directions: int = 24, min_radius: float | None = None) -> trimesh.Trimesh:
    """
    Return the plate body: the convex hull of the magnet footprint grown by a margin.

    Args:
        points_xy: Magnet centres projected to the plate, shape (N, 2).
        margin: Material outside the footprint, in meters.
        z_low: Underside of the plate, in meters.
        z_high: Top of the plate, in meters.
        directions: Corners of the polygon standing in for the disc.
    """
    hull = points_xy[ConvexHull(points_xy).vertices]
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    offsets = margin * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    grown = (hull[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    if min_radius is not None:
        # A fastening ring: the outline grows to at least this radius.
        ring_angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        ring = min_radius * np.stack((np.cos(ring_angles), np.sin(ring_angles)), axis=1)
        grown = np.concatenate([grown, ring])
    outline = grown[ConvexHull(grown).vertices]
    body = trimesh.creation.extrude_polygon(shapely.Polygon(outline), height=z_high - z_low)
    body.apply_translation((0.0, 0.0, z_low))
    return body


def _bore(radius_low: float, radius_high: float, z_low: float, z_high: float,
          segments: int = 96) -> trimesh.Trimesh:
    """Return the bore to subtract, a cone when the two radii differ, so the stack forms one smooth bore."""
    height = (z_high - z_low) + 2 * OVERSHOOT_M
    centre = (z_low + z_high) / 2
    if abs(radius_high - radius_low) < 1e-9:
        bore = trimesh.creation.cylinder(radius=max(radius_low, 1e-4), height=height, sections=segments)
    else:
        profile = [(0.0, -height / 2), (max(radius_low, 1e-4), -height / 2),
                   (max(radius_high, 1e-4), height / 2), (0.0, height / 2)]
        bore = trimesh.creation.revolve(np.array(profile), sections=segments)
    bore.apply_translation((0.0, 0.0, centre))
    return bore


def _upward_normal(axes: np.ndarray) -> np.ndarray:
    """Return the cube face normal pointing most steeply upward."""
    normals = np.concatenate([axes, -axes])
    return normals[int(np.argmax(normals[:, 2]))]


def _pocket_and_channel(centre: np.ndarray, axes: np.ndarray, size: float,
                        plate_top: float, clearance: float) -> trimesh.Trimesh:
    """
    Return the seat and its insertion channel as one prism.

    The cut runs from the underside of the seat along the most upward body axis
    out through the plate top. One prism avoids the membrane two stacked cuts
    leave across the opening.

    Args:
        centre: Magnet centre in meters, shape (3,).
        axes: Body frame, shape (3, 3); rows are local x, y, z in world.
        size: Cube edge length in meters.
        plate_top: Top surface of this plate, in meters.
        clearance: Added to the cross-section for the fit, in meters.
    """
    normal = _upward_normal(axes)
    across = np.array([row for row in axes if abs(row @ normal) < 0.5])
    if len(across) != 2:                                     # numerically degenerate
        across = np.array([row for row in axes if abs(row @ normal) < 0.9])[:2]

    bottom = -size / 2
    top = (plate_top - centre[2]) / max(normal[2], 1e-6) + OVERSHOOT_M
    width = size + clearance
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = across[0], across[1], normal
    frame[:3, 3] = centre + normal * (bottom + top) / 2
    return trimesh.creation.box(extents=(width, width, top - bottom), transform=frame)


def _ejection_hole(centre: np.ndarray, axes: np.ndarray, size: float,
                   plate_bottom: float, radius: float) -> trimesh.Trimesh | None:
    """
    Return the hole through the underside that lets a seated magnet be pushed out.

    The hole follows the insertion axis, so a rod pushes squarely on the face.
    Where that axis is too shallow it runs straight down. Returns None if the
    magnet sits below the plate floor.
    """
    down = -_upward_normal(axes)
    if abs(down[2]) < UPRIGHT_MIN:
        down = np.array([0.0, 0.0, -1.0])
    start = centre + down * (size / 2 - 1.0e-3)
    distance = (plate_bottom - OVERSHOOT_M - start[2]) / down[2]
    if distance <= 0:
        return None
    end = start + down * distance
    return trimesh.creation.cylinder(radius=radius, segment=np.stack([start, end]), sections=24)


def _bore_radii(positions: torch.Tensor, labels: torch.Tensor, plates: list[dict],
                size: float, margin: float) -> list[tuple[float, float]]:
    """Return each plate's bore radius at its underside and top, shared at interfaces."""
    radius = positions[:, :2].norm(dim=1)
    room = []
    for index in range(len(plates)):
        here = labels == index
        room.append(max(float(radius[here].min()) - size * SQRT3 / 2 - margin, 1.0e-3))
    shared = [room[0]] + [min(room[i], room[i + 1]) for i in range(len(room) - 1)] + [room[-1]]
    return [(shared[i], shared[i + 1]) for i in range(len(plates))]


def export_slabs(
    configuration: MagnetConfiguration,
    out_dir: Path,
    roof: float = 1.0e-3,
    floor: float = 1.4e-3,
    clearance: float = 1.0e-4,
    wall: float = 2.5e-3,
    lid_gap: float = 1.0e-3,
    ejector_radius: float = 2.15e-3,
    variant: str = "default",
    outer_radius_m: dict | None = None,
) -> dict:
    """
    Write the trimmed and raw plates as STL in millimetres, and their manifest.

    The trimmed plate has the lid recess taken off its top. The raw plate keeps
    that volume, because the lids are built against it.

    Args:
        configuration: The design to cut up.
        out_dir: Directory for the STLs and the manifest.
        roof: Material above a pocket, in meters.
        floor: Material below a pocket, in meters.
        clearance: Added to each seat for the fit, in meters.
        wall: Material outside the magnets' footprint, in meters.
        lid_gap: Depth of the lid recess, in meters.
        ejector_radius: Radius of the push-out hole, in meters.
        variant: Name recorded in the manifest for this export.
    """
    size = float(configuration.metadata.get("magnet_size_m") or 0.012)
    # The lid may not reach into the magnets, so the roof is at least the lid depth.
    roof = max(roof, lid_gap)
    positions = configuration.positions.double()
    orientations = configuration.orientations.double()
    labels, heights = assign_planes(positions)
    plates = plate_bounds(positions, orientations, size, roof, floor)
    axes = cube_axes_min_z_extent(orientations).numpy()
    centres = positions.numpy()
    bores = _bore_radii(positions, labels, plates, size, wall)

    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, plate in enumerate(plates):
        here = torch.nonzero(labels == index, as_tuple=True)[0].numpy()
        body = _hull_body(centres[here][:, :2], wall + size * SQRT3 / 2,
                          plate["z_min"], plate["z_max"],
                          min_radius=(outer_radius_m or {}).get(index))
        body = _robust_difference(body, [_bore(*bores[index], plate["z_min"], plate["z_max"])])

        cutters = []
        for magnet in here:
            cutters.append(_pocket_and_channel(centres[magnet], axes[magnet], size,
                                               plate["z_max"], clearance))
            ejector = _ejection_hole(centres[magnet], axes[magnet], size,
                                     plate["z_min"], ejector_radius)
            if ejector is not None:
                cutters.append(ejector)
        body = _robust_difference(body, cutters)

        raw_name = f"shell_layer_{index}.stl"
        raw = body.copy()
        raw.apply_scale(1e3)
        raw.export(raw_dir / raw_name)

        trimmed = trimesh.intersections.slice_mesh_plane(
            body, plane_normal=[0, 0, -1],
            plane_origin=[0, 0, plate["z_max"] - lid_gap], cap=True)
        trimmed.apply_scale(1e3)
        trimmed.export(out_dir / raw_name)

        # Checked on the written file, since the float32 export can weld vertices.
        written = trimesh.load(out_dir / raw_name)
        entries.append({
            "file": raw_name,
            "raw_file": f"raw/{raw_name}",
            "z_center_mm": float(plate["height"]) * 1e3,
            "n_magnets": int(len(here)),
            "volume_cm3": float(written.volume) / 1e3,
            "layer": index,
            "shim": False,
            "frozen_as_printed": False,
            "needs_lid": True,
            "watertight": bool(written.is_watertight),
            "broken_faces": int(len(trimesh.repair.broken_faces(written))),
            "z_min_mm": plate["z_min"] * 1e3,
            "z_max_mm": plate["z_max"] * 1e3,
            "lid_bot_mm": (plate["z_max"] - lid_gap) * 1e3,
            "floor_mm": plate["floor"] * 1e3,
            "spacer_below_mm": plate["spacer_below"] * 1e3,
            "roof_mm": plate["roof"] * 1e3,
            "r_outer_mm": float(np.linalg.norm(written.vertices[:, :2], axis=1).max()),
        })

    manifest = {
        "variant": variant,
        "magnet_size_mm": size * 1e3,
        "clearance_mm": clearance * 1e3,
        "lid_gap_mm": lid_gap * 1e3,
        "wall_mm": wall * 1e3,
        "n_magnets": int(len(positions)),
        "slabs": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
