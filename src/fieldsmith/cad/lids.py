# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file builds the lid that closes a plate: a plate-shaped cover carrying
# one pin per magnet, each cut to sit flat on its own tilted magnet.
#
#  lids.py
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

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry.cube_axes import cube_axes_min_z_extent
from fieldsmith.cad.plates import assign_planes

# Grown outward so the lid overlaps the plate wall rather than balancing on it.
OUTLINE_GROW_M = 1.0e-3
# Anything larger than this in the section is the patient bore, not a pocket.
BORE_MIN_AREA_M2 = 3.0e-3
# Collapses near-coincident vertices left by sectioning, far below print resolution.
SIMPLIFY_M = 1.0e-5
PIN_RADIUS_M = 2.5e-3
# The pin reaches this far into the magnet face, which preloads the magnet.
PIN_INTERFERENCE_M = 2.0e-4
# How far below the magnet face the pin blank starts before it is cut back.
PIN_BLANK_DROP_M = 4.0e-3
# Below this the magnet leans too far for a pin to press on its face squarely.
UPRIGHT_MIN = 0.3


def _outline_at(mesh: trimesh.Trimesh, height: float) -> shapely.Polygon | None:
    """
    Return the plate outline at magnet height with the pockets filled in.

    Sectioning at magnet level rather than at the lid plane keeps the outermost
    pins over material, since the plates taper upward. Only the bore is left
    open. Returns None if the plate does not reach that height.
    """
    section = mesh.section(plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, height])
    if section is None:
        return None
    planar, _ = section.to_2D(to_2D=np.eye(4))
    polygons = [polygon for polygon in planar.polygons_full if polygon.area > 0]
    if not polygons:
        return None
    largest = max(polygons, key=lambda polygon: polygon.area)
    outline = shapely.Polygon(largest.exterior).buffer(OUTLINE_GROW_M)
    for interior in largest.interiors:
        hole = shapely.Polygon(interior)
        if hole.area > BORE_MIN_AREA_M2:
            outline = outline.difference(hole)
    if outline.is_empty:
        return None
    # Without this the triangulation yields slivers and every later boolean fails.
    return outline.simplify(SIMPLIFY_M).buffer(0)


def _pin(centre: np.ndarray, axes: np.ndarray, size: float, lid_bottom: float) -> trimesh.Trimesh | None:
    """
    Return the pin that presses one magnet down, cut to its top face.

    Returns None when the magnet leans too far to be pressed squarely or
    already reaches the lid.
    """
    normals = np.concatenate([axes, -axes])
    normal = normals[int(np.argmax(normals[:, 2]))]
    if normal[2] < UPRIGHT_MIN:
        return None
    face_centre = centre + normal * (size / 2)
    # Height at which a vertical line through the magnet leaves its top face.
    exit_height = centre[2] + (size / 2) / normal[2]
    if exit_height - PIN_INTERFERENCE_M >= lid_bottom - 1.0e-4:
        return None                                          # the magnet is already there
    blank = trimesh.creation.cylinder(
        radius=PIN_RADIUS_M, height=lid_bottom - (exit_height - PIN_BLANK_DROP_M), sections=24)
    blank.apply_translation((centre[0], centre[1],
                             (exit_height - PIN_BLANK_DROP_M + lid_bottom) / 2))
    cut = trimesh.intersections.slice_mesh_plane(
        blank, plane_normal=normal,
        plane_origin=face_centre - normal * PIN_INTERFERENCE_M, cap=True)
    return cut if len(cut.faces) else None


def export_lids(configuration: MagnetConfiguration, slab_dir: Path, variant: str = "default") -> dict:
    """
    Write one lid per plate, built against the untrimmed plates in slab_dir.

    Each lid drops into its plate's recess, finishes flush and presses every
    magnet down through a pin. Writes the lid STLs and their manifest.
    """
    manifest = json.loads((slab_dir / "manifest.json").read_text())
    size = float(configuration.metadata.get("magnet_size_m") or 0.012)
    positions = configuration.positions.double()
    orientations = configuration.orientations.double()
    labels, _ = assign_planes(positions)
    axes = cube_axes_min_z_extent(orientations).numpy()
    centres = positions.numpy()

    lid_dir = slab_dir / "lids"
    lid_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for plate in manifest["slabs"]:
        index = plate["layer"]
        raw = trimesh.load(slab_dir / plate["raw_file"])
        raw.apply_scale(1e-3)                                # back to meters to work in
        lid_bottom = plate["lid_bot_mm"] * 1e-3
        outline = _outline_at(raw, plate["z_center_mm"] * 1e-3)
        if outline is None:
            continue
        body = trimesh.creation.extrude_polygon(outline, height=plate["z_max_mm"] * 1e-3 - lid_bottom)
        body.apply_translation((0.0, 0.0, lid_bottom))

        here = torch.nonzero(labels == index, as_tuple=True)[0].numpy()
        pins = [pin for pin in (_pin(centres[i], axes[i], size, lid_bottom) for i in here)
                if pin is not None]
        if pins:
            merged = None
            try:
                merged = trimesh.boolean.union([body, *pins])
            except Exception:
                merged = None
            # A failed union may return the bare cover without raising. The pins
            # reach below the lid, so a body that does not was missing them.
            if merged is None or merged.bounds[0][2] > lid_bottom - 1.0e-4:
                merged = body
                for pin in pins:
                    try:
                        merged = trimesh.boolean.union([merged, pin])
                    except Exception:
                        continue
            body = merged

        body.apply_scale(1e3)
        name = f"lid_layer_{index}.stl"
        body.export(lid_dir / name)
        written = trimesh.load(lid_dir / name)
        entries.append({
            "file": name, "layer": index, "n_pins": len(pins), "n_magnets": plate["n_magnets"],
            "z_center_mm": plate["z_center_mm"], "lid_bot_mm": plate["lid_bot_mm"],
            "volume_cm3": float(written.volume) / 1e3,
            "watertight": bool(written.is_watertight),
            "broken_faces": int(len(trimesh.repair.broken_faces(written))),
        })

    lid_manifest = {"variant": variant, "lid_gap_mm": manifest["lid_gap_mm"], "lids": entries}
    (lid_dir / "manifest.json").write_text(json.dumps(lid_manifest, indent=2))
    return lid_manifest
