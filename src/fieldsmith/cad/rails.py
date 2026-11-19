# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file adds inclined fastening rails to an exported shell set.
#
#  rails.py
#  Marian Frei
#  26.08.2026
# -----------------------------------------------------------------------------
"""
Inclined fastening rails for exported plates.

At every interface brass DIN 84 M4 screws pass through the upper plate's
rail segment into brass nuts captured in side pockets of the segment below.
Each cut is moved along the rail until it breaches no magnet pocket.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import trimesh

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.cad.plates import assign_planes, plate_bounds

MU0 = 4e-7 * math.pi
SIZE = 0.012
RAIL_W_MM = 19.0
PAD_MM = 12.5               # the screw lane must clear the pocket zone
CBORE_R_MM = 3.9            # DIN 84 M4 head is 7.0 across
CBORE_DEPTH_MM = 6.0
SHAFT_R_MM = 2.2
OFFSET_MM = 4.7
CEIL_MM = 3.0               # solid material the nut bears up against
NUT_H_MM = 3.2              # DIN 934 M4
SLOT_W_MM = 7.4             # across flats 7.0 plus play, blocks rotation
SLOT_H_MM = 3.6
NUT_CORNER_MM = 4.05        # half of 8.1 across corners
RELIEF_R_MM = 2.4
STANDARD_LENGTHS_MM = (12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0)
TIP_PAST_NUT_MM = 1.0       # the tip must clear the nut at least this far
WORKING_LOAD_N = 150.0
ULTIMATE_LOAD_N = 450.0
NUT = "DIN 934 brass M4 nut"


def magnet_forces(configuration) -> torch.Tensor:
    """Dipole force on every magnet from all others, Newtons, (M, 3)."""
    br = configuration.remanence_vectors().double()
    volumes = configuration.volumes.double()
    moments = br * volumes[:, None] / MU0
    sources = configuration.positions.double().detach()
    eval_pos = configuration.positions.double().clone().requires_grad_(True)
    diff = eval_pos[:, None, :] - sources[None, :, :]
    dist = diff.norm(dim=-1)
    eye = torch.eye(len(sources), dtype=torch.bool)
    dist = dist.masked_fill(eye, 1.0)
    unit = diff / dist[..., None]
    mdotu = (moments[None, :, :] * unit).sum(-1)
    field = (MU0 / (4 * math.pi)) * (3 * unit * mdotu[..., None]
                                     - moments[None, :, :]) / dist[..., None] ** 3
    field = field.masked_fill(eye[..., None], 0.0)
    energy = (moments * field.sum(1)).sum()
    (force,) = torch.autograd.grad(energy, eval_pos)
    return force


def _rail_segment(angle, r_in, r_out_lo, r_out_hi, z_lo, z_hi):
    w = RAIL_W_MM / 2
    points = [(r_in, -w, z_lo), (r_in, w, z_lo), (r_in, -w, z_hi), (r_in, w, z_hi),
              (r_out_lo, -w, z_lo), (r_out_lo, w, z_lo),
              (r_out_hi, -w, z_hi), (r_out_hi, w, z_hi)]
    solid = trimesh.convex.convex_hull(np.array(points))
    solid.apply_transform(trimesh.transformations.rotation_matrix(angle, (0, 0, 1)))
    return solid


def _cyl(x, y, z0, z1, r):
    body = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=24)
    body.apply_translation([x, y, (z0 + z1) / 2])
    return body


def _slot(angle, off, r_inner, r_outer, z0, z1):
    """The nut pocket: a radial slot entered from the rail's outer face."""
    body = trimesh.creation.box((r_outer - r_inner, SLOT_W_MM, z1 - z0))
    body.apply_translation((r_inner + (r_outer - r_inner) / 2, off, (z0 + z1) / 2))
    body.apply_transform(trimesh.transformations.rotation_matrix(angle, (0, 0, 1)))
    return body


def _cyl_points(x, y, z0, z1, r):
    pts = []
    for rho in (0.0, 0.55 * r, 0.92 * r):
        for j in range(1 if rho == 0.0 else 8):
            a = 2 * math.pi * j / 8
            for z in np.linspace(z0 + 0.15, z1 - 0.15, 4):
                pts.append((x + rho * math.cos(a), y + rho * math.sin(a), z))
    return pts


def _slot_points(angle, off, r_inner, r_outer, z0, z1):
    pts = []
    rot = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    for r in np.linspace(r_inner + 0.2, r_outer - 0.2, 6):
        for t in np.linspace(-SLOT_W_MM / 2 + 0.2, SLOT_W_MM / 2 - 0.2, 4):
            for z in np.linspace(z0 + 0.15, z1 - 0.15, 3):
                x, y = rot @ (r, t)
                pts.append((x, y, z))
    return pts


def _breaches(points, hull, mesh) -> bool:
    """True when a probe point should be solid plate but is a pocket void."""
    pts = np.asarray(points)
    inside_outline = hull.contains(pts)
    if not inside_outline.any():
        return False
    return bool((~mesh.contains(pts[inside_outline])).any())


def add_rails(design_path: Path, shell_dir: Path, n_rails: int = 6) -> dict:
    """Add fastening rails to the exported set in shell_dir, in place."""
    configuration = MagnetConfiguration.from_file(design_path, dtype=torch.float64)
    configuration.metadata.setdefault("magnet_size_m", SIZE)
    labels, _ = assign_planes(configuration.positions.double())
    plates = plate_bounds(configuration.positions.double(),
                          configuration.orientations.double(), SIZE, 1.0e-3, 1.4e-3)
    L = labels.numpy()
    n_plates = len(plates)
    z_lo = [p["z_min"] * 1e3 for p in plates]
    z_hi = [p["z_max"] * 1e3 for p in plates]

    force = magnet_forces(configuration).numpy()
    lid_load = [float(np.clip(force[L == k, 2], 0, None).sum()) for k in range(n_plates)]
    iface = [0.0] + [float(max(force[L > k, 2].sum(), -force[L <= k, 2].sum(), 0.0))
                     for k in range(n_plates - 1)]

    manifest = json.loads((shell_dir / "manifest.json").read_text())
    by_layer = {slab["layer"]: slab for slab in manifest["slabs"]}
    meshes = {k: trimesh.load(shell_dir / by_layer[k]["file"]) for k in by_layer}
    hulls = {k: meshes[k].convex_hull for k in by_layer}
    wall = {k: float(np.linalg.norm(meshes[k].vertices[:, :2], axis=1).max())
            for k in by_layer}

    slope = (wall[0] - wall[n_plates - 1]) / max(z_hi[-1] - z_lo[0], 1e-9)
    offset = max(wall[k] + PAD_MM + slope * z_hi[k] for k in by_layer)
    rail_r = lambda z: offset - slope * z

    # The shortest standard length whose tip clears every interface's nut.
    need = max(z_hi[k] - z_lo[k] - CBORE_DEPTH_MM + CEIL_MM + NUT_H_MM + TIP_PAST_NUT_MM
               for k in range(1, n_plates))
    length = next((l for l in STANDARD_LENGTHS_MM if l >= need), STANDARD_LENGTHS_MM[-1])
    screw = f"DIN 84 brass M4 x {length:.0f}"

    angles = [2 * math.pi * j / n_rails for j in range(n_rails)]
    adds: dict[int, list] = {k: [] for k in by_layer}
    cuts: dict[int, list] = {k: [] for k in by_layer}
    plan = {"design": design_path.name, "screw": screw, "nut": NUT,
            "rails": n_rails,
            "rail_profile": {"A_mm": round(offset, 1), "slope": round(slope, 4)},
            "screw_length_mm": length,
            "assembly": "slide a nut into each pocket, screw each shell down "
                        "empty onto the one below, then insert its magnets, "
                        "then its lid",
            "interfaces": [], "warnings": []}
    spots_view = {"design": design_path.name, "screw": screw, "nut": NUT,
                  "screw_length_mm": length, "cbore_mm": CBORE_DEPTH_MM,
                  "plates": []}

    for k in by_layer:
        for a in angles:
            adds[k].append(_rail_segment(a, wall[k] - 4.0, rail_r(z_lo[k]),
                                         rail_r(z_hi[k]), z_lo[k], z_hi[k]))

    for k in range(1, n_plates):
        r_base = rail_r(z_lo[k]) - 8.5
        sign = 1 if k % 2 else -1
        z_int = z_lo[k]
        tip = length - (z_hi[k] - z_lo[k] - CBORE_DEPTH_MM)   # depth below interface
        slot_top = z_int - CEIL_MM
        slot_bot = slot_top - SLOT_H_MM
        relief_bot = z_int - tip - 1.5
        screws = []
        for a in angles:
            taken: list[float] = []
            for off0 in (sign * OFFSET_MM, -sign * OFFSET_MM):
                placed = False
                for d_off in (0.0, 1.5, -1.5, 3.0, -3.0):
                    off = off0 + d_off
                    if abs(off) > RAIL_W_MM / 2 - SLOT_W_MM / 2 - 1.0:
                        continue
                    if any(abs(off - t) < SLOT_W_MM + 1.2 for t in taken):
                        continue
                    for d_r in (0.0, 1.2, -1.2, 2.4, -2.4, 3.6):
                        r_s = r_base + d_r
                        x = r_s * math.cos(a) - off * math.sin(a)
                        y = r_s * math.sin(a) + off * math.cos(a)
                        outer = rail_r(slot_bot) + 2.0
                        probe_up = (_cyl_points(x, y, z_hi[k] - CBORE_DEPTH_MM,
                                                z_hi[k], CBORE_R_MM)
                                    + _cyl_points(x, y, z_lo[k], z_hi[k], SHAFT_R_MM))
                        probe_dn = (_cyl_points(x, y, slot_top, z_int, SHAFT_R_MM)
                                    + _slot_points(a, off, r_s - NUT_CORNER_MM - 0.35,
                                                   outer - 2.5, slot_bot, slot_top)
                                    + _cyl_points(x, y, relief_bot, slot_bot, RELIEF_R_MM))
                        if _breaches(probe_up, hulls[k], meshes[k]):
                            continue
                        if _breaches(probe_dn, hulls[k - 1], meshes[k - 1]):
                            continue
                        cuts[k].append(_cyl(x, y, z_hi[k] - CBORE_DEPTH_MM,
                                            z_hi[k] + 0.4, CBORE_R_MM))
                        cuts[k].append(_cyl(x, y, z_lo[k] - 0.4, z_hi[k] + 0.4,
                                            SHAFT_R_MM))
                        cuts[k - 1].append(_cyl(x, y, slot_top - 0.05, z_int + 0.4,
                                                SHAFT_R_MM))
                        cuts[k - 1].append(_slot(a, off, r_s - NUT_CORNER_MM - 0.35,
                                                 outer, slot_bot, slot_top))
                        cuts[k - 1].append(_cyl(x, y, relief_bot, slot_bot + 0.1,
                                                RELIEF_R_MM))
                        screws.append({"x_mm": round(x, 1), "y_mm": round(y, 1)})
                        taken.append(off)
                        placed = True
                        break
                    if placed:
                        break
                if not placed:
                    plan["warnings"].append(
                        f"interface {k - 1}->{k}: no clean spot on the rail at "
                        f"{math.degrees(a):.0f} deg, screw dropped")
        joint = iface[k] + lid_load[k - 1]
        plan["interfaces"].append({
            "interface": f"{k - 1}->{k}", "joint_load_n": round(joint, 1),
            "screws": len(screws), "bolt_radius_mm": round(r_base, 1),
            "nut_ceiling_mm": CEIL_MM,
            "tip_past_nut_mm": round(tip - CEIL_MM - NUT_H_MM, 1),
            "margin_working": round(len(screws) * WORKING_LOAD_N / max(joint, 1e-9), 2),
            "margin_ultimate": round(len(screws) * ULTIMATE_LOAD_N / max(joint, 1e-9), 2)})
        spots_view["plates"].append({
            "plate": k, "z_min_mm": round(z_lo[k], 1), "z_max_mm": round(z_hi[k], 1),
            "joint_load_n": round(joint, 1), "nut_top_mm": round(slot_top, 1),
            "lid_screws": screws, "plate_screws": []})

    for k in by_layer:
        mesh = trimesh.boolean.union([meshes[k]] + adds[k], engine="manifold")
        if cuts[k]:
            mesh = trimesh.boolean.difference([mesh] + cuts[k], engine="manifold")
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices()
        mesh.export(shell_dir / by_layer[k]["file"])

    total = sum(entry["screws"] for entry in plan["interfaces"])
    plan["parts"] = {screw: total, NUT: total}
    if not plan["warnings"]:
        del plan["warnings"]
    (shell_dir / "fastener_plan.json").write_text(json.dumps(plan, indent=1))
    (shell_dir / "screw_spots.json").write_text(json.dumps(spots_view, indent=1))
    return plan
