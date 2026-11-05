# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file turns a finished design into printable plates, lids and rails.
#
#  export_plates.py
#  Marian Frei
#  16.10.2026
# -----------------------------------------------------------------------------
"""
Turn a finished design into printable plates, lids and fastening rails.

    python scripts/export_plates.py DESIGN.pt OUTDIR [--lid-mm 2] [--rails 6]

Writes one plate per height level with seats, insertion channels and ejector
holes, a recessed lid per plate with retaining pins, inclined fastening rails,
and the insertion order per plate. Where one magnet's channel crosses a
neighbour's seat, the crossed seat is filled first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from fieldsmith.cad.lids import export_lids
from fieldsmith.cad.rails import add_rails
from fieldsmith.cad.slabs import export_slabs
from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry.collision import candidate_pairs, oriented_box_penetration
from fieldsmith.geometry.cube_axes import cube_axes_min_z_extent
from fieldsmith.cad.plates import assign_planes, plate_bounds

SIZE = 0.012
CLEARANCE = 1.0e-4
EJECTOR_RADIUS = 2.15e-3
UPRIGHT_MIN = 0.3


def assembly_order(configuration) -> dict:
    """Channel crossings, ejector holes through neighbouring seats, and an insertion order."""
    positions = configuration.positions.double()
    axes = cube_axes_min_z_extent(configuration.orientations.double())
    labels, _ = assign_planes(positions)
    plates = plate_bounds(positions, configuration.orientations.double(), SIZE, 1.0e-3, 1.4e-3)
    index_i, index_j = candidate_pairs(positions, reach=2.0 * SIZE * 3 ** 0.5)
    seat = positions.new_full((3,), SIZE + CLEARANCE)
    steps = torch.linspace(0.0, 1.0, 12, dtype=positions.dtype)
    crossings, ejector_hits = [], 0
    for a in range(len(positions)):
        level = int(labels[a])
        normals = torch.cat([axes[a], -axes[a]])
        normal = normals[int(normals[:, 2].argmax())]
        across = [row for row in axes[a] if abs(float(row @ normal)) < 0.5][:2]
        neighbours = torch.cat([index_j[index_i == a], index_i[index_j == a]])
        neighbours = neighbours[labels[neighbours] == level]
        if len(across) < 2 or len(neighbours) == 0:
            continue
        n = len(neighbours)
        length = (plates[level]["z_max"] - float(positions[a, 2])) / max(float(normal[2]), 1e-6) + SIZE / 2
        channel = positions[a] + normal * (length / 2 - SIZE / 2)
        depth = oriented_box_penetration(
            channel.expand(n, 3), positions[neighbours], torch.stack([*across, normal]).expand(n, 3, 3),
            axes[neighbours], positions.new_tensor([SIZE + CLEARANCE, SIZE + CLEARANCE, length]).expand(n, 3),
            seat.expand(n, 3))
        for k in torch.nonzero(depth > 0).flatten().tolist():
            # A crossing shallower than half the clearance cannot reach the magnet and does not constrain the order.
            crossings.append({"insert_first": a, "blocks": int(neighbours[k]),
                              "depth_mm": round(float(depth[k]) * 1e3, 2),
                              "within_clearance": bool(float(depth[k]) < CLEARANCE / 2)})
        down = -normal if abs(float(normal[2])) >= UPRIGHT_MIN else positions.new_tensor([0.0, 0.0, -1.0])
        start = positions[a] + down * (SIZE / 2 - 1.0e-3)
        span = (plates[level]["z_min"] - float(start[2])) / float(down[2]) if float(down[2]) < 0 else 0.0
        hole = start + down * span * steps[:, None]
        local = torch.einsum("nij,ntj->nti", axes[neighbours], hole[None] - positions[neighbours][:, None])
        ejector_hits += int((local.abs() < SIZE / 2 + EJECTOR_RADIUS).all(-1).any(-1).sum())

    first = defaultdict(list)
    for entry in crossings:
        if not entry["within_clearance"]:
            first[entry["insert_first"]].append(entry["blocks"])
    state: dict[int, int] = {}
    sequence: list[int] = []

    def visit(u: int) -> None:
        state[u] = 1
        for v in first[u]:
            if state.get(v) == 1:
                raise SystemExit(f"cycle through magnets {u} and {v}: no assembly order exists")
            if v not in state:
                visit(v)
        state[u] = 2
        sequence.append(u)

    sys.setrecursionlimit(max(10000, 4 * len(positions)))
    for magnet in range(len(positions)):
        if magnet not in state:
            visit(magnet)
    per_plate = defaultdict(list)
    for magnet in reversed(sequence):
        per_plate[int(labels[magnet])].append(magnet)
    return {"n_magnets": len(positions), "channel_crossings": crossings,
            "ejector_through_neighbour_seat": ejector_hits,
            "insertion_order_per_plate": {str(k): v for k, v in sorted(per_plate.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("design", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lid-mm", type=float, default=2.0, help="depth of the lid recess")
    parser.add_argument("--rails", type=int, default=6, help="number of fastening rails, 0 for none")
    args = parser.parse_args()
    configuration = MagnetConfiguration.from_file(args.design, dtype=torch.float64)
    configuration.metadata.setdefault("magnet_size_m", SIZE)
    export_slabs(configuration, args.out, variant=args.out.name, lid_gap=args.lid_mm * 1e-3)
    export_lids(configuration, args.out, variant=args.out.name)
    if args.rails:
        add_rails(args.design, args.out, n_rails=args.rails)
    order = {"design": args.design.name, **assembly_order(configuration)}
    (args.out / "assembly_order.json").write_text(json.dumps(order, indent=1))
    print(f"{args.design.name}: plates, lids{' and rails' if args.rails else ''} in {args.out}, "
          f"{len(order['channel_crossings'])} channel crossings, insertion order written")


if __name__ == "__main__":
    main()
