# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file builds the inputs of the ROMA experiments from the ring files.
#
#  build_roma_inputs.py
#  Marian Frei
#  16.10.2026
# -----------------------------------------------------------------------------
"""
Build every input of the ROMA experiments from the published ring files.

    python scripts/build_roma_inputs.py [--out data/roma] [--force]

Writes, from data/osii_rc2.1 alone:

    roma_true.pt                  the published arrangement, 1,936 cubes and 384 bars
    roma_lsq_inplane_start.pt     its constrained least-squares seed, in the plane
    roma_open90_kept.pt           the arrangement with a 90 degree sector removed
    roma_open90_planar_seed.pt    the opened arrangement's in-plane seed

The closed seed is solved on every 16th point of the pinned set
data/roma/roma_roi_200.npz. The point order matters, so it must be that file.
The opened seed is solved on a 115 mm sphere at 4 mm spacing, offset 0.37 mm,
thinned to about 4,000 points.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry.lsq_seed import lsq_seed_orientations
from fieldsmith.geometry.roma import RomaGeometry

ROOT = Path(__file__).resolve().parents[1]
REMANENCE_T = 1.31
TARGET = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)


def subset(base: MagnetConfiguration, keep: torch.Tensor, orientations: torch.Tensor | None = None):
    """The magnets selected by keep, optionally with new orientations."""
    return MagnetConfiguration(
        positions=base.positions[keep].clone(),
        orientations=(base.orientations[keep] if orientations is None else orientations).clone(),
        volumes=base.volumes[keep].clone(), remanence=base.remanence,
        metadata={"magnet_size_m": 0.012, "magnet_dimensions_m": base.metadata["magnet_dimensions_m"][keep].clone()})


def sphere(radius_mm: float, step_mm: float = 4.0, offset_mm: float = 0.37) -> np.ndarray:
    """Grid points in a sphere, shifted off the planes the magnet faces lie in."""
    step = step_mm * 1e-3
    n = int(math.ceil(radius_mm * 1e-3 / step))
    g = np.arange(-n, n + 1) * step + offset_mm * 1e-3
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    return q[np.linalg.norm(q, axis=1) <= radius_mm * 1e-3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rings", type=Path, default=ROOT / "data" / "osii_rc2.1")
    parser.add_argument("--roi", type=Path, default=ROOT / "data" / "roma" / "roma_roi_200.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "roma")
    parser.add_argument("--wedge", type=float, default=90.0, help="opening in degrees")
    parser.add_argument("--centre", type=float, default=90.0, help="azimuth of the opening in degrees")
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = parser.parse_args()

    built = RomaGeometry(args.rings, REMANENCE_T, device="cpu", dtype=torch.float64).build()
    reference = MagnetConfiguration(
        positions=built.positions, orientations=built.orientations, volumes=built.volumes,
        remanence=REMANENCE_T,
        metadata={"magnet_size_m": 0.012, "magnet_dimensions_m": built.metadata["magnet_dimensions_m"]})
    every = torch.ones(len(reference.positions), dtype=torch.bool)

    points = torch.tensor(np.load(args.roi)["points"][::16], dtype=torch.float64)
    closed_seed = lsq_seed_orientations(positions=reference.positions, roi_points=points,
                                        target_direction=TARGET, volumes=reference.volumes, in_plane=True)

    pos = reference.positions.numpy()
    phi = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
    keep = torch.from_numpy(np.abs((phi - args.centre + 180.0) % 360.0 - 180.0) > args.wedge / 2.0)
    kept = subset(reference, keep)
    q = sphere(115.0)
    open_seed = lsq_seed_orientations(positions=kept.positions, roi_points=torch.tensor(q[::max(1, len(q) // 4000)]),
                                      target_direction=TARGET, volumes=kept.volumes, in_plane=True)
    tag = f"open{int(args.wedge)}"
    print(f"reference {len(pos):,} magnets, the opening keeps {int(keep.sum()):,}", flush=True)

    for name, configuration in ((("roma_true.pt", subset(reference, every)),
                                 ("roma_lsq_inplane_start.pt", subset(reference, every, closed_seed)),
                                 (f"roma_{tag}_kept.pt", kept),
                                 (f"roma_{tag}_planar_seed.pt", subset(reference, keep, open_seed)))):
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / name).exists() and not args.force:
            # Built inputs are kept, so runs that follow start from the same files.
            print(f"  kept the existing {args.out / name}", flush=True)
            continue
        torch.save(configuration.to_dict(), args.out / name)
        print(f"  wrote {args.out / name}", flush=True)


if __name__ == "__main__":
    main()
