# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file gives the cubes of a design a small random tilt.
#
#  kick_tilt.py
#  Marian Frei
#  04.09.2026
# -----------------------------------------------------------------------------
"""
Give every cube a small random tilt, bars untouched.

    python scripts/stages/kick_tilt.py DESIGN OUT --sigma-deg 2 --seed 1

A planar, mirror-symmetric design has zero gradient with respect to the cube
tilts, so the descent never leaves the plane without this kick.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fieldsmith.configurations import MagnetConfiguration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("design"); parser.add_argument("out")
    parser.add_argument("--sigma-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    cfg = MagnetConfiguration.from_file(args.design, dtype=torch.float64)
    o = cfg.orientations.numpy().astype(np.float64); o /= np.linalg.norm(o, axis=1, keepdims=True)
    sizes = cfg.metadata["magnet_dimensions_m"].numpy()
    cube = (sizes.max(1) - sizes.min(1)) < 1e-9
    azimuth = np.arctan2(o[:, 1], o[:, 0])
    polar = np.arccos(np.clip(o[:, 2], -1.0, 1.0))
    kick = np.random.default_rng(args.seed).normal(0.0, np.radians(args.sigma_deg), len(o))
    polar = np.where(cube, polar + kick, polar)
    cfg.orientations = torch.tensor(np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], 1),
                                    dtype=torch.float64)
    torch.save(cfg.to_dict(), args.out)
    tilt = np.degrees(np.abs(np.pi / 2 - polar))
    print(f"{args.out}: {int(cube.sum())} cubes kicked by {args.sigma_deg} deg rms (seed {args.seed}), "
          f"max tilt {tilt[cube].max():.2f} deg, bars max tilt {tilt[~cube].max():.4f} deg")


if __name__ == "__main__":
    main()
