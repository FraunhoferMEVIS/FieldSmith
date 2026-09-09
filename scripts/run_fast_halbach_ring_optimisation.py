# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file runs a fast single-ring Halbach angle optimization.
#
#  run_fast_halbach_ring_optimisation.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.geometry import HalbachRingGeometry
from fieldsmith.geometry.halbach_ring import make_spherical_mask
from fieldsmith.logger import WandbLogger
from fieldsmith.metrics import field_strength_loss, homogeneity_loss
from fieldsmith.models.halbach_ring import SingleRingHalbachModel
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize one Halbach ring with the fast point B-field kernel.")
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--n-magnets", type=int, default=36)
    parser.add_argument("--ring-radius", type=float, default=0.135)
    parser.add_argument("--fov-radius", type=float, default=0.0675)
    parser.add_argument("--magnet-size", type=float, default=0.012)
    parser.add_argument("--remanence", type=float, default=1.3)
    parser.add_argument("--expected-field", type=float, default=0.005)
    parser.add_argument("--double-precision", action="store_true", help="Run in float64; recommended below 1000 ppm.")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--log-each-n-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="fieldsmith")
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=("online", "offline", "disabled"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--symmetric-optimization",
        action="store_true",
        help="Share each angular update with the magnet at phi + pi, optimizing N/2 parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d/%Hh%Mm%Ss_fast_halbach_ring")
    log_dir = Path("logs") / timestamp
    config = vars(args)
    dtype = torch.float64 if args.double_precision else torch.float32

    geometry = HalbachRingGeometry(
        n_magnets=args.n_magnets,
        ring_radius=args.ring_radius,
        magnet_size=args.magnet_size,
        remanence=args.remanence,
        device=args.device,
        dtype=dtype,
    )
    initial_configuration = geometry.build()
    grid_sampler = CartesianGridSampler(
        fov=(0.2, 0.2, 0.0),
        resolution=(0.005, 0.005, 0.005),
        device=args.device,
        dtype=dtype,
    )
    point_sampler = MaskedPointSampler(
        base_sampler=grid_sampler,
        mask_fn=lambda points: make_spherical_mask(points, args.fov_radius),
    )

    print("Using symmetry", args.symmetric_optimization)
    model = SingleRingHalbachModel(
        configuration=initial_configuration,
        device=args.device,
        dtype=dtype,
        point_sampler=point_sampler,
        chunk_points=32768,
        symmetric_optimization=args.symmetric_optimization,
    )
    print(f"FOV evaluation points: {point_sampler.get_points().shape[0]}")
    print(f"Trainable magnet angles: {model.magnet_angles.numel()}")

    optimizer = torch.optim.Adam([{"params": model.magnet_angles, "lr": args.lr, "name": "magnet_angles"}])


    def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
        return homogeneity_loss(output["B"])

    logger = WandbLogger(
        project=args.wandb_project,
        run_name=timestamp.replace("\\", "/"),
        log_dir=log_dir,
        config=config,
        mode=args.wandb_mode,
    )
    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        logger=logger,
        log_each_n_epochs=args.log_each_n_epochs,
        log_dir=log_dir,
    )
    loop.run()


if __name__ == "__main__":
    main()
