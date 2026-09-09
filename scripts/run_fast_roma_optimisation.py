# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file runs a fast ROMA magnet angle optimization.
#
#  run_fast_roma_optimisation.py
#  Kostiantyn Lavronenko
#  12.08.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.geometry import RomaGeometry, make_spherical_mask
from fieldsmith.logger import WandbLogger
from fieldsmith.metrics import field_strength_loss, homogeneity_loss
from fieldsmith.models.halbach_systems import HalbachSystemAngleModel
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop

DEFAULT_RING_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "osii_rc2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the magnet angles of the ROMA geometry."
    )
    parser.add_argument("--ring-directory", type=Path, default=DEFAULT_RING_DIRECTORY)
    parser.add_argument("--num-epochs", type=int, default=40000)
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=None,
        help="Optimization time limit in seconds. When provided, --num-epochs is ignored.",
    )
    parser.add_argument("--fov-radius", type=float, default=0.10)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--magnet-size", type=float, default=0.012)
    parser.add_argument("--outer-magnet-length", type=float, default=0.050)
    parser.add_argument(
        "--use-halbach-orientations",
        action="store_true",
        default=False,
        help="Initialize magnet orientations from their positions using the ideal Halbach pattern.",
    )
    parser.add_argument("--remanence", type=float, default=1.31)
    parser.add_argument("--expected-field", type=float, default=0.03)
    parser.add_argument("--field-loss-weight", type=float, default=0.0)
    parser.add_argument("--double-precision", action="store_true", default=True, help="Run in float64; recommended below 1000 ppm.")
    parser.add_argument(
        "--optimize-out-of-plane",
        action="store_true",
        default=True,
        help="Also optimize each magnet's polar angle instead of keeping it in plane.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--chunk-points", type=int, default=4096)
    parser.add_argument(
        "--field-model",
        choices=("dipole", "cuboid"),
        default="dipole",
        help="Magnetic-field formula used during optimization (default: dipole).",
    )
    parser.add_argument("--log-each-n-epochs", type=int, default=50)
    parser.add_argument("--wandb-project", type=str, default="fieldsmith")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d/%Hh%Mm%Ss_fast_roma")
    log_dir = Path("logs") / timestamp
    config = vars(args).copy()
    config["ring_directory"] = str(args.ring_directory)
    dtype = torch.float64 if args.double_precision else torch.float32
    print("Use Halbach Oriientations: ", args.use_halbach_orientations)
    geometry = RomaGeometry(
        ring_directory=args.ring_directory,
        remanence=args.remanence,
        device=args.device,
        dtype=dtype,
        magnet_size=args.magnet_size,
        outer_magnet_length=args.outer_magnet_length,
        use_halbach_orientations=args.use_halbach_orientations,
    )
    initial_configuration = geometry.build()
    config["n_magnets"] = initial_configuration.num_magnets
    print(f"Total magnets: {config['n_magnets']}")

    fov = (2.0 * args.fov_radius,) * 3
    resolution = (args.resolution,) * 3
    grid_sampler = CartesianGridSampler(
        fov=fov,
        resolution=resolution,
        device=args.device,
        dtype=dtype,
    )
    point_sampler = MaskedPointSampler(
        base_sampler=grid_sampler,
        mask_fn=lambda points: make_spherical_mask(points, args.fov_radius),
    )
    print(f"Out of plane optimization: {args.optimize_out_of_plane}")
    model = HalbachSystemAngleModel(
        configuration=initial_configuration,
        fov=fov,
        resolution=resolution,
        fov_radius=args.fov_radius,
        device=args.device,
        dtype=dtype,
        chunk_points=args.chunk_points,
        point_sampler=point_sampler,
        optimize_out_of_plane=args.optimize_out_of_plane,
        field_model=args.field_model,
    )
    config["n_fov_points"] = int(model.points.shape[0])
    config["n_trainable_angles"] = model.magnet_angles.numel()
    if args.optimize_out_of_plane:
        config["n_trainable_angles"] += model.magnet_polar_angles.numel()
    print(f"FOV evaluation points: {config['n_fov_points']}")
    print(f"Trainable magnet angles: {config['n_trainable_angles']}")

    parameter_groups = [
        {"params": model.magnet_angles, "lr": args.lr, "name": "magnet_angles"}
    ]
    if args.optimize_out_of_plane:
        parameter_groups.append(
            {
                "params": model.magnet_polar_angles,
                "lr": args.lr,
                "name": "magnet_polar_angles",
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)

    def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = homogeneity_loss(output["B"])
        if args.field_loss_weight > 0.0:
            loss = loss + args.field_loss_weight * field_strength_loss(
                output["B"], args.expected_field
            )
        return loss

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
        time_limit_seconds=args.time_limit_seconds,
    )
    loop.run()


if __name__ == "__main__":
    main()
