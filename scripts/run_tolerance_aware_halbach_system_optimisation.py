from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.geometry import HalbachSystemGeometry
from fieldsmith.models import StackedHalbachGeometryModel
from fieldsmith.tolerance_study import EVALUATION_REMANENCE_T, add_study_arguments, run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tolerance-aware optimization of a large stacked Halbach system.")
    parser.add_argument(
        "--initial-configuration", type=Path, default=None,
        help="Start from a saved MagnetConfiguration with mirrored planes of concentric rings; "
             "otherwise the stack is built from the geometry arguments.",
    )
    parser.add_argument("--n-planes", type=int, default=12)
    parser.add_argument("--rings-per-plane", type=int, default=3)
    parser.add_argument("--magnets-per-ring", type=int, default=48, help="Magnets in the innermost ring.")
    parser.add_argument("--additional-magnets-per-ring", type=int, default=7,
                        help="Extra magnets in every further concentric ring.")
    parser.add_argument("--ring-radius", type=float, default=0.15, help="Innermost ring radius in meters.")
    parser.add_argument("--system-length", type=float, default=0.40, help="Distance between the outer planes.")
    parser.add_argument("--magnet-size", type=float, default=0.012)
    parser.add_argument("--remanence", type=float, default=EVALUATION_REMANENCE_T)
    parser.add_argument(
        "--radii-per-plane", action=argparse.BooleanOptionalAction, default=True,
        help="One radius per concentric ring and mirrored plane pair; otherwise one per concentric ring.",
    )
    add_study_arguments(parser)
    parser.set_defaults(
        fov_radius=0.075,
        resolution=0.005,
        train_resolution=0.015,
        min_inner_radius=0.12,
        chunk_samples=8,
        nominal_log_samples=16,
        eval_samples=200,
        wandb_project="halbach-system-magnetopts",
    )
    args = parser.parse_args()
    args.system_name = f"Halbach system ({args.n_planes} planes)"
    args.system_tag = "halbach_system"
    return args


def build_initial(args: argparse.Namespace, device: torch.device) -> MagnetConfiguration:
    if args.initial_configuration is not None:
        initial = MagnetConfiguration.from_file(args.initial_configuration, device=device, dtype=torch.float64)
    else:
        initial = HalbachSystemGeometry(
            n_magnets_per_ring=args.magnets_per_ring,
            n_rings=args.n_planes,
            ring_radius=args.ring_radius,
            system_length=args.system_length,
            magnet_size=args.magnet_size,
            remanence=args.remanence,
            device=device,
            dtype=torch.float64,
            n_rings_per_plane=args.rings_per_plane,
            additional_magnets_per_ring=args.additional_magnets_per_ring,
        ).build()
    initial.metadata.setdefault("magnet_size_m", args.magnet_size)
    return initial


def build_model(args: argparse.Namespace, initial: MagnetConfiguration,
                free: set[str]) -> tuple[nn.Module, list[dict]]:
    model = StackedHalbachGeometryModel(
        initial,
        fov_radius=args.fov_radius,
        optimize_angles="angles" in free,
        optimize_radii="radii" in free,
        optimize_distances="distances" in free,
        radii_per_plane=args.radii_per_plane,
        clearance_m=args.clearance_mm * 1e-3,
        min_inner_radius_m=args.min_inner_radius,
    )
    return model, [
        {"params": [model.magnet_angles_phi], "lr": args.lr, "name": "angles"},
        {"params": [model.radius_slack, model.plane_slack], "lr": args.geometry_lr, "name": "geometry"},
    ]


if __name__ == "__main__":
    run_study(parse_args(), build_initial, build_model)
