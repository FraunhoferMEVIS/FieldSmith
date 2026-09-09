"""Minimal end-to-end example for optimizing a stacked Halbach system."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.evaluation import evaluate_configuration
from fieldsmith.geometry import HalbachSystemGeometry, make_spherical_mask
from fieldsmith.logger import WandbLogger
from fieldsmith.models.halbach_systems import HalbachSystemAngleModel
from fieldsmith.objectives import SmoothedMinimaxObjective
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop


def parse_args() -> argparse.Namespace:
    """Parse the options useful when trying the example."""
    parser = argparse.ArgumentParser(
        description="Build and optimize a generic three-ring Halbach system."
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Build, evaluate, optimize, and re-evaluate a Halbach system."""
    args = parse_args()
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    dtype = torch.float64
    fov = (0.1, 0.1, 0.1)
    resolution = (0.01, 0.01, 0.01)
    fov_radius = 0.05
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_example_system"
    )

    configuration = HalbachSystemGeometry(
        n_magnets_per_ring=36,
        n_rings=10,
        ring_radius=0.135,
        system_length=0.20,
        magnet_size=0.012,
        remanence=1.3,
        device=args.device,
        dtype=dtype,
    ).build()
    grid = CartesianGridSampler(
        fov=fov,
        resolution=resolution,
        device=args.device,
        dtype=dtype,
    )
    sampler = MaskedPointSampler(
        grid,
        lambda points: make_spherical_mask(points, radius=fov_radius),
    )
    points = sampler.get_points()

    print("Initial metrics:", evaluate_configuration(configuration, points, "dipole"))

    model = HalbachSystemAngleModel(
        configuration=configuration,
        fov=fov,
        resolution=resolution,
        fov_radius=fov_radius,
        device=args.device,
        dtype=dtype,
        point_sampler=sampler,
        field_model="dipole",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    logger = WandbLogger(
        project="fieldsmith",
        run_name="example-system",
        log_dir=output_directory,
        mode="disabled",
    )
    TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=SmoothedMinimaxObjective(alpha=2_000.0),
        num_epochs=args.num_epochs,
        logger=logger,
        log_each_n_epochs=args.num_epochs,
        log_dir=output_directory,
    ).run()

    optimized = model.get_configuration()
    print("Optimized metrics:", evaluate_configuration(optimized, points, "dipole"))
    print(f"Checkpoints saved to {(output_directory / 'configurations').resolve()}")


if __name__ == "__main__":
    main()
