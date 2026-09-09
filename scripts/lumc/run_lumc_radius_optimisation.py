"""Optimize one physical LUMC Halbach system's ring radii with Adam."""

from __future__ import annotations

import argparse
from copy import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation.lumc_halbach import INNER_RING_RADII, create_configuration
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.metrics import (
    component_field_strength,
    component_homogeneity_loss,
    component_peak_to_peak_ppm,
    relative_field_strength_shortfall_loss,
)

OUTER_RADIUS_OFFSET_M = 21e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize one LUMC Halbach system's ring radii with Adam.")
    parser.add_argument("--num-epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resolution-mm", type=float, default=5.0)
    parser.add_argument("--dsv-mm", type=float, default=200.0)
    parser.add_argument("--num-rings", type=int, default=23)
    parser.add_argument("--ring-separation-m", type=float, default=0.022)
    parser.add_argument("--magnet-size-m", type=float, default=0.012)
    parser.add_argument("--remanence-t", type=float, default=1.3)
    parser.add_argument("--chunk-points", type=int, default=32768)
    parser.add_argument("--log-each-n-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--seed-file", type=Path, default=Path("data/random_seeds_50.txt"))
    parser.add_argument(
        "--initial-vector",
        type=str,
        default=None,
        help="Comma-separated radius-table indices for nonnegative z positions; random when omitted.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def parse_initial_vector(value: str | None, size: int) -> list[int]:
    """Create one physical initial system using the GA's discrete initialization domain."""
    if value is None:
        return [random.randint(0, INNER_RING_RADII.size - 1) for _ in range(size)]
    vector = [int(item.strip()) for item in value.split(",")]
    if len(vector) != size:
        raise ValueError(f"initial vector must contain {size} indices, received {len(vector)}.")
    if any(index < 0 or index >= INNER_RING_RADII.size for index in vector):
        raise ValueError(f"initial-vector indices must be between 0 and {INNER_RING_RADII.size - 1}.")
    return vector


class LUMCRadiusModel(nn.Module):
    """Differentiable single-system model with fixed magnet counts and trainable radii."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        initial_inner_radii: torch.Tensor,
        points: torch.Tensor,
        chunk_points: int | None,
    ) -> None:
        super().__init__()
        radial_norm = torch.linalg.vector_norm(configuration.positions[:, :2], dim=1)
        ring_indices = configuration.metadata["ring_indices"].to(device=points.device)
        initial_per_magnet = initial_inner_radii[ring_indices]
        outer_offsets = torch.where(
            torch.isclose(radial_norm, initial_per_magnet + OUTER_RADIUS_OFFSET_M),
            OUTER_RADIUS_OFFSET_M,
            0.0,
        )
        minimum = float(INNER_RING_RADII.min())
        maximum = float(INNER_RING_RADII.max())
        self.inner_ring_radii: nn.Parameter = nn.Parameter(initial_inner_radii.clone())
        self.minimum_radius: float = minimum
        self.maximum_radius: float = maximum
        self.chunk_points: int | None = chunk_points
        self.remanence: float = float(configuration.remanence)
        self.register_buffer("radial_units", configuration.positions[:, :2] / radial_norm[:, None])
        self.register_buffer("z_positions", configuration.positions[:, 2])
        self.register_buffer("orientations", configuration.orientations)
        self.register_buffer("volumes", configuration.volumes)
        self.register_buffer("ring_indices", ring_indices)
        self.register_buffer("outer_offsets", outer_offsets)
        self.register_buffer("points", points)

    def inner_radii(self) -> torch.Tensor:
        """Return bounded, continuous inner radii in metres."""
        return self.inner_ring_radii

    def constrain_parameters(self) -> None:
        """Keep optimized radii inside the original radius-table range."""
        self.inner_ring_radii.clamp_(self.minimum_radius, self.maximum_radius)

    def magnet_positions(self) -> torch.Tensor:
        """Build differentiable magnet positions for the current radii."""
        magnet_radii = self.inner_radii()[self.ring_indices] + self.outer_offsets
        xy_positions = self.radial_units * magnet_radii[:, None]
        return torch.cat((xy_positions, self.z_positions[:, None]), dim=1)

    def forward(self) -> torch.Tensor:
        """Evaluate Bx at the original octant DSV points."""
        field = calc_b_at_points_fast(
            self.orientations * self.remanence,
            self.volumes,
            self.magnet_positions(),
            self.points,
            chunk_points=self.chunk_points,
        )
        return field[:, 0]

    def get_configuration(self) -> MagnetConfiguration:
        """Return the current physical magnet configuration."""
        return MagnetConfiguration(
            positions=self.magnet_positions().detach(),
            orientations=self.orientations.detach(),
            volumes=self.volumes.detach(),
            remanence=self.remanence,
            metadata={
                "geometry": "lumc_halbach_continuous_radius_optimisation",
                "selected_inner_outer_radii_m": [
                    (float(radius), float(radius + OUTER_RADIUS_OFFSET_M))
                    for radius in self.inner_radii().detach().cpu().tolist()
                ],
                "ring_indices": self.ring_indices.detach().cpu(),
            },
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before constructing the system."""
    if args.num_epochs <= 0 or args.lr <= 0.0:
        raise ValueError("num epochs and learning rate must be positive.")
    if args.num_rings <= 0 or args.num_rings % 2 == 0:
        raise ValueError("num rings must be a positive odd number.")
    if args.resolution_mm <= 0.0 or args.dsv_mm <= 0.0:
        raise ValueError("resolution and DSV must be positive.")
    if args.log_each_n_epochs <= 0:
        raise ValueError("log interval must be positive.")


def load_run_seeds(seed_file: Path, num_runs: int) -> list[int]:
    """Load the requested number of distinct integer seeds from a text file."""
    seeds = [int(line.strip()) for line in seed_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed file contains duplicate values: {seed_file}")
    if num_runs > len(seeds):
        raise ValueError(f"requested {num_runs} runs, but {seed_file} contains only {len(seeds)} seeds.")
    return seeds[:num_runs]


def run_optimization(args: argparse.Namespace) -> dict[str, object]:
    """Run one Adam optimization and return its saved summary."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach_adam"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    dsv = args.dsv_mm * 1e-3
    coordinate_axis = np.linspace(-dsv / 2.0, dsv / 2.0, int(args.dsv_mm / args.resolution_mm + 1))
    coords = np.meshgrid(coordinate_axis, coordinate_axis, coordinate_axis)
    mask = np.square(coords[0]) + np.square(coords[1]) + np.square(coords[2]) <= (dsv / 2.0) ** 2
    octant_mask = mask & (coords[0] >= 0.0) & (coords[1] >= 0.0) & (coords[2] >= 0.0)
    points = torch.as_tensor(
        np.stack((coords[0][octant_mask], coords[1][octant_mask], coords[2][octant_mask]), axis=1),
        device=args.device,
        dtype=dtype,
    )
    ring_positions = np.linspace(
        -(args.num_rings - 1) * args.ring_separation_m / 2.0,
        (args.num_rings - 1) * args.ring_separation_m / 2.0,
        args.num_rings,
    )
    symmetric_positions = ring_positions[ring_positions >= 0.0]
    initial_vector = parse_initial_vector(args.initial_vector, symmetric_positions.size)
    initial_configuration = create_configuration(
        initial_vector,
        symmetric_positions,
        args.magnet_size_m,
        args.remanence_t,
        args.device,
        dtype,
    )
    initial_radii = torch.as_tensor(
        INNER_RING_RADII[initial_vector], device=args.device, dtype=dtype
    )
    model = LUMCRadiusModel(initial_configuration, initial_radii, points, args.chunk_points).to(args.device)
    optimizer = torch.optim.Adam(
        [{"params": model.inner_ring_radii, "lr": args.lr, "name": "inner_ring_radii"}]
    )

    print(f"Initial vector: {initial_vector}")
    print(f"Physical systems optimized: 1; trainable radii: {model.inner_ring_radii.numel()}")
    with torch.no_grad():
        initial_field_strength = component_field_strength(model())
    print(f"Field-strength reference: {float(initial_field_strength * 1e3):.3f} mT (initial design)")
    tracker = np.zeros(args.num_epochs)
    history: list[dict[str, int | float | str | None]] = []
    best_loss = float("inf")
    best_error = float("inf")
    best_radii = None
    best_field_strength_mt = 0.0
    optimization_start = time.perf_counter()

    for epoch in range(args.num_epochs):
        field_x = model()
        homogeneity = component_homogeneity_loss(field_x)
        strength_loss = relative_field_strength_shortfall_loss(field_x, initial_field_strength)
        loss = homogeneity + strength_loss
        # loss = homogeneity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        current_loss = float(loss.detach().item())
        current_error = float(component_peak_to_peak_ppm(field_x.detach()).item())
        current_field_strength_mt = float(component_field_strength(field_x.detach()).item() * 1e3)
        tracker[epoch] = current_error
        if current_loss < best_loss:
            best_loss = current_loss
            best_error = current_error
            best_radii = model.inner_ring_radii.detach().clone()
            best_field_strength_mt = current_field_strength_mt
        history.append(
            {
                "step": epoch + 1,
                "cumulative_objective_evaluations": epoch + 1,
                "elapsed_seconds": time.perf_counter() - optimization_start,
                "step_best_ppm": current_error,
                "step_inner_radii_m": json.dumps(model.inner_radii().detach().cpu().tolist()),
                "best_so_far_ppm": best_error,
                "population_mean_ppm": None,
                "population_std_ppm": None,
                "best_field_strength_mt": best_field_strength_mt,
                "combined_loss": current_loss,
                "field_strength_shortfall_loss": float(strength_loss.detach().item()),
            }
        )
        optimizer.step()
        with torch.no_grad():
            model.constrain_parameters()
        if epoch == 0 or (epoch + 1) % args.log_each_n_epochs == 0:
            radii_mm = model.inner_radii().detach().cpu().numpy() * 1e3
            print(
                f"Epoch {epoch + 1}/{args.num_epochs}: {current_error:.2f} ppm, "
                f"field strength {current_field_strength_mt:.3f} mT, "
                f"inner radii {np.array2string(radii_mm, precision=2)}, "
                f"elapsed {time.perf_counter() - optimization_start:.1f}s"
            )

    assert best_radii is not None
    with torch.no_grad():
        model.inner_ring_radii.copy_(best_radii)
    configuration = model.get_configuration().detached_cpu()
    best_inner_radii = model.inner_radii().detach().cpu().numpy()
    torch.save(configuration.to_dict(), output_directory / "best_configuration.pt")
    np.save(output_directory / "best_inner_radii_m.npy", best_inner_radii)
    np.save(output_directory / "ppm_by_epoch.npy", tracker)
    with (output_directory / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "optimizer": "Adam",
        "best_peak_to_peak_ppm": best_error,
        "best_combined_loss": best_loss,
        "best_field_strength_mt": best_field_strength_mt,
        "initial_field_strength_mt": float(initial_field_strength.item() * 1e3),
        "initial_vector": initial_vector,
        "best_inner_radii_m": best_inner_radii.tolist(),
        "best_outer_radii_m": (best_inner_radii + OUTER_RADIUS_OFFSET_M).tolist(),
        "fixed_magnet_counts": True,
        "n_magnets": configuration.num_magnets,
        "elapsed_seconds": time.perf_counter() - optimization_start,
        "objective": "relative_peak_to_peak_Bx_plus_initial_field_shortfall",
        "sampling_point_count": int(points.shape[0]),
        "objective_evaluations": args.num_epochs,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    figure, axis = plt.subplots()
    axis.semilogy(tracker)
    axis.set_title("Adam continuous-radius optimization")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Peak-to-peak error (ppm)")
    figure.savefig(output_directory / "convergence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation-ready configuration to {(output_directory / 'best_configuration.pt').resolve()}")
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.num_runs <= 0:
        raise ValueError("number of runs must be positive.")
    if args.seed is not None and args.num_runs != 1:
        raise ValueError("--seed is a single-run override; use it together with --num-runs 1.")
    seeds = [args.seed] if args.seed is not None else load_run_seeds(args.seed_file, args.num_runs)
    if len(seeds) == 1:
        run_args = copy(args)
        run_args.seed = seeds[0]
        run_optimization(run_args)
        return

    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach_adam_runs"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for run_index, seed in enumerate(seeds, start=1):
        run_args = copy(args)
        run_args.seed = seed
        run_args.output_directory = output_directory / f"run_{run_index:02d}_seed_{seed}"
        print(f"Starting run {run_index}/{len(seeds)} with seed {seed}")
        summaries.append(run_optimization(run_args))
    with (output_directory / "runs_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump({"num_runs": len(seeds), "seeds": seeds, "runs": summaries}, output_file, indent=2)
    print(f"Saved {len(seeds)} runs to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
