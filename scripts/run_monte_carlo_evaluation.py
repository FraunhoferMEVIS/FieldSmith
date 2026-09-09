"""Run a manufacturing-tolerance Monte Carlo evaluation from a saved configuration."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation import plot_monte_carlo_distributions, run_monte_carlo
from fieldsmith.geometry import make_spherical_mask
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of magnet manufacturing tolerances.")
    parser.add_argument("configuration", type=Path, help="MagnetConfiguration checkpoint (.pt).")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.1)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--remanence-tolerance-t", type=float, default=0.005)
    parser.add_argument("--fov-radius", type=float, default=0.05, help="Spherical FOV radius in meters.")
    parser.add_argument("--resolution", type=float, default=0.01, help="FOV grid spacing in meters.")
    parser.add_argument("--field-model", choices=("dipole", "cuboid"), default="cuboid")
    parser.add_argument("--chunk-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_monte_carlo"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration = MagnetConfiguration.from_file(
        args.configuration, device=args.device, dtype=torch.float64
    )
    sampler = MaskedPointSampler(
        CartesianGridSampler(
            fov=(2.0 * args.fov_radius,) * 3,
            resolution=(args.resolution,) * 3,
            device=args.device,
            dtype=torch.float64,
        ),
        mask_fn=lambda points: make_spherical_mask(points, args.fov_radius),
    )
    points = sampler.get_points()
    print(f"Loaded {configuration.num_magnets} magnets and {points.shape[0]} FOV points.")
    result = run_monte_carlo(
        configuration=configuration,
        points=points,
        num_samples=args.num_samples,
        position_tolerance_m=args.position_tolerance_mm * 1e-3,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        remanence_tolerance_t=args.remanence_tolerance_t,
        field_model=args.field_model,
        chunk_points=args.chunk_points,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    summary = result.summary()
    with (output_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with (output_directory / "samples.csv").open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.writer(samples_file)
        writer.writerow(("fieldstrength_mT", "peak_to_peak_ppm", "rms_ppm"))
        writer.writerows(zip(result.fieldstrength_mT.tolist(), result.peak_to_peak_ppm.tolist(), result.rms_ppm.tolist()))
    torch.save(result, output_directory / "results.pt")
    figure = plot_monte_carlo_distributions(result, bins=args.bins)
    figure.savefig(output_directory / "distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
