"""Minimal example of tolerance-aware optimization of a stacked Halbach system.

The same system is optimized twice with identical settings: conventionally, on the
unperturbed design only, and tolerance-aware, on randomly perturbed copies of the
design plus the closed-form sensitivity to the tolerances. Magnet angles, ring radii,
and plane heights are free; both keep the initial mean field. Both results are then
evaluated with the same Monte Carlo builds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation import evaluate_configuration, run_monte_carlo
from fieldsmith.geometry import HalbachSystemGeometry, make_spherical_mask
from fieldsmith.logger import WandbLogger
from fieldsmith.models import StackedHalbachGeometryModel, ToleranceAwareModel
from fieldsmith.objectives import RobustObjective, SensitivityRegularized, SmoothedMinimaxObjective
from fieldsmith.perturbation import TolerancePerturbation
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop


def parse_args() -> argparse.Namespace:
    """Parse the options useful when trying the example."""
    parser = argparse.ArgumentParser(
        description="Optimize a generic Halbach system conventionally and tolerance-aware, then compare."
    )
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--num-copies", type=int, default=64, help="Perturbed copies per step.")
    parser.add_argument("--sensitivity-weight", type=float, default=3.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1.2)
    parser.add_argument("--remanence-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--monte-carlo-samples", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def optimize(configuration: MagnetConfiguration, points: torch.Tensor, field_band: tuple[float, float],
             args: argparse.Namespace, tolerance_aware: bool, output_directory: Path) -> MagnetConfiguration:
    """Optimize angles and geometry, either on the unperturbed design or on perturbed copies."""
    model = StackedHalbachGeometryModel(configuration, fov_radius=0.05, min_inner_radius_m=0.1)
    perturbation = TolerancePerturbation(
        num_samples=args.num_copies,
        position_tolerance_m=args.position_tolerance_mm * 1e-3,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        remanence_tolerance_t=args.remanence_tolerance_fraction * model.remanence,
        seed=0,
    )
    # Every forward pass draws new perturbed copies; the conventional loss ignores them.
    wrapped = ToleranceAwareModel(model, perturbation, points, compute_sensitivity=tolerance_aware)
    if tolerance_aware:
        # Mean inhomogeneity of the copies plus the closed-form sensitivity to the tolerances.
        loss_fn = SensitivityRegularized(
            RobustObjective(alpha=2_000.0, cvar_weight=0.0, band_weight=1.0e7, band_t=field_band),
            args.sensitivity_weight,
        )
    else:
        loss_fn = SmoothedMinimaxObjective(alpha=2_000.0, band_weight=1.0e7, band_t=field_band)

    name = "tolerance_aware" if tolerance_aware else "conventional"
    TrainLoop(
        model=wrapped,
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-2),
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        logger=WandbLogger(project="fieldsmith", run_name=name, log_dir=output_directory / name, mode="disabled"),
        log_each_n_epochs=args.num_epochs,
        log_dir=output_directory / name,
    ).run()
    return model.get_configuration()


def main() -> None:
    """Optimize conventionally and tolerance-aware, then compare both under the same tolerances."""
    args = parse_args()
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_example_tolerance_aware"
    )

    configuration = HalbachSystemGeometry(
        n_magnets_per_ring=36,
        n_rings=10,
        ring_radius=0.135,
        system_length=0.20,
        magnet_size=0.012,
        remanence=1.3,
        device=args.device,
        dtype=torch.float64,
    ).build()
    points = MaskedPointSampler(
        CartesianGridSampler(fov=(0.1, 0.1, 0.1), resolution=(0.01, 0.01, 0.01),
                             device=args.device, dtype=torch.float64),
        lambda grid: make_spherical_mask(grid, radius=0.05),
    ).get_points()
    # Both optimizations must keep the initial mean field: no trading field strength for ppm.
    initial_field_t = evaluate_configuration(configuration, points, "dipole")["fieldstrength_mT"] * 1e-3
    field_band = (initial_field_t, 10.0)

    designs = {
        "conventional": optimize(configuration, points, field_band, args, False, output_directory),
        "tolerance-aware": optimize(configuration, points, field_band, args, True, output_directory),
    }

    print(f"\nPeak-to-peak inhomogeneity under the tolerances ({args.monte_carlo_samples} builds each):")
    print(f"{'design':<16} {'unperturbed':>12} {'built mean':>11} {'built std':>10} {'B0':>9}")
    for name, design in designs.items():
        result = run_monte_carlo(
            design.to(args.device, torch.float64), points, num_samples=args.monte_carlo_samples,
            position_tolerance_m=args.position_tolerance_mm * 1e-3,
            rotation_tolerance_deg=args.rotation_tolerance_deg,
            remanence_tolerance_t=args.remanence_tolerance_fraction * 1.3,
            field_model="dipole", seed=1, progress_callback=None,
        )
        p2p = result.summary()["peak_to_peak_ppm"]
        print(f"{name:<16} {p2p['nominal']:>8.0f} ppm {p2p['mean']:>7.0f} ppm {p2p['std']:>6.0f} ppm "
              f"{result.nominal['fieldstrength_mT']:>6.2f} mT")
    print(f"Checkpoints saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
