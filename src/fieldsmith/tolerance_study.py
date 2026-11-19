# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the shared driver of tolerance-aware optimization studies.
#
#  tolerance_study.py
#  Kostiantyn Lavronenko
#  18.10.2026
# -----------------------------------------------------------------------------

"""
Shared driver of tolerance-aware optimization studies.

A study script supplies the system-specific parts -- its geometry arguments, the
initial ``MagnetConfiguration``, and a base model exposing ``magnet_state()`` --
and this module runs the rest: nominal and robust optimization with identical
seeds, held-out tracking of the best design, full-grid Monte Carlo evaluation,
trust metrics, comparison output, and argument sweeps.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation import MonteCarloResult, field_at_points, run_monte_carlo
from fieldsmith.geometry import make_spherical_mask
from fieldsmith.geometry.collision import count_overlaps
from fieldsmith.geometry.cube_axes import cube_axes_from_directions
from fieldsmith.logger import CsvHistoryLogger
from fieldsmith.metrics import field_metrics
from fieldsmith.models import ToleranceAwareModel
from fieldsmith.objectives import (
    DiracWassersteinObjective,
    RobustObjective,
    SensitivityRegularized,
    SmoothedMinimaxObjective,
    batched_peak_to_peak_ppm,
)
from fieldsmith.perturbation import TolerancePerturbation
from fieldsmith.sensitivity import linearized_sensitivity_ppm
from fieldsmith.point_sampler import CartesianGridSampler, FibonacciSphereSampler, MaskedPointSampler
from fieldsmith.train_loop import TrainLoop
from fieldsmith.visualizations import (
    plot_geometry_comparison,
    plot_magnets_and_field_3d,
    plot_magnets_and_field_3d_matplotlib,
    plot_system_layout,
)

EVALUATION_REMANENCE_T = 1.31
MONTE_CARLO_SAMPLES = 100
MONTE_CARLO_POSITION_TOLERANCE_MM = 0.5  # Per-magnet position error of the printed holder, +/- mm
MONTE_CARLO_ROTATION_TOLERANCE_DEG = 1.2  # Per-magnet manufacturing error, +/- degrees
MONTE_CARLO_REMANENCE_TOLERANCE_FRACTION = 0.02  # Per-magnet remanence error, +/- fraction of Br

SERIES_COLORS = ("#2a78d6", "#eb6834")
DESIGN_COLORS = {"initial": "#8a8985", "nominal": SERIES_COLORS[0], "robust": SERIES_COLORS[1]}


def add_study_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the system-independent arguments: objective, tolerances, sampling, evaluation, output."""
    parser.add_argument(
        "--objective", choices=("robust", "nominal", "both"), default="robust",
        help="'both' runs the nominal baseline and then the robust optimization into "
             "<output>/nominal and <output>/robust, and writes a comparison next to them.",
    )
    parser.add_argument(
        "--optimize", type=str, default="angles,radii,distances",
        help="Comma-separated free parameters out of angles, radii, distances. "
             "'angles' alone reproduces the angle-only optimization.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5.0e-3, help="Learning rate of the angles.")
    parser.add_argument("--geometry-lr", type=float, default=1.0e-2,
                        help="Learning rate of the radius and plane-spacing slack parameters.")
    parser.add_argument("--min-inner-radius", type=float, default=0.06,
                        help="Bore: the inner ring radius never falls below this, in meters.")
    parser.add_argument("--clearance-mm", type=float, default=0.5,
                        help="Extra clearance added to every non-overlap bound of the geometry.")
    parser.add_argument("--lr-final-fraction", type=float, default=0.05,
                        help="Cosine-annealed final learning rate as a fraction of --lr.")
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--cvar-weight", type=float, default=1.0)
    parser.add_argument(
        "--robust-loss", choices=("mean_cvar", "wasserstein"), default="mean_cvar",
        help="mean_cvar: mean + CVaR + std of the child peak-to-peak. wasserstein: Sinkhorn "
             "(geomloss) distance of the child peak-to-peak distribution to a Dirac at "
             "--wasserstein-target-ppm, which for p = 2 penalizes mean and breadth together.",
    )
    parser.add_argument("--wasserstein-p", type=int, choices=(1, 2), default=2)
    parser.add_argument("--wasserstein-blur-ppm", type=float, default=50.0)
    parser.add_argument("--wasserstein-target-ppm", type=float, default=0.0,
                        help="Position of the Dirac with --wasserstein-target fixed.")
    parser.add_argument(
        "--wasserstein-target", choices=("fixed", "parent"), default="fixed",
        help="fixed: Dirac at --wasserstein-target-ppm. parent: Dirac at the (detached) homogeneity "
             "of the unperturbed design, so the distance is sqrt(gap^2 + std^2).",
    )
    parser.add_argument("--wasserstein-weight", type=float, default=1.0,
                        help="Weight of the Wasserstein distance.")
    parser.add_argument("--wasserstein-mean-weight", type=float, default=None,
                        help="Weight of the mean child peak-to-peak next to the distance; "
                             "defaults to 0 for a fixed target and 1 for the parent target.")
    parser.add_argument(
        "--sensitivity-weight", type=float, default=0.0,
        help="Weight of the linearized, sampling-free error sensitivity (RMS inhomogeneity error in "
             "ppm) added to the robust loss; 0 disables it.",
    )
    parser.add_argument("--std-weight", type=float, default=1.0,
                        help="Weight of the child peak-to-peak standard deviation; 0 restores mean + CVaR.")
    parser.add_argument("--cvar-level", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=2.0e3)
    parser.add_argument("--alpha-max", type=float, default=2.0e4)
    parser.add_argument("--alpha-growth", type=float, default=1.01)
    parser.add_argument("--min-field-mt", type=float, default=None,
                        help="Optional floor on the nominal mean field, enforced by a quadratic band penalty.")
    parser.add_argument("--band-weight", type=float, default=1.0e7)

    parser.add_argument("--num-samples", type=int, default=MONTE_CARLO_SAMPLES)
    parser.add_argument("--nominal-log-samples", type=int, default=None,
                        help="Children drawn per step by the nominal objective, which only logs them; "
                             "defaults to --num-samples.")
    parser.add_argument("--position-tolerance-mm", type=float, default=MONTE_CARLO_POSITION_TOLERANCE_MM)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=MONTE_CARLO_ROTATION_TOLERANCE_DEG)
    parser.add_argument("--remanence-tolerance-fraction", type=float,
                        default=MONTE_CARLO_REMANENCE_TOLERANCE_FRACTION)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--fov-radius", type=float, default=0.025)
    parser.add_argument("--train-surface-points", type=int, default=1024,
                        help="Fibonacci points on the FOV sphere used for training.")
    parser.add_argument("--train-resolution", type=float, default=0.005,
                        help="Coarse interior grid added to the training points; 0 disables it.")
    parser.add_argument("--resolution", type=float, default=0.001, help="Evaluation grid spacing in meters.")
    parser.add_argument("--heldout-samples", type=int, default=200)
    parser.add_argument("--heldout-every", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--comparison-samples", type=int, default=5000,
                        help="Monte Carlo samples of the final nominal-vs-robust comparison only.")
    parser.add_argument(
        "--comparison-designs", type=str, default="best",
        help="Checkpoints to compare at the end, 'best', 'final', or 'best,final'. With both, the "
             "comparison files carry the suffixes _best and _final.",
    )
    parser.add_argument("--eval-field-model", choices=("dipole", "cuboid"), default="dipole")
    parser.add_argument("--chunk-samples", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-each-n-epochs", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="smh-magnetopts")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument(
        "--sweep", type=str, default=None,
        help="Repeat the experiment for several values of one argument, e.g. "
             "'min_field_mt=33,35,37' or 'std_weight=0,1,3'.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def training_points(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """FOV shell plus an optional coarse interior grid."""
    parts = [FibonacciSphereSampler(args.fov_radius, args.train_surface_points, device, torch.float64).get_points()]
    if args.train_resolution > 0.0:
        interior = MaskedPointSampler(
            CartesianGridSampler((2.0 * args.fov_radius,) * 3, (args.train_resolution,) * 3, device, torch.float64),
            mask_fn=lambda points: make_spherical_mask(points, args.fov_radius),
        )
        parts.append(interior.get_points())
    return torch.cat(parts)


def evaluation_points(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return MaskedPointSampler(
        CartesianGridSampler((2.0 * args.fov_radius,) * 3, (args.resolution,) * 3, device, torch.float64),
        mask_fn=lambda points: make_spherical_mask(points, args.fov_radius),
    ).get_points()


class HeldOutEvaluation:
    """
    Score the design on a fixed set of children, on a cadence, and keep the best.

    The training loss is measured on fresh children every step and is therefore
    noisy; comparing epochs needs the same children every time (common random
    numbers). The selection score matches the objective being optimized, so the
    nominal baseline is not quietly selected for robustness.
    """

    def __init__(self, model: ToleranceAwareModel, logger: CsvHistoryLogger, num_samples: int, seed: int,
                 every: int, score_fn: Callable[[dict[str, float], torch.Tensor], float]) -> None:
        self.model = model
        self.logger = logger
        self.num_samples = num_samples
        self.seed = seed
        self.every = every
        self.score_fn = score_fn
        self.best_score: float = float("inf")
        self.best_epoch: int | None = None
        self.best_configuration: MagnetConfiguration | None = None
        self.best_statistics: dict[str, float] | None = None

    def evaluate(self, epoch: int) -> dict[str, float]:
        generator = torch.Generator(device=self.model.points.device).manual_seed(self.seed)
        with torch.no_grad():
            output = self.model(num_samples=self.num_samples, generator=generator)
            statistics = {name: float(value) for name, value in
                          self.model.child_statistics(output["child_readings"]).items()}
            homogeneity = batched_peak_to_peak_ppm(output["child_readings"])
            nominal = field_metrics(output["B"])
        statistics["nominal_ppm"] = float(nominal["homogeneity_ppm"])
        statistics["nominal_fieldstrength_mT"] = float(nominal["fieldstrength_mT"])
        statistics["children_gap_ppm"] = statistics["children_mean_ppm"] - statistics["nominal_ppm"]
        statistics["sensitivity_ppm"] = self.sensitivity(output["B"])
        geometry_summary = getattr(self.model.base_model, "geometry_summary", None)
        if geometry_summary is not None:
            statistics.update({f"geometry_{name}": value for name, value in geometry_summary().items()})
        score = float(self.score_fn(statistics, homogeneity))
        statistics["score"] = score
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_configuration = self.model.get_configuration().detached_cpu()
            self.best_statistics = dict(statistics)
        self.logger.log_scalars({f"heldout_{name}": value for name, value in statistics.items()}, max(epoch, 0))
        print(
            f"  held-out @ {epoch}: nominal {statistics['nominal_ppm']:.0f} ppm, "
            f"children mean {statistics['children_mean_ppm']:.0f} / "
            f"p95 {statistics['children_p95_ppm']:.0f} / CVaR {statistics['children_cvar_ppm']:.0f} ppm, "
            f"sensitivity {statistics['sensitivity_ppm']:.0f} ppm, B0 {statistics['nominal_fieldstrength_mT']:.2f} mT"
        )
        return statistics

    def sensitivity(self, nominal_field: torch.Tensor) -> float:
        """Linearized RMS inhomogeneity error in ppm, logged for every objective."""
        perturbation = self.model.perturbation
        with torch.no_grad():
            state = self.model.base_model.magnet_state()
            return float(linearized_sensitivity_ppm(
                state["positions"], state["orientations"], state["remanence"], state["volumes"],
                self.model.points, nominal_field, perturbation.position_tolerance_m,
                perturbation.rotation_tolerance_rad, perturbation.remanence_tolerance_t,
            ))

    def __call__(self, epoch: int, **_) -> None:
        if (epoch + 1) % self.every == 0:
            self.evaluate(epoch)


def tolerance_evaluation(configuration: MagnetConfiguration, points: torch.Tensor,
                         args: argparse.Namespace, num_samples: int | None = None) -> MonteCarloResult:
    """Full-grid Monte Carlo of the manufacturing errors with an independent seed."""
    return run_monte_carlo(
        configuration=configuration,
        points=points,
        num_samples=args.eval_samples if num_samples is None else num_samples,
        position_tolerance_m=args.position_tolerance_mm * 1e-3,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        remanence_tolerance_t=args.remanence_tolerance_fraction * args.remanence,
        field_model=args.eval_field_model,
        seed=args.random_seed + 1000,
        progress_callback=None,
    )


def plot_comparison(results: dict[str, dict[str, MonteCarloResult]], labels: tuple[str, str]) -> plt.Figure:
    """Overlay the peak-to-peak distributions of two designs under the manufacturing errors."""
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    values = [results[label]["manufacturing"].peak_to_peak_ppm for label in labels]
    nominals = [results[label]["manufacturing"].nominal["peak_to_peak_ppm"] for label in labels]
    lower = min(min(float(value.min()) for value in values), *nominals)
    upper = max(max(float(value.max()) for value in values), *nominals)
    margin = 0.03 * (upper - lower)
    bins = torch.linspace(lower - margin, upper + margin, 60).numpy()
    for label, value, nominal, color in zip(labels, values, nominals, SERIES_COLORS):
        axis.hist(value.numpy(), bins=bins, color=color, alpha=0.55, edgecolor="white", linewidth=0.5,
                  label=f"{label}: MC mean {float(value.mean()):.0f} ppm")
        axis.axvline(nominal, color=color, linewidth=2, linestyle="--", label=f"{label}: unperturbed")
    axis.set_title("Manufacturing errors")
    axis.set_xlabel("Peak-to-peak homogeneity [ppm]")
    axis.set_ylabel("Monte Carlo samples")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


OPTIMIZABLE = ("angles", "radii", "distances")

InitialBuilder = Callable[[argparse.Namespace, torch.device], MagnetConfiguration]
ModelBuilder = Callable[[argparse.Namespace, MagnetConfiguration, set[str]], tuple[nn.Module, list[dict]]]


def optimized_parameters(args: argparse.Namespace) -> set[str]:
    selected = {name.strip() for name in args.optimize.split(",") if name.strip()}
    unknown = selected.difference(OPTIMIZABLE)
    if not selected or unknown:
        raise ValueError(f"--optimize must name some of {', '.join(OPTIMIZABLE)}; got {args.optimize!r}.")
    return selected


def trust_metrics(result: MonteCarloResult) -> dict[str, float]:
    """How far, and how widely, the built designs fall from the unperturbed one."""
    values = result.peak_to_peak_ppm.to(torch.float64)
    unperturbed = result.nominal["peak_to_peak_ppm"]
    mean = float(values.mean())
    return {
        "unperturbed_ppm": unperturbed,
        "mean_ppm": mean,
        "std_ppm": float(values.std(unbiased=False)),
        "iqr_ppm": float(torch.quantile(values, 0.75) - torch.quantile(values, 0.25)),
        "gap_mean_ppm": mean - unperturbed,
        "gap_p97.5_ppm": float(torch.quantile(values, 0.975)) - unperturbed,
        "relative_gap": (mean - unperturbed) / mean,
        "fieldstrength_mT": result.nominal["fieldstrength_mT"],
    }


def overlap_count(configuration: MagnetConfiguration) -> int:
    """Magnet pairs intersecting by more than the 0.4 mm acceptance tolerance."""
    with torch.no_grad():
        return count_overlaps(
            configuration.positions,
            cube_axes_from_directions(configuration.orientations),
            float(configuration.metadata["magnet_size_m"]),
        )


def print_table(rows: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    print(f"{'design':<16} {'mode':<14} {'nominal':>9} {'mean':>9} {'p97.5':>9} {'B0 [mT]':>9}")
    for name, modes in rows.items():
        for mode, metrics in modes.items():
            p2p = metrics["peak_to_peak_ppm"]
            print(f"{name:<16} {mode:<14} {p2p['nominal']:>9.0f} {p2p['mean']:>9.0f} {p2p['p97.5']:>9.0f} "
                  f"{metrics['fieldstrength_mT']['nominal']:>9.2f}")


def selection_score(args: argparse.Namespace,
                    objective_name: str) -> Callable[[dict[str, float], torch.Tensor], float]:
    """Exact held-out counterpart of the training loss, used to pick the best checkpoint."""
    if objective_name == "nominal":
        return lambda statistics, homogeneity: statistics["nominal_ppm"]
    sensitivity_weight = args.sensitivity_weight
    if args.robust_loss == "wasserstein":
        p, weight, mean_weight = args.wasserstein_p, args.wasserstein_weight, wasserstein_mean_weight(args)

        def score(statistics: dict[str, float], homogeneity: torch.Tensor) -> float:
            target = statistics["nominal_ppm"] if args.wasserstein_target == "parent" else args.wasserstein_target_ppm
            distance = float(((homogeneity - target).abs() ** p).mean() ** (1.0 / p))
            return (weight * distance + mean_weight * statistics["children_mean_ppm"]
                    + sensitivity_weight * statistics["sensitivity_ppm"])
        return score
    return lambda statistics, homogeneity: (
        args.mean_weight * statistics["children_mean_ppm"]
        + args.cvar_weight * statistics["children_cvar_ppm"]
        + args.std_weight * statistics["children_std_ppm"]
        + sensitivity_weight * statistics["sensitivity_ppm"]
    )


def wasserstein_mean_weight(args: argparse.Namespace) -> float:
    if args.wasserstein_mean_weight is not None:
        return args.wasserstein_mean_weight
    return 1.0 if args.wasserstein_target == "parent" else 0.0


def save_system_views(configuration: MagnetConfiguration, args: argparse.Namespace, device: torch.device,
                      directory: Path, name: str, title: str) -> None:
    """
    Save what a design looks like: a static 3D view, an interactive 3D view, and its layout.

    Writes ``system_<name>_3d.png`` (oriented cubes and |B| on the training
    points), ``system_<name>_3d.html`` (the same in Plotly, to rotate and zoom),
    and ``system_<name>_layout.png`` (top view of one plane and r-z section).
    """
    configuration = configuration.to(device=device, dtype=torch.float64)
    points = training_points(args, device)
    magnet_size = float(configuration.metadata["magnet_size_m"])
    with torch.no_grad():
        field = field_at_points(configuration, points, "dipole", chunk_points=None)
    figure = plot_magnets_and_field_3d_matplotlib(
        points, field, configuration.positions, configuration.orientations, magnet_size, title=title,
        figure_size=(9.0, 7.0), show_legend=False,
    )
    figure.suptitle(title)
    figure.savefig(directory / f"system_{name}_3d.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    interactive = plot_magnets_and_field_3d(
        points, field, configuration.positions, configuration.orientations, magnet_size, title=title,
    )
    if hasattr(interactive, "write_html"):
        interactive.write_html(directory / f"system_{name}_3d.html", include_plotlyjs="cdn")
    else:
        plt.close(interactive)
    figure = plot_system_layout(configuration.positions, configuration.orientations, magnet_size,
                                fov_radius=args.fov_radius, title=title)
    figure.savefig(directory / f"system_{name}_layout.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_optimization(
    args: argparse.Namespace,
    objective_name: str,
    initial: MagnetConfiguration,
    output_directory: Path,
    device: torch.device,
    build_model: ModelBuilder,
    initial_results: dict[str, MonteCarloResult] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, MonteCarloResult]]]:
    """
    Run one optimization and its full-grid evaluation into output_directory.

    Every run reseeds identically, so two runs with different objectives see the
    same training children, the same held-out children, and the same evaluation
    draws. ``initial_results`` reuses an evaluation of the shared initial design.
    ``build_model`` returns the base model, which must expose ``magnet_state()``
    and ``get_configuration()``, and its optimizer parameter groups.
    """
    torch.manual_seed(args.random_seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["objective"] = objective_name
    config["output_directory"] = str(output_directory)
    print(f"Device: {device}; objective: {objective_name}; output: {output_directory.resolve()}")

    free = optimized_parameters(args)
    base_model, parameter_groups = build_model(args, initial, free)
    if hasattr(base_model, "geometry_summary"):
        print(f"Free parameters: {', '.join(sorted(free))}; initial geometry: {base_model.geometry_summary()}")
    initial_geometry = base_model.geometry_summary() if hasattr(base_model, "geometry_summary") else {}
    points = training_points(args, device)
    # The nominal objective ignores the children; it only logs them, so it may draw fewer.
    num_samples = args.num_samples
    if objective_name == "nominal" and args.nominal_log_samples is not None:
        num_samples = args.nominal_log_samples
    perturbation = TolerancePerturbation(
        num_samples=num_samples,
        position_tolerance_m=args.position_tolerance_mm * 1e-3,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        remanence_tolerance_t=args.remanence_tolerance_fraction * args.remanence,
        antithetic=args.antithetic,
        seed=args.random_seed,
    )
    model = ToleranceAwareModel(
        base_model, perturbation, points,
        chunk_samples=args.chunk_samples,
        cvar_level=args.cvar_level,
        compute_sensitivity=objective_name == "robust" and args.sensitivity_weight > 0.0,
    )
    print(f"Magnets: {initial.num_magnets}; training points: {points.shape[0]}; children per step: {num_samples}")

    band = None if args.min_field_mt is None else (args.min_field_mt * 1e-3, 10.0)
    band_weight = 0.0 if band is None else args.band_weight
    if objective_name == "robust" and args.robust_loss == "wasserstein":
        objective = DiracWassersteinObjective(
            alpha=args.alpha, alpha_max=args.alpha_max, alpha_growth=args.alpha_growth,
            p=args.wasserstein_p, blur_ppm=args.wasserstein_blur_ppm, target_ppm=args.wasserstein_target_ppm,
            band_weight=band_weight, band_t=band, target=args.wasserstein_target,
            mean_weight=wasserstein_mean_weight(args), distance_weight=args.wasserstein_weight,
        )
    elif objective_name == "robust":
        objective = RobustObjective(
            alpha=args.alpha, alpha_max=args.alpha_max, alpha_growth=args.alpha_growth,
            cvar_level=args.cvar_level, mean_weight=args.mean_weight, cvar_weight=args.cvar_weight,
            std_weight=args.std_weight, band_weight=band_weight, band_t=band,
        )
    else:
        objective = SmoothedMinimaxObjective(
            alpha=args.alpha, alpha_max=args.alpha_max, alpha_growth=args.alpha_growth,
            band_weight=band_weight, band_t=band,
        )
    if objective_name == "robust" and args.sensitivity_weight > 0.0:
        objective = SensitivityRegularized(objective, args.sensitivity_weight)
    parameter_groups = [
        group for group in parameter_groups if any(parameter.requires_grad for parameter in group["params"])
    ]
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_final_fraction,
    )

    logger = CsvHistoryLogger(
        project=args.wandb_project, run_name=f"{output_directory.parent.name}/{output_directory.name}",
        log_dir=output_directory, config=config, mode=args.wandb_mode,
    )
    logger.save_configuration(model.get_configuration(), 0)
    heldout = HeldOutEvaluation(
        model, logger, args.heldout_samples, args.random_seed + 500, args.heldout_every,
        selection_score(args, objective_name),
    )
    initial_heldout = heldout.evaluate(-1)

    def schedule_callback(epoch: int, **_) -> None:
        objective.step()
        scheduler.step()
        logger.log_scalars(
            {"alpha": objective.alpha, **{f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups}},
            epoch,
        )
        if epoch == args.epochs - 1:
            logger.save_configuration(model.get_configuration(), epoch)

    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=objective,
        num_epochs=args.epochs,
        logger=logger,
        log_each_n_epochs=args.log_each_n_epochs,
        log_dir=output_directory,
        callbacks=[schedule_callback, heldout],
    )
    train_start = time.perf_counter()
    loop.run()
    train_seconds = time.perf_counter() - train_start
    print(f"Training took {train_seconds:.1f}s")

    final = model.get_configuration().detached_cpu()
    best = heldout.best_configuration or final
    initial_cpu = initial.detached_cpu()
    for name, configuration in (("initial", initial_cpu), ("final", final), ("best", best)):
        torch.save(configuration.to_dict(), output_directory / f"{name}.pt")

    print(f"Full-grid Monte Carlo ({args.eval_samples} samples, {args.eval_field_model} model)...")
    full_points = evaluation_points(args, device)
    results: dict[str, dict[str, MonteCarloResult]] = {}
    for name, configuration in (("initial", initial_cpu), ("final", final), ("best", best)):
        if name == "initial" and initial_results is not None:
            results[name] = initial_results
            continue
        configuration = configuration.to(device=device, dtype=torch.float64)
        results[name] = {
            "manufacturing": tolerance_evaluation(configuration, full_points, args),
        }
    torch.save(results, output_directory / "monte_carlo_results.pt")
    save_system_views(best, args, device, output_directory, "best",
                      f"{args.system_name}, {objective_name} optimization, best design")

    figure = plot_comparison(results, ("initial", "best"))
    figure.suptitle(f"{args.system_name}, {objective_name} optimization ({args.optimize}) | "
                    f"{args.eval_samples} tolerance samples")
    figure.tight_layout()
    figure.savefig(output_directory / "monte_carlo_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "objective": objective_name,
        "train_seconds": train_seconds,
        "training_points": int(points.shape[0]),
        "evaluation_points": int(full_points.shape[0]),
        "best_epoch": heldout.best_epoch,
        "heldout_initial": initial_heldout,
        "heldout_best": heldout.best_statistics,
        "optimize": sorted(free),
        "robust_loss": args.robust_loss if objective_name == "robust" else None,
        "geometry": {
            "initial": initial_geometry,
            **{name: {key: value for key, value in configuration.metadata.items() if key.endswith("_mm")}
               for name, configuration in (("final", final), ("best", best))},
        },
        "overlaps": {
            name: overlap_count(configuration)
            for name, configuration in (("initial", initial_cpu), ("final", final), ("best", best))
        },
        "monte_carlo": {
            name: {mode: result.summary() for mode, result in modes.items()}
            for name, modes in results.items()
        },
        "trust": {
            name: {mode: trust_metrics(result) for mode, result in modes.items()}
            for name, modes in results.items()
        },
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print_table(summary["monte_carlo"])
    print(f"Overlaps (initial/final/best): {summary['overlaps']}; best geometry: {summary['geometry']['best']}")
    print(f"Results saved to {output_directory.resolve()}")
    return summary, results


def _design_key(configuration: MagnetConfiguration) -> str:
    """Fingerprint of the magnets of a design, to evaluate identical designs only once."""
    digest = hashlib.sha256()
    for tensor in (configuration.positions, configuration.orientations, configuration.remanence_vectors()):
        digest.update(tensor.detach().to("cpu", torch.float64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_comparison(
    args: argparse.Namespace,
    output_directory: Path,
    summaries: dict[str, dict[str, object]],
    device: torch.device,
    design: str = "best",
    suffix: str = "",
    cache: dict[str, MonteCarloResult] | None = None,
) -> None:
    """
    Compare the nominal and robust runs in ``output_directory`` and write the result next to them.

    Works from the files on disk -- ``nominal/`` and ``robust/`` with their
    ``initial.pt`` and ``best.pt`` -- so it can also rebuild the comparison of an
    earlier run. The initial and both best designs are re-evaluated with
    ``--comparison-samples`` Monte Carlo samples (more than the per-run
    evaluation), and the comparison plot, summary, and raw samples use those.
    ``design`` picks the ``best`` or ``final`` checkpoint of both runs, and
    ``suffix`` is appended to every written file name. ``cache`` reuses the
    evaluation of an identical design (same magnets) across several comparisons.
    """
    designs = {
        "initial": MagnetConfiguration.from_file(output_directory / "nominal" / "initial.pt", device, torch.float64),
        "nominal": MagnetConfiguration.from_file(output_directory / "nominal" / f"{design}.pt", device, torch.float64),
        "robust": MagnetConfiguration.from_file(output_directory / "robust" / f"{design}.pt", device, torch.float64),
    }
    for configuration in designs.values():
        configuration.metadata.setdefault("magnet_size_m", args.magnet_size)
    print(f"Comparison Monte Carlo ({args.comparison_samples} samples per design and error model)...")
    points = evaluation_points(args, device)
    cache = {} if cache is None else cache
    results = {}
    for name, configuration in designs.items():
        key = _design_key(configuration)
        if key not in cache:
            cache[key] = tolerance_evaluation(configuration, points, args, args.comparison_samples)
        results[name] = {"manufacturing": cache[key]}
    torch.save(results, output_directory / f"comparison_monte_carlo_results{suffix}.pt")

    comparison = {
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "runs": {name: str(output_directory / name) for name in ("nominal", "robust")},
        "comparison_samples": args.comparison_samples,
        "design": design,
        "evaluation_points": int(points.shape[0]),
        "train_seconds": {name: summaries[name]["train_seconds"] for name in ("nominal", "robust")},
        "best_epoch": {name: summaries[name]["best_epoch"] for name in ("nominal", "robust")},
        "heldout_best": {name: summaries[name]["heldout_best"] for name in ("nominal", "robust")},
        "monte_carlo_best": {
            name: {mode: result.summary() for mode, result in modes.items()} for name, modes in results.items()
        },
        "trust_best": {
            name: {mode: trust_metrics(result) for mode, result in modes.items()} for name, modes in results.items()
        },
        "geometry_best": {name: summaries[name].get("geometry", {}).get("best", {}) for name in ("nominal", "robust")},
        "overlaps_best": {name: summaries[name].get("overlaps", {}).get("best") for name in ("nominal", "robust")},
    }
    with (output_directory / f"comparison_summary{suffix}.json").open("w", encoding="utf-8") as summary_file:
        json.dump(comparison, summary_file, indent=2)

    figure = plot_comparison(results, ("nominal", "robust"))
    figure.suptitle(f"{args.system_name}, nominal vs. tolerance-aware optimization ({args.optimize}), "
                    f"{design} designs | {args.comparison_samples} tolerance samples")
    figure.tight_layout()
    figure.savefig(output_directory / f"comparison_monte_carlo{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure = plot_geometry_comparison(
        {name: configuration.positions for name, configuration in designs.items()},
        float(designs["initial"].metadata["magnet_size_m"]), DESIGN_COLORS, fov_radius=args.fov_radius,
        title=f"{args.system_name}: ring radii and plane heights (z >= 0, mirrored)",
    )
    figure.savefig(output_directory / f"comparison_geometry{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"\nComparison of the {design} designs:")
    print_table(comparison["monte_carlo_best"])
    print(f"{'design':<16} {'mode':<14} {'gap mean':>9} {'gap p97.5':>10} {'std':>7} {'IQR':>7}")
    for name, modes in comparison["trust_best"].items():
        for mode, trust in modes.items():
            print(f"{name:<16} {mode:<14} {trust['gap_mean_ppm']:>9.0f} {trust['gap_p97.5_ppm']:>10.0f} "
                  f"{trust['std_ppm']:>7.0f} {trust['iqr_ppm']:>7.0f}")
    print(f"Comparison saved to {output_directory.resolve()}")


def run_experiment(args: argparse.Namespace, output_directory: Path, device: torch.device,
                   build_initial: InitialBuilder, build_model: ModelBuilder) -> dict[str, dict[str, object]]:
    """Run one objective, or both sequentially with a comparison, and return the summaries."""
    initial = build_initial(args, device)
    if args.objective != "both":
        summary, _ = run_optimization(args, args.objective, initial, output_directory, device, build_model)
        return {args.objective: summary}

    # Sequential, not parallel: both runs share one GPU, each takes minutes, and
    # sequential runs stay deterministic with readable console output.
    summaries: dict[str, dict[str, object]] = {}
    results: dict[str, dict[str, dict[str, MonteCarloResult]]] = {}
    initial_results = None
    for objective_name in ("nominal", "robust"):
        summaries[objective_name], results[objective_name] = run_optimization(
            args, objective_name, initial, output_directory / objective_name, device, build_model,
            initial_results,
        )
        initial_results = results[objective_name]["initial"]
    designs = [name.strip() for name in args.comparison_designs.split(",") if name.strip()]
    if not designs or set(designs) - {"best", "final"}:
        raise ValueError(f"--comparison-designs must name best and/or final; got {args.comparison_designs!r}.")
    cache: dict[str, MonteCarloResult] = {}
    for design in designs:
        write_comparison(args, output_directory, summaries, device, design=design,
                         suffix="" if len(designs) == 1 else f"_{design}", cache=cache)
    return summaries


def parse_sweep(args: argparse.Namespace) -> tuple[str, list[object]]:
    """Split 'name=v1,v2' and convert the values to the type of that argument."""
    name, _, values = args.sweep.partition("=")
    name = name.strip().lstrip("-").replace("-", "_")
    if not values or not hasattr(args, name) or name in {"sweep", "objective", "output_directory"}:
        raise ValueError(f"--sweep must look like 'argument=v1,v2,...' for an existing argument; got {args.sweep!r}.")
    current = getattr(args, name)
    convert = type(current) if isinstance(current, (int, float, str)) and not isinstance(current, bool) else float
    return name, [convert(value.strip()) for value in values.split(",") if value.strip()]


def run_sweep(args: argparse.Namespace, output_directory: Path, device: torch.device,
              build_initial: InitialBuilder, build_model: ModelBuilder) -> None:
    """Repeat the experiment per value and collect every run's trust metrics in one table."""
    name, values = parse_sweep(args)
    rows = []
    for value in values:
        sweep_args = copy.copy(args)
        setattr(sweep_args, name, value)
        sweep_args.sweep = None
        print(f"\n=== {name} = {value} ===")
        summaries = run_experiment(sweep_args, output_directory / f"{name}_{value}", device,
                                   build_initial, build_model)
        for objective_name, summary in summaries.items():
            for design in ("initial", "best"):
                for mode, trust in summary["trust"][design].items():
                    rows.append({
                        name: value, "objective": objective_name, "design": design, "mode": mode,
                        "best_epoch": summary["best_epoch"], "overlaps": summary["overlaps"][design],
                        **trust, **summary["geometry"][design],
                    })
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_directory / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSweep summary saved to {(output_directory / 'sweep_summary.csv').resolve()}")


def run_study(args: argparse.Namespace, build_initial: InitialBuilder, build_model: ModelBuilder) -> None:
    """
    Entry point of a study script, after argument parsing.

    ``args`` must carry ``system_name`` (for titles) and ``system_tag`` (for the
    default output directory) next to the arguments of ``add_study_arguments``.
    """
    device = torch.device(args.device)
    tag = "sweep" if args.sweep else args.objective
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        f"%Y_%m_%d/%Hh%Mm%Ss_tolerance_aware_{args.system_tag}_{tag}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        run_sweep(args, output_directory, device, build_initial, build_model)
    else:
        run_experiment(args, output_directory, device, build_initial, build_model)
