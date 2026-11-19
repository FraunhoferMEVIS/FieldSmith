"""Compare the effect of each manufacturing tolerance on a magnet design.

Every tolerance -- magnet position, magnetization tilt, remanence -- is applied on
its own, and all together, to the same design, and the resulting Monte Carlo
distributions of the peak-to-peak homogeneity are compared. By default the design
is the initial (ideal) system of the large stacked Halbach study; ``--system
small_halbach`` uses the initial small-Halbach system instead, and
``--configuration`` any saved ``MagnetConfiguration`` (.pt), e.g. an optimized
design. ``--tolerance-scales`` repeats the study for scaled tolerances.

For every case it reports the built peak-to-peak (mean, spread, gap to the
unperturbed design), the error floor (peak-to-peak of the error field alone), and
the RMS error from Monte Carlo next to the linearized prediction. The field model
is the dipole model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation import evaluate_configuration
from fieldsmith.evaluation.tolerance_breakdown import CASES, ToleranceCaseResult, run_tolerance_case
from fieldsmith.geometry import HalbachSystemGeometry, make_spherical_mask
from fieldsmith.models.small_halbach import build_small_halbach_configuration
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler
from fieldsmith.tolerance_study import (
    EVALUATION_REMANENCE_T,
    MONTE_CARLO_POSITION_TOLERANCE_MM,
    MONTE_CARLO_REMANENCE_TOLERANCE_FRACTION,
    MONTE_CARLO_ROTATION_TOLERANCE_DEG,
)

CASE_COLORS = {"position": "#2a78d6", "rotation": "#eb6834", "remanence": "#1baf7a", "all": "#52514e"}
CASE_LABELS = {"position": "Position", "rotation": "Rotation", "remanence": "Remanence", "all": "All together"}
SYSTEM_DEFAULTS = {
    "halbach_system": {"fov_radius": 0.075, "resolution": 0.005, "name": "Halbach system (12 planes)"},
    "small_halbach": {"fov_radius": 0.025, "resolution": 0.001, "name": "Small Halbach"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the effect of each manufacturing tolerance on a design.")
    parser.add_argument("--configuration", type=Path, default=None,
                        help="Saved MagnetConfiguration (.pt) to analyse; otherwise the initial --system.")
    parser.add_argument("--system", choices=tuple(SYSTEM_DEFAULTS), default="halbach_system",
                        help="Initial system to build without --configuration; also sets the FOV defaults.")
    parser.add_argument("--fov-radius", type=float, default=None, help="Spherical FOV radius in meters.")
    parser.add_argument("--resolution", type=float, default=None, help="FOV grid spacing in meters.")
    parser.add_argument("--position-tolerance-mm", type=float, default=MONTE_CARLO_POSITION_TOLERANCE_MM)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=MONTE_CARLO_ROTATION_TOLERANCE_DEG)
    parser.add_argument("--remanence-tolerance-fraction", type=float,
                        default=MONTE_CARLO_REMANENCE_TOLERANCE_FRACTION,
                        help="Remanence tolerance as a fraction of the design's mean remanence.")
    parser.add_argument("--tolerance-scales", type=str, default="1",
                        help="Comma-separated multipliers of all tolerances, e.g. '0.5,1,2'.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-points", type=int, default=4096)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    defaults = SYSTEM_DEFAULTS[args.system]
    args.fov_radius = defaults["fov_radius"] if args.fov_radius is None else args.fov_radius
    args.resolution = defaults["resolution"] if args.resolution is None else args.resolution
    args.scales = [float(value) for value in args.tolerance_scales.split(",") if value.strip()]
    return args


def load_design(args: argparse.Namespace, device: torch.device) -> tuple[MagnetConfiguration, str]:
    if args.configuration is not None:
        return (MagnetConfiguration.from_file(args.configuration, device=device, dtype=torch.float64),
                "/".join(args.configuration.with_suffix("").parts[-3:]))
    if args.system == "small_halbach":
        configuration = build_small_halbach_configuration(
            5, (10, 30), 0.08, 0.10, 0.04, 0.012, EVALUATION_REMANENCE_T, device,
        ).to(device=device, dtype=torch.float64)
    else:
        configuration = HalbachSystemGeometry(
            48, 12, 0.15, 0.40, 0.012, EVALUATION_REMANENCE_T, device, torch.float64,
            n_rings_per_plane=3, additional_magnets_per_ring=7,
        ).build()
    return configuration, f"{SYSTEM_DEFAULTS[args.system]['name']}, initial design"


def plot_distributions(results: list[ToleranceCaseResult], unperturbed_ppm: float, title: str) -> plt.Figure:
    """One panel per error case on a shared axis, so their distributions line up."""
    figure, axes = plt.subplots(len(results), 1, figsize=(8.5, 1.9 * len(results) + 0.8), sharex=True)
    values = [result.peak_to_peak_ppm for result in results]
    lower = min(min(float(value.min()) for value in values), unperturbed_ppm)
    upper = max(float(value.max()) for value in values)
    bins = np.linspace(lower - 0.02 * (upper - lower), upper + 0.02 * (upper - lower), 70)
    for axis, result in zip(np.atleast_1d(axes), results):
        color = CASE_COLORS[result.case]
        axis.hist(result.peak_to_peak_ppm.numpy(), bins=bins, color=color, alpha=0.75, edgecolor="white",
                  linewidth=0.4)
        axis.axvline(unperturbed_ppm, color="#0b0b0b", linestyle="--", linewidth=1.5, label="unperturbed")
        mean = float(result.peak_to_peak_ppm.mean())
        std = float(result.peak_to_peak_ppm.std(unbiased=False))
        axis.set_title(f"{CASE_LABELS[result.case]}: mean {mean:.0f} ppm, std {std:.0f} ppm",
                       loc="left", fontsize=10)
        axis.set_ylabel("Samples")
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Peak-to-peak homogeneity [ppm]")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_contributions(results: list[ToleranceCaseResult], unperturbed_ppm: float, title: str) -> plt.Figure:
    """Gap to the unperturbed design, and the RMS error from Monte Carlo and the linear model."""
    figure, (gap_axis, rms_axis) = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [CASE_LABELS[result.case] for result in results]
    colors = [CASE_COLORS[result.case] for result in results]
    positions = np.arange(len(results))
    gaps = [float(result.peak_to_peak_ppm.mean()) - unperturbed_ppm for result in results]
    spreads = [float(result.peak_to_peak_ppm.std(unbiased=False)) for result in results]
    gap_axis.barh(positions, gaps, xerr=spreads, color=colors, height=0.6, capsize=3, error_kw={"linewidth": 1})
    for position, gap, spread in zip(positions, gaps, spreads):
        gap_axis.text(gap + spread, position, f"  {gap:.0f} ± {spread:.0f}", va="center", fontsize=8,
                      color="#52514e")
    gap_axis.set_yticks(positions, labels)
    gap_axis.invert_yaxis()
    gap_axis.set_xlabel("MC mean − unperturbed peak-to-peak [ppm] (bar: std)")
    gap_axis.set_title("Gap to the unperturbed design", loc="left", fontsize=10)
    gap_axis.grid(axis="x", alpha=0.25)
    gap_axis.set_xlim(0.0, max(gap + spread for gap, spread in zip(gaps, spreads)) * 1.3)

    width = 0.38
    monte_carlo = [result.error_rms_ppm for result in results]
    linear = [result.linear_error_rms_ppm for result in results]
    rms_axis.barh(positions - width / 2, monte_carlo, height=width, color=colors, label="Monte Carlo")
    rms_axis.barh(positions + width / 2, linear, height=width, color="white", edgecolor=colors, linewidth=1.5,
                  hatch="//", label="Linear prediction")
    rms_axis.set_yticks(positions, labels)
    rms_axis.invert_yaxis()
    rms_axis.set_xlabel("RMS error across the FOV [ppm]")
    single = [result.error_rms_ppm for result in results if result.case != "all"]
    combined = [result.error_rms_ppm for result in results if result.case == "all"]
    note = ""
    if len(single) == 3 and combined:
        note = f" | quadrature sum {np.sqrt(np.sum(np.square(single))):.0f} vs all {combined[0]:.0f}"
    rms_axis.set_title(f"Error magnitude{note}", loc="left", fontsize=10)
    rms_axis.grid(axis="x", alpha=0.25)
    rms_axis.legend(
        handles=[Patch(facecolor="#8a8985", label="Monte Carlo"),
                 Patch(facecolor="white", edgecolor="#8a8985", hatch="//", label="Linear prediction")],
        frameon=False, fontsize=8, loc="upper right",
    )
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_scaling(rows: list[dict[str, float]], title: str) -> plt.Figure:
    """Gap and spread of the built peak-to-peak against the tolerance scale, per error case."""
    figure, (gap_axis, std_axis) = plt.subplots(1, 2, figsize=(12, 4.2))
    for case in CASES:
        case_rows = sorted((row for row in rows if row["case"] == case), key=lambda row: row["scale"])
        scales = [row["scale"] for row in case_rows]
        for axis, key in ((gap_axis, "gap_mean_ppm"), (std_axis, "std_ppm")):
            values = [row[key] for row in case_rows]
            axis.plot(scales, values, marker="o", markersize=6, linewidth=2, color=CASE_COLORS[case],
                      label=CASE_LABELS[case])
            axis.annotate(CASE_LABELS[case], (scales[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                          va="center", fontsize=8, color="#52514e")
    gap_axis.set_ylabel("MC mean − unperturbed [ppm]")
    std_axis.set_ylabel("Std of peak-to-peak [ppm]")
    for axis in (gap_axis, std_axis):
        axis.set_xlabel("Tolerance scale (1 = given tolerances)")
        axis.grid(alpha=0.25)
    gap_axis.legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_tolerance_breakdown"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration, design_name = load_design(args, device)
    points = MaskedPointSampler(
        CartesianGridSampler((2.0 * args.fov_radius,) * 3, (args.resolution,) * 3, device, torch.float64),
        mask_fn=lambda grid: make_spherical_mask(grid, args.fov_radius),
    ).get_points()
    nominal = evaluate_configuration(configuration, points, "dipole", chunk_points=args.chunk_points)
    mean_remanence = float(torch.linalg.norm(configuration.remanence_vectors(), dim=1).mean())
    print(f"{design_name}: {configuration.num_magnets} magnets, {points.shape[0]} FOV points, "
          f"{nominal['fieldstrength_mT']:.2f} mT, unperturbed {nominal['peak_to_peak_ppm']:.0f} ppm")

    rows: list[dict[str, float]] = []
    samples: dict[str, dict[str, torch.Tensor]] = {}
    for scale in args.scales:
        results = []
        for case in CASES:
            result = run_tolerance_case(
                configuration, points, case,
                position_tolerance_m=args.position_tolerance_mm * 1e-3,
                rotation_tolerance_deg=args.rotation_tolerance_deg,
                remanence_tolerance_t=args.remanence_tolerance_fraction * mean_remanence,
                num_samples=args.num_samples, seed=args.seed, scale=scale, chunk_points=args.chunk_points,
            )
            results.append(result)
            rows.append(result.summary(nominal["peak_to_peak_ppm"]))
            samples[f"scale_{scale}/{case}"] = {"peak_to_peak_ppm": result.peak_to_peak_ppm,
                                                "floor_ppm": result.floor_ppm,
                                                "fieldstrength_mT": result.fieldstrength_mT}
            row = rows[-1]
            print(f"scale {scale:g} | {case:<9}: mean {row['mean_ppm']:7.0f}  std {row['std_ppm']:5.0f}  "
                  f"gap {row['gap_mean_ppm']:6.0f}  floor {row['floor_mean_ppm']:6.0f}  "
                  f"RMS error MC {row['error_rms_ppm']:5.0f} / linear {row['linear_error_rms_ppm']:5.0f} ppm")
        tolerances = (f"±{args.position_tolerance_mm * scale:g} mm, ±{args.rotation_tolerance_deg * scale:g}°, "
                      f"±{args.remanence_tolerance_fraction * scale * 100:g}% Br")
        suffix = "" if len(args.scales) == 1 else f"_scale_{scale:g}"
        title = f"{design_name} | {tolerances} | {args.num_samples} samples per case"
        for name, plot in (("tolerance_distributions", plot_distributions),
                           ("tolerance_contributions", plot_contributions)):
            figure = plot(results, nominal["peak_to_peak_ppm"], title)
            figure.savefig(output_directory / f"{name}{suffix}.png", dpi=180, bbox_inches="tight")
            plt.close(figure)
    if len(args.scales) > 1:
        figure = plot_scaling(rows, f"{design_name} | {args.num_samples} samples per case")
        figure.savefig(output_directory / "tolerance_scaling.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    with (output_directory / "tolerance_breakdown.csv").open("w", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump({
            "design": design_name,
            "configuration": None if args.configuration is None else str(args.configuration),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "evaluation_points": int(points.shape[0]),
            "unperturbed": nominal,
            "cases": rows,
        }, summary_file, indent=2)
    torch.save(samples, output_directory / "tolerance_samples.pt")
    print(f"Results saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
