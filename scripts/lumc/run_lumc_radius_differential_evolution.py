"""Optimize one physical LUMC Halbach system's ring radii with differential evolution."""

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
from scipy.optimize import OptimizeResult, differential_evolution
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fieldsmith.evaluation.lumc_halbach import INNER_RING_RADII, create_configuration
from fieldsmith.metrics import component_field_strength, component_peak_to_peak_ppm

from run_lumc_radius_optimisation import LUMCRadiusModel, parse_initial_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize one LUMC Halbach system's continuous ring radii with differential evolution."
    )
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit per run; when set, overrides --max-generations.",
    )
    parser.add_argument("--popsize", type=int, default=8, help="Population multiplier per optimized radius.")
    parser.add_argument("--mutation", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--recombination", type=float, default=0.7)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--resolution-mm", type=float, default=5.0)
    parser.add_argument("--dsv-mm", type=float, default=200.0)
    parser.add_argument("--num-rings", type=int, default=23)
    parser.add_argument("--ring-separation-m", type=float, default=0.022)
    parser.add_argument("--magnet-size-m", type=float, default=0.012)
    parser.add_argument("--remanence-t", type=float, default=1.3)
    parser.add_argument("--chunk-points", type=int, default=32768)
    parser.add_argument("--log-each-n-generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--seed-file", type=Path, default=Path("data/random_seeds_50.txt"))
    parser.add_argument(
        "--initial-vector",
        type=str,
        default=None,
        help="Comma-separated radius-table indices fixing each ring's magnet count; random when omitted.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def normalize_mutation(value: list[float] | tuple[float, ...]) -> float | tuple[float, float]:
    """Convert one or two CLI mutation values to SciPy's accepted form."""
    if len(value) == 1:
        return value[0]
    if len(value) == 2:
        return (value[0], value[1])
    raise ValueError("--mutation expects one value or a min/max pair.")


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before constructing the LUMC system."""
    if args.max_generations <= 0:
        raise ValueError("max generations must be positive.")
    if args.time_limit_seconds is not None and args.time_limit_seconds <= 0.0:
        raise ValueError("time limit must be positive when provided.")
    if args.popsize <= 0:
        raise ValueError("population multiplier must be positive.")
    if not 0.0 <= args.recombination <= 1.0:
        raise ValueError("recombination must be between zero and one.")
    mutation = normalize_mutation(args.mutation)
    mutation_values = (mutation,) if isinstance(mutation, float) else mutation
    if any(value < 0.0 or value >= 2.0 for value in mutation_values):
        raise ValueError("mutation values must be in [0, 2).")
    if len(mutation_values) == 2 and mutation_values[0] >= mutation_values[1]:
        raise ValueError("mutation minimum must be smaller than its maximum.")
    if args.num_rings <= 0 or args.num_rings % 2 == 0:
        raise ValueError("num rings must be a positive odd number.")
    if args.resolution_mm <= 0.0 or args.dsv_mm <= 0.0:
        raise ValueError("resolution and DSV must be positive.")
    if args.log_each_n_generations <= 0:
        raise ValueError("log interval must be positive.")


def load_run_seeds(seed_file: Path, num_runs: int) -> list[int]:
    """Load the requested number of distinct integer seeds from a text file."""
    seeds = [int(line.strip()) for line in seed_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed file contains duplicate values: {seed_file}")
    if num_runs > len(seeds):
        raise ValueError(f"requested {num_runs} runs, but {seed_file} contains only {len(seeds)} seeds.")
    return seeds[:num_runs]


def move_bound_values_inside(radii: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Move exact endpoint values one float inward for SciPy's x0 normalization."""
    interior_lower = np.nextafter(lower, upper)
    interior_upper = np.nextafter(upper, lower)
    return np.clip(radii, interior_lower, interior_upper)


def run_optimization(args: argparse.Namespace) -> dict[str, object]:
    """Run one continuous-radius differential evolution optimization."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach_de"
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
    initial_radii = INNER_RING_RADII[initial_vector].astype(np.float64)
    model = LUMCRadiusModel(
        initial_configuration,
        torch.as_tensor(initial_radii, device=args.device, dtype=dtype),
        points,
        args.chunk_points,
    ).to(args.device)
    model.eval()

    history = []
    best_error = float("inf")
    best_radii = initial_radii.copy()
    best_field_strength_mt = 0.0
    objective_evaluations = 0
    generation = 0
    optimization_start = time.perf_counter()

    def evaluate(radii: np.ndarray, count_evaluation: bool = True) -> tuple[float, float]:
        nonlocal objective_evaluations
        if count_evaluation:
            objective_evaluations += 1
        radius_tensor = torch.as_tensor(radii, device=args.device, dtype=dtype)
        with torch.inference_mode():
            model.inner_ring_radii.copy_(radius_tensor)
            field_x = model()
            error = float(component_peak_to_peak_ppm(field_x).item())
            field_strength_mt = float(component_field_strength(field_x).item() * 1e3)
        return error, field_strength_mt

    def objective(radii: np.ndarray) -> float:
        nonlocal best_error, best_radii, best_field_strength_mt
        error, field_strength_mt = evaluate(radii)
        if error < best_error:
            best_error = error
            best_radii = radii.copy()
            best_field_strength_mt = field_strength_mt
        return error

    def callback(intermediate_result: OptimizeResult) -> bool:
        nonlocal generation
        generation += 1
        population_energies = np.asarray(intermediate_result.population_energies, dtype=np.float64)
        elapsed = time.perf_counter() - optimization_start
        history.append(
            {
                "step": generation,
                "cumulative_objective_evaluations": objective_evaluations,
                "elapsed_seconds": elapsed,
                "step_best_ppm": float(intermediate_result.fun),
                "step_inner_radii_m": json.dumps(np.asarray(intermediate_result.x).tolist()),
                "best_so_far_ppm": best_error,
                "population_mean_ppm": float(population_energies.mean()),
                "population_std_ppm": float(population_energies.std()),
                "best_field_strength_mt": best_field_strength_mt,
            }
        )
        if generation == 1 or generation % args.log_each_n_generations == 0:
            print(
                f"Generation {generation}: {best_error:.2f} ppm, "
                f"field strength {best_field_strength_mt:.3f} mT, elapsed {elapsed:.1f}s"
            )
        return args.time_limit_seconds is not None and elapsed >= args.time_limit_seconds

    lower_radius = float(INNER_RING_RADII.min())
    upper_radius = float(INNER_RING_RADII.max())
    bounds = [(lower_radius, upper_radius)] * symmetric_positions.size
    initial_de_radii = move_bound_values_inside(initial_radii, lower_radius, upper_radius)
    time_limited = args.time_limit_seconds is not None
    print(f"Initial vector: {initial_vector}")
    print(
        f"Physical systems optimized: 1; continuous radii: {symmetric_positions.size}; "
        f"DE population: {args.popsize * symmetric_positions.size}"
    )
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        maxiter=2_147_483_647 if time_limited else args.max_generations,
        popsize=args.popsize,
        mutation=normalize_mutation(args.mutation),
        recombination=args.recombination,
        tol=-1.0 if time_limited else args.tol,
        atol=-1.0 if time_limited else args.atol,
        seed=args.seed,
        callback=callback,
        polish=args.polish and not time_limited,
        x0=initial_de_radii,
        updating="immediate",
        workers=1,
    )

    final_error, final_field_strength_mt = evaluate(np.asarray(result.x), count_evaluation=False)
    if final_error < best_error:
        best_error = final_error
        best_radii = np.asarray(result.x).copy()
        best_field_strength_mt = final_field_strength_mt
    if not history:
        history.append(
            {
                "step": 1,
                "cumulative_objective_evaluations": int(result.nfev),
                "elapsed_seconds": time.perf_counter() - optimization_start,
                "step_best_ppm": best_error,
                "step_inner_radii_m": json.dumps(best_radii.tolist()),
                "best_so_far_ppm": best_error,
                "population_mean_ppm": best_error,
                "population_std_ppm": 0.0,
                "best_field_strength_mt": best_field_strength_mt,
            }
        )

    with torch.inference_mode():
        model.inner_ring_radii.copy_(torch.as_tensor(best_radii, device=args.device, dtype=dtype))
    configuration = model.get_configuration().detached_cpu()
    elapsed_seconds = time.perf_counter() - optimization_start
    tracker = np.asarray([row["best_so_far_ppm"] for row in history])
    torch.save(configuration.to_dict(), output_directory / "best_configuration.pt")
    np.save(output_directory / "best_inner_radii_m.npy", best_radii)
    np.save(output_directory / "minimum_ppm_by_generation.npy", tracker)
    with (output_directory / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "optimizer": "differential_evolution",
        "best_peak_to_peak_ppm": best_error,
        "best_field_strength_mt": best_field_strength_mt,
        "initial_vector": initial_vector,
        "best_inner_radii_m": best_radii.tolist(),
        "best_outer_radii_m": (best_radii + 21e-3).tolist(),
        "fixed_magnet_counts": True,
        "n_magnets": configuration.num_magnets,
        "elapsed_seconds": elapsed_seconds,
        "objective": "peak_to_peak_Bx_ppm",
        "sampling_point_count": int(points.shape[0]),
        "objective_evaluations": int(result.nfev),
        "generations": generation,
        "success": bool(result.success),
        "message": str(result.message),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    figure, axis = plt.subplots()
    axis.semilogy(np.arange(1, tracker.size + 1), tracker)
    axis.set_title("Differential evolution continuous-radius optimization")
    axis.set_xlabel("Generation")
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
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach_de_runs"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
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
