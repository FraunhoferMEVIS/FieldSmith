"""LUMC Halbach ring-radius genetic optimization with selectable field backend."""

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

from deap import base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fieldsmith.evaluation.lumc_halbach import (
    INNER_NUM_MAGNETS,
    INNER_RING_RADII,
    OUTER_NUM_MAGNETS,
    OUTER_RING_RADII,
    create_configuration,
    create_halbach_upstream,
    fieldsmith_field_x,
)
from fieldsmith.metrics import component_field_strength, component_peak_to_peak_ppm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize LUMC Halbach ring radii using the original genetic algorithm.")
    parser.add_argument("--field-backend", choices=("upstream", "fieldsmith"), default="fieldsmith")
    parser.add_argument(
        "--field-evaluation",
        choices=("precomputed", "direct"),
        default="precomputed",
        help="Use ring-field lookup tables or simulate every candidate configuration directly.",
    )
    parser.add_argument("--population-size", type=int, default=10000)
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--resolution-mm", type=float, default=5.0)
    parser.add_argument("--dsv-mm", type=float, default=200.0)
    parser.add_argument("--num-rings", type=int, default=23)
    parser.add_argument("--ring-separation-m", type=float, default=0.022)
    parser.add_argument("--magnet-size-m", type=float, default=0.012)
    parser.add_argument("--remanence-t", type=float, default=1.3)
    parser.add_argument("--chunk-points", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=None, help="Single-run seed override; requires --num-runs 1.")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed-file", type=Path, default=Path("data/random_seeds.txt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def field_error(field: np.ndarray) -> tuple[float]:
    """Original peak-to-peak Bx objective in ppm."""
    value = component_peak_to_peak_ppm(torch.from_numpy(field))
    return (float(value.item()),)


def load_run_seeds(seed_file: Path, num_runs: int) -> list[int]:
    """Load the requested number of distinct integer seeds from a text file."""
    seeds = [int(line.strip()) for line in seed_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed file contains duplicate values: {seed_file}")
    if num_runs > len(seeds):
        raise ValueError(f"requested {num_runs} runs, but {seed_file} contains only {len(seeds)} seeds.")
    return seeds[:num_runs]


def run_optimization(args: argparse.Namespace) -> dict[str, object]:
    """Run one genetic optimization and return its saved summary."""
    if args.population_size <= 0 or args.max_generations <= 0:
        raise ValueError("population size and max generations must be positive.")
    if args.field_evaluation == "direct" and args.field_backend != "fieldsmith":
        raise ValueError("direct field evaluation requires --field-backend fieldsmith.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_directory = args.output_directory or Path("logs") / datetime.now().strftime(
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    dsv = args.dsv_mm * 1e-3
    sim_dimensions = (dsv, dsv, dsv)
    coordinate_axis = np.linspace(-dsv / 2.0, dsv / 2.0, int(args.dsv_mm / args.resolution_mm + 1))
    coords = np.meshgrid(coordinate_axis, coordinate_axis, coordinate_axis)
    mask = np.square(coords[0]) + np.square(coords[1]) + np.square(coords[2]) <= (dsv / 2.0) ** 2
    octant_mask = mask & (coords[0] >= 0.0) & (coords[1] >= 0.0) & (coords[2] >= 0.0)
    points = torch.as_tensor(
        np.stack((coords[0][octant_mask], coords[1][octant_mask], coords[2][octant_mask]), axis=1),
        device=args.device,
        dtype=torch.float64,
    )
    ring_positions = np.linspace(
        -(args.num_rings - 1) * args.ring_separation_m / 2.0,
        (args.num_rings - 1) * args.ring_separation_m / 2.0,
        args.num_rings,
    )
    symmetric_positions = ring_positions[ring_positions >= 0.0]
    shim_fields = None
    precompute_seconds = 0.0
    if args.field_evaluation == "precomputed":
        shim_fields = np.zeros((int(octant_mask.sum()), symmetric_positions.size, INNER_RING_RADII.size))
        print(f"Precomputing fields with {args.field_backend} backend...")
        precompute_start = time.perf_counter()
        for position_index, position in enumerate(symmetric_positions):
            rings = (0.0,) if position == 0.0 else (-float(position), float(position))
            for size_index in range(INNER_RING_RADII.size):
                if args.field_backend == "upstream":
                    field = create_halbach_upstream(
                        int(INNER_NUM_MAGNETS[size_index]), rings, float(INNER_RING_RADII[size_index]),
                        args.magnet_size_m, resolution=1e3 / args.resolution_mm,
                        b_rem=args.remanence_t, sim_dimensions=sim_dimensions,
                    )
                    field += create_halbach_upstream(
                        int(OUTER_NUM_MAGNETS[size_index]), rings, float(OUTER_RING_RADII[size_index]),
                        args.magnet_size_m, resolution=1e3 / args.resolution_mm,
                        b_rem=args.remanence_t, sim_dimensions=sim_dimensions,
                    )
                    shim_fields[:, position_index, size_index] = field[octant_mask, 0]
                else:
                    partial = create_configuration(
                        [size_index], np.array([position]), args.magnet_size_m, args.remanence_t,
                        args.device, torch.float64,
                    )
                    shim_fields[:, position_index, size_index] = fieldsmith_field_x(
                        partial, points, args.chunk_points
                    )
            print(
                f"Precompute {position_index + 1}/{symmetric_positions.size}: "
                f"{time.perf_counter() - precompute_start:.1f}s elapsed"
            )
        precompute_seconds = time.perf_counter() - precompute_start
    else:
        print(f"Evaluating every candidate directly on {args.device} with the fieldsmith backend.")

    def candidate_field(individual: list[int]) -> np.ndarray:
        """Evaluate one candidate through the selected field-evaluation path."""
        if shim_fields is not None:
            field = np.zeros(shim_fields.shape[0])
            for gene_index, gene in enumerate(individual):
                field += shim_fields[:, gene_index, gene]
            return field
        configuration = create_configuration(
            individual,
            symmetric_positions,
            args.magnet_size_m,
            args.remanence_t,
            args.device,
            torch.float64,
        )
        return fieldsmith_field_x(configuration, points, args.chunk_points)

    def evaluate(individual: list[int]) -> tuple[float]:
        return field_error(candidate_field(individual))

    def field_strength_mt(individual: list[int]) -> float:
        """Return the mean absolute Bx field strength in millitesla."""
        field = candidate_field(individual)
        return float(component_field_strength(torch.from_numpy(field)).item() * 1e3)

    if not hasattr(creator, "FitnessMinLUMC"):
        creator.create("FitnessMinLUMC", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualLUMC"):
        creator.create("IndividualLUMC", list, fitness=creator.FitnessMinLUMC)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, INNER_RING_RADII.size - 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.IndividualLUMC, toolbox.attr_bool, symmetric_positions.size,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=args.population_size)
    for individual, fitness in zip(population, map(toolbox.evaluate, population)):
        individual.fitness.values = fitness
    best_error = np.inf
    best_vector = None
    best_field_strength_mt = 0.0
    min_tracker = np.zeros(args.max_generations)
    history: list[dict[str, int | float | str]] = []
    cumulative_evaluations = args.population_size
    evolution_start = time.perf_counter()
    for generation in range(1, args.max_generations + 1):
        generation_start = time.perf_counter()
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.55:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.4:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness
        cumulative_evaluations += len(invalid)
        population[:] = offspring
        current = tools.selBest(population, 1)[0]
        current_error = current.fitness.values[0]
        current_field_strength_mt = field_strength_mt(current)
        if current_error < best_error:
            best_error = current_error
            best_vector = list(current)
            best_field_strength_mt = current_field_strength_mt
            print(f"BEST VECTOR: {best_vector}")
        min_tracker[generation - 1] = current_error
        elapsed = time.perf_counter() - evolution_start
        fitness_values = np.asarray([individual.fitness.values[0] for individual in population])
        history.append(
            {
                "step": generation,
                "cumulative_objective_evaluations": cumulative_evaluations,
                "elapsed_seconds": elapsed,
                "step_best_ppm": current_error,
                "step_inner_radii_m": json.dumps(INNER_RING_RADII[list(current)].tolist()),
                "best_so_far_ppm": best_error,
                "population_mean_ppm": float(fitness_values.mean()),
                "population_std_ppm": float(fitness_values.std()),
                "best_field_strength_mt": best_field_strength_mt,
            }
        )
        remaining = elapsed / generation * (args.max_generations - generation)
        print(
            f"Generation {generation}/{args.max_generations}: minimum {current_error:.0f} ppm, "
            f"field strength {current_field_strength_mt:.3f} mT, "
            f"generation {time.perf_counter() - generation_start:.2f}s, ETA {remaining:.1f}s"
        )

    assert best_vector is not None
    configuration = create_configuration(
        best_vector, symmetric_positions, args.magnet_size_m, args.remanence_t, "cpu", torch.float64
    )
    torch.save(configuration.to_dict(), output_directory / "best_configuration.pt")
    np.save(output_directory / "best_vector.npy", np.asarray(best_vector, dtype=np.int64))
    np.save(output_directory / "minimum_ppm_by_generation.npy", min_tracker)
    with (output_directory / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "optimizer": "genetic_algorithm",
        "field_backend": args.field_backend,
        "field_evaluation": args.field_evaluation,
        "best_peak_to_peak_ppm": best_error,
        "best_field_strength_mt": best_field_strength_mt,
        "best_vector": best_vector,
        "n_magnets": configuration.num_magnets,
        "elapsed_seconds": time.perf_counter() - evolution_start,
        "precompute_seconds": precompute_seconds,
        "objective": "peak_to_peak_Bx_ppm",
        "sampling_point_count": int(points.shape[0]),
        "objective_evaluations": cumulative_evaluations,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    figure, axis = plt.subplots()
    axis.semilogy(min_tracker)
    axis.set_title("Min error Vs generations")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Error")
    figure.savefig(output_directory / "convergence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation-ready configuration to {(output_directory / 'best_configuration.pt').resolve()}")
    return summary


def main() -> None:
    args = parse_args()
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
        "%Y_%m_%d/%Hh%Mm%Ss_lumc_halbach_runs"
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
