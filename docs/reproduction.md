# Reproducing the reference experiments

This guide describes the commands used to reproduce the FieldSmith reference
experiments and publication results. Run all commands from the repository root
after installing the locked environment as described in the
[README](../README.md#installation):

```bash
uv sync --locked --extra cu128
```

The experiments use stochastic optimization. The radius-comparison experiments
therefore use the same 50 random seeds for every optimizer. They are
computationally intensive and are intended to reproduce the full publication
results, not to serve as quick smoke tests. A CUDA-capable GPU is recommended;
pass `--device cpu` to supported scripts when a GPU is unavailable.

Unless `--output-directory` is supplied, each command writes its checkpoints,
metrics, and plots to a timestamped directory below `logs/`. Consult the
generated `summary.json`, `history.csv`, or training logs for the final metrics
and elapsed time.

## Single-ring benchmark

This experiment optimizes the in-plane magnetization angles of a 36-magnet
Halbach ring over a spherical field of view:

```bash
uv run --locked --extra cu128 python scripts/run_fast_halbach_ring_optimisation.py \
  --num-epochs 10000 \
  --n-magnets 36 \
  --ring-radius 0.135 \
  --fov-radius 0.0675 \
  --magnet-size 0.012 \
  --remanence 1.3 \
  --double-precision \
  --wandb-mode disabled
```

Distances are expressed in metres and remanence in tesla. The run writes
periodic configuration checkpoints below
`logs/<timestamp>_fast_halbach_ring/configurations/`.

## Ring-radius optimizer comparison

The following experiments compare genetic algorithm, differential evolution,
and gradient-based optimization of the LUMC Halbach ring radii. All three
commands use the same geometry, field-of-view sampling, and 50 seeds from
`data/random_seeds_50.txt`. The seed file must contain at least 50 distinct
integers, one per line.

### Genetic algorithm

```bash
uv run --locked --extra cu128 python scripts/lumc/homogeneityOptimization.py \
  --population-size 10000 \
  --max-generations 100 \
  --resolution-mm 5.0 \
  --dsv-mm 200 \
  --num-rings 23 \
  --ring-separation-m 0.022 \
  --magnet-size-m 0.012 \
  --remanence-t 1.3 \
  --num-runs 50 \
  --seed-file data/random_seeds_50.txt
```

### Differential evolution

```bash
uv run --locked --extra cu128 python scripts/lumc/run_lumc_radius_differential_evolution.py \
  --popsize 8 \
  --max-generations 200 \
  --recombination 0.7 \
  --resolution-mm 5.0 \
  --dsv-mm 200 \
  --num-rings 23 \
  --ring-separation-m 0.022 \
  --magnet-size-m 0.012 \
  --remanence-t 1.3 \
  --num-runs 50 \
  --seed-file data/random_seeds_50.txt
```

### Gradient-based optimization

```bash
uv run --locked --extra cu128 python scripts/lumc/run_lumc_radius_optimisation.py \
  --num-epochs 3000 \
  --resolution-mm 5.0 \
  --dsv-mm 200 \
  --num-rings 23 \
  --ring-separation-m 0.022 \
  --magnet-size-m 0.012 \
  --remanence-t 1.3 \
  --num-runs 50 \
  --seed-file data/random_seeds_50.txt
```

The differential-evolution generation limit and gradient-based epoch limit are
deliberately generous. For the publication comparison, their histories are
truncated at the wall-clock time consumed by the corresponding genetic-algorithm
run. This gives each optimizer the same time budget rather than the same number
of update steps. Each run directory contains the optimized configuration,
convergence data and plot, and a `summary.json`; a 50-run experiment additionally
writes `runs_summary.json` in its parent directory.

## Small Halbach demonstrator

This command reproduces the three-stage optimization of the small Halbach
demonstrator. It sequentially optimizes ring distances, ring radii, and magnet
angles:

```bash
uv run --locked --extra cu128 python scripts/run_small_halbach_optimisation.py \
  --fov-size 0.06 \
  --fov-radius 0.025 \
  --resolution 0.001 \
  --wandb-mode disabled
```

The output directory contains the configuration and metrics for each stage,
`small_halbach_final.pt`, and a final `summary.json`.

## Tolerance-aware differentiable magnet optimization

This experiment compares tolerance-aware with conventional optimization of a
stacked Halbach system: 12 planes, mirrored about z = 0, each with three
concentric rings of 48, 55, and 62 cubes of 12 mm (1980 magnets), around a
150 mm diameter spherical volume. All magnet angles, the radius of every
concentric ring per mirrored plane pair, and the plane heights are optimized;
the mean field is held at 47 mT. The tolerances are position ±0.5 mm,
magnetization tilt ±1.2°, and remanence ±2%.

Every run uses `--objective both`: the conventional optimization of the
unperturbed design and the tolerance-aware optimization on 500 perturbed copies
per step, with identical seeds, parameters, and schedule. The tolerance-aware
loss is the mean smoothed peak-to-peak of the copies plus the sensitivity term weighted 
by `w_S`, plus the field penalty. The study varies the sensitivity weight, `w_S` in {0, 3, 10, 30}:

```bash
for w_s in 0 3 10 30; do
  uv run --locked --extra cu128 python scripts/run_tolerance_aware_halbach_system_optimisation.py \
    --objective both \
    --min-field-mt 47 \
    --robust-loss wasserstein --wasserstein-target parent \
    --wasserstein-mean-weight 1 --wasserstein-weight 0 \
    --sensitivity-weight $w_s \
    --num-samples 500 \
    --comparison-samples 5000 --comparison-designs best,final \
    --random-seed 42 \
    --output-directory logs/tadmo/wW0_wS$w_s
done
```

These are the exact settings of the reported runs. The Wasserstein term has
weight 0, so the loss is the mean of the copies plus the sensitivity term; the
same loss without the `geomloss` dependency is
`--robust-loss mean_cvar --cvar-weight 0 --std-weight 0`.

Each run directory contains `nominal/` and `robust/` with their checkpoints
(`initial.pt`, `final.pt`, `best.pt`), histories, and system views, and the
comparison of both designs under 5000 independent Monte Carlo builds on a 5 mm
grid of the DSV: `comparison_summary_final.json`,
`comparison_monte_carlo_final.png`, and the raw samples in
`comparison_monte_carlo_results_final.pt` (and the same with `_best`). The
reported results use the final designs. The conventional design does not depend
on `w_S` and is identical in every run.

A run takes about 2.5 h on an RTX 4090 (about 6 GB of GPU memory, float64) and up
to about 10 h on GPUs with lower float64 throughput.

The effect of each tolerance alone (position, rotation, remanence, all together)
on the conventional and the best tolerance-aware design is evaluated with
2000 Monte Carlo builds per case:

```bash
for design in nominal robust; do
  uv run --locked --extra cu128 python scripts/run_tolerance_breakdown.py \
    --configuration logs/tadmo/wW0_wS30/$design/final.pt \
    --num-samples 2000 \
    --output-directory logs/tadmo/breakdown/$design
done
```

It writes `tolerance_breakdown.csv`, `summary.json`, and the distribution and
contribution plots for each design.

## Short verification runs

To verify the installation without completing the full publication experiments,
reduce `--num-epochs` or `--max-generations` and use one explicit seed. For
example:

```bash
uv run --locked --extra cpu python scripts/lumc/run_lumc_radius_optimisation.py \
  --num-epochs 2 \
  --num-runs 1 \
  --seed 42 \
  --device cpu
```

This confirms that the workflow starts and produces its expected files; its
numerical result is not comparable with the full experiment.
