# Reproducing the reference experiments

This guide describes the commands used to reproduce the FieldSmith reference
experiments and publication results. Run all commands from the repository root
after installing the locked environment as described in the
[README](../README.md#installation):

```bash
uv sync --locked --extra cu128
```

Paper experiment entry points are grouped in `scripts/paper/`; the LUMC
ring-radius comparison scripts are in its `lumc/` subdirectory.

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
uv run --locked --extra cu128 python scripts/paper/run_fast_halbach_ring_optimisation.py \
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
uv run --locked --extra cu128 python scripts/paper/lumc/homogeneityOptimization.py \
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
uv run --locked --extra cu128 python scripts/paper/lumc/run_lumc_radius_differential_evolution.py \
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
uv run --locked --extra cu128 python scripts/paper/lumc/run_lumc_radius_optimisation.py \
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
uv run --locked --extra cu128 python scripts/paper/run_small_halbach_optimisation.py \
  --fov-size 0.06 \
  --fov-radius 0.025 \
  --resolution 0.001 \
  --wandb-mode disabled
```

The output directory contains the configuration and metrics for each stage,
`small_halbach_final.pt`, and a final `summary.json`.

## ROMA array, opened cylinder and dome

Each of these paper results has one shell script in `scripts/paper/`. It runs with no
arguments, builds the inputs it needs, and writes its final design to
`runs/<name>/final.pt`. An output directory can be given as the first argument.

| Paper result | Command | Runtime on one RTX 5090 |
| --- | --- | --- |
| ROMA after least-squares initialization, 521 to 321 ppm | `scripts/paper/roma_from_lsq_seed.sh` | about 35 hours |
| ROMA refinement at the published seats, 332 ppm | `scripts/paper/roma_refined_at_seats.sh` | about 16 hours |
| Cylinder opened along one side, 672 ppm | `scripts/paper/opened_cylinder.sh` | about 8 hours |
| Dome-shaped head array, 980 ppm | `scripts/paper/dome_head_array.sh` | about 15 hours |

Run them inside the locked environment, for example:

```bash
uv run --locked --extra cu128 bash scripts/paper/roma_from_lsq_seed.sh
```

`SMOKE=1` runs every stage for 20 epochs, which checks an installation in minutes.
`DRY=1` prints the optimizer commands without running them.

**ROMA.** The published arrangement of 1,936 cubes of 12 mm and 384 bars of
12 x 12 x 50 mm is read from `data/osii_rc2.1/` with a remanence of 1.31 T. Every
reported inhomogeneity uses ideal magnets, the full cuboid model and the pinned set
of 65,435 points in `data/roma/roma_roi_200.npz`, a 4 mm lattice inside the 200 mm
sphere offset by 0.37 mm so that no point lies in a magnet's face plane.
`scripts/build_roma_inputs.py` builds the reference and the least-squares seeds,
and the ROMA scripts call it when their inputs are missing. It keeps existing files
unless it is given `--force`.

- `roma_from_lsq_seed.sh` turns every magnet in its plane at the published seats in
  14 restarts from the least-squares seed, at 45.00 mT, which ends at 521 ppm. It then
  opens the cube tilts and the centres within boxes of 3/1, 5/2, 7/3 and 9/4 mm around
  the seats, at 45.17 mT, ending at 410, 376, 352 and 321 ppm. The training uses the
  mixed model, point dipoles for the cubes and cuboids for the bars, which agrees with
  the full cuboid model to 0.2 ppm. The paper's run stored its designs as `.npy` and
  decoded their angles on the GPU, so a rerun with `.pt` designs matches its log at
  epoch 0 and departs from it in the last digits after that.
- `roma_refined_at_seats.sh` refines the published angles in the plane with the full
  cuboid model, seats fixed. The seven restarts reach 344, 339, 336, 334, 333, 332 and
  332 ppm at 45.00 mT. Started from the rebuilt reference, the restarts reached the
  same values and the design reads 332.5 ppm.
- `opened_cylinder.sh` removes a 90 degree sector, keeping 1,726 magnets, and runs
  three stages with every freedom open. The stages end at 1,287, 1,081 and 672 ppm at
  20 mT. The 2 mm translation box re-centres every 1,500 epochs, so centres travel up
  to 20.7 mm within their rings while their axial positions stay fixed.
  Started from the rebuilt seed, which differs from the original by about 1e-7, the
  same stages ended at 1,261, 1,174 and 618 ppm. The truth readings swing by
  hundreds of ppm between checkpoints, so the end value of a rerun varies.

**Dome.** `dome_head_array.sh` builds the lattice of 3,987 cubes on eleven shells of
a 230 mm hemisphere, seeds it by least squares and optimizes it in three stages with
first-order mutual demagnetization (susceptibilities 0.05 and 0.10), the clearance
penalty and the mean field pinned at 40 mT. The stages end at 1,977, 1,540 and 980 ppm,
read on the 367,685 points of `data/head_roi.npz`.

The building blocks these scripts chain are in `scripts/stages/`:
`inplane_restarts.sh`, `free_tilts_then_positions.sh` and `all_freedoms_three_stages.sh`.

## Printable plates

The four designs reported in the paper are in `data/paper/`. The dome's plates can be
exported from its design without rerunning the optimization:

```bash
uv run --locked --extra cu128 python scripts/export_plates.py data/paper/dome_980ppm.pt outputs/dome_plates
```

This writes the eleven plates, their lids, the fastening rails and the insertion order
in about half a minute. The STL files are geometrically identical across library
versions, though not byte for byte, because the mesh libraries may triangulate differently.

## Short verification runs

To verify the installation without completing the full publication experiments,
reduce `--num-epochs` or `--max-generations` and use one explicit seed. For
example:

```bash
uv run --locked --extra cpu python scripts/paper/lumc/run_lumc_radius_optimisation.py \
  --num-epochs 2 \
  --num-runs 1 \
  --seed 42 \
  --device cpu
```

This confirms that the workflow starts and produces its expected files; its
numerical result is not comparable with the full experiment.
