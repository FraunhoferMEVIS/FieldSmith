# Examples

Run these examples from the repository root after installing FieldSmith.

## Generic Halbach system optimization

[`scripts/example_system_optimization.py`](../scripts/example_system_optimization.py)
is a compact end-to-end test of the optimization framework. It creates a
three-ring Halbach system, evaluates its initial field, optimizes the in-plane
magnetization angles, and reports the resulting field metrics.

```bash
uv run --extra cpu python scripts/example_system_optimization.py
```

The example uses a CUDA device when one is available and otherwise runs on the
CPU. For a faster smoke test, run one epoch explicitly:

```bash
uv run --extra cpu python scripts/example_system_optimization.py --num-epochs 1 --device cpu
```

Checkpoints are written to a timestamped directory below `logs/`. Use
`--output-directory` to select another location.

## Open-cap/dome array optimization

The dome workflow uses finite cuboid physics, optional mutual demagnetization,
least-squares orientation seeding, and a smoothed minimax objective:

```bash
uv run --extra cu128 python scripts/run_dome_optimisation.py \
  --config configs/dome_geometry.cfg \
  --double-precision \
  --wandb-mode disabled
```

Geometry, field-of-view, seeding, and physics settings live in
[`configs/dome_geometry.cfg`](../configs/dome_geometry.cfg). See the
[dome optimizer documentation](dome_optimizer.md) for detailed design notes.
Run any script with `--help` for all options.

## Evaluation and robustness

Use `field_at_points()` or `evaluate_configuration()` to score a design under
chosen physics. `truth_eval_report()` compares ideal finite-cuboid physics with
a demagnetized calculation and reports overlaps.

Run a manufacturing-tolerance study with:

```bash
uv run --extra cu128 python scripts/run_monte_carlo_evaluation.py path/to/configuration.pt \
  --num-samples 100 \
  --field-model cuboid \
  --output-directory output/monte_carlo
```

The command writes `summary.json`, `samples.csv`, `results.pt`, and
`distributions.png`.

## Geometry verification

Check a saved configuration for minimum spacing and oriented-box intersections:

```bash
uv run --extra cpu python scripts/check_magnet_geometry.py path/to/configuration.pt
```

The verifier can also write a JSON report and images of intersecting magnet
pairs. Use `--help` for its tolerance, body-frame, and output options.
