# Architecture

FieldSmith separates magnet geometry, field evaluation, optimization, and
reporting around a common `MagnetConfiguration` data structure. Most workflows
are assembled by example scripts in `scripts/` rather than by a separate orchestration
framework. `src/fieldsmith` summarizes the functionality used in those examples.

See the following UML-structure, that describes how the optimization is constructed: 
[docs/optimization_stage_structure.puml](docs/optimization_stage_structure.puml). In VSCode
you can run it using `PlantUML`-Plugin.

## Data flow

```text
configuration file or CLI arguments
                |
                v
        geometry builder
                |
                v
      MagnetConfiguration <--------- saved checkpoint
                |
        +-------+--------+
        |                |
        v                v
 optimization model   evaluation and verification
        |                |
        v                v
     TrainLoop       metrics and reports
        |
        v
 optimized MagnetConfiguration
```

`MagnetConfiguration` is the exchange format between these stages. It contains
magnet positions, orientations, volumes, remanence, and geometry-specific
metadata. Models convert configurations to their working device and dtype, then
export optimized configurations back to CPU tensors for storage or evaluation.

## Package responsibilities

- `configurations/` defines `MagnetConfiguration` and loaders for experiment
  configuration files and legacy NumPy setups.
- `geometry/` constructs initial Halbach ring, stacked Halbach, ROMA, and dome
  arrangements. It also contains body-frame, collision, and least-squares
  initialization utilities.
- `point_sampler.py` provides Cartesian, masked, and predefined evaluation-point
  samplers.
- `field_simulations/` implements differentiable dipole, finite-cuboid,
  rectangular-prism, and first-order demagnetization calculations.
- `models/` contains PyTorch models for optimizing magnet orientations,
  positions, ring spacing, and ring radii.
- `objectives.py` and `metrics.py` define optimization losses and reported field
  metrics.
- `evaluation/` evaluates stored configurations, compares physics assumptions,
  and runs manufacturing-tolerance studies.
- `visualizations/` contains Matplotlib and Plotly views of magnets and fields.
- `callbacks.py`, `train_loop.py`, and `logger.py` provide optimization-loop
  services such as periodic evaluation, checkpoint selection, and logging.

## Optimization lifecycle

Optimization models (like `SingleRingHalbachModel` in `src\fieldsmith\models\halbach_ring\` 
and `HalbachSystemAngleModel` in `src\fieldsmith\models\halbach_systems\`) are `torch.nn.Module` 
subclasses. A model owns its trainableparameters and implements `forward()`, returning a 
dictionary containing the field and the geometry required by losses and visualizations. Models used with
`TrainLoop` also provide scalar logging, image logging, and configuration export
methods.

For every optimization step, `TrainLoop`:

1. evaluates the model and loss;
2. performs backpropagation and an optimizer step;
3. applies optional parameter constraints and callbacks;
4. logs metrics, images, and configuration checkpoints.

The executable scripts create the geometry, point sampler, model, objective,
optimizer, callbacks, and logger for a particular experiment. Multi-stage
workflows pass the exported `MagnetConfiguration` from one model to the next.
