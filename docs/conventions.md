# Magnet configuration conventions

`MagnetConfiguration` is FieldSmith's canonical exchange object for a concrete
magnet arrangement. Geometry builders create it, optimization models return it
from `get_configuration()`, checkpoint writers serialize it, and evaluation and
verification tools load it again. Using this object keeps the physical system
separate from optimizer state, field-of-view samples, losses, and results.

The implementation is in
[`src/fieldsmith/configurations/magnet_configuration.py`](../src/fieldsmith/configurations/magnet_configuration.py).

## In-memory representation

A `MagnetConfiguration` contains the following fields, where `M` is the number
of physical magnets:

| Attribute | Type and shape | Meaning |
| --- | --- | --- |
| `positions` | `torch.Tensor`, `(M, 3)` | Magnet-centre coordinates in metres, ordered as x, y, z. |
| `orientations` | `torch.Tensor`, `(M, 3)` | Magnetization directions in world coordinates. These are conventionally unit vectors. |
| `volumes` | `torch.Tensor`, `(M,)` | Magnet volumes in cubic metres. |
| `remanence` | `float` or `torch.Tensor` | Remanence in tesla: one scalar, `(M,)` magnitudes, or `(M, 3)` remanence vectors. |
| `metadata` | `dict[str, Any]` | Geometry, dimensions, parameterization, provenance, and other descriptive information. |

The row at index `i` in `positions`, `orientations`, and `volumes`, and in a
per-magnet `remanence`, always describes the same magnet. `num_magnets` returns
`M`.

`remanence_vectors()` produces the `(M, 3)` vectors used by the field kernels:

- scalar remanence: `orientations * remanence`;
- `(M,)` remanence: `orientations * remanence[:, None]`;
- `(M, 3)` remanence: the stored vectors are returned directly.

Positions and dimensions use metres, volumes use cubic metres, angles use
radians, and magnetic flux density and remanence use tesla unless a name
explicitly declares another unit. Unit suffixes should be used for persisted
dictionary keys and metadata values.

## Canonical `.pt` dictionary

Call `MagnetConfiguration.to_dict()` before saving. It detaches the four core
tensor fields, moves them to the CPU, and returns this dictionary:

```python
{
    "format_version": "0.1.0",
    "magnet_positions_m": torch.Tensor,     # shape (M, 3)
    "magnet_orientations": torch.Tensor,    # shape (M, 3)
    "volumes_m3": torch.Tensor,             # shape (M,)
    "remanence_t": float | torch.Tensor,
    "metadata": dict[str, object],
}
```

`format_version` identifies the checkpoint schema and is mandatory for new
files. The four physical keys are also mandatory. `metadata` is optional when
reading and defaults to an empty dictionary, but writers should include it.
Optional descriptors are stored only in the nested `metadata` dictionary. The
loader accepts unversioned dictionaries created before version `0.1.0` for
backward compatibility, but rejects an explicit unsupported version.

A `.pt` file represents a magnet configuration only when its payload is this
dictionary or, for backward compatibility, a directly pickled
`MagnetConfiguration` instance. The extension alone does not identify the
payload type.

## Metadata

Metadata supplements the physical core; it must not be required to interpret
the four canonical fields. It may be required by geometry-specific operations,
such as collision checking or resuming a specialized optimization stage.

Common metadata keys are:

| Key | Meaning |
| --- | --- |
| `geometry` | Geometry identifier, for example `halbach_ring`, `halbach_system`, `small_halbach`, `dome`, `roma`, or `lumc_halbach_optimisation`. |
| `n_magnets` | Number of physical magnets; redundant with `num_magnets`, but useful for inspection. |
| `magnet_size_m` | Cube edge length in metres when all magnets are equal cubes. |
| `magnet_dimensions_m` | Full body dimensions with shape `(M, 3)` for non-cubic or mixed-size magnets. |
| `initial_magnet_angles_rad` | Initial azimuthal angles used to construct a Halbach geometry. |
| `magnet_angles_rad` | Current optimized azimuthal parameters. |
| `magnet_polar_angles_rad` | Current polar angles when out-of-plane orientations are represented. |
| `field_model` | Field model associated with the exported design, such as `dipole` or `cuboid`. |
| `source` | Human-readable provenance for an imported or derived configuration. |

Geometry-specific keys such as ring radii, ring positions, layer counts,
susceptibilities, or target directions should follow the same unit-bearing
naming convention. Tensor metadata should be detached and stored on the CPU.
Metadata remains separate from the mandatory top-level fields, so it cannot
overwrite the physical configuration or its format version.

`magnet_orientations` records nominal design directions. A model that calculates
effective demagnetized remanence can store those vectors separately as
`metadata["remanence_vectors_t"]`; they do not replace the nominal orientation
and remanence fields.

## Saving a configuration

The canonical explicit save operation is:

```python
from pathlib import Path

import torch

checkpoint_path = Path("configuration.pt")
torch.save(configuration.to_dict(), checkpoint_path)
```

Optimization models should first export their current physical state through
`model.get_configuration()`. Saving `model.state_dict()` is a different
operation: it stores model parameters and buffers, not the stable physical
configuration exchange format.

`WandbLogger.save_configuration()` performs the canonical conversion whenever
the supplied object has a `to_dict()` method. `TrainLoop` uses it for checkpoints
named `configurations/epoch_<N>.pt`. Several runners also save a final
`best_configuration.pt` or `small_halbach_final.pt` with the same dictionary
format.

## Loading a configuration

Use `MagnetConfiguration.from_file()` rather than calling `torch.load()` in
application code:

```python
import torch

from fieldsmith.configurations import MagnetConfiguration

configuration = MagnetConfiguration.from_file(
    "configuration.pt",
    device="cpu",
    dtype=torch.float64,
)
```

The loader:

1. loads onto the requested device;
2. accepts either the canonical dictionary or a legacy pickled
   `MagnetConfiguration`;
3. validates the format version, checks the mandatory dictionary keys, and
   verifies that metadata, when present, is a dictionary;
4. constructs and shape-validates the configuration; and
5. converts its tensor fields to the requested device and optional dtype.

PyTorch checkpoints use pickle internally. Load only files from trusted sources;
the current loader sets `weights_only=False` to support both accepted payload
forms.

## Complete round trip

```python
import torch

from fieldsmith.configurations import MagnetConfiguration

configuration = MagnetConfiguration(
    positions=torch.tensor([[0.10, 0.00, 0.00]]),
    orientations=torch.tensor([[1.00, 0.00, 0.00]]),
    volumes=torch.tensor([0.012**3]),
    remanence=1.3,
    metadata={
        "geometry": "example",
        "n_magnets": 1,
        "magnet_size_m": 0.012,
    },
)

torch.save(configuration.to_dict(), "configuration.pt")
loaded = MagnetConfiguration.from_file(
    "configuration.pt",
    device="cpu",
    dtype=torch.float64,
)
```

The loaded object can be passed to a FieldSmith model, evaluation function,
Monte Carlo workflow, or geometry verifier without knowledge of the optimizer
that created it.

## Importing legacy `.npy` configurations

`load_halbach_setup_npy()` converts the older position-and-spherical-angle NumPy
format into a `MagnetConfiguration`. `save_halbach_setup_npy()` exists for
interoperability with that format. `.npy` is a legacy exchange path; new
FieldSmith checkpoints should use the canonical `.pt` dictionary.
