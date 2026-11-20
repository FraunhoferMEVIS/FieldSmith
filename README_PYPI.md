# FieldSmith

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Version 0.1.4](https://img.shields.io/badge/version-0.1.4-2F6F8F)

FieldSmith is a PyTorch-based framework for designing, optimizing, and evaluating permanent-magnet arrays. It provides differentiable magnetic-field models, reusable geometry and sampling components, gradient-based optimization workflows, and tools for checking designs under realistic physics and manufacturing tolerances.

The package focuses on Halbach rings and multi-ring systems, the ROMA geometry, and open-cap/dome arrays for low-field MRI applications.

## Disclaimer

The software is not qualified for use as a medical product or as part thereof. Provided 'as is' without specific verification or validation.


## Capabilities

- Build and optimize geometries such as Halbach rings, multi-ring, ROMA, and dome magnet arrays.
- Calculate differentiable fields with fast dipole or finite cuboid models.
- Run gradient-based optimization of magnet orientations, positions, and other parameters within the PyTorch framework.
- Evaluate field strength, peak-to-peak homogeneity, and RMS deviation.
- Run Monte Carlo studies of manufacturing tolerances.
- Move magnets within bounded boxes under a clearance penalty, and export a design as printable plates with seats, lids and fastening rails.


## Installation

FieldSmith requires Python 3.11 or newer.

```bash
python -m pip install fieldsmith
```

For a specific CPU or CUDA build, install a compatible version of PyTorch first,
then install FieldSmith. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Quick start

Compute the magnetic field at two points outside a single cubic magnet. Positions
are in meters, volumes are in cubic meters, and the returned field is in tesla.

```python
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation import field_at_points

magnet_size_m = 0.01
configuration = MagnetConfiguration(
    positions=torch.zeros((1, 3), dtype=torch.float64),
    orientations=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    volumes=torch.tensor([magnet_size_m**3], dtype=torch.float64),
    remanence=1.2,
    metadata={"magnet_size_m": magnet_size_m},
)
points = torch.tensor([[0.03, 0.0, 0.0], [0.0, 0.03, 0.0]], dtype=torch.float64)
field = field_at_points(configuration, points, field_model="cuboid")
print(field)
```

The kernels preserve PyTorch gradients and support point chunking to control memory use. Field points can come from a Cartesian grid, a masked region of interest, or an arbitrary point cloud. Optimization objectives include exact and smoothed peak-to-peak homogeneity, RMS deviation, field-strength constraints, and collision penalties.

## Outputs and checkpoints

`TrainLoop` stores periodic `MagnetConfiguration` checkpoints at
`configurations/epoch_<N>.pt` under the configured logger directory, including
when W&B logging is disabled.

## License

FieldSmith is distributed under the **FRAUNHOFER »fieldsmith« LICENSE FOR
SCIENTIFIC NON-COMMERCIAL RESEARCH PURPOSES**. The full license is included in
the package as `LICENSE`.

## Citing

As soon as the paper is published the citing template of the software-related paper will be added here.

## Support or contact

For support with regard to the **FieldSmith** software please contact:

- Kostiantyn Lavronenko: [kostiantyn.lavronenko@mevis.fraunhofer.de](mailto:kostiantyn.lavronenko@mevis.fraunhofer.de)
- Marian Frei: [marian.frei@lfb.rwth-aachen.de](mailto:marian.frei@lfb.rwth-aachen.de)
