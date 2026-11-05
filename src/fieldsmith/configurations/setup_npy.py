# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file loads and saves the legacy optimized-setup .npy exchange format.
#
#  setup_npy.py
#  Marian Frei
#  24.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fieldsmith.configurations import MagnetConfiguration


def load_halbach_setup_npy(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MagnetConfiguration:
    """
    Load a legacy optimized-setup .npy into a MagnetConfiguration.

    The file is a pickled dict of magnet_positions (M, 3), magnet_angles_theta (M,),
    magnet_angles_phi (M,), remanence (scalar) and magnet_amplitudes (M,). The
    spherical angles become unit orientations; the volume is size**3.

    Non-unit amplitudes are rejected: the reused first-order demagnetization assumes
    every magnet is at full remanence.
    """
    data = np.load(Path(path), allow_pickle=True).item()
    positions = torch.tensor(data["magnet_positions"], dtype=dtype, device=device)
    theta = torch.tensor(data["magnet_angles_theta"], dtype=dtype, device=device)
    phi = torch.tensor(data["magnet_angles_phi"], dtype=dtype, device=device)
    amplitudes = torch.tensor(
        data.get("magnet_amplitudes", np.ones(positions.shape[0])), dtype=dtype, device=device
    )
    if not torch.allclose(amplitudes, torch.ones_like(amplitudes)):
        raise ValueError("non-unit magnet_amplitudes are not supported (demag assumes |m| = remanence).")
    orientations = torch.stack(
        (theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), dim=1
    )
    size = float(data.get("magnet_size_m", 0.012))
    # Mixed arrangements such as ROMA store volumes. Without the key every magnet is one cube.
    stored = data.get("magnet_volumes")
    if stored is None:
        volumes = torch.full((positions.shape[0],), size**3, dtype=dtype, device=device)
    else:
        volumes = torch.tensor(np.asarray(stored, dtype=np.float64), dtype=dtype, device=device)
        if volumes.shape != (positions.shape[0],):
            raise ValueError("magnet_volumes must hold one volume per magnet.")
    # They also store side lengths, which the cuboid model needs.
    meta = {"magnet_size_m": size, "source": str(path)}
    shapes = data.get("magnet_sizes_m")
    if shapes is not None:
        sides = torch.tensor(np.asarray(shapes, dtype=np.float64), dtype=dtype, device=device)
        if sides.shape != positions.shape:
            raise ValueError("magnet_sizes_m must hold three side lengths per magnet.")
        meta["magnet_dimensions_m"] = sides
    return MagnetConfiguration(
        positions=positions,
        orientations=orientations,
        volumes=volumes,
        remanence=float(data["remanence"]),
        metadata=meta,
    )


def save_halbach_setup_npy(path: str | Path, configuration: MagnetConfiguration) -> None:
    """Serialize a MagnetConfiguration back to the angle+position .npy exchange format."""
    config = configuration.detached_cpu()
    unit = config.orientations / config.orientations.norm(dim=1, keepdim=True).clamp_min(1e-30)
    theta = torch.arccos(unit[:, 2].clamp(-1.0, 1.0))
    phi = torch.arctan2(unit[:, 1], unit[:, 0])
    remanence = config.remanence
    np.save(
        Path(path),
        {
            "magnet_positions": config.positions.numpy(),
            "magnet_angles_theta": theta.numpy(),
            "magnet_angles_phi": phi.numpy(),
            "remanence": float(remanence.item() if isinstance(remanence, torch.Tensor) else remanence),
            "magnet_amplitudes": np.ones(config.num_magnets),
            "magnet_volumes": config.volumes.numpy(),
            "magnet_size_m": float(config.metadata.get("magnet_size_m", 0.012)),
            **({"magnet_sizes_m": config.metadata["magnet_dimensions_m"].numpy()}
               if isinstance(config.metadata.get("magnet_dimensions_m"), torch.Tensor) else {}),
        },
        allow_pickle=True,
    )
