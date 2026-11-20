# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains LUMC Halbach geometry and magnetic-field backends.
# See her for the original implementation:
# https://github.com/LUMC-LowFieldMRI/HalbachOptimisation
#
#  lumc_halbach.py
#  Kostiantyn Lavronenko
#  18.08.2026
# -----------------------------------------------------------------------------


from __future__ import annotations

import numpy as np
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast

INNER_RING_RADII = np.array(
    [148, 151, 154, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201],
    dtype=np.float64,
) * 1e-3
INNER_NUM_MAGNETS = np.arange(50, 70, dtype=np.int64)
OUTER_RING_RADII = INNER_RING_RADII + 21e-3
OUTER_NUM_MAGNETS = INNER_NUM_MAGNETS + 7


def magnetization(b_rem: float, dimensions: float) -> float:
    """Return the upstream cube dipole moment scaling."""
    return b_rem * dimensions**3 / (4.0 * np.pi * 1e-7)


def single_magnet_upstream(
    position: tuple[float, float, float],
    dipole_moment: np.ndarray,
    sim_dimensions: tuple[float, float, float],
    resolution: float,
) -> np.ndarray:
    """Reproduce upstream ``halbachFields.singleMagnet``."""
    axes = [
        np.linspace(-size / 2.0 + offset, size / 2.0 + offset, int(size * resolution + 1), dtype=np.float32)
        for size, offset in zip(sim_dimensions, position)
    ]
    x, y, z = np.meshgrid(*axes)
    vec_dot_dip = 3.0 * (x * dipole_moment[0] + y * dipole_moment[1])
    vec_mag = np.square(x) + np.square(y) + np.square(z)
    vec_mag_3 = np.power(vec_mag, 1.5)
    vec_mag_5 = np.power(vec_mag, 2.5)
    field = np.zeros((*x.shape, 3), dtype=np.float32)
    field[..., 0] += np.divide(x * vec_dot_dip, vec_mag_5) - np.divide(dipole_moment[0], vec_mag_3)
    field[..., 1] += np.divide(y * vec_dot_dip, vec_mag_5) - np.divide(dipole_moment[1], vec_mag_3)
    field[..., 2] += np.divide(z * vec_dot_dip, vec_mag_5)
    return field


def create_halbach_upstream(
    num_magnets: int = 24,
    rings: tuple[float, ...] = (-0.075, -0.025, 0.025, 0.075),
    radius: float = 0.145,
    magnet_size: float = 0.0254,
    k_value: int = 2,
    resolution: float = 1000.0,
    b_rem: float = 1.3,
    sim_dimensions: tuple[float, float, float] = (0.3, 0.3, 0.2),
) -> np.ndarray:
    """Reproduce upstream ``halbachFields.createHalbach``."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_magnets, endpoint=False)
    dipole_moment = magnetization(b_rem, magnet_size)
    shape = tuple(int(size * resolution) + 1 for size in sim_dimensions)
    field = np.zeros((*shape, 3), dtype=np.float32)
    for row in rings:
        for angle in angles:
            position = (radius * np.cos(angle), radius * np.sin(angle), row)
            dipole_vector = np.array(
                [dipole_moment * np.cos(k_value * angle), dipole_moment * np.sin(k_value * angle)]
            ) * 1e-7
            field += single_magnet_upstream(position, dipole_vector, sim_dimensions, resolution)
    return field


def create_configuration(
    genes: np.ndarray | list[int],
    ring_positions_nonnegative: np.ndarray,
    magnet_size: float = 0.012,
    remanence: float = 1.3,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MagnetConfiguration:
    """Build the two-layer mirrored Halbach array selected by the GA genes."""
    position_parts = []
    orientation_parts = []
    ring_index_parts = []
    selected_radii = []
    for position_index, (z_position, gene) in enumerate(zip(ring_positions_nonnegative, genes)):
        z_positions = (0.0,) if z_position == 0.0 else (-float(z_position), float(z_position))
        for radius, count in (
            (INNER_RING_RADII[gene], INNER_NUM_MAGNETS[gene]),
            (OUTER_RING_RADII[gene], OUTER_NUM_MAGNETS[gene]),
        ):
            angles = torch.arange(int(count), device=device, dtype=dtype) * (2.0 * torch.pi / int(count))
            for z_value in z_positions:
                position_parts.append(
                    torch.stack(
                        (radius * torch.cos(angles), radius * torch.sin(angles), torch.full_like(angles, z_value)), dim=1
                    )
                )
                orientation_parts.append(
                    torch.stack((torch.cos(2.0 * angles), torch.sin(2.0 * angles), torch.zeros_like(angles)), dim=1)
                )
                ring_index_parts.append(torch.full((int(count),), position_index, dtype=torch.int64))
        selected_radii.append((float(INNER_RING_RADII[gene]), float(OUTER_RING_RADII[gene])))
    positions = torch.cat(position_parts)
    orientations = torch.cat(orientation_parts)
    return MagnetConfiguration(
        positions=positions,
        orientations=orientations,
        volumes=torch.full((positions.shape[0],), magnet_size**3, device=device, dtype=dtype),
        remanence=remanence,
        metadata={
            "geometry": "lumc_halbach_optimisation",
            "magnet_size_m": magnet_size,
            "n_magnets": int(positions.shape[0]),
            "best_vector": [int(gene) for gene in genes],
            "ring_positions_nonnegative_m": ring_positions_nonnegative.tolist(),
            "selected_inner_outer_radii_m": selected_radii,
            "ring_indices": torch.cat(ring_index_parts),
            "source": "LUMC-LowFieldMRI/HalbachOptimisation",
        },
    )


def fieldsmith_field_x(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    chunk_points: int | None = 32768,
) -> np.ndarray:
    """Evaluate Bx with Fieldsmith's dipole kernel."""
    field = calc_b_at_points_fast(
        configuration.remanence_vectors(),
        configuration.volumes,
        configuration.positions,
        points.to(device=configuration.positions.device, dtype=configuration.positions.dtype),
        chunk_points=chunk_points,
    )
    return field[:, 0].detach().cpu().numpy()
