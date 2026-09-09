# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains geometry construction for mirrored ROMA magnet arrangements.
#
#  roma.py
#  Kostiantyn Lavronenko
#  13.08.2026
# -----------------------------------------------------------------------------

"""Geometry construction for the mirrored ROMA magnet arrangement."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from fieldsmith.configurations import MagnetConfiguration


class RomaGeometry:
    """Build ROMA v2.1 from the nine supplied ring-setting files."""

    def __init__(
        self,
        ring_directory: str | Path,
        remanence: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        magnet_size: float = 0.012,
        outer_magnet_length: float = 0.050,
        use_halbach_orientations: bool = False,
    ) -> None:
        if min(remanence, magnet_size, outer_magnet_length) <= 0.0:
            raise ValueError("remanence and magnet dimensions must be positive.")
        self.ring_directory: Path = Path(ring_directory)
        self.remanence: float = remanence
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype
        self.magnet_size: float = magnet_size
        self.outer_magnet_length: float = outer_magnet_length
        self.use_halbach_orientations: bool = use_halbach_orientations

    def _ring_files(self) -> list[Path]:
        ring_files = [self.ring_directory / f"ring {index}.txt" for index in range(9)]
        missing_files = [path.name for path in ring_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing ROMA ring files: {', '.join(missing_files)}")
        return ring_files

    @staticmethod
    def _read_ring(path: Path) -> tuple[list[list[float]], list[float]]:
        positions = []
        angles = []
        with path.open(newline="", encoding="utf-8") as ring_file:
            for row_number, row in enumerate(csv.reader(ring_file), start=1):
                if len(row) != 4:
                    raise ValueError(f"{path}:{row_number} must contain X, Y, Z, Angle.")
                try:
                    x_mm, y_mm, z_mm, angle_deg = (float(value) for value in row)
                except ValueError as exc:
                    raise ValueError(f"{path}:{row_number} contains a non-numeric value.") from exc
                positions.append([x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3])
                angles.append(angle_deg)
        if not positions:
            raise ValueError(f"{path} contains no magnets.")
        return positions, angles

    def build(self) -> MagnetConfiguration:
        """Return the source rings mirrored across the xy-plane."""
        position_parts = []
        angle_parts = []
        volume_parts = []
        ring_index_parts = []
        dimension_parts = []
        for ring_index, ring_file in enumerate(self._ring_files()):
            positions_list, angles_list = self._read_ring(ring_file)
            positive_positions = torch.tensor(positions_list, dtype=self.dtype, device=self.device)
            negative_positions = positive_positions * torch.tensor(
                [1.0, 1.0, -1.0], dtype=self.dtype, device=self.device
            )
            if self.use_halbach_orientations:
                angles = 2.0 * torch.atan2(positive_positions[:, 1], positive_positions[:, 0])
            else:
                angles = torch.deg2rad(torch.tensor(angles_list, dtype=self.dtype, device=self.device))
            magnet_count = positive_positions.shape[0]
            length = self.outer_magnet_length if ring_index == 8 else self.magnet_size
            position_parts.extend((negative_positions, positive_positions))
            angle_parts.extend((angles, angles))
            volume_parts.append(
                torch.full(
                    (2 * magnet_count,), self.magnet_size**2 * length, dtype=self.dtype, device=self.device
                )
            )
            ring_index_parts.append(
                torch.full((2 * magnet_count,), ring_index, dtype=torch.int64, device=self.device)
            )
            dimensions = torch.tensor(
                [self.magnet_size, self.magnet_size, length], dtype=self.dtype, device=self.device
            )
            dimension_parts.append(dimensions.repeat((2 * magnet_count, 1)))
        positions = torch.cat(position_parts)
        angles = torch.cat(angle_parts)
        orientations = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=1)
        return MagnetConfiguration(
            positions=positions,
            orientations=orientations,
            volumes=torch.cat(volume_parts),
            remanence=self.remanence,
            metadata={
                "geometry": "roma",
                "n_magnets": int(positions.shape[0]),
                "n_source_rings": 9,
                "n_rings": 18,
                "mirrored_about_z": True,
                "ring_directory": str(self.ring_directory),
                "ring_indices": torch.cat(ring_index_parts).detach().cpu(),
                "magnet_dimensions_m": torch.cat(dimension_parts).detach().cpu(),
                "magnet_size_m": self.magnet_size,
                "outer_magnet_length_m": self.outer_magnet_length,
                "use_halbach_orientations": self.use_halbach_orientations,
                "initial_magnet_angles_rad": angles.detach().cpu(),
            },
        )


def construct_roma(
    ring_directory: str | Path,
    remanence: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    magnet_size: float = 0.012,
    outer_magnet_length: float = 0.050,
    use_halbach_orientations: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ROMA positions, remanence vectors, volumes, and angles."""
    configuration = RomaGeometry(
        ring_directory,
        remanence,
        device,
        dtype,
        magnet_size,
        outer_magnet_length,
        use_halbach_orientations,
    ).build()
    angles = configuration.metadata["initial_magnet_angles_rad"].to(device=device, dtype=dtype)
    return configuration.positions, configuration.remanence_vectors(), configuration.volumes, angles
