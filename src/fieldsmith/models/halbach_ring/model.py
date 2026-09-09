# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the single-ring Halbach optimization model.
#
#  model.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.halbach_ring import make_spherical_mask
from fieldsmith.metrics import field_metrics
from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler, PointSampler
from fieldsmith.visualizations import plot_model_output_3d


class SingleRingHalbachModel(nn.Module):
    """Optimize magnet angles for a provided single-ring Halbach configuration."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        device: torch.device | str,
        point_sampler: PointSampler | None = None,
        dtype: torch.dtype = torch.float32,
        chunk_points: int | None = 65536,
        fov: tuple[float, float, float] = (0.16, 0.16, 0.0),
        resolution: tuple[float, float, float] = (0.008, 0.008, 0.008),
        fov_radius: float = 0.0675,
        symmetric_optimization: bool = False,
    ) -> None:
        super().__init__()
        if configuration is None:
            raise ValueError("SingleRingHalbachModel requires a MagnetConfiguration.")

        configuration = configuration.to(device=device, dtype=dtype)
        if point_sampler is None:
            grid_sampler = CartesianGridSampler(fov=fov, resolution=resolution, device=device, dtype=dtype)
            point_sampler = MaskedPointSampler(grid_sampler, lambda points: make_spherical_mask(points, fov_radius))
        points = point_sampler.get_points()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("The point sampler must return points with shape (P, 3).")
        if points.shape[0] == 0:
            raise ValueError("The FOV, resolution, and fov_radius produced no evaluation points.")

        self.remanence: float = self._scalar_remanence(configuration.remanence)
        self.magnet_size: float = float(configuration.metadata.get("magnet_size_m", 0.0))
        self.chunk_points: int | None = chunk_points
        self.symmetric_optimization: bool = symmetric_optimization
        self.configuration_metadata: dict[str, object] = dict(configuration.metadata)

        initial_angles = torch.atan2(configuration.orientations[:, 1], configuration.orientations[:, 0])
        if symmetric_optimization:
            self._validate_opposite_pairs(configuration.positions)
            parameter_angles = initial_angles[: initial_angles.shape[0] // 2]
        else:
            parameter_angles = initial_angles

        self.register_buffer("magnet_positions", configuration.positions)
        self.register_buffer("volumes", configuration.volumes)
        self.register_buffer("points", points.to(device=device, dtype=dtype))
        self.register_buffer("initial_magnet_angles", initial_angles)
        self.magnet_angles: nn.Parameter = nn.Parameter(parameter_angles.clone())

    @staticmethod
    def _validate_opposite_pairs(positions: torch.Tensor) -> None:
        """Validate that the two configuration halves contain opposite magnets."""
        n_magnets = positions.shape[0]
        if n_magnets % 2 != 0:
            raise ValueError("Symmetric optimization requires an even number of magnets.")
        half = n_magnets // 2
        if not torch.allclose(positions[:half], -positions[half:], rtol=1e-4, atol=1e-6):
            raise ValueError(
                "Symmetric optimization requires magnet i + N/2 to be opposite magnet i."
            )

    def expanded_magnet_angles(self) -> torch.Tensor:
        """Return one current orientation angle for every physical magnet."""
        if not self.symmetric_optimization:
            return self.magnet_angles
        half = self.initial_magnet_angles.shape[0] // 2
        angle_updates = self.magnet_angles - self.initial_magnet_angles[:half]
        return torch.cat(
            (self.magnet_angles, self.initial_magnet_angles[half:] + angle_updates), dim=0
        )

    @staticmethod
    def _scalar_remanence(remanence: torch.Tensor | float) -> float:
        """Return a scalar remanence value for angle-only ring optimization."""
        if isinstance(remanence, torch.Tensor):
            if remanence.ndim == 0:
                return float(remanence.detach().cpu().item())
            if remanence.ndim == 1 and torch.allclose(remanence, remanence[0].expand_as(remanence)):
                return float(remanence[0].detach().cpu().item())
            raise ValueError("SingleRingHalbachModel expects scalar remanence.")
        return float(remanence)

    def create_magnet_orientations(self) -> torch.Tensor:
        """Return unit orientation vectors for current magnet angles."""
        magnet_angles = self.expanded_magnet_angles()
        return torch.stack(
            (
                torch.cos(magnet_angles),
                torch.sin(magnet_angles),
                torch.zeros_like(magnet_angles),
            ),
            dim=1,
        )

    def create_magnet_vectors(self) -> torch.Tensor:
        """Return remanence vectors for current magnet angles."""
        return self.create_magnet_orientations() * self.remanence

    def effective_positions(self) -> torch.Tensor:
        """
        Return the magnet centres the field is evaluated at.

        Here they are the fixed buffer. A subclass that lets positions move
        overrides this, and the field, the logged output and the exported
        configuration all follow without further change.
        """
        return self.magnet_positions

    def forward(self) -> dict[str, torch.Tensor]:
        magnet_vectors = self.create_magnet_vectors()
        magnet_positions = self.effective_positions()
        field = calc_b_at_points_fast(
            dipole_br=magnet_vectors,
            vol=self.volumes,
            dipole_pos=magnet_positions,
            points=self.points,
            chunk_points=self.chunk_points,
        )
        return {
            "B": field,
            "points": self.points,
            "positions": self.points,
            "magnet_positions": magnet_positions,
            "magnet_vectors": magnet_vectors,
        }

    def get_scalars_to_log(self, output: dict[str, torch.Tensor], loss: torch.Tensor) -> dict[str, torch.Tensor]:
        scalars = field_metrics(output["B"])
        scalars["loss"] = loss
        return scalars

    def get_images_to_log(self, output: dict[str, torch.Tensor] | None = None) -> dict[str, object]:
        if output is None:
            output = self.forward()
        return {
            "system_3d": plot_model_output_3d(
                output=output,
                cube_size=self.magnet_size,
                symmetry_type=None,
                title="Halbach ring",
            )
        }

    def get_configuration(self) -> MagnetConfiguration:
        metadata = dict(self.configuration_metadata)
        metadata.update(
            {
                "geometry": metadata.get("geometry", "halbach_ring"),
                "n_magnets": int(self.effective_positions().shape[0]),
                "magnet_angles_rad": self.magnet_angles.detach().cpu(),
                "symmetric_optimization": self.symmetric_optimization,
            }
        )
        return MagnetConfiguration(
            positions=self.effective_positions().detach().cpu(),
            orientations=self.create_magnet_orientations().detach().cpu(),
            volumes=self.volumes.detach().cpu(),
            remanence=self.remanence,
            metadata=metadata,
        )
