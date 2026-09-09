# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains stacked Halbach ring system optimization models.
#
#  model.py
#  Kostiantyn Lavronenko
#  16.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.cube_axes import cube_axes_from_directions
from fieldsmith.models.halbach_ring import SingleRingHalbachModel
from fieldsmith.point_sampler import PointSampler
from fieldsmith.visualizations import plot_model_output_3d


class HalbachSystemAngleModel(SingleRingHalbachModel):
    """Optimize in-plane magnet angles for a stacked Halbach ring system."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov: tuple[float, float, float],
        resolution: tuple[float, float, float],
        fov_radius: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        chunk_points: int | None = 65536,
        point_sampler: PointSampler | None = None,
        optimize_out_of_plane: bool = False,
        field_model: str = "dipole",
    ) -> None:
        if field_model not in {"dipole", "cuboid"}:
            raise ValueError("field_model must be either 'dipole' or 'cuboid'.")
        dimensions = configuration.metadata.get("magnet_dimensions_m")
        if dimensions is None:
            magnet_size = float(configuration.metadata.get("magnet_size_m", 0.0))
            if magnet_size > 0.0:
                dimensions = configuration.positions.new_full(configuration.positions.shape, magnet_size)
            else:
                cube_edges = configuration.volumes.pow(1.0 / 3.0)
                dimensions = cube_edges[:, None].expand_as(configuration.positions)
        initial_polar_angles = torch.acos(
            configuration.orientations[:, 2].to(device=device, dtype=dtype).clamp(-1.0, 1.0)
        )
        super().__init__(
            configuration=configuration,
            fov=fov,
            resolution=resolution,
            fov_radius=fov_radius,
            device=device,
            dtype=dtype,
            chunk_points=chunk_points,
            point_sampler=point_sampler,
        )
        self.title: str = "Halbach system"
        self.optimize_out_of_plane: bool = optimize_out_of_plane
        self.field_model: str = field_model
        self.register_buffer("magnet_dimensions", torch.as_tensor(dimensions, device=device, dtype=dtype))
        self.magnet_polar_angles: nn.Parameter = nn.Parameter(
            initial_polar_angles,
            requires_grad=optimize_out_of_plane,
        )

    def forward(self) -> dict[str, torch.Tensor]:
        """Evaluate the system with the selected magnetic-field model."""
        magnet_vectors = self.create_magnet_vectors()
        magnet_positions = self.effective_positions()
        if self.field_model == "cuboid":
            field = calc_b_at_points_cuboid(
                magnet_br=magnet_vectors,
                magnet_size=self.magnet_dimensions,
                magnet_pos=magnet_positions,
                points=self.points,
                # Is not applicable for the non-cube magnets like in the ROMA geometry
                # Comment that line when realizing with ROMA
                # axes=cube_axes_from_directions(magnet_vectors),
                chunk_points=self.chunk_points,
            )
        else:
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

    def create_magnet_orientations(self) -> torch.Tensor:
        """Return orientations parameterized by azimuthal and polar angles."""
        azimuthal_angles = self.expanded_magnet_angles()
        sin_polar = torch.sin(self.magnet_polar_angles)
        return torch.stack(
            (
                sin_polar * torch.cos(azimuthal_angles),
                sin_polar * torch.sin(azimuthal_angles),
                torch.cos(self.magnet_polar_angles),
            ),
            dim=1,
        )

    def get_images_to_log(self, output: dict[str, torch.Tensor] | None = None) -> dict[str, object]:
        if output is None:
            output = self.forward()
        return {
            "system_3d": plot_model_output_3d(
                output=output,
                cube_size=self.magnet_size,
                symmetry_type=None,
                title=self.title,
            )
        }

    def get_configuration(self) -> MagnetConfiguration:
        configuration = super().get_configuration()
        metadata = dict(configuration.metadata)
        metadata["geometry"] = "halbach_system"
        metadata["n_magnets"] = int(configuration.positions.shape[0])
        metadata["magnet_angles_rad"] = self.magnet_angles.detach().cpu()
        metadata["magnet_polar_angles_rad"] = self.magnet_polar_angles.detach().cpu()
        metadata["optimize_out_of_plane"] = self.optimize_out_of_plane
        metadata["field_model"] = self.field_model
        return MagnetConfiguration(
            positions=configuration.positions,
            orientations=configuration.orientations,
            volumes=configuration.volumes,
            remanence=configuration.remanence,
            metadata=metadata,
        )
