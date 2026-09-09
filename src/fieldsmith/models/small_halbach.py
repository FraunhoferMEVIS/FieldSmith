# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains stacked Halbach ring system optimization models.
#
#  small_halbach.py
#  Kostiantyn Lavronenko
#  25.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points
from fieldsmith.metrics import field_metrics


def _ring_numpy(num_magnets: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((0, 3))
    angles = np.empty((0, 1))
    for index in range(num_magnets):
        angle = index * 360.0 / num_magnets
        position = np.array([
            radius * np.cos(np.deg2rad(angle)),
            radius * np.sin(np.deg2rad(angle)),
            0.0,
        ]).reshape(1, 3)
        positions = np.concatenate((positions, position))
        angles = np.concatenate((angles, np.array([2.0 * angle]).reshape(1, 1)))
    return positions, angles.flatten()


def build_small_halbach_configuration(
    num_rings: int,
    num_magnets_per_ring: Sequence[int],
    ring_radius_inner: float,
    ring_radius_outer: float,
    distance_between_rings: float,
    magnet_size: float,
    remanence: float,
    device: torch.device | str,
) -> MagnetConfiguration:
    """Build the original two-radius, odd-ring small-Halbach arrangement."""
    if len(num_magnets_per_ring) != 2:
        raise ValueError("num_magnets_per_ring must contain inner and outer counts.")
    if num_rings % 2 != 1:
        raise ValueError("The matching small-Halbach workflow requires an odd number of rings.")

    position_parts = []
    angle_parts = []
    radii = (ring_radius_inner, ring_radius_outer)

    def append_plane(z_position: float) -> None:
        for count, radius in zip(num_magnets_per_ring, radii):
            positions, angles = _ring_numpy(int(count), float(radius))
            positions[:, 2] = z_position
            position_parts.append(positions)
            angle_parts.append(angles)

    append_plane(0.0)
    for ring_index in range(1, num_rings // 2 + 1):
        append_plane(-distance_between_rings * ring_index)
        append_plane(distance_between_rings * ring_index)

    positions = torch.tensor(np.concatenate(position_parts), dtype=torch.float32, device=device)
    phi = torch.deg2rad(torch.tensor(
        np.concatenate(angle_parts), dtype=torch.float32, device=device,
    ))
    theta = torch.full_like(phi, torch.pi / 2.0)
    orientations = _orientations(phi, theta)
    volumes = torch.full(
        (positions.shape[0],), magnet_size**3, dtype=torch.float32, device=device,
    )
    return MagnetConfiguration(
        positions=positions,
        orientations=orientations,
        volumes=volumes,
        remanence=remanence,
        metadata={
            "geometry": "small_halbach",
            "magnet_size_m": magnet_size,
            "magnet_angles_phi_rad": phi.detach().cpu(),
            "magnet_angles_theta_rad": theta.detach().cpu(),
        },
    )


def _orientations(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        torch.cos(phi) * torch.sin(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(theta),
    ), dim=1)


def _stage_points(
    fov: Sequence[float],
    resolution: Sequence[float],
    radius: float,
    symmetry: str,
    device: torch.device | str,
) -> torch.Tensor:
    # Preserve the reference sampler's unusual mixed precision exactly: FOV and
    # resolution are first rounded to float32, but linspace emits float64.
    fov_tensor = torch.tensor(fov, dtype=torch.float32, device=device)
    resolution_tensor = torch.tensor(resolution, dtype=torch.float32, device=device)
    axes = [
        torch.linspace(
            -fov_tensor[index] / 2.0,
            fov_tensor[index] / 2.0,
            torch.ceil(torch.round(fov_tensor[index] / resolution_tensor[index] + 1.0, decimals=2)).int(),
            dtype=torch.float64,
            device=device,
        )
        for index in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    grid[torch.abs(grid) < 1.0e-9] = 0.0
    points = grid.reshape(-1, 3)
    points = points[torch.linalg.norm(points, dim=-1) <= radius + 1.0e-3]
    if symmetry == "octant":
        return points[(points >= -1.0e-12).all(dim=-1)]
    if symmetry == "quadrant":
        return points[(points[:, :2] >= -1.0e-12).all(dim=-1)]
    if symmetry != "none":
        raise ValueError('symmetry must be "octant", "quadrant", or "none".')
    return points


class _SmallHalbachModel(nn.Module):
    def __init__(self, points: torch.Tensor, magnet_size: float, remanence: float) -> None:
        super().__init__()
        self.register_buffer("points", points)
        self.magnet_size: float = magnet_size
        self.remanence: float = remanence

    def get_scalars_to_log(
        self, output: dict[str, torch.Tensor], loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metrics = field_metrics(output["B"])
        metrics["loss"] = loss.detach()
        return metrics

    def get_images_to_log(self, output: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        return {}


class SmallHalbachDistanceModel(_SmallHalbachModel):
    """Optimize positive axial levels and mirror them across z=0."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov: Sequence[float],
        resolution: Sequence[float],
        fov_radius: float,
    ) -> None:
        points = _stage_points(fov, resolution, fov_radius, "octant", configuration.positions.device)
        magnet_size = float(configuration.metadata["magnet_size_m"])
        super().__init__(points, magnet_size, float(configuration.remanence))
        self.register_buffer("base_positions", configuration.positions.clone())
        self.register_buffer("orientations", configuration.orientations.clone())
        self.register_buffer(
            "phi", torch.as_tensor(configuration.metadata["magnet_angles_phi_rad"], device=configuration.positions.device),
        )
        self.register_buffer(
            "theta", torch.as_tensor(configuration.metadata["magnet_angles_theta_rad"], device=configuration.positions.device),
        )
        z_levels, z_ids = torch.unique(configuration.positions[:, 2], sorted=True, return_inverse=True)
        self.register_buffer("z_levels_initial", z_levels.clone())
        self.register_buffer("z_ids", z_ids)
        self.register_buffer("positive_mask", z_levels > 0.0)
        self.register_buffer("negative_mask", z_levels < 0.0)
        self.z_levels_trainable: nn.Parameter = nn.Parameter(z_levels[z_levels > 0.0].clone())

    def current_positions(self) -> torch.Tensor:
        z_levels = self.z_levels_initial.clone()
        z_levels[self.positive_mask] = self.z_levels_trainable
        z_levels[self.negative_mask] = torch.flip(-self.z_levels_trainable, dims=(0,))
        positions = self.base_positions.clone()
        positions[:, 2] = z_levels[self.z_ids]
        return positions

    def forward(self) -> dict[str, torch.Tensor]:
        positions = self.current_positions()
        field = calc_b_at_points(
            self.orientations * self.remanence,
            self.magnet_size**3,
            positions,
            self.points,
            chunk_points=32768,
        )
        return {"B": field, "magnet_positions_m": positions}

    def get_configuration(self) -> MagnetConfiguration:
        return MagnetConfiguration(
            positions=self.current_positions().detach(),
            orientations=self.orientations.detach(),
            volumes=torch.full_like(self.orientations[:, 0], self.magnet_size**3),
            remanence=self.remanence,
            metadata={
                "geometry": "small_halbach",
                "magnet_size_m": self.magnet_size,
                "magnet_angles_phi_rad": self.phi.detach().cpu(),
                "magnet_angles_theta_rad": self.theta.detach().cpu(),
            },
        )


class SmallHalbachRadiusModel(_SmallHalbachModel):
    """Apply the original bounded inner/outer radial scale parameterization."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov: Sequence[float],
        resolution: Sequence[float],
        fov_radius: float,
        num_rings: int,
        num_magnets_per_ring: Sequence[int],
        ring_radius_inner: float,
        ring_radius_outer: float,
    ) -> None:
        points = _stage_points(fov, resolution, fov_radius, "quadrant", configuration.positions.device)
        magnet_size = float(configuration.metadata["magnet_size_m"])
        super().__init__(points, magnet_size, float(configuration.remanence))
        self.register_buffer("base_positions", configuration.positions.clone())
        self.register_buffer("orientations", configuration.orientations.clone())
        self.register_buffer(
            "phi", torch.as_tensor(configuration.metadata["magnet_angles_phi_rad"], device=configuration.positions.device),
        )
        self.register_buffer(
            "theta", torch.as_tensor(configuration.metadata["magnet_angles_theta_rad"], device=configuration.positions.device),
        )
        self.inner_count: int = num_rings * int(num_magnets_per_ring[0])
        inner_count = torch.tensor(float(num_magnets_per_ring[0]), dtype=torch.float32, device=configuration.positions.device)
        outer_count = torch.tensor(float(num_magnets_per_ring[1]), dtype=torch.float32, device=configuration.positions.device)
        self.register_buffer("inner_radius", torch.tensor(ring_radius_inner, dtype=torch.float32, device=configuration.positions.device))
        self.register_buffer("outer_radius", torch.tensor(ring_radius_outer, dtype=torch.float32, device=configuration.positions.device))
        self.register_buffer(
            "inner_factor_min",
            (inner_count + 1.0) * magnet_size * np.sqrt(2.0) / (2.0 * torch.pi * self.inner_radius),
        )
        self.register_buffer(
            "outer_factor_min",
            (outer_count + 1.0) * magnet_size * np.sqrt(2.0) / (2.0 * torch.pi * self.outer_radius),
        )
        self.register_buffer("inner_factor_max", 0.18 / self.inner_radius)
        self.register_buffer("outer_factor_max", 0.18 / self.outer_radius)
        inner_initial = torch.logit(
            (1.0 - self.inner_factor_min) / (self.inner_factor_max - self.inner_factor_min)
        )
        outer_initial = torch.logit(
            (1.0 - self.outer_factor_min) / (self.outer_factor_max - self.outer_factor_min)
        )
        self.inner_factor: nn.Parameter = nn.Parameter(inner_initial.reshape(1))
        self.outer_factor: nn.Parameter = nn.Parameter(outer_initial.reshape(1))

    def regularized_inner_factor(self) -> torch.Tensor:
        return self.inner_factor_min + (self.inner_factor_max - self.inner_factor_min) * torch.sigmoid(self.inner_factor)

    def regularized_outer_factor(self) -> torch.Tensor:
        return self.outer_factor_min + (self.outer_factor_max - self.outer_factor_min) * torch.sigmoid(self.outer_factor)

    def ring_distance(self) -> torch.Tensor:
        return self.regularized_outer_factor() * self.outer_radius - self.regularized_inner_factor() * self.inner_radius

    def current_positions(self) -> torch.Tensor:
        factors = torch.cat((
            self.regularized_inner_factor().expand(self.inner_count),
            self.regularized_outer_factor().expand(self.base_positions.shape[0] - self.inner_count),
        ))
        multiplier = torch.stack((factors, factors, torch.ones_like(factors)), dim=1)
        return self.base_positions * multiplier

    def forward(self) -> dict[str, torch.Tensor]:
        positions = self.current_positions()
        field = calc_b_at_points(
            self.orientations * self.remanence,
            self.magnet_size**3,
            positions,
            self.points,
            chunk_points=32768,
        )
        return {"B": field, "ring_distance": self.ring_distance(), "positions": positions}

    def get_configuration(self) -> MagnetConfiguration:
        return MagnetConfiguration(
            positions=self.current_positions().detach(),
            orientations=self.orientations.detach(),
            volumes=torch.full_like(self.orientations[:, 0], self.magnet_size**3),
            remanence=self.remanence,
            metadata={
                "geometry": "small_halbach",
                "magnet_size_m": self.magnet_size,
                "magnet_angles_phi_rad": self.phi.detach().cpu(),
                "magnet_angles_theta_rad": self.theta.detach().cpu(),
            },
        )


class SmallHalbachAngleModel(_SmallHalbachModel):
    """Optimize the in-plane magnet angles of the radius-optimized system."""

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov: Sequence[float],
        resolution: Sequence[float],
        fov_radius: float,
    ) -> None:
        points = _stage_points(fov, resolution, fov_radius, "none", configuration.positions.device)
        magnet_size = float(configuration.metadata["magnet_size_m"])
        super().__init__(points, magnet_size, float(configuration.remanence))
        self.register_buffer("magnet_positions", configuration.positions.clone())
        phi = configuration.metadata.get("magnet_angles_phi_rad")
        theta = configuration.metadata.get("magnet_angles_theta_rad")
        if phi is None or theta is None:
            phi = torch.atan2(configuration.orientations[:, 1], configuration.orientations[:, 0])
            theta = torch.acos(configuration.orientations[:, 2].clamp(-1.0, 1.0))
        self.magnet_angles_phi: nn.Parameter = nn.Parameter(torch.as_tensor(phi, device=self.points.device).clone())
        self.register_buffer("magnet_angles_theta", torch.as_tensor(theta, device=self.points.device).clone())

    def orientations(self) -> torch.Tensor:
        return _orientations(self.magnet_angles_phi, self.magnet_angles_theta)

    def forward(self) -> dict[str, torch.Tensor]:
        field = calc_b_at_points(
            self.orientations() * self.remanence,
            self.magnet_size**3,
            self.magnet_positions,
            self.points,
            chunk_points=32768,
        )
        return {"B": field}

    def get_configuration(self) -> MagnetConfiguration:
        orientations = self.orientations().detach()
        return MagnetConfiguration(
            positions=self.magnet_positions.detach(),
            orientations=orientations,
            volumes=torch.full_like(orientations[:, 0], self.magnet_size**3),
            remanence=self.remanence,
            metadata={
                "geometry": "small_halbach",
                "magnet_size_m": self.magnet_size,
                "magnet_angles_phi_rad": self.magnet_angles_phi.detach().cpu(),
                "magnet_angles_theta_rad": self.magnet_angles_theta.detach().cpu(),
            },
        )


def component_x(field: torch.Tensor) -> torch.Tensor:
    """Return absolute Bx, matching the original small-Halbach losses."""
    return field[..., 0].abs()


def distance_stage_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    field = component_x(output["B"])
    relative = (field - field.mean()) / field.mean()
    variance = relative.square().sum()
    positions = output["magnet_positions_m"].reshape(-1, 3)
    distances = torch.cdist(positions, positions)
    pair_distances = distances[torch.tril(torch.ones_like(distances), diagonal=-1).bool()]
    spacing = 1.0e2 * torch.clamp(0.015 - pair_distances, min=0.0).square().sum()
    field_strength = 0.01 / field.mean()
    return variance + spacing + field_strength


def radius_stage_loss(
    output: dict[str, torch.Tensor], expected_field: float, magnet_size: float = 0.012,
) -> torch.Tensor:
    field = component_x(output["B"].to(torch.float32))
    loss = 100.0 * (field.max() - field.min()) + (field.mean() - expected_field).square()
    minimum_distance = magnet_size * np.sqrt(2.0)
    distance_penalty = 0.1 * torch.relu(minimum_distance - output["ring_distance"])
    return loss.unsqueeze(0) + distance_penalty.squeeze()


__all__ = [
    "SmallHalbachAngleModel",
    "SmallHalbachDistanceModel",
    "SmallHalbachRadiusModel",
    "build_small_halbach_configuration",
    "distance_stage_loss",
    "radius_stage_loss",
]
