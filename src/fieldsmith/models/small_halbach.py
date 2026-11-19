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
        angle_dtype = configuration.positions.dtype
        self.magnet_angles_phi: nn.Parameter = nn.Parameter(
            torch.as_tensor(phi, device=self.points.device, dtype=angle_dtype).clone()
        )
        self.register_buffer(
            "magnet_angles_theta", torch.as_tensor(theta, device=self.points.device, dtype=angle_dtype).clone()
        )

    def orientations(self) -> torch.Tensor:
        return _orientations(self.magnet_angles_phi, self.magnet_angles_theta)

    def magnet_state(self) -> dict[str, torch.Tensor]:
        """Return the differentiable nominal magnets for a tolerance-aware wrapper."""
        orientations = self.orientations()
        return {
            "positions": self.magnet_positions,
            "orientations": orientations,
            "remanence": orientations.new_full((orientations.shape[0],), self.remanence),
            "volumes": orientations.new_full((orientations.shape[0],), self.magnet_size**3),
        }

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


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


class SmallHalbachGeometryModel(_SmallHalbachModel):
    """
    Optimize magnet angles, both ring radii, and the mirrored plane spacing together.

    Built for tolerance-aware optimization: the angles alone hardly change how
    strongly each magnet's manufacturing errors reach the field of view, the
    geometry does. Magnets are assigned to the inner or outer ring by their radius
    and to a plane by their z level, so the model does not depend on magnet order.

    The magnets cannot overlap by construction. Every radius and plane offset is a
    lower bound plus a softplus slack, with bounds for cubes rotated in-plane:

    - adjacent centres in a ring are a chord of at least sqrt(2) * size apart;
    - the outer ring lies at least sqrt(2) * size outside the inner one;
    - neighbouring planes are at least one size apart.

    Each bound carries an extra ``clearance_m``. The inner radius is also kept
    above ``min_inner_radius_m`` (the bore) and never lets a magnet body reach into
    the field of view. Plane 0 stays at z = 0 and the planes above and below
    mirror each other.
    """

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov: Sequence[float],
        resolution: Sequence[float],
        fov_radius: float,
        optimize_angles: bool = True,
        optimize_radii: bool = True,
        optimize_distances: bool = True,
        clearance_m: float = 5.0e-4,
        slack_scale_m: float = 0.01,
        min_inner_radius_m: float = 0.0,
    ) -> None:
        device = configuration.positions.device
        dtype = configuration.positions.dtype
        points = _stage_points(fov, resolution, fov_radius, "none", device).to(dtype)
        magnet_size = float(configuration.metadata["magnet_size_m"])
        super().__init__(points, magnet_size, float(configuration.remanence))
        positions = configuration.positions
        radii = torch.linalg.norm(positions[:, :2], dim=1)
        if bool((radii <= 0.0).any()):
            raise ValueError("every magnet needs a non-zero radius.")
        split = 0.5 * (radii.min() + radii.max())
        ring_outer = radii > split
        if bool(ring_outer.all()) or not bool(ring_outer.any()):
            raise ValueError("SmallHalbachGeometryModel expects an inner and an outer ring.")

        z_levels = torch.unique(torch.round(positions[:, 2], decimals=9), sorted=True)
        positive = z_levels[z_levels > 0.0]
        if not torch.allclose(torch.flip(-positive, dims=(0,)), z_levels[z_levels < 0.0], atol=1e-9) \
                or not bool((z_levels.abs() < 1e-9).any()):
            raise ValueError("SmallHalbachGeometryModel expects a plane at z = 0 and mirrored planes.")
        level = torch.argmin((positions[:, 2].abs()[:, None] - positive[None, :]).abs(), dim=1) + 1
        level = torch.where(positions[:, 2].abs() < 1e-9, torch.zeros_like(level), level)
        plane_index = level * torch.sign(positions[:, 2]).to(level.dtype)

        in_plane_zero = plane_index == 0
        count_inner = int((~ring_outer & in_plane_zero).sum())
        count_outer = int((ring_outer & in_plane_zero).sum())
        diagonal = 2.0**0.5 * magnet_size + clearance_m
        self.clearance_m: float = clearance_m
        self.slack_scale_m: float = slack_scale_m
        self.radial_gap_m: float = diagonal
        self.plane_gap_m: float = magnet_size + clearance_m
        self.inner_radius_min_m: float = max(
            diagonal / (2.0 * np.sin(np.pi / max(count_inner, 2))),
            fov_radius + 0.5 * diagonal,
            min_inner_radius_m,
        )
        self.outer_radius_min_m: float = diagonal / (2.0 * np.sin(np.pi / max(count_outer, 2)))

        self.register_buffer("unit_xy", positions[:, :2] / radii[:, None])
        self.register_buffer("ring_outer", ring_outer)
        self.register_buffer("plane_index", plane_index)

        inner_radius = radii[~ring_outer].mean()
        outer_radius = radii[ring_outer].mean()
        floor = positions.new_tensor(1e-5)
        inner_slack = torch.maximum(inner_radius - self.inner_radius_min_m, floor)
        outer_bound = torch.maximum(inner_radius + self.radial_gap_m, positions.new_tensor(self.outer_radius_min_m))
        outer_slack = torch.maximum(outer_radius - outer_bound, floor)
        self.radius_slack: nn.Parameter = nn.Parameter(
            _inverse_softplus(torch.stack((inner_slack, outer_slack)) / slack_scale_m),
            requires_grad=optimize_radii,
        )
        steps = torch.diff(torch.cat((positive.new_zeros(1), positive)))
        self.plane_slack: nn.Parameter = nn.Parameter(
            _inverse_softplus(torch.maximum(steps - self.plane_gap_m, floor) / slack_scale_m),
            requires_grad=optimize_distances,
        )

        phi = configuration.metadata.get("magnet_angles_phi_rad")
        theta = configuration.metadata.get("magnet_angles_theta_rad")
        if phi is None or theta is None:
            phi = torch.atan2(configuration.orientations[:, 1], configuration.orientations[:, 0])
            theta = torch.acos(configuration.orientations[:, 2].clamp(-1.0, 1.0))
        self.magnet_angles_phi: nn.Parameter = nn.Parameter(
            torch.as_tensor(phi, device=device, dtype=dtype).clone(), requires_grad=optimize_angles,
        )
        self.register_buffer("magnet_angles_theta", torch.as_tensor(theta, device=device, dtype=dtype).clone())

    def ring_radii(self) -> torch.Tensor:
        """Inner and outer ring radii in meters, shape (2,)."""
        slack = self.slack_scale_m * torch.nn.functional.softplus(self.radius_slack)
        inner = self.inner_radius_min_m + slack[0]
        outer_bound = torch.clamp(inner + self.radial_gap_m, min=self.outer_radius_min_m)
        return torch.stack((inner, outer_bound + slack[1]))

    def plane_levels(self) -> torch.Tensor:
        """Positive plane heights in meters, nearest plane first."""
        steps = self.plane_gap_m + self.slack_scale_m * torch.nn.functional.softplus(self.plane_slack)
        return torch.cumsum(steps, dim=0)

    def current_positions(self) -> torch.Tensor:
        inner, outer = self.ring_radii()
        radius = torch.where(self.ring_outer, outer, inner)
        levels = torch.cat((self.plane_levels().new_zeros(1), self.plane_levels()))
        z = torch.sign(self.plane_index).to(levels.dtype) * levels[self.plane_index.abs()]
        return torch.cat((self.unit_xy * radius[:, None], z[:, None]), dim=1)

    def orientations(self) -> torch.Tensor:
        return _orientations(self.magnet_angles_phi, self.magnet_angles_theta)

    def magnet_state(self) -> dict[str, torch.Tensor]:
        """Return the differentiable nominal magnets for a tolerance-aware wrapper."""
        orientations = self.orientations()
        return {
            "positions": self.current_positions(),
            "orientations": orientations,
            "remanence": orientations.new_full((orientations.shape[0],), self.remanence),
            "volumes": orientations.new_full((orientations.shape[0],), self.magnet_size**3),
        }

    def geometry_summary(self) -> dict[str, float]:
        """Current radii and plane heights in millimetres, for logging."""
        with torch.no_grad():
            inner, outer = self.ring_radii()
            summary = {"inner_radius_mm": float(inner) * 1e3, "outer_radius_mm": float(outer) * 1e3}
            for index, level in enumerate(self.plane_levels(), start=1):
                summary[f"plane_{index}_z_mm"] = float(level) * 1e3
        return summary

    def forward(self) -> dict[str, torch.Tensor]:
        positions = self.current_positions()
        field = calc_b_at_points(
            self.orientations() * self.remanence,
            self.magnet_size**3,
            positions,
            self.points,
            chunk_points=32768,
        )
        return {"B": field, "positions": positions}

    def get_configuration(self) -> MagnetConfiguration:
        orientations = self.orientations().detach()
        metadata = {
            "geometry": "small_halbach",
            "magnet_size_m": self.magnet_size,
            "magnet_angles_phi_rad": self.magnet_angles_phi.detach().cpu(),
            "magnet_angles_theta_rad": self.magnet_angles_theta.detach().cpu(),
        }
        metadata.update({name: value for name, value in self.geometry_summary().items()})
        return MagnetConfiguration(
            positions=self.current_positions().detach(),
            orientations=orientations,
            volumes=torch.full_like(orientations[:, 0], self.magnet_size**3),
            remanence=self.remanence,
            metadata=metadata,
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
    "SmallHalbachGeometryModel",
    "SmallHalbachRadiusModel",
    "build_small_halbach_configuration",
    "distance_stage_loss",
    "radius_stage_loss",
]
