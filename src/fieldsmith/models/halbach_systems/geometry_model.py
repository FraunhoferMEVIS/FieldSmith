# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the angle and geometry model of stacked Halbach ring systems.
#
#  geometry_model.py
#  Kostiantyn Lavronenko
#  18.10.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


def _cluster(values: torch.Tensor, tolerance: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted cluster centres of 1D values and each value's cluster index."""
    order = torch.argsort(values)
    sorted_values = values[order]
    starts = torch.cat((sorted_values.new_ones(1, dtype=torch.bool),
                        torch.diff(sorted_values) > tolerance))
    labels_sorted = torch.cumsum(starts.long(), dim=0) - 1
    labels = torch.empty_like(labels_sorted)
    labels[order] = labels_sorted
    count = int(labels_sorted[-1]) + 1
    sums = values.new_zeros(count).index_add_(0, labels, values)
    counts = values.new_zeros(count).index_add_(0, labels, torch.ones_like(values))
    return sums / counts, labels


class StackedHalbachGeometryModel(nn.Module):
    """
    Optimize angles, concentric ring radii, and mirrored plane heights of a stacked system.

    The system is a stack of planes along z, mirror-symmetric about z = 0, each
    holding the same concentric Halbach rings -- the layout of
    ``HalbachSystemGeometry``. Magnets are assigned to a plane by their z level and
    to a concentric ring by their radius, so the model does not depend on magnet
    order. The free parameters are:

    - one in-plane magnetization angle per magnet;
    - one radius per concentric ring and mirrored plane pair (``radii_per_plane``),
      or one per concentric ring shared by all planes;
    - the heights of the planes above z = 0, mirrored below.

    The magnets cannot overlap by construction. Every radius and height is a lower
    bound plus a softplus slack, with bounds for cubes rotated in-plane:

    - adjacent centres in a ring are a chord of at least sqrt(2) * size apart;
    - each concentric ring lies at least sqrt(2) * size outside the previous one;
    - neighbouring planes are at least one size apart, so rings in different
      planes never touch, whatever their radii.

    Each bound carries an extra ``clearance_m``. The innermost radius also stays
    above ``min_inner_radius_m`` (the bore) and keeps every magnet body outside
    the field of view. A stack with an odd number of planes keeps its middle plane
    at z = 0.
    """

    def __init__(
        self,
        configuration: MagnetConfiguration,
        fov_radius: float,
        optimize_angles: bool = True,
        optimize_radii: bool = True,
        optimize_distances: bool = True,
        radii_per_plane: bool = True,
        clearance_m: float = 5.0e-4,
        slack_scale_m: float = 0.01,
        min_inner_radius_m: float = 0.0,
    ) -> None:
        super().__init__()
        positions = configuration.positions
        dtype = positions.dtype
        magnet_size = float(configuration.metadata["magnet_size_m"])
        self.magnet_size: float = magnet_size
        self.remanence: float = float(torch.as_tensor(configuration.remanence).reshape(-1)[0])
        self.radii_per_plane: bool = radii_per_plane
        self.slack_scale_m: float = slack_scale_m
        self.metadata: dict[str, object] = dict(configuration.metadata)

        radii = torch.linalg.norm(positions[:, :2], dim=1)
        if bool((radii <= 0.0).any()):
            raise ValueError("every magnet needs a non-zero radius.")
        ring_radii, ring_index = _cluster(radii, tolerance=0.25 * magnet_size)
        heights, height_index = _cluster(positions[:, 2].abs(), tolerance=0.25 * magnet_size)
        odd = bool(heights[0] < 0.25 * magnet_size)
        z_sign = torch.sign(positions[:, 2])
        for level in range(heights.numel()):
            in_level = height_index == level
            if odd and level == 0:
                continue
            if int((in_level & (z_sign > 0)).sum()) != int((in_level & (z_sign < 0)).sum()):
                raise ValueError("StackedHalbachGeometryModel expects planes mirrored about z = 0.")

        diagonal = math.sqrt(2.0) * magnet_size + clearance_m
        self.radial_gap_m: float = diagonal
        self.plane_gap_m: float = magnet_size + clearance_m
        first_plane = height_index == 0
        if not odd:
            first_plane = first_plane & (z_sign > 0)
        counts = [int((first_plane & (ring_index == ring)).sum()) for ring in range(ring_radii.numel())]
        if min(counts) < 2:
            raise ValueError("every concentric ring needs at least two magnets per plane.")
        bounds = [diagonal / (2.0 * math.sin(math.pi / count)) for count in counts]
        bounds[0] = max(bounds[0], fov_radius + 0.5 * diagonal, min_inner_radius_m)
        self.register_buffer("ring_radius_min", positions.new_tensor(bounds))
        self.register_buffer("ring_index", ring_index)
        self.register_buffer("level_index", height_index)
        self.register_buffer("z_sign", z_sign.to(dtype))
        self.register_buffer("unit_xy", positions[:, :2] / radii[:, None])
        self.odd_planes: bool = odd
        self.magnets_per_ring: list[int] = counts

        n_levels, n_rings = heights.numel(), ring_radii.numel()
        initial = torch.zeros((n_levels, n_rings), dtype=dtype, device=positions.device).index_put_(
            (height_index, ring_index), radii, accumulate=True,
        ) / torch.zeros((n_levels, n_rings), dtype=dtype, device=positions.device).index_put_(
            (height_index, ring_index), torch.ones_like(radii), accumulate=True,
        ).clamp_min(1.0)
        if not radii_per_plane:
            initial = initial.mean(dim=0, keepdim=True)
        floor = positions.new_tensor(1e-5)
        slack = torch.empty_like(initial)
        for ring in range(n_rings):
            bound = self.ring_radius_min[ring].expand(initial.shape[0])
            if ring > 0:
                bound = torch.maximum(initial[:, ring - 1] + self.radial_gap_m, bound)
            slack[:, ring] = torch.maximum(initial[:, ring] - bound, floor)
        self.radius_slack: nn.Parameter = nn.Parameter(
            _inverse_softplus(slack / slack_scale_m), requires_grad=optimize_radii,
        )

        free_heights = heights[1:] if odd else heights
        previous = torch.cat((heights.new_zeros(1), free_heights[:-1]))
        minimum_step = torch.full_like(free_heights, self.plane_gap_m)
        if not odd:
            minimum_step[0] = 0.5 * self.plane_gap_m
        self.register_buffer("minimum_height_step", minimum_step)
        self.plane_slack: nn.Parameter = nn.Parameter(
            _inverse_softplus(torch.maximum(free_heights - previous - minimum_step, floor) / slack_scale_m),
            requires_grad=optimize_distances,
        )

        phi = torch.atan2(configuration.orientations[:, 1], configuration.orientations[:, 0])
        theta = torch.acos(configuration.orientations[:, 2].clamp(-1.0, 1.0))
        self.magnet_angles_phi: nn.Parameter = nn.Parameter(phi.clone(), requires_grad=optimize_angles)
        self.register_buffer("magnet_angles_theta", theta.clone())
        self.register_buffer("volumes", configuration.volumes.clone())

    def ring_radii(self) -> torch.Tensor:
        """Radii in meters, shape (levels, rings), or (1, rings) when shared by all planes."""
        slack = self.slack_scale_m * torch.nn.functional.softplus(self.radius_slack)
        columns = []
        for ring in range(slack.shape[1]):
            bound = self.ring_radius_min[ring].expand(slack.shape[0])
            if ring > 0:
                bound = torch.maximum(columns[-1] + self.radial_gap_m, bound)
            columns.append(bound + slack[:, ring])
        return torch.stack(columns, dim=1)

    def plane_heights(self) -> torch.Tensor:
        """Heights of the plane levels in meters, nearest to z = 0 first (0 for a middle plane)."""
        steps = self.minimum_height_step + self.slack_scale_m * torch.nn.functional.softplus(self.plane_slack)
        heights = torch.cumsum(steps, dim=0)
        return torch.cat((heights.new_zeros(1), heights)) if self.odd_planes else heights

    def current_positions(self) -> torch.Tensor:
        radii = self.ring_radii()
        level = self.level_index if self.radii_per_plane else torch.zeros_like(self.level_index)
        radius = radii[level, self.ring_index]
        z = self.z_sign * self.plane_heights()[self.level_index]
        return torch.cat((self.unit_xy * radius[:, None], z[:, None]), dim=1)

    def orientations(self) -> torch.Tensor:
        sin_theta = torch.sin(self.magnet_angles_theta)
        return torch.stack((
            sin_theta * torch.cos(self.magnet_angles_phi),
            sin_theta * torch.sin(self.magnet_angles_phi),
            torch.cos(self.magnet_angles_theta),
        ), dim=1)

    def magnet_state(self) -> dict[str, torch.Tensor]:
        """Return the differentiable nominal magnets for a tolerance-aware wrapper."""
        orientations = self.orientations()
        return {
            "positions": self.current_positions(),
            "orientations": orientations,
            "remanence": orientations.new_full((orientations.shape[0],), self.remanence),
            "volumes": self.volumes,
        }

    def geometry_summary(self) -> dict[str, float]:
        """Current radii and plane heights in millimetres, for logging."""
        with torch.no_grad():
            summary = {}
            for level, row in enumerate(self.ring_radii()):
                for ring, radius in enumerate(row):
                    name = f"ring_{ring}_radius_mm" if not self.radii_per_plane else f"level_{level}_ring_{ring}_radius_mm"
                    summary[name] = float(radius) * 1e3
            for level, height in enumerate(self.plane_heights()):
                summary[f"level_{level}_z_mm"] = float(height) * 1e3
        return summary

    def get_configuration(self) -> MagnetConfiguration:
        orientations = self.orientations().detach()
        metadata = dict(self.metadata)
        metadata.update({
            "geometry": "halbach_system",
            "magnet_size_m": self.magnet_size,
            "magnet_angles_phi_rad": self.magnet_angles_phi.detach().cpu(),
            "magnet_angles_theta_rad": self.magnet_angles_theta.detach().cpu(),
            **self.geometry_summary(),
        })
        return MagnetConfiguration(
            positions=self.current_positions().detach(),
            orientations=orientations,
            volumes=self.volumes.detach(),
            remanence=self.remanence,
            metadata=metadata,
        )


__all__ = ["StackedHalbachGeometryModel"]
