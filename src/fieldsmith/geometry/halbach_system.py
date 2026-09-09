# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains geometry helpers for stacked Halbach ring systems.
#
#  halbach_system.py
#  Kostiantyn Lavronenko
#  16.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

from fieldsmith.configurations import MagnetConfiguration


class HalbachSystemGeometry:
    """Build concentric Halbach rings in planes stacked along the z-axis."""

    def __init__(
        self,
        n_magnets_per_ring: int,
        n_rings: int,
        ring_radius: float,
        system_length: float,
        magnet_size: float,
        remanence: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        n_rings_per_plane: int = 1,
        additional_magnets_per_ring: int = 7,
    ) -> None:
        if n_magnets_per_ring <= 0:
            raise ValueError("n_magnets_per_ring must be positive.")
        if n_rings <= 0:
            raise ValueError("n_rings must be positive.")
        if n_rings_per_plane <= 0:
            raise ValueError("n_rings_per_plane must be positive.")
        if additional_magnets_per_ring < 0:
            raise ValueError("additional_magnets_per_ring must be non-negative.")
        if ring_radius <= 0.0:
            raise ValueError("ring_radius must be positive.")
        if system_length < 0.0:
            raise ValueError("system_length must be non-negative.")
        if n_rings > 1 and system_length <= 0.0:
            raise ValueError("system_length must be positive when n_rings > 1.")
        if magnet_size <= 0.0:
            raise ValueError("magnet_size must be positive.")

        self.n_magnets_per_ring: int = n_magnets_per_ring
        self.n_rings: int = n_rings
        self.n_rings_per_plane: int = n_rings_per_plane
        self.additional_magnets_per_ring: int = additional_magnets_per_ring
        self.ring_radius: float = ring_radius
        self.system_length: float = system_length
        self.magnet_size: float = magnet_size
        self.remanence: float = remanence
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype

    def build(self) -> MagnetConfiguration:
        """Build the initial stacked Halbach ring configuration."""
        ring_spacing = 1.5 * torch.sqrt(torch.tensor(3.0, dtype=self.dtype, device=self.device)) * self.magnet_size
        ring_radii = self.ring_radius + torch.arange(
            self.n_rings_per_plane,
            dtype=self.dtype,
            device=self.device,
        ) * ring_spacing
        magnets_per_concentric_ring = [
            self.n_magnets_per_ring + ring_index * self.additional_magnets_per_ring
            for ring_index in range(self.n_rings_per_plane)
        ]
        ring_position_parts = []
        ring_angle_parts = []
        for radius, magnet_count in zip(ring_radii, magnets_per_concentric_ring):
            magnet_indices = torch.arange(magnet_count, dtype=self.dtype, device=self.device)
            position_angles = 2.0 * torch.pi * magnet_indices / magnet_count
            ring_position_parts.append(
                torch.stack(
                    (
                        radius * torch.cos(position_angles),
                        radius * torch.sin(position_angles),
                    ),
                    dim=1,
                )
            )
            ring_angle_parts.append(2.0 * position_angles)
        ring_xy_positions = torch.cat(ring_position_parts)
        ring_magnet_angles = torch.cat(ring_angle_parts)
        magnets_per_plane = sum(magnets_per_concentric_ring)

        if self.n_rings == 1:
            ring_z_positions = torch.zeros((1,), dtype=self.dtype, device=self.device)
        else:
            ring_z_positions = torch.linspace(
                -self.system_length / 2.0,
                self.system_length / 2.0,
                self.n_rings,
                dtype=self.dtype,
                device=self.device,
            )

        positions = torch.cat(
            (
                ring_xy_positions.repeat((self.n_rings, 1)),
                ring_z_positions.repeat_interleave(magnets_per_plane)[:, None],
            ),
            dim=1,
        )

        magnet_angles = ring_magnet_angles.repeat(self.n_rings)
        orientations = torch.stack(
            (
                torch.cos(magnet_angles),
                torch.sin(magnet_angles),
                torch.zeros_like(magnet_angles),
            ),
            dim=1,
        )
        volumes = torch.full(
            (magnets_per_plane * self.n_rings,),
            self.magnet_size**3,
            dtype=self.dtype,
            device=self.device,
        )
        return MagnetConfiguration(
            positions=positions,
            orientations=orientations,
            volumes=volumes,
            remanence=self.remanence,
            metadata={
                "geometry": "halbach_system",
                "n_magnets": magnets_per_plane * self.n_rings,
                "n_magnets_per_ring": self.n_magnets_per_ring,
                "n_rings": self.n_rings,
                "n_rings_per_plane": self.n_rings_per_plane,
                "additional_magnets_per_ring": self.additional_magnets_per_ring,
                "magnets_per_concentric_ring": magnets_per_concentric_ring,
                "ring_radius_m": self.ring_radius,
                "ring_radii_m": ring_radii.detach().cpu(),
                "ring_spacing_m": float(ring_spacing.detach().cpu().item()),
                "system_length_m": self.system_length,
                "ring_z_positions_m": ring_z_positions.detach().cpu(),
                "magnet_size_m": self.magnet_size,
                "initial_magnet_angles_rad": magnet_angles.detach().cpu(),
            },
        )


def construct_halbach_system(
    n_magnets_per_ring: int,
    n_rings: int,
    ring_radius: float,
    system_length: float,
    magnet_size: float,
    remanence: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    n_rings_per_plane: int = 1,
    additional_magnets_per_ring: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an ideal stacked Halbach ring system.

    Returns:
        positions, remanence vectors, volumes, initial angles in radians.
    """
    configuration = HalbachSystemGeometry(
        n_magnets_per_ring=n_magnets_per_ring,
        n_rings=n_rings,
        ring_radius=ring_radius,
        system_length=system_length,
        magnet_size=magnet_size,
        remanence=remanence,
        device=device,
        dtype=dtype,
        n_rings_per_plane=n_rings_per_plane,
        additional_magnets_per_ring=additional_magnets_per_ring,
    ).build()
    initial_angles = configuration.metadata["initial_magnet_angles_rad"].to(device=device, dtype=dtype)
    return configuration.positions, configuration.remanence_vectors(), configuration.volumes, initial_angles
