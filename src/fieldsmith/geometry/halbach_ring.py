# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains geometry helpers for Halbach ring optimization.
#
#  halbach_ring.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

from fieldsmith.configurations import MagnetConfiguration


class HalbachRingGeometry:
    """Build an ideal single-ring Halbach magnet configuration."""

    def __init__(
        self,
        n_magnets: int,
        ring_radius: float,
        magnet_size: float,
        remanence: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_magnets: int = n_magnets
        self.ring_radius: float = ring_radius
        self.magnet_size: float = magnet_size
        self.remanence: float = remanence
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype

    def build(self) -> MagnetConfiguration:
        """Build the initial Halbach ring configuration."""
        magnet_indices = torch.arange(self.n_magnets, dtype=self.dtype, device=self.device)
        position_angles = 2.0 * torch.pi * magnet_indices / self.n_magnets
        positions = torch.stack(
            (
                self.ring_radius * torch.cos(position_angles),
                self.ring_radius * torch.sin(position_angles),
                torch.zeros_like(position_angles),
            ),
            dim=1,
        )
        magnet_angles = 2.0 * position_angles
        orientations = torch.stack(
            (
                torch.cos(magnet_angles),
                torch.sin(magnet_angles),
                torch.zeros_like(magnet_angles),
            ),
            dim=1,
        )
        volumes = torch.full((self.n_magnets,), self.magnet_size**3, dtype=self.dtype, device=self.device)
        return MagnetConfiguration(
            positions=positions,
            orientations=orientations,
            volumes=volumes,
            remanence=self.remanence,
            metadata={
                "geometry": "halbach_ring",
                "n_magnets": self.n_magnets,
                "ring_radius_m": self.ring_radius,
                "magnet_size_m": self.magnet_size,
                "initial_magnet_angles_rad": magnet_angles.detach().cpu(),
            },
        )


def construct_halbach_ring(
    n_magnets: int,
    ring_radius: float,
    magnet_size: float,
    remanence: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a single ideal Halbach ring in the xy-plane.

    Returns:
        positions, remanence vectors, volumes, initial angles in radians.
    """
    configuration = HalbachRingGeometry(
        n_magnets=n_magnets,
        ring_radius=ring_radius,
        magnet_size=magnet_size,
        remanence=remanence,
        device=device,
        dtype=dtype,
    ).build()
    initial_angles = configuration.metadata["initial_magnet_angles_rad"].to(device=device, dtype=dtype)
    return configuration.positions, configuration.remanence_vectors(), configuration.volumes, initial_angles


def make_cartesian_points(
    fov: tuple[float, float, float],
    resolution: tuple[float, float, float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create centered Cartesian grid points.

    Returns:
        points with shape (P, 3), grid with shape (Nx, Ny, Nz, 3).
    """
    axes = [
        torch.linspace(-size / 2.0, size / 2.0, int(round(size / step)) + 1, dtype=dtype, device=device)
        for size, step in zip(fov, resolution)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(mesh, dim=-1)
    return grid.reshape(-1, 3), grid


def make_spherical_mask(points: torch.Tensor, radius: float) -> torch.Tensor:
    """Return a boolean mask for points inside a centered sphere."""
    return torch.linalg.norm(points, dim=-1) <= radius
