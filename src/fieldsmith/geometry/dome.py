# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains geometry helpers for open cap magnet arrays on a dome shell.
#
#  dome.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

import torch

from fieldsmith.configurations import MagnetConfiguration




class DomeGeometry:
    """Build concentric rings of magnets on an ellipsoidal dome shell above z = 0."""

    def __init__(
        self,
        aperture_radius: float,
        dome_radius: float,
        dome_height: float,
        n_layers: int,
        magnet_size: float,
        remanence: float,
        min_clearance: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        target_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        if aperture_radius < 0.0 or dome_radius <= aperture_radius or dome_height <= 0.0:
            raise ValueError("dome_radius must exceed a non-negative aperture_radius, dome_height must be positive.")
        if n_layers <= 0 or magnet_size <= 0.0 or min_clearance < magnet_size:
            raise ValueError("n_layers and magnet_size must be positive, min_clearance at least magnet_size.")
        self.aperture_radius: float = aperture_radius
        self.dome_radius: float = dome_radius
        self.dome_height: float = dome_height
        self.n_layers: int = n_layers
        self.magnet_size: float = magnet_size
        self.remanence: float = remanence
        self.min_clearance: float = min_clearance
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype
        self.target_direction: tuple[float, float, float] = target_direction

    def build(self) -> MagnetConfiguration:
        """Build the initial open cap configuration with magic-sphere orientations."""
        # Meridian speed is bounded below by min(dome_radius, dome_height), so this gap clears layers.
        gap = max(self.min_clearance / min(self.dome_radius, self.dome_height), 0.5 * math.pi / (self.n_layers + 1))
        if self.n_layers * gap >= 0.5 * math.pi:
            raise ValueError("n_layers layers cannot be spaced by min_clearance on this dome.")
        polar_angles = [0.5 * math.pi - gap * (index + 1) for index in range(self.n_layers)]
        position_parts, magnets_per_layer = [], []
        for polar_angle in polar_angles:
            ring_radius = self.dome_radius * math.sin(polar_angle)
            if ring_radius < self.aperture_radius:
                magnets_per_layer.append(0)
                continue
            chord_ratio = min(1.0, self.min_clearance / (2.0 * ring_radius))
            count = 1 if chord_ratio >= 1.0 else int(math.pi / math.asin(chord_ratio))
            magnets_per_layer.append(count)
            azimuths = 2.0 * torch.pi * torch.arange(count, dtype=self.dtype, device=self.device) / count
            height = torch.full_like(azimuths, self.dome_height * math.cos(polar_angle))
            position_parts.append(
                torch.stack((ring_radius * torch.cos(azimuths), ring_radius * torch.sin(azimuths), height), dim=1)
            )
        if not position_parts:
            raise ValueError("dome geometry produced no magnets outside the aperture.")
        positions = torch.cat(position_parts)
        target = torch.tensor(self.target_direction, dtype=self.dtype, device=self.device)
        # Placeholders, and deliberately a geometric choice rather than a field
        # one: each magnet points out along the shell it sits on. That is the
        # attitude the ring spacing above assumes, so the arrangement it hands
        # back is collision-free as built. Which way the magnetisation should
        # actually face is left to lsq_seed_orientations, which solves it over
        # the positions and the region that exist rather than from a local rule.
        orientations = positions / positions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        volumes = torch.full((positions.shape[0],), self.magnet_size**3, dtype=self.dtype, device=self.device)
        return MagnetConfiguration(
            positions=positions,
            orientations=orientations,
            volumes=volumes,
            remanence=self.remanence,
            metadata={
                "geometry": "dome",
                "n_magnets": int(positions.shape[0]),
                "magnets_per_layer": magnets_per_layer,
                "layer_polar_angles_rad": polar_angles,
                "aperture_radius_m": self.aperture_radius,
                "dome_semi_axes_m": (self.dome_radius, self.dome_height),
                "magnet_size_m": self.magnet_size,
                "target_direction": self.target_direction,
            },
        )
