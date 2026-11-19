# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file builds an open cap magnet lattice from its geometric rule.
#
#  cap_lattice.py
#  Marian Frei
#  24.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

import torch

from fieldsmith.configurations import MagnetConfiguration


def head_envelope_mm(
    a_bore_mm: float = 115.0,
    c_bore_mm: float = 105.0,
    r_out_mm: float = 144.6,
    margin_mm: float = 10.4,
    z_levels_mm: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0),
    r_min_mm: float = 11.0,
) -> tuple[tuple[float, float, float], ...]:
    """
    Envelope of a hemispherical shell around a head, as (z, r_lo, r_hi) rows in mm.

    Per plate height the outer bound is a hemisphere of radius r_out_mm. The
    inner bound is the head keep-out, a half ellipsoid with semi-axes a_bore_mm
    and c_bore_mm, widened by margin_mm. Above the keep-out the inner bound is
    r_min_mm. The margin is half the space diagonal of a 12 mm cube, so no cube
    corner can cross either bound whatever its orientation.
    """
    if not (0.0 < a_bore_mm < r_out_mm and c_bore_mm > 0.0 and margin_mm >= 0.0):
        raise ValueError("head envelope needs 0 < a_bore_mm < r_out_mm, positive c_bore_mm, non-negative margin.")
    rows = []
    for z in z_levels_mm:
        zc = z - margin_mm
        inside = 1.0 - (zc / c_bore_mm) ** 2 if zc > 0.0 else 1.0
        r_lo = a_bore_mm * math.sqrt(inside) + margin_mm if inside > 0.0 else r_min_mm
        outer = r_out_mm**2 - z**2
        if outer <= 0.0:
            continue
        r_hi = min(math.sqrt(outer), r_out_mm)
        # Round the inner bound up and the outer bound down, so rounding never crosses a bound.
        r_lo, r_hi = math.ceil(r_lo * 10.0) / 10.0, math.floor(r_hi * 10.0) / 10.0
        if r_hi >= r_lo:
            rows.append((float(z), r_lo, r_hi))
    if not rows:
        raise ValueError("head envelope produced no plates; check the bounds.")
    return tuple(rows)


class CapLatticeGeometry:
    """
    Lattice of identical cubes on flat slabs inside an envelope.

    Each slab holds concentric rings from the inner bound outward at a fixed
    radial pitch, and each ring holds round(2 pi r / pitch) magnets starting at
    azimuth zero. Orientations are placeholders along +z for a seeder to replace.
    """

    def __init__(
        self,
        envelope_mm: tuple[tuple[float, float, float], ...],
        ring_pitch_mm: float = 16.5,
        magnet_size: float = 0.012,
        remanence: float = 1.31,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if ring_pitch_mm <= 0.0 or magnet_size <= 0.0:
            raise ValueError("ring_pitch_mm and magnet_size must be positive.")
        if not envelope_mm:
            raise ValueError("envelope_mm must contain at least one slab.")
        self.envelope_mm = envelope_mm
        self.ring_pitch_mm = ring_pitch_mm
        self.magnet_size = magnet_size
        self.remanence = remanence
        self.device = device
        self.dtype = dtype

    def build(self) -> MagnetConfiguration:
        """Build the lattice with placeholder orientations along +z."""
        positions: list[list[float]] = []
        per_slab: list[int] = []
        for z_mm, r_lo_mm, r_hi_mm in self.envelope_mm:
            if r_hi_mm < r_lo_mm:
                raise ValueError(f"slab z={z_mm}: outer radius below inner radius.")
            count_before = len(positions)
            # The tolerance keeps an exact multiple of the pitch from losing a ring to rounding.
            n_rings = int(math.floor((r_hi_mm - r_lo_mm) / self.ring_pitch_mm + 1e-9)) + 1
            for ring in range(n_rings):
                radius_mm = r_lo_mm + ring * self.ring_pitch_mm
                count = max(1, int(round(2.0 * math.pi * radius_mm / self.ring_pitch_mm)))
                for index in range(count):
                    angle = 2.0 * math.pi * index / count
                    positions.append(
                        [radius_mm * math.cos(angle) * 1e-3,
                         radius_mm * math.sin(angle) * 1e-3,
                         z_mm * 1e-3]
                    )
            per_slab.append(len(positions) - count_before)

        pos = torch.tensor(positions, dtype=self.dtype, device=self.device)
        orientations = torch.zeros_like(pos)
        orientations[:, 2] = 1.0
        volumes = torch.full((pos.shape[0],), self.magnet_size**3, dtype=self.dtype, device=self.device)
        return MagnetConfiguration(
            positions=pos,
            orientations=orientations,
            volumes=volumes,
            remanence=self.remanence,
            metadata={
                "geometry": "cap_lattice",
                "n_magnets": int(pos.shape[0]),
                "magnets_per_slab": per_slab,
                "ring_pitch_mm": self.ring_pitch_mm,
                "magnet_size_m": self.magnet_size,
            },
        )
