# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the two-angle, demagnetization-aware dome optimization model.
#
#  model.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.demagnetization import apply_first_order_demag, build_demag_coupling
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.cube_axes import cube_axes_from_directions
from fieldsmith.models.halbach_ring.model import SingleRingHalbachModel
from fieldsmith.point_sampler import PointSampler


class DomeAngleModel(SingleRingHalbachModel):
    """Optimize azimuth and polar magnet angles, optionally with cuboid bodies and demagnetization."""

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
        field_model: str = "dipole",
        chi_par: float = 0.0,
        chi_perp: float = 0.0,
    ) -> None:
        super().__init__(
            configuration=configuration,
            device=device,
            point_sampler=point_sampler,
            dtype=dtype,
            chunk_points=chunk_points,
            fov=fov,
            resolution=resolution,
            fov_radius=fov_radius,
        )
        if field_model not in ("dipole", "cuboid"):
            raise ValueError('field_model must be "dipole" or "cuboid".')
        demagnetizing = chi_par != 0.0 or chi_perp != 0.0
        if (field_model == "cuboid" or demagnetizing) and self.magnet_size <= 0.0:
            raise ValueError('cuboid bodies need a positive "magnet_size_m" in the configuration metadata.')
        self.field_model: str = field_model
        self.chi_par: float = float(chi_par)
        self.chi_perp: float = float(chi_perp)
        self.demagnetizing: bool = demagnetizing

        directions = configuration.orientations.to(self.magnet_positions)
        norm = torch.linalg.norm(directions, dim=1).clamp_min(torch.finfo(directions.dtype).eps)
        self.magnet_polar_angles: nn.Parameter = nn.Parameter(torch.acos((directions[:, 2] / norm).clamp(-1.0, 1.0)))
        # Positions are frozen, so the mutual coupling is built once here and never rebuilt.
        positions = self.effective_positions()
        sizes = positions.new_full(positions.shape, self.magnet_size)
        self.register_buffer("demag_coupling", None)
        self.refresh_demagnetisation()

    def refresh_demagnetisation(self) -> None:
        """
        Rebuild the mutual coupling at the geometry the magnets are at now.

        The coupling depends on where the magnets sit, so it is exact only for
        the arrangement it was built from. Rebuilding it every step would
        dominate the cost, so it is held quasi-static: built once here, and
        rebuilt deliberately when the geometry has moved enough to matter. With
        the centres frozen, once is enough. With them free, a coupling left at
        the starting geometry means the optimiser is being steered by the
        demagnetisation of an array that no longer exists -- and the training
        loss falls the whole time it happens, so nothing signals the error.
        """
        if not self.demagnetizing:
            return
        with torch.no_grad():
            positions = self.effective_positions()
            sizes = positions.new_full(positions.shape, self.magnet_size)
            self.demag_coupling = build_demag_coupling(positions, sizes)

    def create_magnet_orientations(self) -> torch.Tensor:
        """Return full 3-D unit orientation vectors for the current polar and azimuth angles."""
        sin_polar = torch.sin(self.magnet_polar_angles)
        return torch.stack(
            (
                sin_polar * torch.cos(self.magnet_angles),
                sin_polar * torch.sin(self.magnet_angles),
                torch.cos(self.magnet_polar_angles),
            ),
            dim=1,
        )

    def create_magnet_vectors(self) -> torch.Tensor:
        """Return remanence vectors in tesla, corrected for mutual demagnetization when enabled."""
        vectors = self.create_magnet_orientations() * self.remanence
        if self.demag_coupling is None:
            return vectors
        return apply_first_order_demag(vectors, self.demag_coupling, self.chi_par, self.chi_perp)

    def body_axes(self) -> torch.Tensor:
        """
        Return the pocket frames of the current design orientations, without gradient.

        The frame follows the nominal orientation, not the demagnetization-corrected
        moment: the pocket is machined for the design angle. It is detached on purpose
        ("quasi-static axes"): the transverse axes are picked by a discrete argmin, so
        the frame jumps where two direction components cross in magnitude. Optimizing
        through that switch is ill-posed, while refreshing the frame every step -- as
        this does -- keeps it consistent with the design at all times.
        """
        with torch.no_grad():
            return cube_axes_from_directions(self.create_magnet_orientations())

    def forward(self) -> dict[str, torch.Tensor]:
        magnet_vectors = self.create_magnet_vectors()
        if self.field_model == "cuboid":
            positions = self.effective_positions()
            sizes = positions.new_full(positions.shape, self.magnet_size)
            field = calc_b_at_points_cuboid(
                magnet_br=magnet_vectors,
                magnet_size=sizes,
                magnet_pos=positions,
                points=self.points,
                axes=self.body_axes(),
                chunk_points=self.chunk_points,
            )
        else:
            field = calc_b_at_points_fast(
                dipole_br=magnet_vectors, vol=self.volumes, dipole_pos=self.effective_positions(),
                points=self.points, chunk_points=self.chunk_points,
            )
        return {"B": field, "points": self.points, "positions": self.points,
                "magnet_positions": self.effective_positions(), "magnet_vectors": magnet_vectors}

    def get_configuration(self) -> MagnetConfiguration:
        configuration = super().get_configuration()
        configuration.metadata.update({
            "geometry": self.configuration_metadata.get("geometry", "dome"),
            "magnet_polar_angles_rad": self.magnet_polar_angles.detach().cpu(),
            "remanence_vectors_t": self.create_magnet_vectors().detach().cpu(),
            "field_model": self.field_model, "chi_par": self.chi_par, "chi_perp": self.chi_perp,
        })
        return configuration
