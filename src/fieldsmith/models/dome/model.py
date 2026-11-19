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
from fieldsmith.geometry.cube_axes import cube_axes_from_directions, cube_axes_seated
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
        if field_model not in ("dipole", "cuboid", "mixed"):
            raise ValueError('field_model must be "dipole", "cuboid" or "mixed".')
        demagnetizing = chi_par != 0.0 or chi_perp != 0.0
        if (field_model in ("cuboid", "mixed") or demagnetizing) and self.magnet_size <= 0.0:
            raise ValueError('cuboid bodies need a positive "magnet_size_m" in the configuration metadata.')
        self.field_model: str = field_model
        # "argmin" turns each body with its moment, "world" keeps bodies as built.
        self.frame_rule: str = str(self.configuration_metadata.get("frame_rule", "argmin"))
        self.chi_par: float = float(chi_par)
        self.chi_perp: float = float(chi_perp)

        directions = configuration.orientations.to(self.magnet_positions)
        norm = torch.linalg.norm(directions, dim=1).clamp_min(torch.finfo(directions.dtype).eps)
        self.magnet_polar_angles: nn.Parameter = nn.Parameter(torch.acos((directions[:, 2] / norm).clamp(-1.0, 1.0)))
        # Bodies that are not cubes, such as ROMA's bars, may be locked in their
        # plane. The lock holds the polar angle each body arrived with.
        locked = self._locked_polar_mask()
        if locked is None:
            self.register_buffer("polar_locked", None)
            self.register_buffer("polar_locked_values", None)
        else:
            self.register_buffer("polar_locked", locked)
            self.register_buffer("polar_locked_values",
                                 self.magnet_polar_angles.detach().clone())
            print(f"polar angle locked on {int(locked.sum()):,} of {len(locked):,} bodies "
                  f"that are not cubes, free on the rest", flush=True)

        self.demagnetizing: bool = demagnetizing
        self.register_buffer("demag_coupling", None)
        self.refresh_demagnetisation()

    def _locked_polar_mask(self) -> torch.Tensor | None:
        """Which bodies may not leave their plane, or None when all may."""
        if not bool(self.configuration_metadata.get("lock_elongated_polar", False)):
            return None
        sizes = self.configuration_metadata.get("magnet_dimensions_m")
        if sizes is None:
            raise ValueError('locking the elongated bodies needs per-magnet "magnet_dimensions_m".')
        sizes = torch.as_tensor(sizes).to(self.magnet_positions)
        return (sizes.max(dim=1).values - sizes.min(dim=1).values).abs() > 1e-9

    def refresh_demagnetisation(self) -> None:
        """
        Rebuild the mutual coupling at the current geometry.

        The coupling is held fixed between rebuilds for cost. With moving
        positions it has to be rebuilt as the geometry changes.
        """
        if not self.demagnetizing:
            return
        with torch.no_grad():
            positions = self.effective_positions()
            sizes = self._body_sizes(positions)
            # Free the old coupling first, so a refresh never holds two.
            self.demag_coupling = None
            self.demag_coupling = build_demag_coupling(positions, sizes, axes=self.body_axes())

    def create_magnet_orientations(self) -> torch.Tensor:
        """Return full 3-D unit orientation vectors for the current polar and azimuth angles."""
        polar = self.magnet_polar_angles
        if self.polar_locked is not None:
            # A locked body gets no gradient on its polar angle.
            polar = torch.where(self.polar_locked, self.polar_locked_values, polar)
        sin_polar = torch.sin(polar)
        return torch.stack(
            (
                sin_polar * torch.cos(self.magnet_angles),
                sin_polar * torch.sin(self.magnet_angles),
                torch.cos(polar),
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
        if self.frame_rule == "world":
            return None
        with torch.no_grad():
            if self.frame_rule == "seated":
                # Cubes turn with their moments, bars only about the bore axis.
                return cube_axes_seated(self.create_magnet_orientations(),
                                        self._body_sizes(self.magnet_positions))
            return cube_axes_from_directions(self.create_magnet_orientations())


    def _body_sizes(self, positions: torch.Tensor) -> torch.Tensor:
        """Side lengths per magnet, from the arrangement when it carries them."""
        stored = self.configuration_metadata.get("magnet_dimensions_m")
        if stored is None:
            return positions.new_full(positions.shape, self.magnet_size)
        return stored.to(device=positions.device, dtype=positions.dtype)

    def forward(self) -> dict[str, torch.Tensor]:
        magnet_vectors = self.create_magnet_vectors()
        if self.field_model == "mixed":
            # Cuboids for bodies that are not cubes, dipoles for the cubes. On ROMA
            # this is within 1 ppm of the full cuboid model at a fraction of the cost.
            positions = self.effective_positions()
            sizes = self._body_sizes(positions)
            is_cube = (sizes.max(dim=1).values - sizes.min(dim=1).values).abs() < 1e-9
            axes = self.body_axes()
            field = calc_b_at_points_fast(
                dipole_br=magnet_vectors[is_cube], vol=self.volumes[is_cube],
                dipole_pos=positions[is_cube], points=self.points, chunk_points=self.chunk_points)
            if bool((~is_cube).any()):
                field = field + calc_b_at_points_cuboid(
                    magnet_br=magnet_vectors[~is_cube], magnet_size=sizes[~is_cube],
                    magnet_pos=positions[~is_cube], points=self.points,
                    axes=None if axes is None else axes[~is_cube], chunk_points=self.chunk_points)
        elif self.field_model == "cuboid":
            sizes = self._body_sizes(self.magnet_positions)
            field = calc_b_at_points_cuboid(
                magnet_br=magnet_vectors,
                magnet_size=sizes,
                magnet_pos=self.effective_positions(),
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
