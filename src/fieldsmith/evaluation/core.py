
# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains evaluation of a magnet configuration under a chosen physics.
#
#  evaluation.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.demagnetization import apply_first_order_demag, build_demag_coupling
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.collision import count_overlaps
from fieldsmith.geometry.cube_axes import (cube_axes_from_directions, cube_axes_min_z_extent,
                                            cube_axes_seated)
from fieldsmith.objectives import peak_to_peak_ppm, rms_deviation_ppm


def field_at_points(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    field_model: str = "cuboid",
    chi_par: float = 0.0,
    chi_perp: float = 0.0,
    chunk_points: int | None = 4096,
    frame_rule: str = "argmin",
) -> torch.Tensor:
    
    """
    Compute the field of a configuration under an explicitly chosen physics model.

    A design is always optimized under some assumption; this evaluates it under a
    possibly different one, which is how the cost of an assumption is measured. The
    interesting case is a design optimized for ideal magnets and evaluated with
    mutual demagnetization switched on.

    Args:
        configuration: The magnet arrangement to evaluate.
        points: Evaluation points in meters, shape (P, 3).
        field_model: "cuboid" for exact bodies, "dipole" for point sources.
        chi_par: Susceptibility along the easy axis; 0 disables demagnetization.
        chi_perp: Susceptibility across the easy axis.
        chunk_points: Number of evaluation points per chunk.
        frame_rule: Body-frame convention, "argmin" (the ring/dome default) or "gs"
            (the CAD Gram-Schmidt pocket rule). "gs" also couples oriented cube
            bodies in the demagnetization, which is what reproduces the cap reference.

    Returns:
        B-field in tesla, shape (P, 3).
    """
    if field_model not in ("dipole", "cuboid", "mixed"):
        raise ValueError('field_model must be "dipole", "cuboid" or "mixed".')
    if frame_rule not in ("argmin", "gs", "world", "seated"):
        raise ValueError('frame_rule must be "argmin", "gs", "world" or "seated".')
    points = points.to(device=configuration.positions.device, dtype=configuration.positions.dtype)
    magnet_size = configuration.metadata.get("magnet_size_m")
    vectors = configuration.remanence_vectors()
    demagnetizing = chi_par != 0.0 or chi_perp != 0.0
    sizes = None
    axes = None
    if field_model in ("cuboid", "mixed") or demagnetizing:
        if not magnet_size:
            raise ValueError('the configuration metadata needs a positive "magnet_size_m".')
        stored = configuration.metadata.get("magnet_dimensions_m")
        if stored is None:
            sizes = configuration.positions.new_full(configuration.positions.shape, float(magnet_size))
        else:
            # Mixed body shapes need each body's own side lengths.
            sizes = stored.to(device=configuration.positions.device,
                              dtype=configuration.positions.dtype)
        if frame_rule == "world":
            # Bodies keep their built orientation and only the magnetization turns.
            axes = None
        elif frame_rule == "seated":
            # Cubes turn with their moments, bars only about the bore axis.
            axes = cube_axes_seated(configuration.orientations, sizes)
        else:
            build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
            axes = build_axes(configuration.orientations)
    if demagnetizing:
        # The coupling is built in the frame the bodies actually have.
        coupling = build_demag_coupling(
            configuration.positions, sizes,
            axes=axes if frame_rule in ("gs", "seated") else None
        )
        vectors = apply_first_order_demag(vectors, coupling, chi_par, chi_perp)
    if field_model == "mixed":
        # Bars as cuboids, cubes as dipoles. Within 1 ppm of the full cuboid model on ROMA.
        is_cube = (sizes.max(dim=1).values - sizes.min(dim=1).values).abs() < 1e-9
        field = calc_b_at_points_fast(
            dipole_br=vectors[is_cube], vol=configuration.volumes[is_cube],
            dipole_pos=configuration.positions[is_cube], points=points, chunk_points=chunk_points)
        if bool((~is_cube).any()):
            field = field + calc_b_at_points_cuboid(
                magnet_br=vectors[~is_cube], magnet_size=sizes[~is_cube],
                magnet_pos=configuration.positions[~is_cube], points=points,
                axes=None if axes is None else axes[~is_cube], chunk_points=chunk_points)
        return field
    if field_model == "dipole":
        return calc_b_at_points_fast(
            dipole_br=vectors,
            vol=configuration.volumes,
            dipole_pos=configuration.positions,
            points=points,
            chunk_points=chunk_points,
        )
    return calc_b_at_points_cuboid(
        magnet_br=vectors,
        magnet_size=sizes,
        magnet_pos=configuration.positions,
        points=points,
        axes=axes,
        chunk_points=chunk_points,
    )


def evaluate_configuration(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    field_model: str = "cuboid",
    chi_par: float = 0.0,
    chi_perp: float = 0.0,
    chunk_points: int | None = 4096,
    frame_rule: str = "argmin",
) -> dict[str, float]:
    """
    Report mean field strength and homogeneity of a configuration under one physics.

    Returns:
        Mean field in mT, exact peak-to-peak in ppm and rms deviation in ppm.
    """
    with torch.no_grad():
        field = field_at_points(
            configuration, points, field_model, chi_par, chi_perp, chunk_points, frame_rule
        )
        return {
            "fieldstrength_mT": float(torch.linalg.norm(field, dim=-1).mean() * 1e3),
            "peak_to_peak_ppm": float(peak_to_peak_ppm(field)),
            "rms_ppm": float(rms_deviation_ppm(field)),
        }


def _unbuildable_tilt_deg(configuration: MagnetConfiguration, frame_rule: str) -> float:
    """
    The largest out-of-plane magnetization angle, in degrees, among bodies that are not cubes.

    A bar cannot be tilted out of its plane without moving material into its
    neighbour. Returns 0 when every body is a cube or the frame lets bodies tilt.
    """
    if frame_rule not in ("world", "seated"):
        return 0.0
    sizes = configuration.metadata.get("magnet_dimensions_m")
    if sizes is None:
        return 0.0
    sizes = sizes.to(configuration.orientations.device)
    elongated = (sizes.max(dim=1).values - sizes.min(dim=1).values).abs() > 1e-9
    if not bool(elongated.any()):
        return 0.0
    directions = configuration.orientations[elongated]
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    return float(torch.rad2deg(torch.asin(directions[:, 2].abs().clamp(max=1.0))).max())


def truth_eval_report(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    chi_par: float = 0.05,
    chi_perp: float = 0.10,
    chunk_points: int | None = 2048,
    frame_rule: str = "gs",
) -> dict[str, dict[str, float] | int]:
    """
    Score a configuration under both ideal and demagnetized physics.

    This is the cap "truth eval" line: the same two-number report as the reference
    engine's cap_truth_eval (mean |B| and exact peak-to-peak, once with no
    susceptibility and once with first-order mutual demagnetization), using the CAD
    pocket frame by default so a cap reference design reproduces its known number.

    Returns:
        {"chi0": {...}, "demag": {...}, "overlaps": n, "contacts": m} where each
        physics carries fieldstrength_mT / peak_to_peak_ppm / rms_ppm. Overlaps
        counts pairs intersecting by more than 0.4 mm at true body size, the
        acceptance gate. Contacts counts pairs touching at all, which merges
        their pockets in the printed plate.
    """
    build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
    magnet_size = configuration.metadata.get("magnet_size_m")
    with torch.no_grad():
        if magnet_size:
            if frame_rule == "world":
                # Bodies keep their built orientation, so overlaps are checked in the world frame.
                axes = torch.eye(3, dtype=configuration.positions.dtype,
                                 device=configuration.positions.device
                                 ).expand(configuration.positions.shape[0], 3, 3)
            elif frame_rule == "seated":
                stored = configuration.metadata.get("magnet_dimensions_m")
                sizes = (configuration.positions.new_full(configuration.positions.shape, float(magnet_size))
                         if stored is None else stored.to(configuration.positions))
                axes = cube_axes_seated(configuration.orientations, sizes)
            else:
                axes = build_axes(configuration.orientations)
            overlaps = count_overlaps(configuration.positions, axes, float(magnet_size))
            contacts = count_overlaps(configuration.positions, axes, float(magnet_size),
                                      tolerance=0.0)
        else:
            overlaps = contacts = -1
        tilt = _unbuildable_tilt_deg(configuration, frame_rule)
    return {
        "chi0": evaluate_configuration(configuration, points, "cuboid", 0.0, 0.0, chunk_points, frame_rule),
        "demag": evaluate_configuration(
            configuration, points, "cuboid", chi_par, chi_perp, chunk_points, frame_rule
        ),
        "overlaps": overlaps,
        "contacts": contacts,
        "tilt_deg": tilt,
    }
