
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
from fieldsmith.geometry.cube_axes import cube_axes_from_directions, cube_axes_min_z_extent
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
    if field_model not in ("dipole", "cuboid"):
        raise ValueError('field_model must be "dipole" or "cuboid".')
    if frame_rule not in ("argmin", "gs"):
        raise ValueError('frame_rule must be "argmin" or "gs".')
    points = points.to(device=configuration.positions.device, dtype=configuration.positions.dtype)
    magnet_size = configuration.metadata.get("magnet_size_m")
    vectors = configuration.remanence_vectors()
    demagnetizing = chi_par != 0.0 or chi_perp != 0.0
    sizes = None
    axes = None
    if field_model == "cuboid" or demagnetizing:
        if not magnet_size:
            raise ValueError('the configuration metadata needs a positive "magnet_size_m".')
        sizes = configuration.positions.new_full(configuration.positions.shape, float(magnet_size))
        build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
        axes = build_axes(configuration.orientations)
    if demagnetizing:
        # gs couples the oriented cube bodies; argmin keeps the world-aligned
        # coupling the ring and dome models are built on.
        coupling = build_demag_coupling(
            configuration.positions, sizes, axes=axes if frame_rule == "gs" else None
        )
        vectors = apply_first_order_demag(vectors, coupling, chi_par, chi_perp)
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
        {"chi0": {...}, "demag": {...}, "overlaps": n} where each physics carries
        fieldstrength_mT / peak_to_peak_ppm / rms_ppm, and overlaps is the number
        of magnet pairs intersecting by more than 0.4 mm at true body size.
        A design is only a result if that count is zero.
    """
    build_axes = cube_axes_min_z_extent if frame_rule == "gs" else cube_axes_from_directions
    magnet_size = configuration.metadata.get("magnet_size_m")
    with torch.no_grad():
        overlaps = (
            count_overlaps(
                configuration.positions,
                build_axes(configuration.orientations),
                float(magnet_size),
            )
            if magnet_size
            else -1
        )
    return {
        "chi0": evaluate_configuration(configuration, points, "cuboid", 0.0, 0.0, chunk_points, frame_rule),
        "demag": evaluate_configuration(
            configuration, points, "cuboid", chi_par, chi_perp, chunk_points, frame_rule
        ),
        "overlaps": overlaps,
    }
