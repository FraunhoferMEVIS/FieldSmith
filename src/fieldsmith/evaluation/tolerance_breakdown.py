# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the per-error-type breakdown of manufacturing tolerances.
#
#  tolerance_breakdown.py
#  Kostiantyn Lavronenko
#  22.10.2026
# -----------------------------------------------------------------------------

"""
Which manufacturing tolerance hurts a design how much.

Each tolerance -- magnet position, magnetization tilt, remanence -- is applied
alone, and all together, to the same design. For every case the Monte Carlo
distribution of the built peak-to-peak homogeneity is recorded, together with
the error floor (the peak-to-peak of the error field alone, i.e. what the
errors would cause on a perfectly homogeneous design) and the RMS of the
error, which is compared with the linearized, sampling-free prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.perturbation import TolerancePerturbation
from fieldsmith.sensitivity import linearized_error_variance

CASES = ("position", "rotation", "remanence", "all")


@dataclass
class ToleranceCaseResult:
    """Monte Carlo samples of one error case, in ppm of the nominal mean field."""

    case: str
    scale: float
    tolerances: dict[str, float]
    peak_to_peak_ppm: torch.Tensor
    floor_ppm: torch.Tensor
    error_rms_ppm: float
    linear_error_rms_ppm: float
    fieldstrength_mT: torch.Tensor

    def summary(self, unperturbed_ppm: float) -> dict[str, float]:
        values = self.peak_to_peak_ppm.to(torch.float64)
        floor = self.floor_ppm.to(torch.float64)
        mean = float(values.mean())
        return {
            "case": self.case,
            "scale": self.scale,
            **{f"{name}_tolerance": value for name, value in self.tolerances.items()},
            "mean_ppm": mean,
            "std_ppm": float(values.std(unbiased=False)),
            "p2.5_ppm": float(torch.quantile(values, 0.025)),
            "p97.5_ppm": float(torch.quantile(values, 0.975)),
            "iqr_ppm": float(torch.quantile(values, 0.75) - torch.quantile(values, 0.25)),
            "gap_mean_ppm": mean - unperturbed_ppm,
            "floor_mean_ppm": float(floor.mean()),
            "floor_std_ppm": float(floor.std(unbiased=False)),
            "error_rms_ppm": self.error_rms_ppm,
            "linear_error_rms_ppm": self.linear_error_rms_ppm,
            "fieldstrength_mean_mT": float(self.fieldstrength_mT.mean()),
        }


def case_tolerances(case: str, position_tolerance_m: float, rotation_tolerance_deg: float,
                    remanence_tolerance_t: float) -> tuple[float, float, float]:
    """The (position, rotation, remanence) tolerances that are active in a case."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}.")
    return (
        position_tolerance_m if case in ("position", "all") else 0.0,
        rotation_tolerance_deg if case in ("rotation", "all") else 0.0,
        remanence_tolerance_t if case in ("remanence", "all") else 0.0,
    )


def run_tolerance_case(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    case: str,
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
    remanence_tolerance_t: float,
    num_samples: int = 1000,
    seed: int = 0,
    scale: float = 1.0,
    chunk_points: int | None = 4096,
) -> ToleranceCaseResult:
    """
    Monte Carlo of one error case on one design, with the dipole field model.

    The tolerances are uniform plus-or-minus bounds, multiplied by ``scale``; only
    those belonging to ``case`` are active. The error floor of a child is the
    peak-to-peak of |B_child| - |B_nominal| relative to the nominal mean field.
    """
    configuration = configuration.to(device=configuration.positions.device, dtype=torch.float64)
    points = points.to(device=configuration.positions.device, dtype=torch.float64)
    position, rotation, remanence_tolerance = (
        scale * value for value in case_tolerances(case, position_tolerance_m, rotation_tolerance_deg,
                                                   remanence_tolerance_t)
    )
    vectors = configuration.remanence_vectors()
    remanence = torch.linalg.norm(vectors, dim=1)
    directions = vectors / remanence[:, None]
    with torch.no_grad():
        nominal_field = calc_b_at_points_fast(vectors, configuration.volumes, configuration.positions, points,
                                              chunk_points=chunk_points)
        nominal = torch.linalg.norm(nominal_field, dim=-1)
        nominal_mean = nominal.mean()
        perturbation = TolerancePerturbation(num_samples, position, rotation, remanence_tolerance, seed=seed)
        child_positions, child_vectors = perturbation.sample_magnets(configuration.positions, directions, remanence)
        peak_to_peak, floor, strength, error_square = [], [], [], []
        for index in range(num_samples):
            child = torch.linalg.norm(calc_b_at_points_fast(
                child_vectors[index], configuration.volumes, child_positions[index], points,
                chunk_points=chunk_points,
            ), dim=-1)
            child_mean = child.mean()
            peak_to_peak.append((child.max() - child.min()) / child_mean * 1e6)
            error = child - nominal
            floor.append((error.max() - error.min()) / nominal_mean * 1e6)
            error_square.append(((error - error.mean()) ** 2).mean())
            strength.append(child_mean * 1e3)
        linear_variance = linearized_error_variance(
            configuration.positions, directions, remanence, configuration.volumes, points, nominal_field,
            position, math.radians(rotation), remanence_tolerance,
        )
    return ToleranceCaseResult(
        case=case,
        scale=scale,
        tolerances={"position_mm": position * 1e3, "rotation_deg": rotation, "remanence_mT": remanence_tolerance * 1e3},
        peak_to_peak_ppm=torch.stack(peak_to_peak).cpu(),
        floor_ppm=torch.stack(floor).cpu(),
        error_rms_ppm=float(torch.stack(error_square).mean().sqrt() / nominal_mean * 1e6),
        linear_error_rms_ppm=float(linear_variance.mean().sqrt() / nominal_mean * 1e6),
        fieldstrength_mT=torch.stack(strength).cpu(),
    )


__all__ = ["CASES", "ToleranceCaseResult", "case_tolerances", "run_tolerance_case"]
