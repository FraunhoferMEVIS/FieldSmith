# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains Monte Carlo evaluation of magnet manufacturing tolerances.
#
#  monte_carlo.py
#  Kostiantyn Lavronenko
#  13.08.2026
# -----------------------------------------------------------------------------

"""Monte Carlo evaluation of independent magnet manufacturing tolerances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import matplotlib.pyplot as plt
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation.core import evaluate_configuration, field_at_points


@dataclass
class MonteCarloResult:
    """Metric samples and the unperturbed reference values."""

    fieldstrength_mT: torch.Tensor
    peak_to_peak_ppm: torch.Tensor
    rms_ppm: torch.Tensor
    nominal: dict[str, float]

    def summary(self) -> dict[str, dict[str, float]]:
        """Return mean, standard deviation, extrema, and central 95% interval."""
        result = {}
        for name in ("fieldstrength_mT", "peak_to_peak_ppm", "rms_ppm"):
            values = getattr(self, name).to(dtype=torch.float64)
            result[name] = {
                "nominal": self.nominal[name],
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "p2.5": float(torch.quantile(values, 0.025)),
                "p97.5": float(torch.quantile(values, 0.975)),
                "max": float(values.max()),
            }
        return result


def _perturb_configuration(
    configuration: MagnetConfiguration,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    remanence_tolerance_t: float,
    generator: torch.Generator,
) -> MagnetConfiguration:
    positions = configuration.positions
    directions = configuration.remanence_vectors()
    magnitudes = torch.linalg.norm(directions, dim=1)
    directions = directions / magnitudes[:, None]

    position_errors = (2.0 * torch.rand(positions.shape, device=positions.device, dtype=positions.dtype,
                                        generator=generator) - 1.0) * position_tolerance_m
    random_axes = torch.randn(directions.shape, device=directions.device, dtype=directions.dtype,
                              generator=generator)
    random_axes = random_axes - (random_axes * directions).sum(dim=1, keepdim=True) * directions
    random_axes = random_axes / torch.linalg.norm(random_axes, dim=1, keepdim=True).clamp_min(1e-12)
    angle_errors = (2.0 * torch.rand((configuration.num_magnets,), device=positions.device,
                                     dtype=positions.dtype, generator=generator) - 1.0) * rotation_tolerance_rad
    perturbed_directions = (
        directions * torch.cos(angle_errors)[:, None]
        + random_axes * torch.sin(angle_errors)[:, None]
    )
    remanence_errors = (2.0 * torch.rand((configuration.num_magnets,), device=positions.device,
                                         dtype=positions.dtype, generator=generator) - 1.0) * remanence_tolerance_t
    return MagnetConfiguration(
        positions=positions + position_errors,
        orientations=perturbed_directions,
        volumes=configuration.volumes,
        remanence=magnitudes + remanence_errors,
        metadata=dict(configuration.metadata),
    )


def _measurement_metrics(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    field_model: str,
    chunk_points: int | None,
    probe_position_tolerance_m: float,
    probe_tilt_tolerance_rad: float,
    measurement_device_tolerance_t: float,
    generator: torch.Generator,
) -> dict[str, float]:
    """Evaluate scalar probe readings with independent bounded measurement errors."""
    sampled_points = points
    if probe_position_tolerance_m > 0.0:
        position_errors = (
            2.0 * torch.rand(
                points.shape, device=points.device, dtype=points.dtype, generator=generator,
            )
            - 1.0
        ) * probe_position_tolerance_m
        sampled_points = points + position_errors

    field = field_at_points(
        configuration, sampled_points, field_model, chunk_points=chunk_points,
    )
    readings = torch.linalg.norm(field, dim=-1)
    if probe_tilt_tolerance_rad > 0.0:
        tilt_errors = (
            2.0 * torch.rand(
                readings.shape, device=readings.device, dtype=readings.dtype,
                generator=generator,
            )
            - 1.0
        ) * probe_tilt_tolerance_rad
        readings = readings * torch.cos(tilt_errors)
    if measurement_device_tolerance_t > 0.0:
        device_errors = (
            2.0 * torch.rand(
                readings.shape, device=readings.device, dtype=readings.dtype,
                generator=generator,
            )
            - 1.0
        ) * measurement_device_tolerance_t
        readings = readings + device_errors

    readings = readings.abs()
    mean_reading = readings.mean()
    relative_deviation = (readings - mean_reading) / mean_reading.abs().clamp_min(1e-12)
    return {
        "fieldstrength_mT": float(mean_reading * 1e3),
        "peak_to_peak_ppm": float(
            (relative_deviation.max() - relative_deviation.min()) * 1e6
        ),
        "rms_ppm": float(torch.sqrt(torch.mean(relative_deviation**2)) * 1e6),
    }


def run_monte_carlo(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    num_samples: int = 100,
    position_tolerance_m: float = 1e-4,
    rotation_tolerance_deg: float = 1.0,
    remanence_tolerance_t: float = 0.005,
    field_model: str = "dipole",
    chunk_points: int | None = 4096,
    seed: int | None = None,
    progress_every: int = 10,
    progress_callback: Callable[[str], None] | None = print,
    probe_position_tolerance_m: float = 0.0,
    probe_tilt_tolerance_deg: float = 0.0,
    measurement_device_tolerance_t: float = 0.0,
) -> MonteCarloResult:
    """Evaluate independent manufacturing and measurement errors in float64.

    All tolerances describe uniform plus-or-minus bounds. Manufacturing errors are
    sampled independently per magnet. Measurement errors are sampled independently
    per probe point and Monte Carlo realization. Probe tilt models a scalar probe
    nominally aligned with the local field, so its reading is reduced by the cosine
    of the sampled angular error. Device uncertainty is additive in tesla.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    tolerances = (
        position_tolerance_m,
        rotation_tolerance_deg,
        remanence_tolerance_t,
        probe_position_tolerance_m,
        probe_tilt_tolerance_deg,
        measurement_device_tolerance_t,
    )
    if min(tolerances) < 0.0:
        raise ValueError("tolerances must be non-negative.")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
    configuration = configuration.to(device=configuration.positions.device, dtype=torch.float64)
    points = points.to(device=configuration.positions.device, dtype=torch.float64)
    magnitudes = torch.linalg.norm(configuration.remanence_vectors(), dim=1)
    if bool(torch.any(magnitudes <= remanence_tolerance_t)):
        raise ValueError("remanence_tolerance_t must be smaller than every magnet's remanence.")

    generator = torch.Generator(device=configuration.positions.device)
    if seed is not None:
        generator.manual_seed(seed)
    nominal = evaluate_configuration(configuration, points, field_model, chunk_points=chunk_points)
    metric_samples = {name: [] for name in nominal}
    start = time.perf_counter()
    rotation_tolerance_rad = torch.deg2rad(
        configuration.positions.new_tensor(rotation_tolerance_deg)
    ).item()
    probe_tilt_tolerance_rad = torch.deg2rad(
        configuration.positions.new_tensor(probe_tilt_tolerance_deg)
    ).item()
    has_measurement_uncertainty = any((
        probe_position_tolerance_m > 0.0,
        probe_tilt_tolerance_rad > 0.0,
        measurement_device_tolerance_t > 0.0,
    ))

    for sample_index in range(num_samples):
        perturbed = _perturb_configuration(
            configuration,
            position_tolerance_m,
            rotation_tolerance_rad,
            remanence_tolerance_t,
            generator,
        )
        if has_measurement_uncertainty:
            metrics = _measurement_metrics(
                perturbed,
                points,
                field_model,
                chunk_points,
                probe_position_tolerance_m,
                probe_tilt_tolerance_rad,
                measurement_device_tolerance_t,
                generator,
            )
        else:
            metrics = evaluate_configuration(
                perturbed, points, field_model, chunk_points=chunk_points,
            )
        for name, value in metrics.items():
            metric_samples[name].append(value)
        completed = sample_index + 1
        if progress_callback is not None and (completed == 1 or completed % progress_every == 0 or completed == num_samples):
            elapsed = time.perf_counter() - start
            remaining = elapsed / completed * (num_samples - completed)
            progress_callback(
                f"Monte Carlo {completed}/{num_samples} ({100.0 * completed / num_samples:.1f}%): "
                f"elapsed {elapsed:.1f}s, estimated remaining {remaining:.1f}s"
            )

    return MonteCarloResult(
        fieldstrength_mT=torch.tensor(metric_samples["fieldstrength_mT"], dtype=torch.float64),
        peak_to_peak_ppm=torch.tensor(metric_samples["peak_to_peak_ppm"], dtype=torch.float64),
        rms_ppm=torch.tensor(metric_samples["rms_ppm"], dtype=torch.float64),
        nominal=nominal,
    )


def plot_monte_carlo_distributions(
    result: MonteCarloResult,
    bins: int = 30,
) -> plt.Figure:
    """Plot distributions for field strength, peak-to-peak, and RMS homogeneity."""
    if bins <= 0:
        raise ValueError("bins must be positive.")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plots = (
        ("fieldstrength_mT", "Mean field strength [mT]"),
        ("peak_to_peak_ppm", "P-P homogeneity [ppm]"),
        ("rms_ppm", "RMS homogeneity [ppm]"),
    )
    for axis, (name, label) in zip(axes, plots):
        values = getattr(result, name).numpy()
        axis.hist(values, bins=bins, alpha=0.8, edgecolor="black")
        axis.axvline(result.nominal[name], color="red", linestyle="--", label="Nominal")
        axis.set_xlabel(label)
        axis.set_ylabel("Samples")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Monte Carlo magnet tolerance evaluation")
    figure.tight_layout()
    return figure
