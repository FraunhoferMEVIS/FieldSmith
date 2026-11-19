# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains the tolerance-aware wrapper around magnet optimization models.
#
#  tolerance_aware.py
#  Kostiantyn Lavronenko
#  18.10.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points_batched, calc_b_at_points_fast
from fieldsmith.metrics import field_metrics
from fieldsmith.objectives import batched_peak_to_peak_ppm, conditional_value_at_risk
from fieldsmith.perturbation import TolerancePerturbation
from fieldsmith.sensitivity import linearized_sensitivity_ppm


class ToleranceAwareModel(nn.Module):
    """
    Evaluate a nominal design together with N freshly drawn built children.

    The wrapped model owns the trainable parameters and describes its nominal
    magnets through ``magnet_state()``. Every forward pass draws new children, so
    one optimizer step on a loss over the children is one stochastic-gradient step
    on the expected loss over manufacturing errors; no outer resampling loop is
    needed. ``TrainLoop`` sees an ordinary model whose "B" is the nominal field.

    The children are evaluated with independent per-magnet errors, so their fields
    are not symmetric: evaluation points must cover the whole region even when the
    parent uses a symmetric parameterization.
    """

    def __init__(
        self,
        base_model: nn.Module,
        perturbation: TolerancePerturbation,
        points: torch.Tensor,
        chunk_samples: int | None = 16,
        cvar_level: float = 0.95,
        compute_sensitivity: bool = False,
    ) -> None:
        super().__init__()
        if not callable(getattr(base_model, "magnet_state", None)):
            raise TypeError("base_model must implement magnet_state().")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points must have shape (P, 3) with P > 0.")
        self.base_model: nn.Module = base_model
        self.perturbation: TolerancePerturbation = perturbation
        self.chunk_samples: int | None = chunk_samples
        self.cvar_level: float = cvar_level
        self.compute_sensitivity: bool = compute_sensitivity
        self.register_buffer("points", points)

    def forward(
        self,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return the nominal field and the children's fields and field strengths |B|.

        Args:
            num_samples: Number of children; defaults to the perturbation's.
            generator: Random generator, e.g. a freshly seeded one for a fixed
                held-out set; defaults to the perturbation's own.
        """
        state = self.base_model.magnet_state()
        positions = state["positions"]
        orientations = state["orientations"]
        remanence = state["remanence"]
        volumes = state["volumes"]
        nominal_field = calc_b_at_points_fast(
            dipole_br=orientations * remanence[:, None],
            vol=volumes,
            dipole_pos=positions,
            points=self.points,
            chunk_points=None,
        )

        num_samples = self.perturbation.num_samples if num_samples is None else num_samples
        child_positions, child_vectors = self.perturbation.sample_magnets(
            positions, orientations, remanence, num_samples=num_samples, generator=generator,
        )
        child_field = calc_b_at_points_batched(
            dipole_br=child_vectors,
            vol=volumes,
            dipole_pos=child_positions,
            points=self.points,
            chunk_samples=self.chunk_samples,
        )
        readings = torch.linalg.norm(child_field, dim=-1)
        output = {
            "B": nominal_field,
            "B_children": child_field,
            "child_readings": readings,
            "points": self.points,
            "magnet_positions": positions,
            "magnet_vectors": orientations * remanence[:, None],
        }
        if self.compute_sensitivity:
            # Linearized, sampling-free counterpart of the children's spread.
            output["sensitivity_ppm"] = linearized_sensitivity_ppm(
                positions, orientations, remanence, volumes, self.points, nominal_field,
                self.perturbation.position_tolerance_m,
                self.perturbation.rotation_tolerance_rad,
                self.perturbation.remanence_tolerance_t,
            )
        return output

    def child_statistics(self, readings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact peak-to-peak distribution summary of child readings (N, P), in ppm."""
        homogeneity = batched_peak_to_peak_ppm(readings)
        return {
            "children_mean_ppm": homogeneity.mean(),
            "children_std_ppm": homogeneity.std(unbiased=False),
            "children_rms_ppm": homogeneity.square().mean().sqrt(),
            "children_p95_ppm": torch.quantile(homogeneity, 0.95),
            "children_cvar_ppm": conditional_value_at_risk(homogeneity, self.cvar_level),
            "children_max_ppm": homogeneity.max(),
            "children_fieldstrength_mT": readings.mean() * 1e3,
        }

    def get_scalars_to_log(self, output: dict[str, torch.Tensor], loss: torch.Tensor) -> dict[str, torch.Tensor]:
        scalars = field_metrics(output["B"])
        scalars.update(self.child_statistics(output["child_readings"]))
        if "sensitivity_ppm" in output:
            scalars["sensitivity_ppm"] = output["sensitivity_ppm"].detach()
        scalars["loss"] = loss.detach()
        return scalars

    def get_images_to_log(self, output: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        return {}

    def constrain_parameters(self) -> None:
        constrain = getattr(self.base_model, "constrain_parameters", None)
        if constrain is not None:
            constrain()

    def get_configuration(self) -> MagnetConfiguration:
        return self.base_model.get_configuration()


__all__ = ["ToleranceAwareModel"]
