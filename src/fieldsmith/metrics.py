# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains magnetic-field metrics and optimization losses.
#
#  metrics.py
#  Kostiantyn Lavronenko
#  22.06.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch


def component_field_strength(field_component: torch.Tensor) -> torch.Tensor:
    """Return the absolute mean of one signed field component."""
    return field_component.mean().abs()


def component_homogeneity_loss(
    field_component: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """Return dimensionless peak-to-peak homogeneity for one signed component."""
    mean_field = component_field_strength(field_component)
    return (field_component.max() - field_component.min()) / mean_field.clamp_min(eps)


def component_peak_to_peak_ppm(
    field_component: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """Return peak-to-peak homogeneity of one signed component in ppm."""
    return component_homogeneity_loss(field_component, eps) * 1e6


def relative_field_strength_shortfall_loss(
    field_component: torch.Tensor, reference_strength: torch.Tensor | float, eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize relative field loss below a reference, diverging as the field vanishes."""
    reference = torch.as_tensor(
        reference_strength, device=field_component.device, dtype=field_component.dtype,
    )
    if bool((reference <= 0.0).item()):
        raise ValueError("reference_strength must be positive.")
    strength = component_field_strength(field_component)
    shortfall = torch.clamp(reference / strength.clamp_min(eps) - 1.0, min=0.0)
    return shortfall * shortfall


def field_metrics(field: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Calculate field strength and homogeneity metrics."""
    field_abs = torch.linalg.norm(field, dim=-1)
    mean_field = field_abs.mean()
    homogeneity = (field_abs.max() - field_abs.min()) * 1e6 / (mean_field + eps)
    return {
        "fieldstrength_mT": mean_field * 1e3,
        "homogeneity_ppm": homogeneity,
        "field_x_mT": field[:, 0].mean().abs() * 1e3,
        "field_y_mT": field[:, 1].mean().abs() * 1e3,
        "field_z_mT": field[:, 2].mean().abs() * 1e3,
    }


def homogeneity_loss(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative peak-to-peak homogeneity loss on |B|."""
    field_abs = torch.linalg.norm(field, dim=-1)
    # field_abs = field[..., 0].abs()  # Use only the x-component for homogeneity loss
    return (field_abs.max() - field_abs.min()) / (field_abs.mean() + eps)


def field_strength_loss(field: torch.Tensor, expected_field: float) -> torch.Tensor:
    """Normalized mean-field target loss."""
    field_abs = torch.linalg.norm(field, dim=-1)
    return ((field_abs.mean() - expected_field) / expected_field) ** 2
