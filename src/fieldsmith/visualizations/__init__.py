"""Visualization helpers for FieldSmith systems."""

from fieldsmith.visualizations.matplotlib_2d import plot_angles, plot_field_slice_xy, plot_ring_magnets
from fieldsmith.visualizations.system_3d import (
    plot_magnets_and_field_3d,
    plot_magnets_and_field_3d_matplotlib,
    plot_model_output_3d,
)

__all__ = [
    "plot_angles",
    "plot_field_slice_xy",
    "plot_ring_magnets",
    "plot_magnets_and_field_3d",
    "plot_magnets_and_field_3d_matplotlib",
    "plot_model_output_3d",
]
