"""Magnetic field simulation backends."""

from fieldsmith.field_simulations.dipole import calc_b_at_points, calc_b_at_points_fast
from fieldsmith.field_simulations.rectangular_prism import calc_b_at_points_rect_prism_from_cube_volume, calc_b_at_points_rect_prism_potential

__all__ = [
    "calc_b_at_points",
    "calc_b_at_points_fast",
    "calc_b_at_points_rect_prism_from_cube_volume",
    "calc_b_at_points_rect_prism_potential",
]
