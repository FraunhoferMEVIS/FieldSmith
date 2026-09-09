"""Geometry helpers for FieldSmith systems."""

from fieldsmith.geometry.halbach_ring import HalbachRingGeometry, construct_halbach_ring, make_cartesian_points, make_spherical_mask
from fieldsmith.geometry.halbach_system import HalbachSystemGeometry, construct_halbach_system
from fieldsmith.geometry.roma import RomaGeometry, construct_roma
from fieldsmith.geometry.dome import DomeGeometry

__all__ = [
    "construct_halbach_ring",
    "construct_halbach_system",
    "HalbachRingGeometry",
    "HalbachSystemGeometry",
    "RomaGeometry",
    "DomeGeometry"
    "construct_roma",
    "make_cartesian_points",
    "make_spherical_mask",
]
