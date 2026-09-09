"""Optimization models for FieldSmith systems."""

from fieldsmith.models.halbach_ring import SingleRingHalbachModel
from fieldsmith.models.halbach_systems import HalbachSystemAngleModel
from fieldsmith.models.small_halbach import (
    SmallHalbachAngleModel,
    SmallHalbachDistanceModel,
    SmallHalbachRadiusModel,
)

__all__ = [
    "HalbachSystemAngleModel",
    "SingleRingHalbachModel",
    "SmallHalbachAngleModel",
    "SmallHalbachDistanceModel",
    "SmallHalbachRadiusModel",
]
