"""Optimization models for FieldSmith systems."""

from fieldsmith.models.halbach_ring import SingleRingHalbachModel
from fieldsmith.models.halbach_systems import HalbachSystemAngleModel, StackedHalbachGeometryModel
from fieldsmith.models.small_halbach import (
    SmallHalbachAngleModel,
    SmallHalbachDistanceModel,
    SmallHalbachGeometryModel,
    SmallHalbachRadiusModel,
)
from fieldsmith.models.tolerance_aware import ToleranceAwareModel

__all__ = [
    "HalbachSystemAngleModel",
    "SingleRingHalbachModel",
    "SmallHalbachAngleModel",
    "SmallHalbachDistanceModel",
    "SmallHalbachGeometryModel",
    "SmallHalbachRadiusModel",
    "StackedHalbachGeometryModel",
    "ToleranceAwareModel",
]
