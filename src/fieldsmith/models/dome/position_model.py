# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains a model that optimizes magnet positions as well as angles.
#
#  position_model.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn

from fieldsmith.models.dome.model import DomeAngleModel


class DomePositionModel(DomeAngleModel):
    """
    Optimize where the magnets sit as well as which way they point.

    Orientation alone reaches a floor: the arrangement it is given constrains
    what any set of angles can do. Letting the centres move lifts that floor,
    and on the reference cap it is the difference between 2,534 ppm and the
    flagship designs. The displacement stays small -- a few millimetres on a
    lattice of twelve-millimetre cubes -- so this is a refinement of an
    arrangement, not a search for one.

    The offset is bounded by construction rather than by a penalty:

        centre = base + bound * tanh(offset / bound)

    so no magnet can leave its neighbourhood however the optimiser pushes, the
    map is smooth everywhere, and it is the identity for small offsets, so the
    bound does not distort the gradient near the starting point. Optimising an
    unbounded offset and clamping afterwards would instead accumulate gradient
    against a wall.

    A bound applies per training segment, not per run: bake_positions() folds
    the achieved displacement into the base and re-centres the offset, so a long
    optimisation accumulates drift in bounded steps while the coupling and the
    collision pair list can be refreshed at the new geometry.
    """

    def __init__(self, *args, position_bound_m: float = 2.0e-3, **kwargs) -> None:
        if position_bound_m <= 0.0:
            raise ValueError("position_bound_m must be positive.")
        # Set before the parent runs: its constructor builds the demagnetisation
        # coupling, which reads the centres back through effective_positions(),
        # so the bound has to exist by then.
        self.position_bound_m: float = position_bound_m
        super().__init__(*args, **kwargs)
        self.position_offsets: nn.Parameter = nn.Parameter(torch.zeros_like(self.magnet_positions))

    def effective_positions(self) -> torch.Tensor:
        """Return the magnet centres, displaced within the bound."""
        bound = self.position_bound_m
        # The parent constructor builds the demagnetisation coupling, which reads
        # the centres back through here before this class has had a chance to
        # register its offsets. Until it has, the centres are simply the base.
        offsets = getattr(self, "position_offsets", None)
        if offsets is None:
            return self.magnet_positions
        return self.magnet_positions + bound * torch.tanh(offsets / bound)

    @torch.no_grad()
    def bake_positions(self) -> torch.Tensor:
        """
        Fold the achieved displacement into the base and re-centre the offset.

        Returns the displacement, in meters, that was folded in, so a caller can
        report how far the arrangement has travelled.
        """
        displaced = self.effective_positions()
        moved = displaced - self.magnet_positions
        self.magnet_positions.copy_(displaced)
        self.position_offsets.zero_()
        return moved

    def total_displacement(self, reference: torch.Tensor) -> torch.Tensor:
        """Per-magnet distance from a reference arrangement, in meters."""
        return (self.effective_positions() - reference.to(self.magnet_positions)).norm(dim=1)
