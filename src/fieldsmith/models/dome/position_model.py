# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains a model that optimizes magnet positions as well as angles.
#
#  position_model.py
#  Marian Frei
#  28.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Sequence

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

    def __init__(self, *args, position_bound_m: float | Sequence[float] = 2.0e-3, in_plane: bool = False,
                 base_positions: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One bound for all axes, or one per axis (x, y, z).
        bounds = (float(position_bound_m),) * 3 if isinstance(position_bound_m, (int, float)) \
            else tuple(float(v) for v in position_bound_m)
        if len(bounds) != 3 or min(bounds) <= 0.0:
            raise ValueError("position_bound_m must be a positive number or three positive numbers (x, y, z).")
        self.position_bound_m: float = max(bounds)
        # A single bound keeps scalar arithmetic, which the dome's reported run used.
        # Dividing by a scalar or by a tensor differs in the last bit on the GPU.
        self.scalar_bound: bool = isinstance(position_bound_m, (int, float))
        self.register_buffer("position_bound_axes", torch.tensor(bounds, dtype=self.magnet_positions.dtype))
        # In-plane displacement masks the z offset, so every magnet stays on its plate.
        self.in_plane: bool = in_plane
        # With base_positions the box stays around the original seats for a whole
        # chain of stages, and the loaded design becomes an offset inside it.
        loaded = self.magnet_positions.detach().clone()
        if base_positions is None:
            start = torch.zeros_like(loaded)
        else:
            base = torch.as_tensor(base_positions).to(loaded)
            if base.shape != loaded.shape:
                raise ValueError("base_positions must hold one centre per magnet.")
            axes = self.position_bound_axes.to(loaded)
            ratio = ((loaded - base) / axes).clamp(-0.999999, 0.999999)
            if ((loaded - base).abs() > axes).any():
                raise ValueError("the loaded design lies outside the bound around base_positions; widen the bound.")
            with torch.no_grad():
                self.magnet_positions.copy_(base)
            start = axes * torch.atanh(ratio)
        self.position_offsets: nn.Parameter = nn.Parameter(start)

    def effective_positions(self) -> torch.Tensor:
        """Return the magnet centres, displaced within the bound."""
        # The parent constructor evaluates the geometry before this subclass has
        # created its offset, so fall back to the base arrangement until it exists.
        offsets = getattr(self, "position_offsets", None)
        if offsets is None:
            return self.magnet_positions
        bound = self.position_bound_m if self.scalar_bound else self.position_bound_axes.to(offsets)
        displacement = bound * torch.tanh(offsets / bound)
        if self.in_plane:
            displacement = displacement * displacement.new_tensor((1.0, 1.0, 0.0))
        return self.magnet_positions + displacement

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
