# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Reusable samplers for field-of-view evaluation points.
#
#  point_sampler.py
#  Kostiantyn Lavronenko
#  16.07.2026
# -----------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from fieldsmith.geometry.halbach_ring import make_cartesian_points


class PointSampler(ABC):
    """Interface for objects that provide field evaluation points."""

    @abstractmethod
    def get_points(self) -> torch.Tensor:
        """Return evaluation points with shape (P, 3)."""

    @abstractmethod
    def get_grid(self) -> torch.Tensor:
        """Return the underlying Cartesian grid, when available."""


class CartesianGridSampler(PointSampler):
    """Lazily create a centered Cartesian field-of-view grid."""

    def __init__(
        self,
        fov: Sequence[float],
        resolution: Sequence[float],
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if len(fov) != 3 or len(resolution) != 3:
            raise ValueError("fov and resolution must each contain three values.")
        if any(size < 0.0 for size in fov):
            raise ValueError("fov values must be non-negative.")
        if any(step <= 0.0 for step in resolution):
            raise ValueError("resolution values must be positive.")
        self.fov: tuple[float, float, float] = tuple(float(value) for value in fov)
        self.resolution: tuple[float, float, float] = tuple(float(value) for value in resolution)
        self.device: torch.device | str = device
        self.dtype: torch.dtype = dtype
        self._points: torch.Tensor | None = None
        self._grid: torch.Tensor | None = None

    def _create_grid(self) -> None:
        self._points, self._grid = make_cartesian_points(
            fov=self.fov, resolution=self.resolution, device=self.device, dtype=self.dtype
        )

    def get_points(self) -> torch.Tensor:
        if self._points is None:
            self._create_grid()
        assert self._points is not None
        return self._points

    def get_grid(self) -> torch.Tensor:
        if self._grid is None:
            self._create_grid()
        assert self._grid is not None
        return self._grid


class MaskedPointSampler(PointSampler):
    """Select points from another sampler with a boolean mask function."""

    def __init__(self, base_sampler: PointSampler, mask_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.base_sampler: PointSampler = base_sampler
        self.mask_fn: Callable[[torch.Tensor], torch.Tensor] = mask_fn
        self._points: torch.Tensor | None = None

    def get_points(self) -> torch.Tensor:
        if self._points is None:
            points = self.base_sampler.get_points()
            mask = self.mask_fn(points)
            if mask.shape != points.shape[:-1] or mask.dtype != torch.bool:
                raise ValueError("mask_fn must return a boolean tensor with one value per point.")
            self._points = points[mask]
        return self._points

    def get_grid(self) -> torch.Tensor:
        return self.base_sampler.get_grid()

    def get_mask(self) -> torch.Tensor:
        grid = self.get_grid()
        mask = self.mask_fn(grid.reshape(-1, 3))
        if mask.shape != (grid.numel() // 3,) or mask.dtype != torch.bool:
            raise ValueError("mask_fn must return a boolean tensor with one value per point.")
        return mask.reshape(grid.shape[:-1])


class PredefinedPointSampler(PointSampler):
    """Use an existing tensor of arbitrary evaluation points."""

    def __init__(self, points: torch.Tensor) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (P, 3).")
        self.points: torch.Tensor = points

    def get_points(self) -> torch.Tensor:
        return self.points

    def get_grid(self) -> torch.Tensor:
        raise NotImplementedError("PredefinedPointSampler does not support get_grid().")


__all__ = ["CartesianGridSampler", "MaskedPointSampler", "PointSampler", "PredefinedPointSampler"]
