# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains differentiable manufacturing perturbations.
#
#  perturbation.py
#  Kostiantyn Lavronenko
#  18.10.2026
# -----------------------------------------------------------------------------

"""Differentiable Monte Carlo realizations of one nominal magnet arrangement."""

from __future__ import annotations

import math

import torch


class TolerancePerturbation:
    """
    Draw N built children of a nominal design, keeping the graph to the nominal one.

    The error model is the manufacturing part of ``run_monte_carlo``: every
    tolerance is a uniform plus-or-minus bound, sampled independently per magnet.
    The noise does not depend on the nominal design,
    so a child is the nominal design plus a fixed draw, and gradients of any child
    metric flow back to the nominal parameters unchanged (reparameterization).

    Antithetic sampling pairs every draw with its mirror image. The first-order
    effect of the errors then cancels exactly in the child mean, which lowers the
    variance of the mean-homogeneity gradient for the same number of children.
    """

    def __init__(
        self,
        num_samples: int,
        position_tolerance_m: float = 0.0,
        rotation_tolerance_deg: float = 0.0,
        remanence_tolerance_t: float = 0.0,
        antithetic: bool = False,
        seed: int | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if antithetic and num_samples % 2:
            raise ValueError("antithetic sampling requires an even num_samples.")
        if min(position_tolerance_m, rotation_tolerance_deg, remanence_tolerance_t) < 0.0:
            raise ValueError("tolerances must be non-negative.")
        self.num_samples: int = num_samples
        self.position_tolerance_m: float = position_tolerance_m
        self.rotation_tolerance_rad: float = math.radians(rotation_tolerance_deg)
        self.remanence_tolerance_t: float = remanence_tolerance_t
        self.antithetic: bool = antithetic
        self.seed: int | None = seed
        self._generator: torch.Generator | None = None

    def generator(self, device: torch.device) -> torch.Generator:
        """Return the internal generator, created on first use on the given device."""
        if self._generator is None or self._generator.device != torch.device(device):
            self._generator = torch.Generator(device=device)
            if self.seed is None:
                self._generator.seed()
            else:
                self._generator.manual_seed(self.seed)
        return self._generator

    def _uniform(self, num_samples: int, tail: tuple[int, ...], like: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
        """Uniform draws in [-1, 1], shape (N, *tail), mirrored when antithetic."""
        count = num_samples // 2 if self.antithetic else num_samples
        draws = 2.0 * torch.rand((count, *tail), device=like.device, dtype=like.dtype,
                                 generator=generator) - 1.0
        return torch.cat((draws, -draws)) if self.antithetic else draws

    def _normal(self, num_samples: int, tail: tuple[int, ...], like: torch.Tensor,
                generator: torch.Generator) -> torch.Tensor:
        """Standard normal draws, shape (N, *tail), repeated (not mirrored) when antithetic."""
        count = num_samples // 2 if self.antithetic else num_samples
        draws = torch.randn((count, *tail), device=like.device, dtype=like.dtype, generator=generator)
        return torch.cat((draws, draws)) if self.antithetic else draws

    def _resolve(self, like: torch.Tensor, num_samples: int | None,
                 generator: torch.Generator | None) -> tuple[int, torch.Generator]:
        num_samples = self.num_samples if num_samples is None else num_samples
        if self.antithetic and num_samples % 2:
            raise ValueError("antithetic sampling requires an even num_samples.")
        return num_samples, generator if generator is not None else self.generator(like.device)

    def sample_magnets(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        remanence: torch.Tensor,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perturb positions, magnetization directions, and remanence magnitudes.

        Args:
            positions: Nominal magnet centres in meters, shape (M, 3).
            orientations: Nominal unit magnetization directions, shape (M, 3).
            remanence: Nominal remanence magnitudes in tesla, shape (M,).
            num_samples: Number of children; defaults to ``self.num_samples``.
            generator: Random generator; defaults to the internal one.

        Returns:
            Child positions (N, M, 3) and child remanence vectors (N, M, 3), both
            differentiable with respect to the nominal inputs.
        """
        num_samples, generator = self._resolve(positions, num_samples, generator)
        n_magnets = positions.shape[0]
        child_positions = positions.expand(num_samples, -1, -1)
        if self.position_tolerance_m > 0.0:
            child_positions = child_positions + self.position_tolerance_m * self._uniform(
                num_samples, (n_magnets, 3), positions, generator,
            )

        directions = orientations / torch.linalg.norm(orientations, dim=-1, keepdim=True)
        child_directions = directions.expand(num_samples, -1, -1)
        if self.rotation_tolerance_rad > 0.0:
            # Tilt by a uniform angle towards a random direction perpendicular to the
            # nominal one -- the same construction as run_monte_carlo.
            tilt = self._normal(num_samples, (n_magnets, 3), positions, generator)
            tilt = tilt - (tilt * directions).sum(dim=-1, keepdim=True) * directions
            tilt = tilt / torch.linalg.norm(tilt, dim=-1, keepdim=True).clamp_min(1e-12)
            angles = self.rotation_tolerance_rad * self._uniform(
                num_samples, (n_magnets, 1), positions, generator,
            )
            child_directions = directions * torch.cos(angles) + tilt * torch.sin(angles)

        child_remanence = remanence.expand(num_samples, -1)
        if self.remanence_tolerance_t > 0.0:
            child_remanence = child_remanence + self.remanence_tolerance_t * self._uniform(
                num_samples, (n_magnets,), positions, generator,
            )
        return child_positions, child_directions * child_remanence[..., None]


__all__ = ["TolerancePerturbation"]
