# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# This file contains stable magnet configuration exchange objects.
#
#  magnet_configuration.py
#  Kostiantyn Lavronenko
#  01.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


CONFIGURATION_FORMAT_VERSION: str = "0.1.0"


@dataclass
class MagnetConfiguration:
    """Stable handoff format for one concrete magnet arrangement."""

    positions: torch.Tensor
    orientations: torch.Tensor
    volumes: torch.Tensor
    remanence: torch.Tensor | float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (M, 3).")
        if self.orientations.ndim != 2 or self.orientations.shape[1] != 3:
            raise ValueError("orientations must have shape (M, 3).")
        if self.positions.shape[0] != self.orientations.shape[0]:
            raise ValueError("positions and orientations must contain the same number of magnets.")
        if self.volumes.ndim != 1 or self.volumes.shape[0] != self.positions.shape[0]:
            raise ValueError("volumes must have shape (M,).")

    @property
    def num_magnets(self) -> int:
        """Return the number of magnets in this configuration."""
        return int(self.positions.shape[0])

    def remanence_vectors(self) -> torch.Tensor:
        """Return remanence vectors in tesla for all magnets."""
        if isinstance(self.remanence, torch.Tensor):
            remanence = self.remanence.to(device=self.orientations.device, dtype=self.orientations.dtype)
            if remanence.ndim == 0:
                return self.orientations * remanence
            if remanence.ndim == 1:
                return self.orientations * remanence[:, None]
            if remanence.shape == self.orientations.shape:
                return remanence
            raise ValueError("remanence tensor must be scalar, shape (M,), or shape (M, 3).")
        return self.orientations * float(self.remanence)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> MagnetConfiguration:
        """Return a copy with tensor fields moved to the requested device and dtype."""
        remanence = self.remanence
        if isinstance(remanence, torch.Tensor):
            remanence = remanence.to(device=device, dtype=dtype)
        return MagnetConfiguration(
            positions=self.positions.to(device=device, dtype=dtype),
            orientations=self.orientations.to(device=device, dtype=dtype),
            volumes=self.volumes.to(device=device, dtype=dtype),
            remanence=remanence,
            metadata=dict(self.metadata),
        )

    def detached_cpu(self) -> MagnetConfiguration:
        """Return a detached CPU copy suitable for logging or serialization."""
        remanence = self.remanence
        if isinstance(remanence, torch.Tensor):
            remanence = remanence.detach().cpu()
        return MagnetConfiguration(
            positions=self.positions.detach().cpu(),
            orientations=self.orientations.detach().cpu(),
            volumes=self.volumes.detach().cpu(),
            remanence=remanence,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> MagnetConfiguration:
        """Load a configuration checkpoint saved by ``WandbLogger``."""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Magnet configuration not found: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(data, cls):
            return data.to(device=device, dtype=dtype)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid magnet configuration in {checkpoint_path}: expected a dictionary.")

        format_version = data.get("format_version")
        if format_version is not None and format_version != CONFIGURATION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported magnet configuration format version {format_version!r} in "
                f"{checkpoint_path}; expected {CONFIGURATION_FORMAT_VERSION!r}."
            )

        required_keys = {
            "magnet_positions_m",
            "magnet_orientations",
            "volumes_m3",
            "remanence_t",
        }
        missing_keys = required_keys.difference(data)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Invalid magnet configuration in {checkpoint_path}: missing {missing}.")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid magnet configuration in {checkpoint_path}: metadata must be a dictionary.")
        configuration = cls(
            positions=data["magnet_positions_m"],
            orientations=data["magnet_orientations"],
            volumes=data["volumes_m3"],
            remanence=data["remanence_t"],
            metadata=metadata,
        )
        return configuration.to(device=device, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        config = self.detached_cpu()
        return {
            "format_version": CONFIGURATION_FORMAT_VERSION,
            "magnet_positions_m": config.positions,
            "magnet_orientations": config.orientations,
            "volumes_m3": config.volumes,
            "remanence_t": config.remanence,
            "metadata": config.metadata,
        }
