# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Configuration loading for comparable Halbach system experiments.
#
#  halbach_system_config.py
#  Kostiantyn Lavronenko
#  16.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HalbachSystemConfig:
    """Geometry and field-of-view settings shared by optimization methods."""

    n_magnets_per_ring: int
    n_rings: int
    n_rings_per_plane: int
    additional_magnets_per_ring: int
    ring_radius: float
    system_length: float
    magnet_size: float
    remanence: float
    fov: tuple[float, float, float]
    resolution: tuple[float, float, float]
    fov_radius: float

    @classmethod
    def from_file(cls, path: str | Path) -> HalbachSystemConfig:
        """Load and validate a Halbach system configuration file."""
        config_path = Path(path)
        parser = ConfigParser()
        try:
            loaded_files = parser.read(config_path)
            if not loaded_files:
                raise FileNotFoundError(f"Halbach system configuration not found: {config_path}")
            configuration = cls(
                n_magnets_per_ring=parser.getint("geometry", "n_magnets_per_ring"),
                n_rings=parser.getint("geometry", "n_rings"),
                n_rings_per_plane=parser.getint("geometry", "n_rings_per_plane"),
                additional_magnets_per_ring=parser.getint("geometry", "additional_magnets_per_ring"),
                ring_radius=parser.getfloat("geometry", "ring_radius"),
                system_length=parser.getfloat("geometry", "system_length"),
                magnet_size=parser.getfloat("geometry", "magnet_size"),
                remanence=parser.getfloat("geometry", "remanence"),
                fov=(
                    parser.getfloat("fov", "size_x"),
                    parser.getfloat("fov", "size_y"),
                    parser.getfloat("fov", "size_z"),
                ),
                resolution=(
                    parser.getfloat("fov", "resolution_x"),
                    parser.getfloat("fov", "resolution_y"),
                    parser.getfloat("fov", "resolution_z"),
                ),
                fov_radius=parser.getfloat("fov", "radius"),
            )
        except ConfigParserError as error:
            raise ValueError(f"Invalid Halbach system configuration {config_path}: {error}") from error
        configuration._validate()
        return configuration

    def _validate(self) -> None:
        if self.n_magnets_per_ring <= 0 or self.n_rings <= 0 or self.n_rings_per_plane <= 0:
            raise ValueError("Magnet and ring counts must be positive.")
        if self.additional_magnets_per_ring < 0:
            raise ValueError("additional_magnets_per_ring must be non-negative.")
        if self.ring_radius <= 0.0 or self.magnet_size <= 0.0 or self.remanence <= 0.0:
            raise ValueError("Ring radius, magnet size, and remanence must be positive.")
        if self.system_length < 0.0 or (self.n_rings > 1 and self.system_length == 0.0):
            raise ValueError("System length must be positive when multiple rings are configured.")
        if any(size < 0.0 for size in self.fov):
            raise ValueError("FOV sizes must be non-negative.")
        if any(step <= 0.0 for step in self.resolution):
            raise ValueError("FOV resolutions must be positive.")
        if self.fov_radius <= 0.0:
            raise ValueError("FOV radius must be positive.")

    def as_log_config(self) -> dict[str, int | float | tuple[float, float, float]]:
        """Return serializable values for experiment logging."""
        return asdict(self)
