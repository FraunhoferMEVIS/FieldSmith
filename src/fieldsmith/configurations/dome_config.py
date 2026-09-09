# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Configuration loading for open cap (dome) magnet array experiments.
#
#  dome_config.py
#  Marian Frei
#  20.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomeConfig:
    """Geometry, physics and field-of-view settings for an open cap array."""

    aperture_radius: float
    dome_radius: float
    dome_height: float
    n_layers: int
    magnet_size: float
    remanence: float
    min_clearance: float
    field_model: str
    chi_par: float
    chi_perp: float
    fov: tuple[float, float, float]
    resolution: tuple[float, float, float]
    fov_radius: float
    fov_z_min: float
    seed_reg: float
    seed_iterations: int
    seed_mirror: str

    @classmethod
    def from_file(cls, path: str | Path) -> DomeConfig:
        """Load and validate a dome configuration file."""
        config_path = Path(path)
        parser = ConfigParser()
        try:
            if not parser.read(config_path):
                raise FileNotFoundError(f"Dome configuration not found: {config_path}")
            configuration = cls(
                aperture_radius=parser.getfloat("geometry", "aperture_radius"),
                dome_radius=parser.getfloat("geometry", "dome_radius"),
                dome_height=parser.getfloat("geometry", "dome_height"),
                n_layers=parser.getint("geometry", "n_layers"),
                magnet_size=parser.getfloat("geometry", "magnet_size"),
                remanence=parser.getfloat("geometry", "remanence"),
                min_clearance=parser.getfloat("geometry", "min_clearance"),
                field_model=parser.get("physics", "field_model", fallback="dipole"),
                chi_par=parser.getfloat("physics", "chi_par", fallback=0.0),
                chi_perp=parser.getfloat("physics", "chi_perp", fallback=0.0),
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
                fov_z_min=parser.getfloat("fov", "z_min", fallback=0.0),
                seed_reg=parser.getfloat("seed", "reg", fallback=1e-3),
                seed_iterations=parser.getint("seed", "iterations", fallback=40),
                seed_mirror=parser.get("seed", "mirror", fallback="none"),
            )
        except ConfigParserError as error:
            raise ValueError(f"Invalid dome configuration {config_path}: {error}") from error
        configuration._validate()
        return configuration

    def _validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.magnet_size <= 0.0 or self.remanence <= 0.0:
            raise ValueError("Magnet size and remanence must be positive.")
        if self.min_clearance < self.magnet_size:
            raise ValueError("min_clearance must be at least magnet_size.")
        if self.aperture_radius < 0.0 or self.dome_radius <= self.aperture_radius:
            raise ValueError("dome_radius must exceed a non-negative aperture_radius.")
        if self.dome_height <= 0.0:
            raise ValueError("dome_height must be positive.")
        if self.field_model not in ("dipole", "cuboid"):
            raise ValueError('field_model must be "dipole" or "cuboid".')
        if self.chi_par < 0.0 or self.chi_perp < 0.0:
            raise ValueError("Susceptibilities must be non-negative.")
        if any(size < 0.0 for size in self.fov):
            raise ValueError("FOV sizes must be non-negative.")
        if any(step <= 0.0 for step in self.resolution):
            raise ValueError("FOV resolutions must be positive.")
        if self.fov_radius <= 0.0:
            raise ValueError("FOV radius must be positive.")
        if self.seed_reg <= 0.0 or self.seed_iterations <= 0:
            raise ValueError("seed reg and iterations must be positive.")
        if self.seed_mirror not in ("none", "x", "y", "z"):
            raise ValueError('seed mirror must be "none", "x", "y" or "z".')

    def as_log_config(self) -> dict[str, int | float | str | tuple[float, float, float]]:
        """Return serializable values for experiment logging."""
        return asdict(self)
