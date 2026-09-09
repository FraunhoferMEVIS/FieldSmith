from dataclasses import replace
from pathlib import Path

import pytest

from fieldsmith.configurations.halbach_system_config import HalbachSystemConfig


VALID_CONFIG = """\
[geometry]
n_magnets_per_ring = 42
n_rings = 7
n_rings_per_plane = 2
additional_magnets_per_ring = 7
ring_radius = 0.1
system_length = 0.2
magnet_size = 0.012
remanence = 1.3

[fov]
size_x = 0.16
size_y = 0.14
size_z = 0.12
resolution_x = 0.016
resolution_y = 0.014
resolution_z = 0.012
radius = 0.05
"""


@pytest.fixture
def valid_config() -> HalbachSystemConfig:
    return HalbachSystemConfig(
        n_magnets_per_ring=42,
        n_rings=7,
        n_rings_per_plane=2,
        additional_magnets_per_ring=7,
        ring_radius=0.1,
        system_length=0.2,
        magnet_size=0.012,
        remanence=1.3,
        fov=(0.16, 0.14, 0.12),
        resolution=(0.016, 0.014, 0.012),
        fov_radius=0.05,
    )


def test_from_file_loads_all_values(tmp_path: Path, valid_config: HalbachSystemConfig) -> None:
    config_path = tmp_path / "halbach.cfg"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    assert HalbachSystemConfig.from_file(config_path) == valid_config
    assert HalbachSystemConfig.from_file(str(config_path)) == valid_config


def test_from_file_rejects_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.cfg"

    with pytest.raises(FileNotFoundError, match="Halbach system configuration not found"):
        HalbachSystemConfig.from_file(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "n_magnets_per_ring = 42",
        VALID_CONFIG.replace("n_rings = 7\n", ""),
    ],
)
def test_from_file_wraps_parser_errors(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "invalid.cfg"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Halbach system configuration") as error:
        HalbachSystemConfig.from_file(config_path)

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_magnets_per_ring": 0}, "Magnet and ring counts must be positive"),
        ({"n_rings": 0}, "Magnet and ring counts must be positive"),
        ({"n_rings_per_plane": 0}, "Magnet and ring counts must be positive"),
        ({"additional_magnets_per_ring": -1}, "additional_magnets_per_ring must be non-negative"),
        ({"ring_radius": 0.0}, "Ring radius, magnet size, and remanence must be positive"),
        ({"magnet_size": 0.0}, "Ring radius, magnet size, and remanence must be positive"),
        ({"remanence": 0.0}, "Ring radius, magnet size, and remanence must be positive"),
        ({"system_length": -0.1}, "System length must be positive when multiple rings are configured"),
        ({"system_length": 0.0}, "System length must be positive when multiple rings are configured"),
        ({"fov": (-0.1, 0.1, 0.1)}, "FOV sizes must be non-negative"),
        ({"resolution": (0.1, 0.0, 0.1)}, "FOV resolutions must be positive"),
        ({"fov_radius": 0.0}, "FOV radius must be positive"),
    ],
)
def test_validate_rejects_invalid_values(
    valid_config: HalbachSystemConfig,
    changes: dict[str, object],
    message: str,
) -> None:
    configuration = replace(valid_config, **changes)

    with pytest.raises(ValueError, match=message):
        configuration._validate()


def test_validate_allows_zero_length_for_single_ring(valid_config: HalbachSystemConfig) -> None:
    configuration = replace(valid_config, n_rings=1, system_length=0.0)

    configuration._validate()


def test_as_log_config_returns_all_serializable_values(valid_config: HalbachSystemConfig) -> None:
    values = valid_config.as_log_config()

    assert values == {
        "n_magnets_per_ring": 42,
        "n_rings": 7,
        "n_rings_per_plane": 2,
        "additional_magnets_per_ring": 7,
        "ring_radius": 0.1,
        "system_length": 0.2,
        "magnet_size": 0.012,
        "remanence": 1.3,
        "fov": (0.16, 0.14, 0.12),
        "resolution": (0.016, 0.014, 0.012),
        "fov_radius": 0.05,
    }
