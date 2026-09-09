from dataclasses import replace
from pathlib import Path

import pytest

from fieldsmith.configurations.dome_config import DomeConfig


VALID_CONFIG = """\
[geometry]
aperture_radius = 0.06
dome_radius = 0.17
dome_height = 0.19
n_layers = 8
magnet_size = 0.012
remanence = 1.3
min_clearance = 0.0135

[seed]
reg = 0.002
iterations = 50
mirror = y

[physics]
field_model = cuboid
chi_par = 0.05
chi_perp = 0.10

[fov]
size_x = 0.12
size_y = 0.10
size_z = 0.08
resolution_x = 0.008
resolution_y = 0.01
resolution_z = 0.02
radius = 0.05
z_min = 0.012
"""


MINIMAL_CONFIG = """\
[geometry]
aperture_radius = 0.06
dome_radius = 0.17
dome_height = 0.19
n_layers = 8
magnet_size = 0.012
remanence = 1.3
min_clearance = 0.0135

[fov]
size_x = 0.12
size_y = 0.10
size_z = 0.08
resolution_x = 0.008
resolution_y = 0.01
resolution_z = 0.02
radius = 0.05
"""


@pytest.fixture
def valid_config() -> DomeConfig:
    return DomeConfig(
        aperture_radius=0.06,
        dome_radius=0.17,
        dome_height=0.19,
        n_layers=8,
        magnet_size=0.012,
        remanence=1.3,
        min_clearance=0.0135,
        field_model="cuboid",
        chi_par=0.05,
        chi_perp=0.10,
        fov=(0.12, 0.10, 0.08),
        resolution=(0.008, 0.01, 0.02),
        fov_radius=0.05,
        fov_z_min=0.012,
        seed_reg=0.002,
        seed_iterations=50,
        seed_mirror="y",
    )


def test_from_file_loads_all_values(
    tmp_path: Path,
    valid_config: DomeConfig,
) -> None:
    config_path = tmp_path / "dome.cfg"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    assert DomeConfig.from_file(config_path) == valid_config
    assert DomeConfig.from_file(str(config_path)) == valid_config


def test_from_file_applies_optional_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal.cfg"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")

    configuration = DomeConfig.from_file(config_path)

    assert configuration.field_model == "dipole"
    assert configuration.chi_par == pytest.approx(0.0)
    assert configuration.chi_perp == pytest.approx(0.0)
    assert configuration.fov_z_min == pytest.approx(0.0)
    assert configuration.seed_reg == pytest.approx(1e-3)
    assert configuration.seed_iterations == 40
    assert configuration.seed_mirror == "none"


def test_from_file_rejects_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.cfg"

    with pytest.raises(FileNotFoundError, match="Dome configuration not found"):
        DomeConfig.from_file(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "aperture_radius = 0.06",
        VALID_CONFIG.replace("[geometry]\n", "[geometry\n"),
        VALID_CONFIG.replace("dome_radius = 0.17\n", ""),
    ],
)
def test_from_file_wraps_parser_errors(
    tmp_path: Path,
    contents: str,
) -> None:
    config_path = tmp_path / "invalid.cfg"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid dome configuration") as error:
        DomeConfig.from_file(config_path)

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"magnet_size": 0.0}, "Magnet size and remanence must be positive"),
        ({"remanence": 0.0}, "Magnet size and remanence must be positive"),
        ({"min_clearance": 0.011}, "min_clearance must be at least magnet_size"),
        ({"aperture_radius": -0.01}, "dome_radius must exceed a non-negative aperture_radius"),
        ({"dome_radius": 0.06}, "dome_radius must exceed a non-negative aperture_radius"),
        ({"dome_height": 0.0}, "dome_height must be positive"),
        ({"field_model": "sphere"}, 'field_model must be "dipole" or "cuboid"'),
        ({"chi_par": -0.01}, "Susceptibilities must be non-negative"),
        ({"chi_perp": -0.01}, "Susceptibilities must be non-negative"),
        ({"fov": (0.12, -0.1, 0.08)}, "FOV sizes must be non-negative"),
        ({"resolution": (0.008, 0.0, 0.02)}, "FOV resolutions must be positive"),
        ({"fov_radius": 0.0}, "FOV radius must be positive"),
        ({"seed_reg": 0.0}, "seed reg and iterations must be positive"),
        ({"seed_iterations": 0}, "seed reg and iterations must be positive"),
        ({"seed_mirror": "xy"}, 'seed mirror must be "none", "x", "y" or "z"'),
    ],
)
def test_validate_rejects_invalid_values(
    valid_config: DomeConfig,
    changes: dict[str, object],
    message: str,
) -> None:
    configuration = replace(valid_config, **changes)

    with pytest.raises(ValueError, match=message):
        configuration._validate()


@pytest.mark.parametrize("field_model", ["dipole", "cuboid"])
@pytest.mark.parametrize("seed_mirror", ["none", "x", "y", "z"])
def test_validate_accepts_supported_physics_and_mirror_options(
    valid_config: DomeConfig,
    field_model: str,
    seed_mirror: str,
) -> None:
    configuration = replace(
        valid_config,
        field_model=field_model,
        seed_mirror=seed_mirror,
    )

    configuration._validate()


def test_validate_allows_zero_fov_size(
    valid_config: DomeConfig,
) -> None:
    configuration = replace(valid_config, fov=(0.12, 0.10, 0.0))

    configuration._validate()


def test_as_log_config_returns_all_serializable_values(
    valid_config: DomeConfig,
) -> None:
    assert valid_config.as_log_config() == {
        "aperture_radius": 0.06,
        "dome_radius": 0.17,
        "dome_height": 0.19,
        "n_layers": 8,
        "magnet_size": 0.012,
        "remanence": 1.3,
        "min_clearance": 0.0135,
        "field_model": "cuboid",
        "chi_par": 0.05,
        "chi_perp": 0.10,
        "fov": (0.12, 0.10, 0.08),
        "resolution": (0.008, 0.01, 0.02),
        "fov_radius": 0.05,
        "fov_z_min": 0.012,
        "seed_reg": 0.002,
        "seed_iterations": 50,
        "seed_mirror": "y",
    }
