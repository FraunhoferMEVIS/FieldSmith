import math

import pytest
import torch

from fieldsmith.geometry.dome import DomeGeometry


def test_build_creates_expected_single_layer_spherical_dome() -> None:
    geometry = DomeGeometry(
        aperture_radius=0.5,
        dome_radius=2.0,
        dome_height=2.0,
        n_layers=1,
        magnet_size=0.5,
        remanence=1.3,
        min_clearance=1.0,
        device="cpu",
        dtype=torch.float64,
    )

    configuration = geometry.build()

    ring_radius = math.sqrt(2.0)
    expected_first_position = torch.tensor(
        [ring_radius, 0.0, ring_radius],
        dtype=torch.float64,
    )
    radial_distance = torch.linalg.norm(configuration.positions[:, :2], dim=1)

    assert configuration.num_magnets == 8
    assert configuration.positions.shape == (8, 3)
    assert configuration.positions.dtype == torch.float64
    assert torch.allclose(configuration.positions[0], expected_first_position)
    assert torch.allclose(
        radial_distance,
        torch.full((8,), ring_radius, dtype=torch.float64),
    )
    assert torch.allclose(
        configuration.positions[:, 2],
        torch.full((8,), ring_radius, dtype=torch.float64),
    )


def test_build_places_points_on_ellipsoidal_shell() -> None:
    geometry = DomeGeometry(
        aperture_radius=0.2,
        dome_radius=0.8,
        dome_height=0.5,
        n_layers=3,
        magnet_size=0.05,
        remanence=1.2,
        min_clearance=0.08,
        device="cpu",
        dtype=torch.float64,
    )

    configuration = geometry.build()
    positions = configuration.positions
    shell_value = (
        positions[:, 0].square() + positions[:, 1].square()
    ) / geometry.dome_radius**2 + positions[:, 2].square() / geometry.dome_height**2

    assert torch.allclose(shell_value, torch.ones_like(shell_value), atol=1e-12)
    assert torch.all(positions[:, 2] > 0.0)
    assert torch.all(torch.linalg.norm(positions[:, :2], dim=1) >= geometry.aperture_radius)


def test_build_uses_radial_unit_orientations_and_uniform_volumes() -> None:
    geometry = DomeGeometry(
        aperture_radius=0.2,
        dome_radius=0.8,
        dome_height=0.5,
        n_layers=2,
        magnet_size=0.05,
        remanence=1.25,
        min_clearance=0.08,
        device="cpu",
        dtype=torch.float64,
    )

    configuration = geometry.build()
    expected_orientations = configuration.positions / torch.linalg.norm(
        configuration.positions,
        dim=1,
        keepdim=True,
    )

    assert torch.allclose(configuration.orientations, expected_orientations)
    assert torch.allclose(
        torch.linalg.norm(configuration.orientations, dim=1),
        torch.ones(configuration.num_magnets, dtype=torch.float64),
    )
    assert torch.allclose(
        configuration.volumes,
        torch.full(
            (configuration.num_magnets,),
            geometry.magnet_size**3,
            dtype=torch.float64,
        ),
    )
    assert configuration.remanence == pytest.approx(1.25)


def test_build_respects_minimum_clearance_within_layer() -> None:
    geometry = DomeGeometry(
        aperture_radius=0.5,
        dome_radius=2.0,
        dome_height=2.0,
        n_layers=1,
        magnet_size=0.5,
        remanence=1.3,
        min_clearance=1.0,
        device="cpu",
        dtype=torch.float64,
    )

    positions = geometry.build().positions
    neighbour_positions = torch.roll(positions, shifts=-1, dims=0)
    neighbour_distances = torch.linalg.norm(positions - neighbour_positions, dim=1)

    assert torch.all(neighbour_distances >= geometry.min_clearance)


def test_build_records_skipped_layers_and_metadata() -> None:
    target_direction = (1.0, 0.0, 0.0)
    geometry = DomeGeometry(
        aperture_radius=1.2,
        dome_radius=2.0,
        dome_height=2.0,
        n_layers=2,
        magnet_size=0.25,
        remanence=1.3,
        min_clearance=0.5,
        device="cpu",
        dtype=torch.float64,
        target_direction=target_direction,
    )

    configuration = geometry.build()
    metadata = configuration.metadata

    assert metadata["geometry"] == "dome"
    assert metadata["n_magnets"] == configuration.num_magnets
    assert metadata["magnets_per_layer"] == [21, 0]
    assert metadata["layer_polar_angles_rad"] == pytest.approx(
        [math.pi / 3.0, math.pi / 6.0],
    )
    assert metadata["aperture_radius_m"] == pytest.approx(1.2)
    assert metadata["dome_semi_axes_m"] == pytest.approx((2.0, 2.0))
    assert metadata["magnet_size_m"] == pytest.approx(0.25)
    assert metadata["target_direction"] == target_direction


def test_build_rejects_layer_spacing_that_does_not_fit() -> None:
    geometry = DomeGeometry(
        aperture_radius=0.0,
        dome_radius=1.0,
        dome_height=1.0,
        n_layers=2,
        magnet_size=0.5,
        remanence=1.3,
        min_clearance=0.8,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="n_layers layers cannot be spaced by min_clearance",
    ):
        geometry.build()


def test_build_rejects_dome_with_no_magnets_outside_aperture() -> None:
    geometry = DomeGeometry(
        aperture_radius=1.8,
        dome_radius=2.0,
        dome_height=2.0,
        n_layers=2,
        magnet_size=0.25,
        remanence=1.3,
        min_clearance=0.5,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="dome geometry produced no magnets outside the aperture",
    ):
        geometry.build()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"aperture_radius": -0.1},
            "dome_radius must exceed a non-negative aperture_radius",
        ),
        (
            {"dome_radius": 0.2},
            "dome_radius must exceed a non-negative aperture_radius",
        ),
        (
            {"dome_height": 0.0},
            "dome_radius must exceed a non-negative aperture_radius",
        ),
        (
            {"n_layers": 0},
            "n_layers and magnet_size must be positive",
        ),
        (
            {"magnet_size": 0.0},
            "n_layers and magnet_size must be positive",
        ),
        (
            {"min_clearance": 0.04},
            "min_clearance at least magnet_size",
        ),
    ],
)
def test_constructor_rejects_invalid_geometry(
    changes: dict[str, int | float],
    message: str,
) -> None:
    parameters = {
        "aperture_radius": 0.2,
        "dome_radius": 0.8,
        "dome_height": 0.5,
        "n_layers": 3,
        "magnet_size": 0.05,
        "remanence": 1.2,
        "min_clearance": 0.08,
        "device": "cpu",
    }
    parameters.update(changes)

    with pytest.raises(ValueError, match=message):
        DomeGeometry(**parameters)
