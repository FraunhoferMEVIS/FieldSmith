import pytest
import torch

from fieldsmith.geometry.halbach_ring import (
    HalbachRingGeometry,
    construct_halbach_ring,
    make_cartesian_points,
    make_spherical_mask,
)


def test_halbach_ring_geometry_builds_expected_configuration() -> None:
    geometry = HalbachRingGeometry(
        n_magnets=4,
        ring_radius=0.1,
        magnet_size=0.02,
        remanence=1.3,
        device="cpu",
        dtype=torch.float64,
    )

    configuration = geometry.build()

    expected_positions = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    expected_orientations = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert configuration.num_magnets == 4
    assert configuration.positions.dtype == torch.float64
    assert torch.allclose(configuration.positions, expected_positions, atol=1e-12)
    assert torch.allclose(configuration.orientations, expected_orientations, atol=1e-12)
    assert torch.allclose(configuration.volumes, torch.full((4,), 0.02**3, dtype=torch.float64))
    assert configuration.remanence == pytest.approx(1.3)
    assert configuration.metadata["geometry"] == "halbach_ring"
    assert configuration.metadata["n_magnets"] == 4
    assert configuration.metadata["ring_radius_m"] == pytest.approx(0.1)
    assert configuration.metadata["magnet_size_m"] == pytest.approx(0.02)
    assert torch.allclose(
        configuration.metadata["initial_magnet_angles_rad"],
        torch.tensor([0.0, torch.pi, 2.0 * torch.pi, 3.0 * torch.pi], dtype=torch.float64),
    )


def test_construct_halbach_ring_matches_configuration_builder() -> None:
    parameters = {
        "n_magnets": 6,
        "ring_radius": 0.12,
        "magnet_size": 0.01,
        "remanence": 1.25,
        "device": "cpu",
        "dtype": torch.float64,
    }

    configuration = HalbachRingGeometry(**parameters).build()
    positions, remanence_vectors, volumes, initial_angles = construct_halbach_ring(**parameters)

    assert torch.equal(positions, configuration.positions)
    assert torch.equal(remanence_vectors, configuration.remanence_vectors())
    assert torch.equal(volumes, configuration.volumes)
    assert torch.equal(
        initial_angles,
        configuration.metadata["initial_magnet_angles_rad"],
    )


def test_make_cartesian_points_returns_centered_grid_and_flat_view() -> None:
    points, grid = make_cartesian_points(
        fov=(0.2, 0.1, 0.0),
        resolution=(0.1, 0.1, 0.1),
        device="cpu",
        dtype=torch.float64,
    )

    assert grid.shape == (3, 2, 1, 3)
    assert points.shape == (6, 3)
    assert points.dtype == torch.float64
    assert torch.equal(points, grid.reshape(-1, 3))
    assert torch.equal(grid[0, 0, 0], torch.tensor([-0.1, -0.05, 0.0], dtype=torch.float64))
    assert torch.equal(grid[-1, -1, 0], torch.tensor([0.1, 0.05, 0.0], dtype=torch.float64))


def test_make_spherical_mask_includes_boundary_points() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.04, 0.0],
            [0.05, 0.0, 0.0],
            [0.0501, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    mask = make_spherical_mask(points, radius=0.05)

    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([True, True, True, False]))
