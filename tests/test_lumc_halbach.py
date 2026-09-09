from unittest.mock import Mock

import numpy as np
import pytest
import torch

import fieldsmith.evaluation.lumc_halbach as lumc


def test_magnetization_returns_cube_dipole_moment() -> None:
    remanence = 1.3
    magnet_size = 0.012

    result = lumc.magnetization(remanence, magnet_size)

    expected = remanence * magnet_size**3 / (4.0 * np.pi * 1e-7)
    assert result == pytest.approx(expected)


def test_single_magnet_upstream_matches_axial_and_equatorial_dipole_fields() -> None:
    moment = 2.0

    axial = lumc.single_magnet_upstream(
        position=(2.0, 0.0, 0.0),
        dipole_moment=np.array([moment, 0.0]),
        sim_dimensions=(0.0, 0.0, 0.0),
        resolution=1.0,
    )
    equatorial = lumc.single_magnet_upstream(
        position=(0.0, 2.0, 0.0),
        dipole_moment=np.array([moment, 0.0]),
        sim_dimensions=(0.0, 0.0, 0.0),
        resolution=1.0,
    )

    assert axial.shape == (1, 1, 1, 3)
    assert axial.dtype == np.float32
    assert axial[0, 0, 0] == pytest.approx([2.0 * moment / 2.0**3, 0.0, 0.0])
    assert equatorial[0, 0, 0] == pytest.approx([-moment / 2.0**3, 0.0, 0.0])


def test_create_halbach_upstream_sums_every_ring_magnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contributions: list[tuple[tuple[float, float, float], np.ndarray]] = []

    def fake_single_magnet(
        position: tuple[float, float, float],
        dipole_moment: np.ndarray,
        sim_dimensions: tuple[float, float, float],
        resolution: float,
    ) -> np.ndarray:
        contributions.append((position, dipole_moment))
        shape = tuple(int(size * resolution) + 1 for size in sim_dimensions)
        return np.ones((*shape, 3), dtype=np.float32)

    monkeypatch.setattr(lumc, "single_magnet_upstream", fake_single_magnet)

    field = lumc.create_halbach_upstream(
        num_magnets=2,
        rings=(-0.1, 0.1),
        radius=0.2,
        magnet_size=0.01,
        k_value=2,
        resolution=10.0,
        b_rem=1.2,
        sim_dimensions=(0.2, 0.1, 0.0),
    )

    assert field.shape == (3, 2, 1, 3)
    assert field.dtype == np.float32
    assert np.all(field == 4.0)
    assert len(contributions) == 4
    assert contributions[0][0] == pytest.approx((0.2, 0.0, -0.1))
    assert contributions[1][0] == pytest.approx((-0.2, 0.0, -0.1))
    expected_moment = lumc.magnetization(1.2, 0.01) * 1e-7
    assert contributions[0][1] == pytest.approx([expected_moment, 0.0])
    assert contributions[1][1] == pytest.approx([expected_moment, 0.0])


def test_create_configuration_builds_centered_and_mirrored_rings() -> None:
    genes = np.array([0, 1])
    ring_positions = np.array([0.0, 0.02])

    configuration = lumc.create_configuration(
        genes=genes,
        ring_positions_nonnegative=ring_positions,
        magnet_size=0.012,
        remanence=1.3,
        device="cpu",
        dtype=torch.float64,
    )

    centered_count = int(lumc.INNER_NUM_MAGNETS[0] + lumc.OUTER_NUM_MAGNETS[0])
    mirrored_count = 2 * int(
        lumc.INNER_NUM_MAGNETS[1] + lumc.OUTER_NUM_MAGNETS[1]
    )
    assert configuration.num_magnets == centered_count + mirrored_count
    assert configuration.positions.dtype == torch.float64
    assert configuration.orientations.dtype == torch.float64
    assert torch.allclose(
        torch.linalg.vector_norm(configuration.orientations, dim=1),
        torch.ones(configuration.num_magnets, dtype=torch.float64),
    )
    assert torch.allclose(
        configuration.volumes,
        torch.full_like(configuration.volumes, 0.012**3),
    )

    z_positions = configuration.positions[:, 2]
    assert int(torch.count_nonzero(z_positions == 0.0)) == centered_count
    assert int(torch.count_nonzero(z_positions == -0.02)) == mirrored_count // 2
    assert int(torch.count_nonzero(z_positions == 0.02)) == mirrored_count // 2

    radii = torch.linalg.vector_norm(configuration.positions[:, :2], dim=1)
    expected_radii = torch.tensor(
        [
            lumc.INNER_RING_RADII[0],
            lumc.OUTER_RING_RADII[0],
            lumc.INNER_RING_RADII[1],
            lumc.OUTER_RING_RADII[1],
        ],
        dtype=torch.float64,
    )
    for radius in expected_radii:
        assert torch.any(torch.isclose(radii, radius))

    assert configuration.metadata["geometry"] == "lumc_halbach_optimisation"
    assert configuration.metadata["best_vector"] == [0, 1]
    assert configuration.metadata["ring_positions_nonnegative_m"] == [0.0, 0.02]
    assert configuration.metadata["n_magnets"] == configuration.num_magnets
    ring_indices = configuration.metadata["ring_indices"]
    assert int(torch.count_nonzero(ring_indices == 0)) == centered_count
    assert int(torch.count_nonzero(ring_indices == 1)) == mirrored_count


def test_create_configuration_sets_halbach_orientations() -> None:
    configuration = lumc.create_configuration(
        genes=[0],
        ring_positions_nonnegative=np.array([0.0]),
        dtype=torch.float64,
    )
    count = int(lumc.INNER_NUM_MAGNETS[0])
    position_angles = torch.atan2(
        configuration.positions[:count, 1],
        configuration.positions[:count, 0],
    )
    expected = torch.stack(
        (
            torch.cos(2.0 * position_angles),
            torch.sin(2.0 * position_angles),
            torch.zeros_like(position_angles),
        ),
        dim=1,
    )

    assert torch.allclose(configuration.orientations[:count], expected)


def test_fieldsmith_field_x_delegates_to_dipole_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = lumc.create_configuration(
        genes=[0],
        ring_positions_nonnegative=np.array([0.0]),
        dtype=torch.float64,
    )
    points = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]], dtype=torch.float32)
    expected_field = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )
    kernel = Mock(return_value=expected_field)
    monkeypatch.setattr(lumc, "calc_b_at_points_fast", kernel)

    result = lumc.fieldsmith_field_x(configuration, points, chunk_points=17)

    call = kernel.call_args
    assert torch.equal(call.args[0], configuration.remanence_vectors())
    assert call.args[1] is configuration.volumes
    assert call.args[2] is configuration.positions
    assert call.args[3].dtype == configuration.positions.dtype
    assert call.kwargs == {"chunk_points": 17}
    assert isinstance(result, np.ndarray)
    assert result == pytest.approx([1.0, 4.0])
