import pytest
import torch

from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.lsq_seed import dipole_design_operator, lsq_seed_orientations


def test_dipole_design_operator_matches_dipole_field_kernel() -> None:
    positions = torch.tensor(
        [[-0.1, 0.02, 0.0], [0.08, -0.03, 0.01]],
        dtype=torch.float64,
    )
    roi_points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.02, -0.01]],
        dtype=torch.float64,
    )
    volumes = torch.tensor([1.0e-6, 2.0e-6], dtype=torch.float64)
    remanence = torch.tensor(
        [[1.0, 0.2, -0.1], [-0.3, 0.8, 0.4]],
        dtype=torch.float64,
    )

    operator = dipole_design_operator(positions, roi_points, volumes)
    operator_field = (operator @ remanence.reshape(-1)).reshape(-1, 3)
    kernel_field = calc_b_at_points_fast(
        dipole_br=remanence,
        vol=volumes,
        dipole_pos=positions,
        points=roi_points,
        chunk_points=None,
    )

    assert operator.shape == (6, 6)
    assert torch.allclose(operator_field, kernel_field, rtol=1e-12, atol=1e-15)


def test_lsq_seed_returns_unit_vectors_in_input_dtype() -> None:
    positions = torch.tensor(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=torch.float32,
    )
    roi_points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
        dtype=torch.float32,
    )
    volumes = torch.full((2,), 1.0e-6, dtype=torch.float32)

    orientations = lsq_seed_orientations(
        positions,
        roi_points,
        torch.tensor([0.0, 1.0, 0.0]),
        volumes,
        iterations=8,
        rho_end=1e6,
    )

    assert orientations.shape == positions.shape
    assert orientations.dtype == torch.float32
    assert torch.allclose(
        torch.linalg.norm(orientations, dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_lsq_seed_is_invariant_to_target_magnitude() -> None:
    positions = torch.tensor(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    roi_points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]],
        dtype=torch.float64,
    )
    volumes = torch.full((2,), 1.0e-6, dtype=torch.float64)

    unit_target = lsq_seed_orientations(
        positions,
        roi_points,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        volumes,
        iterations=5,
    )
    scaled_target = lsq_seed_orientations(
        positions,
        roi_points,
        torch.tensor([0.0, 7.0, 0.0], dtype=torch.float64),
        volumes,
        iterations=5,
    )

    assert torch.allclose(unit_target, scaled_target, atol=1e-12)


def test_lsq_seed_enforces_requested_mirror_symmetry() -> None:
    positions = torch.tensor(
        [[-0.1, 0.02, 0.0], [0.1, 0.02, 0.0]],
        dtype=torch.float64,
    )
    roi_points = torch.tensor([[0.0, 0.01, 0.0]], dtype=torch.float64)
    volumes = torch.full((2,), 1.0e-6, dtype=torch.float64)

    orientations = lsq_seed_orientations(
        positions,
        roi_points,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        volumes,
        iterations=8,
        rho_end=1e6,
        mirror="x",
    )

    reflected_second = orientations[1] * torch.tensor(
        [-1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    assert torch.allclose(orientations[0], reflected_second, atol=1e-12)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("positions", torch.zeros(3), "positions must have shape"),
        ("roi_points", torch.zeros(3), "roi_points must have shape"),
        ("target_direction", torch.zeros(2), "target_direction must have shape"),
    ],
)
def test_lsq_seed_rejects_invalid_shapes(
    argument: str,
    value: torch.Tensor,
    message: str,
) -> None:
    arguments = {
        "positions": torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "roi_points": torch.zeros((1, 3)),
        "target_direction": torch.tensor([0.0, 1.0, 0.0]),
        "volumes": torch.ones(2),
        "iterations": 2,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        lsq_seed_orientations(**arguments)


def test_lsq_seed_rejects_invalid_mirror_and_incompatible_target() -> None:
    positions = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    roi_points = torch.zeros((1, 3))
    volumes = torch.ones(2)

    with pytest.raises(ValueError, match='mirror must be "x", "y" or "z"'):
        lsq_seed_orientations(
            positions,
            roi_points,
            torch.tensor([0.0, 1.0, 0.0]),
            volumes,
            iterations=2,
            mirror="xy",
        )
    with pytest.raises(ValueError, match="cannot produce a mean field"):
        lsq_seed_orientations(
            positions,
            roi_points,
            torch.tensor([1.0, 0.0, 0.0]),
            volumes,
            iterations=2,
            mirror="x",
        )


def test_lsq_seed_rejects_non_symmetric_positions() -> None:
    positions = torch.tensor([[-0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])

    with pytest.raises(ValueError, match="positions are not symmetric"):
        lsq_seed_orientations(
            positions,
            torch.zeros((1, 3)),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.ones(2),
            iterations=2,
            mirror="x",
        )
