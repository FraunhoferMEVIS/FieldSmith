import math

import pytest
import torch

from fieldsmith.field_simulations.dipole import calc_b_at_points, calc_b_at_points_fast


def test_single_dipole_matches_axial_and_equatorial_fields() -> None:
    remanence = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    positions = torch.zeros((1, 3), dtype=torch.float64)
    points = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )

    field = calc_b_at_points_fast(remanence, 1.0, positions, points)

    axial_strength = 1.0 / (2.0 * math.pi * 2.0**3)
    equatorial_strength = -1.0 / (4.0 * math.pi * 2.0**3)
    expected = torch.tensor(
        [[axial_strength, 0.0, 0.0], [equatorial_strength, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(field, expected)


def test_chunking_matches_unchunked_superposition() -> None:
    remanence = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.8]],
        dtype=torch.float64,
    )
    volumes = torch.tensor([0.5, 0.8, 1.2], dtype=torch.float64)
    positions = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.3, 0.8],
            [-0.4, 0.1, 1.2],
            [0.3, 0.4, 0.9],
            [0.0, -0.2, 1.4],
        ],
        dtype=torch.float64,
    )

    unchunked = calc_b_at_points_fast(
        remanence,
        volumes,
        positions,
        points,
        chunk_points=None,
        chunk_magnets=None,
    )
    chunked = calc_b_at_points_fast(
        remanence,
        volumes,
        positions,
        points,
        chunk_points=2,
        chunk_magnets=1,
    )

    assert torch.allclose(chunked, unchunked)


@pytest.mark.parametrize(
    "volumes",
    [
        0.5,
        [0.5],
        torch.tensor(0.5),
        torch.tensor([0.5]),
        [0.5, 0.5],
        torch.tensor([0.5, 0.5]),
    ],
)
def test_volume_representations_produce_the_same_field(
    volumes: torch.Tensor | list[float] | float,
) -> None:
    remanence = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    positions = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    field = calc_b_at_points_fast(remanence, volumes, positions, points)
    expected = calc_b_at_points_fast(
        torch.tensor(remanence),
        torch.tensor([0.5, 0.5]),
        torch.tensor(positions),
        points,
    )

    assert field.dtype == points.dtype
    assert field.device == points.device
    assert torch.allclose(field, expected)


def test_eps_keeps_field_finite_at_dipole_position() -> None:
    remanence = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    positions = torch.zeros((1, 3), dtype=torch.float64)

    field = calc_b_at_points_fast(
        remanence,
        1.0,
        positions,
        positions.clone(),
        eps=0.1,
    )

    expected = torch.tensor([[-1.0 / (4.0 * math.pi * 0.1**3), 0.0, 0.0]], dtype=torch.float64)
    assert torch.isfinite(field).all()
    assert torch.allclose(field, expected)


def test_field_calculation_preserves_gradients() -> None:
    remanence = torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    positions = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    field = calc_b_at_points_fast(remanence, 0.5, positions, points)
    field.square().sum().backward()

    assert remanence.grad is not None
    assert positions.grad is not None
    assert torch.isfinite(remanence.grad).all()
    assert torch.isfinite(positions.grad).all()


def test_calc_b_at_points_matches_fast_implementation() -> None:
    remanence = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    positions = torch.zeros((1, 3), dtype=torch.float64)
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    expected = calc_b_at_points_fast(
        remanence,
        0.5,
        positions,
        points,
        eps=1e-9,
        chunk_points=1,
    )
    result = calc_b_at_points(
        remanence,
        0.5,
        positions,
        points,
        eps=1e-9,
        chunk_points=1,
    )

    assert torch.equal(result, expected)


def test_rejects_mismatched_dipole_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"dipole_br and dipole_pos must both have shape \(M, 3\)",
    ):
        calc_b_at_points_fast(
            torch.zeros((2, 3)),
            1.0,
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
        )


def test_rejects_incorrect_volume_count() -> None:
    with pytest.raises(ValueError, match=r"vol must be scalar or have shape \(M,\)"):
        calc_b_at_points_fast(
            torch.zeros((2, 3)),
            torch.ones(3),
            torch.zeros((2, 3)),
            torch.zeros((1, 3)),
        )


@pytest.mark.parametrize(
    "points",
    [
        torch.zeros(3),
        torch.zeros((2, 2)),
    ],
)
def test_rejects_invalid_point_shapes(points: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"points must have shape \(P, 3\)"):
        calc_b_at_points_fast(
            torch.zeros((1, 3)),
            1.0,
            torch.zeros((1, 3)),
            points,
        )
