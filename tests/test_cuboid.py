from typing import Any

import pytest
import torch

from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid


def test_axis_aligned_cube_field_has_expected_symmetry() -> None:
    magnet_br = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    magnet_size = torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float64)
    magnet_pos = torch.zeros((1, 3), dtype=torch.float64)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.0, -0.1],
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    field = calc_b_at_points_cuboid(magnet_br, magnet_size, magnet_pos, points)

    assert torch.allclose(field[:, :2], torch.zeros_like(field[:, :2]), atol=1e-14)
    assert field[0, 2] > 0.0
    assert field[2, 2] < 0.0
    assert field[0, 2] == pytest.approx(field[1, 2].item())
    assert field[2, 2] == pytest.approx(field[3, 2].item())


def test_multiple_magnets_equal_sum_of_individual_fields() -> None:
    magnet_br = torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 0.5, 1.2]],
        dtype=torch.float64,
    )
    magnet_size = torch.tensor(
        [[0.02, 0.03, 0.04], [0.03, 0.02, 0.025]],
        dtype=torch.float64,
    )
    magnet_pos = torch.tensor(
        [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [[0.0, 0.0, 0.1], [0.02, -0.03, 0.08]],
        dtype=torch.float64,
    )

    combined = calc_b_at_points_cuboid(magnet_br, magnet_size, magnet_pos, points)
    separate = sum(
        (
            calc_b_at_points_cuboid(
                magnet_br[index:index + 1],
                magnet_size[index:index + 1],
                magnet_pos[index:index + 1],
                points,
            )
            for index in range(2)
        ),
        start=torch.zeros_like(combined),
    )

    assert torch.allclose(combined, separate, rtol=1e-12, atol=1e-14)


def test_point_chunking_matches_unchunked_field() -> None:
    magnet_br = torch.tensor(
        [[1.0, 0.0, 0.2], [0.0, 1.1, 0.3]],
        dtype=torch.float64,
    )
    magnet_size = torch.tensor(
        [[0.02, 0.03, 0.04], [0.025, 0.02, 0.03]],
        dtype=torch.float64,
    )
    magnet_pos = torch.tensor(
        [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.08],
            [0.01, 0.02, 0.09],
            [-0.03, 0.01, 0.1],
            [0.04, -0.02, 0.11],
            [0.0, 0.03, 0.12],
        ],
        dtype=torch.float64,
    )

    unchunked = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
        chunk_points=None,
    )
    chunked = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
        chunk_points=2,
    )

    assert torch.equal(chunked, unchunked)


def test_identity_axes_match_axis_aligned_calculation() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.1], [0.03, -0.02, 0.08]],
        dtype=torch.float64,
    )
    magnet_br = [[0.5, 0.2, 1.0]]
    magnet_size = [[0.02, 0.03, 0.04]]
    magnet_pos = [[0.0, 0.0, 0.0]]
    axes = [[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]]

    axis_aligned = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
    )
    oriented = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
        axes=axes,
    )

    assert oriented.dtype == points.dtype
    assert oriented.device == points.device
    assert torch.allclose(oriented, axis_aligned, rtol=1e-12, atol=1e-14)


def test_rotating_magnet_and_points_rotates_field() -> None:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    magnet_br = torch.tensor([[0.7, 0.2, 1.1]], dtype=torch.float64)
    magnet_size = torch.tensor([[0.02, 0.04, 0.06]], dtype=torch.float64)
    magnet_pos = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64)
    points = torch.tensor(
        [[0.08, 0.01, 0.12], [-0.04, 0.07, 0.1]],
        dtype=torch.float64,
    )

    base_field = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
    )
    rotated_field = calc_b_at_points_cuboid(
        magnet_br @ rotation.T,
        magnet_size,
        magnet_pos @ rotation.T,
        points @ rotation.T,
        axes=rotation.T.unsqueeze(0),
    )

    assert torch.allclose(
        rotated_field,
        base_field @ rotation.T,
        rtol=1e-11,
        atol=1e-13,
    )


def test_field_preserves_magnet_parameter_gradients() -> None:
    magnet_br = torch.tensor(
        [[0.8, 0.2, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    magnet_size = torch.tensor(
        [[0.02, 0.03, 0.04]],
        dtype=torch.float64,
        requires_grad=True,
    )
    magnet_pos = torch.tensor(
        [[0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    points = torch.tensor([[0.03, 0.02, 0.1]], dtype=torch.float64)

    field = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
    )
    field.square().sum().backward()

    assert magnet_br.grad is not None
    assert magnet_size.grad is not None
    assert magnet_pos.grad is not None
    assert torch.isfinite(magnet_br.grad).all()
    assert torch.isfinite(magnet_size.grad).all()
    assert torch.isfinite(magnet_pos.grad).all()


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("magnet_br", torch.zeros(3), r"magnet_br must have shape \(M, 3\)"),
        ("magnet_br", torch.zeros((1, 2)), r"magnet_br must have shape \(M, 3\)"),
        ("magnet_size", torch.zeros(3), r"magnet_size must have shape \(M, 3\)"),
        ("magnet_size", torch.zeros((1, 2)), r"magnet_size must have shape \(M, 3\)"),
        (
            "magnet_pos",
            torch.zeros((2, 3)),
            r"magnet_pos and magnet_br must both have shape \(M, 3\)",
        ),
        ("magnet_size", torch.zeros((2, 3)), r"magnet_size must have shape \(M, 3\)"),
        ("points", torch.zeros(3), r"points must have shape \(P, 3\)"),
        ("points", torch.zeros((1, 2)), r"points must have shape \(P, 3\)"),
        ("axes", torch.eye(3), r"axes must have shape \(M, 3, 3\)"),
    ],
)
def test_rejects_invalid_shapes(
    argument: str,
    value: torch.Tensor,
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "magnet_br": torch.zeros((1, 3)),
        "magnet_size": torch.ones((1, 3)),
        "magnet_pos": torch.zeros((1, 3)),
        "points": torch.ones((1, 3)),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        calc_b_at_points_cuboid(**arguments)
