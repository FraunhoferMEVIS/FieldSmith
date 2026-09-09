from typing import Any

import pytest
import torch

from fieldsmith.field_simulations.rectangular_prism import (
    calc_b_at_points_rect_prism_from_cube_volume,
    calc_b_at_points_rect_prism_potential,
)


def test_point_chunking_matches_unchunked_field() -> None:
    magnet_br = torch.tensor([[0.5, 0.3, 1.0]], dtype=torch.float64)
    magnet_size = torch.tensor([[0.02, 0.03, 0.04]], dtype=torch.float64)
    magnet_pos = torch.zeros((1, 3), dtype=torch.float64)
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

    unchunked = calc_b_at_points_rect_prism_potential(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
        chunk_points=None,
    )
    chunked = calc_b_at_points_rect_prism_potential(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
        chunk_points=2,
    )

    assert torch.equal(chunked, unchunked)


def test_list_inputs_are_converted_to_point_dtype_and_device() -> None:
    points = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)

    field = calc_b_at_points_rect_prism_potential(
        magnet_br=[[0.0, 0.0, 1.2]],
        magnet_size=[[0.02, 0.02, 0.02]],
        magnet_pos=[[0.0, 0.0, 0.0]],
        points=points,
    )

    assert field.shape == (1, 3)
    assert field.dtype == points.dtype
    assert field.device == points.device
    assert torch.isfinite(field).all()


def test_field_is_finite_on_face_normal_extension() -> None:
    field = calc_b_at_points_rect_prism_potential(
        magnet_br=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        magnet_size=torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float64),
        magnet_pos=torch.zeros((1, 3), dtype=torch.float64),
        points=torch.tensor([[0.03, 0.0, 0.0]], dtype=torch.float64),
    )

    assert torch.isfinite(field).all()
    assert field[0, 0] > 0.0


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

    field = calc_b_at_points_rect_prism_potential(
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


def test_field_can_be_computed_inside_no_grad_context() -> None:
    magnet_br = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    magnet_size = torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float64)
    magnet_pos = torch.zeros((1, 3), dtype=torch.float64)
    points = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)

    with torch.no_grad():
        field = calc_b_at_points_rect_prism_potential(
            magnet_br,
            magnet_size,
            magnet_pos,
            points,
        )

    assert field.shape == (1, 3)
    assert not field.requires_grad
    assert torch.isfinite(field).all()


@pytest.mark.parametrize(
    "volumes",
    [
        0.008,
        torch.tensor(0.008, dtype=torch.float64),
        [0.008, 0.027],
        torch.tensor([0.008, 0.027], dtype=torch.float64),
    ],
)
def test_cube_volume_wrapper_matches_explicit_side_lengths(
    volumes: torch.Tensor | list[float] | float,
) -> None:
    magnet_br = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    magnet_pos = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    points = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    volume_tensor = torch.as_tensor(volumes, dtype=torch.float64).reshape(-1)
    if volume_tensor.numel() == 1:
        volume_tensor = volume_tensor.expand(2)
    side_lengths = torch.pow(volume_tensor, 1.0 / 3.0)

    wrapped = calc_b_at_points_rect_prism_from_cube_volume(
        magnet_br,
        volumes,
        magnet_pos,
        points,
        chunk_points=1,
    )
    explicit = calc_b_at_points_rect_prism_potential(
        magnet_br,
        side_lengths[:, None].expand(-1, 3),
        magnet_pos,
        points,
        chunk_points=1,
    )

    assert torch.equal(wrapped, explicit)


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
        calc_b_at_points_rect_prism_potential(**arguments)


def test_cube_volume_wrapper_rejects_incorrect_volume_count() -> None:
    with pytest.raises(ValueError, match=r"magnet_size must have shape \(M, 3\)"):
        calc_b_at_points_rect_prism_from_cube_volume(
            magnet_br=torch.zeros((2, 3)),
            vol=torch.ones(3),
            magnet_pos=torch.zeros((2, 3)),
            points=torch.ones((1, 3)),
        )
