import pytest
import torch

from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.demagnetization import (
    apply_first_order_demag,
    build_demag_coupling,
)


def test_single_magnet_has_no_mutual_demagnetization_coupling() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    sizes = torch.full((1, 3), 0.012, dtype=torch.float64)

    coupling = build_demag_coupling(positions, sizes)

    assert coupling.shape == (3, 3)
    assert torch.equal(coupling, torch.zeros_like(coupling))


def test_coupling_matches_cuboid_fields_between_two_magnets() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.04, 0.01, -0.005]],
        dtype=torch.float64,
    )
    sizes = torch.tensor(
        [[0.012, 0.010, 0.008], [0.012, 0.010, 0.008]],
        dtype=torch.float64,
    )
    coupling = build_demag_coupling(positions, sizes, chunk_magnets=1)
    expected = torch.zeros_like(coupling)

    for source_index in range(2):
        target_index = 1 - source_index
        for component in range(3):
            remanence = torch.zeros((1, 3), dtype=torch.float64)
            remanence[0, component] = 1.0
            field = calc_b_at_points_cuboid(
                magnet_br=remanence,
                magnet_size=sizes[source_index:source_index + 1],
                magnet_pos=positions[source_index:source_index + 1],
                points=positions[target_index:target_index + 1],
                eps=1e-12,
                chunk_points=None,
            )
            expected[
                3 * target_index:3 * target_index + 3,
                3 * source_index + component,
            ] = field[0]
    expected = 0.5 * (expected + expected.T)

    blocks = coupling.view(2, 3, 2, 3)
    assert torch.equal(blocks[0, :, 0, :], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.equal(blocks[1, :, 1, :], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.allclose(coupling, coupling.T, atol=1e-14)
    assert torch.allclose(coupling, expected, rtol=1e-10, atol=1e-12)


def test_identity_body_axes_preserve_coupling_and_dtype_override() -> None:
    positions = torch.tensor(
        [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=torch.float32,
    )
    sizes = torch.full((2, 3), 0.01, dtype=torch.float32)
    axes = torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone()

    world_aligned = build_demag_coupling(positions, sizes, dtype=torch.float64)
    body_aligned = build_demag_coupling(
        positions,
        sizes,
        axes=axes,
        chunk_magnets=1,
        dtype=torch.float64,
    )

    assert world_aligned.dtype == torch.float64
    assert torch.allclose(body_aligned, world_aligned, rtol=1e-12, atol=1e-14)


def test_apply_first_order_demag_weights_parallel_and_transverse_fields() -> None:
    magnet_br = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    coupling = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    corrected = apply_first_order_demag(
        magnet_br,
        coupling,
        chi_par=0.5,
        chi_perp=0.25,
    )

    assert corrected.dtype == torch.float64
    assert torch.allclose(
        corrected,
        torch.tensor([[2.5, 0.75, 1.0]], dtype=torch.float64),
    )


def test_isotropic_demagnetization_matches_linear_coupling_and_has_gradients() -> None:
    magnet_br = torch.tensor(
        [[1.0, 0.2, -0.1], [-0.3, 0.8, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coupling = torch.arange(36, dtype=torch.float64).reshape(6, 6) / 100.0
    susceptibility = 0.07

    corrected = apply_first_order_demag(
        magnet_br,
        coupling,
        chi_par=susceptibility,
        chi_perp=susceptibility,
    )
    expected = magnet_br + susceptibility * (
        coupling @ magnet_br.reshape(-1)
    ).reshape_as(magnet_br)
    corrected.square().sum().backward()

    assert torch.allclose(corrected, expected)
    assert magnet_br.grad is not None
    assert torch.isfinite(magnet_br.grad).all()


@pytest.mark.parametrize(
    ("positions", "sizes", "axes", "message"),
    [
        (torch.zeros(3), torch.ones((1, 3)), None, "magnet_pos"),
        (torch.zeros((1, 3)), torch.ones(3), None, "magnet_size"),
        (torch.zeros((1, 3)), torch.ones((1, 3)), torch.eye(2), "axes"),
    ],
)
def test_build_demag_coupling_validates_input_shapes(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    axes: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_demag_coupling(positions, sizes, axes=axes)


def test_apply_first_order_demag_validates_input_shapes() -> None:
    with pytest.raises(ValueError, match="magnet_br"):
        apply_first_order_demag(torch.ones(3), torch.eye(3), 0.1, 0.1)
    with pytest.raises(ValueError, match="coupling"):
        apply_first_order_demag(torch.ones((2, 3)), torch.eye(3), 0.1, 0.1)
