import torch

from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.rectangular_prism import (
    calc_b_at_points_rect_prism_potential,
)


def test_rectangular_prism_matches_exact_cuboid_kernel_outside_magnets() -> None:
    magnet_br = torch.tensor(
        [[0.8, 0.2, 1.1], [-0.3, 0.9, 0.4]],
        dtype=torch.float64,
    )
    magnet_size = torch.tensor(
        [[0.02, 0.03, 0.04], [0.03, 0.025, 0.02]],
        dtype=torch.float64,
    )
    magnet_pos = torch.tensor(
        [[-0.04, 0.0, 0.0], [0.04, 0.01, -0.01]],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.1],
            [0.03, -0.04, 0.08],
            [-0.06, 0.03, 0.07],
        ],
        dtype=torch.float64,
    )

    potential_field = calc_b_at_points_rect_prism_potential(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
    )
    exact_field = calc_b_at_points_cuboid(
        magnet_br,
        magnet_size,
        magnet_pos,
        points,
    )

    assert torch.allclose(potential_field, exact_field, rtol=1e-9, atol=1e-12)
