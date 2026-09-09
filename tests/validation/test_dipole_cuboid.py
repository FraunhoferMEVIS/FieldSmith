import torch

from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.cube_axes import cube_axes_from_directions


def _assert_far_fields_are_similar(
    name: str,
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
) -> None:
    volumes = torch.prod(magnet_size, dim=1)
    axes = cube_axes_from_directions(magnet_br)
    dipole_field = calc_b_at_points_fast(
        dipole_br=magnet_br,
        vol=volumes,
        dipole_pos=magnet_pos,
        points=points,
        chunk_points=None,
        chunk_magnets=None,
    )
    cuboid_field = calc_b_at_points_cuboid(
        magnet_br=magnet_br,
        magnet_size=magnet_size,
        magnet_pos=magnet_pos,
        points=points,
        axes=axes,
        chunk_points=None,
    )

    difference = dipole_field - cuboid_field
    relative_l2_error = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        cuboid_field
    )
    cosine_similarity = torch.nn.functional.cosine_similarity(
        dipole_field,
        cuboid_field,
        dim=1,
    )

    assert relative_l2_error < 5e-3, (
        f"{name}: relative L2 field error is {relative_l2_error.item():.3e}"
    )
    assert torch.all(cosine_similarity > 0.99999), (
        f"{name}: minimum field-vector cosine similarity is "
        f"{cosine_similarity.min().item():.8f}"
    )


def test_dipole_approximates_cuboid_field_far_from_magnets() -> None:
    dtype = torch.float64

    _assert_far_fields_are_similar(
        name="single cube",
        magnet_br=torch.tensor([[0.3, -0.4, 1.2]], dtype=dtype),
        magnet_size=torch.full((1, 3), 0.01, dtype=dtype),
        magnet_pos=torch.zeros((1, 3), dtype=dtype),
        points=torch.tensor(
            [
                [0.2, 0.0, 0.0],
                [0.0, -0.25, 0.1],
                [0.15, 0.12, 0.2],
            ],
            dtype=dtype,
        ),
    )

    directions = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [-0.3, 1.0, 0.4],
            [0.2, -0.5, 1.0],
        ],
        dtype=dtype,
    )
    directions = directions / torch.linalg.vector_norm(
        directions,
        dim=1,
        keepdim=True,
    )
    _assert_far_fields_are_similar(
        name="three translated cubes",
        magnet_br=directions * torch.tensor([[1.1], [1.25], [0.95]], dtype=dtype),
        magnet_size=torch.tensor(
            [
                [0.010, 0.010, 0.010],
                [0.012, 0.012, 0.012],
                [0.008, 0.008, 0.008],
            ],
            dtype=dtype,
        ),
        magnet_pos=torch.tensor(
            [
                [-0.02, 0.00, 0.00],
                [0.02, 0.01, -0.01],
                [0.00, -0.02, 0.015],
            ],
            dtype=dtype,
        ),
        points=torch.tensor(
            [
                [0.25, 0.10, 0.20],
                [-0.20, 0.18, 0.22],
                [0.12, -0.25, 0.18],
                [0.05, 0.15, 0.30],
            ],
            dtype=dtype,
        ),
    )
