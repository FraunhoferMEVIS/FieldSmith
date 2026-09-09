from pathlib import Path

import magpylib as magpy
import torch
from scipy.spatial.transform import Rotation

from fieldsmith.configurations.halbach_system_config import HalbachSystemConfig
from fieldsmith.field_simulations.cuboid import calc_b_at_points_cuboid
from fieldsmith.geometry.cube_axes import cube_axes_from_directions
from fieldsmith.geometry.halbach_system import HalbachSystemGeometry


ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
SMALL_HALBACH_CONFIG = ROOT_DIRECTORY / "configs" / "smh_geometry.cfg"


def _magpylib_field(
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    axes: torch.Tensor,
) -> torch.Tensor:
    local_br = torch.einsum("mkj,mj->mk", axes, magnet_br)
    sources = [
        magpy.magnet.Cuboid(
            position=position.detach().cpu().numpy(),
            orientation=Rotation.from_matrix(body_axes.detach().cpu().numpy().T),
            dimension=dimensions.detach().cpu().numpy(),
            polarization=polarization.detach().cpu().numpy(),
        )
        for position, body_axes, dimensions, polarization in zip(
            magnet_pos,
            axes,
            magnet_size,
            local_br,
        )
    ]
    reference = magpy.Collection(*sources).getB(points.detach().cpu().numpy())
    return torch.as_tensor(reference, dtype=points.dtype, device=points.device)


def _assert_matches_magpylib(
    name: str,
    magnet_br: torch.Tensor,
    magnet_size: torch.Tensor,
    magnet_pos: torch.Tensor,
    points: torch.Tensor,
    axes: torch.Tensor,
) -> None:
    fieldsmith_field = calc_b_at_points_cuboid(
        magnet_br=magnet_br,
        magnet_size=magnet_size,
        magnet_pos=magnet_pos,
        points=points,
        axes=axes,
        chunk_points=8,
    )
    magpylib_field = _magpylib_field(
        magnet_br=magnet_br,
        magnet_size=magnet_size,
        magnet_pos=magnet_pos,
        points=points,
        axes=axes,
    )

    difference = fieldsmith_field - magpylib_field
    relative_l2_error = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        magpylib_field
    ).clamp_min(torch.finfo(points.dtype).eps)
    reference_strength = torch.linalg.vector_norm(magpylib_field, dim=1)
    compared_points = reference_strength > 1e-10
    cosine_similarity = torch.nn.functional.cosine_similarity(
        fieldsmith_field[compared_points],
        magpylib_field[compared_points],
        dim=1,
    )

    assert relative_l2_error < 2e-5, (
        f"{name}: relative L2 field error is {relative_l2_error.item():.3e}"
    )
    assert torch.all(cosine_similarity > 1.0 - 1e-8), (
        f"{name}: minimum field-vector cosine similarity is "
        f"{cosine_similarity.min().item():.12f}"
    )


def _halbach_sample_points(config: HalbachSystemConfig) -> torch.Tensor:
    extent = min(
        config.fov[0] / 2.0,
        config.fov[1] / 2.0,
        config.fov[2] / 2.0,
        config.fov_radius / 3.0**0.5,
    )
    sample_axis = torch.linspace(-extent, extent, 3, dtype=torch.float64)
    return torch.cartesian_prod(sample_axis, sample_axis, sample_axis)


def test_cuboid_field_matches_magpylib_for_magnet_setups_and_halbach_system() -> None:
    dtype = torch.float64

    single_orientation = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
    single_axes = cube_axes_from_directions(single_orientation)
    _assert_matches_magpylib(
        name="single axis-aligned cube",
        magnet_br=1.2 * single_orientation,
        magnet_size=torch.full((1, 3), 0.012, dtype=dtype),
        magnet_pos=torch.zeros((1, 3), dtype=dtype),
        points=torch.tensor(
            [[0.03, 0.01, 0.02], [-0.025, 0.015, 0.04]],
            dtype=dtype,
        ),
        axes=single_axes,
    )

    orientations = torch.tensor(
        [[1.0, 2.0, 0.5], [-0.3, 0.4, 1.0]],
        dtype=dtype,
    )
    orientations = orientations / torch.linalg.vector_norm(
        orientations,
        dim=1,
        keepdim=True,
    )
    _assert_matches_magpylib(
        name="two translated and rotated cubes",
        magnet_br=orientations * torch.tensor([[1.1], [1.35]], dtype=dtype),
        magnet_size=torch.tensor(
            [[0.012, 0.012, 0.012], [0.016, 0.016, 0.016]],
            dtype=dtype,
        ),
        magnet_pos=torch.tensor(
            [[-0.02, 0.01, 0.0], [0.025, -0.015, 0.01]],
            dtype=dtype,
        ),
        points=torch.tensor(
            [[0.0, 0.0, 0.05], [0.04, 0.03, 0.06], [-0.05, 0.02, 0.04]],
            dtype=dtype,
        ),
        axes=cube_axes_from_directions(orientations),
    )

    system_config = HalbachSystemConfig.from_file(SMALL_HALBACH_CONFIG)
    system = HalbachSystemGeometry(
        n_magnets_per_ring=system_config.n_magnets_per_ring,
        n_rings=system_config.n_rings,
        n_rings_per_plane=system_config.n_rings_per_plane,
        additional_magnets_per_ring=system_config.additional_magnets_per_ring,
        ring_radius=system_config.ring_radius,
        system_length=system_config.system_length,
        magnet_size=system_config.magnet_size,
        remanence=system_config.remanence,
        device="cpu",
        dtype=dtype,
    ).build()
    system_axes = cube_axes_from_directions(system.orientations)

    assert system.num_magnets == 200
    _assert_matches_magpylib(
        name="configured small Halbach system",
        magnet_br=system.remanence_vectors(),
        magnet_size=torch.full_like(system.positions, system_config.magnet_size),
        magnet_pos=system.positions,
        points=_halbach_sample_points(system_config),
        axes=system_axes,
    )
