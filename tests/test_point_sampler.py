import pytest
import torch

from fieldsmith.point_sampler import CartesianGridSampler, MaskedPointSampler, PredefinedPointSampler


def test_cartesian_grid_sampler_returns_grid_and_flat_points() -> None:
    sampler = CartesianGridSampler(
        fov=(0.2, 0.1, 0.0),
        resolution=(0.1, 0.1, 0.1),
        device="cpu",
        dtype=torch.float64,
    )

    grid = sampler.get_grid()
    points = sampler.get_points()

    assert grid.shape == (3, 2, 1, 3)
    assert points.shape == (6, 3)
    assert points.dtype == torch.float64
    assert torch.equal(points, grid.reshape(-1, 3))


def test_masked_point_sampler_exposes_mask_and_selected_points() -> None:
    base_sampler = CartesianGridSampler(
        fov=(0.2, 0.2, 0.0),
        resolution=(0.1, 0.1, 0.1),
        device="cpu",
    )
    sampler = MaskedPointSampler(base_sampler, lambda points: torch.linalg.norm(points, dim=-1) <= 0.1)

    mask = sampler.get_mask()
    points = sampler.get_points()

    assert mask.shape == base_sampler.get_grid().shape[:-1]
    assert points.shape == (5, 3)
    assert torch.all(torch.linalg.norm(points, dim=-1) <= 0.1)


def test_predefined_point_sampler_returns_input() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    sampler = PredefinedPointSampler(points)

    assert sampler.get_points() is points
    with pytest.raises(NotImplementedError):
        sampler.get_grid()


def test_samplers_validate_shapes_and_masks() -> None:
    with pytest.raises(ValueError):
        PredefinedPointSampler(torch.zeros(3))

    sampler = MaskedPointSampler(
        CartesianGridSampler((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), device="cpu"),
        lambda points: torch.ones(points.shape[0]),
    )
    with pytest.raises(ValueError):
        sampler.get_points()
