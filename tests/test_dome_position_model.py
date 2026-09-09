import pytest
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.models.dome.position_model import DomePositionModel
from fieldsmith.point_sampler import PredefinedPointSampler


def position_configuration() -> MagnetConfiguration:
    return MagnetConfiguration(
        positions=torch.tensor([[-0.1, 0.0, 0.05], [0.1, 0.0, 0.05]], dtype=torch.float64),
        orientations=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64),
        volumes=torch.full((2,), 1.0e-6, dtype=torch.float64),
        remanence=1.2,
        metadata={"geometry": "dome", "magnet_size_m": 0.01},
    )


def make_position_model(
    position_bound_m: float = 0.002,
) -> DomePositionModel:
    return DomePositionModel(
        configuration=position_configuration(),
        fov=(0.0, 0.0, 0.0),
        resolution=(0.01, 0.01, 0.01),
        fov_radius=0.01,
        device="cpu",
        dtype=torch.float64,
        point_sampler=PredefinedPointSampler(torch.zeros((1, 3), dtype=torch.float64)),
        position_bound_m=position_bound_m,
    )


def test_effective_positions_apply_smooth_bounded_offsets() -> None:
    model = make_position_model(position_bound_m=0.002)
    with torch.no_grad():
        model.position_offsets.copy_(
            torch.tensor(
                [[0.002, -0.002, 0.0], [0.004, 0.0, -0.004]],
                dtype=torch.float64,
            ),
        )

    displacement = model.effective_positions() - model.magnet_positions
    expected = 0.002 * torch.tanh(model.position_offsets / 0.002)

    assert torch.allclose(displacement, expected)
    assert torch.all(displacement.abs() < 0.002)


def test_position_offsets_receive_field_gradients() -> None:
    model = make_position_model()

    model()["B"].square().sum().backward()

    assert model.position_offsets.grad is not None
    assert torch.isfinite(model.position_offsets.grad).all()


def test_bake_positions_preserves_effective_geometry_and_resets_offsets() -> None:
    model = make_position_model()
    with torch.no_grad():
        model.position_offsets.fill_(0.001)
    before = model.effective_positions().clone()

    moved = model.bake_positions()

    assert torch.allclose(model.magnet_positions, before)
    assert torch.allclose(model.effective_positions(), before)
    assert torch.equal(model.position_offsets, torch.zeros_like(model.position_offsets))
    assert torch.allclose(moved, before - position_configuration().positions)


def test_total_displacement_and_configuration_use_effective_positions() -> None:
    model = make_position_model()
    reference = model.magnet_positions.clone()
    with torch.no_grad():
        model.position_offsets[:, 0] = 0.001

    expected_positions = model.effective_positions().detach()
    expected_distance = 0.002 * torch.tanh(torch.tensor(0.5, dtype=torch.float64))
    distances = model.total_displacement(reference)
    configuration = model.get_configuration()

    assert torch.allclose(
        distances,
        torch.full((2,), expected_distance, dtype=torch.float64),
    )
    assert torch.allclose(configuration.positions, expected_positions)


@pytest.mark.parametrize("position_bound_m", [0.0, -0.001])
def test_model_rejects_nonpositive_position_bound(
    position_bound_m: float,
) -> None:
    with pytest.raises(ValueError, match="position_bound_m must be positive"):
        make_position_model(position_bound_m)
