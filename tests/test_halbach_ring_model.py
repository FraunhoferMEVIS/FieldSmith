from typing import Any

import pytest
import torch

import fieldsmith.models.halbach_ring.model as ring_model_module
from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.field_simulations.dipole import calc_b_at_points_fast
from fieldsmith.geometry.halbach_ring import HalbachRingGeometry
from fieldsmith.models.halbach_ring import SingleRingHalbachModel
from fieldsmith.point_sampler import PointSampler, PredefinedPointSampler


class InvalidSampler(PointSampler):
    def __init__(self, points: torch.Tensor) -> None:
        self.points: torch.Tensor = points

    def get_points(self) -> torch.Tensor:
        return self.points

    def get_grid(self) -> torch.Tensor:
        return self.points


def ring_configuration(
    n_magnets: int = 4,
    remanence: torch.Tensor | float = 1.3,
) -> MagnetConfiguration:
    configuration = HalbachRingGeometry(
        n_magnets=n_magnets,
        ring_radius=0.1,
        magnet_size=0.01,
        remanence=1.3,
        device="cpu",
        dtype=torch.float64,
    ).build()
    configuration.remanence = remanence
    return configuration


def point_sampler() -> PredefinedPointSampler:
    return PredefinedPointSampler(
        torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=torch.float64),
    )


def test_forward_matches_dipole_kernel_and_backpropagates() -> None:
    configuration = ring_configuration()
    model = SingleRingHalbachModel(
        configuration=configuration,
        device="cpu",
        dtype=torch.float64,
        point_sampler=point_sampler(),
        chunk_points=None,
    )

    output = model()
    expected = calc_b_at_points_fast(
        dipole_br=model.create_magnet_vectors(),
        vol=configuration.volumes,
        dipole_pos=configuration.positions,
        points=model.points,
        chunk_points=None,
    )
    output["B"].square().sum().backward()

    assert torch.allclose(output["B"], expected)
    assert output["points"] is model.points
    assert output["magnet_positions"] is model.magnet_positions
    assert model.magnet_angles.grad is not None
    assert torch.isfinite(model.magnet_angles.grad).all()


def test_symmetric_optimization_shares_angle_updates() -> None:
    model = SingleRingHalbachModel(
        configuration=ring_configuration(),
        device="cpu",
        dtype=torch.float64,
        point_sampler=point_sampler(),
        symmetric_optimization=True,
    )

    with torch.no_grad():
        model.magnet_angles.add_(torch.tensor([0.1, -0.2], dtype=torch.float64))
    expanded = model.expanded_magnet_angles()
    updates = expanded - model.initial_magnet_angles

    assert model.magnet_angles.numel() == 2
    assert torch.allclose(updates[:2], torch.tensor([0.1, -0.2], dtype=torch.float64))
    assert torch.allclose(updates[2:], updates[:2])


def test_get_configuration_exports_current_design_and_metadata() -> None:
    model = SingleRingHalbachModel(
        configuration=ring_configuration(),
        device="cpu",
        dtype=torch.float64,
        point_sampler=point_sampler(),
        symmetric_optimization=True,
    )

    configuration = model.get_configuration()

    assert torch.equal(configuration.positions, model.effective_positions())
    assert torch.allclose(configuration.orientations, model.create_magnet_orientations())
    assert configuration.metadata["geometry"] == "halbach_ring"
    assert configuration.metadata["n_magnets"] == 4
    assert configuration.metadata["symmetric_optimization"] is True
    assert torch.equal(configuration.metadata["magnet_angles_rad"], model.magnet_angles.detach())


def test_get_images_to_log_uses_existing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SingleRingHalbachModel(
        configuration=ring_configuration(),
        device="cpu",
        dtype=torch.float64,
        point_sampler=point_sampler(),
    )
    output = model()
    captured = {}
    sentinel = object()

    def fake_plot(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ring_model_module, "plot_model_output_3d", fake_plot)

    assert model.get_images_to_log(output) == {"system_3d": sentinel}
    assert captured["output"] is output
    assert captured["cube_size"] == pytest.approx(0.01)
    assert captured["title"] == "Halbach ring"


@pytest.mark.parametrize(
    "remanence",
    [
        torch.tensor(1.3),
        torch.full((4,), 1.3),
    ],
)
def test_model_accepts_uniform_tensor_remanence(
    remanence: torch.Tensor,
) -> None:
    model = SingleRingHalbachModel(
        configuration=ring_configuration(remanence=remanence),
        device="cpu",
        point_sampler=point_sampler(),
    )

    assert model.remanence == pytest.approx(1.3)


def test_model_rejects_nonuniform_remanence() -> None:
    with pytest.raises(ValueError, match="expects scalar remanence"):
        SingleRingHalbachModel(
            configuration=ring_configuration(
                remanence=torch.tensor([1.0, 1.1, 1.0, 1.0]),
            ),
            device="cpu",
            point_sampler=point_sampler(),
        )


def test_model_rejects_invalid_or_empty_sampler() -> None:
    with pytest.raises(ValueError, match=r"shape \(P, 3\)"):
        SingleRingHalbachModel(
            configuration=ring_configuration(),
            device="cpu",
            point_sampler=InvalidSampler(torch.zeros(3)),
        )
    with pytest.raises(ValueError, match="produced no evaluation points"):
        SingleRingHalbachModel(
            configuration=ring_configuration(),
            device="cpu",
            point_sampler=InvalidSampler(torch.empty((0, 3))),
        )


def test_symmetric_optimization_validates_opposite_pairs() -> None:
    with pytest.raises(ValueError, match="even number"):
        SingleRingHalbachModel(
            configuration=ring_configuration(n_magnets=3),
            device="cpu",
            point_sampler=point_sampler(),
            symmetric_optimization=True,
        )

    configuration = ring_configuration()
    configuration.positions[3] = torch.tensor([0.2, 0.2, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="to be opposite"):
        SingleRingHalbachModel(
            configuration=configuration,
            device="cpu",
            point_sampler=point_sampler(),
            symmetric_optimization=True,
        )
