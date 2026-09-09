from typing import Any

import pytest
import torch

import fieldsmith.models.halbach_systems.model as system_model_module
from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.models.halbach_systems import HalbachSystemAngleModel
from fieldsmith.point_sampler import PredefinedPointSampler


def system_configuration(
    metadata: dict[str, object] | None = None,
) -> MagnetConfiguration:
    azimuth = torch.tensor([0.0, torch.pi], dtype=torch.float64)
    polar = torch.tensor([torch.pi / 3.0, 2.0 * torch.pi / 3.0], dtype=torch.float64)
    orientations = torch.stack(
        (
            torch.sin(polar) * torch.cos(azimuth),
            torch.sin(polar) * torch.sin(azimuth),
            torch.cos(polar),
        ),
        dim=1,
    )
    return MagnetConfiguration(
        positions=torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float64),
        orientations=orientations,
        volumes=torch.tensor([1.0e-6, 8.0e-6], dtype=torch.float64),
        remanence=1.2,
        metadata=dict(metadata or {}),
    )


def make_model(
    configuration: MagnetConfiguration,
    **kwargs: Any,
) -> HalbachSystemAngleModel:
    return HalbachSystemAngleModel(
        configuration=configuration,
        fov=(0.0, 0.0, 0.0),
        resolution=(0.01, 0.01, 0.01),
        fov_radius=0.01,
        device="cpu",
        dtype=torch.float64,
        point_sampler=PredefinedPointSampler(torch.zeros((1, 3), dtype=torch.float64)),
        **kwargs,
    )


def test_model_initializes_polar_angles_and_trainability() -> None:
    frozen = make_model(system_configuration(), optimize_out_of_plane=False)
    trainable = make_model(system_configuration(), optimize_out_of_plane=True)

    assert torch.allclose(
        frozen.create_magnet_orientations(),
        system_configuration().orientations,
        atol=1e-12,
    )
    assert frozen.magnet_polar_angles.requires_grad is False
    assert trainable.magnet_polar_angles.requires_grad is True


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {"magnet_dimensions_m": torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])},
            torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=torch.float64),
        ),
        (
            {"magnet_size_m": 0.02},
            torch.full((2, 3), 0.02, dtype=torch.float64),
        ),
        (
            {},
            torch.tensor([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=torch.float64),
        ),
    ],
)
def test_model_resolves_magnet_dimensions(
    metadata: dict[str, object],
    expected: torch.Tensor,
) -> None:
    model = make_model(system_configuration(metadata))

    assert torch.allclose(model.magnet_dimensions, expected)


@pytest.mark.parametrize("field_model", ["dipole", "cuboid"])
def test_forward_uses_selected_field_model(
    monkeypatch: pytest.MonkeyPatch,
    field_model: str,
) -> None:
    calls = []

    def fake_field(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        points = kwargs["points"]
        return torch.full_like(points, 7.0)

    monkeypatch.setattr(system_model_module, "calc_b_at_points_fast", fake_field)
    monkeypatch.setattr(system_model_module, "calc_b_at_points_cuboid", fake_field)
    model = make_model(
        system_configuration({"magnet_size_m": 0.01}),
        field_model=field_model,
    )

    output = model()

    assert torch.equal(output["B"], torch.full((1, 3), 7.0, dtype=torch.float64))
    assert len(calls) == 1
    if field_model == "cuboid":
        assert calls[0]["magnet_size"] is model.magnet_dimensions
        assert "axes" not in calls[0]
    else:
        assert calls[0]["vol"] is model.volumes


def test_get_configuration_records_system_settings() -> None:
    model = make_model(
        system_configuration({"magnet_size_m": 0.01}),
        optimize_out_of_plane=True,
        field_model="cuboid",
    )

    configuration = model.get_configuration()

    assert configuration.metadata["geometry"] == "halbach_system"
    assert configuration.metadata["n_magnets"] == 2
    assert configuration.metadata["optimize_out_of_plane"] is True
    assert configuration.metadata["field_model"] == "cuboid"
    assert torch.equal(
        configuration.metadata["magnet_polar_angles_rad"],
        model.magnet_polar_angles.detach(),
    )


def test_model_rejects_unknown_field_model() -> None:
    with pytest.raises(ValueError, match="field_model must be either"):
        make_model(system_configuration(), field_model="sphere")
