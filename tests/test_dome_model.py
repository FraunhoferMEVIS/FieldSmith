from typing import Any

import pytest
import torch

import fieldsmith.models.dome.model as dome_model_module
from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.models.dome import DomeAngleModel
from fieldsmith.point_sampler import PredefinedPointSampler


def dome_configuration(
    include_size: bool = True,
) -> MagnetConfiguration:
    orientations = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    orientations = orientations / torch.linalg.norm(orientations, dim=1, keepdim=True)
    metadata = {"geometry": "dome"}
    if include_size:
        metadata["magnet_size_m"] = 0.01
    return MagnetConfiguration(
        positions=torch.tensor([[-0.1, 0.0, 0.05], [0.1, 0.0, 0.05]], dtype=torch.float64),
        orientations=orientations,
        volumes=torch.full((2,), 1.0e-6, dtype=torch.float64),
        remanence=1.2,
        metadata=metadata,
    )


def make_model(
    configuration: MagnetConfiguration,
    **kwargs: Any,
) -> DomeAngleModel:
    return DomeAngleModel(
        configuration=configuration,
        fov=(0.0, 0.0, 0.0),
        resolution=(0.01, 0.01, 0.01),
        fov_radius=0.01,
        device="cpu",
        dtype=torch.float64,
        point_sampler=PredefinedPointSampler(torch.zeros((1, 3), dtype=torch.float64)),
        **kwargs,
    )


def test_model_reconstructs_initial_three_dimensional_orientations() -> None:
    configuration = dome_configuration()
    model = make_model(configuration)

    assert torch.allclose(
        model.create_magnet_orientations(),
        configuration.orientations,
        atol=1e-12,
    )
    assert model.magnet_angles.requires_grad
    assert model.magnet_polar_angles.requires_grad
    assert model.demag_coupling is None


@pytest.mark.parametrize("field_model", ["dipole", "cuboid"])
def test_forward_uses_selected_field_model(
    monkeypatch: pytest.MonkeyPatch,
    field_model: str,
) -> None:
    calls = []

    def fake_field(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return torch.full_like(kwargs["points"], 3.0)

    monkeypatch.setattr(dome_model_module, "calc_b_at_points_fast", fake_field)
    monkeypatch.setattr(dome_model_module, "calc_b_at_points_cuboid", fake_field)
    model = make_model(dome_configuration(), field_model=field_model)

    output = model()

    assert torch.equal(output["B"], torch.full((1, 3), 3.0, dtype=torch.float64))
    assert len(calls) == 1
    if field_model == "cuboid":
        assert calls[0]["magnet_size"].shape == (2, 3)
        assert calls[0]["axes"].shape == (2, 3, 3)
        assert calls[0]["axes"].requires_grad is False
    else:
        assert calls[0]["vol"] is model.volumes


def test_demagnetization_coupling_is_built_and_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = []
    apply_calls = []
    coupling = torch.eye(6, dtype=torch.float64)

    def fake_build(
        positions: torch.Tensor,
        sizes: torch.Tensor,
    ) -> torch.Tensor:
        build_calls.append((positions.clone(), sizes.clone()))
        return coupling

    def fake_apply(
        vectors: torch.Tensor,
        actual_coupling: torch.Tensor,
        chi_par: float,
        chi_perp: float,
    ) -> torch.Tensor:
        apply_calls.append((actual_coupling, chi_par, chi_perp))
        return vectors + 0.25

    monkeypatch.setattr(dome_model_module, "build_demag_coupling", fake_build)
    monkeypatch.setattr(dome_model_module, "apply_first_order_demag", fake_apply)
    model = make_model(dome_configuration(), chi_par=0.05, chi_perp=0.1)

    vectors = model.create_magnet_vectors()
    model.refresh_demagnetisation()

    assert len(build_calls) == 2
    assert torch.equal(model.demag_coupling, coupling)
    assert apply_calls == [(coupling, 0.05, 0.1)]
    assert torch.allclose(
        vectors,
        model.create_magnet_orientations() * model.remanence + 0.25,
    )


def test_get_configuration_records_dome_physics() -> None:
    model = make_model(
        dome_configuration(),
        field_model="cuboid",
        chi_par=0.05,
        chi_perp=0.1,
    )

    configuration = model.get_configuration()

    assert configuration.metadata["geometry"] == "dome"
    assert configuration.metadata["field_model"] == "cuboid"
    assert configuration.metadata["chi_par"] == pytest.approx(0.05)
    assert configuration.metadata["chi_perp"] == pytest.approx(0.1)
    assert torch.equal(
        configuration.metadata["magnet_polar_angles_rad"],
        model.magnet_polar_angles.detach(),
    )
    assert torch.allclose(
        configuration.metadata["remanence_vectors_t"],
        model.create_magnet_vectors().detach(),
    )


def test_model_rejects_invalid_field_model() -> None:
    with pytest.raises(ValueError, match='field_model must be "dipole" or "cuboid"'):
        make_model(dome_configuration(), field_model="sphere")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field_model": "cuboid"},
        {"chi_par": 0.1},
        {"chi_perp": 0.1},
    ],
)
def test_model_requires_magnet_size_for_body_physics(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        make_model(dome_configuration(include_size=False), **kwargs)
