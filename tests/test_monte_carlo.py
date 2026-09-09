import pytest
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.evaluation.monte_carlo import run_monte_carlo


@pytest.fixture
def configuration() -> MagnetConfiguration:
    return MagnetConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        orientations=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        volumes=torch.tensor([1e-6], dtype=torch.float64),
        remanence=1.0,
    )


@pytest.fixture
def points() -> torch.Tensor:
    return torch.tensor(
        [[0.05, 0.01, 0.0], [0.06, 0.0, 0.01], [0.07, -0.01, 0.0]],
        dtype=torch.float64,
    )


def test_zero_measurement_tolerances_preserve_seeded_results(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
) -> None:
    common = {
        "configuration": configuration,
        "points": points,
        "num_samples": 3,
        "position_tolerance_m": 0.0,
        "rotation_tolerance_deg": 0.0,
        "remanence_tolerance_t": 0.0,
        "seed": 12,
        "progress_callback": None,
    }

    default_result = run_monte_carlo(**common)
    explicit_result = run_monte_carlo(
        **common,
        probe_position_tolerance_m=0.0,
        probe_tilt_tolerance_deg=0.0,
        measurement_device_tolerance_t=0.0,
    )

    assert torch.equal(default_result.fieldstrength_mT, explicit_result.fieldstrength_mT)
    assert torch.equal(default_result.peak_to_peak_ppm, explicit_result.peak_to_peak_ppm)
    assert torch.equal(default_result.rms_ppm, explicit_result.rms_ppm)


def test_probe_tilt_reduces_scalar_probe_readings(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
) -> None:
    result = run_monte_carlo(
        configuration,
        points,
        num_samples=4,
        position_tolerance_m=0.0,
        rotation_tolerance_deg=0.0,
        remanence_tolerance_t=0.0,
        probe_tilt_tolerance_deg=10.0,
        seed=4,
        progress_callback=None,
    )

    assert torch.all(result.fieldstrength_mT < result.nominal["fieldstrength_mT"])


@pytest.mark.parametrize(
    ("measurement_parameters"),
    [
        {"probe_position_tolerance_m": 1e-3},
        {"measurement_device_tolerance_t": 1e-4},
    ],
)
def test_measurement_uncertainty_changes_seeded_samples(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    measurement_parameters: dict[str, float],
) -> None:
    common = {
        "configuration": configuration,
        "points": points,
        "num_samples": 3,
        "position_tolerance_m": 0.0,
        "rotation_tolerance_deg": 0.0,
        "remanence_tolerance_t": 0.0,
        "seed": 8,
        "progress_callback": None,
    }
    baseline = run_monte_carlo(**common)
    perturbed = run_monte_carlo(**common, **measurement_parameters)

    assert not torch.equal(perturbed.peak_to_peak_ppm, baseline.peak_to_peak_ppm)


@pytest.mark.parametrize(
    "parameter",
    [
        "probe_position_tolerance_m",
        "probe_tilt_tolerance_deg",
        "measurement_device_tolerance_t",
    ],
)
def test_negative_measurement_tolerance_is_rejected(
    configuration: MagnetConfiguration,
    points: torch.Tensor,
    parameter: str,
) -> None:
    with pytest.raises(ValueError, match="tolerances must be non-negative"):
        run_monte_carlo(
            configuration,
            points,
            num_samples=1,
            position_tolerance_m=0.0,
            rotation_tolerance_deg=0.0,
            remanence_tolerance_t=0.0,
            progress_callback=None,
            **{parameter: -1.0},
        )
