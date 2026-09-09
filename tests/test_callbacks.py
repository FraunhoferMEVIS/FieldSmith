from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import fieldsmith.callbacks as callbacks
from fieldsmith.callbacks import BestStateTracker, GeometryRefresh, PeriodicTruthEvaluation


def test_periodic_truth_evaluation_skips_epochs_outside_cadence() -> None:
    evaluation = PeriodicTruthEvaluation(torch.zeros((1, 3)), every=3)
    model = Mock()

    evaluation(epoch=0, model=model)

    model.get_configuration.assert_not_called()
    assert evaluation.latest is None
    assert evaluation.history == []


def test_periodic_truth_evaluation_records_demagnetized_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.zeros((2, 3), dtype=torch.float64)
    configuration = Mock()
    configuration.metadata = {}
    configuration.to.return_value = configuration
    model = Mock()
    model.get_configuration.return_value = configuration
    model.magnet_size = 0.012

    def fake_truth_eval_report(
        evaluated_configuration: object,
        evaluated_points: torch.Tensor,
        chi_par: float,
        chi_perp: float,
        chunk_points: int,
        frame_rule: str,
    ) -> dict[str, object]:
        assert not torch.is_grad_enabled()
        assert evaluated_configuration is configuration
        assert evaluated_points is points
        assert chi_par == 0.04
        assert chi_perp == 0.08
        assert chunk_points == 128
        assert frame_rule == "argmin"
        return {
            "demag": {
                "peak_to_peak_ppm": 321.5,
                "fieldstrength_mT": 42.25,
            },
            "overlaps": 0,
        }

    monkeypatch.setattr(callbacks, "truth_eval_report", fake_truth_eval_report)
    evaluation = PeriodicTruthEvaluation(
        points,
        every=2,
        frame_rule="argmin",
        chi_par=0.04,
        chi_perp=0.08,
        chunk_points=128,
    )

    evaluation(epoch=1, model=model, ignored_argument=True)

    model.get_configuration.assert_called_once_with()
    configuration.to.assert_called_once_with(points.device, points.dtype)
    assert configuration.metadata["magnet_size_m"] == 0.012
    assert evaluation.latest == {
        "epoch": 2,
        "ppm": 321.5,
        "field_mT": 42.25,
        "overlaps": 0,
    }
    assert evaluation.history == [evaluation.latest]


def test_periodic_truth_evaluation_preserves_existing_magnet_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.zeros((1, 3))
    configuration = Mock()
    configuration.metadata = {"magnet_size_m": 0.02}
    configuration.to.return_value = configuration
    model = Mock()
    model.get_configuration.return_value = configuration
    model.magnet_size = 0.012
    monkeypatch.setattr(
        callbacks,
        "truth_eval_report",
        Mock(
            return_value={
                "demag": {
                    "peak_to_peak_ppm": 10.0,
                    "fieldstrength_mT": 20.0,
                },
                "overlaps": 0,
            }
        ),
    )

    PeriodicTruthEvaluation(points, every=1)(epoch=0, model=model)

    assert configuration.metadata["magnet_size_m"] == 0.02


@pytest.mark.parametrize(
    "latest",
    [
        None,
        {"epoch": 3, "ppm": 100.0, "field_mT": 40.0, "overlaps": 0},
        {"epoch": 4, "ppm": 100.0, "field_mT": 40.0, "overlaps": 2},
    ],
)
def test_best_state_tracker_skips_ineligible_evaluations(
    latest: dict[str, float] | None,
) -> None:
    evaluation = SimpleNamespace(latest=latest)
    tracker = BestStateTracker(evaluation)
    model = Mock()

    tracker(epoch=3, model=model)

    model.get_configuration.assert_not_called()
    assert tracker.best is None
    assert tracker.best_configuration is None


def test_best_state_tracker_keeps_best_buildable_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = {"epoch": 5, "ppm": 125.0, "field_mT": 39.5, "overlaps": 0}
    evaluation = SimpleNamespace(latest=latest)
    detached_configuration = Mock()
    detached_configuration.metadata = {}
    configuration = Mock()
    configuration.detached_cpu.return_value = detached_configuration
    model = Mock()
    model.get_configuration.return_value = configuration
    model.magnet_size = 0.014
    save = Mock()
    monkeypatch.setattr(callbacks, "save_halbach_setup_npy", save)
    output_path = tmp_path / "best.npy"
    tracker = BestStateTracker(evaluation, output_path)

    tracker(epoch=4, model=model)

    assert tracker.best == latest
    assert tracker.best is not latest
    assert tracker.best_configuration is detached_configuration
    assert detached_configuration.metadata["magnet_size_m"] == 0.014
    save.assert_called_once_with(output_path, detached_configuration)

    evaluation.latest = {
        "epoch": 6,
        "ppm": 125.0,
        "field_mT": 40.0,
        "overlaps": 0,
    }
    tracker(epoch=5, model=model)

    assert model.get_configuration.call_count == 1
    assert tracker.best == latest
    save.assert_called_once()


def test_best_state_tracker_can_accept_overlapping_configuration() -> None:
    latest = {"epoch": 1, "ppm": 200.0, "field_mT": 35.0, "overlaps": 3}
    evaluation = SimpleNamespace(latest=latest)
    detached_configuration = Mock()
    detached_configuration.metadata = {"magnet_size_m": 0.01}
    configuration = Mock()
    configuration.detached_cpu.return_value = detached_configuration
    model = Mock()
    model.get_configuration.return_value = configuration
    tracker = BestStateTracker(evaluation, require_buildable=False)

    tracker(epoch=0, model=model)

    assert tracker.best == latest
    assert tracker.best_configuration is detached_configuration


def test_geometry_refresh_skips_epochs_outside_cadence() -> None:
    pair_builder = Mock()
    refresh = GeometryRefresh(every=3, pair_builder=pair_builder)
    model = Mock()

    refresh(epoch=0, model=model)

    model.bake_positions.assert_not_called()
    model.refresh_demagnetisation.assert_not_called()
    pair_builder.assert_not_called()
    assert refresh.baked == 0


def test_geometry_refresh_updates_available_geometry_state() -> None:
    pair_builder = Mock()
    refresh = GeometryRefresh(every=2, pair_builder=pair_builder)
    model = Mock()

    refresh(epoch=1, model=model, ignored_argument=True)
    refresh(epoch=3, model=model)

    assert model.bake_positions.call_count == 2
    assert model.refresh_demagnetisation.call_count == 2
    assert pair_builder.call_count == 2
    pair_builder.assert_called_with(model)
    assert refresh.baked == 2


def test_geometry_refresh_supports_models_without_optional_methods() -> None:
    pair_builder = Mock()
    refresh = GeometryRefresh(every=1, pair_builder=pair_builder)
    model = object()

    refresh(epoch=0, model=model)

    pair_builder.assert_called_once_with(model)
    assert refresh.baked == 0


def test_geometry_refresh_does_not_require_pair_builder() -> None:
    model = Mock()
    refresh = GeometryRefresh(every=1)

    refresh(epoch=0, model=model)

    model.bake_positions.assert_called_once_with()
    model.refresh_demagnetisation.assert_called_once_with()
    assert refresh.baked == 1
