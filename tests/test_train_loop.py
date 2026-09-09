from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import fieldsmith.train_loop as train_loop_module
from fieldsmith.train_loop import TrainLoop


class RecordingLogger:
    def __init__(self) -> None:
        self.scalar_epochs: list[int] = []
        self.image_epochs: list[int] = []
        self.configuration_epochs: list[int] = []
        self.closed: bool = False

    def log_scalars(self, scalars: dict[str, Any], epoch: int) -> None:
        self.scalar_epochs.append(epoch)

    def log_images(self, images: dict[str, Any], epoch: int) -> None:
        self.image_epochs.append(epoch)

    def save_configuration(self, configuration: Any, epoch: int) -> None:
        self.configuration_epochs.append(epoch)

    def close(self) -> None:
        self.closed = True


class ScalarModel(nn.Module):
    def __init__(self, initial_value: float = 0.0) -> None:
        super().__init__()
        self.value: nn.Parameter = nn.Parameter(torch.tensor(initial_value))
        self.constraint_calls: int = 0

    def forward(self) -> dict[str, torch.Tensor]:
        return {"value": self.value}

    def constrain_parameters(self) -> None:
        self.constraint_calls += 1

    def get_scalars_to_log(
        self,
        output: dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "loss": loss,
            "fieldstrength_mT": output["value"].detach().abs(),
            "homogeneity_ppm": loss.detach().abs(),
        }

    def get_images_to_log(
        self,
        output: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        return {"image": object()}

    def get_configuration(self) -> dict[str, float]:
        return {"value": float(self.value.detach())}


def make_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    logger: RecordingLogger,
    tmp_path: Path,
    **kwargs: Any,
) -> TrainLoop:
    return TrainLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=kwargs.pop("num_epochs", 1),
        logger=logger,
        log_each_n_epochs=kwargs.pop("log_each_n_epochs", 10),
        log_dir=tmp_path,
        **kwargs,
    )


def test_run_optimizes_for_all_epochs_and_logs_on_schedule(tmp_path: Path) -> None:
    model = ScalarModel()
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    callback_epochs = []

    def callback(epoch: int, model: nn.Module, loss: torch.Tensor, loop: TrainLoop) -> None:
        callback_epochs.append(epoch)
        assert model is loop.model
        assert not loss.requires_grad

    loop = make_loop(
        model,
        optimizer,
        lambda output: (output["value"] - 2.0).square(),
        logger,
        tmp_path,
        num_epochs=3,
        log_each_n_epochs=2,
        callbacks=[callback],
    )

    final_loss = loop.run()

    assert model.training
    assert model.value.item() == pytest.approx(1.75)
    assert final_loss.item() == pytest.approx(0.25)
    assert callback_epochs == [0, 1, 2]
    assert model.constraint_calls == 3
    assert logger.scalar_epochs == [0, 1, 2]
    assert logger.image_epochs == [0, 1]
    assert logger.configuration_epochs == [0, 1]
    assert logger.closed


def test_run_constrains_parameters_after_optimizer_step(tmp_path: Path) -> None:
    class ClampedModel(ScalarModel):
        def constrain_parameters(self) -> None:
            super().constrain_parameters()
            self.value.clamp_(max=1.0)

    model = ClampedModel(initial_value=0.9)
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loop = make_loop(
        model,
        optimizer,
        lambda output: -output["value"],
        logger,
        tmp_path,
    )

    loop.run()

    assert model.value.item() == pytest.approx(1.0)
    assert model.constraint_calls == 1


def test_run_applies_gradient_clipping(tmp_path: Path) -> None:
    model = ScalarModel()
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loop = make_loop(
        model,
        optimizer,
        lambda output: 100.0 * output["value"],
        logger,
        tmp_path,
        grad_clip_norm=0.5,
    )

    loop.run()

    assert model.value.item() == pytest.approx(-0.5)


def test_run_stops_when_time_limit_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    times = iter((0.0, 0.0, 0.0, 0.1, 1.1))
    monkeypatch.setattr(train_loop_module.time, "monotonic", lambda: next(times))
    model = ScalarModel()
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback_epochs = []
    loop = make_loop(
        model,
        optimizer,
        lambda output: output["value"].square(),
        logger,
        tmp_path,
        num_epochs=100,
        time_limit_seconds=1.0,
        callbacks=[lambda epoch, **_: callback_epochs.append(epoch)],
    )

    loop.run()

    assert callback_epochs == [0]
    assert "Time limit reached after 1 epochs." in capsys.readouterr().out
    assert logger.closed


def test_run_closes_logger_when_loss_raises(tmp_path: Path) -> None:
    model = ScalarModel()
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def failing_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError("loss failed")

    loop = make_loop(model, optimizer, failing_loss, logger, tmp_path)

    with pytest.raises(RuntimeError, match="loss failed"):
        loop.run()

    assert logger.closed


@pytest.mark.parametrize("time_limit", [0.0, -1.0])
def test_init_rejects_nonpositive_time_limit(
    tmp_path: Path,
    time_limit: float,
) -> None:
    model = ScalarModel()
    logger = RecordingLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="time_limit_seconds must be positive"):
        make_loop(
            model,
            optimizer,
            lambda output: output["value"].square(),
            logger,
            tmp_path,
            time_limit_seconds=time_limit,
        )
