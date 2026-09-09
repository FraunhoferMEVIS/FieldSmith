import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
import torch

from fieldsmith.logger import WandbLogger


class FakeRun:
    def __init__(self) -> None:
        self.artifacts: list[FakeArtifact] = []
        self.finished: bool = False

    def log_artifact(self, artifact: "FakeArtifact") -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name: str = name
        self.type: str = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeWandb(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run: FakeRun = FakeRun()
        self.init_kwargs: dict[str, Any] = {}
        self.log_calls: list[tuple[dict[str, Any], int]] = []

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run

    def log(self, values: dict[str, Any], step: int) -> None:
        self.log_calls.append((values, step))

    def Image(self, figure: object) -> tuple[str, object]:
        return ("image", figure)

    def Plotly(self, figure: object) -> tuple[str, object]:
        return ("plotly", figure)

    def Artifact(self, name: str, type: str) -> FakeArtifact:
        return FakeArtifact(name, type)


class SerializableConfiguration:
    def to_dict(self) -> dict[str, int]:
        return {"value": 42}


def test_disabled_logger_creates_directories_and_saves_configuration(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "run"
    logger = WandbLogger(
        project="fieldsmith",
        run_name="test",
        log_dir=log_dir,
        mode="disabled",
    )

    logger.save_configuration(SerializableConfiguration(), epoch=3)
    saved = torch.load(
        log_dir / "configurations" / "epoch_3.pt",
        weights_only=False,
    )

    assert log_dir.is_dir()
    assert (log_dir / "configurations").is_dir()
    assert saved == {"value": 42}
    assert logger.run is None
    assert logger.wandb is None


def test_disabled_logger_closes_matplotlib_images_but_not_plotly(
    tmp_path: Path,
) -> None:
    logger = WandbLogger("fieldsmith", "test", tmp_path, mode="disabled")
    matplotlib_figure = plt.figure()
    plotly_figure = go.Figure()
    figure_number = matplotlib_figure.number

    logger.log_images(
        {"matplotlib": matplotlib_figure, "plotly": plotly_figure},
        epoch=0,
    )

    assert not plt.fignum_exists(figure_number)
    assert isinstance(plotly_figure, go.Figure)


def test_active_logger_initializes_and_logs_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = WandbLogger(
        project="fieldsmith",
        run_name="active",
        log_dir=tmp_path,
        config={"lr": 0.1},
        mode="offline",
    )

    logger.log_scalars(
        {"loss": torch.tensor(1.5, requires_grad=True), "epoch_time": 2.0},
        epoch=7,
    )

    assert fake_wandb.init_kwargs == {
        "project": "fieldsmith",
        "name": "active",
        "config": {"lr": 0.1},
        "dir": str(tmp_path),
        "mode": "offline",
        "reinit": True,
    }
    assert fake_wandb.log_calls == [
        ({"Scalar/loss": 1.5, "Scalar/epoch_time": 2.0}, 7),
    ]


def test_active_logger_wraps_images_and_logs_configuration_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = WandbLogger("fieldsmith", "active", tmp_path, mode="offline")
    matplotlib_figure = plt.figure()
    plotly_figure = go.Figure()
    figure_number = matplotlib_figure.number

    logger.log_images(
        {"slice": matplotlib_figure, "system": plotly_figure},
        epoch=2,
    )
    logger.save_configuration({"answer": 42}, epoch=2)

    image_log, image_step = fake_wandb.log_calls[0]
    artifact = fake_wandb.run.artifacts[0]
    assert image_step == 2
    assert image_log["Images/slice"] == ("image", matplotlib_figure)
    assert image_log["Images/system"] == ("plotly", plotly_figure)
    assert not plt.fignum_exists(figure_number)
    assert artifact.name == "configuration_epoch_2"
    assert artifact.type == "configuration"
    assert artifact.files == [
        str(tmp_path / "configurations" / "epoch_2.pt"),
    ]


def test_close_finishes_active_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = WandbLogger("fieldsmith", "active", tmp_path, mode="offline")

    logger.close()

    assert fake_wandb.run.finished


def test_wandb_initialization_failure_falls_back_to_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_wandb = FakeWandb()

    def failing_init(**kwargs: Any) -> FakeRun:
        raise RuntimeError("service unavailable")

    fake_wandb.init = failing_init
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger("fieldsmith", "failed", tmp_path, mode="online")

    assert logger.mode == "disabled"
    assert logger.run is None
    assert "wandb disabled: service unavailable" in capsys.readouterr().out
