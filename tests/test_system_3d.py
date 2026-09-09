import builtins
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import fieldsmith.visualizations.plotly_3d as plotly_module
import fieldsmith.visualizations.system_3d as system_3d_module
from fieldsmith.visualizations.system_3d import (
    _plot_magnets_matplotlib,
    _to_numpy,
    plot_magnets_and_field_3d,
    plot_model_output_3d,
)


def test_to_numpy_detaches_tensors_and_preserves_other_values() -> None:
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    value = {"unchanged": True}

    converted = _to_numpy(tensor)

    assert isinstance(converted, np.ndarray)
    assert np.array_equal(converted, [1.0, 2.0])
    assert _to_numpy(value) is value


def test_plot_magnets_matplotlib_creates_labeled_3d_figure() -> None:
    positions = torch.tensor([[-0.1, 0.0, -0.02], [0.1, 0.0, 0.05]])
    orientations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    figure = _plot_magnets_matplotlib(
        positions,
        orientations,
        cube_size=torch.tensor([0.01, 0.02]),
        title="Fallback",
    )
    axis = figure.axes[0]

    assert axis.name == "3d"
    assert axis.get_xlabel() == "x [m]"
    assert axis.get_ylabel() == "y [m]"
    assert axis.get_zlabel() == "z [m]"
    assert axis.get_title() == "Fallback"
    assert axis.get_zlim()[0] == pytest.approx(-0.02)
    plt.close(figure)


def test_plot_magnets_and_field_uses_plotly_and_converts_tesla_to_millitesla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    sentinel = object()

    def fake_plotly(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(plotly_module, "plot_magnets_and_field_3d_plotly", fake_plotly)
    grid = torch.tensor([[0.0, 0.0, 0.0]])
    field = torch.tensor([[0.001, 0.002, 0.003]])
    positions = torch.tensor([[0.1, 0.0, 0.0]])
    orientations = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([True])

    result = plot_magnets_and_field_3d(
        grid,
        field,
        positions,
        orientations,
        cube_size=torch.tensor([0.01, 0.02, 0.03]),
        mask=mask,
        symmetry_type="octant",
        title="System",
    )

    assert result is sentinel
    assert np.array_equal(captured["voxel_grid"], grid.numpy())
    assert np.array_equal(captured["B"], field.numpy() * 1e3)
    assert np.array_equal(captured["magnet_positions"], positions.numpy())
    assert np.array_equal(captured["magnet_orientations"], orientations.numpy())
    assert np.array_equal(captured["mask"], mask.numpy())
    assert captured["center"] == [0.0, 0.0, 0.0]
    assert captured["title"] == "System"


def test_plot_magnets_and_field_falls_back_when_plotly_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_plotly(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "fieldsmith.visualizations.plotly_3d":
            error = ModuleNotFoundError("No module named 'plotly'")
            error.name = "plotly"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_plotly)

    figure = plot_magnets_and_field_3d(
        voxel_grid=torch.zeros((1, 3)),
        field=torch.zeros((1, 3)),
        magnet_positions=torch.tensor([[0.1, 0.0, 0.0]]),
        magnet_orientations=torch.tensor([[1.0, 0.0, 0.0]]),
        cube_size=0.01,
        title="Fallback",
    )

    assert isinstance(figure, plt.Figure)
    assert figure.axes[0].get_title() == "Fallback"
    plt.close(figure)


def test_plot_magnets_and_field_reraises_unrelated_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "fieldsmith.visualizations.plotly_3d":
            error = ModuleNotFoundError("No module named 'other_dependency'")
            error.name = "other_dependency"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(ModuleNotFoundError, match="other_dependency"):
        plot_magnets_and_field_3d(
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.tensor([[0.1, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            cube_size=0.01,
        )


def test_plot_model_output_prefers_positions_and_magnet_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    sentinel = object()

    def fake_system_plot(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(system_3d_module, "plot_magnets_and_field_3d", fake_system_plot)
    output = {
        "positions": torch.ones((2, 3)),
        "points": torch.zeros((2, 3)),
        "B": torch.full((2, 3), 2.0),
        "magnet_positions": torch.full((1, 3), 3.0),
        "magnet_vectors": torch.full((1, 3), 4.0),
        "magnet_orientations": torch.full((1, 3), 5.0),
    }

    result = plot_model_output_3d(output, cube_size=0.01, title="Model")

    assert result is sentinel
    assert captured["voxel_grid"] is output["positions"]
    assert captured["magnet_orientations"] is output["magnet_vectors"]
    assert captured["title"] == "Model"


def test_plot_model_output_uses_points_and_orientations_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_system_plot(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(system_3d_module, "plot_magnets_and_field_3d", fake_system_plot)
    output = {
        "points": torch.zeros((2, 3)),
        "B": torch.zeros((2, 3)),
        "magnet_positions": torch.zeros((1, 3)),
        "magnet_orientations": torch.ones((1, 3)),
    }

    plot_model_output_3d(output, cube_size=0.01)

    assert captured["voxel_grid"] is output["points"]
    assert captured["magnet_orientations"] is output["magnet_orientations"]


def test_plot_model_output_requires_field_points() -> None:
    with pytest.raises(ValueError, match="must contain 'positions' or 'points'"):
        plot_model_output_3d(
            {
                "B": torch.zeros((1, 3)),
                "magnet_positions": torch.zeros((1, 3)),
                "magnet_vectors": torch.ones((1, 3)),
            },
            cube_size=0.01,
        )
