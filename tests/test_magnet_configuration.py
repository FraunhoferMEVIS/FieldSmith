from pathlib import Path
from typing import Any

import pytest
import torch

from fieldsmith.configurations import CONFIGURATION_FORMAT_VERSION, MagnetConfiguration


@pytest.fixture
def configuration() -> MagnetConfiguration:
    return MagnetConfiguration(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        orientations=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        volumes=torch.tensor([1.0, 2.0], dtype=torch.float32),
        remanence=1.3,
        metadata={"name": "test configuration"},
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"positions": torch.zeros(3)}, r"positions must have shape \(M, 3\)"),
        ({"positions": torch.zeros((2, 2))}, r"positions must have shape \(M, 3\)"),
        ({"orientations": torch.zeros(3)}, r"orientations must have shape \(M, 3\)"),
        ({"orientations": torch.zeros((2, 2))}, r"orientations must have shape \(M, 3\)"),
        (
            {"orientations": torch.zeros((3, 3))},
            "positions and orientations must contain the same number of magnets",
        ),
        ({"volumes": torch.zeros((2, 1))}, r"volumes must have shape \(M,\)"),
        ({"volumes": torch.zeros(3)}, r"volumes must have shape \(M,\)"),
    ],
)
def test_initialization_rejects_invalid_shapes(
    configuration: MagnetConfiguration,
    changes: dict[str, Any],
    message: str,
) -> None:
    values = {
        "positions": configuration.positions,
        "orientations": configuration.orientations,
        "volumes": configuration.volumes,
        "remanence": configuration.remanence,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        MagnetConfiguration(**values)


def test_num_magnets_returns_position_count(configuration: MagnetConfiguration) -> None:
    assert configuration.num_magnets == 2


def test_remanence_vectors_expands_float(configuration: MagnetConfiguration) -> None:
    expected = configuration.orientations * 1.3

    assert torch.allclose(configuration.remanence_vectors(), expected)


@pytest.mark.parametrize(
    ("remanence", "expected"),
    [
        (
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        ),
        (
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ),
        (
            torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                dtype=torch.float64,
            ),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        ),
    ],
)
def test_remanence_vectors_accepts_supported_tensor_shapes(
    configuration: MagnetConfiguration,
    remanence: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    configuration.remanence = remanence

    result = configuration.remanence_vectors()

    assert result.dtype == configuration.orientations.dtype
    assert result.device == configuration.orientations.device
    assert torch.allclose(result, expected)


def test_remanence_vectors_rejects_invalid_tensor_shape(
    configuration: MagnetConfiguration,
) -> None:
    configuration.remanence = torch.ones((2, 2))

    with pytest.raises(
        ValueError,
        match=r"remanence tensor must be scalar, shape \(M,\), or shape \(M, 3\)",
    ):
        configuration.remanence_vectors()


@pytest.mark.parametrize("tensor_remanence", [False, True])
def test_to_converts_tensor_fields_and_copies_metadata(
    configuration: MagnetConfiguration,
    tensor_remanence: bool,
) -> None:
    if tensor_remanence:
        configuration.remanence = torch.tensor([1.2, 1.4], dtype=torch.float32)

    converted = configuration.to("cpu", dtype=torch.float64)

    assert converted is not configuration
    assert converted.positions.dtype == torch.float64
    assert converted.orientations.dtype == torch.float64
    assert converted.volumes.dtype == torch.float64
    if tensor_remanence:
        assert isinstance(converted.remanence, torch.Tensor)
        assert converted.remanence.dtype == torch.float64
    else:
        assert converted.remanence == configuration.remanence
    assert converted.metadata == configuration.metadata
    assert converted.metadata is not configuration.metadata


@pytest.mark.parametrize("tensor_remanence", [False, True])
def test_detached_cpu_removes_gradients_and_copies_metadata(
    configuration: MagnetConfiguration,
    tensor_remanence: bool,
) -> None:
    configuration.positions.requires_grad_()
    configuration.orientations.requires_grad_()
    configuration.volumes.requires_grad_()
    if tensor_remanence:
        configuration.remanence = torch.tensor([1.2, 1.4], requires_grad=True)

    detached = configuration.detached_cpu()

    assert detached.positions.device.type == "cpu"
    assert detached.orientations.device.type == "cpu"
    assert detached.volumes.device.type == "cpu"
    assert not detached.positions.requires_grad
    assert not detached.orientations.requires_grad
    assert not detached.volumes.requires_grad
    if tensor_remanence:
        assert isinstance(detached.remanence, torch.Tensor)
        assert not detached.remanence.requires_grad
    else:
        assert detached.remanence == configuration.remanence
    assert detached.metadata == configuration.metadata
    assert detached.metadata is not configuration.metadata


def test_from_file_loads_saved_configuration(
    tmp_path: Path,
    configuration: MagnetConfiguration,
) -> None:
    checkpoint_path = tmp_path / "configuration.pt"
    torch.save(configuration, checkpoint_path)

    loaded = MagnetConfiguration.from_file(checkpoint_path, dtype=torch.float64)

    assert loaded.positions.dtype == torch.float64
    assert torch.equal(loaded.positions, configuration.positions.to(torch.float64))
    assert loaded.metadata == configuration.metadata
    assert loaded.metadata is not configuration.metadata


def test_from_file_loads_checkpoint_dictionary(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "configuration.pt"
    torch.save(
        {
            "format_version": CONFIGURATION_FORMAT_VERSION,
            "magnet_positions_m": torch.zeros((2, 3), dtype=torch.float32),
            "magnet_orientations": torch.ones((2, 3), dtype=torch.float32),
            "volumes_m3": torch.ones(2, dtype=torch.float32),
            "remanence_t": torch.tensor([1.2, 1.3], dtype=torch.float32),
            "metadata": {"stage": "angle"},
        },
        checkpoint_path,
    )

    loaded = MagnetConfiguration.from_file(str(checkpoint_path), dtype=torch.float64)

    assert loaded.positions.dtype == torch.float64
    assert isinstance(loaded.remanence, torch.Tensor)
    assert loaded.remanence.dtype == torch.float64
    assert loaded.metadata == {"stage": "angle"}


def test_from_file_loads_legacy_unversioned_dictionary(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "configuration.pt"
    torch.save(
        {
            "magnet_positions_m": torch.zeros((1, 3)),
            "magnet_orientations": torch.ones((1, 3)),
            "volumes_m3": torch.ones(1),
            "remanence_t": 1.3,
        },
        checkpoint_path,
    )

    loaded = MagnetConfiguration.from_file(checkpoint_path)

    assert loaded.num_magnets == 1


def test_from_file_rejects_unsupported_format_version(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "configuration.pt"
    torch.save(
        {
            "format_version": "1.0.0",
            "magnet_positions_m": torch.zeros((1, 3)),
            "magnet_orientations": torch.ones((1, 3)),
            "volumes_m3": torch.ones(1),
            "remanence_t": 1.3,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="Unsupported magnet configuration format version"):
        MagnetConfiguration.from_file(checkpoint_path)


def test_from_file_rejects_missing_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Magnet configuration not found"):
        MagnetConfiguration.from_file(checkpoint_path)


def test_from_file_rejects_non_dictionary_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save(["not", "a", "configuration"], checkpoint_path)

    with pytest.raises(ValueError, match="expected a dictionary"):
        MagnetConfiguration.from_file(checkpoint_path)


def test_from_file_reports_all_missing_keys(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incomplete.pt"
    torch.save({"magnet_positions_m": torch.zeros((1, 3))}, checkpoint_path)

    with pytest.raises(ValueError) as error:
        MagnetConfiguration.from_file(checkpoint_path)

    assert str(error.value).endswith(
        "missing magnet_orientations, remanence_t, volumes_m3."
    )


def test_from_file_rejects_non_dictionary_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid_metadata.pt"
    torch.save(
        {
            "magnet_positions_m": torch.zeros((1, 3)),
            "magnet_orientations": torch.ones((1, 3)),
            "volumes_m3": torch.ones(1),
            "remanence_t": 1.3,
            "metadata": "invalid",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        MagnetConfiguration.from_file(checkpoint_path)


def test_to_dict_detaches_tensors_and_includes_metadata(
    configuration: MagnetConfiguration,
) -> None:
    configuration.positions.requires_grad_()

    data = configuration.to_dict()

    assert data["format_version"] == CONFIGURATION_FORMAT_VERSION
    assert data["magnet_positions_m"].device.type == "cpu"
    assert not data["magnet_positions_m"].requires_grad
    assert torch.equal(data["magnet_orientations"], configuration.orientations)
    assert torch.equal(data["volumes_m3"], configuration.volumes)
    assert data["remanence_t"] == 1.3
    assert data["metadata"] == configuration.metadata
    assert "name" not in data
