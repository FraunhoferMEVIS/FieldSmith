from pathlib import Path

import pytest
import torch

from fieldsmith.geometry.roma import RomaGeometry, construct_roma


def write_ring_files(
    directory: Path,
    rows: list[str] | None = None,
) -> None:
    directory.mkdir()
    for ring_index in range(9):
        row = (
            rows[ring_index]
            if rows is not None
            else f"{10 * (ring_index + 1)},0,{ring_index + 1},{10 * ring_index}\n"
        )
        (directory / f"ring {ring_index}.txt").write_text(row, encoding="utf-8")


def test_read_ring_converts_units_and_preserves_angles(tmp_path: Path) -> None:
    ring_path = tmp_path / "ring.txt"
    ring_path.write_text(
        "10,-20,30,45\n-5,2.5,0,-90\n",
        encoding="utf-8",
    )

    positions, angles = RomaGeometry._read_ring(ring_path)

    assert torch.allclose(
        torch.tensor(positions, dtype=torch.float64),
        torch.tensor(
            [[0.010, -0.020, 0.030], [-0.005, 0.0025, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert angles == pytest.approx([45.0, -90.0])


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("1,2,3\n", "must contain X, Y, Z, Angle"),
        ("1,2,3,not-a-number\n", "contains a non-numeric value"),
        ("", "contains no magnets"),
    ],
)
def test_read_ring_rejects_invalid_files(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    ring_path = tmp_path / "invalid.txt"
    ring_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        RomaGeometry._read_ring(ring_path)


def test_ring_files_reports_every_missing_file(tmp_path: Path) -> None:
    ring_directory = tmp_path / "rings"
    ring_directory.mkdir()
    (ring_directory / "ring 0.txt").write_text("1,2,3,4\n", encoding="utf-8")
    geometry = RomaGeometry(
        ring_directory=ring_directory,
        remanence=1.3,
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="Missing ROMA ring files") as error:
        geometry._ring_files()

    assert "ring 1.txt" in str(error.value)
    assert "ring 8.txt" in str(error.value)
    assert "ring 0.txt" not in str(error.value)


def test_build_mirrors_each_source_ring_and_uses_file_angles(
    tmp_path: Path,
) -> None:
    ring_directory = tmp_path / "rings"
    write_ring_files(ring_directory)
    geometry = RomaGeometry(
        ring_directory=ring_directory,
        remanence=1.3,
        device="cpu",
        dtype=torch.float64,
        magnet_size=0.012,
        outer_magnet_length=0.05,
    )

    configuration = geometry.build()

    assert configuration.num_magnets == 18
    assert configuration.positions.dtype == torch.float64
    for ring_index in range(9):
        negative_index = 2 * ring_index
        positive_index = negative_index + 1
        expected_x = 0.01 * (ring_index + 1)
        expected_z = 0.001 * (ring_index + 1)
        expected_angle = torch.deg2rad(torch.tensor(10.0 * ring_index, dtype=torch.float64))
        expected_orientation = torch.stack(
            (
                torch.cos(expected_angle),
                torch.sin(expected_angle),
                torch.zeros((), dtype=torch.float64),
            ),
        )

        assert torch.allclose(
            configuration.positions[negative_index],
            torch.tensor([expected_x, 0.0, -expected_z], dtype=torch.float64),
        )
        assert torch.allclose(
            configuration.positions[positive_index],
            torch.tensor([expected_x, 0.0, expected_z], dtype=torch.float64),
        )
        assert torch.allclose(configuration.orientations[negative_index], expected_orientation)
        assert torch.allclose(configuration.orientations[positive_index], expected_orientation)


def test_build_uses_long_outer_magnets_and_records_metadata(
    tmp_path: Path,
) -> None:
    ring_directory = tmp_path / "rings"
    write_ring_files(ring_directory)
    geometry = RomaGeometry(
        ring_directory=ring_directory,
        remanence=1.25,
        device="cpu",
        dtype=torch.float64,
        magnet_size=0.012,
        outer_magnet_length=0.05,
    )

    configuration = geometry.build()
    metadata = configuration.metadata
    dimensions = metadata["magnet_dimensions_m"]

    assert torch.allclose(
        configuration.volumes[:16],
        torch.full((16,), 0.012**3, dtype=torch.float64),
    )
    assert torch.allclose(
        configuration.volumes[16:],
        torch.full((2,), 0.012**2 * 0.05, dtype=torch.float64),
    )
    assert dimensions.shape == (18, 3)
    assert torch.allclose(
        dimensions[:16],
        torch.tensor([0.012, 0.012, 0.012], dtype=torch.float64).expand(16, 3),
    )
    assert torch.allclose(
        dimensions[16:],
        torch.tensor([0.012, 0.012, 0.05], dtype=torch.float64).expand(2, 3),
    )
    assert metadata["geometry"] == "roma"
    assert metadata["n_magnets"] == 18
    assert metadata["n_source_rings"] == 9
    assert metadata["n_rings"] == 18
    assert metadata["mirrored_about_z"] is True
    assert metadata["ring_directory"] == str(ring_directory)
    assert torch.equal(metadata["ring_indices"], torch.arange(9).repeat_interleave(2))
    assert metadata["magnet_size_m"] == pytest.approx(0.012)
    assert metadata["outer_magnet_length_m"] == pytest.approx(0.05)
    assert metadata["use_halbach_orientations"] is False
    assert configuration.remanence == pytest.approx(1.25)


def test_build_can_derive_halbach_orientations_from_positions(
    tmp_path: Path,
) -> None:
    ring_directory = tmp_path / "rings"
    write_ring_files(ring_directory, ["10,10,5,123\n"] * 9)
    geometry = RomaGeometry(
        ring_directory=ring_directory,
        remanence=1.3,
        device="cpu",
        dtype=torch.float64,
        use_halbach_orientations=True,
    )

    configuration = geometry.build()

    expected_orientation = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert torch.allclose(
        configuration.orientations,
        expected_orientation.expand(18, 3),
        atol=1e-12,
    )
    assert torch.allclose(
        configuration.metadata["initial_magnet_angles_rad"],
        torch.full((18,), torch.pi / 2.0, dtype=torch.float64),
    )
    assert configuration.metadata["use_halbach_orientations"] is True


def test_construct_roma_matches_geometry_builder(tmp_path: Path) -> None:
    ring_directory = tmp_path / "rings"
    write_ring_files(ring_directory)
    parameters = {
        "ring_directory": ring_directory,
        "remanence": 1.3,
        "device": "cpu",
        "dtype": torch.float64,
        "magnet_size": 0.012,
        "outer_magnet_length": 0.05,
        "use_halbach_orientations": False,
    }

    configuration = RomaGeometry(**parameters).build()
    positions, remanence_vectors, volumes, angles = construct_roma(**parameters)

    assert torch.equal(positions, configuration.positions)
    assert torch.equal(remanence_vectors, configuration.remanence_vectors())
    assert torch.equal(volumes, configuration.volumes)
    assert torch.equal(angles, configuration.metadata["initial_magnet_angles_rad"])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"remanence": 0.0}, "remanence and magnet dimensions must be positive"),
        ({"magnet_size": 0.0}, "remanence and magnet dimensions must be positive"),
        ({"outer_magnet_length": 0.0}, "remanence and magnet dimensions must be positive"),
    ],
)
def test_constructor_rejects_nonpositive_physical_values(
    tmp_path: Path,
    changes: dict[str, float],
    message: str,
) -> None:
    parameters = {
        "ring_directory": tmp_path,
        "remanence": 1.3,
        "device": "cpu",
        "magnet_size": 0.012,
        "outer_magnet_length": 0.05,
    }
    parameters.update(changes)

    with pytest.raises(ValueError, match=message):
        RomaGeometry(**parameters)
