# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Tests for the legacy optimized-setup .npy loader/saver.
#
#  test_setup_npy.py
#  Marian Frei
#  24.07.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fieldsmith.configurations import MagnetConfiguration
from fieldsmith.configurations.setup_npy import load_halbach_setup_npy, save_halbach_setup_npy

DATA = Path(__file__).resolve().parents[1] / "data" / "paper"


@pytest.fixture
def legacy_dome(tmp_path) -> Path:
    """The paper's dome written in the legacy .npy format."""
    path = tmp_path / "dome.npy"
    save_halbach_setup_npy(path, MagnetConfiguration.from_file(DATA / "dome_980ppm.pt"))
    return path


def test_loads_a_legacy_design(legacy_dome) -> None:
    config = load_halbach_setup_npy(legacy_dome)
    assert config.num_magnets == 3987
    assert float(config.remanence) == pytest.approx(1.31)
    assert config.metadata["magnet_size_m"] == pytest.approx(0.012)
    norms = config.orientations.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)
    assert torch.allclose(config.volumes, torch.full_like(config.volumes, 0.012**3))


def test_round_trips_through_save(legacy_dome, tmp_path) -> None:
    config = load_halbach_setup_npy(legacy_dome)
    out = tmp_path / "round.npy"
    save_halbach_setup_npy(out, config)
    again = load_halbach_setup_npy(out)
    assert torch.allclose(again.positions, config.positions, atol=1e-12)
    # orientations reconstruct through angles, so a few ulp apart, not more.
    assert torch.allclose(again.orientations, config.orientations, atol=1e-9)


def test_keeps_the_body_dimensions(tmp_path) -> None:
    config = MagnetConfiguration.from_file(DATA / "roma_refined_332ppm.pt")
    out = tmp_path / "roma.npy"
    save_halbach_setup_npy(out, config)
    again = load_halbach_setup_npy(out)
    assert torch.equal(again.metadata["magnet_dimensions_m"], config.metadata["magnet_dimensions_m"])
    assert torch.equal(again.volumes, config.volumes)


def test_rejects_non_unit_amplitudes(tmp_path) -> None:
    import numpy as np

    out = tmp_path / "amp.npy"
    np.save(out, {"magnet_positions": np.zeros((2, 3)), "magnet_angles_theta": np.zeros(2),
                  "magnet_angles_phi": np.zeros(2), "remanence": 1.31,
                  "magnet_amplitudes": np.array([1.0, 0.5])}, allow_pickle=True)
    with pytest.raises(ValueError, match="amplitudes"):
        load_halbach_setup_npy(out)
