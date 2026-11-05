# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# -----------------------------------------------------------------------------
# Tests for exporting a design as printable plates, lids and rails.
#
#  test_cad_export.py
#  Marian Frei
#  16.10.2026
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import trimesh

from fieldsmith.cad.lids import export_lids
from fieldsmith.cad.plates import assign_planes
from fieldsmith.cad.rails import add_rails
from fieldsmith.cad.slabs import export_slabs
from fieldsmith.configurations import MagnetConfiguration

DOME = Path(__file__).resolve().parents[1] / "data" / "paper" / "dome_980ppm.pt"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """The top two plates of the paper's dome, exported with lids and rails."""
    design = MagnetConfiguration.from_file(DOME)
    labels, _ = assign_planes(design.positions)
    keep = labels >= labels.max() - 1
    top = MagnetConfiguration(positions=design.positions[keep], orientations=design.orientations[keep],
                              volumes=design.volumes[keep], remanence=design.remanence,
                              metadata={"magnet_size_m": 0.012})
    root = tmp_path_factory.mktemp("export")
    torch.save(top.to_dict(), root / "top.pt")
    out = root / "plates"
    export_slabs(top, out, variant="top", lid_gap=2.0e-3)
    export_lids(top, out, variant="top")
    add_rails(root / "top.pt", out)
    return top, out


def test_every_level_becomes_one_watertight_plate(exported) -> None:
    top, out = exported
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_magnets"] == top.num_magnets
    assert manifest["clearance_mm"] == pytest.approx(0.1) and manifest["lid_gap_mm"] == pytest.approx(2.0)
    assert len(manifest["slabs"]) == 2
    for slab in manifest["slabs"]:
        assert trimesh.load(out / slab["file"]).is_watertight


def test_each_plate_gets_a_lid_and_the_stack_gets_rails(exported) -> None:
    _, out = exported
    lids = json.loads((out / "lids" / "manifest.json").read_text())["lids"]
    assert len(lids) == 2
    plan = json.loads((out / "fastener_plan.json").read_text())
    assert plan["rails"] == 6 and len(plan["interfaces"]) == 1
