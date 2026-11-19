import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_SCRIPTS = (
    Path("run_fast_halbach_ring_optimisation.py"),
    Path("run_small_halbach_optimisation.py"),
    Path("run_fast_roma_optimisation.py"),
    Path("lumc") / "homogeneityOptimization.py",
    Path("lumc") / "run_lumc_radius_differential_evolution.py",
    Path("lumc") / "run_lumc_radius_optimisation.py",
)


@pytest.mark.parametrize("relative_path", PAPER_SCRIPTS, ids=str)
def test_paper_script_help_from_another_directory(
    relative_path: Path,
    tmp_path: Path,
) -> None:
    script = REPOSITORY_ROOT / "scripts" / "paper" / relative_path
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("relative_path", PAPER_SCRIPTS, ids=str)
def test_paper_script_resolves_repository_paths(
    relative_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = REPOSITORY_ROOT / "scripts" / "paper" / relative_path
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(script.parent))
    namespace = runpy.run_path(str(script))

    assert Path(sys.path[0]) == REPOSITORY_ROOT / "src"
    if relative_path.name == "run_fast_roma_optimisation.py":
        monkeypatch.setattr(sys, "argv", [str(script)])
        args = namespace["parse_args"]()
        assert args.ring_directory == REPOSITORY_ROOT / "data" / "osii_rc2.1"
        assert args.ring_directory.is_dir()
