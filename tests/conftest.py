import os
from pathlib import Path

import pytest


os.environ.setdefault("MPLBACKEND", "Agg")

VALIDATION_TESTS = Path(__file__).parent / "validation"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests by their suite directory."""
    for item in items:
        if Path(item.path).is_relative_to(VALIDATION_TESTS):
            item.add_marker(pytest.mark.validation)
        else:
            item.add_marker(pytest.mark.unit)
