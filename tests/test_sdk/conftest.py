from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_CATEGORY_MARKERS = {
    "unit": "Unit tests for the SDK package.",
    "e2e": "End-to-end tests.",
    "integration": "Integration tests.",
    "regression": "Regression tests.",
    "canary": "Canary tests.",
    "slow": "Slow-running tests.",
    "skip_in_ci": "Tests skipped in CI environments.",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker_name, description in _CATEGORY_MARKERS.items():
        config.addinivalue_line("markers", f"{marker_name}: {description}")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        # Pytest passes the full repo item list here once this conftest is loaded,
        # so restrict the auto-unit marker to the SDK's own tests.
        if item_path != _TESTS_DIR and _TESTS_DIR not in item_path.parents:
            continue

        marker_names = {marker.name for marker in item.iter_markers()}
        if not marker_names.intersection(_CATEGORY_MARKERS):
            item.add_marker(pytest.mark.unit)
