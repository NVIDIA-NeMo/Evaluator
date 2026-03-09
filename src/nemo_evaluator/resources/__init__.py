"""Bundled resource files (container mappings, etc.)."""

from pathlib import Path

RESOURCES_DIR = Path(__file__).parent


def containers_toml_path() -> Path:
    return RESOURCES_DIR / "containers.toml"
