"""Version information for nemo-evaluator-launcher."""

import sys
from pathlib import Path

# Conditional import for TOML parsing
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.logging_utils import logger

# Get the path to pyproject.toml relative to this file
_pyproject_path = Path(__file__).parents[3] / "pyproject.toml"


def _get_name() -> str:
    """Read version from pyproject.toml.

    Raises:
        FileNotFoundError, KeyError, tomllib.TOMLDecodeError: if pyproject can't be found
           this fails fast if the structure has changed
    """
    with open(_pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["name"]


def _get_version() -> str:
    """Read version from pyproject.toml.

    Raises:
        FileNotFoundError, KeyError, tomllib.TOMLDecodeError: if pyproject can't be found
           this fails fast if the structure has changed
    """
    with open(_pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]


__main_pkg_name__ = _get_name()
__version__ = _get_version()

logger.info("Version info", pkg=__main_pkg_name__, ver=__version__)
