"""NeMo Evaluator Launcher - Public API.

This package provides the public API for the NeMo Evaluator Launcher.
It automatically initializes logging and conditionally loads internal components.
"""

import importlib

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.version_utils import __version__

try:
    importlib.import_module("nemo_evaluator_launcher_internal")
    logger.debug(
        "Successfully loaded internal package",
        package="nemo_evaluator_launcher_internal",
    )
except ImportError as e:
    logger.debug(
        "Internal package not available",
        package="nemo_evaluator_launcher_internal",
        error=str(e),
    )
except Exception as e:
    logger.warning(
        "Failed to load internal package",
        package="nemo_evaluator_launcher_internal",
        error=str(e),
    )


__all__ = ["__version__"]
