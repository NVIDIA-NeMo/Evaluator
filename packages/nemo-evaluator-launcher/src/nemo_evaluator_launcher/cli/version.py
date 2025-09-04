"""Version command for nemo-evaluator-launcher."""

import importlib
from dataclasses import dataclass

from nemo_evaluator_launcher.common.version_utils import __main_pkg_name__, __version__


@dataclass
class Cmd:
    """Show version information for nemo-evaluator-launcher and internal packages."""

    def execute(self) -> None:
        """Execute the version command."""
        print(f"{__main_pkg_name__}: {__version__}")

        # Check for internal package
        try:
            internal_module = importlib.import_module(
                "nemo_evaluator_launcher_internal"
            )
            # Try to get version from internal package
            try:
                internal_version = getattr(internal_module, "__version__", None)
                if internal_version:
                    print(f"nemo-evaluator-launcher-internal: {internal_version}")
                else:
                    print(
                        "nemo-evaluator-launcher-internal: available (version unknown)"
                    )
            except Exception:
                print("nemo-evaluator-launcher-internal: available (version unknown)")
        except ImportError:
            # Internal package not available - this is expected in many cases
            pass
        except Exception as e:
            print(f"nemo-evaluator-launcher-internal: error loading ({e})")
