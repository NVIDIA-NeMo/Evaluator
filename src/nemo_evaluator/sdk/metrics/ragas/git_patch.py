"""
Patch for GitPython to avoid Git dependency in containerized environments.

This module patches the git.refresh function to be a no-op, preventing
GitPython from trying to find the git executable during import.
This is particularly useful when using ragas which depends on GitPython
but doesn't actually need git functionality in a containerized environment.
"""

import sys
from types import ModuleType


def apply_git_patch():
    """
    Apply patch so that git is not available.

    This must be called before any imports of ragas or other packages
    that depend on GitPython.

    If GitPython is already installed and available, this function does nothing
    to avoid breaking code that actually uses GitPython.
    """
    # If git module is already loaded (real or patched), leave it alone
    if "git" in sys.modules:
        return

    # Try to import GitPython
    try:
        import git  # noqa: F401
    except ImportError:
        # GitPython not available, apply patch
        sys.modules["git"] = ModuleType("git")
