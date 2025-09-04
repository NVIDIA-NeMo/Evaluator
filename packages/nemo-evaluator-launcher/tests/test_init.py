"""Tests for the __init__.py module."""

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import nemo_evaluator_launcher


def test_version_import():
    """Test that __version__ is properly imported and accessible."""
    # Test that __version__ is available at the package level
    assert hasattr(nemo_evaluator_launcher, "__version__")

    # Test that __version__ is a string
    assert isinstance(nemo_evaluator_launcher.__version__, str)

    # Test that __version__ is not empty
    assert len(nemo_evaluator_launcher.__version__) > 0


@patch("importlib.import_module")
def test_internal_package_successful_import(mock_import_module):
    """Test successful import of internal package."""
    # Mock successful import
    mock_import_module.return_value = MagicMock()

    # Re-import the module to trigger the import logic
    importlib.reload(nemo_evaluator_launcher)

    # Verify import was called
    mock_import_module.assert_called_with("nemo_evaluator_launcher_internal")


@patch("importlib.import_module")
def test_internal_package_import_error(mock_import_module):
    """Test ImportError when internal package is not available."""
    # Mock ImportError
    mock_import_module.side_effect = ImportError(
        "No module named 'nemo_evaluator_launcher_internal'"
    )

    # Re-import the module to trigger the import logic
    # This should not raise an exception
    importlib.reload(nemo_evaluator_launcher)

    # Verify import was called
    mock_import_module.assert_called_with("nemo_evaluator_launcher_internal")


@patch("importlib.import_module")
def test_internal_package_successful_import_logging(mock_import_module, caplog):
    """Test that proper success logging occurs when internal package import succeeds."""
    # Mock successful import
    mock_import_module.return_value = MagicMock()

    # Set log level to DEBUG to capture debug messages
    with caplog.at_level(logging.DEBUG):
        # Re-import the module to trigger the import logic
        importlib.reload(nemo_evaluator_launcher)

    # Verify import was called
    mock_import_module.assert_called_with("nemo_evaluator_launcher_internal")

    # Verify that the expected log message appears in the captured logs
    log_messages = caplog.text
    assert "Successfully loaded internal package" in log_messages
    assert "nemo_evaluator_launcher_internal" in log_messages


@patch("importlib.import_module")
def test_internal_package_import_error_logging(mock_import_module, caplog):
    """Test that proper error logging occurs when internal package import fails."""
    # Mock ImportError
    error_message = "No module named 'nemo_evaluator_launcher_internal'"
    mock_import_module.side_effect = ImportError(error_message)

    # Set log level to DEBUG to capture debug messages
    with caplog.at_level(logging.DEBUG):
        # Re-import the module to trigger the import logic
        importlib.reload(nemo_evaluator_launcher)

    # Verify import was called
    mock_import_module.assert_called_with("nemo_evaluator_launcher_internal")

    # Verify that the expected log message appears in the captured logs
    # The structlog output should contain the error event and package info
    log_messages = caplog.text
    assert "Internal package not available" in log_messages
    assert "nemo_evaluator_launcher_internal" in log_messages
    assert error_message in log_messages


def test_version_consistency():
    """Test that version from __init__.py matches version_utils."""
    from nemo_evaluator_launcher.common.version_utils import (
        __version__ as version_utils_version,
    )

    assert nemo_evaluator_launcher.__version__ == version_utils_version
