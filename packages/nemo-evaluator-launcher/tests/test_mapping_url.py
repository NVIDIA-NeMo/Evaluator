"""Tests for mapping URL validation."""

import os
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.mapping import MAPPING_URL, _download_latest_mapping


def test_mapping_url_contains_main():
    """Test that MAPPING_URL contains 'main' substring in the branch path."""
    # Check that the URL contains 'main' in the branch part (after /raw/)
    assert (
        "ref=main" in MAPPING_URL
    ), f"MAPPING_URL '{MAPPING_URL}' must contain '/raw/main/'"


def test_mapping_url_is_reachable():
    """Test that MAPPING_URL is reachable and returns valid TOML."""
    # Get GitLab token from environment
    gitlab_token = os.environ.get("GITLAB_TOKEN", "")
    if not gitlab_token:
        pytest.skip("GITLAB_TOKEN not set, skipping network test")

    # Test the actual download function
    result = _download_latest_mapping()
    assert result is not None, f"Failed to download mapping from '{MAPPING_URL}'"
    assert isinstance(result, bytes), "Downloaded mapping should be raw bytes"
    assert len(result) > 0, "Downloaded mapping should not be empty"
    # Test the content
    mapping_str = result.decode("utf-8")
    mapping = tomllib.loads(mapping_str)
