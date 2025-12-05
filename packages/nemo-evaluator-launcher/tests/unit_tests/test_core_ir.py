# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for core Intermediate Representation (IR) structures.

IR stands for Intermediate Representation - structured data representations
that bridge between raw framework.yml files and the launcher's internal data structures.
"""

import hashlib
import importlib.resources

import pytest
import yaml

from nemo_evaluator_launcher.common.task_ir import (
    _calculate_mapping_checksum,
    load_tasks_from_tasks_file,
)


def test_calculate_mapping_checksum(tmp_path):
    """Test that checksum calculation works correctly."""
    mapping_file = tmp_path / "mapping.toml"
    mapping_file.write_text("[test]\ncontainer = 'test:latest'")

    checksum = _calculate_mapping_checksum(mapping_file)

    assert checksum is not None
    assert checksum.startswith("sha256:")
    assert len(checksum) == 71  # sha256: + 64 hex chars

    # Verify checksum is deterministic
    checksum2 = _calculate_mapping_checksum(mapping_file)
    assert checksum == checksum2


def test_load_tasks_mismatched_checksum(tmp_path, caplog):
    """Test that mismatched checksums log warning and return mapping_verified=False."""
    tasks_file = tmp_path / "all_tasks_irs.yaml"
    mapping_file = tmp_path / "mapping.toml"
    mapping_file.write_text("[test]\ncontainer = 'test:latest'")

    # Create tasks file with mismatched checksum
    tasks_data = {
        "metadata": {
            "mapping_toml_checksum": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "num_tasks": 1,
        },
        "tasks": [
            {
                "name": "test_task",
                "description": "",
                "harness": "test",
                "container": "test:latest",
                "defaults": {},
            }
        ],
    }
    tasks_file.write_text(yaml.dump(tasks_data))

    tasks, mapping_verified = load_tasks_from_tasks_file(tasks_file)

    assert len(tasks) == 1
    assert mapping_verified is False
    assert "mapping.toml checksum mismatch detected" in caplog.text


def test_load_tasks_matching_checksum(tmp_path, caplog):
    """Test that matching checksums return mapping_verified=True."""
    tasks_file = tmp_path / "all_tasks_irs.yaml"
    mapping_file = tmp_path / "mapping.toml"
    mapping_file.write_text("[test]\ncontainer = 'test:latest'")

    checksum = _calculate_mapping_checksum(mapping_file)

    tasks_data = {
        "metadata": {
            "mapping_toml_checksum": checksum,
            "num_tasks": 1,
        },
        "tasks": [
            {
                "name": "test_task",
                "description": "",
                "harness": "test",
                "container": "test:latest",
                "defaults": {},
            }
        ],
    }
    tasks_file.write_text(yaml.dump(tasks_data))

    tasks, mapping_verified = load_tasks_from_tasks_file(tasks_file)

    assert len(tasks) == 1
    assert mapping_verified is True
    assert "mapping.toml checksum matches all_tasks_irs.yaml" in caplog.text


def test_load_tasks_missing_checksum(tmp_path, caplog):
    """Test that missing checksum returns mapping_verified=False."""
    tasks_file = tmp_path / "all_tasks_irs.yaml"
    mapping_file = tmp_path / "mapping.toml"
    mapping_file.write_text("[test]\ncontainer = 'test:latest'")

    # Create tasks file without checksum
    tasks_data = {
        "metadata": {
            "num_tasks": 1,
        },
        "tasks": [
            {
                "name": "test_task",
                "description": "",
                "harness": "test",
                "container": "test:latest",
                "defaults": {},
            }
        ],
    }
    tasks_file.write_text(yaml.dump(tasks_data))

    tasks, mapping_verified = load_tasks_from_tasks_file(tasks_file)

    assert len(tasks) == 1
    assert mapping_verified is False


def test_load_tasks_missing_mapping_file(tmp_path, caplog):
    """Test that missing mapping.toml returns mapping_verified=False."""
    tasks_file = tmp_path / "all_tasks_irs.yaml"
    # Don't create mapping.toml

    tasks_data = {
        "metadata": {
            "mapping_toml_checksum": "sha256:abc123",
            "num_tasks": 1,
        },
        "tasks": [
            {
                "name": "test_task",
                "description": "",
                "harness": "test",
                "container": "test:latest",
                "defaults": {},
            }
        ],
    }
    tasks_file.write_text(yaml.dump(tasks_data))

    tasks, mapping_verified = load_tasks_from_tasks_file(tasks_file)

    assert len(tasks) == 1
    assert mapping_verified is False


def test_packaged_mapping_toml_checksum_match():
    """CI guard: Ensure packaged mapping.toml matches packaged all_tasks_irs.yaml.

    This test validates that the packaged artifacts are in sync.
    If this test fails, it means mapping.toml was changed but all_tasks_irs.yaml
    wasn't regenerated - CI should catch this.
    """
    # Load packaged all_tasks_irs.yaml
    try:
        tasks_content = importlib.resources.read_text(
            "nemo_evaluator_launcher.resources",
            "all_tasks_irs.yaml",
            encoding="utf-8",
        )
        tasks_data = yaml.safe_load(tasks_content)
    except Exception as e:
        pytest.fail(f"Could not load packaged all_tasks_irs.yaml: {e}")

    # Get stored checksum from metadata
    stored_checksum = tasks_data.get("metadata", {}).get("mapping_toml_checksum")
    if not stored_checksum:
        pytest.fail(
            "packaged all_tasks_irs.yaml missing mapping_toml_checksum in metadata"
        )

    # Load packaged mapping.toml
    try:
        mapping_content = importlib.resources.read_text(
            "nemo_evaluator_launcher.resources",
            "mapping.toml",
            encoding="utf-8",
        )
    except Exception as e:
        pytest.fail(f"Could not load packaged mapping.toml: {e}")

    # Calculate current checksum of packaged mapping.toml
    current_checksum = hashlib.sha256(mapping_content.encode("utf-8")).hexdigest()
    current_checksum = f"sha256:{current_checksum}"

    # Assert checksums match - FAIL if they don't (CI guard)
    assert stored_checksum == current_checksum, (
        f"packaged mapping.toml checksum mismatch!\n"
        f"  stored (in all_tasks_irs.yaml): {stored_checksum}\n"
        f"  current (mapping.toml): {current_checksum}\n"
        f"This means mapping.toml was changed but all_tasks_irs.yaml wasn't regenerated.\n"
        f"Please regenerate all_tasks_irs.yaml by running:\n"
        f"  python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py"
    )
