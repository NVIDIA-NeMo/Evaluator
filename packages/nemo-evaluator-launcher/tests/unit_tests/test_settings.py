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
"""Tests for the settings module."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml


class TestSlurmProfile:
    """Tests for SlurmProfile dataclass."""

    def test_slurm_profile_default_values(self):
        """Test SlurmProfile has correct default values."""
        from nemo_evaluator_launcher.common.settings import SlurmProfile

        profile = SlurmProfile(hostname="cluster.example.com")

        assert profile.hostname == "cluster.example.com"
        assert profile.username is None
        assert profile.account is None
        assert profile.partition == "batch"
        assert profile.walltime == "01:00:00"
        assert profile.gres is None

    def test_slurm_profile_all_values(self):
        """Test SlurmProfile with all values set."""
        from nemo_evaluator_launcher.common.settings import SlurmProfile

        profile = SlurmProfile(
            hostname="cluster.example.com",
            username="testuser",
            account="myaccount",
            partition="gpu",
            walltime="04:00:00",
            gres="gpu:8",
        )

        assert profile.hostname == "cluster.example.com"
        assert profile.username == "testuser"
        assert profile.account == "myaccount"
        assert profile.partition == "gpu"
        assert profile.walltime == "04:00:00"
        assert profile.gres == "gpu:8"


class TestEnvVarSettings:
    """Tests for EnvVarSettings dataclass."""

    def test_env_var_settings_default_values(self):
        """Test EnvVarSettings has empty lists by default."""
        from nemo_evaluator_launcher.common.settings import EnvVarSettings

        settings = EnvVarSettings()

        assert settings.deployment == []
        assert settings.task == []

    def test_env_var_settings_with_values(self):
        """Test EnvVarSettings with values set."""
        from nemo_evaluator_launcher.common.settings import EnvVarSettings

        settings = EnvVarSettings(
            deployment=["NGC_API_KEY", "HF_TOKEN"],
            task=["HF_TOKEN", "JUDGE_API_KEY"],
        )

        assert settings.deployment == ["NGC_API_KEY", "HF_TOKEN"]
        assert settings.task == ["HF_TOKEN", "JUDGE_API_KEY"]


class TestSettingsStore:
    """Tests for SettingsStore class."""

    def test_settings_store_is_singleton(self):
        """Test that SettingsStore is a singleton."""
        from nemo_evaluator_launcher.common.settings import SettingsStore

        # Reset singleton for test
        SettingsStore._instance = None

        store1 = SettingsStore()
        store2 = SettingsStore()

        assert store1 is store2

    def test_settings_store_list_profiles_empty(self):
        """Test listing profiles when none exist."""
        from nemo_evaluator_launcher.common.settings import SettingsStore

        # Reset singleton for test
        SettingsStore._instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "nemo_evaluator_launcher.common.settings.SETTINGS_DIR",
                Path(tmpdir),
            ):
                with mock.patch(
                    "nemo_evaluator_launcher.common.settings.SETTINGS_FILE",
                    Path(tmpdir) / "settings.yaml",
                ):
                    store = SettingsStore()
                    assert store.list_slurm_profiles() == []

    def test_settings_store_save_and_get_profile(self):
        """Test saving and retrieving a SLURM profile."""
        from nemo_evaluator_launcher.common.settings import (
            SettingsStore,
            SlurmProfile,
        )

        # Reset singleton for test
        SettingsStore._instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "nemo_evaluator_launcher.common.settings.SETTINGS_DIR",
                Path(tmpdir),
            ):
                with mock.patch(
                    "nemo_evaluator_launcher.common.settings.SETTINGS_FILE",
                    Path(tmpdir) / "settings.yaml",
                ):
                    store = SettingsStore()

                    profile = SlurmProfile(
                        hostname="cluster.example.com",
                        account="myaccount",
                        partition="gpu",
                        walltime="04:00:00",
                    )
                    store.save_slurm_profile("test-cluster", profile)

                    # Verify profile is listed
                    assert "test-cluster" in store.list_slurm_profiles()

                    # Verify profile can be retrieved
                    retrieved = store.get_slurm_profile("test-cluster")
                    assert retrieved is not None
                    assert retrieved.hostname == "cluster.example.com"
                    assert retrieved.account == "myaccount"

    def test_settings_store_delete_profile(self):
        """Test deleting a SLURM profile."""
        from nemo_evaluator_launcher.common.settings import (
            SettingsStore,
            SlurmProfile,
        )

        # Reset singleton for test
        SettingsStore._instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "nemo_evaluator_launcher.common.settings.SETTINGS_DIR",
                Path(tmpdir),
            ):
                with mock.patch(
                    "nemo_evaluator_launcher.common.settings.SETTINGS_FILE",
                    Path(tmpdir) / "settings.yaml",
                ):
                    store = SettingsStore()

                    profile = SlurmProfile(hostname="cluster.example.com")
                    store.save_slurm_profile("test-cluster", profile)

                    assert "test-cluster" in store.list_slurm_profiles()

                    store.delete_slurm_profile("test-cluster")

                    assert "test-cluster" not in store.list_slurm_profiles()
                    assert store.get_slurm_profile("test-cluster") is None

    def test_settings_store_save_and_get_env_vars(self):
        """Test saving and retrieving env var settings."""
        from nemo_evaluator_launcher.common.settings import (
            EnvVarSettings,
            SettingsStore,
        )

        # Reset singleton for test
        SettingsStore._instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "nemo_evaluator_launcher.common.settings.SETTINGS_DIR",
                Path(tmpdir),
            ):
                with mock.patch(
                    "nemo_evaluator_launcher.common.settings.SETTINGS_FILE",
                    Path(tmpdir) / "settings.yaml",
                ):
                    store = SettingsStore()

                    env_settings = EnvVarSettings(
                        deployment=["NGC_API_KEY"],
                        task=["HF_TOKEN"],
                    )
                    store.save_env_vars(env_settings)

                    retrieved = store.get_env_vars()
                    assert retrieved.deployment == ["NGC_API_KEY"]
                    assert retrieved.task == ["HF_TOKEN"]

    def test_settings_store_persistence(self):
        """Test that settings persist to disk."""
        from nemo_evaluator_launcher.common.settings import (
            SettingsStore,
            SlurmProfile,
        )

        # Reset singleton for test
        SettingsStore._instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.yaml"
            with mock.patch(
                "nemo_evaluator_launcher.common.settings.SETTINGS_DIR",
                Path(tmpdir),
            ):
                with mock.patch(
                    "nemo_evaluator_launcher.common.settings.SETTINGS_FILE",
                    settings_file,
                ):
                    store = SettingsStore()

                    profile = SlurmProfile(
                        hostname="cluster.example.com",
                        account="myaccount",
                    )
                    store.save_slurm_profile("test-cluster", profile)

                    # Verify file was created
                    assert settings_file.exists()

                    # Verify file content
                    with open(settings_file) as f:
                        data = yaml.safe_load(f)

                    assert "slurm_profiles" in data
                    assert "test-cluster" in data["slurm_profiles"]
                    assert data["slurm_profiles"]["test-cluster"]["hostname"] == "cluster.example.com"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_store(self):
        """Test that get_settings returns a SettingsStore instance."""
        from nemo_evaluator_launcher.common.settings import SettingsStore, get_settings

        # Reset singleton for test
        SettingsStore._instance = None

        store = get_settings()
        assert isinstance(store, SettingsStore)

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance."""
        from nemo_evaluator_launcher.common.settings import SettingsStore, get_settings

        # Reset singleton for test
        SettingsStore._instance = None

        store1 = get_settings()
        store2 = get_settings()
        assert store1 is store2
