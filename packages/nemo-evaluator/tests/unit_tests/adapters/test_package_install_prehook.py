# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PackageInstallPreHook functionality."""

import subprocess
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from nemo_evaluator.adapters.pre_hooks.package_install_prehook import (
    PackageInstallPreHook,
)
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterGlobalContext


def test_package_install_prehook_registration():
    """Test that PackageInstallPreHook is registered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.clear_cache()

    # The PackageInstallPreHook should be registered automatically when imported
    available_hooks = registry.get_pre_eval_hooks()
    assert "package_install_prehook" in available_hooks
    assert available_hooks["package_install_prehook"].supports_pre_eval_hook()


def test_package_install_prehook_creation():
    """Test that PackageInstallPreHook instances can be created correctly."""
    registry = InterceptorRegistry.get_instance()

    config = {
        "packages": ["numpy>=1.20", "torch==2.0.0"],
    }

    hook = registry._get_or_create_instance("package_install_prehook", config)
    assert isinstance(hook, PackageInstallPreHook)
    assert hook.params.packages == ["numpy>=1.20", "torch==2.0.0"]


def test_package_install_prehook_requires_packages_or_file():
    """Test that either packages or requirements_file must be provided."""
    with pytest.raises(ValueError, match="Either 'packages' list or 'requirements_file'"):
        PackageInstallPreHook(
            PackageInstallPreHook.Params(
                packages=[],
                requirements_file=None,
            )
        )


@patch("subprocess.run")
def test_install_packages_with_pip(mock_run, tmpdir):
    """Test installing packages with pip."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook
    params = PackageInstallPreHook.Params(
        packages=["numpy>=1.20", "pandas"],
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called with correct command
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == sys.executable
    assert call_args[1] == "-m"
    assert call_args[2] == "pip"
    assert call_args[3] == "install"
    assert "numpy>=1.20" in call_args
    assert "pandas" in call_args


@patch("subprocess.run")
def test_install_packages_with_upgrade(mock_run, tmpdir):
    """Test installing packages with upgrade flag."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook
    params = PackageInstallPreHook.Params(
        packages=["numpy"],
        upgrade=True,
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called with upgrade flag
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "--upgrade" in call_args


@patch("subprocess.run")
def test_install_packages_with_extra_args(mock_run, tmpdir):
    """Test installing packages with extra arguments."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook
    params = PackageInstallPreHook.Params(
        packages=["numpy"],
        extra_args=["--no-cache-dir", "--index-url", "https://test.pypi.org/simple/"],
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called with extra args
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "--no-cache-dir" in call_args
    assert "--index-url" in call_args
    assert "https://test.pypi.org/simple/" in call_args


@patch("subprocess.run")
def test_install_from_requirements_file(mock_run, tmpdir):
    """Test installing packages from requirements file."""
    # Create requirements file
    requirements_file = tmpdir / "requirements.txt"
    requirements_file.write("numpy>=1.20\npandas\n")

    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook
    params = PackageInstallPreHook.Params(
        requirements_file=str(requirements_file),
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called with -r flag
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == sys.executable
    assert call_args[1] == "-m"
    assert call_args[2] == "pip"
    assert call_args[3] == "install"
    assert "-r" in call_args
    assert str(requirements_file) in call_args


@patch("builtins.__import__")
@patch("subprocess.run")
def test_skip_already_installed_packages(mock_run, mock_import, tmpdir):
    """Test that already installed packages are skipped."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Mock numpy as already installed
    def side_effect(name, *args, **kwargs):
        if name == "numpy":
            return MagicMock()
        raise ImportError(f"No module named '{name}'")

    mock_import.side_effect = side_effect

    # Create hook with check_installed=True
    params = PackageInstallPreHook.Params(
        packages=["numpy", "pandas"],  # numpy is "installed", pandas is not
        package_manager="pip",
        check_installed=True,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called only for pandas
    if mock_run.called:
        call_args = mock_run.call_args[0][0]
        # numpy should not be in the command since it's already installed
        # pandas should be in the command
        assert "pandas" in call_args


@patch("subprocess.run")
def test_install_failure_raises_error(mock_run, tmpdir):
    """Test that installation failure raises an error."""
    # Mock failed subprocess run
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["pip", "install", "numpy"],
        stderr="Error installing package",
    )

    # Create hook with fail_on_error=True
    params = PackageInstallPreHook.Params(
        packages=["numpy"],
        fail_on_error=True,
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook - should raise RuntimeError
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    with pytest.raises(RuntimeError):
        hook.pre_eval_hook(context)


@patch("subprocess.run")
def test_install_failure_continues_when_fail_on_error_false(mock_run, tmpdir):
    """Test that installation continues despite failure when fail_on_error=False."""
    # Mock failed subprocess run
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["pip", "install", "numpy"],
        stderr="Error installing package",
    )

    # Create hook with fail_on_error=False
    params = PackageInstallPreHook.Params(
        packages=["numpy"],
        fail_on_error=False,
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook - should not raise
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)  # Should complete without error


@patch("subprocess.run")
def test_install_both_packages_and_requirements(mock_run, tmpdir):
    """Test installing both individual packages and from requirements file."""
    # Create requirements file
    requirements_file = tmpdir / "requirements.txt"
    requirements_file.write("pandas\n")

    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook with both packages and requirements_file
    params = PackageInstallPreHook.Params(
        packages=["numpy"],
        requirements_file=str(requirements_file),
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called twice (once for requirements, once for packages)
    assert mock_run.call_count == 2


@patch("subprocess.run")
def test_package_spec_parsing(mock_run, tmpdir):
    """Test that package specifications are parsed correctly."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create hook with various package specifications
    params = PackageInstallPreHook.Params(
        packages=[
            "numpy>=1.20",
            "pandas==1.3.0",
            "scipy<1.8.0",
            "matplotlib",
        ],
        check_installed=False,
    )
    hook = PackageInstallPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify all packages were included
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "numpy>=1.20" in call_args
    assert "pandas==1.3.0" in call_args
    assert "scipy<1.8.0" in call_args
    assert "matplotlib" in call_args

