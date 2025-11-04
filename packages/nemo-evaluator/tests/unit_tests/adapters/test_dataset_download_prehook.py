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

"""Tests for DatasetDownloadPreHook functionality."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.adapters.pre_hooks.dataset_download_prehook import (
    DatasetDownloadPreHook,
)
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterGlobalContext


def test_dataset_download_prehook_registration():
    """Test that DatasetDownloadPreHook is registered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.clear_cache()

    # The DatasetDownloadPreHook should be registered automatically when imported
    available_hooks = registry.get_pre_eval_hooks()
    assert "dataset_download_prehook" in available_hooks
    assert available_hooks["dataset_download_prehook"].supports_pre_eval_hook()


def test_dataset_download_prehook_creation():
    """Test that DatasetDownloadPreHook instances can be created correctly."""
    registry = InterceptorRegistry.get_instance()

    config = {
        "target_path": "/tmp/test_dataset",
        "source_type": "local",
        "source_path": "/tmp/source_dataset",
    }

    hook = registry._get_or_create_instance("dataset_download_prehook", config)
    assert isinstance(hook, DatasetDownloadPreHook)
    assert hook.params.target_path == "/tmp/test_dataset"
    assert hook.params.source_type == "local"
    assert hook.params.source_path == "/tmp/source_dataset"


def test_local_copy_file(tmpdir):
    """Test copying a file from local path."""
    # Create source file
    source_file = tmpdir / "source" / "dataset.txt"
    source_file.dirpath().mkdir()
    source_file.write("test data")

    target_file = tmpdir / "target" / "dataset.txt"

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_file),
        source_type="local",
        source_path=str(source_file),
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify file was copied
    assert target_file.exists()
    assert target_file.read() == "test data"


def test_local_copy_directory(tmpdir):
    """Test copying a directory from local path."""
    # Create source directory
    source_dir = tmpdir / "source"
    source_dir.mkdir()
    (source_dir / "file1.txt").write("data1")
    (source_dir / "file2.txt").write("data2")

    target_dir = tmpdir / "target"

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="local",
        source_path=str(source_dir),
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify directory was copied
    assert target_dir.exists()
    assert (target_dir / "file1.txt").exists()
    assert (target_dir / "file2.txt").exists()
    assert (target_dir / "file1.txt").read() == "data1"
    assert (target_dir / "file2.txt").read() == "data2"


def test_skip_download_if_exists(tmpdir):
    """Test that download is skipped if target already exists."""
    # Create target directory
    target_dir = tmpdir / "target"
    target_dir.mkdir()
    (target_dir / "existing.txt").write("existing data")

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="local",
        source_path=str(tmpdir / "source"),
        force_download=False,
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify directory still has only existing file
    assert target_dir.exists()
    assert (target_dir / "existing.txt").exists()
    assert (target_dir / "existing.txt").read() == "existing data"


def test_force_download(tmpdir):
    """Test that force_download re-downloads even if target exists."""
    # Create source directory
    source_dir = tmpdir / "source"
    source_dir.mkdir()
    (source_dir / "new_file.txt").write("new data")

    # Create target directory with existing content
    target_dir = tmpdir / "target"
    target_dir.mkdir()
    (target_dir / "old_file.txt").write("old data")

    # Create hook with force_download
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="local",
        source_path=str(source_dir),
        force_download=True,
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify directory was replaced
    assert target_dir.exists()
    assert (target_dir / "new_file.txt").exists()
    assert not (target_dir / "old_file.txt").exists()


def test_local_source_not_found(tmpdir):
    """Test that appropriate error is raised when source doesn't exist."""
    target_dir = tmpdir / "target"

    # Create hook with non-existent source
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="local",
        source_path=str(tmpdir / "nonexistent"),
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook - should raise FileNotFoundError
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    with pytest.raises(FileNotFoundError):
        hook.pre_eval_hook(context)


@patch("datasets.load_dataset")
def test_huggingface_download(mock_load_dataset, tmpdir):
    """Test downloading from HuggingFace."""
    target_dir = tmpdir / "target"

    # Mock dataset object
    mock_dataset = MagicMock()
    mock_load_dataset.return_value = mock_dataset

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="huggingface",
        source_path="test/dataset",
        source_config={"split": "train", "token": "test_token"},
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify load_dataset was called correctly
    mock_load_dataset.assert_called_once_with(
        "test/dataset", split="train", token="test_token", cache_dir=None
    )
    # Verify save_to_disk was called
    mock_dataset.save_to_disk.assert_called_once_with(str(target_dir))


@patch("urllib.request.urlretrieve")
def test_url_download(mock_urlretrieve, tmpdir):
    """Test downloading from URL."""
    target_file = tmpdir / "dataset.zip"

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_file),
        source_type="url",
        source_path="http://example.com/dataset.zip",
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify urlretrieve was called
    mock_urlretrieve.assert_called_once_with(
        "http://example.com/dataset.zip", str(target_file)
    )


@patch("boto3.Session")
def test_s3_download(mock_session, tmpdir):
    """Test downloading from S3."""
    target_file = tmpdir / "dataset.txt"

    # Mock S3 client
    mock_s3_client = MagicMock()
    mock_session.return_value.client.return_value = mock_s3_client

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_file),
        source_type="s3",
        source_path="s3://test-bucket/dataset.txt",
        source_config={
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "region_name": "us-west-2",
        },
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify boto3 session was created correctly
    mock_session.assert_called_once_with(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-west-2",
    )
    # Verify download_file was called
    mock_s3_client.download_file.assert_called_once_with(
        "test-bucket", "dataset.txt", str(target_file)
    )


@patch("subprocess.run")
def test_ngc_download(mock_run, tmpdir):
    """Test downloading from NGC."""
    target_dir = tmpdir / "dataset"

    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0)

    # Create hook with force_download to ensure download happens
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="ngc",
        source_path="nvidia/test-dataset:v1",
        source_config={"api_key": "test_api_key"},
        force_download=True,  # Force download to bypass existence check
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify subprocess.run was called with correct command
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "ngc"
    assert call_args[1] == "registry"
    assert call_args[2] == "resource"
    assert call_args[3] == "download-version"
    assert call_args[4] == "nvidia/test-dataset:v1"
    assert "--api-key" in call_args
    assert "test_api_key" in call_args


@patch("wandb.init")
@patch("wandb.login")
def test_wandb_download(mock_login, mock_init, tmpdir):
    """Test downloading from Weights & Biases."""
    target_dir = tmpdir / "dataset"

    # Mock W&B run and artifact
    mock_artifact = MagicMock()
    mock_run = MagicMock()
    mock_run.use_artifact.return_value = mock_artifact
    mock_init.return_value = mock_run

    # Create hook
    params = DatasetDownloadPreHook.Params(
        target_path=str(target_dir),
        source_type="wandb",
        source_path="entity/project/artifact:v1",
        source_config={"api_key": "test_key", "project": "test_project"},
    )
    hook = DatasetDownloadPreHook(params)

    # Execute hook
    context = AdapterGlobalContext(output_dir=str(tmpdir), url="http://test.com")
    hook.pre_eval_hook(context)

    # Verify W&B functions were called
    mock_login.assert_called_once_with(key="test_key")
    mock_init.assert_called_once_with(project="test_project")
    mock_run.use_artifact.assert_called_once_with("entity/project/artifact:v1")
    mock_artifact.download.assert_called_once_with(root=str(target_dir))
    mock_run.finish.assert_called_once()


def test_invalid_source_type(tmpdir):
    """Test that invalid source type raises ValueError."""
    target_dir = tmpdir / "target"

    # Create hook with invalid source type
    with pytest.raises(ValueError):
        DatasetDownloadPreHook.Params(
            target_path=str(target_dir),
            source_type="invalid_type",  # This should fail validation
            source_path="/tmp/source",
        )

