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

"""Dataset download pre-evaluation hook."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import AdapterGlobalContext, PreEvalHook
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


@register_for_adapter(
    name="dataset_download_prehook",
    description="Downloads datasets from various sources (HuggingFace, local, S3, NGC, URL, WandB) before evaluation",
)
class DatasetDownloadPreHook(PreEvalHook):
    """Pre-evaluation hook that downloads datasets from various sources."""

    class Params(BaseModel):
        """Configuration parameters for dataset download."""

        target_path: str = Field(
            description="A path where the dataset should be placed to be correctly loaded"
        )

        source_type: Literal["ngc", "s3", "local", "wandb", "huggingface", "url"] = (
            Field(description="Type of the dataset download backend to be used")
        )

        source_path: str = Field(
            description="A path or URI pointing to the dataset location"
        )

        source_config: dict[str, Any] = Field(
            default_factory=dict,
            description="Source-specific config, e.g., api_key, token, split, etc.",
        )

        force_download: bool = Field(
            default=False,
            description="If True, re-download even if target_path already exists",
        )

        @field_validator("target_path")
        @classmethod
        def validate_target_path(cls, v: str) -> str:
            """Validate and expand target path."""
            return os.path.expanduser(os.path.expandvars(v))

    def __init__(self, params: Params):
        """
        Initialize the dataset download pre-hook.

        Args:
            params: Configuration parameters
        """
        self.params = params
        logger.info(
            "Initialized DatasetDownloadPreHook",
            source_type=params.source_type,
            source_path=params.source_path,
            target_path=params.target_path,
        )

    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Download dataset from configured source."""
        target_path = Path(self.params.target_path)

        # Check if dataset already exists
        if target_path.exists() and not self.params.force_download:
            logger.info(
                "Dataset already exists, skipping download",
                target_path=str(target_path),
            )
            return

        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting dataset download",
            source_type=self.params.source_type,
            source_path=self.params.source_path,
            target_path=str(target_path),
        )

        try:
            if self.params.source_type == "huggingface":
                self._download_from_huggingface(target_path)
            elif self.params.source_type == "local":
                self._copy_from_local(target_path)
            elif self.params.source_type == "s3":
                self._download_from_s3(target_path)
            elif self.params.source_type == "ngc":
                self._download_from_ngc(target_path)
            elif self.params.source_type == "url":
                self._download_from_url(target_path)
            elif self.params.source_type == "wandb":
                self._download_from_wandb(target_path)
            else:
                raise ValueError(f"Unsupported source type: {self.params.source_type}")

            logger.info("Dataset download completed", target_path=str(target_path))

        except Exception as e:
            logger.error(
                "Dataset download failed",
                source_type=self.params.source_type,
                error=str(e),
            )
            raise

    def _download_from_huggingface(self, target_path: Path) -> None:
        """Download dataset from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace downloads. "
                "Install it with: pip install datasets"
            )

        # Parse source_path as dataset_name
        dataset_name = self.params.source_path
        split = self.params.source_config.get("split", None)
        token = self.params.source_config.get("token", None)
        cache_dir = self.params.source_config.get("cache_dir", None)

        logger.info(
            "Downloading from HuggingFace",
            dataset=dataset_name,
            split=split,
        )

        # Load dataset
        dataset = load_dataset(
            dataset_name,
            split=split,
            token=token,
            cache_dir=cache_dir,
        )

        # Save to target path
        if hasattr(dataset, "save_to_disk"):
            dataset.save_to_disk(str(target_path))
        else:
            # For older versions of datasets
            dataset.save(str(target_path))

    def _copy_from_local(self, target_path: Path) -> None:
        """Copy dataset from local path."""
        source_path = Path(self.params.source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        logger.info("Copying from local path", source=str(source_path))

        if source_path.is_file():
            # Copy single file
            shutil.copy2(source_path, target_path)
        else:
            # Copy directory
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)

    def _download_from_s3(self, target_path: Path) -> None:
        """Download dataset from S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 downloads. Install it with: pip install boto3"
            )

        # Parse S3 URI
        s3_uri = self.params.source_path
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with 's3://'")

        # Extract bucket and key
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # Get AWS credentials from source_config
        aws_access_key_id = self.params.source_config.get("aws_access_key_id")
        aws_secret_access_key = self.params.source_config.get("aws_secret_access_key")
        region_name = self.params.source_config.get("region_name", "us-east-1")

        logger.info("Downloading from S3", bucket=bucket, key=key)

        # Create S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        s3 = session.client("s3")

        # Download file or directory
        s3.download_file(bucket, key, str(target_path))

    def _download_from_ngc(self, target_path: Path) -> None:
        """Download dataset from NVIDIA NGC."""
        # NGC CLI should be installed and configured
        ngc_path = self.params.source_path
        api_key = self.params.source_config.get("api_key")

        logger.info("Downloading from NGC", path=ngc_path)

        # Build NGC CLI command
        cmd = ["ngc", "registry", "resource", "download-version", ngc_path]

        if api_key:
            cmd.extend(["--api-key", api_key])

        # Set destination
        cmd.extend(["--dest", str(target_path.parent)])

        # Execute NGC CLI
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.returncode != 0:
            raise RuntimeError(f"NGC download failed: {result.stderr}")

    def _download_from_url(self, target_path: Path) -> None:
        """Download dataset from URL."""
        import urllib.request

        url = self.params.source_path
        logger.info("Downloading from URL", url=url)

        # Download with urllib
        urllib.request.urlretrieve(url, str(target_path))

        # If it's a compressed file, extract it
        if self.params.source_config.get("extract", False):
            self._extract_archive(target_path)

    def _download_from_wandb(self, target_path: Path) -> None:
        """Download dataset from Weights & Biases."""
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for W&B downloads. Install it with: pip install wandb"
            )

        # Parse artifact path
        artifact_path = self.params.source_path
        api_key = self.params.source_config.get("api_key")

        if api_key:
            wandb.login(key=api_key)

        logger.info("Downloading from W&B", artifact=artifact_path)

        # Initialize W&B run
        run = wandb.init(project=self.params.source_config.get("project", "default"))

        # Download artifact
        artifact = run.use_artifact(artifact_path)
        artifact.download(root=str(target_path))

        # Finish run
        run.finish()

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract compressed archive."""
        import tarfile
        import zipfile

        logger.info("Extracting archive", path=str(archive_path))

        extract_dir = archive_path.parent / archive_path.stem

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in [".tar", ".gz", ".bz2", ".xz"]:
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            logger.warning(
                "Unknown archive format, skipping extraction",
                suffix=archive_path.suffix,
            )
            return

        # If extract succeeded and we want to replace the archive
        if self.params.source_config.get("remove_archive", False):
            archive_path.unlink()
