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
"""Tests for the Hugging Face Hub exporter."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nemo_evaluator_launcher.exporters.huggingface import (
    HuggingFaceExporter,
    HuggingFaceExporterConfig,
    _build_eval_result_entry,
    _format_eval_results_yaml,
    _sanitize_filename,
    _select_metric,
)
from nemo_evaluator_launcher.exporters.utils import DataForExport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    return HuggingFaceExporterConfig(
        dataset_id="nvidia/compute-eval",
        task_id="compute_eval",
        model_repo="nvidia/test-model",
        metric_name="pass_at_1",
        token="hf_test_token",
        create_pr=True,
        notes="Pass@1",
    )


@pytest.fixture
def sample_data_for_export():
    return DataForExport(
        artifacts_dir=Path("/tmp/artifacts"),
        logs_dir=None,
        config={"model": {"api_endpoint": {"model_id": "test-model"}}},
        model_id="meta/llama-3.1-70b-instruct",
        metrics={
            "compute_eval_pass_at_1": 0.61,
            "compute_eval_pass_at_3": 0.74,
        },
        harness="compute-eval",
        task="compute-eval.CUDA",
        container="nvcr.io/nvidia/eval-factory/compute-eval:26.01",
        executor="local",
        invocation_id="abc123",
        job_id="abc123.0",
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_replaces_slashes(self):
        assert _sanitize_filename("nvidia/compute-eval") == "nvidia_compute-eval"

    def test_preserves_hyphens_and_underscores(self):
        assert _sanitize_filename("my-dataset_v2") == "my-dataset_v2"

    def test_replaces_special_chars(self):
        assert _sanitize_filename("org/name.with.dots") == "org_name_with_dots"


# ---------------------------------------------------------------------------
# _select_metric
# ---------------------------------------------------------------------------


class TestSelectMetric:
    def test_exact_substring_match(self):
        metrics = {"compute_eval_pass_at_1": 0.61, "compute_eval_pass_at_3": 0.74}
        key, value = _select_metric(metrics, "pass_at_1")
        assert key == "compute_eval_pass_at_1"
        assert value == 0.61

    def test_first_metric_when_none(self):
        metrics = {"accuracy": 0.85, "f1": 0.82}
        key, value = _select_metric(metrics, None)
        assert key == "accuracy"
        assert value == 0.85

    def test_raises_on_empty_metrics(self):
        with pytest.raises(ValueError, match="No metrics found"):
            _select_metric({}, "anything")

    def test_raises_on_missing_metric(self):
        metrics = {"accuracy": 0.85}
        with pytest.raises(ValueError, match="not found in results"):
            _select_metric(metrics, "pass_at_1")


# ---------------------------------------------------------------------------
# _build_eval_result_entry
# ---------------------------------------------------------------------------


class TestBuildEvalResultEntry:
    def test_minimal_entry(self):
        entry = _build_eval_result_entry(
            dataset_id="nvidia/compute-eval",
            task_id="compute_eval",
            value=61.0,
            date="2026-03-03",
        )
        assert entry == {
            "dataset": {
                "id": "nvidia/compute-eval",
                "task_id": "compute_eval",
            },
            "value": 61.0,
            "date": "2026-03-03",
        }

    def test_full_entry(self):
        entry = _build_eval_result_entry(
            dataset_id="nvidia/compute-eval",
            task_id="compute_eval",
            value=61.0,
            date="2026-03-03",
            notes="Pass@1",
            source_url="https://example.com/logs",
            source_name="Eval Logs",
        )
        assert entry["notes"] == "Pass@1"
        assert entry["source"]["url"] == "https://example.com/logs"
        assert entry["source"]["name"] == "Eval Logs"

    def test_source_without_name(self):
        entry = _build_eval_result_entry(
            dataset_id="d",
            task_id="t",
            value=0.5,
            date="2026-01-01",
            source_url="https://example.com",
        )
        assert entry["source"] == {"url": "https://example.com"}

    def test_no_source_or_notes_when_none(self):
        entry = _build_eval_result_entry(
            dataset_id="d",
            task_id="t",
            value=0.5,
            date="2026-01-01",
        )
        assert "source" not in entry
        assert "notes" not in entry


# ---------------------------------------------------------------------------
# _format_eval_results_yaml
# ---------------------------------------------------------------------------


class TestFormatEvalResultsYaml:
    def test_produces_valid_yaml(self):
        entries = [
            _build_eval_result_entry(
                dataset_id="nvidia/compute-eval",
                task_id="compute_eval",
                value=61.0,
                date="2026-03-03",
            )
        ]
        yaml_str = _format_eval_results_yaml(entries)
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["dataset"]["id"] == "nvidia/compute-eval"
        assert parsed[0]["value"] == 61.0

    def test_multiple_entries(self):
        entries = [
            _build_eval_result_entry("d", "t1", 0.5, "2026-01-01"),
            _build_eval_result_entry("d", "t2", 0.7, "2026-01-01"),
        ]
        yaml_str = _format_eval_results_yaml(entries)
        parsed = yaml.safe_load(yaml_str)
        assert len(parsed) == 2
        assert parsed[0]["dataset"]["task_id"] == "t1"
        assert parsed[1]["dataset"]["task_id"] == "t2"


# ---------------------------------------------------------------------------
# HuggingFaceExporterConfig
# ---------------------------------------------------------------------------


class TestHuggingFaceExporterConfig:
    def test_required_fields(self):
        with pytest.raises(Exception):
            HuggingFaceExporterConfig()

    def test_valid_config(self, sample_config):
        assert sample_config.dataset_id == "nvidia/compute-eval"
        assert sample_config.task_id == "compute_eval"
        assert sample_config.create_pr is True

    def test_token_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        config = HuggingFaceExporterConfig(
            dataset_id="d",
            task_id="t",
            model_repo="m",
        )
        assert config.token == "hf_env_token"

    def test_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        config = HuggingFaceExporterConfig(
            dataset_id="d",
            task_id="t",
            model_repo="m",
            token="hf_explicit",
        )
        assert config.token == "hf_explicit"


# ---------------------------------------------------------------------------
# HuggingFaceExporter.export_jobs
# ---------------------------------------------------------------------------


class TestHuggingFaceExporterExportJobs:
    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_successful_export(
        self, mock_hf_api_cls, sample_config, sample_data_for_export
    ):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        exporter = HuggingFaceExporter(sample_config)

        # Patch the lazy import inside export_jobs
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            # Re-patch to use our mock
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                success, failed, skipped = exporter.export_jobs(
                    [sample_data_for_export]
                )

        assert sample_data_for_export.job_id in success
        assert len(failed) == 0
        assert len(skipped) == 0

        mock_api.upload_file.assert_called_once()
        call_kwargs = mock_api.upload_file.call_args[1]
        assert call_kwargs["repo_id"] == "nvidia/test-model"
        assert call_kwargs["repo_type"] == "model"
        assert call_kwargs["create_pr"] is True
        assert ".eval_results/" in call_kwargs["path_in_repo"]

        uploaded_yaml = call_kwargs["path_or_fileobj"].decode("utf-8")
        parsed = yaml.safe_load(uploaded_yaml)
        assert parsed[0]["dataset"]["id"] == "nvidia/compute-eval"
        assert parsed[0]["dataset"]["task_id"] == "compute_eval"
        assert parsed[0]["value"] == 0.61

    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_upload_failure(
        self, mock_hf_api_cls, sample_config, sample_data_for_export
    ):
        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception("403 Forbidden")
        mock_hf_api_cls.return_value = mock_api

        exporter = HuggingFaceExporter(sample_config)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                success, failed, skipped = exporter.export_jobs(
                    [sample_data_for_export]
                )

        assert len(success) == 0
        assert sample_data_for_export.job_id in failed

    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_metric_not_found(self, mock_hf_api_cls, sample_data_for_export):
        config = HuggingFaceExporterConfig(
            dataset_id="nvidia/compute-eval",
            task_id="compute_eval",
            model_repo="nvidia/test-model",
            metric_name="nonexistent_metric",
            token="hf_test",
        )
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        exporter = HuggingFaceExporter(config)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                success, failed, skipped = exporter.export_jobs(
                    [sample_data_for_export]
                )

        assert len(success) == 0
        assert sample_data_for_export.job_id in failed
        mock_api.upload_file.assert_not_called()

    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_multiple_jobs(
        self, mock_hf_api_cls, sample_config, sample_data_for_export
    ):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        data2 = DataForExport(
            artifacts_dir=Path("/tmp/artifacts2"),
            logs_dir=None,
            config={},
            model_id="meta/llama-3.1-70b-instruct",
            metrics={"compute_eval_pass_at_1": 0.55},
            harness="compute-eval",
            task="compute-eval.CCCL",
            container="nvcr.io/nvidia/eval-factory/compute-eval:26.01",
            executor="local",
            invocation_id="def456",
            job_id="def456.0",
            timestamp=time.time(),
        )

        exporter = HuggingFaceExporter(sample_config)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                success, failed, skipped = exporter.export_jobs(
                    [sample_data_for_export, data2]
                )

        assert len(success) == 2
        assert len(failed) == 0

        uploaded_yaml = mock_api.upload_file.call_args[1]["path_or_fileobj"].decode(
            "utf-8"
        )
        parsed = yaml.safe_load(uploaded_yaml)
        assert len(parsed) == 2

    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_direct_push_mode(self, mock_hf_api_cls, sample_data_for_export):
        config = HuggingFaceExporterConfig(
            dataset_id="nvidia/compute-eval",
            task_id="compute_eval",
            model_repo="nvidia/test-model",
            metric_name="pass_at_1",
            token="hf_test",
            create_pr=False,
        )
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        exporter = HuggingFaceExporter(config)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                exporter.export_jobs([sample_data_for_export])

        assert mock_api.upload_file.call_args[1]["create_pr"] is False

    @patch("nemo_evaluator_launcher.exporters.huggingface.HfApi")
    def test_notes_include_container(
        self, mock_hf_api_cls, sample_config, sample_data_for_export
    ):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        exporter = HuggingFaceExporter(sample_config)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            with patch(
                "nemo_evaluator_launcher.exporters.huggingface.HfApi",
                return_value=mock_api,
            ):
                exporter.export_jobs([sample_data_for_export])

        uploaded_yaml = mock_api.upload_file.call_args[1]["path_or_fileobj"].decode(
            "utf-8"
        )
        parsed = yaml.safe_load(uploaded_yaml)
        notes = parsed[0]["notes"]
        assert "Pass@1" in notes
        assert "compute-eval:26.01" in notes
