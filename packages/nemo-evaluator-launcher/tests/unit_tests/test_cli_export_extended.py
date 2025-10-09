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
"""Extended tests for export CLI command to increase coverage."""

from unittest.mock import patch

from nemo_evaluator_launcher.cli.export import ExportCmd


class TestExportCmdConfiguration:
    """Test export command configuration building."""

    def test_basic_config_building(self, capsys):
        """Test basic configuration building with minimal args."""
        mock_result = {
            "success": True,
            "jobs": {"test123.0": {"success": True, "message": "Success"}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 1 invocation to local" in captured.out

    def test_all_config_options(self, capsys):
        """Test configuration with all options set."""

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                output_dir="/custom/path",
                output_filename="custom_results.json",
                format="json",
                copy_logs=True,
                log_metrics=["score", "accuracy"],
                only_required=False,
            )
            cmd.execute()

        # Verify the config was built correctly
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        config = call_args[0][2]  # Third argument is config

        assert config["output_dir"] == "/custom/path"
        assert config["output_filename"] == "custom_results.json"
        assert config["format"] == "json"
        assert config["copy_logs"] is True
        assert config["log_metrics"] == ["score", "accuracy"]
        assert config["only_required"] is False

    def test_config_with_none_values(self, capsys):
        """Test configuration when optional values are None."""

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                output_dir=None,
                output_filename=None,
                format=None,
                copy_logs=False,
                log_metrics=[],
                only_required=True,
            )
            cmd.execute()

        # Verify None values are not included in config
        call_args = mock_export.call_args
        config = call_args[0][2]

        assert "output_dir" not in config
        assert "output_filename" not in config
        assert "format" not in config
        assert config["copy_logs"] is False
        assert "only_required" not in config
        assert "log_metrics" not in config  # Empty list should not be included


class TestExportCmdOutputMessages:
    """Test export command output messages."""

    def test_format_warning_for_non_local_dest(self, capsys):
        """Test warning message when format is used with non-local destination."""
        mock_result = {
            "success": True,
            "jobs": {"test123.0": {"success": True, "message": "Success"}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                format="json",
            )
            cmd.execute()

        captured = capsys.readouterr()
        assert "Note: --format is only used by --dest local" in captured.out
        assert "will be ignored for other destinations" in captured.out

    def test_no_format_warning_for_local_dest(self, capsys):
        """Test no warning when format is used with local destination."""
        mock_result = {
            "success": True,
            "jobs": {"test123.0": {"success": True, "message": "Success"}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                format="json",
            )
            cmd.execute()

        captured = capsys.readouterr()
        assert "will be ignored" not in captured.out

    def test_single_invocation_export_message(self, capsys):
        """Test export message for single invocation."""
        mock_result = {
            "success": True,
            "jobs": {"test123.0": {"success": True, "message": "Success"}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 1 invocation to local" in captured.out

    def test_multiple_invocations_export_message(self, capsys):
        """Test export message for multiple invocations."""
        mock_result = {"success": True, "invocations": {}}

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123", "test456"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 2 invocations to local" in captured.out


class TestExportCmdFailureHandling:
    """Test export command failure handling."""

    def test_export_failure(self, capsys):
        """Test handling of export failure."""
        mock_result = {"success": False, "error": "Network connection failed"}

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export failed: Network connection failed" in captured.out

    def test_export_failure_no_error_message(self, capsys):
        """Test handling of export failure without error message."""
        mock_result = {"success": False}

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export failed: Unknown error" in captured.out


class TestExportCmdSingleInvocationResults:
    """Test export command output for single invocation results."""

    def test_single_invocation_success_with_url(self, capsys):
        """Test single invocation success output with URL."""
        mock_result = {
            "success": True,
            "jobs": {
                "test123.0": {
                    "success": True,
                    "message": "Exported successfully",
                    "metadata": {
                        "run_url": "https://wandb.ai/test/project/runs/test123",
                        "summary_path": "/path/to/summary.json",
                    },
                }
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for test123" in captured.out
        assert "test123.0: Exported successfully" in captured.out
        assert "URL: https://wandb.ai/test/project/runs/test123" in captured.out
        assert "Summary: /path/to/summary.json" in captured.out

    def test_single_invocation_success_without_metadata(self, capsys):
        """Test single invocation success output without metadata."""
        mock_result = {
            "success": True,
            "jobs": {
                "test123.0": {"success": True, "message": "Exported successfully"}
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for test123" in captured.out
        assert "test123.0: Exported successfully" in captured.out
        assert "URL:" not in captured.out
        assert "Summary:" not in captured.out

    def test_single_invocation_job_failure(self, capsys):
        """Test single invocation with job failure."""
        mock_result = {
            "success": True,
            "jobs": {"test123.0": {"success": False, "message": "Job export failed"}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for test123" in captured.out
        assert "test123.0 failed: Job export failed" in captured.out

    def test_single_invocation_multiple_jobs(self, capsys):
        """Test single invocation with multiple jobs."""
        mock_result = {
            "success": True,
            "jobs": {
                "test123.0": {"success": True, "message": "Job 0 exported"},
                "test123.1": {
                    "success": True,
                    "message": "Job 1 exported",
                    "metadata": {"run_url": "https://example.com/run1"},
                },
                "test123.2": {"success": False, "message": "Job 2 failed"},
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for test123" in captured.out
        assert "test123.0: Job 0 exported" in captured.out
        assert "test123.1: Job 1 exported" in captured.out
        assert "URL: https://example.com/run1" in captured.out
        assert "test123.2 failed: Job 2 failed" in captured.out


class TestExportCmdMultipleInvocationsResults:
    """Test export command output for multiple invocations results."""

    def test_multiple_invocations_success(self, capsys):
        """Test multiple invocations success output."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_invocations": 2,
                "total_invocations": 2,
                "summary_path": "/path/to/combined_summary.json",
            },
            "invocations": {
                "test123": {"success": True, "jobs": {"test123.0": {"success": True}}},
                "test456": {"success": True, "jobs": {"test456.0": {"success": True}}},
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123", "test456"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 2/2 successful" in captured.out
        assert "Summary: /path/to/combined_summary.json" in captured.out
        assert "test123: 1 jobs" in captured.out
        assert "test456: 1 jobs" in captured.out

    def test_multiple_invocations_partial_success(self, capsys):
        """Test multiple invocations with partial success."""
        mock_result = {
            "success": True,
            "metadata": {"successful_invocations": 1, "total_invocations": 2},
            "invocations": {
                "test123": {
                    "success": True,
                    "jobs": {
                        "test123.0": {"success": True},
                        "test123.1": {"success": True},
                    },
                },
                "test456": {"success": False, "error": "Invocation not found"},
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123", "test456"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 1/2 successful" in captured.out
        assert "test123: 2 jobs" in captured.out
        assert "test456: failed, Invocation not found" in captured.out

    def test_multiple_invocations_no_summary_path(self, capsys):
        """Test multiple invocations without summary path in metadata."""
        mock_result = {
            "success": True,
            "metadata": {"successful_invocations": 1, "total_invocations": 1},
            "invocations": {
                "test123": {"success": True, "jobs": {"test123.0": {"success": True}}}
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(
                invocation_ids=["test123", "test456"], dest="wandb"
            )  # Multiple invocations
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 1/1 successful" in captured.out
        assert "Summary:" not in captured.out

    def test_multiple_invocations_missing_metadata(self, capsys):
        """Test multiple invocations with missing metadata fields."""
        mock_result = {
            "success": True,
            "metadata": {},  # Missing required fields
            "invocations": {"test123": {"success": True, "jobs": {}}},
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(
                invocation_ids=["test123", "test456"], dest="local"
            )  # Multiple invocations
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 0/0 successful" in captured.out

    def test_multiple_invocations_failure_without_error(self, capsys):
        """Test multiple invocations with failure but no error message."""
        mock_result = {
            "success": True,
            "metadata": {"successful_invocations": 0, "total_invocations": 1},
            "invocations": {
                "test123": {
                    "success": False
                    # Missing error field
                }
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(
                invocation_ids=["test123", "test456"], dest="local"
            )  # Multiple invocations
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 0/1 successful" in captured.out
        assert "test123: failed, Unknown error" in captured.out


class TestExportCmdEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_invocation_ids_list(self, capsys):
        """Test with empty invocation IDs list."""
        mock_result = {"success": True, "invocations": {}}

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=[], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        # empty IDs returns early with error
        assert "Error: No IDs provided" in captured.out
        assert "Usage:" in captured.out

    def test_single_invocation_no_jobs_in_result(self, capsys):
        """Test single invocation with no jobs in result."""
        mock_result = {"success": True, "jobs": {}}

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for test123" in captured.out

    def test_result_missing_expected_fields(self, capsys):
        """Test handling of result missing expected fields."""
        mock_result = {
            "success": True,
            "jobs": {},
        }  # Missing some jobs but has jobs key

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        # Should handle gracefully without crashing
        assert "Export completed" in captured.out

    def test_job_metadata_missing_fields(self, capsys):
        """Test job with partial metadata."""
        mock_result = {
            "success": True,
            "jobs": {
                "test123.0": {
                    "success": True,
                    "message": "Success",
                    "metadata": {
                        "run_url": "https://example.com/run"
                        # Missing summary_path
                    },
                }
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "URL: https://example.com/run" in captured.out
        assert "Summary:" not in captured.out


class TestExportCmdOverrides:
    """Test export command override functionality."""

    def test_override_with_export_prefix(self, capsys):
        """Test override using export.<dest>.key=value format."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.tracking_uri=http://mlflow:5000",
                    "export.mlflow.experiment_name=test-exp",
                    "export.mlflow.log_logs=true",
                ],
            )
            cmd.execute()

        # Verify overrides were applied correctly
        call_args = mock_export.call_args
        config = call_args[0][2]  # Third argument is config

        assert config["tracking_uri"] == "http://mlflow:5000"
        assert config["experiment_name"] == "test-exp"
        assert config["log_logs"] is True

    def test_override_with_plain_keys(self, capsys):
        """Test override using plain keys (no export. prefix)."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                override=[
                    "entity=org-name",
                    "project=proj-name",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        assert config["entity"] == "org-name"
        assert config["project"] == "proj-name"

    def test_override_with_list_values(self, capsys):
        """Test override with list values."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                override=[
                    # Hydra list syntax: use brackets and commas
                    "export.wandb.tags=[exp1,model1]",
                    "export.wandb.log_metrics=[accuracy,pass@1]",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        # Verify list parsing works correctly
        assert isinstance(config["tags"], list)
        assert "exp1" in config["tags"]
        assert "model1" in config["tags"]
        assert config["log_metrics"] == ["accuracy", "pass@1"]

    def test_override_with_nested_dict(self, capsys):
        """Test override with nested dictionary values."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.tags.framework=vllm",
                    "export.mlflow.tags.precision=bf16",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        assert config["tags"]["framework"] == "vllm"
        assert config["tags"]["precision"] == "bf16"

    def test_override_ignores_other_destinations(self, capsys):
        """Test that overrides for other destinations raise clear error."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.tracking_uri=http://mlflow:5000",
                    "export.wandb.entity=other-org",  # wrong destination should raise error
                ],
            )
            cmd.execute()

        # Should error with clear message, and not call export_results
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "export.wandb" in captured.out
        assert "mlflow" in captured.out
        # Validation should prevent export_results from being called
        mock_export.assert_not_called()

    def test_override_merges_with_existing_flags(self, capsys):
        """Test that overrides merge with existing CLI flags."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                output_dir="/tmp/results",
                format="json",
                copy_logs=True,
                override=[
                    "export.local.log_metrics=[score,accuracy]",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        # Both CLI flags and overrides should be present
        assert config["output_dir"] == "/tmp/results"
        assert config["format"] == "json"
        assert config["copy_logs"] is True
        assert config["log_metrics"] == ["score", "accuracy"]

    def test_override_with_numeric_values(self, capsys):
        """Test override with numeric values."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.timeout=300",
                    "export.mlflow.max_retries=3",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        assert config["timeout"] == 300
        assert config["max_retries"] == 3

    def test_multiple_overrides_same_key(self, capsys):
        """Test that later overrides win when same key is specified multiple times."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                override=[
                    "export.wandb.entity=first-org",
                    "export.wandb.entity=second-org",  # Should win
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        # Last override should win
        assert config["entity"] == "second-org"

    def test_override_with_boolean_values(self, capsys):
        """Test override with boolean values."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.log_artifacts=false",
                    "export.mlflow.log_logs=true",
                ],
            )
            cmd.execute()

        call_args = mock_export.call_args
        config = call_args[0][2]

        assert config["log_artifacts"] is False
        assert config["log_logs"] is True

    def test_override_empty_list(self, capsys):
        """Test that empty override list works correctly."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                output_dir="/tmp",
                override=[],  # Empty override list
            )
            cmd.execute()

        # Should work normally with just CLI flags
        call_args = mock_export.call_args
        config = call_args[0][2]
        assert config["output_dir"] == "/tmp"
        assert len(config) >= 2  # At least output_dir + standard flags


class TestExportCmdErrorMessages:
    """Test improved error messages for common issues."""

    def test_mlflow_tracking_uri_error_message(self, capsys):
        """Test helpful message when MLflow tracking_uri is missing."""
        mock_result = {
            "success": False,
            "error": "tracking_uri is required",
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="mlflow")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export failed: tracking_uri is required" in captured.out
        assert "MLflow requires 'tracking_uri'" in captured.out
        assert "export.mlflow.tracking_uri=" in captured.out

    def test_wandb_entity_error_message(self, capsys):
        """Test helpful message when W&B entity/project is missing."""
        mock_result = {
            "success": False,
            "error": "entity and project are required",
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Export failed:" in captured.out
        assert "W&B requires 'entity' and 'project'" in captured.out
        assert "export.wandb.entity=" in captured.out

    def test_package_not_installed_error_message(self, capsys):
        """Test helpful message when exporter package is not installed."""
        mock_result = {
            "success": False,
            "error": "mlflow package not installed",
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="mlflow")
            cmd.execute()

        captured = capsys.readouterr()
        assert "MLflow package not installed" in captured.out
        assert "pip install nemo-evaluator-launcher[mlflow]" in captured.out

    def test_empty_ids_validation_message(self, capsys):
        """Test validation message when no IDs are provided."""
        cmd = ExportCmd(invocation_ids=[], dest="local")
        cmd.execute()

        captured = capsys.readouterr()
        assert "Error: No IDs provided" in captured.out
        assert "Usage: nemo-evaluator-launcher export" in captured.out


class TestExportCmdOverrideEdgeCases:
    """Test edge cases for override functionality."""

    def test_override_with_urls_and_special_chars(self, capsys):
        """Test override handles URLs with ports, paths, special characters."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.tracking_uri=http://mlflow.example.com:5000/api/v2"
                ],
            )
            cmd.execute()

        config = mock_export.call_args[0][2]
        assert config["tracking_uri"] == "http://mlflow.example.com:5000/api/v2"

    def test_override_destination_mismatch_raises_err(self, capsys):
        """Test that override for wrong destination raises clear error."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=["export.wandb.entity=wrong"],  # Wrong dest!
            )
            cmd.execute()

        captured = capsys.readouterr()
        # Should raise error msg
        assert "Error:" in captured.out
        assert "export.wandb" in captured.out
        assert "mlflow" in captured.out
        # Should NOT call export_results due to validation error
        mock_export.assert_not_called()

    def test_override_nested_metadata_paths(self, capsys):
        """Test deep nested paths for complex metadata."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                override=[
                    "export.wandb.extra_metadata.training.hardware=H100",
                    "export.wandb.extra_metadata.training.nodes=8",
                    "export.wandb.extra_metadata.dataset.name=custom",
                ],
            )
            cmd.execute()

        config = mock_export.call_args[0][2]
        assert config["extra_metadata"]["training"]["hardware"] == "H100"
        assert config["extra_metadata"]["training"]["nodes"] == 8
        assert config["extra_metadata"]["dataset"]["name"] == "custom"

    def test_plain_keys_shortcut_works(self, capsys):
        """Test that plain keys (no prefix) work as shortcut for current destination."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "tracking_uri=http://mlflow:5000",
                    "experiment_name=my-exp",
                ],
            )
            cmd.execute()

        config = mock_export.call_args[0][2]
        assert config["tracking_uri"] == "http://mlflow:5000"
        assert config["experiment_name"] == "my-exp"

    def test_override_precedence_over_cli_flags(self, capsys):
        """Test that -o overrides take precedence over convenience flags (documented behavior)."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                output_dir="/from-flag",
                override=["export.local.output_dir=/from-override"],
            )
            cmd.execute()

        config = mock_export.call_args[0][2]
        # Overrides applied after flags, so they win (dict.update semantics)
        assert config["output_dir"] == "/from-override"

    def test_default_destination_used_when_not_specified(self, capsys):
        """Test that default destination (local) is used when --dest not provided."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"t.0": {"success": True, "message": "ok"}},
            }

            # Don't specify --dest, should default to "local"
            cmd = ExportCmd(
                invocation_ids=["test123"],
                # dest defaults to "local"
                override=["export.local.format=json"],
            )
            cmd.execute()

        # Should call with dest="local"
        call_args = mock_export.call_args
        assert call_args[0][1] == "local"

    def test_override_only_for_different_dest_raises_error(self, capsys):
        """Test that ALL overrides for different destination raise error (no fallback merge)."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "jobs": {"test123.0": {"success": True, "message": "Success"}},
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.wandb.entity=my-org",  # Wrong destination
                    "export.wandb.project=my-proj",
                ],
            )
            cmd.execute()

        captured = capsys.readouterr()
        # Validation should error and prevent export
        assert "Error:" in captured.out
        assert "export.wandb" in captured.out
        assert "mlflow" in captured.out
        # Should NOT call export_results
        mock_export.assert_not_called()
