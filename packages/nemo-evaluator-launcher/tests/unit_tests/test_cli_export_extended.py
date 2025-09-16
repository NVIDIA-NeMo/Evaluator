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
            "nemo_evaluator_launcher.cli.export.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 1 invocation to local" in captured.out

    def test_all_config_options(self, capsys):
        """Test configuration with all options set."""

        with patch("nemo_evaluator_launcher.cli.export.export_results") as mock_export:
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

        with patch("nemo_evaluator_launcher.cli.export.export_results") as mock_export:
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
        assert config["only_required"] is True
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=[], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        # The logic uses "invocation" vs "invocations" based on count
        assert "Exporting 0 invocation to local" in captured.out

    def test_single_invocation_no_jobs_in_result(self, capsys):
        """Test single invocation with no jobs in result."""
        mock_result = {"success": True, "jobs": {}}

        with patch(
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
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
            "nemo_evaluator_launcher.cli.export.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "URL: https://example.com/run" in captured.out
        assert "Summary:" not in captured.out
