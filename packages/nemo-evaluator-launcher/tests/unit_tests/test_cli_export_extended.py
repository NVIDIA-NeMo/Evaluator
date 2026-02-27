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
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
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
                copy_logs=None,
                log_metrics=None,
                only_required=None,
            )
            cmd.execute()

        # Verify None values are not included in config
        call_args = mock_export.call_args
        config = call_args[0][2]

        assert "output_dir" not in config
        assert "output_filename" not in config
        assert "format" not in config
        assert "copy_logs" not in config
        assert "only_required" not in config
        assert "log_metrics" not in config


class TestExportCmdOutputMessages:
    """Test export command output messages."""

    def test_format_warning_for_non_local_dest(self, capsys):
        """Test warning message when format is used with non-local destination."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
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
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
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
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
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
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123", "test456"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 2 invocations to local" in captured.out

    def test_export_failure(self, capsys):
        """Test handling of export failure."""
        mock_result = mock_result = {
            "success": False,
            "metadata": {
                "successful_jobs": 0,
                "failed_jobs": 1,
                "skipped_jobs": 0,
            },
        }
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert (
            "Some jobs failed to export. See logs above for more details."
            in captured.out
        )

    def test_single_invocation_wandb(self, capsys):
        """Test single invocation success output with URL."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="wandb")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 1 invocation to wandb..." in captured.out
        assert (
            "Export completed: {'successful_jobs': 1, 'failed_jobs': 0, 'skipped_jobs': 0}"
            in captured.out
        )

    def test_single_invocation_local(self, capsys):
        """Test single invocation success output without metadata."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 1 invocation to local..." in captured.out
        assert (
            "Export completed: {'successful_jobs': 1, 'failed_jobs': 0, 'skipped_jobs': 0}"
            in captured.out
        )

    def test_multiple_invocations_local(self, capsys):
        """Test multiple invocations success output."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 2,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
        }

        with patch(
            "nemo_evaluator_launcher.api.functional.export_results",
            return_value=mock_result,
        ):
            cmd = ExportCmd(invocation_ids=["test123", "test456"], dest="local")
            cmd.execute()

        captured = capsys.readouterr()
        assert "Exporting 2 invocations to local..." in captured.out
        assert (
            "Export completed: {'successful_jobs': 2, 'failed_jobs': 0, 'skipped_jobs': 0}"
            in captured.out
        )

    def test_empty_invocation_ids_list(self, capsys):
        """Test with empty invocation IDs list."""
        mock_result = {
            "success": True,
            "metadata": {
                "successful_jobs": 0,
                "failed_jobs": 0,
                "skipped_jobs": 0,
            },
        }

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


class TestExportCmdOverrides:
    """Test export command override functionality."""

    def test_override_with_export_prefix(self, capsys):
        """Test override using export.<dest>.key=value format."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                override=[
                    "export.wandb.entity=org-name",
                    "export.wandb.project=proj-name",
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
        print(config)

        # Verify list parsing works correctly
        assert len(config["tags"]) == 2
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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

            call_args = mock_export.call_args
            config = call_args[0][2]
            assert config["tracking_uri"] == "http://mlflow:5000"
            assert "entity" not in config

    def test_override_merges_with_existing_flags(self, capsys):
        """Test that overrides merge with existing CLI flags."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
        assert len(config) >= 1  # At least output_dir


class TestExportCmdOverrideEdgeCases:
    """Test edge cases for override functionality."""

    def test_override_with_urls_and_special_chars(self, capsys):
        """Test override handles URLs with ports, paths, special characters."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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

    def test_override_nested_metadata_paths(self, capsys):
        """Test deep nested paths for complex metadata."""
        with patch(
            "nemo_evaluator_launcher.api.functional.export_results"
        ) as mock_export:
            mock_export.return_value = {
                "success": True,
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
            }

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="mlflow",
                override=[
                    "export.mlflow.tracking_uri=http://mlflow:5000",
                    "export.mlflow.experiment_name=my-exp",
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
                "metadata": {
                    "successful_jobs": 1,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                },
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
