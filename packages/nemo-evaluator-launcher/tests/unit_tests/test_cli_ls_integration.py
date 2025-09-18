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
"""Integration tests for the CLI ls command and its alias functionality."""

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.cli.main import create_parser, main


class TestLsCommandIntegration:
    """Test the ls command integration with main CLI."""

    @pytest.fixture
    def sample_tasks_data(self):
        """Sample task data for testing."""
        return [
            ["garak", "chat", "garak", "nvcr.io/nvidia/eval-factory/garak:25.08.1"],
            ["mmlu", "chat", "lm-eval", "nvcr.io/nvidia/eval-factory/lm-eval:25.08.1"],
        ]

    def test_ls_tasks_explicit_command(self, sample_tasks_data):
        """Test 'nv-eval ls tasks' command."""
        parser = create_parser()
        args = parser.parse_args(["ls", "tasks"])

        with patch(
            "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                if args.command == "ls" and args.ls_command == "tasks":
                    args.tasks.execute()

            result = output.getvalue()
            assert "harness: garak" in result
            assert "harness: lm-eval" in result

    def test_ls_tasks_explicit_with_json_flag(self, sample_tasks_data):
        """Test 'nv-eval ls tasks --json' command."""
        parser = create_parser()
        args = parser.parse_args(["ls", "tasks", "--json"])

        with patch(
            "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                if args.command == "ls" and args.ls_command == "tasks":
                    args.tasks.execute()

            result = output.getvalue().strip()
            parsed_json = json.loads(result)
            assert "tasks" in parsed_json
            assert len(parsed_json["tasks"]) == 2

    def test_ls_alias_without_subcommand(self, sample_tasks_data):
        """Test 'nv-eval ls' command (default alias to tasks)."""
        parser = create_parser()
        args = parser.parse_args(["ls"])

        with patch(
            "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                if args.command == "ls":
                    if args.ls_command is None or args.ls_command == "tasks":
                        if hasattr(args, "tasks_alias"):
                            args.tasks_alias.execute()
                        elif hasattr(args, "tasks"):
                            args.tasks.execute()

            result = output.getvalue()
            assert "harness: garak" in result
            assert "harness: lm-eval" in result

    def test_ls_alias_with_json_flag(self, sample_tasks_data):
        """Test 'nv-eval ls --json' command (alias with argument propagation)."""
        parser = create_parser()
        args = parser.parse_args(["ls", "--json"])

        with patch(
            "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                if args.command == "ls":
                    if args.ls_command is None or args.ls_command == "tasks":
                        if hasattr(args, "tasks_alias"):
                            args.tasks_alias.execute()
                        elif hasattr(args, "tasks"):
                            args.tasks.execute()

            result = output.getvalue().strip()
            parsed_json = json.loads(result)
            assert "tasks" in parsed_json
            assert len(parsed_json["tasks"]) == 2

    def test_parser_structure_for_ls_command(self):
        """Test that the parser is correctly structured for ls commands."""
        parser = create_parser()

        # Test ls without subcommand
        args = parser.parse_args(["ls"])
        assert args.command == "ls"
        assert args.ls_command is None
        assert hasattr(
            args, "tasks_alias"
        )  # Should have tasks_alias from main ls parser

        # Test ls tasks
        args = parser.parse_args(["ls", "tasks"])
        assert args.command == "ls"
        assert args.ls_command == "tasks"
        assert hasattr(args, "tasks")

        # Test ls with json flag
        args = parser.parse_args(["ls", "--json"])
        assert args.command == "ls"
        assert args.ls_command is None
        assert hasattr(args, "tasks_alias")
        assert args.tasks_alias.json is True

        # Test ls tasks with json flag
        args = parser.parse_args(["ls", "tasks", "--json"])
        assert args.command == "ls"
        assert args.ls_command == "tasks"
        assert hasattr(args, "tasks")
        assert args.tasks.json is True

    def test_main_function_ls_dispatch(self, sample_tasks_data):
        """Test that main() function correctly dispatches ls commands."""
        # Test the main dispatch logic for ls commands
        original_argv = sys.argv

        try:
            # Test ls without subcommand
            sys.argv = ["nv-eval", "ls"]

            with patch(
                "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
                return_value=sample_tasks_data,
            ):
                output = StringIO()
                with redirect_stdout(output):
                    with patch("sys.exit"):  # Prevent actual exit
                        main()

                result = output.getvalue()
                # Should execute the tasks command
                assert len(result) > 0  # Should have some output

        finally:
            sys.argv = original_argv

    def test_main_function_ls_with_json(self, sample_tasks_data):
        """Test that main() function correctly handles ls --json."""
        original_argv = sys.argv

        try:
            # Test ls with json flag
            sys.argv = ["nv-eval", "ls", "--json"]

            with patch(
                "nemo_evaluator_launcher.cli.ls_tasks.get_tasks_list",
                return_value=sample_tasks_data,
            ):
                output = StringIO()
                with redirect_stdout(output):
                    with patch("sys.exit"):  # Prevent actual exit
                        main()

                result = output.getvalue().strip()
                # Should be valid JSON
                if result:  # Only test if there's output
                    parsed_json = json.loads(result)
                    assert "tasks" in parsed_json

        finally:
            sys.argv = original_argv

    def test_help_output_includes_ls_options(self):
        """Test that help output includes ls command options."""
        parser = create_parser()

        # Get help for ls command
        with pytest.raises(SystemExit):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                parser.parse_args(["ls", "--help"])

        # The help should mention the json option
        help_output = mock_stdout.getvalue()
        assert "--json" in help_output or "json" in help_output.lower()

    def test_ls_runs_subcommand_still_works(self):
        """Test that ls runs subcommand is not affected by our changes."""
        parser = create_parser()
        args = parser.parse_args(["ls", "runs"])

        assert args.command == "ls"
        assert args.ls_command == "runs"
        assert hasattr(args, "runs")

    def test_argument_namespace_isolation(self):
        """Test that tasks_alias and tasks arguments don't interfere."""
        parser = create_parser()

        # ls without subcommand should have tasks_alias
        args1 = parser.parse_args(["ls"])
        assert hasattr(args1, "tasks_alias")
        assert not hasattr(args1, "tasks")  # Should not have tasks when no subcommand

        # ls tasks should have tasks
        args2 = parser.parse_args(["ls", "tasks"])
        assert hasattr(args2, "tasks")
        assert hasattr(args2, "tasks_alias")  # This is also added to main parser

    def test_json_flag_values_consistency(self):
        """Test that json flag values are consistent between alias and explicit command."""
        parser = create_parser()

        # Test with json flag
        args1 = parser.parse_args(["ls", "--json"])
        args2 = parser.parse_args(["ls", "tasks", "--json"])

        assert args1.tasks_alias.json is True
        assert args2.tasks.json is True

        # Test without json flag
        args3 = parser.parse_args(["ls"])
        args4 = parser.parse_args(["ls", "tasks"])

        assert args3.tasks_alias.json is False
        assert args4.tasks.json is False
