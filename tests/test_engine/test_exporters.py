"""Tests for engine/exporters — contract tests with mocked backends."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestExporterRegistry:
    def test_lazy_load_populates(self):
        from nemo_evaluator.engine.exporters import _lazy_load, _REGISTRY

        _lazy_load()
        assert "wandb" in _REGISTRY
        assert "mlflow" in _REGISTRY
        assert "inspect" in _REGISTRY

    def test_get_unknown_raises(self):
        from nemo_evaluator.engine.exporters import get_exporter

        with pytest.raises(KeyError, match="Unknown exporter"):
            get_exporter("nonexistent_exporter_xyz")


class TestWandBExporter:
    def test_init(self):
        from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

        exp = WandBExporter(project="test-proj", entity="test-entity", tags=["ci"])
        assert exp._project == "test-proj"
        assert exp._entity == "test-entity"
        assert exp._tags == ["ci"]

    def test_export_calls_wandb(self):
        from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        exp = WandBExporter(project="test")
        bundles = [
            {
                "benchmark": {
                    "name": "mmlu",
                    "samples": 10,
                    "scores": {"pass@1": {"value": 0.8}},
                },
                "_results": [],
            }
        ]
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            exp.export(bundles)
        mock_wandb.init.assert_called_once()


# MLflow exporter tests live in tests/test_engine/test_mlflow_exporter.py.


class TestInspectExporter:
    def test_default_format_is_eval(self):
        pytest.importorskip("inspect_ai")
        from nemo_evaluator.engine.exporters.inspect_export import InspectExporter

        assert InspectExporter()._format == "eval"
        assert InspectExporter(format="json")._format == "json"

    def test_unsupported_format_raises(self):
        pytest.importorskip("inspect_ai")
        from nemo_evaluator.engine.exporters.inspect_export import InspectExporter

        with pytest.raises(ValueError, match="unsupported format"):
            InspectExporter(format="yaml")

    def test_atif_tool_calls_become_inspect_tool_calls(self):
        pytest.importorskip("inspect_ai")
        from nemo_evaluator.engine.exporters.inspect_export import (
            _atif_steps_to_messages,
        )

        trajectory = [
            {
                "steps": [
                    {"source": "user", "message": "hi"},
                    {
                        "source": "agent",
                        "message": "Executed Bash toolu_bdrk_0151rmB4xGKjdDaM4eEbVmJh",
                        "tool_calls": [
                            {
                                "tool_call_id": "toolu_bdrk_0151rmB4xGKjdDaM4eEbVmJh",
                                "function_name": "Bash",
                                "arguments": {"command": "ls /tmp", "description": "list"},
                            }
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "toolu_bdrk_0151rmB4xGKjdDaM4eEbVmJh",
                                    "content": "file1\nfile2\n",
                                }
                            ]
                        },
                    },
                ]
            }
        ]

        messages = _atif_steps_to_messages(trajectory)
        assert [m.role for m in messages] == ["user", "assistant", "tool"]

        assistant = messages[1]
        assert assistant.content == "", "synthetic 'Executed …' label must be stripped"
        assert assistant.tool_calls and len(assistant.tool_calls) == 1
        call = assistant.tool_calls[0]
        assert call.id == "toolu_bdrk_0151rmB4xGKjdDaM4eEbVmJh"
        assert call.function == "Bash"
        assert call.arguments == {"command": "ls /tmp", "description": "list"}

        tool_msg = messages[2]
        assert tool_msg.tool_call_id == "toolu_bdrk_0151rmB4xGKjdDaM4eEbVmJh"
        assert tool_msg.function == "Bash"
        assert tool_msg.content == "file1\nfile2\n"

    def test_agent_message_without_tool_calls_preserved_verbatim(self):
        pytest.importorskip("inspect_ai")
        from nemo_evaluator.engine.exporters.inspect_export import (
            _atif_steps_to_messages,
        )

        trajectory = [
            {
                "steps": [
                    {"source": "agent", "message": "Thinking out loud here."},
                ]
            }
        ]
        messages = _atif_steps_to_messages(trajectory)
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Thinking out loud here."
        assert not messages[0].tool_calls
