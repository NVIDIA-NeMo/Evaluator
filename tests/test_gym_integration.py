"""Tests for Gym environment integration: protocol, client, lifecycle, registry.

After the senior-engineer review the Gym integration uses:

- ``GymDataset`` -- JSONL loader (no HTTP)
- ``GymEnvironment`` -- single HTTP client with ``protocol`` param
  (``"evaluator"`` or ``"native"``) and optional ``GymDataset``
- ``ManagedGymEnvironment`` -- subprocess lifecycle, delegates to above

Tests are grouped by component.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest



# ---------------------------------------------------------------------------
# gym_protocol helpers
# ---------------------------------------------------------------------------

class TestGymProtocol:

    def test_wrap_text_as_gym_response(self):
        from nemo_evaluator.environments.gym_protocol import wrap_text_as_gym_response

        result = wrap_text_as_gym_response("SELECT * FROM users;")

        assert result["output_text"] == "SELECT * FROM users;"
        assert result["status"] == "completed"
        msg = result["output"][0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "SELECT * FROM users;"

    def test_wrap_text_as_gym_response_empty(self):
        from nemo_evaluator.environments.gym_protocol import wrap_text_as_gym_response
        assert wrap_text_as_gym_response("")["output_text"] == ""

    def test_wrap_text_as_responses_create_params(self):
        from nemo_evaluator.environments.gym_protocol import wrap_text_as_responses_create_params

        result = wrap_text_as_responses_create_params("What is SQL?")
        assert result["input"] == [{"role": "user", "content": "What is SQL?"}]
        assert result["model"] == "evaluator"

    def test_extract_prompt_from_rcp(self):
        from nemo_evaluator.environments.gym_protocol import extract_prompt_from_rcp

        assert extract_prompt_from_rcp({"input": "hello"}) == "hello"
        assert extract_prompt_from_rcp({
            "input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        }) == "hi"
        assert extract_prompt_from_rcp({}) == ""

    def test_messages_from_rcp(self):
        from nemo_evaluator.environments.gym_protocol import messages_from_rcp

        msgs = messages_from_rcp({
            "input": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        })
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "q"}

    def test_extract_assistant_text_plain_string(self):
        from nemo_evaluator.environments.gym_protocol import extract_assistant_text
        assert extract_assistant_text("hello") == "hello"

    def test_extract_assistant_text_gym_response(self):
        from nemo_evaluator.environments.gym_protocol import extract_assistant_text, wrap_text_as_gym_response
        assert extract_assistant_text(wrap_text_as_gym_response("SQL;")) == "SQL;"

    def test_extract_assistant_text_openai_format(self):
        from nemo_evaluator.environments.gym_protocol import extract_assistant_text
        resp = {"choices": [{"message": {"content": "42"}}]}
        assert extract_assistant_text(resp) == "42"

    def test_extract_assistant_text_fallback(self):
        from nemo_evaluator.environments.gym_protocol import extract_assistant_text
        assert extract_assistant_text(12345) == "12345"


# ---------------------------------------------------------------------------
# GymDataset
# ---------------------------------------------------------------------------

class TestGymDataset:

    def test_loads_jsonl(self, tmp_path):
        from nemo_evaluator.environments.gym import GymDataset

        path = tmp_path / "data.jsonl"
        rows = [
            {"responses_create_params": {"input": "q1"}, "expected_answer": "a1"},
            {"responses_create_params": {"input": "q2"}, "expected_answer": "a2"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        ds = GymDataset(path)
        assert len(ds) == 2
        assert ds[0]["expected_answer"] == "a1"
        assert ds.name == "data"

    def test_missing_file_yields_empty(self, tmp_path):
        from nemo_evaluator.environments.gym import GymDataset

        ds = GymDataset(tmp_path / "nope.jsonl")
        assert len(ds) == 0

    def test_skips_blank_lines(self, tmp_path):
        from nemo_evaluator.environments.gym import GymDataset

        path = tmp_path / "data.jsonl"
        path.write_text('{"a":1}\n\n{"b":2}\n  \n')

        ds = GymDataset(path)
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# GymEnvironment -- evaluator protocol (default)
# ---------------------------------------------------------------------------

class TestGymEnvironmentEvaluator:

    @pytest.mark.asyncio
    async def test_seed_from_server(self):
        from nemo_evaluator.environments.gym import GymEnvironment

        env = GymEnvironment("http://fake:8000")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "prompt": "What is 2+2?",
            "expected_answer": "4",
            "metadata": {"cat": "math"},
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mc.return_value = client

            seed = await env.seed(0)

        assert seed.prompt == "What is 2+2?"
        assert seed.expected_answer == "4"

    @pytest.mark.asyncio
    async def test_verify_evaluator_protocol(self):
        from nemo_evaluator.environments.gym import GymEnvironment

        env = GymEnvironment("http://fake:8000")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"reward": 1.0, "extracted_answer": "4"}

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mc.return_value = client

            vr = await env.verify("4", "4")

        assert vr.reward == 1.0

    @pytest.mark.asyncio
    async def test_dataset_size_from_server(self):
        from nemo_evaluator.environments.gym import GymEnvironment

        env = GymEnvironment("http://fake:8000")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"size": 100}

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_resp)
            mc.return_value = client

            assert await env.dataset_size() == 100

    @pytest.mark.asyncio
    async def test_dataset_size_error_returns_negative(self):
        from nemo_evaluator.environments.gym import GymEnvironment

        env = GymEnvironment("http://fake:8000")

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=Exception("refused"))
            mc.return_value = client

            assert await env.dataset_size() == -1


# ---------------------------------------------------------------------------
# GymEnvironment -- native protocol (with GymDataset)
# ---------------------------------------------------------------------------

class TestGymEnvironmentNative:

    @pytest.fixture
    def spider_data(self, tmp_path):
        rows = [
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "List employees"}],
                    "model": "test-model",
                },
                "expected_answer": "",
                "db_id": "hr",
                "gold_sql": "SELECT * FROM employees;",
                "uuid": "s2-0",
            },
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Count orders"}],
                    "model": "test-model",
                },
                "expected_answer": "",
                "db_id": "sales",
                "uuid": "s2-1",
            },
        ]
        path = tmp_path / "spider2.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    @pytest.mark.asyncio
    async def test_seed_from_dataset(self, spider_data):
        from nemo_evaluator.environments.gym import GymDataset, GymEnvironment

        env = GymEnvironment(
            "http://fake:8000",
            protocol="native",
            dataset=GymDataset(spider_data),
        )
        assert await env.dataset_size() == 2

        seed = await env.seed(0)
        assert seed.prompt == "List employees"
        assert seed.messages is not None
        assert len(seed.messages) == 1

    @pytest.mark.asyncio
    async def test_verify_native_wraps_response(self, spider_data):
        """Native verify should wrap response in NeMoGymResponse and include rcp from dataset."""
        from nemo_evaluator.environments.gym import GymDataset, GymEnvironment

        ds = GymDataset(spider_data)
        env = GymEnvironment("http://fake:8000", protocol="native", dataset=ds)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "reward": 1.0,
            "extracted_sql": "SELECT * FROM employees;",
        }

        captured: dict[str, Any] = {}

        async def capture_post(url, json=None, **kw):
            captured.update(json or {})
            return mock_resp

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.post = capture_post
            mc.return_value = client

            vr = await env.verify("SELECT * FROM employees;", "", problem_idx=0)

        assert vr.reward == 1.0
        assert vr.extracted_answer == "SELECT * FROM employees;"

        # Body should contain NeMoGymResponse envelope and original rcp
        assert captured["response"]["output_text"] == "SELECT * FROM employees;"
        assert captured["responses_create_params"]["model"] == "test-model"
        # Benchmark-specific fields forwarded from dataset row
        assert captured["db_id"] == "hr"

    @pytest.mark.asyncio
    async def test_verify_native_without_dataset_uses_fallback_rcp(self):
        """If no dataset is attached, native verify builds a synthetic rcp."""
        from nemo_evaluator.environments.gym import GymEnvironment

        env = GymEnvironment("http://fake:8000", protocol="native")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"reward": 0.5}

        captured: dict[str, Any] = {}

        async def capture_post(url, json=None, **kw):
            captured.update(json or {})
            return mock_resp

        with patch.object(env, "_get_client") as mc:
            client = AsyncMock()
            client.post = capture_post
            mc.return_value = client

            await env.verify("answer", "expected", prompt="What?")

        assert captured["responses_create_params"]["input"][0]["content"] == "What?"
        assert captured["response"]["output_text"] == "answer"

    @pytest.mark.asyncio
    async def test_metadata_smuggling_fixed(self, spider_data):
        """The old NativeGymEnvironment smuggled _responses_create_params through
        metadata.  The unified GymEnvironment should NOT put it in metadata."""
        from nemo_evaluator.environments.gym import GymDataset, GymEnvironment

        ds = GymDataset(spider_data)
        env = GymEnvironment("http://fake:8000", protocol="native", dataset=ds)

        seed = await env.seed(0)

        # No underscore-prefixed internal fields should appear in metadata
        for key in seed.metadata:
            assert not key.startswith("_"), f"Internal key {key!r} leaked into metadata"


# ---------------------------------------------------------------------------
# ManagedGymEnvironment
# ---------------------------------------------------------------------------

class TestManagedGymEnvironment:

    def test_build_cmd_nel_benchmark(self):
        from nemo_evaluator.environments.gym import ManagedGymEnvironment
        env = ManagedGymEnvironment(nel_benchmark="gsm8k", port=9999)
        cmd = env._build_cmd()
        assert isinstance(cmd, list)
        assert "--benchmark" in cmd and "gsm8k" in cmd

    def test_build_cmd_server_module(self):
        from nemo_evaluator.environments.gym import ManagedGymEnvironment
        env = ManagedGymEnvironment(server_module="my_app.server", port=9999)
        cmd = env._build_cmd()
        assert "uvicorn" in cmd and "my_app.server:app" in cmd

    def test_build_cmd_server_cmd(self):
        from nemo_evaluator.environments.gym import ManagedGymEnvironment
        env = ManagedGymEnvironment(server_cmd="python run_server.py", port=9999)
        cmd = env._build_cmd()
        assert isinstance(cmd, str) and "--port 9999" in cmd

    def test_build_cmd_requires_at_least_one(self):
        from nemo_evaluator.environments.gym import ManagedGymEnvironment
        with pytest.raises(ValueError, match="requires one of"):
            ManagedGymEnvironment(port=9999)._build_cmd()

    def test_passes_protocol_and_dataset_to_inner(self, tmp_path):
        """ManagedGymEnvironment should propagate protocol + dataset to inner GymEnvironment."""
        from nemo_evaluator.environments.gym import GymDataset, ManagedGymEnvironment

        path = tmp_path / "d.jsonl"
        path.write_text('{"responses_create_params":{"input":"x"}}\n')
        ds = GymDataset(path)

        env = ManagedGymEnvironment(
            nel_benchmark="test", port=9999,
            protocol="native", dataset=ds,
        )
        assert env._protocol == "native"
        assert env._dataset_obj is ds

    def test_auto_port_selection(self):
        from nemo_evaluator.environments.gym import ManagedGymEnvironment
        env = ManagedGymEnvironment(nel_benchmark="test")
        assert 0 < env._port < 65536


# ---------------------------------------------------------------------------
# Registry: gym:// URI resolution
# ---------------------------------------------------------------------------

class TestGymUriResolution:

    def test_host_port_evaluator(self):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import GymEnvironment

        env = _make_gym("localhost:8080")
        assert isinstance(env, GymEnvironment)
        assert env.endpoint == "http://localhost:8080"
        assert env.protocol == "evaluator"

    def test_host_port_native_via_query(self, tmp_path):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import GymEnvironment

        data_path = tmp_path / "d.jsonl"
        data_path.write_text('{"responses_create_params":{"input":"x"}}\n')

        env = _make_gym(f"localhost:8080?protocol=native&data={data_path}")
        assert isinstance(env, GymEnvironment)
        assert env.protocol == "native"
        assert env._dataset is not None
        assert len(env._dataset) == 1

    def test_module_managed(self):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import ManagedGymEnvironment

        env = _make_gym("module:my_app.server")
        assert isinstance(env, ManagedGymEnvironment)

    def test_module_native_via_query(self, tmp_path):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import ManagedGymEnvironment

        data_path = tmp_path / "d.jsonl"
        data_path.write_text('{"responses_create_params":{"input":"x"}}\n')

        env = _make_gym(f"module:my_app.server?protocol=native&data={data_path}")
        assert isinstance(env, ManagedGymEnvironment)
        assert env._protocol == "native"

    def test_cmd_managed(self):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import ManagedGymEnvironment

        env = _make_gym("cmd:python my_server.py")
        assert isinstance(env, ManagedGymEnvironment)

    def test_bare_name_managed(self):
        from nemo_evaluator.environments.registry import _make_gym
        from nemo_evaluator.environments.gym import ManagedGymEnvironment

        env = _make_gym("spider2_lite")
        assert isinstance(env, ManagedGymEnvironment)

    def test_protocol_kwarg(self):
        from nemo_evaluator.environments.registry import _make_gym
        env = _make_gym("localhost:8080", protocol="native")
        assert env.protocol == "native"


# ---------------------------------------------------------------------------
# Gym resource server inventory (verify servers exist in the fork)
# ---------------------------------------------------------------------------

GYM_ROOT = Path(__file__).parent.parent.parent / "Gym"
RESOURCE_SERVERS_DIR = GYM_ROOT / "resources_servers"

EXPECTED_RESOURCE_SERVERS = [
    "spider2_lite",
    "xstest",
    "workplace_assistant",
    "text_to_sql",
    "finance_sec_search",
    "code_gen",
    "math_with_code",
    "math_with_judge",
    "jailbreak_detection",
    "over_refusal_detection",
]


@pytest.mark.skipif(
    not RESOURCE_SERVERS_DIR.exists(),
    reason="Gym fork not available at expected path",
)
class TestGymResourceServerInventory:

    def test_resource_servers_dir_exists(self):
        assert RESOURCE_SERVERS_DIR.is_dir()

    @pytest.mark.parametrize("server_name", EXPECTED_RESOURCE_SERVERS)
    def test_expected_server_exists(self, server_name):
        assert (RESOURCE_SERVERS_DIR / server_name).is_dir()

    @pytest.mark.parametrize("server_name", EXPECTED_RESOURCE_SERVERS)
    def test_server_has_app_py(self, server_name):
        assert (RESOURCE_SERVERS_DIR / server_name / "app.py").is_file()

    def test_spider2_lite_has_eval_script(self):
        assert (RESOURCE_SERVERS_DIR / "spider2_lite" / "scripts" / "run_eval.sh").is_file()

    def test_spider2_lite_has_example_data(self):
        assert (RESOURCE_SERVERS_DIR / "spider2_lite" / "data" / "example.jsonl").is_file()

    def test_total_resource_server_count(self):
        servers = [d for d in RESOURCE_SERVERS_DIR.iterdir()
                   if d.is_dir() and (d / "app.py").exists()]
        assert len(servers) >= 30, f"Only {len(servers)} resource servers found"
