# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the composed-config reproducibility snapshot."""

import yaml

from nemo_evaluator.config.snapshot import (
    SNAPSHOT_FILENAME,
    _mask_cli_text,
    build_provenance,
    build_snapshot_text,
    mask_secrets,
    record_output_dir_override,
    write_config_snapshot,
)


class _StubConfig:
    """Minimal stand-in for EvalConfig: private attrs + output.dir."""

    def __init__(self, raw=None, provenance=None, output_dir="unused"):
        self._composed_raw = raw
        self._snapshot_provenance = provenance or {}

        class _Out:
            dir = output_dir

        self.output = _Out()

    def model_dump(self, **_kw):
        return {"services": {"m": {"type": "api", "api_key": "nvapi-" + "x" * 20}}}


class TestMaskSecrets:
    def test_env_refs_preserved(self):
        raw = {"services": {"m": {"api_key": "${NVIDIA_API_KEY}"}}}
        assert mask_secrets(raw) == raw

    def test_literal_secret_key_masked(self):
        raw = {"services": {"m": {"api_key": "nvapi-not-a-real-key"}}}
        assert mask_secrets(raw)["services"]["m"]["api_key"] == "<redacted>"

    def test_env_var_names_masked_when_literal(self):
        raw = {"env_vars": {"HF_TOKEN": "hf_literalvalue", "JUDGE_API_KEY": "${JUDGE_API_KEY}"}}
        out = mask_secrets(raw)
        assert out["env_vars"]["HF_TOKEN"] == "<redacted>"  # key matches token
        assert out["env_vars"]["JUDGE_API_KEY"] == "${JUDGE_API_KEY}"

    def test_pattern_backstop_inside_free_form_string(self):
        raw = {"extra": {"args": "++server.key=nvapi-" + "a" * 24 + " ++x=1"}}
        out = mask_secrets(raw)
        assert "nvapi-" not in out["extra"]["args"]
        assert "++x=1" in out["extra"]["args"]

    def test_non_secret_values_untouched(self):
        raw = {"model": "nvidia/nemotron", "temperature": 1.0, "flags": [1, "a"]}
        assert mask_secrets(raw) == raw

    def test_tokenizer_fields_untouched(self):
        """'token' must not substring-match tokenizer/tokenizer_backend."""
        raw = {"extra": {"tokenizer": "zai-org/GLM-4.7", "tokenizer_backend": "huggingface"}}
        assert mask_secrets(raw) == raw

    def test_template_markers_and_empty_untouched(self):
        """Reasoning-token markers and empty strings under secret-named keys stay."""
        raw = {
            "adapter_config": {"start_reasoning_token": "<|channel>thought\n", "end_reasoning_token": "<channel|>"},
            "model": {"api_key": ""},
            "env": {"TIKTOKEN_ENCODINGS_DIR": "/cache/vllm/tiktoken_encodings"},
            "sandbox": {"tunnel_secret_arn": "arn:aws:secretsmanager:us-west-2:123:secret:x"},
        }
        assert mask_secrets(raw) == raw

    def test_reference_shaped_values_untouched(self):
        """Env-var names, bare $VAR and host: mappings are references, not secrets."""
        raw = {
            "judge": {"api_key": "JUDGE_API_KEY"},
            "env_vars": {"HF_TOKEN": "$HF_TOKEN", "NGC_API_KEY": "host:NGC_API_KEY"},
        }
        assert mask_secrets(raw) == raw

    def test_plural_secret_keys_masked(self):
        raw = {"credentials": "some-literal-cred", "secrets": "another-literal"}
        assert mask_secrets(raw) == {"credentials": "<redacted>", "secrets": "<redacted>"}

    def test_all_caps_blob_masked(self):
        """Uppercase alnum without underscores is not an env-var name."""
        raw = {"m": {"api_key": "ABCDEF123456XYZ"}}
        assert mask_secrets(raw)["m"]["api_key"] == "<redacted>"


class TestSnapshotText:
    def test_header_and_reparseable_body(self):
        raw = {"services": {"m": {"type": "api", "url": "${U}/v1", "model": "x"}}}
        text = build_snapshot_text(raw, build_provenance(source_config="/tmp/c.yaml"))
        assert text.startswith("#")
        assert "nemo-evaluator version" in text
        assert "source config: /tmp/c.yaml" in text
        # Comments are ignored by YAML: the body must round-trip.
        assert yaml.safe_load(text) == raw


class TestWriteConfigSnapshot:
    def test_writes_composed_raw(self, tmp_path):
        raw = {"services": {"m": {"type": "api", "api_key": "${KEY}"}}}
        cfg = _StubConfig(raw=raw, provenance={"run_id": "r1"})
        path = write_config_snapshot(cfg, tmp_path)
        assert path == tmp_path / SNAPSHOT_FILENAME
        text = path.read_text()
        assert "${KEY}" in text
        assert "run_id: r1" in text

    def test_skipped_in_inner_execution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEL_INNER_EXECUTION", "1")
        cfg = _StubConfig(raw={"a": 1})
        assert write_config_snapshot(cfg, tmp_path) is None
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()

    def test_fallback_reconstruction_masks_secrets(self, tmp_path):
        """Quick-mode configs (no composed raw) fall back to model_dump, masked."""
        cfg = _StubConfig(raw=None)
        path = write_config_snapshot(cfg, tmp_path)
        text = path.read_text()
        assert "nvapi-" not in text
        assert "<redacted>" in text

    def test_never_raises(self):
        """A broken config must not raise (snapshot is best-effort)."""

        class _Broken:
            output = None  # config.output.dir access will fail

        assert write_config_snapshot(_Broken(), None) is None


class TestMaskCliText:
    def test_override_kv_secret_masked(self):
        out = _mask_cli_text("nel eval run c.yaml -O services.m.api_key=hunter2 -O benchmarks.0.repeats=2")
        assert "hunter2" not in out
        assert "api_key=<redacted>" in out
        assert "repeats=2" in out  # non-secret override untouched

    def test_override_env_ref_kept(self):
        out = _mask_cli_text("-O services.m.api_key=${NVIDIA_API_KEY}")
        assert "${NVIDIA_API_KEY}" in out

    def test_tokens_param_not_masked(self):
        """'tokens' plural is deliberately excluded: *_tokens params are benign."""
        out = _mask_cli_text("-O benchmarks.0.params.max_new_tokens=2048")
        assert "max_new_tokens=2048" in out


class TestForceOverwrite:
    def test_fresh_run_overwrites_stale_snapshot(self, tmp_path):
        write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path, force=True)
        path = write_config_snapshot(_StubConfig(raw={"a": 2}), tmp_path, force=True)
        assert "a: 2" in path.read_text()

    def test_resume_preserves_original(self, tmp_path):
        first = write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path, force=True)
        original = first.read_text()
        write_config_snapshot(_StubConfig(raw={"a": 2}), tmp_path, force=False)
        assert first.read_text() == original


class TestProvenanceLeakRegression:
    def test_secret_cli_forms_never_reach_snapshot_text(self, monkeypatch):
        """argv/override secrets must not survive into the rendered snapshot."""
        import sys

        monkeypatch.setattr(
            sys,
            "argv",
            ["nel", "eval", "run", "c.yaml", "--api-key=nvapi-ARGVSECRET111111", "--token", "plainSecret123"],
        )
        prov = build_provenance(overrides=("services.x.api_key=overrideSecret99",))
        text = build_snapshot_text({"services": {}}, prov)
        assert "nvapi-" not in text
        assert "plainSecret123" not in text
        assert "overrideSecret99" not in text
        assert "<redacted>" in text


class TestOutputDirOverride:
    def test_cli_override_reflected_in_snapshot(self, tmp_path):
        """-o overrides config.output.dir after compose; the snapshot must match."""
        cfg = _StubConfig(raw={"output": {"dir": "./from_yaml"}})
        record_output_dir_override(cfg, "/actual/run/dir")
        path = write_config_snapshot(cfg, tmp_path)
        text = path.read_text()
        assert "/actual/run/dir" in text
        assert "from_yaml" not in text
