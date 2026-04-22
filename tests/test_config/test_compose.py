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
"""Tests for config composition (defaults, base inheritance, deep merge, self-refs)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from nemo_evaluator.config.compose import (
    _deep_merge,
    _prune_nulls,
    _resolve_self_refs,
    _strip_private_keys,
    compose_config,
)


# ── _deep_merge ───────────────────────────────────────────────────────────


class TestDeepMerge:
    def test_simple_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_disjoint_keys(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dicts_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        overlay = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, overlay) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_lists_replace(self):
        base = {"items": [1, 2, 3]}
        overlay = {"items": [4, 5]}
        assert _deep_merge(base, overlay) == {"items": [4, 5]}

    def test_type_conflict_overlay_wins(self):
        assert _deep_merge({"x": [1]}, {"x": {"a": 1}}) == {"x": {"a": 1}}
        assert _deep_merge({"x": {"a": 1}}, {"x": 42}) == {"x": 42}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
        overlay = {"a": {"b": {"c": {"e": 3, "f": 4}}}}
        expected = {"a": {"b": {"c": {"d": 1, "e": 3, "f": 4}}}}
        assert _deep_merge(base, overlay) == expected

    def test_empty_overlay(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        overlay = {"x": {"b": 2}}
        base_copy = {"x": {"a": 1}}
        _deep_merge(base, overlay)
        assert base == base_copy


# ── _prune_nulls ──────────────────────────────────────────────────────────


class TestPruneNulls:
    def test_removes_top_level_nulls(self):
        d = {"a": 1, "b": None, "c": 3}
        _prune_nulls(d)
        assert d == {"a": 1, "c": 3}

    def test_removes_nested_nulls(self):
        d = {"a": {"b": None, "c": 2}}
        _prune_nulls(d)
        assert d == {"a": {"c": 2}}

    def test_no_nulls_unchanged(self):
        d = {"a": 1, "b": {"c": 2}}
        _prune_nulls(d)
        assert d == {"a": 1, "b": {"c": 2}}

    def test_null_in_list_preserved(self):
        d = {"items": [1, None, 3]}
        _prune_nulls(d)
        assert d == {"items": [1, None, 3]}


# ── _resolve_self_refs ────────────────────────────────────────────────────


class TestResolveSelfRefs:
    def test_simple_ref(self):
        data = {"a": "hello", "b": "${.a}-world"}
        result = _resolve_self_refs(data, data)
        assert result == {"a": "hello", "b": "hello-world"}

    def test_nested_ref(self):
        data = {"x": {"y": "val"}, "z": "${.x.y}"}
        result = _resolve_self_refs(data, data)
        assert result == {"x": {"y": "val"}, "z": "val"}

    def test_unresolved_ref_left_as_is(self):
        data = {"a": "${.nonexistent.path}"}
        result = _resolve_self_refs(data, data)
        assert result == {"a": "${.nonexistent.path}"}

    def test_env_var_syntax_untouched(self):
        data = {"a": "${HF_TOKEN}", "b": "${HOME:-/root}"}
        result = _resolve_self_refs(data, data)
        assert result == {"a": "${HF_TOKEN}", "b": "${HOME:-/root}"}

    def test_numeric_value_to_string(self):
        data = {"port": 5000, "url": "http://localhost:${.port}"}
        result = _resolve_self_refs(data, data)
        assert result == {"port": 5000, "url": "http://localhost:5000"}

    def test_refs_in_list(self):
        data = {"name": "foo", "items": ["${.name}-1", "${.name}-2"]}
        result = _resolve_self_refs(data, data)
        assert result == {"name": "foo", "items": ["foo-1", "foo-2"]}


# ── compose_config (integration with YAML files) ─────────────────────────


@pytest.fixture
def config_dir(tmp_path: Path):
    """Create a temporary config directory with fragments."""
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "clusters").mkdir()
    (conf / "services").mkdir()
    return tmp_path


def _write_yaml(path: Path, data: dict | str) -> Path:
    if isinstance(data, str):
        path.write_text(textwrap.dedent(data))
    else:
        path.write_text(yaml.dump(data, default_flow_style=False))
    return path


class TestComposeConfig:
    def test_flat_config_no_defaults(self, config_dir: Path):
        """Backward-compat: config without defaults: works identically."""
        cfg = _write_yaml(
            config_dir / "flat.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "test"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
            },
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["type"] == "api"
        assert len(raw["benchmarks"]) == 1

    def test_fragment_loading(self, config_dir: Path):
        """Fragments from conf/ are loaded and merged under the correct key."""
        _write_yaml(
            config_dir / "conf" / "clusters" / "local.yaml",
            {
                "type": "local",
            },
        )
        cfg = _write_yaml(
            config_dir / "main.yaml",
            """\
            defaults:
              - clusters/local
            services:
              model:
                type: api
                url: http://localhost:8000/chat/completions
                model: test
            benchmarks:
              - name: gsm8k
                solver: {type: simple, service: model}
        """,
        )
        raw = compose_config(cfg)
        assert raw["cluster"]["type"] == "local"
        assert raw["services"]["model"]["type"] == "api"

    def test_base_inheritance(self, config_dir: Path):
        """_base_: loads a full config and overlays the current one."""
        _write_yaml(
            config_dir / "base.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "base-model"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "/base/output"},
            },
        )
        cfg = _write_yaml(
            config_dir / "child.yaml",
            """\
            defaults:
              - _base_: base
            output:
              dir: /child/output
        """,
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["model"] == "base-model"
        assert raw["output"]["dir"] == "/child/output"
        assert raw["benchmarks"][0]["name"] == "gsm8k"

    def test_base_deep_merge(self, config_dir: Path):
        """Child's dict fields are deep-merged on top of base's."""
        _write_yaml(
            config_dir / "base.yaml",
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://localhost:8000/chat/completions",
                        "model": "base-model",
                        "extra_field": "keep",
                    },
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
            },
        )
        cfg = _write_yaml(
            config_dir / "child.yaml",
            """\
            defaults:
              - _base_: base
            services:
              model:
                model: child-model
        """,
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["model"] == "child-model"
        assert raw["services"]["model"]["extra_field"] == "keep"
        assert raw["services"]["model"]["type"] == "api"

    def test_null_deletes_key(self, config_dir: Path):
        """Setting a key to null removes it from the inherited base."""
        _write_yaml(
            config_dir / "base.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "test"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "/base", "extra": "remove-me"},
            },
        )
        cfg = _write_yaml(
            config_dir / "child.yaml",
            """\
            defaults:
              - _base_: base
            output:
              extra: null
        """,
        )
        raw = compose_config(cfg)
        assert "extra" not in raw["output"]
        assert raw["output"]["dir"] == "/base"

    def test_self_position(self, config_dir: Path):
        """_self_ controls merge priority."""
        _write_yaml(
            config_dir / "conf" / "clusters" / "remote.yaml",
            {
                "type": "local",
                "extra": "from-fragment",
            },
        )
        cfg = _write_yaml(
            config_dir / "main.yaml",
            """\
            defaults:
              - _self_
              - clusters/remote
            services:
              model:
                type: api
                url: http://localhost:8000/chat/completions
                model: test
            benchmarks:
              - name: gsm8k
                solver: {type: simple, service: model}
            cluster:
              type: local
              extra: from-main
        """,
        )
        raw = compose_config(cfg)
        # Fragment comes after _self_, so fragment wins
        assert raw["cluster"]["extra"] == "from-fragment"

    def test_circular_reference_detected(self, config_dir: Path):
        _write_yaml(
            config_dir / "a.yaml",
            """\
            defaults:
              - _base_: b
            services:
              model: {type: api, url: http://x/chat/completions, model: test}
            benchmarks:
              - {name: x, solver: {type: simple, service: model}}
        """,
        )
        _write_yaml(
            config_dir / "b.yaml",
            """\
            defaults:
              - _base_: a
            services:
              model: {type: api, url: http://x/chat/completions, model: test}
            benchmarks:
              - {name: x, solver: {type: simple, service: model}}
        """,
        )
        with pytest.raises(ValueError, match="Circular config reference"):
            compose_config(config_dir / "a.yaml")

    def test_missing_fragment_raises(self, config_dir: Path):
        cfg = _write_yaml(
            config_dir / "bad.yaml",
            """\
            defaults:
              - clusters/nonexistent
            services:
              model: {type: api, url: http://x/chat/completions, model: test}
            benchmarks:
              - {name: x, solver: {type: simple, service: model}}
        """,
        )
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            compose_config(cfg)

    def test_multiple_fragments_merge(self, config_dir: Path):
        """Multiple fragment defaults are all merged."""
        _write_yaml(
            config_dir / "conf" / "clusters" / "local.yaml",
            {
                "type": "local",
            },
        )
        _write_yaml(
            config_dir / "conf" / "services" / "model.yaml",
            {
                "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "test"},
            },
        )
        cfg = _write_yaml(
            config_dir / "multi.yaml",
            """\
            defaults:
              - clusters/local
              - services/model
            benchmarks:
              - name: gsm8k
                solver: {type: simple, service: model}
        """,
        )
        raw = compose_config(cfg)
        assert raw["cluster"]["type"] == "local"
        assert raw["services"]["model"]["type"] == "api"

    def test_self_refs_resolved(self, config_dir: Path):
        """${.path} references are resolved in the final output."""
        cfg = _write_yaml(
            config_dir / "selfref.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "my-model"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "./results/${.services.model.model}"},
            },
        )
        raw = compose_config(cfg)
        assert raw["output"]["dir"] == "./results/my-model"

    def test_env_vars_not_expanded(self, config_dir: Path):
        """compose_config does NOT expand ${ENV_VAR} — that's parse_eval_config's job."""
        cfg = _write_yaml(
            config_dir / "envvar.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "test"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "/path/${HF_HOME}"},
            },
        )
        raw = compose_config(cfg)
        assert raw["output"]["dir"] == "/path/${HF_HOME}"

    def test_chained_base_inheritance(self, config_dir: Path):
        """base -> child -> grandchild all compose correctly."""
        _write_yaml(
            config_dir / "grandparent.yaml",
            {
                "services": {
                    "model": {"type": "api", "url": "http://localhost:8000/chat/completions", "model": "gp-model"}
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "/gp"},
            },
        )
        _write_yaml(
            config_dir / "parent.yaml",
            """\
            defaults:
              - _base_: grandparent
            output:
              dir: /parent
        """,
        )
        cfg = _write_yaml(
            config_dir / "child.yaml",
            """\
            defaults:
              - _base_: parent
            output:
              dir: /child
        """,
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["model"] == "gp-model"
        assert raw["output"]["dir"] == "/child"


# ── _strip_private_keys ──────────────────────────────────────────────────


class TestStripPrivateKeys:
    def test_removes_underscore_prefixed(self):
        d = {"_model": "foo", "services": {"a": 1}}
        _strip_private_keys(d)
        assert d == {"services": {"a": 1}}

    def test_keeps_non_underscore_keys(self):
        d = {"services": {"a": 1}, "output": {"dir": "/tmp"}}
        _strip_private_keys(d)
        assert d == {"services": {"a": 1}, "output": {"dir": "/tmp"}}

    def test_nested_underscore_keys_preserved(self):
        d = {"services": {"_internal": "keep"}}
        _strip_private_keys(d)
        assert d == {"services": {"_internal": "keep"}}

    def test_multiple_private_keys(self):
        d = {"_a": 1, "_b": 2, "real": 3}
        _strip_private_keys(d)
        assert d == {"real": 3}


# ── compose_config user vars (integration) ──────────────────────────────


class TestUserVars:
    def test_underscore_vars_resolved_then_stripped(self, config_dir: Path):
        cfg = _write_yaml(
            config_dir / "uservars.yaml",
            {
                "_model_name": "nemotron-super",
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://localhost:8000/chat/completions",
                        "model": "${._model_name}",
                    }
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
                "output": {"dir": "./results/${._model_name}"},
            },
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["model"] == "nemotron-super"
        assert raw["output"]["dir"] == "./results/nemotron-super"
        assert "_model_name" not in raw

    def test_nested_underscore_var_resolved(self, config_dir: Path):
        cfg = _write_yaml(
            config_dir / "nested_uv.yaml",
            {
                "_common": {"model": "nemotron", "base_url": "http://vllm:8000"},
                "services": {
                    "model": {
                        "type": "api",
                        "url": "${._common.base_url}/v1/chat/completions",
                        "model": "${._common.model}",
                    }
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
            },
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["url"] == "http://vllm:8000/v1/chat/completions"
        assert raw["services"]["model"]["model"] == "nemotron"
        assert "_common" not in raw

    def test_underscore_vars_in_base_inherited(self, config_dir: Path):
        _write_yaml(
            config_dir / "base_uv.yaml",
            {
                "_model_id": "base-model",
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://localhost:8000/chat/completions",
                        "model": "${._model_id}",
                    }
                },
                "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
            },
        )
        cfg = _write_yaml(
            config_dir / "child_uv.yaml",
            """\
            defaults:
              - _base_: base_uv
            _model_id: child-model
        """,
        )
        raw = compose_config(cfg)
        assert raw["services"]["model"]["model"] == "child-model"
        assert "_model_id" not in raw
