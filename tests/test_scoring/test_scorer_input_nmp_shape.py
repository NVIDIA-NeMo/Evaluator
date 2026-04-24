# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NMP-shape access on :class:`ScorerInput`.

This is Phase 1 of the ScorerInput → NMP shape migration. The named NEL
fields (``response``, ``target``, ``metadata``, ``config``, ``sandbox``)
remain the authoritative source. We add derived properties :attr:`item`
and :attr:`sample`, plus a :meth:`jinja_context` helper, so NMP-style
metric classes and Jinja templates can read the input in their native
vocabulary without any change to NEL's existing scorer functions.

See also: the integration RFC (`Metrics resolution and Milestone 1`).
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from nemo_evaluator.scoring.types import ScorerInput


# ---------------------------------------------------------------------------
# .item property — NMP-native dataset-row view
# ---------------------------------------------------------------------------


def test_item_contains_reference_from_target():
    s = ScorerInput(response="foo", target="bar")
    assert s.item == {"reference": "bar"}


def test_item_merges_metadata():
    s = ScorerInput(
        response="foo",
        target="bar",
        metadata={"problem_id": 42, "category": "safety"},
    )
    assert s.item == {"reference": "bar", "problem_id": 42, "category": "safety"}


def test_item_does_not_leak_config_or_response():
    s = ScorerInput(response="r", target="t", metadata={"m": 1}, config={"c": 2})
    assert "c" not in s.item
    assert "response" not in s.item
    assert "output_text" not in s.item


def test_item_reference_key_can_be_overridden_via_metadata():
    """If metadata already has a 'reference' key, it overrides target — callers
    with their own target-field conventions can populate metadata directly."""
    s = ScorerInput(
        response="foo",
        target="ignored",
        metadata={"reference": "from-metadata"},
    )
    assert s.item["reference"] == "from-metadata"


# ---------------------------------------------------------------------------
# .sample property — NMP-native inference-payload view
# ---------------------------------------------------------------------------


def test_sample_contains_output_text_from_response():
    s = ScorerInput(response="hello world", target="irrelevant")
    assert s.sample["output_text"] == "hello world"
    assert s.sample["response"] == "hello world"


def test_sample_does_not_leak_target_or_metadata():
    s = ScorerInput(response="r", target="t", metadata={"m": 1})
    assert "target" not in s.sample
    assert "m" not in s.sample
    assert "reference" not in s.sample


# ---------------------------------------------------------------------------
# .jinja_context() — template rendering context
# ---------------------------------------------------------------------------


def test_jinja_context_exposes_flat_nel_keys():
    s = ScorerInput(response="r", target="t", metadata={"m": 1}, config={"c": 2})
    ctx = s.jinja_context()
    assert ctx["response"] == "r"
    assert ctx["target"] == "t"
    assert ctx["metadata"] == {"m": 1}
    assert ctx["config"] == {"c": 2}


def test_jinja_context_exposes_flat_nmp_keys():
    s = ScorerInput(response="model output", target="ground truth")
    ctx = s.jinja_context()
    assert ctx["reference"] == "ground truth"
    assert ctx["output_text"] == "model output"


def test_jinja_context_exposes_item_and_sample_namespaces():
    s = ScorerInput(
        response="model output",
        target="ground truth",
        metadata={"problem_id": 7},
    )
    ctx = s.jinja_context()
    assert ctx["item"] == {"reference": "ground truth", "problem_id": 7}
    assert ctx["sample"] == {"output_text": "model output", "response": "model output"}


def test_jinja_render_nel_native_template():
    """NEL-native templates render unchanged."""
    s = ScorerInput(response="hi", target="hello", metadata={"k": "v"})
    env = Environment(undefined=StrictUndefined)
    out = env.from_string("{{ response }} / {{ target }} / {{ metadata.k }}").render(
        **s.jinja_context()
    )
    assert out == "hi / hello / v"


def test_jinja_render_nmp_native_template():
    """NMP-style templates using item/sample namespaces render unchanged."""
    s = ScorerInput(
        response="model said this",
        target="gold answer",
        metadata={"problem_id": 42, "context": "paragraph"},
    )
    env = Environment(undefined=StrictUndefined)
    template = "ref={{ item.reference }} | out={{ sample.output_text }} | id={{ item.problem_id }}"
    out = env.from_string(template).render(**s.jinja_context())
    assert out == "ref=gold answer | out=model said this | id=42"


def test_jinja_render_mixed_vocabulary_template():
    """A template mixing NEL and NMP names renders consistently."""
    s = ScorerInput(response="A", target="B")
    env = Environment(undefined=StrictUndefined)
    out = env.from_string("{{ response }}-{{ reference }}-{{ output_text }}-{{ target }}").render(
        **s.jinja_context()
    )
    # response == output_text == "A"; target == reference == "B"
    assert out == "A-B-A-B"


# ---------------------------------------------------------------------------
# Backward-compat: existing named-field access still works
# ---------------------------------------------------------------------------


def test_existing_field_access_unchanged():
    """E2e guarantee: code reading .response / .target / .metadata / .config /
    .sandbox works exactly as before. Phase 1 only adds properties."""
    s = ScorerInput(
        response="r",
        target="t",
        metadata={"m": 1},
        config={"c": 2},
    )
    assert s.response == "r"
    assert s.target == "t"
    assert s.metadata == {"m": 1}
    assert s.config == {"c": 2}
    assert s.sandbox is None


def test_dataclass_construction_accepts_only_required_fields():
    """Minimal construction still works — properties are opt-in."""
    s = ScorerInput(response="r", target="t")
    assert s.item == {"reference": "t"}
    assert s.sample == {"output_text": "r", "response": "r"}
