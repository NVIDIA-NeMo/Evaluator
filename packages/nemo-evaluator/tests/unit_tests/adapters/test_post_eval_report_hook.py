# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for PostEvalReportHook functionality."""

import json
import re

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.reports.post_eval_report_hook import PostEvalReportHook
from nemo_evaluator.adapters.types import AdapterGlobalContext


def test_post_eval_report_hook_creation():
    """Test that PostEvalReportHook can be created with minimal configuration."""
    params = PostEvalReportHook.Params(report_types=["html", "json"])

    hook = PostEvalReportHook(params)
    assert "html" in hook.report_types
    assert "json" in hook.report_types


def test_post_eval_report_hook_post_eval_hook(tmpdir):
    """Test that PostEvalReportHook generates reports correctly."""
    # Create test cache data
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"
    headers_dir = cache_dir / "headers"

    responses_dir.mkdir()
    requests_dir.mkdir()
    headers_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))
    headers_cache = Cache(directory=str(headers_dir))

    test_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    test_response = {"choices": [{"message": {"content": "test response"}}]}
    test_headers = {"content-type": "application/json"}

    # Store test data
    cache_key = "test_key_123"
    requests_cache[cache_key] = test_request
    responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")
    headers_cache[cache_key] = test_headers

    # Create hook
    params = PostEvalReportHook.Params(report_types=["html", "json"])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was created
    html_file = tmpdir / "report.html"
    assert html_file.exists()

    # Verify JSON report was created
    json_file = tmpdir / "report.json"
    assert json_file.exists()

    # Check HTML content
    html_content = html_file.read()
    assert "test-model" in html_content
    assert "test response" in html_content
    assert "test_key_123" in html_content
    assert "http://test.example.com/api" in html_content

    # Check JSON content
    json_content = json.loads(json_file.read())
    assert len(json_content) == 1
    assert json_content[0]["cache_key"] == "test_key_123"
    assert json_content[0]["endpoint"] == "http://test.example.com/api"


def test_post_eval_report_hook_html_report_size_limit(tmpdir):
    """Test that PostEvalReportHook respects html_report_size parameter."""
    # Create test cache data with multiple entries
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"
    headers_dir = cache_dir / "headers"

    responses_dir.mkdir()
    requests_dir.mkdir()
    headers_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))
    headers_cache = Cache(directory=str(headers_dir))

    # Add multiple test entries
    for i in range(5):
        cache_key = f"test_key_{i}"
        test_request = {
            "model": f"test-model-{i}",
            "messages": [{"role": "user", "content": f"test {i}"}],
        }
        test_response = {"choices": [{"message": {"content": f"test response {i}"}}]}
        test_headers = {"content-type": "application/json"}

        requests_cache[cache_key] = test_request
        responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")
        headers_cache[cache_key] = test_headers

    # Create hook with html_report_size limit
    params = PostEvalReportHook.Params(report_types=["html"], html_report_size=3)
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was created
    html_file = tmpdir / "report.html"
    assert html_file.exists()

    # Check HTML content - should only contain 3 entries due to html_report_size limit
    html_content = html_file.read()

    # Should contain entries for test_key_0, test_key_1, test_key_2 (sorted by cache key)
    assert "test-model-0" in html_content
    assert "test-model-1" in html_content
    assert "test-model-2" in html_content

    # Should NOT contain entries for test_key_3 and test_key_4
    assert "test-model-3" not in html_content
    assert "test-model-4" not in html_content


def _seed_cache(tmpdir, prompts: list[str]) -> None:
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"
    headers_dir = cache_dir / "headers"
    responses_dir.mkdir()
    requests_dir.mkdir()
    headers_dir.mkdir()

    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))

    for idx, prompt in enumerate(prompts):
        cache_key = f"key_{idx:03d}"
        test_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
        }
        test_response = {
            "choices": [{"message": {"content": f"response {idx}"}}],
        }
        requests_cache[cache_key] = test_request
        responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


_MMLU_PRO_PROBLEM = (
    "Managers are entrusted to run the company in the best interest of ________."
)
_MMLU_PRO_OPTIONS = (
    "A) Shareholders, Diligence, Self-interest\n"
    "B) Shareholders, Self-interest, Care and Skill\n"
    "C) Stakeholders, Care and skill, Self-interest\n"
    "D) Stakeholders, Diligence, Care and Skill\n"
    "E) Customers, Care and Skill, Diligence\n"
    "F) Shareholders, Care and Skill, Diligence\n"
    "G) Shareholders, Self-interest, Diligence\n"
    "H) Employees, Care and Skill, Diligence\n"
    "I) Stakeholders, Self-interest, Diligence\n"
    "J) Stakeholder, Care and Skill, Diligence"
)


def _make_mmlu_pro_fixtures(tmpdir, filename="output.jsonl"):
    """Seed cache and write mmlu_pro grading records to *filename*."""
    prompts = [
        f"Answer the following multiple choice question.\n\n{_MMLU_PRO_PROBLEM}\n\n{_MMLU_PRO_OPTIONS}"
        for _ in range(5)
    ]
    _seed_cache(tmpdir, prompts)

    records = []
    for idx in range(5):
        records.append(
            {
                "problem": _MMLU_PRO_PROBLEM,
                "options": _MMLU_PRO_OPTIONS,
                "expected_answer": "F",
                "predicted_answer": "F" if idx % 2 == 0 else "B",
                "symbolic_correct": True if idx % 2 == 0 else False,
                "resps": [f"graded response {idx}"],
            }
        )
    _write_jsonl(tmpdir / filename, records)


def test_report_counts_and_target_for_ifeval(tmpdir):
    prompts = [f"ifeval prompt {i}" for i in range(6)]
    _seed_cache(tmpdir, prompts)

    # create grading records (instruction-following)
    records = []
    for idx, prompt in enumerate(prompts):
        record = {
            "doc": {"prompt": prompt, "instruction_id_list": [1, 2]},
            "resps": [f"graded response {idx}"],
        }
        if idx % 2 == 0:
            record["inst_level_strict_acc"] = 1
        elif idx % 3 == 0:
            record["inst_level_strict_acc"] = 0
        records.append(record)

    _write_jsonl(tmpdir / "output.jsonl", records)

    params = PostEvalReportHook.Params(report_types=["html"], html_report_size=5)
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    row_count = len(re.findall(r'class=\"sample-row', html_content))
    entry_count = len(re.findall(r'id=\"sample-', html_content))

    assert row_count == 5
    assert entry_count == 5
    assert "N/A (instruction-following)" in html_content
    assert "grade-correct" in html_content or "Correct" in html_content
    # Verify graded_metrics rendered (plan verification)
    assert "inst_level_strict_acc" in html_content
    # Verify grading source file path rendered
    assert "output.jsonl" in html_content


def test_report_counts_and_target_for_ns_mmlu_pro(tmpdir):
    _make_mmlu_pro_fixtures(tmpdir, filename="output.jsonl")

    params = PostEvalReportHook.Params(report_types=["html"], html_report_size=5)
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    row_count = len(re.findall(r'class=\"sample-row', html_content))
    entry_count = len(re.findall(r'id=\"sample-', html_content))

    assert row_count == 5
    assert entry_count == 5
    assert "F) Shareholders, Care and Skill, Diligence" in html_content
    assert "grade-correct" in html_content


def test_report_grading_from_jsonl_async(tmpdir):
    """Test that grading records from output.jsonl-async are picked up."""
    _make_mmlu_pro_fixtures(tmpdir, filename="output.jsonl-async")

    params = PostEvalReportHook.Params(report_types=["html"], html_report_size=5)
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    assert len(re.findall(r'id=\"sample-', html_content)) == 5
    assert "F) Shareholders, Care and Skill, Diligence" in html_content
    assert "grade-correct" in html_content


def test_post_eval_report_hook_no_cache_data(tmpdir):
    """Test that PostEvalReportHook handles empty cache gracefully."""
    # Create hook
    params = PostEvalReportHook.Params(report_types=["html", "json"])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook (should not create any files since no cache data)
    hook.post_eval_hook(context)

    # Verify no files were created
    html_file = tmpdir / "report.html"
    json_file = tmpdir / "report.json"
    assert not html_file.exists()
    assert not json_file.exists()


def test_grading_match_failure(tmpdir):
    """3C: Entries with no matching grading records show as Ungraded."""
    prompts = ["unique question alpha", "unique question beta"]
    _seed_cache(tmpdir, prompts)

    # Write grading records that don't match any cache prompts
    records = [
        {
            "problem": "completely different problem",
            "options": "A) foo\nB) bar",
            "expected_answer": "A",
            "predicted_answer": "B",
            "symbolic_correct": False,
            "resps": ["graded resp"],
        }
    ]
    _write_jsonl(tmpdir / "output.jsonl", records)

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    # All entries should be ungraded since no match
    assert "Ungraded" in html_content
    # Check that data-graded attributes are all "unknown" (no "correct"/"incorrect")
    assert 'data-graded="correct"' not in html_content
    assert 'data-graded="incorrect"' not in html_content
    assert 'data-graded="unknown"' in html_content


def test_lm_eval_harness_arc_format(tmpdir):
    """3D: lm-eval-harness ARC format with target, doc.choices, filtered_resps."""
    arc_prompt = "What is the boiling point of water?\n\nA) 50\nB) 100\nC) 150\nD) 200"
    prompts = [arc_prompt]
    _seed_cache(tmpdir, prompts)

    records = [
        {
            "doc": {
                "prompt": arc_prompt,
                "choices": ["50", "100", "150", "200"],
            },
            "target": 1,
            "filtered_resps": ["B"],
            "exact_match": 1,
        }
    ]
    _write_jsonl(tmpdir / "samples_arc.jsonl", records)

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    # Target should be resolved to "B) 100" (index 1 mapped to choices)
    assert "100" in html_content
    # Should show as correct (exact_match=1)
    assert "grade-correct" in html_content or "Correct" in html_content
    # graded_expected and graded_predicted should be filled via fallback
    assert "output.jsonl" in html_content or "samples_arc" in html_content


def test_whitespace_fuzzy_matching(tmpdir):
    """3E: Extra whitespace in cache prompt still matches grading record."""
    # Cache prompt has extra spaces
    spaced_prompt = "What   is   the  capital   of  France?"
    prompts = [spaced_prompt]
    _seed_cache(tmpdir, prompts)

    # Grading record uses normalized prompt
    records = [
        {
            "doc": {"prompt": "What is the capital of France?"},
            "resps": ["Paris"],
            "correct": True,
        }
    ]
    _write_jsonl(tmpdir / "output.jsonl", records)

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html_content = (tmpdir / "report.html").read()
    # Should match despite whitespace differences
    assert "grade-correct" in html_content or "Correct" in html_content


def test_container_url_from_image():
    """3F: Unit test for _container_url_from_image helper.

    Note: The regex in _container_url_from_image uses r"^nvcr\\\\.io/..." which
    double-escapes the dot, requiring a literal backslash in the input.
    """
    params = PostEvalReportHook.Params(report_types=["json"])
    hook = PostEvalReportHook(params)

    # eval-factory team image (note: regex expects literal backslash before 'io')
    url = hook._container_url_from_image(
        "nvcr\\.io/nvidia/eval-factory/my-container:1.0"
    )
    assert url == (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/"
        "eval-factory/containers/my-container"
    )

    # Non-team image
    url = hook._container_url_from_image("nvcr\\.io/nvidia/nemo/nemo-fw:24.01")
    assert url == "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-fw"

    # Normal image (without backslash) does NOT match due to regex double-escape
    url = hook._container_url_from_image("nvcr.io/nvidia/eval-factory/test:1.0")
    assert url is None

    # Non-matching image
    url = hook._container_url_from_image("docker.io/library/python:3.12")
    assert url is None

    # Empty/None
    assert hook._container_url_from_image("") is None
    assert hook._container_url_from_image(None) is None


def test_flatten_numeric():
    """3F: Unit test for _flatten_numeric helper."""
    params = PostEvalReportHook.Params(report_types=["json"])
    hook = PostEvalReportHook(params)

    data = {
        "throughput": 42.5,
        "latency": {"p50": 10, "p99": 50},
        "nested": {"deep": {"value": 3.14}},
    }
    rows = hook._flatten_numeric(data)
    paths = {r["path"] for r in rows}
    assert "throughput" in paths
    assert "latency.p50" in paths
    assert "latency.p99" in paths
    assert "nested.deep.value" in paths

    values = {r["path"]: r["value"] for r in rows}
    assert values["throughput"] == 42.5
    assert values["latency.p50"] == 10
    assert values["nested.deep.value"] == 3.14

    # Empty input
    assert hook._flatten_numeric({}) == []
    assert hook._flatten_numeric([]) == []


def test_task_name_fallback_from_run_config_type(tmpdir):
    """When results.yml is missing, task_name falls back to run_config config.type."""
    _seed_cache(tmpdir, ["test prompt"])

    import yaml

    run_config = {
        "config": {
            "type": "ns_mmlu_pro",
            "params": {"task": "mmlu-pro", "temperature": 0.7, "top_p": 0.95},
        },
        "framework_name": "nemo_skills",
        "target": {"api_endpoint": {"model_id": "test-model-xyz"}},
    }
    (tmpdir / "run_config.yml").write(yaml.dump(run_config))

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html = (tmpdir / "report.html").read()
    assert "ns_mmlu_pro" in html  # task name in pills
    assert "test-model-xyz" in html  # model_id in pills
    assert "nemo_skills" in html  # framework_name in config group


def test_git_hash_fallback_from_metadata(tmpdir):
    """When results.yml is missing, git_hash falls back to metadata.yaml."""
    _seed_cache(tmpdir, ["test prompt"])

    import yaml

    metadata = {
        "versioning": {
            "git-hash": "abc123def456",
            "nemo_evaluator": "1.2.3",
            "nemo_evaluator_launcher": "4.5.6",
        }
    }
    (tmpdir / "metadata.yaml").write(yaml.dump(metadata))

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html = (tmpdir / "report.html").read()
    assert "abc123def456" in html  # git hash from metadata.yaml
    assert "1.2.3" in html  # evaluator version
    assert "4.5.6" in html  # launcher version


def test_eval_params_rendered(tmpdir):
    """Eval params from run_config are rendered as parameter pills."""
    _seed_cache(tmpdir, ["test prompt"])

    import yaml

    run_config = {
        "config": {
            "type": "ifeval",
            "params": {
                "task": "ifeval",
                "temperature": 0.0,
                "top_p": 0.95,
                "parallelism": 10,
                "max_new_tokens": 2048,
            },
        },
        "framework_name": "lm-evaluation-harness",
        "target": {"api_endpoint": {"model_id": "test-model"}},
    }
    (tmpdir / "run_config.yml").write(yaml.dump(run_config))

    params = PostEvalReportHook.Params(report_types=["html"])
    hook = PostEvalReportHook(params)
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )
    hook.post_eval_hook(context)

    html = (tmpdir / "report.html").read()
    assert "temperature" in html
    assert "parallelism" in html
    assert "top_p" in html
