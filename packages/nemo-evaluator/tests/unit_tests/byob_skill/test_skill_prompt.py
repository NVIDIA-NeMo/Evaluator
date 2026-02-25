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

"""Static validation tests for the BYOB skill prompt file.

Tests T001-T006: Verify the skill prompt exists, is within budget, and
contains all required sections.
"""

import os
import re


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3 (GPT-family average)."""
    words = len(text.split())
    return int(words * 1.3)


def test_skill_prompt_exists_and_within_budget(skill_prompt_path):
    """T001: Skill prompt file exists and is within token budget.

    Validates:
    - File exists at .claude/commands/byob.md
    - Content is non-empty (>100 chars)
    - File size < 20KB (safety upper bound)
    - Estimated token count < 4,000
    """
    assert os.path.isfile(skill_prompt_path), (
        f"Skill prompt not found at: {skill_prompt_path}. "
        f"Expected .claude/commands/byob.md relative to package root."
    )

    with open(skill_prompt_path) as f:
        content = f.read()

    assert len(content) > 100, (
        f"Skill prompt is suspiciously short ({len(content)} chars). "
        f"Expected at least 100 characters."
    )

    byte_count = len(content.encode("utf-8"))
    assert byte_count < 20_000, (
        f"Skill prompt exceeds 20KB safety bound ({byte_count} bytes). "
        f"This suggests the file may be corrupted or includes large embedded data."
    )

    token_est = estimate_tokens(content)
    assert token_est < 4_000, (
        f"Estimated {token_est} tokens exceeds 4,000 budget. "
        f"The skill prompt must fit within Claude Code's skill prompt limit. "
        f"Consider compressing sections or moving content to templates."
    )


def test_skill_prompt_contains_required_sections(skill_prompt_path):
    """T002: Skill prompt contains all required sections.

    Required sections (identified by keywords, case-insensitive):
    1. Workflow / 5 steps
    2. API reference
    3. Scorer selection logic
    4. Dataset handling
    5. Prompt patterns
    6. Compilation commands
    7. Error fixes
    8. Rules
    9. Template references
    """
    with open(skill_prompt_path) as f:
        content = f.read()

    content_lower = content.lower()

    # Section 1: Workflow steps
    assert "step" in content_lower, (
        "Skill prompt missing 'step' keyword. "
        "Expected a workflow section describing the 5-step process."
    )
    workflow_keywords = ["understand", "data", "prompt", "score", "ship"]
    found_workflow = sum(1 for kw in workflow_keywords if kw in content_lower)
    assert found_workflow >= 3, (
        f"Skill prompt workflow section incomplete. "
        f"Expected at least 3 of {workflow_keywords}, found {found_workflow}. "
        f"Check that the 5-step workflow (Understand, Data, Prompt, Score, Ship) is documented."
    )

    # Section 2: API reference
    assert "@benchmark" in content, (
        "Skill prompt missing '@benchmark' decorator reference. "
        "Expected BYOB API documentation."
    )
    assert "@scorer" in content, (
        "Skill prompt missing '@scorer' decorator reference. "
        "Expected BYOB API documentation."
    )
    assert "nemo_evaluator.contrib.byob" in content, (
        "Skill prompt missing 'nemo_evaluator.contrib.byob' import path. "
        "Expected BYOB API import instructions."
    )

    # Section 3: Scorer selection
    builtin_scorers = ["exact_match", "contains"]
    found_scorers = sum(1 for scorer in builtin_scorers if scorer in content_lower)
    assert found_scorers >= 2, (
        f"Skill prompt missing scorer selection logic. "
        f"Expected references to built-in scorers {builtin_scorers}. "
        f"Found {found_scorers} of 2."
    )

    # Section 4: Dataset handling
    assert "jsonl" in content_lower, (
        "Skill prompt missing 'jsonl' keyword. Expected dataset format documentation."
    )

    # Section 5: Prompt patterns (check for benchmark type keywords)
    benchmark_types = ["math", "multichoice", "qa", "classification", "safety"]
    found_types = sum(1 for btype in benchmark_types if btype in content_lower)
    assert found_types >= 3, (
        f"Skill prompt missing prompt pattern examples. "
        f"Expected at least 3 of {benchmark_types}, found {found_types}. "
        f"Check that template categories are documented."
    )

    # Section 6: Compilation
    assert "nemo-evaluator-byob" in content, (
        "Skill prompt missing 'nemo-evaluator-byob' CLI command. "
        "Expected compilation instructions."
    )

    # Section 7: Error fixes
    error_keywords = ["error", "fix", "diagnostic"]
    found_error_docs = sum(1 for kw in error_keywords if kw in content_lower)
    assert found_error_docs >= 1, (
        f"Skill prompt missing error handling documentation. "
        f"Expected at least one of {error_keywords}."
    )

    # Section 8: Rules
    assert "always" in content_lower, (
        "Skill prompt missing 'always' keyword. "
        "Expected rules/guidelines section with imperative directives."
    )

    # Section 9: Template references
    assert "template" in content_lower, (
        "Skill prompt missing 'template' keyword. "
        "Expected references to the template library."
    )


def test_skill_prompt_file_references_exist(skill_prompt_path):
    """T003: All template file references in prompt match actual files.

    Extracts file paths matching examples/byob/templates/*.py or *.jsonl
    and verifies each file exists on disk.
    """
    with open(skill_prompt_path) as f:
        content = f.read()

    # Extract file paths that look like examples/byob/...
    # Pattern: examples/byob/WORD/*.EXTENSION
    paths = re.findall(r"(examples/byob/\S+\.(?:py|jsonl))", content)
    assert len(paths) >= 6, (
        f"Expected at least 6 file references in skill prompt, found {len(paths)}. "
        f"Check that all 6 templates are referenced. Found: {paths}"
    )

    # Resolve paths relative to REPO_ROOT (packages/nemo-evaluator/)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(skill_prompt_path)))
    for rel_path in paths:
        full_path = os.path.join(repo_root, rel_path)
        assert os.path.isfile(full_path), (
            f"Referenced file not found: {rel_path}. "
            f"Full path checked: {full_path}. "
            f"Ensure template files exist and skill prompt references are accurate."
        )


def test_skill_prompt_contains_version_marker(skill_prompt_path):
    """T004: Skill prompt contains version marker.

    Validates presence of a version marker for tracking (e.g., v1.0, version 1.0).
    """
    with open(skill_prompt_path) as f:
        content = f.read()

    content_lower = content.lower()

    # Check for version patterns: "v1.0", "version 1.0", "byob-skill v1.0", etc.
    version_patterns = [
        r"v\d+\.\d+",  # v1.0, v2.3
        r"version\s+\d+\.\d+",  # version 1.0
        r"byob-skill\s+v\d+",  # byob-skill v1
    ]

    has_version = any(re.search(pattern, content_lower) for pattern in version_patterns)
    assert has_version, (
        "Skill prompt missing version marker. "
        "Expected a pattern like 'v1.0', 'version 1.0', or 'byob-skill v1.0'. "
        "This is needed for tracking changes across skill prompt revisions."
    )


def test_skill_prompt_references_critical_cli_commands(skill_prompt_path):
    """T005: Skill prompt references critical CLI commands.

    Validates that key commands and paths users need are documented.
    """
    with open(skill_prompt_path) as f:
        content = f.read()

    # Check for critical references
    assert "nemo-evaluator-byob" in content, (
        "Skill prompt missing 'nemo-evaluator-byob' compilation command. "
        "Users need this to compile benchmarks."
    )

    assert "nemo_evaluator.contrib.byob" in content, (
        "Skill prompt missing 'nemo_evaluator.contrib.byob' import path. "
        "Users need this for @benchmark and @scorer imports."
    )

    assert "run_eval" in content, (
        "Skill prompt missing 'run_eval' command reference. "
        "Users need this to execute compiled benchmarks."
    )

    # Check for byob_ prefix (package naming convention)
    assert "byob_" in content, (
        "Skill prompt missing 'byob_' prefix reference. "
        "This is the standard package naming convention for compiled benchmarks."
    )


def test_skill_prompt_encodes_scorer_smoke_test_pattern(skill_prompt_path):
    """T006: Skill prompt encodes scorer smoke test pattern.

    Validates that the prompt instructs Claude to validate scorer return types
    before compilation. This catches the most common failure mode.
    """
    with open(skill_prompt_path) as f:
        content = f.read()

    content_lower = content.lower()

    # Check for smoke test pattern
    assert "smoke test" in content_lower or "smoke-test" in content_lower, (
        "Skill prompt missing 'smoke test' reference. "
        "Expected instructions to validate scorers before compilation."
    )

    # Check for validation approach (python execution)
    assert "python" in content_lower, (
        "Skill prompt missing 'python' keyword in testing context. "
        "Expected instructions to run scorer validation via python."
    )

    # Check for return type validation
    assert "dict" in content_lower, (
        "Skill prompt missing 'dict' keyword. "
        "Expected instructions to validate scorer returns a dict."
    )
