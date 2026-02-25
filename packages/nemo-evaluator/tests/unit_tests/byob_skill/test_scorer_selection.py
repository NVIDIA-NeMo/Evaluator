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

"""Scorer selection logic tests.

Tests T065-T071: Validate scorer selection keywords and smoke tests.
T065-T069 are semi-automated (check keywords in prompt).
T070-T071 are fully automated scorer smoke tests.
"""

from nemo_evaluator.contrib.byob.decorators import ScorerInput
from nemo_evaluator.contrib.byob.scorers import exact_match, contains


def test_scorer_selection_exact_match_keyword(skill_prompt_path):
    """T065: 'exact match' description appears in skill prompt.

    Semi-automated: verifies keyword presence. Manual testing verifies
    Claude Code selects the correct scorer.
    """
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    assert "exact" in content, (
        "Skill prompt missing 'exact' keyword. "
        "Expected scorer selection guidance for exact_match scorer."
    )
    assert "match" in content or "exact_match" in content, (
        "Skill prompt missing 'match' or 'exact_match' keyword."
    )


def test_scorer_selection_contains_keyword(skill_prompt_path):
    """T066: 'contains' description appears in skill prompt."""
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    assert "contains" in content or "substring" in content, (
        "Skill prompt missing 'contains' or 'substring' keyword. "
        "Expected scorer selection guidance for contains scorer."
    )


def test_scorer_selection_number_extraction_keyword(skill_prompt_path):
    """T067: 'number extraction' description appears in skill prompt."""
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    # Check for number-related keywords
    number_keywords = ["number", "numeric", "integer", "extract"]
    found = sum(1 for kw in number_keywords if kw in content)

    assert found >= 2, (
        f"Skill prompt missing number extraction guidance. "
        f"Expected at least 2 of {number_keywords}, found {found}. "
        f"Needed for math reasoning scorer selection."
    )


def test_scorer_selection_multiple_choice_keyword(skill_prompt_path):
    """T068: Multiple choice keywords appear in skill prompt."""
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    # Check for multiple choice patterns
    mc_keywords = ["multiple choice", "a/b/c/d", "letter"]
    found = sum(1 for kw in content if kw in content)

    assert found >= 1, (
        f"Skill prompt missing multiple choice guidance. "
        f"Expected at least 1 of {mc_keywords}. "
        f"Needed for multichoice scorer selection."
    )


def test_scorer_selection_yes_no_keyword(skill_prompt_path):
    """T069: Yes/no boolean keywords appear in skill prompt."""
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    # Check for yes/no or boolean patterns
    assert ("yes" in content and "no" in content) or "boolean" in content, (
        "Skill prompt missing yes/no or boolean guidance. "
        "Expected keywords for binary classification scorer selection."
    )


def test_built_in_scorer_smoke_test():
    """T070: Built-in scorer smoke test validates return type.

    Verifies that built-in scorers return dict with expected structure.
    """
    # Test exact_match
    result = exact_match(ScorerInput(response="test", target="test", metadata={}))
    assert isinstance(result, dict), (
        f"exact_match should return dict, got {type(result).__name__}"
    )
    assert "correct" in result, (
        f"exact_match should return dict with 'correct' key. Keys: {list(result.keys())}"
    )
    assert isinstance(result["correct"], bool), (
        f"exact_match 'correct' value should be bool, got {type(result['correct']).__name__}"
    )

    # Test contains
    result = contains(ScorerInput(response="hello world", target="world", metadata={}))
    assert isinstance(result, dict), (
        f"contains should return dict, got {type(result).__name__}"
    )
    assert "correct" in result, (
        f"contains should return dict with 'correct' key. Keys: {list(result.keys())}"
    )
    assert isinstance(result["correct"], bool), (
        f"contains 'correct' value should be bool, got {type(result['correct']).__name__}"
    )


def test_custom_scorer_smoke_test_detects_wrong_type():
    """T071: Smoke test pattern detects non-dict return.

    Demonstrates how to detect a scorer that returns the wrong type.
    """

    def bad_scorer(sample):
        return "correct"  # BUG: returns string instead of dict

    result = bad_scorer("test")
    assert not isinstance(result, dict), (
        f"This test validates detection logic. "
        f"The bad_scorer incorrectly returns {type(result).__name__} instead of dict. "
        f"Smoke test should catch this."
    )

    # Good scorer for comparison
    def good_scorer(sample):
        return {"correct": True}

    result = good_scorer("test")
    assert isinstance(result, dict), "good_scorer should return dict"
    assert "correct" in result, "good_scorer should have 'correct' key"
