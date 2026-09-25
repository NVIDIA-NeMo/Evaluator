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
"""MCQ scoring must distinguish "no verdict" from "wrong answer" (issue #1140)."""

import pytest

from nemo_evaluator.environments.vlmevalkit import VLMEvalKitEnvironment

CHOICES = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}


def _score(response: str, expected: str = "B") -> tuple[float, bool, str | None]:
    # ``_score_mcq`` never touches ``self``; calling it unbound keeps the test off
    # ``__init__``, which requires the optional ``vlmeval`` package and a dataset download.
    vr = VLMEvalKitEnvironment._score_mcq(None, response, expected, CHOICES)
    details = vr.scoring_details
    return vr.reward, details.get("unscored", False), details.get("unscored_reason")


@pytest.mark.parametrize(
    "response,reward,unscored,reason",
    [
        ("The answer is B", 1.0, False, None),
        ("B", 1.0, False, None),
        ("The answer is C", 0.0, False, None),
        ("I cannot tell.", 0.0, True, "format_error"),
        ("The image is too blurry to read.", 0.0, True, "format_error"),
        ("Maybe A, or possibly C.", 0.0, True, "format_error"),
        ("", 0.0, True, "empty_final_response"),
        ("  \n\t ", 0.0, True, "empty_final_response"),
    ],
    ids=[
        "correct_letter_in_sentence",
        "correct_bare_letter",
        "wrong_but_extracted",
        "refusal_no_option",
        "hedge_no_option",
        "ambiguous_two_options",
        "empty_response",
        "whitespace_only_response",
    ],
)
def test_mcq_unscored_tagging(response, reward, unscored, reason):
    assert _score(response) == (reward, unscored, reason)


def test_scored_samples_carry_no_unscored_keys():
    """A scored sample must not grow the keys, or every consumer has to special-case them."""
    details = VLMEvalKitEnvironment._score_mcq(None, "The answer is B", "B", CHOICES).scoring_details
    assert "unscored" not in details
    assert "unscored_reason" not in details


def test_unscored_keeps_the_existing_details():
    details = VLMEvalKitEnvironment._score_mcq(None, "I cannot tell.", "B", CHOICES).scoring_details
    assert details["method"] == "vlmevalkit_mcq"
    assert details["predicted_option"] is None
    assert details["expected_option"] == "B"
    assert details["exact_match"] is False
