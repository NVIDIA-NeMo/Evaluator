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
"""SkillsEnvironment must not silently degrade to exact string matching.

Regression tests for #1141: an evaluator-dependent eval_type whose evaluator
cannot be constructed, or whose evaluator raises at scoring time, must fail
loudly unless the exact-match fallback was explicitly opted into.

No GPU, network, or nemo_skills install required: construction helpers are
patched and scoring-time behavior is tested on a directly assembled instance.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_evaluator.environments.skills import SkillsEnvironment, SkillsEvaluatorError

MOD = "nemo_evaluator.environments.skills"


def _construct(eval_type, evaluator, **kwargs):
    module = SimpleNamespace(METRICS_TYPE=eval_type)
    with (
        patch(f"{MOD}._require_skills", return_value=SimpleNamespace(get_dataset_module=lambda b: module)),
        patch(f"{MOD}._unpack_module", return_value=(module, None)),
        patch(f"{MOD}._load_dataset", return_value=[{"problem": "p", "expected_answer": "a"}]),
        patch(f"{MOD}._get_evaluator", return_value=evaluator),
    ):
        return SkillsEnvironment(benchmark="dummy_bench", **kwargs)


def _assembled(eval_type, evaluator, allow_fallback):
    env = SkillsEnvironment.__new__(SkillsEnvironment)
    env._benchmark = "dummy_bench"
    env._eval_type = eval_type
    env._evaluator = evaluator
    env._allow_exact_match_fallback = allow_fallback
    return env


class _RaisingEvaluator:
    def eval_single(self, sample):
        raise ValueError("judge endpoint not configured")


class TestConstructionGate:
    def test_missing_evaluator_for_evaluator_eval_type_raises(self):
        with pytest.raises(SkillsEvaluatorError, match="exact string matching"):
            _construct("code", evaluator=None)

    def test_missing_evaluator_allowed_with_optin(self):
        env = _construct("code", evaluator=None, allow_exact_match_fallback=True)
        reward, details = env._score("Answer", "answer", {})
        assert reward == 1.0
        assert details["method"] == "exact_match_fallback"

    def test_handler_eval_types_do_not_require_an_evaluator(self):
        env = _construct("multichoice", evaluator=None)
        assert env.eval_type == "multichoice"

    def test_constructed_evaluator_is_accepted(self):
        env = _construct("code", evaluator=_RaisingEvaluator())
        assert env._evaluator is not None


class TestScoringTimeFailure:
    def test_evaluator_failure_raises_without_optin(self):
        env = _assembled("code", _RaisingEvaluator(), allow_fallback=False)
        with pytest.raises(SkillsEvaluatorError) as excinfo:
            env._score("def f(): return 1", "reference solution", {})
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_evaluator_failure_falls_back_with_optin(self):
        env = _assembled("code", _RaisingEvaluator(), allow_fallback=True)
        reward, details = env._score("def f(): return 1", "reference solution", {})
        assert reward == 0.0
        assert details["method"] == "exact_match_fallback"

    def test_working_evaluator_unaffected(self):
        class _Working:
            def eval_single(self, sample):
                return {"is_correct": True}

        env = _assembled("code", _Working(), allow_fallback=False)
        reward, details = env._score("anything", "anything else", {})
        assert reward == 1.0
        assert details["method"] == "skills_code"
