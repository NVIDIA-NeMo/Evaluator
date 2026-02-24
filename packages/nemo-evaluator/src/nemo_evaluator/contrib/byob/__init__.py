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

"""BYOB (Bring Your Own Benchmark) framework for NeMo Evaluator.

Quick start::

    from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

    @benchmark(name="my-qa", dataset="data.jsonl", prompt="Q: {q}\\nA:")
    @scorer
    def check(inp: ScorerInput):
        return {"correct": inp.target.lower() in inp.response.lower()}

Built-in scorers (import from nemo_evaluator.contrib.byob.scorers)::

    contains(response, target, metadata) -> {"correct": bool}
        Case-insensitive substring match.

    exact_match(response, target, metadata) -> {"correct": bool}
        Case-insensitive, whitespace-stripped equality.

    f1_token(response, target, metadata) -> {"f1": float, "precision": float, "recall": float}
        Token-level F1 score.

    regex_match(response, target, metadata) -> {"correct": bool}
        Regex pattern match (target is the pattern).

Scorer composition::

    from nemo_evaluator.contrib.byob import any_of, all_of
    from nemo_evaluator.contrib.byob.scorers import contains

    combined = any_of(contains, my_custom_scorer)
"""
from nemo_evaluator.contrib.byob.decorators import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.scorers import any_of, all_of
from nemo_evaluator.contrib.byob.judge import judge_score

__all__ = ["benchmark", "scorer", "ScorerInput", "any_of", "all_of", "judge_score"]
