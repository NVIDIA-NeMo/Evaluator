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
"""HumanEval -- code generation with Docker-sandboxed test execution."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, code_sandbox

_PROMPT = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n\n{prompt}"
)


def _prepare(row, idx, rng):
    return {**row, "_prompt": row["prompt"], "_test": row["test"], "_entry_point": row["entry_point"]}


@benchmark(
    name="humaneval",
    dataset="hf://openai/openai_humaneval?split=test",
    prompt=_PROMPT,
    target_field="entry_point",
    prepare_row=_prepare,
)
@scorer
def humaneval_scorer(sample: ScorerInput) -> dict:
    return code_sandbox(sample)
