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

"""Code generation benchmark template.

Suitable for: HumanEval, MBPP, coding interview problems.
Scoring: Extract code from response, execute test assertion.

WARNING: This template uses exec() to run generated code. Only use with
trusted test cases. Never run on untrusted input.

Dataset fields: prompt (str), test (str), entry_point (str)
Target field: test
"""

import os

from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="code_generation",
    dataset=os.path.join(_SCRIPT_DIR, "code_generation_data.jsonl"),
    prompt=(
        "Write a Python function to solve the following problem.\n"
        "Return only the function definition, no explanation.\n\n"
        "{prompt}"
    ),
    target_field="test",
    endpoint_type="chat",
)
@scorer
def code_scorer(sample: ScorerInput) -> dict:
    """Extract Python code from response and run test assertions.

    Extracts code from markdown code blocks if present.
    Executes code + test string in isolated namespace.
    """
    import re

    # Extract code from markdown block if present
    code_match = re.search(r"```(?:python)?\n(.*?)```", sample.response, re.DOTALL)
    code = code_match.group(1) if code_match else sample.response

    # Clean up common artifacts
    code = code.strip()

    try:
        namespace = {}
        exec(code, namespace)
        exec(sample.target, namespace)
        return {"correct": True, "parsed": True, "error": False}
    except AssertionError:
        return {"correct": False, "parsed": True, "error": False}
    except Exception:
        return {"correct": False, "parsed": bool(code_match), "error": True}
