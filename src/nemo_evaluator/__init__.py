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
"""NeMo Evaluator -- environments, solvers, evaluation orchestration."""

__version__ = "0.12.0"

from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.environments.registry import get_environment, list_environments, load_benchmark_file, register
from nemo_evaluator.scoring import (
    BooleanValue,
    CandidateOutput,
    ContinuousScore,
    DatasetRow,
    DiscreteScore,
    Label,
    Metric,
    MetricDescriptor,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    MetricScorerFunction,
    ScorerCallable,
    ScorerConfig,
    ScorerFunctionMetric,
    ScorerInput,
    ScorerReturn,
    answer_line,
    code_sandbox,
    code_sandbox_async,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    needs_judge,
    numeric_match,
    score_names_from_output_spec,
)
from nemo_evaluator.solvers import (
    ChatSolver,
    CompletionSolver,
    NatSolver,
    OpenClawSolver,
    Solver,
    SolveResult,
    VLMSolver,
)

__all__ = [
    # Core
    "EvalEnvironment",
    "SeedResult",
    "VerifyResult",
    "register",
    "get_environment",
    "list_environments",
    "load_benchmark_file",
    "run_evaluation",
    "ModelClient",
    # Solver
    "Solver",
    "ChatSolver",
    "CompletionSolver",
    "NatSolver",
    "OpenClawSolver",
    "VLMSolver",
    "SolveResult",
    # Benchmark definition API
    "benchmark",
    "scorer",
    "ScorerInput",
    "Metric",
    "BooleanValue",
    "DatasetRow",
    "CandidateOutput",
    "ContinuousScore",
    "DiscreteScore",
    "Label",
    "MetricInput",
    "MetricOutput",
    "MetricOutputSpec",
    "MetricDescriptor",
    "MetricResult",
    "MetricScorerFunction",
    "ScorerCallable",
    "ScorerConfig",
    "ScorerFunctionMetric",
    "ScorerReturn",
    "score_names_from_output_spec",
    # Scoring primitives
    "exact_match",
    "multichoice_regex",
    "answer_line",
    "fuzzy_match",
    "numeric_match",
    "code_sandbox",
    "code_sandbox_async",
    "needs_judge",
]
