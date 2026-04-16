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
from nemo_evaluator.engine.bundles import (
    discover_bundles,
    extract_benchmark_name,
    match_bundles,
)
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.gate import (
    BenchmarkGateResult,
    GateReport,
    gate_runs,
    write_gate_report,
)
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.engine.artifacts import write_all
from nemo_evaluator.engine.comparison import (
    FlipReport,
    RegressionReport,
    build_flip_report,
    compare_results,
    compare_runs,
    load_paired_records,
    write_regression,
)

__all__ = [
    "BenchmarkGateResult",
    "FlipReport",
    "GateReport",
    "ModelClient",
    "RegressionReport",
    "build_flip_report",
    "compare_results",
    "compare_runs",
    "discover_bundles",
    "extract_benchmark_name",
    "gate_runs",
    "load_paired_records",
    "match_bundles",
    "run_evaluation",
    "write_all",
    "write_gate_report",
    "write_regression",
]
