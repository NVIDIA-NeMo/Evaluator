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

"""Output parser for subprocess mode nemo-skills evaluations."""

from pathlib import Path

from nemo_evaluator._nemo_skills._adapters.file_utils import jload
from nemo_evaluator.api.api_dataclasses import EvaluationResult
from nemo_evaluator.plugins.nemo_skills.metrics_adapter import translate


def parse_output(output_dir: str) -> EvaluationResult:
    """Parses ns_results.json and returns EvaluationResult.

    Per C-033: reads ns_results.json from output_dir, extracts benchmark_name,
    calls metrics_adapter.translate() to produce EvaluationResult.

    Args:
        output_dir: Directory containing ns_results.json

    Returns:
        EvaluationResult with tasks populated

    Raises:
        FileNotFoundError: If ns_results.json does not exist
    """
    results_path = Path(output_dir) / "ns_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"ns_results.json not found in {output_dir}")

    ns_results = jload(str(results_path))
    benchmark_name = ns_results.get("benchmark_name", "unknown")
    ns_metrics = ns_results.get("_all_", {})
    config = ns_results.get("config", {})

    # Wrap ns_metrics in expected structure for translate
    wrapped_metrics = {"_all_": ns_metrics}

    return translate(wrapped_metrics, benchmark_name, config)
