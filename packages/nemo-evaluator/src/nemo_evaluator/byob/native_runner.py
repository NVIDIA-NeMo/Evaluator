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

"""BYOB native harness: runs BYOB evaluations in-process."""

import sys

from nemo_evaluator.api.api_dataclasses import Evaluation, EvaluationResult
from nemo_evaluator.byob.decorators import clear_registry
from nemo_evaluator.byob.eval_logic import (
    build_evaluation_result,
    import_benchmark,
    run_eval_loop,
)


class ByobNativeHarness:
    """NativeHarness implementation for BYOB benchmarks.

    Executes the BYOB evaluation loop in-process, receiving model calls
    through an injected model_call_fn instead of making raw HTTP requests.
    This enables the engine to route calls through the adapter pipeline
    transparently.

    Invariants:
    - Registry is cleared before and after execution to prevent state pollution
    - sys.path and sys.modules are restored after execution
    - config.params.extra must contain: benchmark_module, benchmark_name, dataset
    - Returns EvaluationResult directly (no JSON file intermediary)
    """

    def execute(
        self,
        evaluation: Evaluation,
        model_call_fn,  # ModelCallFn type - not imported to avoid circular dependency
    ) -> EvaluationResult:
        """Run BYOB evaluation in-process.

        Extracts benchmark configuration from evaluation.config.params.extra,
        imports the benchmark module, loads the dataset, runs the evaluation
        loop using the injected model_call_fn, and returns the result directly.

        Args:
            evaluation: Fully-merged evaluation configuration.
            model_call_fn: Callable(prompt, endpoint_type) -> response_text.
                          Injected by the engine; routes through adapter pipeline.

        Returns:
            EvaluationResult with tasks populated.

        Raises:
            ValueError: If required config.params.extra fields are missing.
            Any exception from benchmark import, dataset loading, or eval loop.
        """
        # Extract required config from evaluation.config.params.extra
        extra = evaluation.config.params.extra or {}
        benchmark_module = extra.get("benchmark_module")
        benchmark_name = extra.get("benchmark_name")
        dataset_path = extra.get("dataset")

        if not benchmark_module or not benchmark_name or not dataset_path:
            raise ValueError(
                "Native BYOB evaluation requires config.params.extra fields: "
                "benchmark_module, benchmark_name, dataset. "
                f"Got: {extra}"
            )

        # Determine endpoint type from target configuration
        endpoint_type = "chat"
        if (evaluation.target.api_endpoint
                and evaluation.target.api_endpoint.type):
            endpoint_type = str(evaluation.target.api_endpoint.type)

        # Determine sample limit
        limit_samples = evaluation.config.params.limit_samples
        if isinstance(limit_samples, float):
            limit_samples = int(limit_samples)

        # Capture sys.path and sys.modules for restoration after execution
        saved_path = sys.path[:]
        saved_modules = set(sys.modules.keys())

        try:
            # Import benchmark (clears registry internally)
            bench = import_benchmark(benchmark_module, benchmark_name)

            # Load dataset (import here to avoid circular dependency)
            from nemo_evaluator.byob.runner import load_dataset
            dataset = load_dataset(dataset_path, limit=limit_samples)

            # Run evaluation loop (returns tuple: scores, predictions)
            scores, _predictions = run_eval_loop(
                bench=bench,
                dataset=dataset,
                model_call_fn=model_call_fn,
                endpoint_type=endpoint_type,
            )

            # Build result
            result = build_evaluation_result(scores, benchmark_name)

            return result

        finally:
            # Always clean up registry to prevent state leakage between evaluations
            clear_registry()

            # Restore sys.path
            sys.path[:] = saved_path

            # Remove modules added during execution (but keep nemo_evaluator.*)
            new_modules = set(sys.modules.keys()) - saved_modules
            for mod_name in new_modules:
                if not mod_name.startswith("nemo_evaluator."):
                    sys.modules.pop(mod_name, None)
