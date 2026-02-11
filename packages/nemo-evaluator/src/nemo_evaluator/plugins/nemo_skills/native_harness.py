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

"""NeMo-Skills native harness: runs nemo-skills evaluations in-process."""

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    EndpointModelConfig,
    Evaluation,
    EvaluationResult,
)
from nemo_evaluator.client.client import NeMoEvaluatorClient
from nemo_evaluator.plugins.nemo_skills.runner import evaluate as ns_evaluate


class SkillsNativeHarness:
    """NativeHarness implementation for nemo-skills evaluations.

    Per F-003 remediation: constructs NeMoEvaluatorClient from
    evaluation.target.api_endpoint for full adapter pipeline access.
    The injected model_call_fn is NOT used.

    Invariants:
    - evaluation.config.params.extra must contain: benchmark_name, eval_type, data_dir
    - Returns EvaluationResult directly (no JSON file intermediary)
    """

    def execute(
        self,
        evaluation: Evaluation,
        model_call_fn,  # ModelCallFn type - not imported to avoid circular dependency
    ) -> EvaluationResult:
        """Run nemo-skills evaluation in-process.

        Per F-003: model_call_fn is NOT used. The runner constructs its own
        NeMoEvaluatorClient from evaluation.target for full adapter pipeline access.

        Args:
            evaluation: Fully-merged evaluation configuration
            model_call_fn: Injected by engine (NOT USED per F-003)

        Returns:
            EvaluationResult with tasks populated

        Raises:
            ValueError: If required config.params.extra fields are missing
            Any exception from runner execution
        """
        # Construct NeMoEvaluatorClient from evaluation.target
        endpoint = evaluation.target.api_endpoint
        if endpoint is None:
            raise ValueError("evaluation.target.api_endpoint must not be None")

        # Build EndpointModelConfig from ApiEndpoint
        endpoint_config = EndpointModelConfig(
            model_id=endpoint.model_id,
            url=endpoint.url,
            type=endpoint.type,
            adapter_config=endpoint.adapter_config,
            api_key_name=endpoint.api_key_name,
            temperature=evaluation.config.params.temperature,
            top_p=evaluation.config.params.top_p,
            max_new_tokens=evaluation.config.params.max_new_tokens,
            max_retries=evaluation.config.params.max_retries,
            parallelism=evaluation.config.params.parallelism,
            request_timeout=evaluation.config.params.request_timeout,
        )

        client = NeMoEvaluatorClient(endpoint_config, evaluation.config.output_dir)

        # Delegate to runner
        return ns_evaluate(evaluation, client, evaluation.config.output_dir)
