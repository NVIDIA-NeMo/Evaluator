# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
Multi-task evaluation: Evaluate a model across multiple academic benchmarks.
"""

import os

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

# [snippet-start]
from nemo_evaluator.core.evaluate import evaluate

# Configure target endpoint (reused for all tasks)
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        type=EndpointType.CHAT,
        api_key="YOUR_API_KEY",
    )
)

# Define academic benchmark suite
academic_tasks = ["mmlu_pro", "gsm8k", "arc_challenge"]
results = {}

# Run evaluations
for task in academic_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"./results/{task}/",
        params=ConfigParams(
            limit_samples=50,  # Quick testing
            temperature=0.01,  # Deterministic
            parallelism=4,
        ),
    )

    results[task] = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"✓ Completed {task}")

# Summary report
print("\nAcademic Benchmark Results:")
for task_name, result in results.items():
    if task_name in result.tasks:
        task_result = result.tasks[task_name]
        if "acc" in task_result.metrics:
            acc = task_result.metrics["acc"].scores["acc"].value
            print(f"{task_name:20s}: {acc:.2%}")
# [snippet-end]


if __name__ == "__main__":
    api_key_name = os.getenv("API_KEY_NAME", "YOUR_API_KEY")

    if not os.getenv(api_key_name):
        print(f"Warning: Environment variable {api_key_name} not set")
        print("Set it before running: export YOUR_API_KEY='your-key-here'")
        exit(1)
