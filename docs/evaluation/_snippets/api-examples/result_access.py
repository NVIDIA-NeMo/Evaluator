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
Result access example: How to access and interpret evaluation results.
"""
# Assumes you have already run an evaluation and have a result object

evaluate = None
eval_config = None
target_config = None

# [snippet-start]
# Access evaluation results
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)

# Access task-level metrics
task_result = result.tasks["mmlu_pro"]
accuracy = task_result.metrics["acc"].scores["acc"].value
print(f"MMLU Pro Accuracy: {accuracy:.2%}")

# Access metrics with statistics
acc_metric = task_result.metrics["acc"]
acc = acc_metric.scores["acc"].value
stderr = acc_metric.scores["acc"].stats.stderr
print(f"Accuracy: {acc:.3f} ± {stderr:.3f}")
# [snippet-end]
