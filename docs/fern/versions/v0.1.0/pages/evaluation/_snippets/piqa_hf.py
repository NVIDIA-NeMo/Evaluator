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

# pip install nvidia-lm-eval

# [snippet-start]
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8000/v1/completions/",
    type="completions",
    model_id="meta-llama/Llama-3.1-8B",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(
    extra={
        "tokenizer": "meta-llama/Llama-3.1-8B",  # or path to locally stored checkpoint with tokenizer
        "tokenizer_backend": "huggingface",  # or "tiktoken"
    },
)
eval_config = EvaluationConfig(type="piqa", params=eval_params, output_dir="results")

evaluate(target_cfg=eval_target, eval_cfg=eval_config)
# [snippet-end]
