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

## Run the evaluation
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(
    top_p=0,
    temperature=0,
    limit_samples=1,
    parallelism=1,
    extra={
        "tokenizer": "/workspace/mbridge_llama3_8b/iter_0000000/tokenizer",
        "tokenizer_backend": "huggingface",
    },
)
eval_config = EvaluationConfig(
    type="arc_challenge", params=eval_params, output_dir="results"
)

if __name__ == "__main__":
    check_endpoint(
        endpoint_url=eval_target.api_endpoint.url,
        endpoint_type=eval_target.api_endpoint.type,
        model_name=eval_target.api_endpoint.model_id,
    )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
