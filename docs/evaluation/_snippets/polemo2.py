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
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
completions_url = "https://8d02791a-18f8-41eb-9a42-c21d748e0356.invocation.api.nvcf.nvidia.com/v1/completions"


target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url=completions_url,
        type=EndpointType.COMPLETIONS,
        model_id=model_name,
        api_key="API_KEY",
    )
)

eval_config = EvaluationConfig(
    type="lm-evaluation-harness.polemo2",
    output_dir="/results/",
)


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
