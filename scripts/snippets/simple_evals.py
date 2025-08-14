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

# pip install nvidia-simple-evals==25.6

## Export the required variables
## Key with access to https://build.nvidia.com/ endpoints
# export JUDGE_API_KEY=...
## Run the evaluation
from nvidia_eval_commons.api.api_dataclasses import EvaluationConfig, EvaluationTarget

from nemo_eval.api import evaluate

model_name = "megatron_model"
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"


target_config = EvaluationTarget(
    api_endpoint={
        "url": chat_url,
        "type": "chat",
    }
)
eval_config = EvaluationConfig(
    type="AIME_2025",
    output_dir="/results/",
    params={"limit_samples": 10},
)


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
