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

# pip install nvidia-lm-eval==25.6

## Run the evaluation
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, EvaluationTarget

model_name = "megatron_model"
completions_url = "http://0.0.0.0:8080/v1/completions/"


target_config = EvaluationTarget(
    api_endpoint={
        "url": completions_url,
        "type": "completions",
    }
)
eval_config = EvaluationConfig(
    type="lm-evaluation-harness.lambada_openai",
    output_dir="/results/",
    params={
        "limit_samples": 10,
        "extra": {
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
        },
    },
)


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
