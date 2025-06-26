# pip install nvidia-lm-eval==25.5

## Export the required variables
# export HF_TOKEN=...
## Run the evaluation
from nemo.collections.llm import api
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget


model_name = "triton_model"
completions_url = "http://0.0.0.0:8886/v1/completions/"


target_config = EvaluationTarget(
    api_endpoint={
        "url": completions_url, 
        "type": "completions",
        }
    )
eval_config = EvaluationConfig(
    type="arc_challenge",
    output_dir="/results/",
    params={
        "limit_samples": 10, 
        "extra": {
            "tokenizer": "meta-llama/Llama-3.1-70B-Instruct", 
            "tokenizer_backend": "huggingface"
            }
        }
    )


results = api.evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
