# pip install nvidia-lm-eval==25.5

## Export the required variables
# export HF_TOKEN=...
## Run the evaluation
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, EvaluationTarget


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
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
        },
    },
)


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
