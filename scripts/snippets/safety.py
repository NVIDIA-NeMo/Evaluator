# pip install nvidia-safety-harness==25.6

## Export the required variables
## Key with access to https://build.nvidia.com/ endpoints
# export JUDGE_API_KEY=...
# export HF_TOKEN=...
## Run the evaluation
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, EvaluationTarget


model_name = "megatron_model"
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"


target_config = EvaluationTarget(
    api_endpoint={
        "url": chat_url,
        "type": "chat",
    }
)
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="/results/",
    params={
        "limit_samples": 10,
        "extra": {
            "judge": {
                "model_id": "nvidia/llama-3.1-nemoguard-8b-content-safety",
                "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            }
        },
    },
)


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
