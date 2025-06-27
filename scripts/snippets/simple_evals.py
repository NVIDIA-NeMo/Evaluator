# pip install nvidia-simple-evals==25.6

## Export the required variables
## Key with access to https://build.nvidia.com/ endpoints
# export JUDGE_API_KEY=...
## Run the evaluation
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, EvaluationTarget


model_name = "triton_model"
chat_url = "http://0.0.0.0:8886/v1/chat/completions/"


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
