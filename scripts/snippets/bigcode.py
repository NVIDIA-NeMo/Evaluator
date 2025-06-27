# pip install nvidia-bigcode-eval==25.5

## Export the required variables
# No environment variables are required
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
eval_config = EvaluationConfig(type="mbpp", output_dir="/results/", params={"limit_samples": 10})


results = evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
