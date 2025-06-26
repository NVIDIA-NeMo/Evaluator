# pip install nvidia-eval-factory-garak==25.5

## Export the required variables
# No environment variables are required
## Run the evaluation
from nemo.collections.llm import api
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget


model_name = "triton_model"
chat_url = "http://0.0.0.0:8886/v1/chat/completions/"


target_config = EvaluationTarget(
    api_endpoint={
        "url": chat_url, 
        "type": "chat",
        }
    )
eval_config = EvaluationConfig(
    type="garak",
    output_dir="/results/",
    params={"extra": {"probes": "ansiescape.AnsiEscaped"}},
)


results = api.evaluate(target_cfg=target_config, eval_cfg=eval_config)


print(results)
