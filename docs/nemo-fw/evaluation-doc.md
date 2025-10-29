# Evaluate Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) checkpoints trained using the NeMo Framework with the Megatron Core backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

Here, we focus on benchmarks in the `lm-evaluation-harness` that rely on text generation.
In this approach, the model is given a prompt such as a question to answer, an instruction to follow, or a text to continue, and its response is then evaluated for correctness.

An alternative approach to LLM evaluation utilizes **log-probabilities**.
To learn more, please refer to ["Evaluate LLMs Using Log-Probabilities"](logprobs.md).

Use the `show_available_tasks` function to list the evaluation configs available in your environment:

```python
from nemo_evaluator import show_available_tasks

show_available_tasks()
```

This will print a list of eval harnesses and configs available in each of them:

```
lm-evaluation-harness: 
  * mmlu
  * mmlu_instruct
  * mmlu_cot_0_shot_chat
  * ifeval
  * mmlu_pro
  * mmlu_pro_instruct
  * mmlu_redux
  ...
  * bbq
  * arc_multilingual
  * hellaswag_multilingual
  * mmlu_prox
```

## Deploy and Evaluate NeMo Checkpoints

The evaluation process employs a server-client approach, comprising two main phases. 
- **Phase 1: Model Deployment**
    - Deployment via PyTriton: The Megatron Bridge checkpoint is deployed in-framework on a PyTriton server by exposing OpenAI API (OAI)-compatible endpoints. Both completions (`v1/completions`) and chat-completions (`v1/chat/completions`) endpoints are exposed, enabling evaluation on both completion and chat benchmarks.
    - Deployment via Ray: The Megatron Bridge checkpoint can also be deployed in-framework on a Ray server. Ray Serve provides support for multi-instance evaluations, along with OpenAI API (OAI)-compatible endpoints. Both completions (`v1/completions`) and chat-completions (`v1/chat/completions`) endpoints are exposed. For more details on evaluations with Ray Serve, refer to ["Use Ray Serve for Multi-Instance Evaluations"](evaluation-with-ray.md).

- **Phase 2: Model Evaluation**
    - Evaluation via OAI Endpoints: Once the model is deployed, evaluation is performed by sending benchmark requests to the exposed OAI-compatible endpoints using their respective port. This allows assessment across a range of tasks and harnesses.

> **Note:** Some of the benchmarks (e.g., GPQA) use a gated dataset. To access them, you must authenticate to the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) before launching the evaluation.

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) includes [`nvidia-lm-eval`](https://pypi.org/project/nvidia-lm-eval/), which comes pre-installed. This tool provides predefined configurations for evaluating the completions endpoint, such as:

- `gsm8k`
- `mgsm`
- `mmlu`
- `mmlu_pro`
- `mmlu_redux`

It also provides predefined configurations for evaluating the chat endpoint, such as:

- `gpqa_diamond_cot`
- `gsm8k_cot_instruct`
- `ifeval`
- `mgsm_cot`
- `mmlu_instruct`
- `mmlu_pro_instruct`
- `mmlu_redux_instruct`
- `wikilingua`

When defining a task in EvaluationConfig (see detailed examples in the [Evaluate Models Locally on Your Workstation](#evaluate-models-locally-on-your-workstation) section), you can use either the task name from the list above or prefix it with the harness name. For example:

```python
eval_config = EvaluationConfig(type="mmlu")
eval_config = EvaluationConfig(type="lm-evaluation-harness.mmlu")
```

A subtask of a benchmark (e.g., `mmlu_str_high_school_european_history` under `mmlu`) can also be specified similar to above:

```python
eval_config = EvaluationConfig(type="mmlu_str_high_school_european_history")
eval_config = EvaluationConfig(type="lm-evaluation-harness.mmlu_str_high_school_european_history")
```

To enable additional evaluation harnesses, like  `simple-evals`, `BFCL`, `garak`, `BigCode`, or `safety-harness`, you need to install them. For example:

```bash
pip install nvidia-simple-evals
```

For more information on enabling additional evaluation harnesses, see the ["Add On-Demand Evaluation Packages"](optional-eval-package.md) section.
If multiple harnesses are installed in your environment and they define a task with the same name, you must use the `<harness>.<task>` format to avoid ambiguity. For example:

```python
eval_config = EvaluationConfig(type="lm-evaluation-harness.mmlu")
eval_config = EvaluationConfig(type="simple-evals.mmlu")
```

To evaluate your model on a task without a pre-defined config, see ["Run Evaluation Using Task Without Pre-Defined Config"](custom-task.md).

## Evaluate Models Locally on Your Workstation

This section outlines the steps to deploy and evaluate the Megatron Bridge using Python commands. This method is quick and easy, making it ideal for evaluation on a local workstation with GPUs, as it facilitates easier debugging. However, for running evaluations on clusters, it is recommended to use NeMo Run for its ease of use (see the next section).

The deployment scripts are available inside the [`/opt/Export-Deploy/scripts/deploy/nlp/`](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts/deploy/nlp) directory. Below is an example command for deployment. It uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to Megatron Bridge format. To evaluate a checkpoint saved during pretraining or fine-tuning with [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html), provide the path to the saved checkpoint using the `--megatron_checkpoint` flag in the command below.

```shell
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --megatron_checkpoint "/workspace/llama3_8b/iter_0000000" \
  --triton_model_name "megatron_model" \
  --server_port 8080 \
  --num_gpus 1 \
  --max_batch_size 4 \
  --inference_max_seq_length 4096
```

> **Note:** Megatron Bridge creates checkpoints in directories named `iter_N`, where *N* is the iteration number. Each `iter_N` directory contains model weights and related artifacts. When using a checkpoint, make sure to provide the path to the appropriate `iter_N` directory. Hugging Face checkpoints converted for Megatron Bridge are typically stored in a directory named `iter_0000000`, as shown in the command above.

> **Note:** To evaluate NeMo 2.0 checkpoints, replace the `--megatron_checkpoint` flag in the deployment command above with `--nemo_checkpoint` and provide the path for the NeMo checkpoint.

The entry point for evaluation is the [`evaluate`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/evaluate.py) function. To run evaluations on the deployed model, use the following command. Make sure to open a new terminal within the same container to execute it. For longer evaluations, it is advisable to run both the deploy and evaluate commands in tmux sessions to prevent the processes from being terminated unexpectedly and aborting the runs.
It is recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests before starting the evaluation.

```python
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(top_p=0, temperature=0, limit_samples=2, parallelism=1)
eval_config = EvaluationConfig(type='mmlu', params=eval_params, output_dir="results")

if __name__ == "__main__":
    check_endpoint(
            endpoint_url=eval_target.api_endpoint.url,
            endpoint_type=eval_target.api_endpoint.type,
            model_name=eval_target.api_endpoint.model_id,
        )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```

> **Note:** To evaluate the chat endpoint, update the url by replacing `/v1/completions/` with `/v1/chat/completions/`. Additionally, set the `type` field to `"chat"` in both `ApiEndpoint` and `EvaluationConfig` to indicate a chat benchmark. A list of available chat benchmarks can be found in the [Deploy and Evaluate NeMo Checkpoints](#deploy-and-evaluate-nemo-checkpoints) section above.

> **Note:** Please refer to the [`deploy_inframework_triton.py`](https://github.com/NVIDIA-NeMo/Export-Deploy/blob/main/scripts/deploy/nlp/deploy_inframework_triton.py) script and [`evaluate`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/evaluate.py) function to review all available argument options, as the provided commands are only examples and do not include all arguments or their default values. For more detailed information on the arguments used in the ApiEndpoint and ConfigParams classes for evaluation, see the [`api_dataclasses`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/api/api_dataclasses.py) submodule.

> **Tip:** If you encounter a TimeoutError on the eval client side, please increase the `request_timeout` parameter in `ConfigParams` class to a larger value like `1000` or `1200` seconds (the default is 300).

## Run Evaluations with NeMo Run

This section explains how to run evaluations with NeMo Run. For detailed information about [NeMo Run](https://github.com/NVIDIA/NeMo-Run), please refer to its documentation. Below is a concise guide focused on using NeMo Run to perform evaluations in NeMo.

The [evaluation_with_nemo_run.py](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/scripts/evaluation_with_nemo_run.py) script serves as a reference for launching evaluations with NeMo Run. This script demonstrates how to use NeMo Run with both local executors (your local workstation) and Slurm-based executors like clusters. In this setup, the deploy and evaluate processes are launched as two separate jobs with NeMo Run. The evaluate method waits until the PyTriton server is accessible and the model is deployed before starting the evaluations.

> **Note:** If you're using the `local_executor`, update the `HF_TOKEN` in the NeMo Run script's [local_executor `env_vars`](https://github.com/nvidia-nemo/evaluator/blob/main/scripts/evaluation_with_nemo_run.py#l267). If you're using the `slurm_executor`, update it in the [slurm_executor's `env_vars`](https://github.com/nvidia-nemo/evaluator/blob/main/scripts/evaluation_with_nemo_run.py#l232) instead.

### Run Locally with NeMo Run

To run evaluations on your local workstation, use the following command:

```bash
cd scripts
python evaluation_with_nemo_run.py --nemo_checkpoint '/workspace/llama3_8b_nemo2/' --eval_task 'gsm8k' --devices 2
```

> **Note:** When running locally with NeMo Run, you will need to manually terminate the deploy process once evaluations are complete.

### Run on Slurm-based Clusters

To run evaluations on Slurm-based clusters, add the `--slurm` flag to your command and specify any custom parameters such as user, host, remote_job_dir, account, mounts, etc. Refer to the `evaluation_with_nemo_run.py` script for further details. Below is an example command:

```bash
cd scripts
python evaluation_with_nemo_run.py --nemo_checkpoint='/workspace/llama3_8b_nemo2' --slurm --nodes 1 \
  --devices 8 --container_image "nvcr.io/nvidia/nemo:25.07" --tensor_parallelism_size 8
```
By following these commands, you can successfully run evaluations using NeMo Run on both local and Slurm-based environments.
