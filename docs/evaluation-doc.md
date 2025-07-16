# Evaluate Checkpoints Trained by NeMo Framework

This guide provides detailed instructions on evaluating checkpoints trained by NeMo Framework via the megatron-core backend. This section in particular focuses on using [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/) for evaluation, which is a wrapper around [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).

In this section we will focus on benchmarks in `lm-evaluation-harness` that use **generation**.
In such approach the model is presented with a question it needs to answer, an instruction it should follow, or a text that it should continue.
Then the model's answer is evaluated for its correctness.

An alternative approach to LLM evaluation utilizes **log-probabilities**.
To learn more please refer to ["Evaluate LLMs Using Log-Probabilities"](logprobs.md)

Use `list_available_evaluations` function to list the evaluation configs available in your evironment:

```python
from nemo_eval.utils.base import list_available_evaluations

list_available_evaluations()
```

This will return a dictionary with eval packages as keys, and list of available configs as values:

```
{'core_evals.lm_evaluation_harness': ['mmlu',
  'mmlu_instruct',
  'mmlu_cot_0_shot_chat',
  'ifeval',
  'mmlu_pro',
  'mmlu_pro_instruct',
  'mmlu_redux',
  ...
  'bbq',
  'arc_multilingual',
  'hellaswag_multilingual',
  'mmlu_prox']}
```

## Introduction

The evaluation process employs a server-client approach, comprising two main phases. 
- **Phase 1: Model Deployment**
    - Deployment via PyTriton: NeMo FW checkpoint is deployed in-framework on a PyTriton server by exposing OpenAI API (OAI) compatible endpoints. Both completions (`v1/completions`) and chat-completions (`v1/chat/completions`) endpoints are exposed, enabling evaluation on both completion and chat benchmarks.
    - Deployment via Ray: NeMo FW checkpoint can also be deployed in-framework on a Ray server. Ray serve provides support for multi instance evaluation along with OpenAI API (OAI) compatible endpoints. Both completions (`v1/completions`) and chat-completions (`v1/chat/completions`) endpoints are exposed. For more details on evaluations with Ray serve refer
    to ["Multi instance evaluations with Ray Serve"](evaluation-with-ray.md).

- **Phase 2: Model Evaluation**
    - Involves running the evaluation on the model using the OAI endpoint and port.

> **Note:** Some of the benchmarks (e.g. GPQA) use a gated dataset. To use them, you must authenticate to the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) before launching the evaluation.

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) includes [`nvidia-lm-eval`](https://pypi.org/project/nvidia-lm-eval/) pre-installed. `nvidia-lm-eval` contains predefined configurations for evaluating the completions endpoint, e.g.:

- `gsm8k`
- `mgsm`
- `mmlu`
- `mmlu_pro`
- `mmlu_redux`

It also provides predefined configurations for evaluating the chat endpoint, e.g.:

- `gpqa_diamond_cot`
- `gsm8k_cot_instruct`
- `ifeval`
- `mgsm_cot`
- `mmlu_instruct`
- `mmlu_pro_instruct`
- `mmlu_redux_instruct`
- `wikilingua`


When specifying the task, you can either use the task name from the list above or prepend it with the harness name. For example:

```python
task = "mmlu"
task = "lm-evaluation-harness.mmlu"
task = "lm_evaluation_harness.mmlu"
```

To enable additional evaluation harnesses like  `simple-evals`, `BFCL`, `garak`, `BigCode`, `safety-harness`, you need to install them. For example:

```bash
pip install nvidia-simple-evals
```

For more information on enabling additional evaluation harnesses, see ["Add On-Demand Evaluation Packages"](optional-eval-package.md) section.
If multiple harnesses are installed in your environment and they define a task with the same name, you must use the `<harness>.<task>` format to avoid ambiguity. For example:

```python
task = "lm-evaluation-harness.mmlu"
task = "simple-evals.mmlu"
```

To evaluate your model on a task without pre-defined config, see ["Run Evaluation Using Task Without Pre-Defined Config"](custom-task.md)

## Evaluate Models Locally on Your Workstation

This section outlines the steps to deploy and evaluate a checkpoint trained by Nemo Framework directly using Python commands. This method is quick and easy, making it ideal for evaluation on a local workstation with GPUs, as it facilitates easier debugging. However, for running evaluations on clusters, it is recommended to use NeMo-Run for its ease of use (see next section).

The entry point for deployment is the `deploy` method defined in `nemo/collections/llm/api.py`. Below is an example command for deployment. It uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to NeMo format. To evaluate a checkpoint saved during [pretraining](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#pretraining) or [fine-tuning](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#fine-tuning), provide the path to the saved checkpoint using the `nemo_checkpoint` argument in the `deploy` command below.

```python
from nemo_eval.api import deploy

if __name__ == "__main__":
    deploy(
        nemo_checkpoint='/workspace/llama3_8b_nemo2',
        max_input_len=4096,
        max_batch_size=4,
        server_port=8080,
        num_gpus=1,)
```

The entry point for evaluation is the `evaluate` method defined in `nemo_eval/api.py`. To run evaluations on the deployed model, use the following command. Make sure to open a new terminal within the same container to execute it. For longer evaluations, it is advisable to run both the deploy and evaluate commands in tmux sessions to prevent the processes from being terminated unexpectedly and aborting the runs.

```python
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions"
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(top_p=1, temperature=1, limit_samples=2, parallelism=1)
eval_config = EvaluationConfig(type='mmlu', params=eval_params)

if __name__ == "__main__":
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```

> **Note:** For evaluating the chat endpoint, replace `/v1/completions/` with `/v1/chat/completions/` in the `url` and update the `type` as `chat` in `ApiEndpoint`. Also update `type` in `EvaluationConfig` to be a chat benchmark. Available chat benchmarks are listed in the [Introduction](#introduction) section above.

> **Note:** Please refer to `deploy` and `evaluate` method in `nemo_eval/api.py` to review all available argument options, as the provided commands are only examples and do not include all arguments or their default values. For more detailed information on the arguments used in the ApiEndpoint and ConfigParams classes for evaluation, see the source code at [nemo_eval/utils/api.py](https://github.com/NVIDIA-NeMo/Eval/blob/main/src/nemo_eval/utils/api.py).

## Run Evaluations with NeMo-Run

This section explains how to run evaluations with NeMo-Run. For detailed information about [NeMo-Run](https://github.com/NVIDIA/NeMo-Run), please refer to its documentation. Below is a concise guide focused on using NeMo-Run to perform evaluations in NeMo.

The [evaluation_with_nemo_run.py](https://github.com/NVIDIA-NeMo/Eval/blob/main/scripts/evaluation_with_nemo_run.py) script serves as a reference for launching evaluations with NeMo-Run. This script demonstrates how to use NeMo-Run with both local executors (your local workstation) and Slurm-based executors like clusters. In this setup, the deploy and evaluate processes are launched as two separate jobs with NeMo-Run. The evaluate method waits until the PyTriton server is accessible and the model is deployed before starting the evaluations.

> **Note:** Please make sure to update HF_TOKEN in the NeMo-Run script's [local_executor env_vars](https://github.com/NVIDIA-NeMo/Eval/blob/main/scripts/evaluation_with_nemo_run.py#L197) with your HF_TOKEN if using local executor or in the [slurm_executor's env_vars](https://github.com/NVIDIA-NeMo/Eval/blob/main/scripts/evaluation_with_nemo_run.py#L164) if using slurm_executor.

### Run Locally with NeMo-Run

To run evaluations on your local workstation, use the following command:

```bash
python scripts/evaluation_with_nemo_run.py --nemo_checkpoint '/workspace/llama3_8b_nemo2/' --eval_task 'gsm8k' --devices 2
```

> **Note:** When running locally with NeMo-Run, you will need to manually terminate the deploy process once evaluations are complete.

### Run on Slurm-based Clusters

To run evaluations on Slurm-based clusters, add the `--slurm` flag to your command and specify any custom parameters such as user, host, remote_job_dir, account, mounts, etc. Refer to the evaluation.py script for further details. Below is an example command:

```bash
python scripts/evaluation_with_nemo_run.py --nemo_checkpoint='/workspace/llama3_8b_nemo2' --slurm --nodes 1
--devices 8 --container_image "nvcr.io/nvidia/nemo:25.07" --tensor_parallelism_size 8
```
By following these commands, you can successfully run evaluations using NeMo-Run on both local and Slurm-based environments.
