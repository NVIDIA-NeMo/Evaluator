# Run Evaluations with NeMo Run

This tutorial explains how to run evaluations inside NeMo Framework container with NeMo Run.
For detailed information about [NeMo Run](https://github.com/NVIDIA/NeMo-Run), please refer to its documentation.
Below is a concise guide focused on using NeMo Run to perform evaluations in NeMo.

## Prerequisites

- Docker installed
- [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- Access to a NeMo 2.0 checkpoint (tutorials use Llama 3.2 1B Instruct)
- CUDA-compatible GPU with sufficient memory (for running locally) or access to a Slurm-based Cluster (for running on cluster).
- NeMo Evaluator repository cloned (for access to [scripts](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/scripts))
  ```bash
  git clone https://github.com/NVIDIA-NeMo/Evaluator.git
  ```
- (Optional) Your Hugging Face token if you are using gated datasets (e.g. [GPQA-Diamond dataset](https://huggingface.co/datasets/Idavidrein/gpqa)).


## How it works

The [evaluation_with_nemo_run.py](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/scripts/evaluation_with_nemo_run.py) script serves as a reference for launching evaluations with NeMo Run.
This script demonstrates how to use NeMo Run with both local executors (your local workstation) and Slurm-based executors like clusters.
In this setup, the deploy and evaluate processes are launched as two separate jobs with NeMo Run. The evaluate method waits until the PyTriton server is accessible and the model is deployed before starting the evaluations.

For this purpose we define a helper function:

```{literalinclude} ../../../scripts/helpers.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

The script supports two types of serving: with Triton (default) and with Ray (pass `--serving_backend ray` flag).
User-provided arguments are mapped onto flags exptected by the scripts:

```{literalinclude} ../../../scripts/evaluation_with_nemo_run.py
:language: python
:start-after: "# # [snippet-deploy-start]"
:end-before: "# # [snippet-deploy-end]"
```

The script supports two modes of running the experiment:

- locally, using your environment
- remotely, sending the job to the Slurm-based cluster

First, an executor is selected based on the arguments provided by the user, either a local one:

```{literalinclude} ../../../scripts/evaluation_with_nemo_run.py
:language: python
:start-after: "# [snippet-local-executor-start]"
:end-before: "# [snippet-local-executor-end]"
```
or a Slurm one:

```{literalinclude} ../../../scripts/evaluation_with_nemo_run.py
:language: python
:start-after: "# [snippet-slurm-executor-start]"
:end-before: "# [snippet-slurm-executor-end]"
```

:::{note}
Please make sure to update `HF_TOKEN` with your token

- in the NeMo Run script's [local_executor env_vars](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/scripts/evaluation_with_nemo_run.py#L274) if using local executor
- in the [slurm_executor's env_vars](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/scripts/evaluation_with_nemo_run.py#L237) if using slurm_executor.
:::

Then, the two jobs are configured:

```{literalinclude} ../../../scripts/evaluation_with_nemo_run.py
:language: python
:start-after: "# [snippet-config-start]"
:end-before: "# [snippet-config-end]"
```

Finally, the experiment is started:

```{literalinclude} ../../../scripts/evaluation_with_nemo_run.py
:language: python
:start-after: "# [snippet-experiment-start]"
:end-before: "# [snippet-experiment-end]"
```

## Run Locally

To run evaluations on your local workstation, use the following command:

```bash
cd Evaluator/scripts
python evaluation_with_nemo_run.py \
  --nemo_checkpoint '/workspace/llama3_8b_nemo2/' \
  --eval_task 'gsm8k' \
  --devices 2
```

:::{note}
When running locally with NeMo Run, you will need to manually terminate the deploy process once evaluations are complete.
:::

## Run on Slurm-based Clusters

To run evaluations on Slurm-based clusters, add the `--slurm` flag to your command and specify any custom parameters such as user, host, remote_job_dir, account, mounts, etc. Refer to the `evaluation_with_nemo_run.py` script for further details. Below is an example command:

```bash
cd Evaluator/scripts
python evaluation_with_nemo_run.py \
  --nemo_checkpoint='/workspace/llama3_8b_nemo2' \
  --slurm --nodes 1 \
  --devices 8 \
  --container_image "nvcr.io/nvidia/nemo:25.11" \
  --tensor_parallelism_size 8
```
