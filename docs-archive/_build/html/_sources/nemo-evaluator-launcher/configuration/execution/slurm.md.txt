# Slurm Execution

The Slurm executor submits evaluation jobs to an HPC cluster managed by Slurm.

## Configuration

See the complete configuration structure in the [Slurm Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/slurm/default.yaml).

## Key Settings

- **`hostname`**: Slurm cluster hostname (required)
- **`username`**: Username (defaults to environment variable `${oc.env:USER}`)
- **`account`**: Slurm account for resource allocation (required)
- **`partition`**: Slurm partition/queue (default: batch)
- **`num_nodes`**: Number of nodes (default: 1) - **Note**: Currently only `num_nodes` is supported
- **`ntasks_per_node`**: Number of tasks per node (default: 1)
- **`gres`**: GPU resources (default: gpu:8)
- **`walltime`**: Maximum job runtime in HH:MM:SS format (default: 01:00:00)
- **`subproject`**: Subproject identifier (default: nemo-evaluator-launcher)
- **`output_dir`**: Directory where evaluation results will be saved (use `???` for required values)
- **`env_vars.deployment`**: Environment variables for deployment container
- **`env_vars.evaluation`**: Environment variables for evaluation container
- **`mounts.deployment`**: Mount points for deployment container
- **`mounts.evaluation`**: Mount points for evaluation container
- **`mount_home`**: Whether to mount home directory (default: true)

Tips:
- configure `hostname`, `account`, and `partition` for your specific Slurm cluster
- configure `mounts` to access shared storage on your cluster
- use `env_vars` to pass environment variables to containers
- **`env_vars` should be used for setting explicit values only** (e.g., `VERBOSE: "true"`, `LOG_LEVEL: "debug"`) - do not use for secrets or sensitive data
- adjust `gres` (GPU resources) based on your cluster's GPU configuration
- set appropriate `walltime` for your evaluation jobs

Examples:
- [Slurm vLLM Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct.yaml) - Slurm execution with vLLM deployment and model loaded from local checkpoint
- [Slurm vLLM HF Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct_hf.yaml) - Slurm execution with vLLM and model loaded from Hugging Face
- [Slurm None Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_3_1_8b_instruct.yaml) - Slurm execution with existing endpoint
- [Slurm Nemotron Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_nemotron_super_v1_nemotron_benchmarks.yaml) - Slurm execution with Nemotron model

## Reference

- [Slurm Config Directory](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/slurm)
