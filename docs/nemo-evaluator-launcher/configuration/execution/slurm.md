# SLURM Execution

The SLURM executor submits evaluation jobs to a high-performance computing (HPC) cluster that SLURM manages.

## Configuration

Refer to the default configuration file: [Default SLURM configuration (YAML)](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/slurm/default.yaml).

### Key Settings

- `hostname`: SLURM cluster host name (required).
- `username`: User name (defaults to environment variable `${oc.env:USER}`).
- `account`: SLURM account for resource allocation (required).
- `partition`: SLURM partition or queue (default: `batch`).
- `num_nodes`: Number of nodes (default: 1). The executor supports `num_nodes`.
- `ntasks_per_node`: Number of tasks per node (default: 1).
- `gres`: GPU resources (default: `gpu:8`).
- `walltime`: Job run time limit in `HH:MM:SS` format (default: `01:00:00`).
- `subproject`: Identifier for the sub-project name (default: `nemo-evaluator-launcher`).
- `output_dir`: Set the directory to save evaluation results. Use `???` for required values.
- `env_vars.deployment`: Environment variables for the deployment container.
- `env_vars.evaluation`: Environment variables for the evaluation container.
- `mounts.deployment`: Mount points for the deployment container.
- `mounts.evaluation`: Mount points for the evaluation container.
- `mount_home`: Whether to mount the home directory (default: `true`).

### Tips

- Configure `hostname`, `account`, and `partition` for your SLURM cluster.
- Configure `mounts` to access shared storage on your cluster.
- Use `env_vars` to pass environment variables to containers.
- Use `env_vars` for explicit values, such as `VERBOSE: "true"` and `LOG_LEVEL: "debug"`. Do not include secrets or sensitive data.
- Adjust `gres` based on your cluster's GPU configuration.
- Set an appropriate `walltime` for evaluation jobs.

## Examples

- SLURM with vLLM and a local checkpoint: [slurm_llama_3_1_8b_instruct.yaml](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct.yaml).
- SLURM with vLLM and a model from Hugging Face: [slurm_llama_3_1_8b_instruct_hf.yaml](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct_hf.yaml).
- SLURM with an existing endpoint (no deployment): [slurm_no_deployment_llama_3_1_8b_instruct.yaml](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_3_1_8b_instruct.yaml).
- SLURM example for a benchmark model: [benchmark configuration YAML](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_nemotron_super_v1_nemotron_benchmarks.yaml).

## Reference

- SLURM configuration directory: [configs/execution/slurm](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/slurm)
