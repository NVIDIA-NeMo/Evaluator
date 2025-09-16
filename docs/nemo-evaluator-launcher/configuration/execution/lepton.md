# Lepton Execution

The Lepton executor deploys endpoints and runs evaluations on Lepton AI platform.

## Configuration

See the complete configuration structure in the [Lepton Config File](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/execution/lepton/default.yaml?ref_type=heads).

## Key Settings

- **`output_dir`**: Directory where evaluation results will be saved (required)
- **`env_var_names`**: Environment variable names to pass to evaluation containers
- **`evaluation_tasks.resource_shape`**: Resource shape for evaluation tasks (default: "cpu.small")
- **`evaluation_tasks.timeout`**: Timeout for individual evaluation tasks in seconds (default: 3600)
- **`evaluation_tasks.use_shared_storage`**: Whether to use shared storage for results (default: true)
- **`lepton_platform.deployment.node_group`**: Node group for endpoint deployments
- **`lepton_platform.deployment.endpoint_readiness_timeout`**: Endpoint readiness timeout in seconds (default: 1200)
- **`lepton_platform.tasks.node_group`**: Node group for evaluation tasks
- **`lepton_platform.tasks.env_vars`**: Default environment variables for all tasks
- **`lepton_platform.tasks.mounts`**: Storage mounts for task execution
- **`lepton_platform.tasks.image_pull_secrets`**: Image pull secrets for task containers

Tips:
- adjust `evaluation_tasks.resource_shape` based on your evaluation requirements (cpu.small, gpu.small, gpu.1xh200, etc.)
- set appropriate `evaluation_tasks.timeout` for your evaluation tasks
- configure `lepton_platform.deployment.node_group` and `lepton_platform.tasks.node_group` for your Lepton environment
- use `env_var_names` to pass environment variables to evaluation containers

Examples:
- [Lepton vLLM Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/lepton_vllm_llama_3_1_8b_instruct.yaml?ref_type=heads) - Lepton execution with vLLM deployment
- [Lepton NIM Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/lepton_nim_llama_3_1_8b_instruct.yaml?ref_type=heads) - Lepton execution with NIM deployment
- [Lepton None Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/lepton_none_llama_3_1_8b_instruct.yaml?ref_type=heads) - Lepton execution with already deployed in Lepton endpoint

## Reference

- [Lepton Documentation](https://lepton.ai/docs)
- [Lepton Config Directory](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/tree/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/execution/lepton?ref_type=heads)
