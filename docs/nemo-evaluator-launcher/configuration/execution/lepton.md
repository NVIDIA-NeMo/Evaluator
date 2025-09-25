# Lepton Execution

The Lepton executor deploys endpoints and runs evaluations on the Lepton AI platform.

## Configuration

Refer to the default Lepton execution configuration in [`default.yaml`](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/lepton/default.yaml).

## Key Settings

- **`output_dir`**: Directory that stores evaluation results (required)
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

Use the following tips to configure Lepton execution:

- Adjust `evaluation_tasks.resource_shape` based on your evaluation requirements (`cpu.small`, `gpu.small`, `gpu.1xh200`, and so on).
- Set appropriate `evaluation_tasks.timeout` for your evaluation tasks.
- Configure `lepton_platform.deployment.node_group` and `lepton_platform.tasks.node_group` for your Lepton environment.
- Use `env_var_names` to pass environment variables to evaluation containers.

Use the following examples as starting points:

- [Lepton vLLM example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/lepton_vllm_llama_3_1_8b_instruct.yaml): Lepton execution with vLLM deployment.
- [Lepton NIM example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/lepton_nim_llama_3_1_8b_instruct.yaml): Lepton execution with NIM deployment.
- [Lepton `none` example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/lepton_none_llama_3_1_8b_instruct.yaml): Lepton execution with an endpoint already deployed in Lepton.

## Reference

- [Lepton Documentation](https://lepton.ai/docs)
- [Lepton Configuration Directory](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/lepton)
