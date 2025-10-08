# Local Execution

The Local executor runs evaluations on your local machine using Docker containers.

## Configuration

See the complete configuration structure in the [Local Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml).

## Key Settings

- **`output_dir`**: Directory where evaluation results will be saved (required)
- **`extra_docker_args`**: Additional arguments to pass to the `docker run` command (optional). This flag allows advanced users to customize their setup (see [Advanced configuration](#advanced-configuration)).

Tips:
- ensure Docker is running on your local machine
- local execution is ideal for development and testing

Examples:
- [Local vLLM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local execution with vLLM deployment
- [Local None Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local execution with existing endpoint
- [Auto Export Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml) - Local execution with automatic result export
- [Limit Samples Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/local_limit_samples.yaml) - Local execution with limited samples

## Advanced configuration

You can customize your local executor by specifying `extra_docker_args`.
This parameter allows you to pass any flag to the `docker run` command that is executed by the NeMo Evaluator Launcher.
You can use it to mount additional volumes, set environment variables or customize your netowrk settings.

For example, if you would like your job to use a specific docker network, you can specify:

```yaml
execution:
  extra_docker_args: "--network my-custom-network"
```

Replace `my-custom-network` with `host` to access the host network.

## Reference

- [Local Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml)
