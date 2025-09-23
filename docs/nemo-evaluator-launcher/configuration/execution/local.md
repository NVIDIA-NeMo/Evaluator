# Local Execution

The local executor runs evaluations on your local machine using Docker containers.

## Configuration

Refer to the [local configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml) for the full schema.

## Key Settings

Key settings include:

- **`output_dir`**: Directory where the launcher saves evaluation results. Required.

### Tips

- Ensure Docker is running on your local machine.
- Use local execution for development and testing.

### Examples

- [Local Llama 3.1 8B Instruct](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml): Local execution of Llama 3.1 8B Instruct.
- [Local with metadata](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_with_user_provided_metadata.yaml): Local execution with custom metadata.
- [Auto export example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml): Local execution with automatic result export.
- [Limit samples example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_limit_samples.yaml): Local execution with limited samples.

## Reference

- [Local configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml)
