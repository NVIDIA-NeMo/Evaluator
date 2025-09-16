# Local Execution

The Local executor runs evaluations on your local machine using Docker containers.

## Configuration

See the complete configuration structure in the [Local Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml).

## Key Settings

- **`output_dir`**: Directory where evaluation results will be saved (required)

Tips:
- ensure Docker is running on your local machine
- local execution is ideal for development and testing

Examples:
- [Local vLLM Example](../../../../packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local execution with vLLM deployment
- [Local None Example](../../../../packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local execution with existing endpoint
- [Local with Metadata](../../../../packages/nemo-evaluator-launcher/examples/local_with_user_provided_metadata.yaml) - Local execution with custom metadata
- [Auto Export Example](../../../../packages/nemo-evaluator-launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml) - Local execution with automatic result export
- [Limit Samples Example](../../../../packages/nemo-evaluator-launcher/examples/local_limit_samples.yaml) - Local execution with limited samples

## Reference

- [Local Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml)
