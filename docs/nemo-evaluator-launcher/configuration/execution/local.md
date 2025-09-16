# Local Execution

The Local executor runs evaluations on your local machine using Docker containers.

## Configuration

See the complete configuration structure in the [Local Config File](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml?ref_type=heads).

## Key Settings

- **`output_dir`**: Directory where evaluation results will be saved (required)

Tips:
- ensure Docker is running on your local machine
- local execution is ideal for development and testing

Examples:
- [Local vLLM Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_llama_3_1_8b_instruct.yaml?ref_type=heads) - Local execution with vLLM deployment
- [Local None Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_llama_3_1_8b_instruct.yaml?ref_type=heads) - Local execution with existing endpoint
- [Local with Metadata](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_with_user_provided_metadata.yaml?ref_type=heads) - Local execution with custom metadata
- [Auto Export Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml?ref_type=heads) - Local execution with automatic result export
- [Limit Samples Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_limit_samples.yaml?ref_type=heads) - Local execution with limited samples

## Reference

- [Local Config File](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/execution/local.yaml?ref_type=heads)
