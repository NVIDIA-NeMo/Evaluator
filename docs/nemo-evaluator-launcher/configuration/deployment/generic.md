# Generic Deployment

Generic deployment provides flexible configuration for deploying any custom server that isn't covered by built-in deployment configurations.

## Configuration

See [Generic Config](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/generic.yaml) for all available parameters.

Key arguments:
- **`image`**: Docker image to use for deployment (required)
- **`command`**: Command to run the server with template variables (required)
- **`served_model_name`**: Name of the served model (required)
- **`endpoints`**: API endpoint paths (chat, completions, health)
- **`checkpoint_path`**: Path to model checkpoint for mounting (default: null)
- **`extra_args`**: Additional command line arguments
- **`env_vars`**: Environment variables as {name: value} dict

## Best Practices
- Ensure server responds to health check endpoint (ensure that health endpoint is correctly parametrized)
- Test configuration with `--dry_run`
