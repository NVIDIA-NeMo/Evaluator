# Pre-Evaluation Hooks

Prepare the evaluation environment before evaluations start.

Pre-evaluation hooks execute in the parent process before the adapter server starts. This eliminates race conditions and ensures all setup completes before evaluation requests can be processed. Common use cases include downloading datasets, installing packages, and prefilling caches.

```{note}
Pre-evaluation hooks run **before** the server starts, ensuring atomic setup with no race conditions.
```

## Unified Architecture

Pre-evaluation hooks are unified with interceptors. Any component that implements the `PreEvalHook` interface is automatically recognized and executed at the appropriate time. You can configure them either:

1. **As interceptors** (recommended): Add them to the `interceptors` list
2. **Legacy syntax**: Use `pre_eval_hooks` for backward compatibility (automatically merged into interceptors)

Both approaches work identically - the system automatically detects which interceptors implement the `PreEvalHook` interface and executes them before evaluation.

## Available Pre-Hooks

### Dataset Download

Download datasets from various sources before evaluation begins.

#### YAML Configuration (Recommended)

Configure as an interceptor:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
      - name: "dataset_download_prehook"
        enabled: true
        config:
          target_path: "/workspace/datasets/squad"
          source_type: "huggingface"
          source_path: "squad"
          source_config:
            split: "validation"
            token: "${HF_TOKEN}"
          force_download: false
```

#### Legacy Configuration (Backward Compatible)

```yaml
target:
  api_endpoint:
    adapter_config:
      pre_eval_hooks:
      - name: "dataset_download_prehook"
        enabled: true
        config:
          target_path: "/workspace/datasets/squad"
          source_type: "huggingface"
          source_path: "squad"
          source_config:
            split: "validation"
            token: "${HF_TOKEN}"
          force_download: false
```

Both configurations work identically - `pre_eval_hooks` are automatically merged into `interceptors`.

#### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_path` | Path where dataset should be placed | Required |
| `source_type` | Source backend: `huggingface`, `local`, `s3`, `ngc`, `url`, `wandb` | Required |
| `source_path` | Path/URI to dataset location | Required |
| `source_config` | Source-specific configuration (tokens, credentials, etc.) | `{}` |
| `force_download` | Re-download even if target exists | `false` |

#### Supported Sources

**HuggingFace**
```yaml
source_type: "huggingface"
source_path: "squad"
source_config:
  split: "validation"
  token: "${HF_TOKEN}"
  cache_dir: "/tmp/hf_cache"
```

**Local Copy**
```yaml
source_type: "local"
source_path: "/data/source/dataset"
```

**AWS S3**
```yaml
source_type: "s3"
source_path: "s3://my-bucket/datasets/data.zip"
source_config:
  aws_access_key_id: "${AWS_KEY}"
  aws_secret_access_key: "${AWS_SECRET}"
  region_name: "us-west-2"
```

**NVIDIA NGC**
```yaml
source_type: "ngc"
source_path: "nvidia/dataset:v1"
source_config:
  api_key: "${NGC_API_KEY}"
```

**HTTP/HTTPS URL**
```yaml
source_type: "url"
source_path: "https://example.com/dataset.zip"
source_config:
  extract: true
  remove_archive: true
```

**Weights & Biases**
```yaml
source_type: "wandb"
source_path: "entity/project/artifact:v1"
source_config:
  api_key: "${WANDB_API_KEY}"
  project: "my-project"
```

### Package Installation

Install Python packages using pip before evaluation starts.

#### YAML Configuration (Recommended)

Configure as an interceptor:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
      - name: "package_install_prehook"
        enabled: true
        config:
          packages:
            - "transformers>=4.30"
            - "accelerate"
          check_installed: true
          upgrade: false
          fail_on_error: true
```

#### Legacy Configuration (Backward Compatible)

```yaml
target:
  api_endpoint:
    adapter_config:
      pre_eval_hooks:
      - name: "package_install_prehook"
        enabled: true
        config:
          packages:
            - "transformers>=4.30"
            - "accelerate"
          check_installed: true
          upgrade: false
          fail_on_error: true
```

#### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `packages` | List of package specifications to install | `[]` |
| `requirements_file` | Path to requirements.txt file | `None` |
| `upgrade` | Upgrade packages if already installed | `false` |
| `check_installed` | Skip packages already installed | `true` |
| `fail_on_error` | Fail evaluation if installation fails | `true` |
| `extra_args` | Additional arguments for pip | `[]` |

#### Using Requirements File

```yaml
pre_eval_hooks:
- name: "package_install_prehook"
  config:
    requirements_file: "/workspace/requirements.txt"
```

#### Advanced Configuration

```yaml
pre_eval_hooks:
- name: "package_install_prehook"
  config:
    packages: ["torch==2.0.0"]
    upgrade: true
    extra_args: 
      - "--index-url"
      - "https://download.pytorch.org/whl/cu118"
```

## Multiple Pre-Hooks

Chain multiple pre-hooks in the interceptors list. They execute in the order specified:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        # Pre-hooks run first (in order)
        - name: "package_install_prehook"
          config:
            packages: ["datasets", "boto3"]
            check_installed: true
        
        - name: "dataset_download_prehook"
          config:
            target_path: "/workspace/data/custom"
            source_type: "s3"
            source_path: "s3://my-bucket/dataset.zip"
            source_config:
              aws_access_key_id: "${AWS_KEY}"
              aws_secret_access_key: "${AWS_SECRET}"
        
        # Then regular interceptors
        - name: "caching"
        - name: "endpoint"
```

The system automatically recognizes which interceptors implement the `PreEvalHook` interface and executes them before the server starts.

## Execution Model

Pre-evaluation hooks run in the parent process **before** the adapter server spawns:

1. Evaluation framework calls `AdapterServerProcess.__enter__()`
2. Pre-hooks execute sequentially in parent process
3. All hooks must complete successfully
4. Only then does the adapter server start
5. Evaluation requests begin

This design eliminates race conditions where evaluation requests could arrive before setup completes.

```{important}
If any pre-hook fails, the server never starts and the evaluation terminates immediately.
```

## Creating Custom Pre-Hooks

Create custom pre-hooks by implementing the `PreEvalHook` interface:

```python
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import PreEvalHook, AdapterGlobalContext
from pydantic import BaseModel, Field

@register_for_adapter(
    name="my_custom_prehook",
    description="My custom pre-evaluation setup"
)
class MyCustomPreHook(PreEvalHook):
    class Params(BaseModel):
        """Configuration parameters."""
        setup_path: str = Field(..., description="Path for setup")
    
    def __init__(self, params: Params):
        self.setup_path = params.setup_path
    
    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Execute pre-evaluation setup."""
        # Your setup logic here
        print(f"Setting up at {self.setup_path}")
```

Then use it in configuration:

```yaml
pre_eval_hooks:
  - name: "my_custom_prehook"
    config:
      setup_path: "/workspace/setup"
```

## Best Practices

1. **Use environment variables** for sensitive data like API keys and credentials
2. **Set `force_download: false`** for datasets to avoid unnecessary re-downloads
3. **Enable `check_installed: true`** for packages to skip what's already available
4. **Order matters**: Install packages before downloading datasets if downloads need those packages
5. **Use `fail_on_error: true`** to catch setup issues early before evaluation starts

## Troubleshooting

**Pre-hook fails silently**
- Check logs for error messages
- Ensure all required credentials are set in environment variables
- Verify paths have proper permissions

**Dataset already exists but still downloading**
- Set `force_download: false` in configuration
- Verify `target_path` matches existing location exactly

**Package installation fails**
- Check network connectivity
- Verify package names and versions exist
- Try with `extra_args: ["--verbose"]` for detailed output

**Server never starts**
- Check if pre-hook is failing
- Look for exceptions in logs before "Adapter server started" message
- Pre-hooks must complete successfully for server to start

