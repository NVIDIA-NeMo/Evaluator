(deployment-nim)=

# NIM Deployment

NIM (NVIDIA Inference Microservices) provides optimized inference microservices with OpenAI-compatible application programming interfaces. NIM deployments automatically handle model optimization, scaling, and resource management on supported platforms.

## Execution Flow

When using NIM deployments, the launcher follows this execution flow:

1. **Deploy**: Deploy the specified NIM container to the target platform endpoint
2. **Wait**: Wait for the endpoint to be ready and accepting requests
3. **Execute**: Run evaluation tasks as parallel jobs that connect to the deployed NIM
4. **Cleanup**: Clean up the endpoint when done (on failure) or remind you to clean up (on success)

## Prerequisites

Before using NIM deployments, ensure you have:

### Platform-Specific Requirements

- **Lepton**: Install Lepton AI (`pip install leptonai`) and configure credentials (`lep login`)
- **Other platforms**: Refer to platform-specific documentation

### API Access

- **NGC API Key**: Required for NIM container access
- **HuggingFace Token**: Required for model access (if using HF models)

### Resource Access

- Appropriate GPU resources for your chosen NIM container
- Storage access for model caching (recommended)

## Configuration

### Basic Settings

- **`image`**: NIM container image from [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim) (required)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the NIM server (default: 8000)

### Platform Integration

NIM deployments integrate with execution platform configurations:

```yaml
defaults:
  - execution: lepton/default  # or other platform
  - deployment: nim
  - _self_

deployment:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  
  # Platform-specific settings
  lepton_config:
    endpoint_name: nim-llama-3-1-8b-eval
    resource_shape: gpu.1xh200
    # ... additional platform settings
```

### Environment Variables

NIM containers require specific environment variables:

```yaml
deployment:
  lepton_config:
    envs:
      # Required for NIM operation
      OMPI_ALLOW_RUN_AS_ROOT: "1"
      OMPI_ALLOW_RUN_AS_ROOT_CONFIRM: "1"
      
      # API access (use secrets for security)
      NGC_API_KEY:
        value_from:
          secret_name_ref: "NGC_API_KEY"
      HF_TOKEN:
        value_from:
          secret_name_ref: "HUGGING_FACE_HUB_TOKEN_read"
```

**Auto-populated Variables:**

- `SERVED_MODEL_NAME`: Set from `deployment.served_model_name`
- `NIM_MODEL_NAME`: Set from `deployment.served_model_name` for NIM compatibility
- `MODEL_PORT`: Set from `deployment.port` (default: 8000)

### Resource Management

#### Auto-scaling Configuration

```yaml
deployment:
  lepton_config:
    min_replicas: 1
    max_replicas: 3
    
    auto_scaler:
      scale_down:
        no_traffic_timeout: 3600
        scale_from_zero: false
      target_gpu_utilization_percentage: 0
      target_throughput:
        qpm: 2.5
```

#### Storage Mounts

Enable model caching for faster startup:

```yaml
deployment:
  lepton_config:
    mounts:
      enabled: true
      cache_path: "/path/to/model/cache"
      mount_path: "/opt/nim/.cache"
```

### Security Configuration

#### API Tokens

```yaml
deployment:
  lepton_config:
    api_tokens:
      - value: "UNIQUE_ENDPOINT_TOKEN"
```

#### Image Pull Secrets

```yaml
execution:
  lepton_platform:
    tasks:
      image_pull_secrets:
        - "lepton-nvidia-registry-secret"
```

## Complete Example

```yaml
defaults:
  - execution: lepton/default
  - deployment: nim
  - _self_

execution:
  output_dir: lepton_nim_llama_3_1_8b_results
  env_var_names:
    - HF_TOKEN
    - API_KEY

deployment:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  
  lepton_config:
    endpoint_name: nim-llama-3-1-8b-eval
    resource_shape: gpu.1xh200
    min_replicas: 1
    max_replicas: 3
    
    api_tokens:
      - value: "UNIQUE_ENDPOINT_TOKEN"
    
    envs:
      NGC_API_KEY:
        value_from:
          secret_name_ref: "NGC_API_KEY"
      HF_TOKEN:
        value_from:
          secret_name_ref: "HUGGING_FACE_HUB_TOKEN_read"
    
    mounts:
      enabled: true
      cache_path: "/path/to/model/cache"
      mount_path: "/opt/nim/.cache"

evaluation:
  tasks:
    - name: gpqa_diamond
    - name: math_test_500_nemo
```

## Tips and Best Practices

- **Image Selection**: Choose the appropriate NIM image for your model from [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim)
- **Resource Optimization**: NIM automatically selects optimal tensor and data parallelism based on available hardware
- **Model Caching**: Use storage mounts to cache models and reduce startup time
- **Security**: Always use secret references for sensitive data like API keys
- **Scaling**: Configure auto-scaling based on your expected workload patterns

## Examples

- `lepton_nim_llama_3_1_8b_instruct.yaml` - Complete NIM deployment on Lepton platform

## Reference

- [NIM Documentation](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html#)
- The launcher package installation includes configuration files
