(deployment-nim)=

# NIM Deployment

NIM (NVIDIA Inference Microservices) provides optimized inference microservices with OpenAI-compatible application programming interfaces. NIM deployments automatically handle model optimization, scaling, and resource management on supported platforms.


## Configuration

### Basic Settings

```yaml
deployment:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  port: 8000
```

- **`image`**: NIM container image from [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim) (required)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the NIM server (default: 8000)

### Endpoints

```yaml
endpoints:
  chat: /v1/chat/completions
  completions: /v1/completions
  health: /health
```

## Integration with Lepton

NIM deployment with Lepton executor:

```yaml
defaults:
  - execution: lepton/default
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

Configure environment variables for NIM container operation:

```yaml
deployment:
  lepton_config:
    envs:
      HF_TOKEN:
        value_from:
          secret_name_ref: "HUGGING_FACE_HUB_TOKEN"
```

**Auto-populated Variables:**

The launcher automatically sets these environment variables from your deployment configuration:

- `SERVED_MODEL_NAME`: Set from `deployment.served_model_name`
- `NIM_MODEL_NAME`: Set from `deployment.served_model_name`
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

### Complete Example

```yaml
defaults:
  - execution: lepton/default
  - deployment: nim
  - _self_

execution:
  output_dir: lepton_nim_llama_3_1_8b_results

deployment:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  
  lepton_config:
    endpoint_name: llama-3-1-8b
    resource_shape: gpu.1xh200
    min_replicas: 1
    max_replicas: 3
    
    api_tokens:
      - value_from:
          token_name_ref: "ENDPOINT_API_KEY"
    
    envs:
      HF_TOKEN:
        value_from:
          secret_name_ref: "HUGGING_FACE_HUB_TOKEN"
    
    mounts:
      enabled: true
      cache_path: "/path/to/model/cache"
      mount_path: "/opt/nim/.cache"

evaluation:
  tasks:
    - name: ifeval
```

## Examples

Refer to `packages/nemo-evaluator-launcher/examples/lepton_nim.yaml` for a complete NIM deployment example.

## Reference

- [NIM Documentation](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html)
