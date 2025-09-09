(launcher-orchestrated-lepton)=

# Lepton AI Deployment via Launcher

Deploy and evaluate models on Lepton AI cloud platform using NeMo Evaluator Launcher orchestration. This approach provides scalable cloud inference with managed infrastructure.

## Overview

Lepton launcher-orchestrated deployment:
- Deploys models on Lepton AI cloud platform
- Provides managed infrastructure and scaling
- Supports various resource shapes and configurations
- Handles deployment lifecycle in the cloud

Based on PR #108's Lepton execution backend implementation.

## Quick Start

```bash
# Deploy and evaluate on Lepton AI
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name lepton_vllm_llama_3_1_8b_instruct \
    -o deployment.model_path=meta-llama/Llama-3.1-8B-Instruct \
    -o execution.resource_shape=gpu.a100.1x
```

## Prerequisites

### Lepton AI Setup

```bash
# Install Lepton AI CLI
pip install leptonai

# Login to Lepton AI
lepton login

# Set API token (if needed)
export LEPTON_API_TOKEN="your-api-token"
```

### Authentication Configuration

```yaml
# config/lepton_auth.yaml
execution:
  backend: lepton
  api_token: ${LEPTON_API_TOKEN}  # From environment
  workspace: "your-workspace"     # Optional workspace
```

## Deployment Types

### vLLM Lepton Deployment

High-performance inference with cloud scaling:

```yaml
# config/lepton_vllm.yaml
deployment:
  type: vllm
  model_path: meta-llama/Llama-3.1-8B-Instruct
  port: 8080
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  served_model_name: llama-3.1-8b

execution:
  backend: lepton
  resource_shape: gpu.a100.1x    # Single A100 GPU
  min_replicas: 1                # Minimum instances
  max_replicas: 3                # Auto-scaling limit
  timeout: 300                   # Deployment timeout

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 500
    - name: gsm8k
      params:
        limit_samples: 200

target:
  api_endpoint:
    url: "https://${LEPTON_DEPLOYMENT_URL}/v1/chat/completions"
    model_id: llama-3.1-8b
```

### NIM Lepton Deployment

Enterprise-grade serving in the cloud:

```yaml
# config/lepton_nim.yaml
deployment:
  type: nim
  model_path: meta-llama/Llama-3.1-8B-Instruct
  container_image: nvcr.io/nim/llama-3.1-8b-instruct
  port: 8000

execution:
  backend: lepton
  resource_shape: gpu.a100.2x    # Dual A100 setup
  min_replicas: 1
  max_replicas: 5
  auto_scale: true               # Enable auto-scaling
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
    - name: humaneval

target:
  api_endpoint:
    url: "https://${LEPTON_DEPLOYMENT_URL}/v1/chat/completions"
    model_id: llama-3.1-8b-instruct
```

### Custom Model Deployment

Deploy your own models to Lepton:

```yaml
# config/lepton_custom.yaml
deployment:
  type: vllm
  model_path: /workspace/my-custom-model  # Local path or HF model
  custom_image: "my-registry/custom-vllm:latest"
  
execution:
  backend: lepton
  resource_shape: gpu.h100.1x    # H100 for custom models
  
  # Custom environment
  env_vars:
    CUSTOM_CONFIG: "production"
    MODEL_CACHE_DIR: "/tmp/model_cache"
    
  # Storage mounting
  mounts:
    - source: "/shared/models"
      target: "/workspace/models"
      type: "volume"

evaluation:
  tasks:
    - name: custom_benchmark
```

## Resource Shapes

### Available GPU Configurations

```yaml
execution:
  backend: lepton
  
  # Single GPU options
  resource_shape: gpu.a100.1x     # 1x A100 40GB
  resource_shape: gpu.h100.1x     # 1x H100 80GB
  resource_shape: gpu.l4.1x       # 1x L4 24GB
  
  # Multi-GPU options
  resource_shape: gpu.a100.2x     # 2x A100 40GB
  resource_shape: gpu.a100.4x     # 4x A100 40GB
  resource_shape: gpu.h100.2x     # 2x H100 80GB
  resource_shape: gpu.h100.4x     # 4x H100 80GB
  resource_shape: gpu.h100.8x     # 8x H100 80GB
```

### CPU-Only Options

```yaml
execution:
  backend: lepton
  
  # CPU configurations
  resource_shape: cpu.small       # 2 vCPU, 4GB RAM
  resource_shape: cpu.medium      # 4 vCPU, 8GB RAM
  resource_shape: cpu.large       # 8 vCPU, 16GB RAM
  resource_shape: cpu.xlarge      # 16 vCPU, 32GB RAM
```

## Configuration Examples

### Auto-Scaling Deployment

```yaml
# config/lepton_autoscale.yaml
deployment:
  type: vllm
  model_path: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 2

execution:
  backend: lepton
  resource_shape: gpu.a100.2x
  
  # Auto-scaling configuration
  min_replicas: 1                # Always keep 1 instance
  max_replicas: 10               # Scale up to 10 instances
  target_concurrency: 50         # Target requests per instance
  scale_down_delay: 300          # Wait 5 min before scaling down
  
  # Health checks
  health_check_path: "/health"
  health_check_timeout: 30

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 5000  # Large evaluation triggers scaling
  
  parallelism: 20  # High parallelism to test scaling
```

### Multi-Model Evaluation

```yaml
# config/lepton_multi_model.yaml
# Use Lepton's model serving for comparison study

models:
  - name: llama-3.1-8b
    deployment:
      type: vllm
      model_path: meta-llama/Llama-3.1-8B-Instruct
      resource_shape: gpu.a100.1x
      
  - name: llama-3.1-70b  
    deployment:
      type: vllm
      model_path: meta-llama/Llama-3.1-70B-Instruct
      resource_shape: gpu.a100.4x
      tensor_parallel_size: 4

execution:
  backend: lepton
  parallel_deployments: true  # Deploy models in parallel

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
  
  # Compare models on same tasks
  compare_models: true
```

### Cost-Optimized Setup

```yaml
# config/lepton_cost_optimized.yaml
deployment:
  type: vllm
  model_path: meta-llama/Llama-3.1-8B-Instruct
  gpu_memory_utilization: 0.95  # Maximize GPU usage

execution:
  backend: lepton
  resource_shape: gpu.l4.1x      # Cost-effective L4 GPU
  
  # Minimize costs
  min_replicas: 0                # Scale to zero when idle
  max_replicas: 2                # Limit maximum scale
  idle_timeout: 300              # Shutdown after 5 min idle
  
  # Spot instances (if available)
  use_spot_instances: true
  spot_max_price: 0.50           # Maximum spot price

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 100  # Smaller evaluation
        
  # Sequential processing to minimize replicas
  parallelism: 1
```

## Advanced Configuration

### Custom Docker Images

```yaml
# config/lepton_custom_image.yaml
deployment:
  type: custom
  custom_image: "my-registry/custom-inference:v1.0"
  
  # Build configuration
  dockerfile: |
    FROM vllm/vllm-openai:latest
    COPY custom_config.json /app/config.json
    RUN pip install custom-package
    
  build_context: "./docker_context"

execution:
  backend: lepton
  resource_shape: gpu.a100.1x
  
  # Custom startup command
  command: ["python", "/app/custom_server.py"]
  
  # Environment variables
  env_vars:
    CUSTOM_CONFIG_PATH: "/app/config.json"
    LOG_LEVEL: "INFO"
```

### Storage and Networking

```yaml
# config/lepton_storage.yaml
execution:
  backend: lepton
  
  # Persistent storage
  storage:
    - name: "model-cache"
      size: "100Gi"
      mount_path: "/cache"
      storage_class: "fast-ssd"
      
    - name: "results"
      size: "50Gi" 
      mount_path: "/results"
      storage_class: "standard"
  
  # Networking
  network:
    ingress:
      - path: "/metrics"
        port: 9090
        public: false  # Internal only
        
    egress:
      - destination: "*.huggingface.co"
        protocol: "https"
```

## Monitoring and Management

### Deployment Status

```bash
# Check deployment status
nemo-evaluator-launcher status <job_id>

# List Lepton deployments
lepton deployment list

# Get deployment details
lepton deployment get <deployment-name>

# View deployment logs
lepton deployment logs <deployment-name>
```

### Resource Monitoring

```bash
# Monitor resource usage
lepton deployment metrics <deployment-name>

# Check scaling events
lepton deployment events <deployment-name>

# View billing information
lepton billing usage --deployment <deployment-name>
```

### Debugging

```bash
# Access deployment shell (if enabled)
lepton deployment exec <deployment-name> -- /bin/bash

# Port forward for local debugging
lepton deployment port-forward <deployment-name> 8080:8080

# Download deployment artifacts
lepton deployment download-artifacts <deployment-name>
```

## Cost Management

### Cost Optimization Strategies

```yaml
# config/lepton_cost_aware.yaml
execution:
  backend: lepton
  
  # Use smaller instances for development
  resource_shape: gpu.l4.1x
  
  # Aggressive auto-scaling
  min_replicas: 0
  scale_to_zero_timeout: 180  # 3 minutes
  
  # Spot instances
  use_spot_instances: true
  spot_interruption_handler: true
  
  # Resource limits
  cpu_limit: "4000m"
  memory_limit: "16Gi"
  gpu_limit: 1

evaluation:
  # Batch evaluations to minimize deployment time
  batch_size: 100
  max_concurrent_batches: 2
```

### Cost Monitoring

```yaml
# config/lepton_with_budget.yaml
execution:
  backend: lepton
  
  # Budget controls
  max_cost_per_hour: 5.0        # USD per hour limit
  daily_budget: 50.0            # USD per day limit
  
  # Alerts
  cost_alerts:
    - threshold: 80  # 80% of budget
      action: "warn"
    - threshold: 95  # 95% of budget  
      action: "pause"
    - threshold: 100 # 100% of budget
      action: "stop"
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/lepton-evaluation.yml
name: Model Evaluation on Lepton
on:
  push:
    branches: [main]
    
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install nemo-evaluator-launcher leptonai
          
      - name: Run evaluation
        env:
          LEPTON_API_TOKEN: ${{ secrets.LEPTON_API_TOKEN }}
        run: |
          nemo-evaluator-launcher run \
            --config-dir configs \
            --config-name lepton_ci_evaluation \
            -o deployment.model_path="${{ github.event.head_commit.message }}"
```

### MLflow Integration

```yaml
# config/lepton_mlflow.yaml
execution:
  backend: lepton
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
    
  # MLflow tracking
  mlflow:
    tracking_uri: "https://mlflow.company.com"
    experiment_name: "lepton-evaluations"
    
  # Auto-export results
  exporters:
    - type: mlflow
      config:
        tracking_uri: "https://mlflow.company.com"
        experiment_name: "lepton-evaluations"
        tags:
          deployment_platform: "lepton"
          resource_shape: "${execution.resource_shape}"
```

## Troubleshooting

### Common Lepton Issues

**Deployment Timeout:**
```bash
# Increase deployment timeout
-o execution.timeout=600

# Check deployment logs
lepton deployment logs <deployment-name>
```

**Resource Unavailable:**
```bash
# Try different resource shape
-o execution.resource_shape=gpu.a100.1x

# Check resource availability
lepton resource list --available
```

**Authentication Issues:**
```bash
# Re-authenticate
lepton login

# Check token
lepton auth status

# Set token explicitly
export LEPTON_API_TOKEN="your-token"
```

**Scaling Issues:**
```bash
# Check scaling events
lepton deployment events <deployment-name>

# Adjust scaling parameters
-o execution.target_concurrency=100
-o execution.scale_down_delay=600
```

### Performance Issues

**High Latency:**
```yaml
# Optimize for latency
deployment:
  type: vllm
  max_batch_size: 1          # Reduce batch size
  max_waiting_time: 0.01     # Reduce wait time

execution:
  resource_shape: gpu.h100.1x  # Use faster GPU
  min_replicas: 2              # Keep instances warm
```

**Low Throughput:**
```yaml
# Optimize for throughput
deployment:
  type: vllm
  max_batch_size: 64         # Larger batches
  tensor_parallel_size: 2    # Use more GPUs

execution:
  resource_shape: gpu.a100.2x
  max_replicas: 5            # Allow more scaling
```

## Best Practices

### Resource Management
- **Right-size instances**: Choose appropriate resource shapes for your models
- **Use auto-scaling**: Configure scaling based on actual usage patterns
- **Monitor costs**: Set up budget alerts and cost monitoring
- **Optimize for spot**: Use spot instances for cost-sensitive workloads

### Deployment Strategy
- **Test locally first**: Validate configurations locally before deploying
- **Use staging environments**: Test deployments in staging before production
- **Implement health checks**: Ensure proper health check endpoints
- **Plan for failures**: Handle spot interruptions and deployment failures

### Security and Compliance
- **Use private deployments**: Keep sensitive models private
- **Implement access controls**: Use proper authentication and authorization
- **Monitor access**: Log and monitor API access patterns
- **Data residency**: Ensure compliance with data location requirements

## Next Steps

- **Compare costs**: Analyze costs vs [Slurm deployment](slurm.md) for your use case
- **Optimize performance**: Fine-tune deployment configurations for your workloads
- **Automate workflows**: Integrate Lepton deployments into your ML pipelines
- **Scale globally**: Explore multi-region deployments for global access
