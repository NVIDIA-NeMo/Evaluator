(config-examples)=

# Configuration Examples

Complete configuration examples for different deployment scenarios and use cases.

## Local Development

```yaml
# configs/local_dev.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./dev_results

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: my-local-model

evaluation:
  overrides:
    config.params.limit_samples: 10    # Small test run
  tasks:
    - name: hellaswag
    - name: arc_challenge
```

## Production Slurm Cluster

```yaml
# configs/production_slurm.yaml
defaults:
  - execution: slurm
  - deployment: vllm
  - _self_

execution:
  output_dir: /shared/production_results
  partition: gpu
  nodes: 1
  gpus_per_node: 8
  time_limit: "12:00:00"
  account: research_project

deployment:
  type: vllm
  image: nvcr.io/nvidia/vllm:24.01
  model_path: /shared/models/llama-3.1-70b
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    VLLM_USE_V2_BLOCK_MANAGER: "1"
    HF_TOKEN: ${oc.env:HF_TOKEN}

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta-llama/Llama-3.1-70B-Instruct

evaluation:
  overrides:
    config.params.temperature: 0.0
    config.params.max_new_tokens: 4096
    config.params.parallelism: 64
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
    - name: gsm8k
```

## Lepton AI Cloud

```yaml
# configs/lepton_cloud.yaml
defaults:
  - execution: lepton
  - deployment: nim
  - _self_

execution:
  output_dir: cloud_results

deployment:
  type: nim
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
  envs:
    NGC_API_KEY: ${oc.env:NGC_API_KEY}

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct

evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
    - name: truthfulqa_mc2
```

## Development Testing with Sample Limits

```yaml
# configs/local_limit_samples.yaml  
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./dev_results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  # Use small sample sizes for quick development testing
  tasks:
    - name: hellaswag
      params:
        limit_samples: 10
    - name: arc_challenge  
      params:
        limit_samples: 10
    - name: winogrande
      params:
        limit_samples: 10
    - name: gsm8k
      params:
        limit_samples: 5  # Math problems take longer
```

## Reasoning vs Non-Reasoning Task Separation

```yaml
# configs/local_aa_reasoning.yaml - For reasoning-heavy tasks
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./reasoning_results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  # Tasks that benefit from chain-of-thought reasoning
  tasks:
    - name: gsm8k
      params:
        temperature: 0.1
        max_tokens: 1024
        enable_cot: true
    - name: math
      params:
        temperature: 0.1  
        max_tokens: 2048
        enable_cot: true
    - name: arc_challenge
      params:
        temperature: 0.1
        max_tokens: 512
        enable_cot: true
```

## See Also

- [Structure & Sections](structure.md) - Core configuration sections
- [Advanced Patterns](advanced.md) - Complex real-world configurations
- [Overrides & Composition](overrides.md) - Configuration customization
