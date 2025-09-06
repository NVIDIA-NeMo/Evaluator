# Executors Overview

Executors run the evaluation for you by taking the appropriate Docker image (which contains the evaluation harness) and executing the selected benchmark in your environment. They orchestrate containerized runs, manage resources and IO paths, and ensure evaluations are reproducible across machines and clusters. Optionally, an executor can also provision and host the model endpoint as part of the workflow.

- **Core ideas**:
  - Your model is separate from the evaluation container; communication is via an OpenAI‑compatible API
  - Each benchmark runs in an open‑sourced Docker container for reproducibility
  - Execution backends can also optionally manage model deployment

## Supported execution backends
- `Local`: run on your workstation with Docker
- `Slurm`: submit jobs to an HPC cluster managed by Slurm
- `Lepton`: deploy endpoints and run evaluations on Lepton AI
- `Custom`: build your own executor for any environment (e.g., AWS, Google Cloud, Azure, Kubernetes)

Tip: The simplest way to get started is the Local executor. It pulls the evaluation container to your machine and runs it locally. Your model can live anywhere—as long as it exposes an OpenAI‑compatible endpoint, the local run can call it.

Some executors (e.g., Slurm, Lepton) can optionally host the model on‑the‑fly for the duration of the evaluation. In these cases, two containers are used:
- one for the containerized evaluation (benchmark harness), and
- one for serving the model endpoint.

When on‑the‑fly hosting is enabled, the evaluation configuration also includes a deployment section. See the examples in the examples/ folder for Slurm and Lepton.

## Common workflow
1. Choose an executor and example config
2. Point the target to your model endpoint
3. Run and monitor logs
4. Optionally export results to dashboards or files

## Multi-Backend Configuration Examples

Choose the execution backend that matches your infrastructure. All examples use the same model and evaluation tasks, showing how easy it is to switch between backends:

### Local Execution
```yaml
# examples/local_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
```

### Slurm HPC Cluster
```yaml
# examples/slurm_llama_3_1_8b_instruct.yaml
defaults:
  - execution: slurm
  - deployment: vllm
  - _self_

execution:
  output_dir: /shared/results
  partition: gpu
  nodes: 1
  gpus_per_node: 8
  time_limit: "04:00:00"

deployment:
  type: vllm
  model_path: /shared/models/llama-3.1-8b-instruct
  
target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge  
    - name: winogrande
```

### Lepton AI Cloud
```yaml
# examples/lepton_vllm_llama_3_1_8b_instruct.yaml
defaults:
  - execution: lepton
  - deployment: vllm
  - _self_

execution:
  output_dir: ./cloud_results

deployment:
  type: vllm
  model_id: meta-llama/Llama-3.1-8B-Instruct
  
target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
```

## Key Commands
```bash
# List available benchmarks
nv-eval ls tasks

# Run evaluations using the examples above
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
nv-eval run --config-dir examples --config-name slurm_llama_3_1_8b_instruct  
nv-eval run --config-dir examples --config-name lepton_vllm_llama_3_1_8b_instruct

# Status and management
nv-eval status <invocation_id>
nv-eval kill <invocation_id>

# Export results
nv-eval export <invocation_id> --dest local --format json
```

## Test runs
Use a small subset to validate your setup before running full benchmarks:
```bash
nv-eval run --config-dir examples --config-name <your_config> -o +config.params.limit_samples=10
```


## Configuration Overrides

You can override values from the command line (Hydra syntax):
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

## Troubleshooting

View the fully resolved configuration without running:
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

## Create your own executor
- Define how to provision resources and launch the evaluation containers
- Implement log and artifact handling
- Expose configuration for resource sizing, networking, and credentials
- Optionally handle model deployment (bring‑up/tear‑down)

See specific guides:
- [Local](local.md)
- [Slurm](slurm.md)
- [Lepton](lepton.md)
 - Custom: write your own executor to run evaluations anywhere
