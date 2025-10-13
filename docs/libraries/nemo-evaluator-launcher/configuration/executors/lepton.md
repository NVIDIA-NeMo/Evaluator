(executor-lepton)=

# Lepton Executor

The Lepton executor deploys endpoints and runs evaluations on Lepton AI. It's designed for fast, isolated, parallel evaluations using hosted or deployed endpoints.

See common concepts and commands in the executors overview.

## Prerequisites

- Lepton AI account and workspace access
- Lepton AI credentials configured
- Appropriate container images and permissions (for deployment flows)

## Install Lepton AI SDK

Install the Lepton AI SDK:

```bash
pip install leptonai
```

## Authenticate with Your Lepton Workspace

Log in to your Lepton AI workspace:

```bash
lep login
```

Follow the prompts to authenticate with your Lepton AI credentials.

## Quick Start

Run a Lepton evaluation using the provided examples:

```bash
# Deploy NIM model and run evaluation
nemo-evaluator-launcher run --config-dir examples --config-name lepton_nim_llama_3_1_8b_instruct

# Deploy vLLM model and run evaluation
nemo-evaluator-launcher run --config-dir examples --config-name lepton_vllm_llama_3_1_8b_instruct

# Use an existing endpoint (no deployment)
nemo-evaluator-launcher run --config-dir examples --config-name lepton_none_llama_3_1_8b_instruct
```

## Parallel Deployment Strategy

- Dedicated endpoints: Each task gets its own endpoint of the same model
- Parallel deployment: All endpoints are created simultaneously (~3x faster)
- Resource isolation: Independent tasks avoid mutual interference
- Storage isolation: Per-invocation subdirectories are created in your configured mount paths
- Simple cleanup: Single command tears down endpoints and storage

```{mermaid}
graph TD
    A["nemo-evaluator-launcher run"] --> B["Load Tasks"]
    B --> D["Endpoints Deployment"]
    D --> E1["Deployment 1: Create Endpoint 1"]
    D --> E2["Deployment 2: Create Endpoint 2"]
    D --> E3["Deployment 3: Create Endpoint 3"]
    E1 --> F["Wait for All Ready"]
    E2 --> F
    E3 --> F
    F --> G["Mount Storage per Task"]
    G --> H["Parallel Tasks Creation as Jobs in Lepton"]
    H --> J1["Task 1: Job 1 Evaluation"]
    H --> J2["Task 2: Job 2 Evaluation"]
    H --> J3["Task 3: Job 3 Evaluation"]
    J1 --> K["Execute in Parallel"]
    J2 --> K
    J3 --> K
    K --> L["Finish"]
```

## Configuration

Lepton executor configurations require:

- **Execution backend**: `execution: lepton/default`
- **Lepton platform settings**: Node groups, resource shapes, secrets, and storage mounts

Refer to the complete working examples in the `examples/` directory:

- `lepton_vllm_llama_3_1_8b_instruct.yaml` - vLLM deployment
- `lepton_nim_llama_3_1_8b_instruct.yaml` - NIM container deployment
- `lepton_none_llama_3_1_8b_instruct.yaml` - Use existing endpoint

These example files include:

- Lepton-specific resource configuration (`lepton_config.resource_shape`, node groups)
- Environment variable references to secrets (HuggingFace tokens, API keys)
- Storage mount configurations for model caching
- Auto-scaling settings for deployments

## Monitoring and Troubleshooting

Check the status of your evaluation runs:

```bash
# Check status of a specific invocation
nemo-evaluator-launcher status <invocation_id>

# Kill running jobs and cleanup endpoints
nemo-evaluator-launcher kill <invocation_id>
```

Common issues:

- Ensure Lepton credentials are valid (`lep login`)
- Verify container images are accessible from your Lepton workspace
- Check that endpoints reach Ready state before jobs start
- Confirm secrets are configured in Lepton UI (Settings → Secrets)
