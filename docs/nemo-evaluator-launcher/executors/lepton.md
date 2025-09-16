# Lepton Executor

The Lepton executor deploys endpoints and runs evaluations on Lepton AI. It’s designed for fast, isolated, parallel evaluations using hosted or deployed endpoints. To learn more, visit [Lepton Page](https://www.nvidia.com/en-us/data-center/dgx-cloud-lepton/).

See common concepts and commands in the executors overview.

## Prerequisites

- Lepton account and credentials configured
- Appropriate container images and permissions (for deployment flows)

## Authenticate with Your Lepton Workspace

Generate an API token and log in:

1. Open your Lepton AI dashboard.
2. Go to Settings → Tokens.
3. Create a new token.
4. Copy and securely save the token. It starts with "nvapi-".
5. Log in from your terminal:

```bash
lep login -c <workspace_id>:<your_token>
```

/// note | Workspace ID
It is actually workspace ID, not workspace name. It is important as it will not work with name.
///

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

Please note some additional steps might be necessary, such as creating registry keys in Lepton.

## Parallel Deployment Strategy

- Dedicated endpoints: Each task gets its own endpoint of the same model
- Parallel deployment: All endpoints are created simultaneously (~3x faster)
- Resource isolation: Independent tasks avoid mutual interference
- Storage isolation: Per-invocation directory `/shared/nemo-evaluator-launcher-workspace/{invocation_id}`
- Simple cleanup: Single command tears down endpoints and storage

Mermaid overview:

```mermaid
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

## Example Configuration

```yaml
evaluation:
  tasks:
    - name: gpqa_diamond
    - name: hellaswag
    - name: winogrande
```

## Configuration Notes

- Supports Hydra overrides for endpoint URLs, API keys, and execution output paths
- Environment variables can be passed to deployment containers via `deployment.envs`

## Troubleshooting

- Ensure Lepton credentials are valid and images are accessible
- Check that endpoints reach Ready state before jobs start
- Use launcher status commands to inspect job progress
