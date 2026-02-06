# Implementation Plan: verify_config.py

## Overview
Create a Pydantic-based config validation script for NeMo Evaluator Launcher (NEL) that:
1. Resolves configs using Hydra (same as the launcher)
2. Validates against comprehensive Pydantic models
3. Provides descriptive error messages
4. Passes validation for all example configs

## File to Create
`.claude/skills/nel-config-generator/scripts/verify_config.py`

## Pydantic Model Architecture

### Design Principles
- Use discriminated unions for `execution` and `deployment` based on `type` field
- Use `extra="forbid"` on well-defined structures to catch unexpected fields
- Use `extra="allow"` on flexible structures (like task-specific configs)
- Document all fields with `Field(description=...)` for descriptive errors

### Model Hierarchy

```
NELConfig (root)
├── defaults: list[str | dict] (optional, Hydra-specific)
├── execution: ExecutionConfig (discriminated union by type)
│   ├── LocalExecutionConfig (type="local")
│   ├── SlurmExecutionConfig (type="slurm")
│   └── LeptonExecutionConfig (type="lepton")
├── deployment: DeploymentConfig (discriminated union by type)
│   ├── NoneDeploymentConfig (type="none")
│   ├── VLLMDeploymentConfig (type="vllm")
│   ├── SGLangDeploymentConfig (type="sglang")
│   ├── NIMDeploymentConfig (type="nim")
│   ├── TRTLLMDeploymentConfig (type="trtllm")
│   └── GenericDeploymentConfig (type="generic")
├── target: TargetConfig
│   └── api_endpoint: ApiEndpointConfig
├── evaluation: EvaluationSectionConfig
│   ├── nemo_evaluator_config: NemoEvaluatorConfig (optional)
│   └── tasks: list[TaskConfig]
└── export: ExportConfig (optional)
    ├── mlflow: MLflowExportConfig
    └── wandb: WandbExportConfig
```

### Execution Models

```python
class LocalExecutionConfig(BaseModel):
    """Local Docker execution configuration."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["local"] = Field(description="Execution type: local")
    output_dir: str = Field(description="Directory for output files")
    extra_docker_args: str = Field(default="", description="Additional Docker arguments")
    mode: str = Field(default="sequential", description="Execution mode")
    auto_export: Optional[AutoExportConfig] = Field(default=None, description="Auto-export configuration")
    env_vars: Optional[dict] = Field(default=None, description="Environment variables")

class SlurmExecutionConfig(BaseModel):
    """SLURM cluster execution configuration."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["slurm"] = Field(description="Execution type: slurm")
    hostname: str = Field(description="SLURM headnode hostname (required)")
    username: str = Field(description="Cluster username")
    account: str = Field(description="SLURM account allocation (required)")
    output_dir: str = Field(description="Absolute path on compute nodes (required)")
    partition: str = Field(default="batch", description="SLURM partition")
    num_nodes: int = Field(default=1, description="Number of nodes")
    # ... more fields
    deployment: Optional[SlurmDeploymentSettings] = Field(default=None)
    env_vars: Optional[SlurmEnvVars] = Field(default=None)
    mounts: Optional[SlurmMounts] = Field(default=None)
    proxy: Optional[ProxyConfig] = Field(default=None)

class LeptonExecutionConfig(BaseModel):
    """Lepton cloud execution configuration."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["lepton"] = Field(description="Execution type: lepton")
    output_dir: str = Field(description="Output directory")
    env_var_names: list[str] = Field(default_factory=list)
    evaluation_tasks: Optional[LeptonEvaluationTasks] = Field(default=None)
    lepton_platform: Optional[LeptonPlatform] = Field(default=None)
```

### Deployment Models

```python
class NoneDeploymentConfig(BaseModel):
    """No deployment - use external endpoint."""
    type: Literal["none"] = Field(description="Deployment type: none (external API)")

class VLLMDeploymentConfig(BaseModel):
    """vLLM deployment configuration."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["vllm"] = Field(description="Deployment type: vllm")
    image: str = Field(default="vllm/vllm-openai:latest", description="Docker image")
    checkpoint_path: Optional[str] = Field(default=None, description="Model checkpoint path")
    hf_model_handle: Optional[str] = Field(default=None, description="HuggingFace model ID")
    served_model_name: str = Field(description="Model name for serving")
    port: int = Field(default=8000, description="Server port")
    tensor_parallel_size: int = Field(default=8, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size")
    data_parallel_size: int = Field(default=1, description="Data parallel size")
    data_parallel_size_local: Optional[int] = Field(default=None, description="Local DP size")
    gpu_memory_utilization: float = Field(default=0.95, description="GPU memory utilization")
    extra_args: str = Field(default="", description="Extra vLLM arguments")
    env_vars: dict = Field(default_factory=dict, description="Environment variables")
    endpoints: Optional[EndpointsConfig] = Field(default=None, description="API endpoints")
    command: Optional[str] = Field(default=None, description="Custom command")
```

### Target & Evaluation Models

```python
class ApiEndpointConfig(BaseModel):
    """API endpoint configuration."""
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = Field(default=None, description="Endpoint URL")
    model_id: Optional[str] = Field(default=None, description="Model identifier")
    api_key_name: Optional[str] = Field(default=None, description="Env var name for API key")
    api_key: Optional[str] = Field(default=None, description="[DEPRECATED] Use api_key_name")
    type: Optional[str] = Field(default=None, description="Endpoint type (chat/completions)")
    adapter_config: Optional[AdapterConfig] = Field(default=None, description="Adapter configuration")

class TaskConfig(BaseModel):
    """Individual evaluation task configuration."""
    model_config = ConfigDict(extra="allow")  # Allow task-specific fields

    name: str = Field(description="Task name (e.g., ifeval, gsm8k, mmlu)")
    nemo_evaluator_config: Optional[NemoEvaluatorConfig] = Field(default=None)
    env_vars: Optional[dict[str, str]] = Field(default=None, description="Task-specific env vars")
```

## Main Script Logic

```python
#!/usr/bin/env python3
"""
NEL Config Validator - Validates NeMo Evaluator Launcher configurations.

Usage:
    python verify_config.py <config_path> [--mock-required]

Options:
    --mock-required  Auto-fill required (???) fields with mock values
"""

import sys
from pathlib import Path
from typing import Any

from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

# Import our models
from models import NELConfig, get_mock_overrides

def resolve_config(config_path: str, mock_required: bool = False) -> DictConfig:
    """Resolve config using Hydra, same as NEL launcher."""
    GlobalHydra.instance().clear()

    overrides = []
    if mock_required:
        overrides = get_mock_overrides(config_path)

    # Reuse NEL's config resolution logic
    from nemo_evaluator_launcher.api.types import RunConfig
    cfg = RunConfig.from_hydra(config=config_path, hydra_overrides=overrides)

    GlobalHydra.instance().clear()
    return cfg

def validate_config(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Validate resolved config against Pydantic models."""
    errors = []

    # Convert to dict for Pydantic
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    try:
        NELConfig(**config_dict)
        return True, []
    except ValidationError as e:
        for err in e.errors():
            path = ".".join(str(p) for p in err["loc"])
            msg = err["msg"]
            error_type = err["type"]

            if error_type == "extra_forbidden":
                errors.append(f"Unexpected field '{path}': This field is not recognized. Check for typos.")
            elif error_type == "missing":
                errors.append(f"Missing required field '{path}': {msg}")
            else:
                errors.append(f"Invalid value at '{path}': {msg}")

        return False, errors

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_config.py <config_path> [--mock-required]")
        sys.exit(1)

    config_path = sys.argv[1]
    mock_required = "--mock-required" in sys.argv

    print(f"Validating: {config_path}")

    try:
        cfg = resolve_config(config_path, mock_required)
        valid, errors = validate_config(cfg)

        if valid:
            print("✓ Configuration is valid!")
            sys.exit(0)
        else:
            print("✗ Configuration has errors:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to process config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Mock Overrides for Required Fields

```python
def get_mock_overrides(config_path: str) -> list[str]:
    """Generate mock overrides for required (???) fields based on config type."""
    overrides = [
        "execution.output_dir=/tmp/nel-test-results",
    ]

    config_name = Path(config_path).stem

    # SLURM configs need additional mocks
    if config_name.startswith("slurm_"):
        overrides.extend([
            "execution.hostname=test-slurm-host.example.com",
            "execution.account=test-account",
            "++deployment.checkpoint_path=null",
        ])

    # Auto-export configs need MLflow URI
    if "auto_export" in config_name:
        overrides.extend([
            "export.mlflow.tracking_uri=http://test-mlflow:5000",
        ])

    # Safety configs need judge URL
    if "safety" in config_name:
        overrides.append(
            "++evaluation.tasks.1.nemo_evaluator_config.config.params.extra.judge.url=https://test-judge.example.com/v1"
        )

    return overrides
```

## Error Message Formatting

Key principle: Make errors actionable by including:
1. The exact path to the problematic field
2. What's wrong (missing, unexpected, wrong type)
3. Suggestion for how to fix it

Example error outputs:
```
✗ Configuration has errors:
  - Unexpected field 'execution.hostnam': This field is not recognized. Did you mean 'hostname'?
  - Missing required field 'execution.account': SLURM account allocation is required
  - Invalid value at 'deployment.tensor_parallel_size': Input should be a valid integer, got 'eight'
```

## Verification Plan

1. **Unit test the script** against all example configs:
```bash
cd packages/nemo-evaluator-launcher/examples
for config in *.yaml; do
    uv run python ../../../.claude/skills/nel-config-generator/scripts/verify_config.py "$config" --mock-required
done
```

2. **Test error detection** with intentionally broken configs:
   - Add unknown field → should report "Unexpected field"
   - Remove required field → should report "Missing required field"
   - Wrong type → should report "Invalid value"

3. **Integration with existing tests**: The script should pass on all configs that pass `test_example_configs.py`

## Files to Create

1. `.claude/skills/nel-config-generator/scripts/verify_config.py` - Main script
2. `.claude/skills/nel-config-generator/scripts/models.py` - Pydantic models (or inline in verify_config.py)

## Dependencies
- pydantic (already in project)
- hydra-core (already in project)
- omegaconf (already in project)
- nemo_evaluator_launcher (for RunConfig.from_hydra)

## Implementation Notes

1. **Discriminated Unions**: Use Pydantic's `Discriminator` to route to correct model based on `type`:
```python
ExecutionConfig = Annotated[
    LocalExecutionConfig | SlurmExecutionConfig | LeptonExecutionConfig,
    Field(discriminator="type")
]
```

2. **Handling OmegaConf interpolations**: The `${oc.env:USER}` and similar interpolations are resolved by Hydra before validation, so Pydantic sees the final values.

3. **Strict validation everywhere** (user confirmed): Use `extra="forbid"` on ALL models to catch typos and unexpected fields. This includes:
   - Top-level NELConfig
   - All execution configs (local, slurm, lepton)
   - All deployment configs (none, vllm, sglang, nim, trtllm, generic)
   - Target and evaluation configs
   - Adapter configs and interceptor configs

4. **Exception for deeply nested dynamic configs**: Only use `extra="allow"` for:
   - `InterceptorConfig.config` dict (varies by interceptor type)
   - `params.extra` dict (framework-specific parameters)
   - These are explicitly designed to be extensible

5. **Hydra metadata fields to whitelist**: Some fields are added by Hydra and should be allowed:
   - `user_config_path` (added by RunConfig.from_hydra)
   - Fields starting with `_` (Hydra internal)
