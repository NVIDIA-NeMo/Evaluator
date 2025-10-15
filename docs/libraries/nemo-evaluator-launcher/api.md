# Python API

The NeMo Evaluator Launcher provides a Python API for programmatic access to evaluation functionality. This allows you to integrate evaluations into your Python workflows, Jupyter notebooks, and automated pipelines.

## Installation

```bash
pip install nemo-evaluator-launcher

# With optional exporters
pip install nemo-evaluator-launcher[mlflow,wandb,gsheets]
```

## Core Functions

### Running Evaluations

```python
from nemo_evaluator_launcher.api import RunConfig, run_eval

# Run evaluation with configuration
config = RunConfig.from_hydra(
    config_name="local_llama_3_1_8b_instruct",
    config_dir="examples",
    hydra_overrides=[
        "execution.output_dir=my_results"
    ]
)
invocation_id = run_eval(config)

# Returns invocation ID for tracking
print(f"Started evaluation: {invocation_id}")
```

### Listing Available Tasks

```python
from nemo_evaluator_launcher.api import get_tasks_list

# Get all available evaluation tasks
tasks = get_tasks_list()

# Each task contains: [task_name, endpoint_type, harness, container]
for task in tasks[:5]:
    task_name, endpoint_type, harness, container = task
    print(f"Task: {task_name}, Type: {endpoint_type}")
```

### Checking Job Status

```python
from nemo_evaluator_launcher.api import get_status

# Check status of a specific invocation or job
status = get_status(["abc12345"])

# Returns list of status dictionaries with keys: invocation, job_id, status, progress, data
for job_status in status:
    print(f"Job {job_status['job_id']}: {job_status['status']}")
```

## Configuration Management

### Creating Configuration with Hydra

```python
from nemo_evaluator_launcher.api import RunConfig
from omegaconf import OmegaConf

# Load default configuration
config = RunConfig.from_hydra()
print(OmegaConf.to_yaml(config))
```

### Loading Existing Configuration

```python
from nemo_evaluator_launcher.api import RunConfig

# Load a specific configuration file
config = RunConfig.from_hydra(
    config_name="local_llama_3_1_8b_instruct",
    config_dir="examples"
)
```

### Configuration with Overrides

```python
import tempfile
from nemo_evaluator_launcher.api import RunConfig, run_eval

# Create configuration with both Hydra overrides and dictionary overrides
config = RunConfig.from_hydra(
    hydra_overrides=[
        "execution.output_dir=" + tempfile.mkdtemp()
    ],
    dict_overrides={
        "target": {
            "api_endpoint": {
                "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "model_id": "meta/llama-3.1-8b-instruct",
                "api_key_name": "NGC_API_KEY"
            }
        },
        "evaluation": [
            {
                "name": "ifeval",
                "overrides": {
                    "config.params.limit_samples": 10
                }
            }
        ]
    }
)

# Run evaluation
invocation_id = run_eval(config)
```

### Exploring Deployment Options

```python
from nemo_evaluator_launcher.api import RunConfig
from omegaconf import OmegaConf

# Load configuration with different deployment backend
config = RunConfig.from_hydra(
    hydra_overrides=["deployment=vllm"]
)
print(OmegaConf.to_yaml(config))
```

## Jupyter Notebook Integration

```python
# Cell 1: Setup
import tempfile
from omegaconf import OmegaConf
from nemo_evaluator_launcher.api import RunConfig, get_status, get_tasks_list, run_eval

# Cell 2: List available tasks
tasks = get_tasks_list()
print("Available tasks:")
for task in tasks[:10]:  # Show first 10
    print(f"  - {task[0]} ({task[1]})")

# Cell 3: Create and run evaluation
config = RunConfig.from_hydra(
    hydra_overrides=[
        "execution.output_dir=" + tempfile.mkdtemp()
    ],
    dict_overrides={
        "target": {
            "api_endpoint": {
                "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "model_id": "meta/llama-3.1-8b-instruct",
                "api_key_name": "NGC_API_KEY"
            }
        },
        "evaluation": [
            {
                "name": "ifeval",
                "overrides": {
                    "config.params.limit_samples": 10
                }
            }
        ]
    }
)
invocation_id = run_eval(config)
print(f"Started evaluation: {invocation_id}")

# Cell 4: Check status
status_list = get_status([invocation_id])
status = status_list[0]
print(f"Status: {status['status']}")
print(f"Output directory: {status['data']['output_dir']}")
```

## See Also

- [CLI Reference](index.md) - Command-line interface documentation
- [Configuration](configuration/index.md) - Configuration system overview
- [Exporters](exporters/index.md) - Result export options
