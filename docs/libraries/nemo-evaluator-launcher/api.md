# Python API

The NeMo Evaluator Launcher provides a comprehensive Python API for programmatic access to evaluation functionality. This allows you to integrate evaluations into your Python workflows, Jupyter notebooks, and automated pipelines.

## Installation

```bash
pip install nemo-evaluator-launcher

# With optional exporters
pip install nemo-evaluator-launcher[mlflow,wandb,gsheets]
```

## Core Functions

### Running Evaluations

```python
from nemo_evaluator_launcher.api.functional import run_eval
from nemo_evaluator_launcher.api.types import RunConfig

# Run evaluation with configuration
config = RunConfig.from_hydra(
    config_dir="examples",
    config_name="local_llama_3_1_8b_instruct",
    hydra_overrides=[
        "execution.output_dir=my_results",
        "target.api_endpoint.model_id=meta/llama-3.1-8b-instruct"
    ]
)
result = run_eval(config)

# Returns invocation ID for tracking
print(f"Started evaluation: {result}")
```

### Listing Available Tasks

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list

# Get all available evaluation tasks
tasks = get_tasks_list()

# Each task contains: [task_name, endpoint_type, harness, container]
for task in tasks:
    task_name, endpoint_type, harness, container = task
    print(f"Task: {task_name}, Type: {endpoint_type}")
```

### Checking Job Status

```python
from nemo_evaluator_launcher.api.functional import get_status

# Check status of a specific job or invocation
status = get_status(["abc12345"])
print(f"Status: {status}")

# Returns list of status dictionaries with keys: invocation, job_id, status, progress, data
```

### Exporting Results

```python
from nemo_evaluator_launcher.api.functional import export_results

# Export to local files (JSON format)
export_results(
    invocation_ids=["abc12345"],
    dest="local",
    config={
        "format": "json",
        "output_dir": "./results",
        "copy_logs": True
    }
)

# Export to MLflow
export_results(
    invocation_ids=["abc12345"],
    dest="mlflow",
    config={
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "my_evaluation"
    }
)

# Export to Weights & Biases
export_results(
    invocation_ids=["abc12345"],
    dest="wandb",
    config={
        "entity": "my_org",
        "project": "evaluations",
        "log_mode": "per_task"
    }
)

# Export to Google Sheets
export_results(
    invocation_ids=["abc12345"],
    dest="gsheets",
    config={
        "spreadsheet_name": "Evaluation Results",
        "service_account_file": "path/to/credentials.json"
    }
)
```

## Configuration Management

### Loading and Modifying Configs

```python
from omegaconf import OmegaConf
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.api.functional import run_eval

# Load a configuration file
config = RunConfig.from_hydra(
    config_dir="examples",
    config_name="local_llama_3_1_8b_instruct"
)

# Modify configuration programmatically
config.target.api_endpoint.model_id = "my-custom-model"
config.evaluation.tasks = [
    {"name": "hellaswag"},
    {"name": "winogrande"}
]

# Run with modified config
result = run_eval(config=config)
```

### Dynamic Configuration

```python
from omegaconf import OmegaConf
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.api.functional import run_eval

# Create configuration programmatically using OmegaConf
config_dict = {
    "execution": {"output_dir": "my_results"},
    "target": {
        "api_endpoint": {
            "url": "http://localhost:8000/v1/chat/completions",
            "model_id": "my-model"
        }
    },
    "evaluation": {
        "tasks": [
            {"name": "arc_challenge"},
            {"name": "hellaswag"}
        ]
    }
}
config = OmegaConf.create(config_dict)

# Run evaluation
result = run_eval(config=config)
```

## Jupyter Notebook Integration

```python
# Cell 1: Setup
from nemo_evaluator_launcher.api.functional import (
    get_tasks_list, run_eval, get_status, export_results
)
from nemo_evaluator_launcher.api.types import RunConfig
import time

# Cell 2: List available tasks
tasks = get_tasks_list()
print("Available tasks:")
for task in tasks[:10]:  # Show first 10
    print(f"  - {task[0]} ({task[1]})")

# Cell 3: Run evaluation
config = RunConfig.from_hydra(
    config_dir="examples",
    config_name="local_llama_3_1_8b_instruct",
    hydra_overrides=[
        "execution.output_dir=notebook_results",
        "+config.params.limit_samples=10"  # Small test run
    ]
)
invocation_id = run_eval(config)
print(f"Started evaluation: {invocation_id}")

# Cell 4: Monitor progress
while True:
    status_list = get_status([invocation_id])
    status = status_list[0]
    print(f"Status: {status['status']}")
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(30)

# Cell 5: Export results
if status['status'] == 'completed':
    export_results(
        invocation_ids=[invocation_id],
        dest="local",
        config={"format": "json", "output_dir": "./results"}
    )
    print("Results exported to ./results/")
```

## Advanced Usage

### Custom Executors

```python
from nemo_evaluator_launcher.executors.registry import register_executor
from nemo_evaluator_launcher.executors.base import BaseExecutor

class MyCustomExecutor(BaseExecutor):
    def run(self, config):
        # Custom execution logic
        pass

# Register custom executor
register_executor("my_executor", MyCustomExecutor)

# Use in configuration
config.execution.type = "my_executor"
```

### Custom Exporters

```python
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.base import BaseExporter

class MyCustomExporter(BaseExporter):
    def export(self, invocation_ids, config):
        # Custom export logic
        pass

# Register custom exporter
register_exporter("my_exporter", MyCustomExporter)

# Use for export
export_results(["abc123"], dest="my_exporter", config={})
```

### Batch Processing

```python
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.api.functional import run_eval, export_results

# Run multiple evaluations in parallel
configs = [
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model1"]),
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model2"]),
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model3"])
]

invocation_ids = []
for config_name, overrides in configs:
    config = RunConfig.from_hydra(
        config_dir="examples",
        config_name=config_name,
        hydra_overrides=overrides
    )
    inv_id = run_eval(config)
    invocation_ids.append(inv_id)

# Export all results together
export_results(
    invocation_ids=invocation_ids,
    dest="mlflow",
    config={"experiment_name": "model_comparison"}
)
```

## Error Handling

```python
from nemo_evaluator_launcher.api.functional import run_eval
from nemo_evaluator_launcher.api.types import RunConfig

try:
    config = RunConfig.from_hydra(
        config_dir="examples",
        config_name="invalid_config"
    )
    result = run_eval(config)
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Execution error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints and Validation

```python
from typing import List, Dict, Any
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.api.functional import run_eval, export_results

def run_batch_evaluation(
    configs: List[RunConfig],
    export_dest: str,
    export_config: Dict[str, Any]
) -> List[str]:
    """Run multiple evaluations and export results."""
    invocation_ids = []
    
    for config in configs:
        # Validate configuration
        if not config.target.api_endpoint.url:
            raise ValueError("API endpoint URL is required")
        
        inv_id = run_eval(config=config)
        invocation_ids.append(inv_id)
    
    # Export results
    export_results(
        invocation_ids=invocation_ids,
        dest=export_dest,
        config=export_config
    )
    
    return invocation_ids
```

## See Also

- {ref}`launcher-quickstart` - Getting started with the CLI
- {ref}`executors-overview` - Available execution backends  
- {ref}`exporters-overview` - Result export options
- {ref}`configuration-overview` - Full configuration reference
