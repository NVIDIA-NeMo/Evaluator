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
from nemo_evaluator_launcher.api.functional import run_evaluation

# Run evaluation with configuration
result = run_evaluation(
    config_dir="examples",
    config_name="local_llama_3_1_8b_instruct",
    overrides=[
        "execution.output_dir=my_results",
        "target.api_endpoint.model_id=meta/llama-3.1-8b-instruct"
    ]
)

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
from nemo_evaluator_launcher.api.functional import get_job_status

# Check status of a specific job or invocation
status = get_job_status("abc12345")
print(f"Status: {status}")

# Status includes: job_id, status, progress, logs_path, etc.
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
from nemo_evaluator_launcher.api.utils import load_config

# Load a configuration file
config = load_config("examples", "local_llama_3_1_8b_instruct")

# Modify configuration programmatically
config.target.api_endpoint.model_id = "my-custom-model"
config.evaluation.tasks = [
    {"name": "hellaswag"},
    {"name": "winogrande"}
]

# Run with modified config
result = run_evaluation(config=config)
```

### Dynamic Configuration

```python
from nemo_evaluator_launcher.api.types import RunConfig

# Create configuration programmatically
config = RunConfig()
config.execution.output_dir = "my_results"
config.target.api_endpoint.url = "http://localhost:8000/v1/chat/completions"
config.target.api_endpoint.model_id = "my-model"
config.evaluation.tasks = [
    {"name": "arc_challenge"},
    {"name": "hellaswag"}
]

# Run evaluation
result = run_evaluation(config=config)
```

## Jupyter Notebook Integration

```python
# Cell 1: Setup
from nemo_evaluator_launcher.api.functional import (
    get_tasks_list, run_evaluation, get_job_status, export_results
)
import time

# Cell 2: List available tasks
tasks = get_tasks_list()
print("Available tasks:")
for task in tasks[:10]:  # Show first 10
    print(f"  - {task[0]} ({task[1]})")

# Cell 3: Run evaluation
invocation_id = run_evaluation(
    config_dir="examples",
    config_name="local_llama_3_1_8b_instruct",
    overrides=[
        "execution.output_dir=notebook_results",
        "+config.params.limit_samples=10"  # Small test run
    ]
)
print(f"Started evaluation: {invocation_id}")

# Cell 4: Monitor progress
while True:
    status = get_job_status(invocation_id)
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
# Run multiple evaluations in parallel
configs = [
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model1"]),
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model2"]),
    ("local_llama_3_1_8b_instruct", ["target.api_endpoint.model_id=model3"])
]

invocation_ids = []
for config_name, overrides in configs:
    inv_id = run_evaluation(
        config_dir="examples",
        config_name=config_name,
        overrides=overrides
    )
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
from nemo_evaluator_launcher.api.functional import run_evaluation
from nemo_evaluator_launcher.common.exceptions import ConfigurationError

try:
    result = run_evaluation(
        config_dir="examples",
        config_name="invalid_config"
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints and Validation

```python
from typing import List, Dict, Any
from nemo_evaluator_launcher.api.types import RunConfig, ExportConfig

def run_batch_evaluation(
    configs: List[RunConfig],
    export_config: ExportConfig
) -> List[str]:
    """Run multiple evaluations and export results."""
    invocation_ids = []
    
    for config in configs:
        # Validate configuration
        if not config.target.api_endpoint.url:
            raise ValueError("API endpoint URL is required")
        
        inv_id = run_evaluation(config=config)
        invocation_ids.append(inv_id)
    
    # Export results
    export_results(
        invocation_ids=invocation_ids,
        dest=export_config.dest,
        config=export_config.config
    )
    
    return invocation_ids
```

## See Also

- {ref}`launcher-quickstart` - Getting started with the CLI
- {ref}`executors-overview` - Available execution backends  
- {ref}`exporters-overview` - Result export options
- {ref}`configuration-overview` - Full configuration reference
