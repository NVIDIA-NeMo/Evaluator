(nemo-evaluator-cli)=

# NeMo Evaluator CLI Reference (nemo-evaluator)

This document provides a comprehensive reference for the `nemo-evaluator` command-line interface, which is the primary way to interact with NeMo Evaluator from the terminal.

## Prerequisites

- **Container way**: Use evaluation containers mentioned in {ref}`nemo-evaluator-containers`
- **Package way**:

  ```bash
  pip install nemo-evaluator
  ```

  To run evaluations, you also need to install an evaluation framework package (for example, `nvidia-simple-evals`):
  ```bash
  pip install nvidia-simple-evals
  ```

## Overview

The CLI provides a unified interface for managing evaluations and frameworks. It's built on top of the Python API and provides full feature parity with it.

## Command Structure

```bash
nemo-evaluator [command] [options]
```

## Available Commands

### `ls` - List Available Evaluations

List all available evaluation types and frameworks.

```bash
nemo-evaluator ls
```

**Output Example:**
```
nvidia-simple-evals: 
  * mmlu_pro
...
human_eval: 
  * human_eval
```

### `run_eval` - Run Evaluation

Execute an evaluation with the specified configuration.

```bash
nemo-evaluator run_eval [options]
```

To see the list of options, run:
```bash
nemo-evaluator run_eval --help
```

**Required Options:**
- `--eval_type`: Type of evaluation to run
- `--model_id`: Model identifier
- `--model_url`: API endpoint URL
- `--model_type`: Endpoint type (chat, completions, vlm, embedding)
- `--output_dir`: Output directory for results

**Optional Options:**
- `--api_key_name`: Environment variable name for API key
- `--run_config`: Path to YAML configuration file
- `--overrides`: Comma-separated parameter overrides
- `--dry_run`: Show configuration without running
- `--debug`: Enable debug mode (deprecated, use NEMO_EVALUATOR_LOG_LEVEL)

**Example Usage:**
```bash
# Basic evaluation
nemo-evaluator run_eval \
  --eval_type mmlu_pro \
  --model_id "meta/llama-3.2-3b-instruct" \
  --model_url "https://integrate.api.nvidia.com/v1/chat/completions" \
  --model_type chat \
  --api_key_name MY_API_KEY \
  --output_dir ./results

# With parameter overrides
nemo-evaluator run_eval \
  --eval_type mmlu_pro \
  --model_id "meta/llama-3.2-3b-instruct" \
  --model_url "https://integrate.api.nvidia.com/v1/chat/completions" \
  --model_type chat \
  --api_key_name MY_API_KEY \
  --output_dir ./results \
  --overrides "config.params.limit_samples=100,config.params.temperature=0.1"

# Dry run to see configuration
nemo-evaluator run_eval \
  --eval_type mmlu_pro \
  --model_id "meta/llama-3.2-3b-instruct" \
  --model_url "https://integrate.api.nvidia.com/v1/chat/completions" \
  --model_type chat \
  --api_key_name MY_API_KEY \
  --output_dir ./results \
  --dry_run
```

For execution with run configuration:
```bash
# Using YAML configuration file
nemo-evaluator run_eval \
  --eval_type mmlu_pro \
  --output_dir ./results \
  --run_config ./config/eval_config.yml
```
To check the structure of the run configuration, refer to the [Run Configuration](#run-configuration) section below.

(run-configuration)=

## Run Configuration

Run configurations are stored in YAML files with the following structure:

```yaml
config:
  type: mmlu_pro
  params:
    limit_samples: 10
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.2-3b-instruct
    type: chat
    api_key: MY_API_KEY
    adapter_config:
      interceptors:
        - name: "request_logging"
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./cache"
        - name: "endpoint"
        - name: "response_logging"
          enabled: true
          config:
            output_dir: "./cache/responses"
```

Run configurations can be specified in YAML files and executed with following syntax:

```bash
nemo-evaluator run_eval \
    --run_config config.yml \
    --output_dir `mktemp -d`
```

(parameter-overrides)=

## Parameter Overrides

Parameter overrides use a dot-notation format to specify configuration paths:

```bash
# Basic parameter overrides
--overrides "config.params.limit_samples=100,config.params.temperature=0.1"

# Adapter configuration overrides
--overrides "target.api_endpoint.adapter_config.interceptors.0.config.output_dir=./logs"

# Multiple complex overrides
--overrides "config.params.limit_samples=100,config.params.max_tokens=512,target.api_endpoint.adapter_config.use_caching=true"
```

### Override Format

```
section.subsection.parameter=value
```

**Examples:**
- `config.params.limit_samples=100`
- `target.api_endpoint.adapter_config.use_caching=true`

## Handle Errors

### Debug Mode

Enable debug mode for detailed error information:

```bash
# Set environment variable (recommended)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Or use deprecated debug flag
nemo-evaluator run_eval --debug [options]
```

## Examples

### Complete Evaluation Workflow

```bash
# 1. List available evaluations
nemo-evaluator ls

# 2. Run evaluation
nemo-evaluator run_eval \
  --eval_type mmlu_pro \
  --model_id "meta/llama-3.2-3b-instruct" \
  --model_url "https://integrate.api.nvidia.com/v1/chat/completions" \
  --model_type chat \
  --api_key_name MY_API_KEY \
  --output_dir ./results \
  --overrides "config.params.limit_samples=100"

# 3. Show results
ls -la ./results/
```

### Batch Evaluation Script

```bash
#!/bin/bash

# Batch evaluation script
models=("meta/llama-3.1-8b-instruct" "meta/llama-3.1-70b-instruct")
eval_types=("mmlu_pro" "gsm8k")

for model in "${models[@]}"; do
  for eval_type in "${eval_types[@]}"; do
    echo "Running $eval_type on $model..."
    
    nemo-evaluator run_eval \
      --eval_type "$eval_type" \
      --model_id "$model" \
      --model_url "https://integrate.api.nvidia.com/v1/chat/completions" \
      --model_type chat \
      --api_key_name MY_API_KEY \
      --output_dir "./results/${model//\//_}_${eval_type}" \
      --overrides "config.params.limit_samples=50"
    
    echo "Completed $eval_type on $model"
  done
done

echo "All evaluations completed!"
```

### Framework Development

```bash
# Setup new framework
nemo-evaluator-example my_custom_eval .

# This creates the basic structure:
# core_evals/my_custom_eval/
# ├── framework.yml
# ├── output.py
# └── __init__.py

# Edit framework.yml to configure your evaluation
# Edit output.py to implement result parsing
# Test your framework
nemo-evaluator run_eval \
  --eval_type my_custom_eval \
  --model_id "test-model" \
  --model_url "https://api.example.com/v1/chat/completions" \
  --model_type chat \
  --api_key_name MY_API_KEY \
  --output_dir ./results
```

## Framework Setup Command

### `nemo-evaluator-example` - Setup Framework

Set up NVIDIA framework files in a destination folder.

```bash
nemo-evaluator-example [package_name] [destination]
```

**Arguments:**
- `package_name`: Python package-like name for the framework
- `destination`: Destination folder where to create framework files

**Example Usage:**
```bash
# Setup framework in specific directory
nemo-evaluator-example my_package /path/to/destination

# Setup framework in current directory
nemo-evaluator-example my_package .
```

**What it creates:**
- `core_evals/my_package/framework.yml` - Framework configuration
- `core_evals/my_package/output.py` - Output parsing logic
- `core_evals/my_package/__init__.py` - Package initialization

## Environment Variables

### Logging Configuration

```bash
# Set log level (recommended over --debug flag)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
```
