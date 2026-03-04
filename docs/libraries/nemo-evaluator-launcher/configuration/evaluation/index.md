(evaluation-configuration)=

# Evaluation Configuration

Evaluation configuration defines which benchmarks to run and their configuration. It is common for all executors and can be reused between them to launch the exact same tasks.

**Important**: Each task has its own default values that you can override. For comprehensive override options, see {ref}`parameter-overrides`.

## Configuration Structure

```yaml
evaluation:
  nemo_evaluator_config:  # Global overrides for all tasks
    config:
      params:
        request_timeout: 3600
  tasks:
    - name: task_name  # Use default benchmark configuration
    - name: another_task
      nemo_evaluator_config:  # Task-specific overrides
        config:
          params:  # Task-specific overrides
            temperature: 0.6
            top_p: 0.95
      env_vars:  # Task-specific environment variables
        HF_TOKEN: $host:HF_TOKEN
```

## Key Components

### Global Overrides

- **`overrides`**: Parameter overrides that apply to all tasks
- **`env_vars`**: Environment variables that apply to all tasks

### Task Configuration

- **`tasks`**: List of evaluation tasks to run
- **`name`**: Name of the benchmark task
- **`overrides`**: Task-specific parameter overrides
- **`env_vars`**: Task-specific environment variables

For a comprehensive list of available tasks, their descriptions, and task-specific parameters, see {ref}`nemo-evaluator-containers`.

## Advanced Task Configuration

### Parameter Overrides

The overrides system is crucial for leveraging the full flexibility of the common endpoint interceptors and task configuration layer. This is where nemo-evaluator intersects with nemo-evaluator-launcher, providing a unified configuration interface.

#### Global Overrides

Settings applied to all tasks listed in the config.

```yaml
evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        temperature: 0.7
```

#### Task-Specific Overrides

Parameters passed to a job for a single task. They take precedence over global evaluation settings.

```yaml
evaluation:
  tasks:
    - name: gpqa_diamond
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.6
            top_p: 0.95
            max_new_tokens: 8192
            parallelism: 32
    - name: mbpp
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.2
            top_p: 0.95
            max_new_tokens: 2048
            extra:
              n_samples: 5
```

(env-vars-configuration)=

### Environment Variables

Environment variables can be declared at multiple levels. Values at more specific levels override broader ones (last wins):

```yaml
# 1. Top-level — applies to ALL jobs (deployment + evaluation)
env_vars:
  HF_TOKEN: $host:HF_TOKEN
  CACHE_DIR: $lit:/cache/huggingface

# 2. Evaluation-level — applies to all evaluation tasks
evaluation:
  env_vars:
    CUSTOM_VAR: $lit:some_value

  # 3. Task-level — applies to a single task only
  tasks:
    - name: task_name1
      env_vars:
        HF_TOKEN: $host:HF_TOKEN_FOR_GPQA_DIAMOND  # overrides top-level
    - name: task_name2   # inherits top-level HF_TOKEN
```

#### Value Prefixes

Every value must use one of three explicit prefixes:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `$host:VAR` | Resolved from the host environment (or `.env` file) at config-load time. Fails if the variable is not set. | `$host:HF_TOKEN` |
| `$lit:value` | Literal value, written as-is. Use for paths, URLs, flags. | `$lit:/cache/huggingface` |
| `$runtime:VAR` | Late-bound — resolved by the execution environment at runtime (e.g., a variable set by SLURM or the deployment container). | `$runtime:SLURM_JOB_ID` |

Bare (unprefixed) values still work for backward compatibility but emit deprecation warnings.

:::{tip}
Use the migration script to automatically add prefixes to existing configs:
```bash
python scripts/migrate_config.py your_config.yaml        # preview
python scripts/migrate_config.py your_config.yaml --write # overwrite
```
:::

#### API Key (`api_key_name`)

The `target.api_endpoint.api_key_name` field specifies which host environment variable holds the API key for the model endpoint. The launcher automatically includes it in the evaluation environment — you do not need to add it to `env_vars` manually:

```yaml
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key_name: NGC_API_KEY   # resolved from $NGC_API_KEY on the host
```

If you need to override the API key for a specific task, declare it explicitly in that task's `env_vars`.

#### Loading from `.env` Files

The launcher can load environment variables from a `.env` file before resolving `$host:` references. This is useful for keeping secrets out of your shell history:

```bash
# Loads $PWD/.env by default (if it exists)
nemo-evaluator-launcher run --config config.yaml

# Or specify a path explicitly
nemo-evaluator-launcher run --config config.yaml --env-file /path/to/.env
```

Variables already set in the shell environment take precedence over `.env` file values.

#### Secrets Handling

Secrets (`$host:` values) are never written into generated scripts (`run.sh`, `run.sub`). Instead, they are stored in a separate `.secrets.env` file alongside the script and sourced at runtime. This prevents accidental exposure in logs, artifacts, and dry-run output.

### Dataset Directory Mounting

Some evaluation tasks require access to local datasets that must be mounted into the evaluation container. Tasks that require dataset mounting will have `NEMO_EVALUATOR_DATASET_DIR` in their `required_env_vars`.

When using such tasks, you must specify:
- **`dataset_dir`**: Path to the dataset on the host machine
- **`dataset_mount_path`** (optional): Path where the dataset should be mounted inside the container (defaults to `/datasets`)

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /path/to/your/techqa/dataset
      # dataset_mount_path: /datasets  # Optional, defaults to /datasets
```

The system will:
1. Mount the host path (`dataset_dir`) to the container path (`dataset_mount_path`)
2. Automatically set the `NEMO_EVALUATOR_DATASET_DIR` environment variable to point to the mounted path inside the container
3. Validate that the required environment variable is properly configured

**Example with custom mount path:**

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /mnt/data/techqa
      dataset_mount_path: /data/techqa  # Custom container path
```

## When to Use

Use evaluation configuration when you want to:

- **Change Default Sampling Parameters**: Adjust temperature, top_p, max_new_tokens for different tasks
- **Change Default Task Values**: Override benchmark-specific default configurations
- **Configure Task-Specific Parameters**: Set custom parameters for individual benchmarks (e.g., n_samples for code generation tasks)
- **Debug and Test**: Launch with limited samples for validation
- **Adjust Endpoint Capabilities**: Configure request timeouts, max retries, and parallel request limits

:::{tip}
For overriding long strings, use YAML multiline syntax with `>-`:

```yaml
config.params.extra.custom_field: >-
  This is a long string that spans multiple lines
  and will be passed as a single value with spaces
  replacing the newlines.
```

This preserves formatting and allows for complex multi-line configurations.
:::

## Reference

- **Parameter Overrides**: {ref}`parameter-overrides` - Complete guide to available parameters and override syntax
- **Adapter Configuration**: For advanced request/response modification (system prompts, payload modification, reasoning handling), see {ref}`nemo-evaluator-interceptors`
- **Task Configuration**: {ref}`lib-core` - Complete nemo-evaluator documentation
- **Available Tasks**: {ref}`nemo-evaluator-containers` - Browse all available evaluation tasks and benchmarks
