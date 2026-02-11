# NEL Config Consistency Reference

## How This Document Is Used

This document is piped into `claude -p` by `make update-configs` along with the
full branch git diff. Claude analyzes whether changes require updates to config
templates or examples.

---

## Config Layers (must stay in sync)

### Layer 1: Hydra Config Defaults (source of truth for field structure)

These YAML files define the complete schema — every field name, its type, and its
default value. Hydra merges them at runtime based on the `defaults` list.

- `src/nemo_evaluator_launcher/configs/default.yaml` — root config; sets
  `defaults: [execution: local, deployment: none, _self_]`
- `src/nemo_evaluator_launcher/configs/deployment/{none,vllm,sglang,nim,trtllm,generic}.yaml`
- `src/nemo_evaluator_launcher/configs/execution/{local,slurm/default,lepton/default}.yaml`

### Layer 2: Python Code (source of truth for field usage)

Key files that read config fields at runtime:

- `src/nemo_evaluator_launcher/executors/` — reads `cfg.execution.*`,
  `cfg.deployment.*` to dispatch jobs (local, slurm, lepton executors)
- `src/nemo_evaluator_launcher/api/` — `RunConfig.from_hydra()` loads configs,
  `run_eval()` orchestrates execution
- `src/nemo_evaluator_launcher/cli/` — CLI argument mapping to Hydra overrides
- `src/nemo_evaluator_launcher/exporters/` — reads `cfg.export.*`
  (mlflow, wandb exporters)
- `src/nemo_evaluator_launcher/common/helpers.py` — dispatches on
  `cfg.deployment.type` (`get_endpoint_url`, `get_served_model_name`,
  `get_api_key_name`, `get_health_url`)

### Layer 3: Config Templates (composed by `nel skills build-config`)

These partial YAML snippets are merged by `build_config()` to produce a complete
config file. Located in:

`src/nemo_evaluator_launcher/resources/config_templates/`

- **deployment/** — `none.yaml`, `vllm.yaml`, `sglang.yaml`, `nim.yaml`, `trtllm.yaml`
- **execution/** — `local.yaml`, `slurm.yaml`
- **export/** — `none.yaml`, `mlflow.yaml`, `wandb.yaml`
- **evaluation/**
  - `base/` — `default.yaml`, `standard.yaml`, `code.yaml`, `multilingual.yaml`
  - `chat/` — `default.yaml`, `standard.yaml`, `code.yaml`, `math_reasoning.yaml`,
    `safety.yaml`, `multilingual.yaml`
  - `reasoning/` — `default.yaml`, `standard.yaml`, `code.yaml`,
    `math_reasoning.yaml`, `safety.yaml`, `multilingual.yaml`

### Layer 4: Examples (user-facing documentation)

Complete, runnable configs demonstrating real use cases:

`examples/*.yaml` and `examples/nemotron/*.yaml` (24 files total)

---

## Template Conventions

### Defaults list

Deployment and execution templates include a Hydra `defaults:` key at the top
that selects the matching schema file:

```yaml
defaults:
  - deployment: vllm
```

`build_config()` collects all `defaults` entries and ensures `_self_` appears
exactly once, at the end.

### `???` placeholders

Required fields that the user **must** fill in are set to `???` (Hydra's
MISSING sentinel). Each is followed by an inline comment explaining what to
provide:

```yaml
deployment:
  hf_model_handle: ??? # HuggingFace model handle (e.g., meta-llama/Llama-3.1-8B)
```

### Environment variables in deployment templates

Deployment templates that need runtime secrets set them under
`execution.env_vars.deployment`:

```yaml
execution:
  env_vars:
    deployment:
      HF_TOKEN: $HF_TOKEN # Required for gated HuggingFace models
```

NIM additionally sets `execution.mounts.deployment` for cache directories.

### Evaluation templates

- **`default.yaml`** in each model-type directory sets global evaluation params
  (parallelism, timeout, temperature, max_new_tokens). It does **not** define
  tasks.
- **Benchmark files** (`standard.yaml`, `code.yaml`, etc.) define
  `evaluation.tasks` lists. Each task entry has at minimum a `name` field.
- Task names use the fully-qualified form
  `<harness>.<task_name>` (e.g., `lm-evaluation-harness.ifeval`,
  `nemo_skills.ns_mmlu_pro`, `simple_evals.humanevalplus`,
  `bigcode-evaluation-harness.mbppplus_nemo`).
- Per-task overrides go under `nemo_evaluator_config` nested within the task.
- Per-task environment variables go under `env_vars` within the task.
- `gpqa_diamond` (and its nemo variant) appears only in `math_reasoning.yaml`
  to avoid duplicates when benchmarks are combined. Each file that excludes it
  includes a comment explaining this.

### Export templates

- `export/none.yaml` is comment-only (no YAML keys).
- `export/mlflow.yaml` and `export/wandb.yaml` set
  `execution.auto_export.destinations` and the corresponding `export.<name>`
  config block with `???` placeholders for required fields.

### Comment conventions

- Leading `#` comment at the top of each template describes its purpose and any
  prerequisites (endpoint type, log-probs requirement, etc.).
- Optional/advanced settings are commented out with `#` and a brief explanation.
- Safety templates include commented-out quick-test configurations.

---

## Consistency Rules

1. **Hydra schema ↔ Templates**: If a field is added, removed, or renamed in a
   Hydra config YAML (`configs/`), the corresponding config template
   (`resources/config_templates/`) must be updated to match.

2. **Python code ↔ Hydra schema**: If Python code starts accessing a new config
   field (e.g., `cfg.deployment.new_field`), the Hydra config for that
   deployment/execution type must define it (with a default or `???`), and the
   template should expose it if users need to set it.

3. **Templates ↔ Examples**: If a config template adds a new `???` placeholder,
   at least one example using that deployment/execution type should demonstrate
   the field with a concrete value.

4. **Deprecation**: If Python code removes usage of a config field, consider
   deprecating it from Hydra defaults, templates, and examples.

5. **Task names**: Task names in evaluation templates must be valid
   fully-qualified names (harness + task) that exist in NEL's task registry.
   Run `nel ls tasks` to verify.

6. **Example validation**: Examples must pass dry-run validation (tested by
   `tests/unit_tests/test_example_configs.py`). If a new example is added or an
   existing one changes structure, the test parametrization may need updating.

7. **build_config tests**: If template files are added, removed, or restructured,
   `tests/unit_tests/test_build_config.py` may need corresponding updates
   (e.g., new parametrized cases, updated edge-case assertions).

8. **No duplicate tasks**: When benchmark templates are designed to be combined
   (e.g., standard + math_reasoning), tasks must not appear in multiple files.
   Use comments to document where shared tasks live.

---

## Instructions for Claude

Given the git diff below, analyze ALL changes (Python AND YAML) and report:

1. Which config templates need updating and what specific changes are needed
2. Which examples need updating and what specific changes are needed
3. Whether `test_build_config.py` or `test_example_configs.py` need updating
4. Any new config fields accessed in Python that lack Hydra defaults or
   template entries

If no updates are needed, output exactly `CONFIGS_OK` (one line, no other text).
This token is parsed by CI to determine pass/fail.

Otherwise, format findings as a unified git patch (`diff --git` format) that can
be applied with `git apply`. Add inline YAML comments only when the reason for a
change is non-obvious; omit them for straightforward additions or renames.
