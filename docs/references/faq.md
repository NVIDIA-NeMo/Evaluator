# Frequently Asked Questions

## **What benchmarks and harnesses are supported?**

The docs list hundreds of benchmarks across multiple harnesses, available via curated NGC evaluation containers and the unified Launcher.

Reference: {ref}`eval-benchmarks`

:::{tip}
Discover available tasks with

```bash
nemo-evaluator-launcher ls tasks
```
:::

---

## **How do I set log dir and verbose logging?**

Set these environment variables for logging configuration:

```bash
# Set log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
export LOG_LEVEL=DEBUG
# or (legacy, still supported)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
```

Reference: {ref}`nemo-evaluator-logging`.

---

## **Can I run distributed or on a scheduler?**

Yes. Launcher supports multiple executors. For optimal performance, the SLURM executor is recommended. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility via containerized benchmarks.

See {ref}`executor-slurm` for details.

---

## **Can I point Evaluator at my own endpoint?**

Yes. Provide your OpenAI‑compatible endpoint. The "none" deployment option means no model deployment is performed as part of the evaluation job. Instead, you provide an existing OpenAI-compatible endpoint. The launcher handles running evaluation tasks while connecting to your existing endpoint.

```yaml
target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct    # Model identifier (required)
    url: https://your-endpoint.com/v1/chat/completions  # Endpoint URL (required)
    api_key_name: API_KEY                    # Environment variable name (recommended)

```

Reference: {ref}`deployment-none`.

---

**Can I test my endpoint for OpenAI compatibility?**

Yes. Preview the full resolved configuration without executing using `--dry-run` :

```bash
nemo-evaluator-launcher run \
  --config-dir packages/nemo-evaluator-launcher/examples \
  --config-name local_llama_3_1_8b_instruct --dry-run
```

Reference: {ref}`launcher-cli-dry-run`.

---

## **Can I store and retrieve per-sample results, not just the summary?**

Yes. Capture full request/response artifacts and retrieve them from the run's artifacts folder.

Enable detailed logging with `nemo_evaluator_config`:

```yaml
evaluation:
  # Request + response logging (example at 1k each)
  nemo_evaluator_config:
    target:
      api_endpoint:
        adapter_config:
          use_request_logging: True
          max_saved_requests: 1000
          use_response_logging: True
          max_saved_responses: 1000
```

These enable the **RequestLoggingInterceptor** and **ResponseLoggingInterceptor** so each prompt/response pair is saved alongside the evaluation job.

Retrieve artifacts after the run:

```bash
nemo-evaluator-launcher export <invocation_id> --dest local --output-dir ./artifacts --copy-logs
```

Look under `./artifacts/` for `results.yml`, reports, logs, and saved request/response files.

Reference: {ref}`interceptor-request-logging`.

---

## **Where do I find evaluation results?**

After a run completes, copy artifacts locally:

```bash
nemo-evaluator-launcher info <invocation_id> --copy-artifacts ./artifacts
```

Inside `./artifacts/` you'll see the run config, `results.yaml` (main output file), HTML/JSON reports, logs, and cached request/response files, if caching was used.

Where the output is structured:

```bash
  <output_dir>/
  │   ├── eval_factory_metrics.json
  │   ├── report.html
  │   ├── report.json
  │   ├── results.yml
  │   ├── run_config.yml
  │   └── <Task specific arifacts>/
```

Reference: {ref}`evaluation-output`.

---

## **Can I export a consolidated JSON of scores?**

Yes. JSON is included in the standard output exporter, along with automatic exporters for MLflow, Weights & Biases, and Google Sheets.

```bash
nemo-evaluator-launcher export <invocation_id> --dest local --format json
```

This creates `processed_results.json` (you can also pass multiple invocation IDs to merge).

**Exporter docs:** Local files, W&B, MLflow, GSheets are listed under **Launcher → Exporters** in the docs.

Reference: {ref}`exporters-overview`.

---

## **What's the difference between Launcher and Core?**

* **Launcher (`nemo-evaluator-launcher`)**: Unified CLI with config/exec backends (local/Slurm/Lepton), container orchestration, and exporters. Best for most users. See {ref}`lib-launcher`.
* **Core (`nemo-evaluator`)**: Direct access to the evaluation engine and adapters—useful for custom programmatic pipelines and advanced interceptor use. See {ref}`lib-core`.

---

## **Can I add a new benchmark?**

Yes. Use a **Framework Definition File (FDF)**—a YAML that declares framework metadata, default commands/params, and one or more evaluation tasks. Minimal flow:

1. Create an FDF with `framework`, `defaults`, and `evaluations` sections.
2. Point the launcher/Core at your FDF and run.
3. (Recommended) Package as a container for reproducibility and shareability. See {ref}`extending-evaluator`.

**Skeleton FDF (excerpt):**

```yaml
framework:
  name: my-custom-eval
  pkg_name: my_custom_eval
defaults:
  command: >-
    my-eval-cli --model {{target.api_endpoint.model_id}}
                --task {{config.params.task}}
                --output {{config.output_dir}}
evaluations:
  - name: my_task_1
    defaults:
      config:
        params:
          task: my_task_1
```

See the "Framework Definition File (FDF)" page for the full example and field reference.

Reference: {ref}`framework-definition-file`.

---

## **Why aren't exporters included in the main wheel?**

Exporters target **external systems** (e.g., W&B, MLflow, Google Sheets). Each of those adds heavy/optional dependencies and auth integrations. To keep the base install lightweight and avoid forcing unused deps on every user, exporters ship as **optional extras**:

```bash
# Only what you need
pip install "nemo-evaluator-launcher[wandb]"
pip install "nemo-evaluator-launcher[mlflow]"
pip install "nemo-evaluator-launcher[gsheets]"

# Or everything
pip install "nemo-evaluator-launcher[all]"
```

**Exporter docs:** Local files, W&B, MLflow, GSheets are listed under {ref}`exporters-overview`.

---

## **How is input configuration managed?**

NeMo Evaluator uses **Hydra** for configuration management, allowing flexible composition, inheritance, and command-line overrides.

Each evaluation is defined by a YAML configuration file that includes four primary sections:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: results

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY

evaluation:
  - name: gpqa_diamond
  - name: ifeval
```

This structure defines **where to run**, **how to serve the model**, **which model or endpoint to evaluate**, and **what benchmarks to execute**.

You can start from a provided example config or compose your own using Hydra's `defaults` list to combine deployment, execution, and benchmark modules.

Reference: {ref}`configuration-overview`.

---

## **Can I customize or override configuration values?**

Yes. You can override any field in the YAML file directly from the command line using the `-o` flag:

```bash
# Override output directory
nemo-evaluator-launcher run --config-name your_config \
  -o execution.output_dir=my_results

# Override multiple fields
nemo-evaluator-launcher run --config-name your_config \
  -o target.api_endpoint.url="https://new-endpoint.com/v1/chat/completions" \
  -o target.api_endpoint.model_id=openai/gpt-4o
```

Overrides are merged dynamically at runtime—ideal for testing new endpoints, swapping models, or changing output destinations without editing your base config.

:::{tip}
Always start with a dry run to validate your configuration before launching a full evaluation:

```bash
nemo-evaluator-launcher run --config-name your_config --dry-run
```
:::

Reference: {ref}`configuration-overview`.

---

## **How do I choose the right deployment and execution configuration?**

NeMo Evaluator separates **deployment** (how your model is served) from **execution** (where your evaluations are run). These are configured in the `defaults` section of your YAML file:

```yaml
defaults:
  - execution: local      # Where to run: local, lepton, or slurm
  - deployment: none      # How to serve the model: none, vllm, sglang, nim, trtllm, generic

```

**Deployment Options — How your model is served**

| Option | Description | Best for |
| ----- | ----- | ----- |
| `none` | Uses an existing API endpoint (e.g., NVIDIA API Catalog, OpenAI, Anthropic). No deployment needed. | External APIs or already-deployed services |
| `vllm` | High-performance inference server for LLMs with tensor parallelism and caching. | Fast local/cluster inference, production workloads |
| `sglang` | Lightweight structured generation server optimized for throughput. | Evaluating structured or long-form text generation |
| `nim` | NVIDIA Inference Microservice (NIM) – optimized for enterprise-grade serving with autoscaling and telemetry. | Enterprise, production, and reproducible benchmarks |
| `trtllm` | TensorRT-LLM backend using GPU-optimized kernels. | Lowest latency and highest GPU efficiency |
| `generic` | Use a custom serving stack of your choice. | Custom frameworks or experimental endpoints |

**Execution Platforms — Where evaluations run**

| Platform | Description | Use case |
| ----- | ----- | ----- |
| `local` | Runs Docker-based evaluation locally. | Development, testing, or small-scale benchmarking |
| `lepton` | Runs on NVIDIA Lepton for on-demand GPU execution. | Scalable, production-grade evaluations |
| `slurm` | Uses your HPC cluster's job scheduler. | Research clusters or large batch evaluations |

**Example:**

```yaml
defaults:
  - execution: lepton
  - deployment: vllm
```

This configuration launches the model with **vLLM serving** and runs benchmarks remotely on **Lepton GPUs**.

When in doubt:

* Use `deployment: none` + `execution: local` for your **first run** (quickest setup).
* Use `vllm` or `nim` once you need **scalability and speed**.

Always test first:


```bash
nemo-evaluator-launcher run --config-name your_config --dry-run
```

Reference: {ref}`configuration-overview`.

---
