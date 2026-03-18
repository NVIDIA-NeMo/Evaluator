(byob-cli)=

# CLI Reference

The `nemo-evaluator-byob` CLI compiles, validates, lists, and containerizes BYOB benchmarks.

## Commands

### Compile a Benchmark

```bash
nemo-evaluator-byob my_benchmark.py
```

Compiles the benchmark definition and auto-installs the resulting package via pip.
After compilation, the benchmark is immediately available to `nemo-evaluator run_eval`.

### Validate Without Installing

```bash
nemo-evaluator-byob my_benchmark.py --dry-run
```

Validates the benchmark definition and shows dataset info without installing the
compiled package. Useful for checking your benchmark before committing changes.

### Compile Without Auto-Install

```bash
nemo-evaluator-byob my_benchmark.py --no-install
```

Compiles the benchmark but does not install it. You must manually add the output
directory to your Python path:

```bash
export PYTHONPATH="~/.nemo-evaluator/byob_packages/byob_<name>:$PYTHONPATH"
```

### List Installed Benchmarks

```bash
nemo-evaluator-byob --list
```

Prints all currently installed BYOB benchmarks and their `eval_type` identifiers.

### Containerize

```bash
nemo-evaluator-byob my_benchmark.py --containerize
```

Builds a Docker image from the compiled benchmark. See
{ref}`byob-containerization` for details on the image layout and deployment.

### Containerize and Push

```bash
nemo-evaluator-byob my_benchmark.py --push registry.example.com/my-bench:latest
```

Builds the Docker image and pushes it to the specified registry in one step.
The `--push` flag implies `--containerize` automatically.

### Additional Flags

| Flag | Description |
|------|-------------|
| `--install-dir DIR` | Custom installation directory |
| `--base-image IMAGE` | Base Docker image (default: `python:3.12-slim`) |
| `--tag TAG` | Docker image tag (default: `byob_<name>:latest`). The target platform is always appended as a suffix (e.g. `byob_qa:latest-linux-amd64`) |
| `--platform PLATFORM` | Target platform for Docker build (e.g. `linux/amd64`). Uses `buildx` when set; plain `docker build` otherwise |
| `--check-requirements` | Verify declared requirements are importable |
| `--version` | Show version |

## Running Evaluations

After compiling a benchmark, run it with the standard `nemo-evaluator` CLI:

```bash
nemo-evaluator run_eval \
  --eval_type byob_<name>.<benchmark_name> \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type chat \
  --output_dir ./results \
  --api_key_name API_KEY \
  --save-predictions
```

The `--eval_type` follows the pattern `byob_<normalized_name>.<original_name>`,
where `<normalized_name>` is the lowercased, underscore-separated form of the
name you passed to `@benchmark(name=...)`.

:::{tip}
Use `nemo-evaluator-byob --list` to see the exact `eval_type` for each installed
benchmark. This avoids guessing the normalized name.
:::

### Runner Flags

These flags control the evaluation runner behavior:

| Flag | Default | Description |
|------|---------|-------------|
| `--save-predictions` | `false` | Save per-sample predictions — including per-question scores and full sample metadata — to `byob_predictions.jsonl` in the output directory (see below) |
| `--timeout-per-sample` | `120` | Timeout in seconds for each model call |
| `--max-retries` | `3` | Maximum number of HTTP retries per model call |
| `--fail-on-skip` | `false` | Raise an error if any sample is skipped (missing field or model error) |
| `--parallelism` | `1` | Number of concurrent evaluation threads. Values greater than 1 evaluate samples in parallel |
| `--n-repeats` | `1` | Number of times to repeat the evaluation. When greater than 1, each prediction includes a `_repeat` key in `metadata` and `sample_id` values are offset per repeat |

### Per-Sample Predictions

When you pass `--save-predictions`, the runner writes a `byob_predictions.jsonl`
file to the output directory alongside the standard `byob_results.json`.
Each line is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | `int` | Zero-based index of the sample in the dataset |
| `prompt` | `str` | Rendered prompt string sent to the model |
| `response` | `str \| null` | Model response text, or `null` if the model call failed |
| `target` | `any` | Ground-truth target value from the dataset |
| `scores` | `dict \| null` | Score dict returned by the scorer function, or `null` if not scored |
| `status` | `str` | One of `"scored"`, `"skipped_missing_field"`, `"skipped_model_error"`, `"skipped_scorer_error"` |
| `error` | `str \| null` | Error message if the sample was skipped |
| `metadata` | `dict` | The full sample row from the dataset. When using `--n-repeats`, also includes a `_repeat` index |

This is useful for profiling results, debugging scorer behavior, or inspecting
individual model responses and their per-question scores.

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-containerization` -- Packaging benchmarks as Docker images
