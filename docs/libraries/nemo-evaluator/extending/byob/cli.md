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
  --eval_type byob_<normalized_name>.<normalized_name> \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type chat \
  --output_dir ./results \
  --api_key_name API_KEY
```

The `--eval_type` follows the pattern `byob_<normalized_name>.<normalized_name>`,
where `<normalized_name>` is the lowercased, underscore-separated form of the
name you passed to `@benchmark(name=...)` (e.g. `my-benchmark` becomes `my_benchmark`).

:::{tip}
Use `nemo-evaluator-byob --list` to see the exact `eval_type` for each installed
benchmark. This avoids guessing the normalized name.
:::

For logprob-based multiple-choice benchmarks, use a completions endpoint that
supports `echo` and `logprobs`:

```bash
nemo-evaluator run_eval \
  --eval_type byob_<normalized_name>.<normalized_name> \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type completions_logprob \
  --output_dir ./results \
  --api_key_name API_KEY
```

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-containerization` -- Packaging benchmarks as Docker images
