(byob-containerization)=

# Containerization

Package BYOB benchmarks as Docker images for deployment with NeMo Evaluator
Launcher. Containerization bundles the benchmark code, dataset, and all pip
dependencies into a single reproducible image.

## Building an Image

The simplest way to build a container is with the `--containerize` flag:

```bash
nemo-evaluator-byob my_benchmark.py --containerize
```

Customize the base image and tag:

```bash
nemo-evaluator-byob my_benchmark.py \
  --containerize \
  --base-image python:3.12-slim \
  --tag myrepo/byob_qa:latest
```

The target platform is always appended to the tag (e.g. `myrepo/byob_qa:latest-linux-amd64`). To cross-build for a different platform, use `--platform`:

```bash
nemo-evaluator-byob my_benchmark.py \
  --containerize \
  --platform linux/amd64
```

## Image Layout

The generated Docker image uses the following directory structure:

| Path | Contents |
|------|----------|
| `/opt/byob/code/` | Benchmark Python module(s) |
| `/opt/byob/data/` | Dataset file(s) |
| `/opt/byob_pkg/` | Compiled namespace package |
| `/opt/metadata/` | Framework metadata for harness discovery |

:::{note}
The image entrypoint is pre-configured so that `nemo-evaluator run_eval` can
discover the benchmark without any additional path setup.
:::

## Push to Registry

Build and push in a single command:

```bash
nemo-evaluator-byob my_benchmark.py --push registry.example.com/my-bench:latest
```

The `--push` flag implies `--containerize` automatically. Make sure you are
authenticated with the target registry before pushing (e.g., `docker login`).

## Requirements

Pip dependencies declared in the `@benchmark(requirements=[...])` parameter are
installed inside the container during the build step. This ensures the runtime
environment matches the declared dependencies exactly.

```python
@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt="Q: {question}\nA:",
    target_field="cop",
    requirements=["datasets"],
)
```

:::{warning}
If a requirement is missing from the `requirements` list, the container will
fail at runtime with an `ImportError`. Use `--check-requirements` during
compilation to verify that all declared packages are importable.
:::

## Running with Launcher

Once the image is pushed to a registry, reference it in a NeMo Evaluator
Launcher configuration file:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: byob_my_qa.my-qa
      container: registry.example.com/my-bench:latest # or just local container name
```

The `name` field must match the `eval_type` produced by compilation. Use
`nemo-evaluator-byob --list` to confirm the exact value.

:::{tip}
For multi-task evaluation, add multiple entries under `tasks`, each pointing to
a different containerized benchmark image.
:::

## See Also

- {ref}`byob-cli` -- CLI flags for building and pushing images
- {ref}`byob` -- BYOB overview and quickstart
