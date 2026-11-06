# RASB 26H1 NeMo Evaluator Integration

This package makes the RASB 26H1 benchmark snapshot runnable through NeMo
Evaluator as the task `rasb-26h1.rasb_26h1`.

## What Was Added

- `framework.yml` registers the RASB harness with NeMo Evaluator.
- `runner.py` adapts NeMo Evaluator target/model configuration to RASB's
  `benchmark.py` API.
- `output.py` converts RASB's `summary.json` into NeMo Evaluator metrics.
- `docker/rasb-26h1-local/` builds a development image containing NeMo
  Evaluator, Docker CLI support, and a RASB snapshot.
- `packages/nemo-evaluator-launcher/examples/local_rasb_26h1.yaml` is a local
  launcher example. The same task shape will work with an NGC-published image
  once that image is available.

## How Execution Works

RASB uses two Docker layers when run through NeMo Evaluator Launcher:

1. NeMo Evaluator Launcher starts the outer evaluation container
   specified by the task's `container` field. During development this is
   `localhost/rasb-26h1:local`; after publication it should be the NGC image,
   for example `nvcr.io/nvidia/eval-factory/rasb-26h1:<version>`.
2. Inside that container, NeMo Evaluator loads `rasb-26h1.rasb_26h1` and runs
   `python -m nemo_evaluator.rasb_26h1.runner`.
3. The runner loads RASB's `benchmark.py`, discovers environments under
   `26h1/`, selects a callable package, and calls RASB's `run_in_docker()` for
   each selected environment.
4. RASB starts the inner per-environment Docker containers, extracts
   `results.json`, computes an aggregate summary, and writes
   `<output_dir>/summary.json`.
5. NeMo Evaluator calls `output.py`, which reports pass-rate and count metrics.

Because RASB starts nested containers through the host Docker socket, paths used
for nested Docker volume mounts must be host-valid paths. For launcher runs
that mount the host Docker socket, keep `dataset_mount_path` identical to
`dataset_dir`.

## Container Options

There are two intended ways to provide the outer evaluation image.

### Development: Local Image

Build the local image from this repository and a RASB snapshot:

```bash
export DATASET_DIR=/path/to/rasb-26h1
docker/rasb-26h1-local/build.sh "$DATASET_DIR"
```

Use this task container:

```yaml
container: localhost/rasb-26h1:local
```

This is the current working path while the RASB evaluator image is not yet
published.

### Published: NGC Image

Once the RASB evaluator image is published to NGC, use that image directly:

```yaml
container: nvcr.io/nvidia/eval-factory/rasb-26h1:<version>
```

Replace `<version>` with the actual published tag. If the image is private, log
in before running:

```bash
docker login nvcr.io
```

The NGC image should contain the NeMo Evaluator RASB harness and RASB runtime
dependencies. It removes the local image build step, but with the current
host-Docker-socket execution model it does not remove the need to mount the
RASB snapshot from a host path. RASB's inner containers still need access to
environment input directories through paths that exist on the host.

## Dependencies

- Docker daemon access on the host.
- Permission to mount `/var/run/docker.sock` into the outer evaluation
  container.
- A RASB 26H1 snapshot containing `benchmark.py`, `26h1/`, `callables/`,
  `evaluate.py`, `judge.py`, `lm.py`, and `requirements.txt`.
- RASB Python dependencies installed in the evaluation image. The local image
  builder installs `requirements.txt`; the future NGC image should include
  equivalent dependencies.
- API credentials for the model endpoint, normally passed with
  `target.api_endpoint.api_key_name`.
- Enough disk space for RASB base and overlay Docker images.
- NGC registry access if using the future published image.

## Model Endpoint

For Claude/Anthropic models served through NVIDIA, use the NVIDIA inference API
base endpoint:

```yaml
target:
  api_endpoint:
    model_id: azure/anthropic/claude-opus-4-5
    url: https://inference-api.nvidia.com
    api_key_name: NVAPI_KEY
```

`NVAPI_KEY` is the environment variable holding the API key. You may use a
different variable name, but `api_key_name` must match it. The RASB wrapper
maps that value into the runtime variables expected by the RASB callables.

RASB uses provider SDK callables such as `callables/anthropic_lm`. Those
callables expect a base URL and append provider-specific request paths
themselves, for example `/v1/messages` for Anthropic. NeMo Evaluator therefore
runs this harness so the callable receives the configured base endpoint
directly.

## Dataset Location

The input dataset is the RASB snapshot directory. The runner resolves it in this
order:

1. `config.params.extra.rasb_root`
2. `NEMO_EVALUATOR_DATASET_DIR`
3. `NEMO_EVALUATOR_RASB_ROOT`
4. `/rasb-26h1`

For launcher runs, specify the dataset in the task entry:

```yaml
evaluation:
  tasks:
    - name: rasb-26h1.rasb_26h1
      container: localhost/rasb-26h1:local
      dataset_dir: ${oc.env:DATASET_DIR}
      dataset_mount_path: ${oc.env:DATASET_DIR}
```

`DATASET_DIR` is the host path to the RASB 26H1 snapshot containing
`benchmark.py`, `26h1/`, and `callables/`.

When the NGC image is available, the dataset mount stays the same and only the
container changes:

```yaml
evaluation:
  tasks:
    - name: rasb-26h1.rasb_26h1
      container: nvcr.io/nvidia/eval-factory/rasb-26h1:<version>
      dataset_dir: ${oc.env:DATASET_DIR}
      dataset_mount_path: ${oc.env:DATASET_DIR}
```

For direct `nemo-evaluator run_eval` runs, set
`config.params.extra.rasb_root=$DATASET_DIR`.

## Useful Parameters

- `config.params.limit_samples`: run the first N environments.
- `config.params.extra.slice`: run a Python-style slice, for example `1`,
  `10`, `10:20`, or `:20`.
- `config.params.extra.only`: run environments whose ID contains this string.
- `config.params.extra.environments`: override the environments directory,
  relative to the RASB root unless absolute.
- `config.params.extra.callable`: override the callable package. By default,
  Claude/Anthropic model IDs use `callables/anthropic_lm`; other model IDs use
  `callables/openai_lm`.
- `config.params.parallelism` or `config.params.extra.benchmark_workers`:
  number of RASB environments to evaluate concurrently.
- `config.params.request_timeout` or `config.params.extra.benchmark_timeout`:
  per-environment Docker run timeout in seconds.
- `config.params.extra.clean`: remove existing RASB result artifacts before
  running.
- `config.params.extra.redo_on_error`: rerun cached environments that have
  errors.
- `config.params.extra.env_file`: optional explicit `.env` file for RASB. By
  default, the wrapper uses an empty temporary `.env` and passes NeMo Evaluator
  credentials as runtime environment variables.

## Workflow Compared With SWE-bench

For a normal packaged NeMo Evaluator task such as codec's `swebench_test`, the
workflow is mostly:

1. choose a listed task,
2. point NeMo Evaluator at a model endpoint,
3. run one evaluator container,
4. collect artifacts from `/results`.

RASB is different in the runtime layer:

- During development, the task is a local/custom harness rather than a
  pre-listed task in the launcher's packaged IR mapping. With a published NGC
  image, the same task can be referenced through that image's `framework.yml`;
  if it is later added to packaged launcher IRs, it can become a normal listed
  task as well.
- The benchmark data is a RASB snapshot mounted into the evaluator container,
  not just a task name baked into a remote container.
- The outer evaluator container must have Docker installed and must receive the
  host Docker socket.
- Every RASB environment runs in its own inner Docker container.
- The dataset mount path must also be valid on the host because the inner
  containers are launched by the host Docker daemon.

Agentic SWE-bench harnesses also use sandbox/runtime infrastructure, so they are
conceptually closer to RASB than simple QA benchmarks. The practical RASB
difference is that this integration delegates the per-environment lifecycle to
RASB's existing `benchmark.py` and nested Docker implementation.
