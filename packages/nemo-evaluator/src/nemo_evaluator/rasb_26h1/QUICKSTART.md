# Quickstart: Run Opus 4.5 on RASB 26H1

This quickstart runs `azure/anthropic/claude-opus-4-5` against RASB 26H1 through
NeMo Evaluator Launcher.

The commands below support two outer evaluator image modes:

- local development image: `localhost/rasb-26h1:local`
- future NGC image: `nvcr.io/nvidia/eval-factory/rasb-26h1:<version>`

The NGC image is not available yet. Replace `<version>` with the published tag
when it exists.

## 1. Set Paths

Set and enter the Evaluator checkout:

```bash
export IMAGE_DIR=/path/to/Evaluator
cd "$IMAGE_DIR"
```

Set the RASB snapshot path:

```bash
export DATASET_DIR=/path/to/rasb-26h1
```

`IMAGE_DIR` is the local Evaluator checkout containing this integration.
`DATASET_DIR` is the host path to the RASB 26H1 snapshot containing
`benchmark.py`, `26h1/`, and `callables/`.

The example config reads `DATASET_DIR` with OmegaConf interpolation:

```bash
packages/nemo-evaluator-launcher/examples/local_rasb_26h1.yaml
```

Keep those two paths identical for launcher runs that mount the host Docker
socket. RASB launches nested Docker containers through that socket, so inner
Docker volume paths must exist on the host.

This same-path mount is still required when using the future NGC image unless
the runtime is changed away from the host Docker socket model.

## 2. Choose the Evaluator Image

### Option A: Local Development Image

```bash
docker/rasb-26h1-local/build.sh "$DATASET_DIR"
```

This builds:

```bash
localhost/rasb-26h1:local
```

The image contains NeMo Evaluator, the RASB snapshot, Docker CLI support, and
RASB's Python dependencies.

Use this in the launcher config:

```yaml
container: localhost/rasb-26h1:local
```

### Option B: Future NGC Image

Once the RASB evaluator image is published, skip the local build and use the NGC
image in the launcher config:

```yaml
container: nvcr.io/nvidia/eval-factory/rasb-26h1:<version>
```

If the image is private, authenticate first:

```bash
docker login nvcr.io
```

The rest of the launcher config remains the same. Keep `dataset_dir` and
`dataset_mount_path` pointed at the host RASB snapshot:

```yaml
evaluation:
  tasks:
    - name: rasb-26h1.rasb_26h1
      container: nvcr.io/nvidia/eval-factory/rasb-26h1:<version>
      dataset_dir: ${oc.env:DATASET_DIR}
      dataset_mount_path: ${oc.env:DATASET_DIR}
```

## 3. Set the Endpoint and API Key

For NVIDIA-hosted model endpoints:

```bash
export NVAPI_KEY=replace-with-your-key
```

The launcher config uses `https://inference-api.nvidia.com` as the model base
URL and passes the key through `target.api_endpoint.api_key_name`. `NVAPI_KEY`
is the environment variable holding the API key. You may use a different
variable name, but `api_key_name` must match it.

```yaml
target:
  api_endpoint:
    model_id: azure/anthropic/claude-opus-4-5
    url: https://inference-api.nvidia.com
    api_key_name: NVAPI_KEY
```

## 4. Run a One-Environment Smoke Test

The provided local example config intentionally runs only one environment:

```yaml
extra:
  slice: "1"
```

Run:

```bash
.venv/bin/nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/local_rasb_26h1.yaml
```

When using the future NGC image, use the same command with a config whose
`container` value points at the NGC image.

The relevant config values are:

```yaml
target:
  api_endpoint:
    model_id: azure/anthropic/claude-opus-4-5
    url: https://inference-api.nvidia.com
    api_key_name: NVAPI_KEY
```

## 5. Run the Full Benchmark

After the smoke test succeeds, remove the `slice: "1"` line from:

```bash
packages/nemo-evaluator-launcher/examples/local_rasb_26h1.yaml
```

Then rerun the same launcher command.

## 6. Outputs

Launcher outputs go under:

```bash
nel-results/
```

Inside the task artifacts, RASB writes:

```bash
summary.json
rasb_results/environments/<environment_id>/results.json
```

NeMo Evaluator reports these metric groups:

- `pass_rate`
- `counts`

The main score is:

```text
pass_rate.overall_pass_rate
```

## Direct NeMo Evaluator Run

If NeMo Evaluator and RASB dependencies are installed on the host, you can run
without the outer launcher container:

```bash
.venv/bin/nemo-evaluator run_eval \
  --eval_type rasb-26h1.rasb_26h1 \
  --model_id azure/anthropic/claude-opus-4-5 \
  --model_url https://inference-api.nvidia.com \
  --model_type chat \
  --output_dir /tmp/rasb-opus-4-5 \
  --api_key_name NVAPI_KEY \
  --overrides config.params.extra.rasb_root="$DATASET_DIR",config.params.extra.slice=1
```

This still starts RASB's per-environment Docker containers, so host Docker
access is required.
