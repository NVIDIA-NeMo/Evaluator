(rasb-26h1)=

# RASB 26H1

RASB 26H1 evaluates models on Real Agent Scaffolds Bench tasks. The benchmark is
available in NeMo Evaluator as:

```text
rasb-26h1.rasb_26h1
```

RASB has one extra runtime requirement compared with most evaluator tasks: each
benchmark environment starts its own Docker container. When running with NeMo
Evaluator Launcher, mount the host Docker socket into the outer evaluation
container and keep the dataset mount path identical to the host path.

## Requirements

- Docker daemon access on the host.
- Permission to mount `/var/run/docker.sock` into the evaluation container.
- A RASB 26H1 snapshot containing `benchmark.py`, `26h1/`, and `callables/`.
- A model endpoint and API key. For NVIDIA-hosted Claude/Anthropic models, use
  `https://inference-api.nvidia.com` as the endpoint base URL.

## Local Image

Build the local RASB evaluator image from this repository and the RASB snapshot:

```bash
export DATASET_DIR=/path/to/rasb-26h1
docker/rasb-26h1-local/build.sh "$DATASET_DIR"
```

Set the model API key:

```bash
export NVAPI_KEY=replace-with-your-key
```

Run the provided one-environment smoke-test config:

```bash
nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/local_rasb_26h1.yaml
```

The example config uses:

```yaml
target:
  api_endpoint:
    model_id: azure/anthropic/claude-opus-4-5
    url: https://inference-api.nvidia.com
    api_key_name: NVAPI_KEY

evaluation:
  tasks:
    - name: rasb-26h1.rasb_26h1
      container: localhost/rasb-26h1:local
      dataset_dir: ${oc.env:DATASET_DIR}
      dataset_mount_path: ${oc.env:DATASET_DIR}
```

The checked-in example limits the run to one RASB environment with
`config.params.extra.slice: "1"`. Remove that override to run the full RASB
26H1 snapshot.

## NGC Image

When a RASB evaluator image is published to NGC, use the same launcher task
shape and replace only the task container:

```yaml
evaluation:
  tasks:
    - name: rasb-26h1.rasb_26h1
      container: nvcr.io/nvidia/eval-factory/rasb-26h1:<version>
      dataset_dir: ${oc.env:DATASET_DIR}
      dataset_mount_path: ${oc.env:DATASET_DIR}
```

Replace `<version>` with the published image tag.
