# Evaluate Multimodal Models with NeMo Framework

This guide shows how to convert, deploy, and evaluate a [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) vision-language model using NeMo Framework and the [OCRBench](https://huggingface.co/datasets/echo840/OCRBench) benchmark.

## Prerequisites

- Access to a SLURM cluster with the NeMo Framework container (`nvcr.io/nvidia/nemo:26.02` or later)
- Sufficient GPU resources (at least one GPU with enough VRAM for the chosen model size)
- A Hugging Face account with access to the model weights

## Installation

The [nvidia-vlmeval](https://pypi.org/project/nvidia-vlmeval/) package provides multimodal evaluation tasks, including OCRBench, and must be installed inside the NeMo Framework container:

```bash
cd /opt/NeMo-FW && uv pip install nvidia-vlmeval
```

:::{note}
The NeMo Framework Docker image does not include `nvidia-vlmeval` by default.
Install it before running any multimodal evaluation.
:::

## Checkpoint Conversion (optional)

To evaluate a checkpoint saved during pretraining or fine-tuning with [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html), provide the path to the saved checkpoint to the deployment command below. Otherwise, Hugging Face checkpoints can be converted to Megatron Bridge using the single shell command:

```bash
python -c 'from megatron.bridge import AutoBridge; AutoBridge.import_ckpt("Qwen/Qwen2.5-VL-3B-Instruct", "/workspace/qwen2_5_vl_3b_instruct/")'
```

## Deployment

Deploy the model using the multimodal deployment script included in the NeMo Framework container. Open a terminal inside the container:

```bash
python /opt/Export-Deploy/scripts/deploy/multimodal/deploy_ray_inframework.py \
  --megatron_checkpoint /workspace/qwen2_5_vl_3b_instruct/ \
  --host 0.0.0.0 \
  --port 8886
```

Wait until the server is ready before proceeding. You can verify readiness with:

```python
from nemo_evaluator.api import check_endpoint

check_endpoint(
    endpoint_url="http://0.0.0.0:8886/v1/chat/completions/",
    endpoint_type="chat",
    model_name="megatron_model",
)
```

:::{tip}
Open a second terminal in the same container session to run the evaluation while the deployment server is running in the first terminal.
:::

## Evaluation

### OCRBench

OCRBench is a comprehensive benchmark designed to assess the OCR (Optical Character Recognition) capabilities of large multimodal models. During evaluation, the harness sends requests to the model's chat completions endpoint. Each request follows the OpenAI multimodal message format and contains:

1. An **image** payload — a base64-encoded image or an image URL included as a `image_url` content part
2. A **text question** — a prompt asking the model to transcribe or answer questions about text visible in the image

For example, a typical request message content looks like:

```json
[
  {
    "type": "image_url",
    "image_url": {"url": "data:image/jpeg;base64,<encoded-image>", "detail": "low"}
  },
  {
    "type": "text",
    "text": "What is written in the image?"
  }
]
```

The model's text response is compared against the ground-truth answer and scored using string matching.


### Run the Evaluation

```{literalinclude} _snippets/vlmeval_ocrbench.py
:language: python
:start-after: "## Run the evaluation"
```

:::{tip}
The example above uses `limit_samples=2` to run on a small subset.
Remove the `limit_samples` parameter to evaluate on the full OCRBench dataset.
:::

Results are saved to the directory specified in `output_dir`.
