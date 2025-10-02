# NeMo Evaluator

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/downloads/)
[![Tests](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![nemo-evaluator PyPI version](https://img.shields.io/pypi/v/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator-launcher PyPI version](https://img.shields.io/pypi/v/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![nemo-evaluator-launcher PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)](#)

NeMo Evaluator is an open-source platform for robust, reproducible, and scalable evaluation of Large Language Models. It enables you to run hundreds of benchmarks across popular evaluation harnesses against any OpenAI-compatible model API. Evaluations execute in open-source Docker containers for auditable and trustworthy results. The platform's containerized architecture allows for the rapid integration of public benchmarks and private datasets.

[Tutorial](./docs/nemo-evaluator-launcher/tutorial.md) | [Supported Benchmarks](#supported-benchmarks-and-evaluation-harnesses) | [Configuration Examples](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples) | [Contribution Guide](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md)

## ✨ Key Pillars

NeMo Evaluator is built on four core principles to provide a reliable and versatile evaluation experience:

- **Reproducibility by Default**: All configurations, random seeds, and software provenance are captured automatically for auditable and repeatable evaluations.
- **Scale Anywhere**: Run evaluations from a local machine to a Slurm cluster or cloud-native backends like Lepton AI without changing your workflow.
- **State-of-the-Art Benchmarking**: Access a comprehensive suite of over 100 benchmarks from 18 popular open-source evaluation harnesses. See the full list of [Supported benchmarks and evaluation harnesses](#supported-benchmarks-and-evaluation-harnesses).
- **Extensible and Customizable**: Integrate new evaluation harnesses, add custom benchmarks with proprietary data, and define custom result exporters for existing MLOps tooling.

## How It Works: Launcher and Core Engine

The platform consists of two main components:

- **`nemo-evaluator` ([The Evaluation Core Engine](./docs/nemo-evaluator/index.md))**: A Python library that manages the interaction between an evaluation harness and the model being tested.
- **`nemo-evaluator-launcher` ([The CLI and Orchestration](./docs/nemo-evaluator-launcher/index.md))**: The primary user interface and orchestration layer. It handles configuration, selects the execution environment, and launches the appropriate container to run the evaluation.

Most users typically interact with `nemo-evaluator-launcher`, which serves as a universal gateway to different benchmarks and harnesses. However, it is also possible to interact directly with `nemo-evaluator` by following this [guide](./docs/nemo-evaluator/workflows/using-containers.md).

```mermaid
graph TD
    A[User] --> B{NeMo Evaluator Launcher};
    B -- " " --> C{Local};
    B -- " " --> D{Slurm};
    B -- " " --> E{Lepton};
    subgraph Execution Environment
        C -- "Launches Container" --> F[Evaluation Container];        
        D -- "Launches Container" --> F;
        E -- "Launches Container" --> F;
    end
    subgraph F[Evaluation Container]
        G[Nemo Evaluator] -- " Runs " --> H[Evaluation Harness]
    end
    H -- "Sends Requests To" --> I[🤖 Model Endpoint];
    I -- "Returns Responses" --> H;
```


## 🚀 Quickstart

Get your first evaluation result in minutes. This guide uses your local machine to run a small benchmark against an OpenAI API-compatible endpoint.

#### 1. Install the Launcher

The launcher is the only package required to get started.

```bash
pip install nemo-evaluator-launcher
```

### 2. Set Up Your Model Endpoint

NeMo Evaluator works with any model that exposes an OpenAI-compatible endpoint. For this quickstart, we will use the OpenAI API.

**What is an OpenAI-compatible endpoint?** A server that exposes /v1/chat/completions and /v1/completions endpoints, matching the OpenAI API specification.

**Options for model endpoints:**

- **Hosted endpoints** (fastest): Use ready-to-use hosted models from providers like [build.nvidia.com](https://build.nvidia.com) that expose OpenAI-compatible APIs with no hosting required.
- **Self-hosted options**: Host your own models using tools like NVIDIA NIM, vLLM, or TensorRT-LLM for full control over your evaluation environment.

For detailed setup instructions including self-hosted configurations, see the [tutorial guide](./docs/nemo-evaluator-launcher/tutorial.md).

**Getting an NGC API Key for build.nvidia.com:**

To use out-of-the-box build.nvidia.com APIs, you need an API key:

1. Register an account at [build.nvidia.com](https://build.nvidia.com).
2. In the Setup menu under Keys/Secrets, generate an API key.
3. Set the environment variable by executing `export NGC_API_KEY=<YOUR_API_KEY>`.

### 3. Run Your First Evaluation

Run a small evaluation on your local machine. The launcher automatically pulls the correct container and executes the benchmark. The list of benchmarks is directly configured in the YAML file.

**Configuration Examples**: Explore ready-to-use configuration files in [`packages/nemo-evaluator-launcher/examples/`](./packages/nemo-evaluator-launcher/examples/) for local, Lepton, and Slurm deployments with various model hosting options (vLLM, NIM, hosted endpoints).

Once you have the example configuration file, either by cloning this repository or downloading one directly such as `local_nvidia_nemotron_nano_9b_v2.yaml`, you can run the following command:


```bash
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_nvidia_nemotron_nano_9b_v2 --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

After running this command, you will see a `job_id`, which can be used to track the job and its results. All logs will be available in your `<YOUR_OUTPUT_LOCAL_DIR>`.

### 4. Check Your Results

Results, logs, and run configurations are saved locally. Inspect the status of the evaluation job by using the corresponding `job_id`:

```bash
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

### Next Steps

- List all supported benchmarks:

  ```bash
  nemo-evaluator-launcher ls tasks
  ```

- Explore the [Supported Benchmarks](#supported-benchmarks-and-evaluation-harnesses) to see all available harnesses and benchmarks.
- Scale up your evaluations using the [Slurm Executor](./docs/nemo-evaluator-launcher/executors/slurm.md) or [Lepton Executor](./docs/nemo-evaluator-launcher/executors/lepton.md).
- Learn to evaluate self-hosted models in the extended [Tutorial guide](./docs/nemo-evaluator-launcher/tutorial.md) for nemo-evaluator-launcher.
- Customize your workflow with [Custom Exporters](./docs/nemo-evaluator-launcher/exporters/overview.md) or by evaluating with [proprietary data](./docs/nemo-evaluator/extending/framework-definition-file.md).


## Run with NeMo Framework

The NeMo Framework is NVIDIA’s GPU-accelerated, end-to-end training platform for large language models (LLMs), multimodal models, and speech models. It enables seamless scaling of both pretraining and post-training workloads, from a single GPU to clusters with thousands of nodes, supporting Hugging Face/PyTorch and Megatron models. NeMo includes a suite of libraries and curated training recipes to help users build models from start to finish.

The NeMo Evaluator is integrated within NeMo Framework, offering streamlined deployment and advanced evaluation capabilities for models trained using NeMo, leveraging state-of-the-art evaluation harnesses.

### Features

- **Multi-Backend Deployment**: Supports PyTriton and multi-instance evaluations using the Ray Serve deployment backend
- **Production-Ready**: Supports high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU and Multi-Node Support**: Enables distributed inference across multiple GPUs and compute nodes
- **OpenAI-Compatible API**: Provides RESTful endpoints aligned with OpenAI API specifications

### 1. Start NeMo Framework Container

For optimal performance and user experience, use the latest version of the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent $TAG and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

### 2. Deploy a Model

```bash
# Deploy a NeMo checkpoint
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "/path/to/your/checkpoint" \
  --model_id megatron_model \
  --port 8080 \
  --host 0.0.0.0
```

### 3. Evaluate the Model

```python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(results)
```


## 📊 Supported Benchmarks and Evaluation Harnesses

NeMo Evaluator Launcher provides pre-built evaluation containers for different evaluation harnesses through the NVIDIA NGC catalog. Each harness supports a variety of benchmarks, which can then be called via `nemo-evaluator`. This table provides a list of benchmark names per harness. A more detailed list of task names can be found in the [list of NGC containers](./docs/nemo-evaluator/index.md#ngc-containers).

| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **agentic_eval** | Agentic AI evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) | `25.08.1` | Agentic Eval Topic Adherence, Agentic Eval Tool Call, Agentic Eval Goal and Answer Accuracy |
| **bfcl** | Function calling | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) | `25.08.1` | BFCL v2 and v3 |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.08.1` | MBPP, MBPP-Plus, HumanEval, HumanEval+, Multiple (cpp, cs, d, go, java, jl, js, lua, php, pl, py, r, rb, rkt, rs, scala, sh, swift, ts) |
| **garak** | Safety and vulnerability testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.08.1` | Garak |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.08.1` | MedHelm |
| **hle** | Academic knowledge and problem solving | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle) | `25.08.1` | HLE |
| **ifbench** | Instruction following | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench) | `25.08.1` | IFBench |
| **livecodebench** | Coding | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench) | `25.08.1` | LiveCodeBench (v1-v6, 0724_0125, 0824_0225) |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.08.1` | ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MINERVA MMMLU-Pro, RACE, TruthfulQA, AGIEval, BBH, BBQ, CSQA, Frames, Global MMMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande|
| **mmath** | Multilingual math reasoning | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath) | `25.08.1` | EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.08.1` | MT-Bench |
| **rag_retriever_eval** | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) | `25.08.1` | RAG, Retriever |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.08.1` | Aegis v2, BBQ, WildGuard |
| **scicode** | Coding for scientific research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode) | `25.08.1` | SciCode |
| **simple-evals** | Common evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.08.1` | GPQA-D, MATH-500, AIME 24 & 25, HumanEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.08.1` | ToolTalk |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.08.1` | AI2D, ChartQA, OCRBench, SlideVQA |



## 🤝 Contribution Guide

We welcome community contributions. Please see our [Contribution Guide](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) for instructions on submitting pull requests, reporting issues, and suggesting features.


## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE) file for details.


## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions)
- **Documentation**: [NeMo Documentation](https://docs.nvidia.com/nemo/evaluator/latest/index.html)