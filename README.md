# NeMo Evaluator SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/downloads/)
[![Tests](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![nemo-evaluator PyPI version](https://img.shields.io/pypi/v/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator-launcher PyPI version](https://img.shields.io/pypi/v/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![nemo-evaluator-launcher PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)](#)

## [📖 Documentation](https://docs.nvidia.com/nemo/evaluator/latest/) 

NeMo Evaluator SDK is an open-source platform for robust, reproducible, and scalable evaluation of Large Language Models. It enables you to run hundreds of benchmarks across popular evaluation harnesses against any OpenAI-compatible model API. Evaluations execute in open-source Docker containers for auditable and trustworthy results. The platform's containerized architecture allows for the rapid integration of public benchmarks and private datasets.

NeMo Evaluator SDK is built on four core principles to provide a reliable and versatile evaluation experience:

- **Reproducibility by Default**: All configurations, random seeds, and software provenance are captured automatically for auditable and repeatable evaluations.
- **Scale Anywhere**: Run evaluations from a local machine to a Slurm cluster or cloud-native backends like Lepton AI without changing your workflow.
- **State-of-the-Art Benchmarking**: Access a comprehensive suite of over 100 benchmarks from 18 popular open-source evaluation harnesses. See the full list of [Supported benchmarks and evaluation harnesses](#-supported-benchmarks-and-evaluation-harnesses).
- **Extensible and Customizable**: Integrate new evaluation harnesses, add custom benchmarks with proprietary data, and define custom result exporters for existing MLOps tooling.

## ⚙️ How It Works: Launcher and Core Engine

The platform consists of two main components:

- **`nemo-evaluator` ([The Evaluation Core Engine](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/index.html))**: A Python library that manages the interaction between an evaluation harness and the model being tested.
- **`nemo-evaluator-launcher` ([The CLI and Orchestration](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/index.html))**: The primary user interface and orchestration layer. It handles configuration, selects the execution environment, and launches the appropriate container to run the evaluation.

Most users typically interact with `nemo-evaluator-launcher`, which serves as a universal gateway to different benchmarks and harnesses. However, it is also possible to interact directly with `nemo-evaluator` by following this [guide](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/workflows/cli.html).


## 📊 Supported Benchmarks and Evaluation Harnesses

NeMo Evaluator Launcher provides pre-built evaluation containers for different evaluation harnesses through the NVIDIA NGC catalog. Each harness supports a variety of benchmarks, which can then be called via `nemo-evaluator`. This table provides a list of benchmark names per harness. A more detailed list of task names can be found in the [list of NGC containers](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/containers/index.html).

| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **bfcl** | Function calling | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) | `25.10` | BFCL v2 and v3 |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.10` | MBPP, MBPP-Plus, HumanEval, HumanEval+, Multiple (cpp, cs, d, go, java, jl, js, lua, php, pl, py, r, rb, rkt, rs, scala, sh, swift, ts) |
| **compute-eval** | CUDA code evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/compute-eval) | `25.10` | CCCL, Combined Problems, CUDA |
| **garak** | Safety and vulnerability testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.10` | Garak |
| **genai-perf** | GenAI performance benchmarking | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf) | `25.10` | GenAI Perf Generation & Summarization |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.10` | MedHelm |
| **hle** | Academic knowledge and problem solving | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle) | `25.10` | HLE |
| **ifbench** | Instruction following | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench) | `25.10` | IFBench |
| **livecodebench** | Coding | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench) | `25.10` | LiveCodeBench (v1-v6, 0724_0125, 0824_0225) |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.10` | ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MINERVA MMMLU-Pro, RACE, TruthfulQA, AGIEval, BBH, BBQ, CSQA, Frames, Global MMMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande|
| **mmath** | Multilingual math reasoning | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath) | `25.10` | EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.10` | MT-Bench |
| **nemo-skills** | Language model benchmarks (science, math, agentic)  | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills) | `25.10` | AIME 24 & 25, BFCL_v3, GPQA, HLE, LiveCodeBench, MMLU, MMLU-Pro |
| **profbench** | Professional domains in Business and Scientific Research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench) | `25.10` | ProfBench |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.10` | Aegis v2, BBQ, WildGuard |
| **scicode** | Coding for scientific research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode) | `25.10` | SciCode |
| **simple-evals** | Common evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.10` | GPQA-D, MATH-500, AIME 24 & 25, HumanEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.10` | ToolTalk |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.10` | AI2D, ChartQA, OCRBench, SlideVQA |

## 🚀 Quickstart

Get your first evaluation result in minutes. This guide uses your local machine to run a small benchmark against an OpenAI API-compatible endpoint.

### 1. Install the Launcher

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
- **Models trained with NeMo framework**: Host your models trained with NeMo framework by deploying them as OpenAI-compatible endpoints using [NeMo Export-Deploy](https://github.com/nvidia-nemo/export-deploy/tree/main). More detailed user guide [here](https://github.com/nvidia-nemo/evaluator/tree/main/docs/nemo-fw).

<!-- TODO(martas): uncomment once publish -->
<!-- For detailed setup instructions including self-hosted configurations, see the [tutorials](https://docs.nvidia.com/nemo/evaluator/latest/tutorials/). -->

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

## 🤝 Contribution Guide

We welcome community contributions. Please see our [Contribution Guide](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) for instructions on submitting pull requests, reporting issues, and suggesting features.


## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE) file for details.


## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions)
- **Documentation**: [NeMo Evaluator Documentation](https://docs.nvidia.com/nemo/evaluator/latest/)
