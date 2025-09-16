# NeMo Evaluator: The Universal Platform for LLM Evaluation

NeMo Evaluator is an open-source platform for robust, reproducible, and scalable evaluation of Large Language Models. It enables you to run hundreds of benchmarks across popular evaluation harnesses against any OpenAI-compatible model API. Evaluations execute in open-source Docker containers for auditable and trustworthy results. The platform's containerized architecture allows for the rapid integration of public benchmarks and private datasets.

[Tutorial](./docs/nemo-evaluator-launcher/quickstart.md) | [Supported Benchmarks](#supported-benchmarks-and-evaluation-harnesses) | [Configuration Examples](./docs/nemo-evaluator/reference/containers.md) | [Contribution Guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md)

## Key Pillars

NeMo Evaluator is built on four core principles to provide a reliable and versatile evaluation experience.

- **Reproducibility by Default** -- All configurations, random seeds, and software provenance are captured automatically for auditable and repeatable evaluations.
- **Scale Anywhere** --  Run evaluations from a local machine to a Slurm cluster or cloud-native backends like Lepton AI without changing your workflow.
- **State-of-the-Art Benchmarking** --  Access a comprehensive suite of over 100 benchmarks from 18 popular open-source evaluation harnesses. See the full list of [Supported benchmarks and evaluation harnesses](#supported-benchmarks-and-evaluation-harnesses).
- **Extensible and Customizable** --  Integrate new evaluation harnesses, add custom benchmarks with proprietary data, and define custom result exporters for existing MLOps tooling.

## How It Works: Launcher and Core Engine

The platform consists of two main components:

- **`nemo-evaluator` ([The Evaluation Core Engine](./docs/nemo-evaluator/index.md))**: A Python library that manages the interaction between an evaluation harness and the model being tested.
- **`nemo-evaluator-launcher` ([The CLI and Orchestration](./docs/nemo-evaluator-launcher/index.md))**: The primary user interface and orchestration layer. It handles configuration, selects the execution environment, and launches the appropriate container to run the evaluation.

Most users only need to interact with the `nemo-evaluator-launcher` as universal gateway to different benchmarks and harnesses. It is however possible to interact directly with `nemo-evaluator` by following this [guide](./docs/nemo-evaluator/workflows/using-containers.md).

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

### 1. Install the Launcher

The launcher is the only package required to get started.

```bash
pip install nemo-evaluator-launcher
```

### 2. Set Up Your Model Endpoint

NeMo Evaluator works with any model that exposes an OpenAI-compatible endpoint. For this quickstart, we will use the OpenAI API.

**What is an OpenAI-compatible endpoint?** A server that exposes /v1/chat/completions and /v1/completions endpoints, matching the OpenAI API specification.

**Options for model endpoints:**

- **Hosted endpoints** (fastest): Use ready-to-use hosted models from providers like build.nvidia.com that expose OpenAI-compatible APIs with no hosting required.
- **Self-hosted options**: Host your own models using tools like NVIDIA NIM, vLLM, or TensorRT-LLM for full control over your evaluation environment.

For detailed setup instructions including self-hosted configurations, see the [quickstart guide](./docs/nemo-evaluator-launcher/quickstart.md).

**Getting an NGC API Key for build.nvidia.com:**
To use out-of-the-box build.nvidia.com APIs, you need an API key:

1. Register an account at [build.nvidia.com](https://build.nvidia.com)
2. In the Setup menu under Keys/Secrets, generate an API key
3. Set the environment variable by executing `export NGC_API_KEY=<<YOUR_API_KEY>>`

### 3. Run Your First Evaluation

Run a small evaluation on your local machine. The launcher automatically pulls the correct container and executes the benchmark. The list of benchmarks is directly configured in the yaml file.

```bash
nemo-evaluator-launcher run --config-dir examples --config-name nvidia-nemotron-nano-9b-v2 --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

Upon running this command, you will be able to see a job_id, which can then be used for tracking the job.

### 4. Check Your Results

Results, logs, and run configurations are saved locally. Inspect the status of the evaluation job by using the corresponding job id:

```bash
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

### Next Steps

- List all supported benchmarks:

```bash
nemo-evaluator-launcher ls tasks
```

- Explore the [Supported Benchmarks](#supported-benchmarks-and-evaluation-harnesses) to see all available harnesses and benchmarks.
- Scale up your evaluations using the [Slurm Executor] or [Lepton Executor](./docs/nemo-evaluator-launcher/executors/slurm.md).
- Learn to evaluate self-hosted models in the extended [Tutorial and quickstart guide](./docs/nemo-evaluator-launcher/quickstart.md) for nemo-evaluator-launcher.
- Customize your workflow with [Custom Exporters](./docs/nemo-evaluator-launcher/exporters/overview.md) or by evaluating with [proprietary data](./docs/nemo-evaluator/extending/framework-definition-file.md).

## Supported Benchmarks and Evaluation Harnesses

NeMo Evaluator provides pre-built evaluation containers for different evaluation harnesses through the NVIDIA NGC catalog. Each harness supports a variety of benchmarks, which can then be called via `nemo-evaluator`. This table provides a list of benchmark names per harness. A more detailed list of task names can be found in the [list of NGC containers](./docs/nemo-evaluator/index.md#ngc-containers).

| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **agentic_eval** | Agentic AI evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) | `25.08.0` | Agentic Eval Topic Adherence, Agentic Eval Tool Call, Agentic Eval Goal and Answer Accuracy |
| **bfcl** | Function calling | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) | `25.08.0` | BFCL v2 and v3 |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.08.0` | MBPP, MBPP-Plus, HumanEval, HumanEval+, Multiple (cpp, cs, d, go, java, jl, js, lua, php, pl, py, r, rb, rkt, rs, scala, sh, swift, ts) |
| **garak** | Safety and vulnerability testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.08.0` | Garak |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.08.0` | MedHelm |
| **hle** | Academic knowledge and problem solving | [Link]() | `25.08.0` | HLE |
| **ifbench** | Instruction following | [Link]() | `25.08.0` | IFBench |
| **livecodebench** | Coding | [Link]() | `25.08.0` | LiveCodeBench (v1-v6, 0724_0125, 0824_0225) |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.08.0` | ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MINERVA MMMLU-Pro, RACE, TruthfulQA, AGIEval, BBH, BBQ, CSQA, Frames, Global MMMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande|
| **mmath** | Multilingual math reasoning | [Link]() | `25.08.0` | EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.08.0` | MT-Bench |
| **rag_retriever_eval** | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) | `25.08.0` | RAG, Retriever |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.08.0` | Aegis v2, BBQ, WildGuard |
| **scicode** | Coding for scientific research | [Link]() | `25.08.0` | SciCode |
| **simple-evals** | Common evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.08.0` | GPQA-D, MATH-500, AIME 24 & 25, HumanEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.08.0` | ToolTalk |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.08.0` | AI2D, ChartQA, OCRBench, SlideVQA |

## Contribution Guide

We welcome community contributions. Please see our [Contribution Guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md) for instructions on submitting pull requests, reporting issues, and suggesting features.
