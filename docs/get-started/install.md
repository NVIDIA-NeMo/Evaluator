(gs-install)=
# Installation Guide

NeMo Evaluator provides multiple installation paths depending on your needs. Choose the approach that best fits your use case.

## Choose Your Installation Path

```{list-table} Installation Path Comparison
:header-rows: 1
:widths: 25 25 50

* - **Installation Path**
  - **Best For**
  - **Key Features**
* - **NeMo Evaluator Launcher** (Recommended)
  - Most users who want unified CLI and orchestration across backends
  - • Unified CLI for 100+ benchmarks  
    • Multi-backend execution (local, Slurm, cloud)  
    • Built-in result export to MLflow, W&B, etc.  
    • Configuration management with examples
* - **NeMo Evaluator Core**
  - Developers building custom evaluation pipelines
  - • Programmatic Python API  
    • Direct container access  
    • Custom framework integration  
    • Advanced adapter configuration
* - **Container Direct**
  - Users who prefer container-based workflows
  - • Pre-built NGC evaluation containers  
    • Guaranteed reproducibility  
    • No local installation required  
    • Isolated evaluation environments
```

---

## Prerequisites

### System Requirements

- Python 3.10 or higher (supports 3.10, 3.11, 3.12, and 3.13)
- CUDA-compatible GPU(s) (tested on RTX A6000, A100, H100)
- Docker (for container-based workflows)

### Recommended Environment

- Python 3.12
- CUDA 12.9
- Ubuntu 24.04

---

## Installation Methods


::::{tab-set}

:::{tab-item} Launcher (Recommended)

Install NeMo Evaluator Launcher for unified CLI and orchestration:

```{literalinclude} _snippets/install_launcher.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

Quick verification:

```{literalinclude} _snippets/verify_launcher.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::

:::{tab-item} Core Library

Install NeMo Evaluator Core for programmatic access:

```{literalinclude} _snippets/install_core.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

Quick verification:

```{literalinclude} _snippets/verify_core.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::

:::{tab-item} NGC Containers

Use pre-built evaluation containers from NVIDIA NGC for guaranteed reproducibility:

```bash
# Pull evaluation containers (no local installation needed)
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:{{ docker_compose_latest }}
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
```

```bash
# Run container interactively
docker run --rm -it \
    -v $(pwd)/results:/workspace/results \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

# Or run evaluation directly
docker run --rm \
    -v $(pwd)/results:/workspace/results \
    -e NGC_API_KEY=nvapi-xxx \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
    nemo-evaluator run_eval \
        --eval_type mmlu_pro \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_id meta/llama-3.1-8b-instruct \
        --api_key_name NGC_API_KEY \
        --output_dir /workspace/results
```

Quick verification:

```bash
# Test container access
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
    nemo-evaluator ls | head -5
echo " Container access verified"
```

:::

::::

---

## Clone the Repository

Clone the NeMo Evaluator repository to get easy access to our ready-to-use examples:

```bash
git clone https://github.com/NVIDIA-NeMo/Evaluator.git
```

Run the example:

```bash
cd Evaluator/

export NGC_API_KEY=nvapi-...  # API Key with access to build.nvidia.com
nemo-evaluator-launcher run \
  --config-dir packages/nemo-evaluator-launcher/examples \
  --config-name local_nvidia_nemotron_nano_9b_v2 \
  --override execution.output_dir=nemotron-eval
```

## Add Evaluation Harnesses to Your Environment 

Build your custom evaluation pipeline by adding evaluation harness packages to your environment of choice:

```bash
pip install nemo-evaluator <evaluation-package>
```

Available PyPI packages:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Package Name
  - PyPI URL
* - nvidia-bfcl
  - <https://pypi.org/project/nvidia-bfcl/>
* - nvidia-bigcode-eval
  - <https://pypi.org/project/nvidia-bigcode-eval/>
* - nvidia-compute-eval
  - <https://pypi.org/project/nvidia-compute-eval/>
* - nvidia-eval-factory-garak
  - <https://pypi.org/project/nvidia-eval-factory-garak/>
* - nvidia-genai-perf-eval
  - <https://pypi.org/project/nvidia-genai-perf-eval/>
* - nvidia-crfm-helm
  - <https://pypi.org/project/nvidia-crfm-helm/>
* - nvidia-hle
  - <https://pypi.org/project/nvidia-hle/>
* - nvidia-ifbench
  - <https://pypi.org/project/nvidia-ifbench/>
* - nvidia-livecodebench
  - <https://pypi.org/project/nvidia-livecodebench/>
* - nvidia-lm-eval
  - <https://pypi.org/project/nvidia-lm-eval/>
* - nvidia-mmath
  - <https://pypi.org/project/nvidia-mmath/>
* - nvidia-mtbench-evaluator
  - <https://pypi.org/project/nvidia-mtbench-evaluator/>
* - nvidia-eval-factory-nemo-skills
  - <https://pypi.org/project/nvidia-eval-factory-nemo-skills/>
* - nvidia-safety-harness
  - <https://pypi.org/project/nvidia-safety-harness/>
* - nvidia-scicode
  - <https://pypi.org/project/nvidia-scicode/>
* - nvidia-simple-evals
  - <https://pypi.org/project/nvidia-simple-evals/>
* - nvidia-tooltalk
  - <https://pypi.org/project/nvidia-tooltalk/>
* - nvidia-vlmeval
  - <https://pypi.org/project/nvidia-vlmeval/>
```

:::{note}
Evaluation harnessess that require complex environments are not available as packages but only as containers.
:::