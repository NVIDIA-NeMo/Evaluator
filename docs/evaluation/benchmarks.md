(eval-benchmarks)=

# Benchmark Catalog

Comprehensive catalog of hundreds of benchmarks across popular evaluation harnesses, all available through NGC containers and the NeMo Evaluator platform.


## Available via Launcher

```{literalinclude} _snippets/commands/list_tasks.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Available via Direct Container Access

```{literalinclude} _snippets/commands/list_tasks_core.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Choosing Benchmarks for Academic Research

:::{admonition} Benchmark Selection Guide
:class: tip

**For General Knowledge**:
- `mmlu_pro` - Expert-level knowledge across 14 domains
- `gpqa_diamond` - Graduate-level science questions

**For Mathematical & Quantitative Reasoning**:
- `AIME_2025` - American Invitational Mathematics Examination (AIME) 2025 questions
- `mgsm` - Multilingual math reasoning

**For Instruction Following & Alignment**:
- `ifbench` - Precise instruction following
- `mtbench` - Multi-turn conversation quality

See benchmark categories below and {ref}`benchmarks-full-list` for more details.
:::

## Benchmark Categories

###  **Academic and Reasoning**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **simple-evals**
  - Common evaluation tasks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
  - GPQA-D, MATH-500, AIME 24 & 25, HumanEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA 
* - **lm-evaluation-harness**
  - Language model benchmarks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
  - ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MINERVA MMMLU-Pro, RACE, TruthfulQA, AGIEval, BBH, BBQ, CSQA, Frames, Global MMMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMMLU, MMMLU-Pro, MMMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande
* - **hle**
  - Academic knowledge and problem solving
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)
  - HLE 
* - **ifbench**
  - Instruction following
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)
  - IFBench 
* - **mtbench**
  - Multi-turn conversation evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
  - MT-Bench
* - **nemo-skills**
  - Language model benchmarks (science, math, agentic) 
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills)
  - AIME 24 & 25, BFCL_v3, GPQA, HLE, LiveCodeBench, MMLU, MMLU-Pro 
* - **profbench**
  - Evaluation of professional knowledge accross Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
  - Report Gerenation, LLM Judge
```

:::{note}
BFCL tasks from the nemo-skills container require function calling capabilities. See {ref}`deployment-testing-compatibility` for checking if your endpoint is compatible.
:::

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: ifeval
    - name: gsm8k_cot_instruct
    - name: gpqa_diamond
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...
export HF_TOKEN=hf_...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

###  **Code Generation**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **bigcode-evaluation-harness**
  - Code generation evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)
  - MBPP, MBPP-Plus, HumanEval, HumanEval+, Multiple (cpp, cs, d, go, java, jl, js, lua, php, pl, py, r, rb, rkt, rs, scala, sh, swift, ts) 
* - **livecodebench**
  - Coding
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)
  - LiveCodeBench (v1-v6, 0724_0125, 0824_0225) 
* - **scicode**
  - Coding for scientific research
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)
  - SciCode 
```

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: humaneval_instruct
    - name: mbbp
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

###  **Safety and Security**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **garak**
  - Safety and vulnerability testing
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)
  - Garak
* - **safety-harness**
  - Safety and bias evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
  - Aegis v2, BBQ, WildGuard
```

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: aegis_v2
    - name: garak
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...
export HF_TOKEN=hf_...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

###  **Function Calling**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **bfcl**
  - Function calling
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)
  - BFCL v2 and v3 
* - **tooltalk**
 - Tool usage evaluation
 - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)
 - ToolTalk 
```

:::{note}
Some of the tasks in this category require function calling capabilities. See {ref}`deployment-testing-compatibility` for checking if your endpoint is compatible.
:::

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: bfclv2_ast_prompting
    - name: tooltalk
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```


###  **Vision-Language Models**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **vlmevalkit**
  - Vision-language model evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
  - AI2D, ChartQA, OCRBench, SlideVQA
```

:::{note}
The tasks in this category require a VLM chat endpoint. See {ref}`deployment-testing-compatibility` for checking if your endpoint is compatible.
:::

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: ocrbench
    - name: chartqa
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

###  **Domain-Specific**

```{list-table}
:header-rows: 1
:widths: 20 30 30 50

* - Container
  - Description
  - NGC Catalog
  - Benchmarks
* - **helm**
  - Holistic evaluation framework
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)
  - MedHelm 
```

**Example Usage:**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: pubmed_qa
    - name: medcalc_bench
```

Run evaluation:

```bash
export NGC_API_KEY=nvapi-...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

## Container Details

For detailed specifications of each container, see {ref}`nemo-evaluator-containers`.

### Quick Container Access

Pull and run any evaluation container directly:

```bash
# Academic benchmarks
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker run --rm -it nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# Code generation
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
docker run --rm -it nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}

# Safety evaluation
docker pull nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}
docker run --rm -it nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}
```

### Available Tasks by Container

For a complete list of available tasks in each container:

```bash
# List tasks in any container
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} nemo-evaluator ls

# Or use the launcher for unified access
nemo-evaluator-launcher ls tasks
```

## Integration Patterns

NeMo Evaluator provides multiple integration options to fit your workflow:

```bash
# Launcher CLI (recommended for most users)
nemo-evaluator-launcher ls tasks
nemo-evaluator-launcher run --config-dir . --config-name local_mmlu_evaluation.yaml

# Container direct execution
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} nemo-evaluator ls

# Python API (for programmatic control)
# See the Python API documentation for details
```

## Benchmark Selection Best Practices

### For Model Development

**Iterative Testing**:
- Start with `limit_samples=100` for quick feedback during development
- Run full evaluations before major releases
- Track metrics over time to measure improvement

**Configuration**:
```python
# Development testing
params = ConfigParams(
    limit_samples=100,      # Quick iteration
    temperature=0.01,       # Deterministic
    parallelism=4
)

# Production evaluation
params = ConfigParams(
    limit_samples=None,     # Full dataset
    temperature=0.01,       # Deterministic
    parallelism=8          # Higher throughput
)
```

### For Specialized Domains

- **Code Models**: Focus on `humaneval`, `mbpp`, `livecodebench`
- **Instruction Models**: Emphasize `ifbench`, `mtbench`
- **Multilingual Models**: Include `arc_multilingual`, `hellaswag_multilingual`, `mgsm`
- **Safety-Critical**: Prioritize `safety-harness` and `garak` evaluations

(benchmarks-full-list)=
## Full Benchmarks List

```{include} ../_resources/tasks-table.md
```

## Next Steps

- **Container Details**: Browse {ref}`nemo-evaluator-containers` for complete specifications
- **Custom Benchmarks**: Learn {ref}`framework-definition-file` for custom evaluations
