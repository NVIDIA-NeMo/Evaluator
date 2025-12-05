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
  - GPQA-D, MATH-500, AIME 24 & 25, HumanEval, HumanEval+, MGSM, MMLU (also multilingual), MMLU-Pro, MMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA, BrowseComp, HealthBench 
* - **lm-evaluation-harness**
  - Language model benchmarks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
  - ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MBPP+, MINERVA Math, RACE, AGIEval, BBH, BBQ, CSQA, Frames, Global MMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMLU, MMLU-Pro, MMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande
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
    --config ./config.yml \
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
    --config ./config.yml \
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
  - Aegis v2, WildGuard
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
    --config ./config.yml \
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
    --config ./config.yml \
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
  - AI2D, ChartQA, MMMU, MathVista-MINI, OCRBench, SlideVQA
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
    --config ./config.yml \
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
    --config ./config.yml \
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
nemo-evaluator-launcher run --config ./local_mmlu_evaluation.yaml

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

<!-- BEGIN_BENCH_TABLE -->
```{list-table}
:header-rows: 1
:widths: 20 25 15 15 25

* - Container
  - Description
  - NGC Catalog
  - Latest Tag
  - Tasks
* - <a href="../task_catalog/harnesses/bfcl.html#bfcl"><strong>bfcl</strong></a>
  - BFCL v2 with Single-turn, Live and Non-Live, AST and Exec evaluation. Not using native function calling.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv2">bfclv2</a>, <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv2-ast">bfclv2_ast</a>, <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv2-ast-prompting">bfclv2_ast_prompting</a>, <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv3">bfclv3</a>, <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv3-ast">bfclv3_ast</a>, <a href="../task_catalog/harnesses/bfcl.html#bfcl-bfclv3-ast-prompting">bfclv3_ast_prompting</a>
* - <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness"><strong>bigcode-evaluation-harness</strong></a>
  - HumanEval is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-humaneval">humaneval</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-humaneval-instruct">humaneval_instruct</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-humanevalplus">humanevalplus</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-mbpp">mbpp</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-mbppplus">mbppplus</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-mbppplus-nemo">mbppplus_nemo</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-cpp">multiple-cpp</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-cs">multiple-cs</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-d">multiple-d</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-go">multiple-go</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-java">multiple-java</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-jl">multiple-jl</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-js">multiple-js</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-lua">multiple-lua</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-php">multiple-php</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-pl">multiple-pl</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-py">multiple-py</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-r">multiple-r</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-rb">multiple-rb</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-rkt">multiple-rkt</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-rs">multiple-rs</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-scala">multiple-scala</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-sh">multiple-sh</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-swift">multiple-swift</a>, <a href="../task_catalog/harnesses/bigcode-evaluation-harness.html#bigcode-evaluation-harness-multiple-ts">multiple-ts</a>
* - <a href="../task_catalog/harnesses/garak.html#garak"><strong>garak</strong></a>
  - Task for running the default set of Garak probes
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/garak.html#garak-garak">garak</a>
* - <a href="../task_catalog/harnesses/helm.html#helm"><strong>helm</strong></a>
  - Extract and structure information from patient-doctor conversations
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/helm.html#helm-aci-bench">aci_bench</a>, <a href="../task_catalog/harnesses/helm.html#helm-ehr-sql">ehr_sql</a>, <a href="../task_catalog/harnesses/helm.html#helm-head-qa">head_qa</a>, <a href="../task_catalog/harnesses/helm.html#helm-med-dialog-healthcaremagic">med_dialog_healthcaremagic</a>, <a href="../task_catalog/harnesses/helm.html#helm-med-dialog-icliniq">med_dialog_icliniq</a>, <a href="../task_catalog/harnesses/helm.html#helm-medbullets">medbullets</a>, <a href="../task_catalog/harnesses/helm.html#helm-medcalc-bench">medcalc_bench</a>, <a href="../task_catalog/harnesses/helm.html#helm-medec">medec</a>, <a href="../task_catalog/harnesses/helm.html#helm-medhallu">medhallu</a>, <a href="../task_catalog/harnesses/helm.html#helm-medi-qa">medi_qa</a>, <a href="../task_catalog/harnesses/helm.html#helm-medication-qa">medication_qa</a>, <a href="../task_catalog/harnesses/helm.html#helm-mtsamples-procedures">mtsamples_procedures</a>, <a href="../task_catalog/harnesses/helm.html#helm-mtsamples-replicate">mtsamples_replicate</a>, <a href="../task_catalog/harnesses/helm.html#helm-pubmed-qa">pubmed_qa</a>, <a href="../task_catalog/harnesses/helm.html#helm-race-based-med">race_based_med</a>
* - <a href="../task_catalog/harnesses/hle.html#hle"><strong>hle</strong></a>
  - Text-only questions from Humanity's Last Exam
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/hle.html#hle-hle">hle</a>, <a href="../task_catalog/harnesses/hle.html#hle-hle-aa-v2">hle_aa_v2</a>
* - <a href="../task_catalog/harnesses/ifbench.html#ifbench"><strong>ifbench</strong></a>
  - IFBench with vanilla settings
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/ifbench.html#ifbench-ifbench">ifbench</a>, <a href="../task_catalog/harnesses/ifbench.html#ifbench-ifbench-aa-v2">ifbench_aa_v2</a>
* - <a href="../task_catalog/harnesses/livecodebench.html#livecodebench"><strong>livecodebench</strong></a>
  - “Execute” a program on an input, evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codeexecution-v2">codeexecution_v2</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codeexecution-v2-cot">codeexecution_v2_cot</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-notfast">codegeneration_notfast</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-latest">codegeneration_release_latest</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v1">codegeneration_release_v1</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v2">codegeneration_release_v2</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v3">codegeneration_release_v3</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v4">codegeneration_release_v4</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v5">codegeneration_release_v5</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-codegeneration-release-v6">codegeneration_release_v6</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-livecodebench-0724-0125">livecodebench_0724_0125</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-livecodebench-0824-0225">livecodebench_0824_0225</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-livecodebench-aa-v2">livecodebench_aa_v2</a>, <a href="../task_catalog/harnesses/livecodebench.html#livecodebench-testoutputprediction">testoutputprediction</a>
* - <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness"><strong>lm-evaluation-harness</strong></a>
  - Version of the AGIEval-EN-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-agieval-en-cot">adlr_agieval_en_cot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-arc-challenge-llama-25-shot">adlr_arc_challenge_llama_25_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-commonsense-qa-7-shot">adlr_commonsense_qa_7_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-global-mmlu-lite-5-shot">adlr_global_mmlu_lite_5_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-gpqa-diamond-cot-5-shot">adlr_gpqa_diamond_cot_5_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-gsm8k-cot-8-shot">adlr_gsm8k_cot_8_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-humaneval-greedy">adlr_humaneval_greedy</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-humaneval-sampled">adlr_humaneval_sampled</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-math-500-4-shot-sampled">adlr_math_500_4_shot_sampled</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-greedy">adlr_mbpp_sanitized_3_shot_greedy</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-sampled">adlr_mbpp_sanitized_3_shot_sampled</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-mgsm-native-cot-8-shot">adlr_mgsm_native_cot_8_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-minerva-math-nemo-4-shot">adlr_minerva_math_nemo_4_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-mmlu">adlr_mmlu</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-mmlu-pro-5-shot-base">adlr_mmlu_pro_5_shot_base</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-race">adlr_race</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-truthfulqa-mc2">adlr_truthfulqa_mc2</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-adlr-winogrande-5-shot">adlr_winogrande_5_shot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-agieval">agieval</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-arc-challenge">arc_challenge</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-arc-challenge-chat">arc_challenge_chat</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-arc-multilingual">arc_multilingual</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-bbh">bbh</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-bbh-instruct">bbh_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-bbq">bbq</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-commonsense-qa">commonsense_qa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-frames-naive">frames_naive</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-frames-naive-with-links">frames_naive_with_links</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-frames-oracle">frames_oracle</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu">global_mmlu</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-ar">global_mmlu_ar</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-bn">global_mmlu_bn</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-de">global_mmlu_de</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-en">global_mmlu_en</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-es">global_mmlu_es</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-fr">global_mmlu_fr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full">global_mmlu_full</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-am">global_mmlu_full_am</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ar">global_mmlu_full_ar</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-bn">global_mmlu_full_bn</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-cs">global_mmlu_full_cs</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-de">global_mmlu_full_de</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-el">global_mmlu_full_el</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-en">global_mmlu_full_en</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-es">global_mmlu_full_es</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-fa">global_mmlu_full_fa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-fil">global_mmlu_full_fil</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-fr">global_mmlu_full_fr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ha">global_mmlu_full_ha</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-he">global_mmlu_full_he</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-hi">global_mmlu_full_hi</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-id">global_mmlu_full_id</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ig">global_mmlu_full_ig</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-it">global_mmlu_full_it</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ja">global_mmlu_full_ja</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ko">global_mmlu_full_ko</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ky">global_mmlu_full_ky</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-lt">global_mmlu_full_lt</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-mg">global_mmlu_full_mg</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ms">global_mmlu_full_ms</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ne">global_mmlu_full_ne</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-nl">global_mmlu_full_nl</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ny">global_mmlu_full_ny</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-pl">global_mmlu_full_pl</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-pt">global_mmlu_full_pt</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ro">global_mmlu_full_ro</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-ru">global_mmlu_full_ru</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-si">global_mmlu_full_si</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-sn">global_mmlu_full_sn</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-so">global_mmlu_full_so</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-sr">global_mmlu_full_sr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-sv">global_mmlu_full_sv</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-sw">global_mmlu_full_sw</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-te">global_mmlu_full_te</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-tr">global_mmlu_full_tr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-uk">global_mmlu_full_uk</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-vi">global_mmlu_full_vi</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-yo">global_mmlu_full_yo</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-full-zh">global_mmlu_full_zh</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-hi">global_mmlu_hi</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-id">global_mmlu_id</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-it">global_mmlu_it</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-ja">global_mmlu_ja</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-ko">global_mmlu_ko</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-pt">global_mmlu_pt</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-sw">global_mmlu_sw</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-yo">global_mmlu_yo</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-global-mmlu-zh">global_mmlu_zh</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gpqa">gpqa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gpqa-diamond-cot">gpqa_diamond_cot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gsm8k">gsm8k</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gsm8k-cot-instruct">gsm8k_cot_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gsm8k-cot-llama">gsm8k_cot_llama</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gsm8k-cot-zeroshot">gsm8k_cot_zeroshot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-gsm8k-cot-zeroshot-llama">gsm8k_cot_zeroshot_llama</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-hellaswag">hellaswag</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-hellaswag-multilingual">hellaswag_multilingual</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-humaneval-instruct">humaneval_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-ifeval">ifeval</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-m-mmlu-id-str">m_mmlu_id_str</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mbpp-plus">mbpp_plus</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mgsm">mgsm</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mgsm-cot">mgsm_cot</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu">mmlu</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-cot-0-shot-chat">mmlu_cot_0_shot_chat</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-instruct">mmlu_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-logits">mmlu_logits</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-pro">mmlu_pro</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-pro-instruct">mmlu_pro_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox">mmlu_prox</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox-de">mmlu_prox_de</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox-es">mmlu_prox_es</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox-fr">mmlu_prox_fr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox-it">mmlu_prox_it</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-prox-ja">mmlu_prox_ja</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-redux">mmlu_redux</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-mmlu-redux-instruct">mmlu_redux_instruct</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-musr">musr</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-openbookqa">openbookqa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-piqa">piqa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-social-iqa">social_iqa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-truthfulqa">truthfulqa</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-wikilingua">wikilingua</a>, <a href="../task_catalog/harnesses/lm-evaluation-harness.html#lm-evaluation-harness-winogrande">winogrande</a>
* - <a href="../task_catalog/harnesses/mtbench.html#mtbench"><strong>mtbench</strong></a>
  - Standard MT-Bench
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/mtbench.html#mtbench-mtbench">mtbench</a>, <a href="../task_catalog/harnesses/mtbench.html#mtbench-mtbench-cor1">mtbench-cor1</a>
* - <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills"><strong>nemo-skills</strong></a>
  - AA-LCR
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-aa-lcr">ns_aa_lcr</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-aime2024">ns_aime2024</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-aime2025">ns_aime2025</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-bfcl-v3">ns_bfcl_v3</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-bfcl-v4">ns_bfcl_v4</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-gpqa">ns_gpqa</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-hle">ns_hle</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-ifbench">ns_ifbench</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-livecodebench">ns_livecodebench</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-mmlu">ns_mmlu</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-mmlu-pro">ns_mmlu_pro</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-ruler">ns_ruler</a>, <a href="../task_catalog/harnesses/nemo-skills.html#nemo-skills-ns-scicode">ns_scicode</a>
* - <a href="../task_catalog/harnesses/profbench.html#profbench"><strong>profbench</strong></a>
  - Run LLM judge on provided ProfBench reports and score them
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/profbench.html#profbench-llm-judge">llm_judge</a>, <a href="../task_catalog/harnesses/profbench.html#profbench-report-generation">report_generation</a>
* - <a href="../task_catalog/harnesses/safety-eval.html#safety-eval"><strong>safety-eval</strong></a>
  - Aegis V2 without evaluating reasoning traces. This version is used by the NeMo Safety Toolkit.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/safety-eval.html#safety-eval-aegis-v2">aegis_v2</a>, <a href="../task_catalog/harnesses/safety-eval.html#safety-eval-aegis-v2-reasoning">aegis_v2_reasoning</a>, <a href="../task_catalog/harnesses/safety-eval.html#safety-eval-wildguard">wildguard</a>
* - <a href="../task_catalog/harnesses/scicode.html#scicode"><strong>scicode</strong></a>
  - - SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant does not include scientist-annotated background in the prompts.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/scicode.html#scicode-scicode">scicode</a>, <a href="../task_catalog/harnesses/scicode.html#scicode-scicode-aa-v2">scicode_aa_v2</a>, <a href="../task_catalog/harnesses/scicode.html#scicode-scicode-background">scicode_background</a>
* - <a href="../task_catalog/harnesses/simple-evals.html#simple-evals"><strong>simple-evals</strong></a>
  - AIME 2024 questions, math, using Artificial Analysis's setup.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aa-aime-2024">AA_AIME_2024</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aa-math-test-500">AA_math_test_500</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aime-2024">AIME_2024</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aime-2025">AIME_2025</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aime-2025-aa-v2">AIME_2025_aa_v2</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aime-2024-nemo">aime_2024_nemo</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-aime-2025-nemo">aime_2025_nemo</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-browsecomp">browsecomp</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-diamond">gpqa_diamond</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-diamond-aa-v2">gpqa_diamond_aa_v2</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-diamond-aa-v2-llama-4">gpqa_diamond_aa_v2_llama_4</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-diamond-nemo">gpqa_diamond_nemo</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-extended">gpqa_extended</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-gpqa-main">gpqa_main</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-healthbench">healthbench</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-healthbench-consensus">healthbench_consensus</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-healthbench-hard">healthbench_hard</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-humaneval">humaneval</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-humanevalplus">humanevalplus</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-math-test-500">math_test_500</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-math-test-500-nemo">math_test_500_nemo</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mgsm">mgsm</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mgsm-aa-v2">mgsm_aa_v2</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu">mmlu</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-am">mmlu_am</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ar">mmlu_ar</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ar-lite">mmlu_ar-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-bn">mmlu_bn</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-bn-lite">mmlu_bn-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-cs">mmlu_cs</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-de">mmlu_de</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-de-lite">mmlu_de-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-el">mmlu_el</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-en">mmlu_en</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-en-lite">mmlu_en-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-es">mmlu_es</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-es-lite">mmlu_es-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-fa">mmlu_fa</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-fil">mmlu_fil</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-fr">mmlu_fr</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-fr-lite">mmlu_fr-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ha">mmlu_ha</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-he">mmlu_he</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-hi">mmlu_hi</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-hi-lite">mmlu_hi-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-id">mmlu_id</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-id-lite">mmlu_id-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ig">mmlu_ig</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-it">mmlu_it</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-it-lite">mmlu_it-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ja">mmlu_ja</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ja-lite">mmlu_ja-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ko">mmlu_ko</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ko-lite">mmlu_ko-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ky">mmlu_ky</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-llama-4">mmlu_llama_4</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-lt">mmlu_lt</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-mg">mmlu_mg</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ms">mmlu_ms</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-my-lite">mmlu_my-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ne">mmlu_ne</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-nl">mmlu_nl</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ny">mmlu_ny</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pl">mmlu_pl</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pro">mmlu_pro</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pro-aa-v2">mmlu_pro_aa_v2</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pro-llama-4">mmlu_pro_llama_4</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pt">mmlu_pt</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-pt-lite">mmlu_pt-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ro">mmlu_ro</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-ru">mmlu_ru</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-si">mmlu_si</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-sn">mmlu_sn</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-so">mmlu_so</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-sr">mmlu_sr</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-sv">mmlu_sv</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-sw">mmlu_sw</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-sw-lite">mmlu_sw-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-te">mmlu_te</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-tr">mmlu_tr</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-uk">mmlu_uk</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-vi">mmlu_vi</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-yo">mmlu_yo</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-yo-lite">mmlu_yo-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-mmlu-zh-lite">mmlu_zh-lite</a>, <a href="../task_catalog/harnesses/simple-evals.html#simple-evals-simpleqa">simpleqa</a>
* - <a href="../task_catalog/harnesses/tooltalk.html#tooltalk"><strong>tooltalk</strong></a>
  - ToolTalk task with default settings.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/tooltalk.html#tooltalk-tooltalk">tooltalk</a>
* - <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit"><strong>vlmevalkit</strong></a>
  - A benchmark for evaluating diagram understanding capabilities of large vision-language models.
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit?version=25.11)
  - {{ docker_compose_latest }}
  - <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-ai2d-judge">ai2d_judge</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-chartqa">chartqa</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-mathvista-mini">mathvista-mini</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-mmmu-judge">mmmu_judge</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-ocr-reasoning">ocr_reasoning</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-ocrbench">ocrbench</a>, <a href="../task_catalog/harnesses/vlmevalkit.html#vlmevalkit-slidevqa">slidevqa</a>
```
<!-- END_BENCH_TABLE -->

## Next Steps

- **Container Details**: Browse {ref}`nemo-evaluator-containers` for complete specifications
- **Custom Benchmarks**: Learn {ref}`framework-definition-file` for custom evaluations
