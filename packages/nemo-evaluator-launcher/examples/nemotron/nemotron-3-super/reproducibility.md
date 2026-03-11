# NVIDIA Nemotron 3 Super 120B A12B — Reproducing Model Card Evaluation Results

This tutorial demonstrates how to reproduce the evaluation results for the [**NVIDIA Nemotron 3 Super 120B A12B**](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) model using the NeMo Evaluator Launcher.

## Overview

The Nemotron 3 Super 120B A12B is a powerful reasoning model from NVIDIA. This configuration runs a comprehensive evaluation suite covering:

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **GPQA** | Science | Graduate-level science questions |
| **HLE** | Humanity's Last Exam | Expert-level questions across domains |
| **MMLU-Pro** | Knowledge | Multi-task language understanding (10-choice) |
| **AIME 2025** | Mathematics | American Invitational Mathematics Exam 2025 |
| **AIME 2026** | Mathematics | American Invitational Mathematics Exam 2026 |
| **HMMT Feb 2025** | Mathematics | Harvard-MIT Mathematics Tournament |
| **Math 500** | Mathematics | MATH benchmark (500 problems) |
| **CritPt** | Physics | Critical Point evaluation |
| **SciCode** | Scientific Coding | Scientific programming challenges |
| **LiveCodeBench** | Coding | Real-world coding problems evaluation |
| **IFBench** | Instruction Following | Instruction following benchmark |
| **BFCL v4** | Function Calling | Berkeley Function Calling Leaderboard v4 |
| **Tau2 Bench Telecom** | Telecom / Tool Use | Tau2 benchmark (telecom domain) |
| **Tau2 Bench Retail** | Retail / Tool Use | Tau2 benchmark (retail domain) |
| **Tau2 Bench Airline** | Airline / Tool Use | Tau2 benchmark (airline domain) |
| **Arena Hard v2** | Chat Preference | Arena Hard v2 benchmark |
| **AA LCR** | Long-context | Artificial Analysis Long-Context-Reasoning benchmark |
| **MMLU-Pro X** | Knowledge | MMLU-Pro extended multilingual variant |
| **AA-Omniscience** | Knowledge / hallucination| AA Omniscience benchmark with Gemini-3-Flash-Preview judge |
| **WMT24++** | Translation | WMT24 translation benchmark scored with XCOMET-XXL |

The open source container on NeMo Skills packaged via NVIDIA's NeMo Evaluator SDK used for evaluations can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo-skills/tags?version=latest).

---

## Prerequisites

### 1. Install the NeMo Evaluator Launcher

```bash
pip install nemo-evaluator-launcher
```

Or install from source:

```bash
git clone https://github.com/NVIDIA-NeMo/Evaluator.git
cd Evaluator/packages/nemo-evaluator-launcher
pip install -e .
```

### 2. Required API Keys

Set the following environment variables before running the evaluation:

```bash
# Required: NGC API key for accessing the model endpoint
export NGC_API_KEY="your-ngc-api-key"

# Required: HuggingFace token for accessing datasets
export HF_TOKEN="your-huggingface-token"

# Required for HLE and Multi-Challenge benchmarks: OpenAI/Azure API key for GPT-4o judge
export JUDGE_API_KEY="your-judge-api-key"

# Required for Omniscience benchmark: Gemini API key for Gemini-3-Flash-Preview judge
export GEMINI_API_KEY="your-gemini-api-key"

# Required for AA LCR benchmark: API key with access to non-reasoning Qwen3-235B-A22B judge
export QWEN_API_KEY="your-qwen-api-key"

# Required for CritPt benchmark
export ARTIFICIAL_ANALYSIS_API_KEY="your-artificial-analysis-api-key"
```

> **Note:** Not all API keys are required for every benchmark. `JUDGE_API_KEY` is needed for HLE; `GEMINI_API_KEY` for AA-Omniscience; `QWEN_API_KEY` for AA LCR and tau2-bench; and `ARTIFICIAL_ANALYSIS_API_KEY` for CritPt. The `NGC_API_KEY` is also reused as the `JUDGE_API_KEY` for AA Math Test 500.

### 3. HuggingFace Cache (Optional)

For faster subsequent runs, you can set a persistent cache directory:

```bash
export HF_HOME="/path/to/your/huggingface/cache"
```

---

## Running the Full Evaluation Suite

### 1. Get the Configuration

The config files are located in this directory:

| Config File | Purpose |
|-------------|---------|
| `local_nemotron-3-super-120b-a12b.yaml` | Main evaluation suite |
| `local_nemotron-3-super-120b-a12b_tools.yaml` | Tool usage evaluation |
| `local_nemotron-3-super-120b-a12b_low_budget.yaml` | Low-budget / low-effort thinking evaluation |

### 2. Run the Evaluation

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml
```

### 3. Dry Run (Preview Configuration)

To preview the configuration without running the evaluation:

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml \
  --dry-run
```

---

## Running Individual Benchmarks

You can run specific benchmarks using the `-t` flag:

```bash
# Run only MMLU-Pro
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b.yaml -t nemo_skills.ns_mmlu_pro

# Run only coding benchmarks
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b.yaml -t ns_livecodebench

# Run multiple specific benchmarks
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b.yaml -t nemo_skills.ns_gpqa -t nemo_skills.ns_aime2025
```

### Available Task Names

| Task Name | Benchmark |
|-----------|-----------|
| `nemo_skills.ns_gpqa` | GPQA Diamond (4-choice MCQ) |
| `ns_hle` | Humanity's Last Exam |
| `nemo_skills.ns_mmlu_pro` | MMLU-Pro (10-choice format) |
| `nemo_skills.ns_aime2025` | AIME 2025 |
| `nemo_skills.ns_aime2026` | AIME 2026 |
| `ns_hmmt_feb2025` | HMMT Feb 2025 |
| `AA_math_test_500` | Math 500 |
| `nemo_skills.ns_critpt` | CritPt |
| `ns_scicode` | SciCode |
| `ns_livecodebench_v5` | LiveCodeBench v5 (Jul–Dec 2024) |
| `ns_livecodebench` | LiveCodeBench v6 (Aug 2024–May 2025) |
| `ns_ifbench` | IFBench |
| `nemo_skills.ns_bfcl_v4` | BFCL v4 |
| `tau2_bench_telecom` | Tau2 Bench Telecom |
| `tau2_bench_retail` | Tau2 Bench Retail |
| `tau2_bench_airline` | Tau2 Bench Airline |
| `ns_arena_hard_v2` | Arena Hard v2 |
| `ns_aa_lcr` | AA LCR |
| `ns_omniscience` | AA-Omniscience |
| `ns_mmlu_prox` | MMLU-Pro X |
| `ns_wmt24pp_comet` | WMT24++ (COMET score) |

---

## Configuration Details

### Model Endpoint

The evaluation uses the NVIDIA API endpoint:

```yaml
target:
  api_endpoint:
    model_id: nvidia/nemotron-super-120b-a12b
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY
```

### Default Parameters

The configuration uses the following default inference parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_new_tokens` | 131072 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `parallelism` | 1 | Concurrent requests |
| `request_timeout` | 3600 | Request timeout in seconds |
| `max_retries` | 10 | Maximum retry attempts |

The adapter config enables thinking mode and response logging:

```yaml
adapter_config:
  params_to_add: {"chat_template_kwargs": {"enable_thinking": true}, "skip_special_tokens": false}
  use_caching: true
  tracking_requests_stats: true
  log_failed_requests: true
  use_request_logging: true
  max_logged_requests: 10
  use_response_logging: true
  max_logged_responses: 10
```

### Benchmark-Specific Configurations

Different benchmarks use tailored parameters:

#### GPQA
- 8 repeated samples for pass@1
- 4-choice MCQ prompt format (`eval/aai/mcq-4choices`)

#### HLE
- GPT-4o judge via OpenAI API
- Judge parallelism: 16
- Requires `JUDGE_API_KEY` and configuring the judge URL

#### MMLU-Pro
- 1 repeat
- 10-choice boxed prompt format (`eval/aai/mcq-10choices-boxed`)

#### AIME 2025
- 64 repeated samples (`num_repeats: 64`)
- Math-specific prompt template (`math-oai.yaml`)

#### AIME 2026
- 64 repeated samples
- Math-specific prompt template + `tokens_to_generate=null`

#### HMMT Feb 2025
- 64 repeated samples

#### Math 500
- 5 samples per item (`n_samples: 5`)
- Uses `NGC_API_KEY` as `JUDGE_API_KEY`

#### CritPt
- 3 repeated samples
- Requires `ARTIFICIAL_ANALYSIS_API_KEY`

#### SciCode
- 8 repeated samples

#### LiveCodeBench v5 / v6
- 8 repeated samples
- v5 split: `test_v5_2407_2412`; v6 split: `test_v6_2408_2505`

#### IFBench
- 8 repeated samples

#### BFCL v4 (Function Calling)
- `max_new_tokens`: 8192
- `parallelism`: 128
- Client parsing disabled

#### Tau2 Bench (Telecom / Retail / Airline)
- 8 samples per item
- Caching enabled; `skip_failed_samples: true`
- `use_reasoning: true` in adapter config
- Requires `QWEN_API_KEY` and configuring the user URL/model_id

#### Arena Hard v2
- Default parameters (no task-specific overrides)

#### AA LCR
- 16 repeated samples
- Judge: non-reasoning Qwen3-235B-A22B
- Requires `QWEN_API_KEY` and configuring the judge URL/model_id

#### Omniscience
- Reasoning parsing disabled (`++parse_reasoning=False`)
- Judge: Gemini-3-Flash-Preview
- Requires `GEMINI_API_KEY` and configuring the judge URL

#### MMLU-Pro X
- 1 repeat

#### WMT24++ (COMET)
- Scored using XCOMET-XXL
- Requires downloading the [XCOMET-XXL model](https://huggingface.co/Unbabel/XCOMET-XXL) and setting the `model_path` in the config

---

## Monitoring and Results

### Check Job Status

```bash
nemo-evaluator-launcher status
```

### View Logs

```bash
# Stream logs for a specific job
nemo-evaluator-launcher logs <job-id>
```

### Results Location

Results are saved to the output directory specified in the config:

```
./results_nvidia_nemotron_3_super_120b_a12b/
├── artifacts/
│   └── <task_name>/
│       └── results.json
└── logs/
    └── stdout.log
```

---

## Customization

### Override Output Directory

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml \
  -o execution.output_dir=/custom/output/path
```

### Limit Samples (for Testing)

For quick testing, you can limit the number of samples:

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml \
  -o evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

### Use a Different API Endpoint

If you're hosting the model locally or using a different endpoint:

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml \
  -o target.api_endpoint.url=http://localhost:8000/v1/chat/completions
```

---

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure environment variables are exported in the current shell
   - Verify the key names match exactly: `NGC_API_KEY`, `HF_TOKEN`, `JUDGE_API_KEY`, `GEMINI_API_KEY`, `QWEN_API_KEY`, `ARTIFICIAL_ANALYSIS_API_KEY`

2. **Timeout Errors**
   - The model generates long reasoning chains; timeouts are set to 3600s
   - Reduce `parallelism` if hitting rate limits

3. **HLE Judge Failures**
   - Verify `JUDGE_API_KEY` has access to GPT-4o
   - Check the judge endpoint URL is accessible

4. **WMT24++ COMET Scoring**
   - Download the XCOMET-XXL checkpoint from HuggingFace and update `model_path` in the config

5. **HuggingFace Dataset Access**
   - Some datasets require accepting terms on HuggingFace Hub
   - Ensure your `HF_TOKEN` has appropriate permissions

### Getting Help

```bash
# View all available commands
nemo-evaluator-launcher --help

# View run command options
nemo-evaluator-launcher run --help

# List available evaluation tasks
nemo-evaluator-launcher ls tasks
```

---

## Expected Results

After running the full evaluation suite, you should obtain results comparable to those reported in the NVIDIA Nemotron 3 Super 120B A12B Model Card.

> **Important:** Due to the stochastic nature of sampling (temperature > 0) and the use of `num_repeats` for consensus-based scoring, slight variations in results are expected between runs.

---

## Tool Usage Evaluation

The Nemotron 3 Super 120B A12B model supports tool usage, allowing it to call a Python interpreter to solve math and science problems step by step.

### Tool Usage Config

| Config File | Model |
|-------------|-------|
| `local_nemotron-3-super-120b-a12b_tools.yaml` | nvidia/nemotron-3-super-120b-a12b |

### Tool Usage Benchmarks

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **GPQA (tools)** | Science + Tools | GPQA with Python code execution |
| **HLE (tools)** | Humanity's Last Exam + Tools | HLE with Python code execution and GPT-4o judge |
| **AIME 2025 (tools)** | Mathematics + Tools | AIME with Python code execution |
| **AIME 2026 (tools)** | Mathematics + Tools | AIME 2026 with Python code execution |
| **HMMT Feb 2025 (tools)** | Mathematics + Tools | HMMT with Python code execution |

### Available Task Names (Tool Usage)

| Task Name | Benchmark |
|-----------|-----------|
| `nemo_skills.ns_gpqa` | GPQA Diamond (with sandbox + Python tool) |
| `ns_hle` | HLE (with sandbox + Python tool + GPT-4o judge) |
| `nemo_skills.ns_aime2025` | AIME 2025 (with sandbox + Python tool) |
| `nemo_skills.ns_aime2026` | AIME 2026 (with sandbox + Python tool) |
| `ns_hmmt_feb2025` | HMMT Feb 2025 (with sandbox + Python tool) |

### Additional API Keys for Tool Usage

```bash
# Required for HLE (tools): OpenAI/Azure API key for GPT-4o judge
export JUDGE_API_KEY="your-judge-api-key"
```

### Running Tool Usage Evaluations

```bash
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b_tools.yaml

# Run only AIME 2025 with tools
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b_tools.yaml -t nemo_skills.ns_aime2025

# Run only GPQA with tools
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b_tools.yaml -t nemo_skills.ns_gpqa

```

### Tool Usage Configuration Details

Tool usage is enabled per-task via these extra parameters:

```yaml
extra:
  use_sandbox: true
  args: "++inference.tokens_to_generate=null ++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool]"
```

The model generates tool calls (Python code), which are executed in a sandboxed environment, and the results are fed back to the model for further reasoning.

---

## Low-Budget / Low-Effort Thinking Evaluation

A separate configuration runs a subset of benchmarks with low-effort thinking mode enabled, useful for evaluating the model's performance under constrained compute budgets.

### Low-Budget Config

| Config File | Model |
|-------------|-------|
| `local_nemotron-3-super-120b-a12b_low_budget.yaml` | nvidia/nemotron-3-super-120b-a12b |

### Low-Budget Benchmarks

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **AIME 2025** | Mathematics | AIME 2025 with low-effort thinking |
| **Tau2 Bench Telecom** | Telecom / Tool Use | Tau2 telecom with low-effort thinking |
| **LiveCodeBench v6** | Coding | LiveCodeBench with low-effort thinking |

### Running Low-Budget Evaluations

```bash
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b_low_budget.yaml

# Run only AIME 2025 (low effort)
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b_low_budget.yaml -t nemo_skills.ns_aime2025
```

### Low-Budget Configuration Details

The low-budget config overrides the adapter to enable low-effort thinking:

```yaml
adapter_config:
  params_to_add: {"chat_template_kwargs": {"enable_thinking": true, "low_effort": true}}
```

---

--

## Base Model Evaluation

In addition to the instruct model, you can reproduce evaluation results for the base (pre-trained) models using local vLLM deployment.

### Available Base Model Configs

| Config File | Model |
|-------------|-------|
| `local_nemotron-3-super-120b-a12b-base.yaml` | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base |

### Base Model Benchmarks

The base model evaluation suite includes traditional pre-training benchmarks:

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **MMLU (5-shot)** | General Knowledge | Massive Multitask Language Understanding |
| **MMLU-Pro (5-shot, CoT)** | General Knowledge | Multi-task language understanding |
| **AGIEval-En (CoT)** | General Knowledge | Human-centric benchmark with chain-of-thought |
| **GPQA-Diamond (CoT)** | General Knowledge | Graduate-Level Google-Proof Q&A |
| **GSM8K (8-shot)** | Math | Grade school math word problems |
| **MATH (4-shot)** | Math | Competition mathematics |
| **MATH-500 (4-shot)** | Math | Subset of MATH benchmark |
| **HumanEval (0-shot)** | Code | Code generation |
| **MBPP Sanitized (3-shot)** | Code | Python programming problems |
| **ARC-Challenge (25-shot)** | Commonsense Understanding | AI2 Reasoning Challenge |
| **HellaSwag (10-shot)** | Commonsense Understanding | Sentence completion |
| **OpenBookQA (0-shot)** | Commonsense Understanding | Open-book question answering |
| **PIQA (0-shot)** | Commonsense Understanding | Physical intuition |
| **WinoGrande (5-shot)** | Commonsense Understanding | Coreference resolution |
| **RACE (0-shot)** | Reading Comprehension | Reading comprehension |
| **Global MMLU Lite (5-shot)** | Multilingual | Multilingual understanding |
| **MGSM (8-shot)** | Multilingual | Multilingual grade school math |
| **RULER 64k** | Long Context | Long context benchmarking |
| **RULER 128k** | Long Context | Long context benchmarking |
| **RULER 256k** | Long Context | Long context benchmarking |
| **RULER 512k** | Long Context | Long context benchmarking |
| **RULER 1m** | Long Context | Long context benchmarking |


### Available Task Names (Base Models)

| Task Name | Benchmark |
|-----------|-----------|
| `adlr_mmlu` | MMLU (5-shot) |
| `adlr_mmlu_pro_5_shot_base` | MMLU-Pro (5-shot, CoT) |
| `adlr_agieval_en_cot` | AGIEval-En (3/5-shot, CoT) |
| `adlr_gpqa_diamond_cot_5_shot` | GPQA-Diamond (5-shot, CoT) |
| `adlr_gsm8k_cot_8_shot` | GSM8K (8-shot) |
| `adlr_minerva_math_nemo_4_shot` | MATH (4-shot) |
| `adlr_math_500_4_shot_sampled` | MATH-500 (4-shot) |
| `adlr_humaneval_greedy` | HumanEval (0-shot, pass@1, n=32) |
| `adlr_mbpp_sanitized_3_shot_greedy` | MBPP Sanitized (3-shot, pass@1, n=32) |
| `adlr_arc_challenge_llama_25_shot` | ARC-Challenge (25-shot) |
| `hellaswag` | HellaSwag (10-shot) |
| `openbookqa` | OpenBookQA (0-shot) |
| `piqa` | PIQA (0-shot) |
| `adlr_winogrande_5_shot` | WinoGrande (5-shot) |
| `adlr_race` | RACE (0-shot) |
| `adlr_global_mmlu_lite_5_shot` | Global MMLU Lite (5-shot) |
| `adlr_mgsm_native_cot_8_shot` | MGSM (8-shot) |
| `ruler-64k-completions` | RULER 64k |
| `ruler-128k-completions` | RULER 128k |
| `ruler-256k-completions` | RULER 256k |
| `ruler-512k-completions` | RULER 512k |
| `ruler-1m-completions` | RULER 1m |
### Running Base Model Evaluations

Base model configs use local vLLM deployment instead of an API endpoint:

```bash
cd packages/nemo-evaluator-launcher/examples/nemotron

# Run NVIDIA Nemotron base model evaluation
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b-base.yaml


### Base Model Configuration Details

#### Deployment (vLLM)

The base models are deployed locally using vLLM:

```yaml
deployment:
  image: vllm/vllm-openai:v0.13.0
  hf_model_handle: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base
  tensor_parallel_size: 8
  data_parallel_size: 1
```

#### Inference Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_retries` | 5 | Number of retries for API requests |
| `request_timeout` | 360 | Request timeout in seconds |
| `parallelism` | 4 | Concurrent requests |

### Running Individual Base Model Benchmarks

```bash
# Run only MMLU
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b-base.yaml -t adlr_mmlu

# Run only coding benchmarks
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b-base.yaml -t adlr_humaneval_greedy -t adlr_mbpp_sanitized_3_shot_greedy

# Run math benchmarks
nemo-evaluator-launcher run --config local_nemotron-3-super-120b-a12b-base.yaml -t adlr_gsm8k_cot_8_shot -t adlr_minerva_math_nemo_4_shot -t adlr_math_500_4_shot_sampled
```

### Quick Testing (Base Models)

For quick configuration testing, uncomment `limit_samples` in the config file or override via CLI:

```bash
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b-base.yaml \
  -o evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

> **Warning:** Always run full evaluations (without `limit_samples`) for actual benchmark results. Results from test runs with limited samples should never be used to compare models.

---

## License

This configuration and tutorial are provided under the Apache License 2.0. See the main repository LICENSE file for details.
