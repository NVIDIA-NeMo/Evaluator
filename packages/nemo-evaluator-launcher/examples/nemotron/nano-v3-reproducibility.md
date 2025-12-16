# NVIDIA Nemotron 3 Nano 30B A3B — Reproducing Model Card Evaluation Results

This tutorial demonstrates how to reproduce the evaluation results for the [**NVIDIA Nemotron 3 Nano 30B A3B**](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) model using the NeMo Evaluator Launcher.

## Overview

The Nemotron 3 Nano 30B A3B is a compact yet powerful reasoning model from NVIDIA. This configuration runs a comprehensive evaluation suite covering:

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **BFCL v3** | Function Calling | Berkeley Function Calling Leaderboard v3 |
| **BFCL v4** | Function Calling | Berkeley Function Calling Leaderboard v4 |
| **LiveCodeBench** | Coding | Real-world coding problems evaluation |
| **MMLU-Pro** | Knowledge | Multi-task language understanding (10-choice) |
| **GPQA** | Science | Graduate-level science questions |
| **AIME 2025** | Mathematics | American Invitational Mathematics Exam |
| **SciCode** | Scientific Coding | Scientific programming challenges |
| **IFBench** | Instruction Following | Instruction following benchmark |
| **HLE** | Humanity's Last Exam | Expert-level questions across domains |

The open source container on NeMo Skills packaged via NVIDIA's NeMo Evaluator SDK used for evaluations can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills?version=25.11).

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

# Required for HLE benchmark: OpenAI/Azure API key for GPT-4o judge
export JUDGE_API_KEY="your-judge-api-key"
```

> **Note:** The `JUDGE_API_KEY` is only required if you're running the HLE (Humanity's Last Exam) benchmark, which uses GPT-4o as a judge model.

### 3. HuggingFace Cache (Optional)

For faster subsequent runs, you can set a persistent cache directory:

```bash
export HF_HOME="/path/to/your/huggingface/cache"
```

---

## Running the Full Evaluation Suite

### 1. Get the Configuration

Clone the repository or download the [example config file](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/local_nvidia_nemotron_3_nano_30b_a3b.yaml):

```bash
git clone https://github.com/NVIDIA-NeMo/Evaluator.git
cd Evaluator  # or navigate to where you placed the config file
```

### 2. Run the Evaluation

```bash
nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/nemotron/local_nvidia_nemotron_3_nano_30b_a3b.yaml
# Or point to your path if you placed the file under a different location
```

### 3. Dry Run (Preview Configuration)

To preview the configuration without running the evaluation:

```bash
nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/nemotron/local_nvidia_nemotron_3_nano_30b_a3b.yaml \
  --dry-run
```

---

## Running Individual Benchmarks

You can run specific benchmarks using the `-t` flag (from the `examples/nemotron` directory):

```bash
# Run only MMLU-Pro
nemo-evaluator-launcher run --config local_nvidia_nemotron_3_nano_30b_a3b.yaml -t ns_mmlu_pro

# Run only coding benchmarks
nemo-evaluator-launcher run --config local_nvidia_nemotron_3_nano_30b_a3b.yaml -t ns_livecodebench

# Run multiple specific benchmarks
nemo-evaluator-launcher run --config local_nvidia_nemotron_3_nano_30b_a3b.yaml -t ns_gpqa -t ns_aime2025
```

### Available Task Names

| Task Name | Benchmark |
|-----------|-----------|
| `ns_bfcl_v3` | Berkeley Function Calling Leaderboard v3 |
| `ns_bfcl_v4` | Berkeley Function Calling Leaderboard v4 |
| `ns_livecodebench` | LiveCodeBench |
| `ns_mmlu_pro` | MMLU-Pro (10-choice format) |
| `ns_gpqa` | GPQA Diamond |
| `ns_aime2025` | AIME 2025 |
| `ns_scicode` | SciCode |
| `ns_ifbench` | IFBench |
| `ns_hle` | Humanity's Last Exam |

---

## Configuration Details

### Model Endpoint

The evaluation uses the NVIDIA API endpoint:

```yaml
target:
  api_endpoint:
    model_id: nvidia/nemotron-3-nano-30b-a3b
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY
```

### Default Parameters

The configuration uses the following default inference parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_new_tokens` | 131072 | Maximum tokens to generate |
| `temperature` | 0.99999 | Sampling temperature |
| `top_p` | 0.99999 | Nucleus sampling threshold |
| `parallelism` | 512 | Concurrent requests |
| `request_timeout` | 3600 | Request timeout in seconds |
| `max_retries` | 10 | Maximum retry attempts |

### Benchmark-Specific Configurations

Different benchmarks use tailored parameters:

#### BFCL (Function Calling)
- Temperature: 0.6
- Top-p: 0.95
- Client parsing disabled

#### LiveCodeBench
- 8 repeated samples (`num_repeats: 8`)
- Test split: `test_v5_2407_2412`

#### GPQA
- 8 repeated samples for pass@1
- 4-choice MCQ prompt format

#### AIME 2025
- 64 repeated samples (`num_repeats: 64`)
- Math-specific prompt template

#### HLE
- GPT-4o judge via OpenAI API
- Judge parallelism: 16
- Requires `JUDGE_API_KEY` and configuring the judge URL

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
./results_nvidia_nemotron_3_nano_30b_a3b/
├── artifacts/
│   └── <task_name>/
│       └── results.json
└── logs/
    └── stdout.log
```

---

## Customization

All examples below assume you are in the `examples/nemotron` directory.

### Override Output Directory

```bash
nemo-evaluator-launcher run \
  --config local_nvidia_nemotron_3_nano_30b_a3b.yaml \
  -o execution.output_dir=/custom/output/path
```

### Limit Samples (for Testing)

For quick testing, you can limit the number of samples:

```bash
nemo-evaluator-launcher run \
  --config local_nvidia_nemotron_3_nano_30b_a3b.yaml \
  -o evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

### Use a Different API Endpoint

If you're hosting the model locally or using a different endpoint:

```bash
nemo-evaluator-launcher run \
  --config local_nvidia_nemotron_3_nano_30b_a3b.yaml \
  -o target.api_endpoint.url=http://localhost:8000/v1/chat/completions
```

---

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure environment variables are exported in the current shell
   - Verify the key names match exactly: `NGC_API_KEY`, `HF_TOKEN`, `JUDGE_API_KEY`

2. **Timeout Errors**
   - The model generates long reasoning chains; timeouts are set to 3600s
   - Reduce `parallelism` if hitting rate limits

3. **HLE Judge Failures**
   - Verify `JUDGE_API_KEY` has access to GPT-4o
   - Check the judge endpoint URL is accessible

4. **HuggingFace Dataset Access**
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

After running the full evaluation suite, you should obtain results comparable to those reported in the NVIDIA Nemotron 3 Nano 30B A3B Model Card.

> **Important:** Due to the stochastic nature of sampling (temperature > 0) and the use of `num_repeats` for consensus-based scoring, slight variations in results are expected between runs.

---

## Base Model Evaluation

In addition to the instruct model, you can reproduce evaluation results for the base (pre-trained) models using local vLLM deployment.

### Available Base Model Configs

| Config File | Model |
|-------------|-------|
| `local_nvidia-nemotron-3-nano-30b-a3b-base.yaml` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 |
| `local_qwen3-30b-a3b-base.yaml` | Qwen/Qwen3-30B-A3B-Base |

### Base Model Benchmarks

The base model evaluation suite includes traditional pre-training benchmarks:

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **MMLU (5-shot)** | General Knowledge | Massive Multitask Language Understanding |
| **MMLU-Pro (5-shot, CoT)** | General Knowledge | Multi-task language understanding |
| **AGIEval-En (CoT)** | General Knowledge | Human-centric benchmark with chain-of-thought |
| **HumanEval (0-shot)** | Code | Code generation |
| **MBPP Sanitized (3-shot)** | Code | Python programming problems |
| **GSM8K (8-shot)** | Math | Grade school math word problems |
| **MATH (4-shot)** | Math | Competition mathematics |
| **MATH-500 (4-shot)** | Math | Subset of MATH benchmark |
| **ARC-Challenge (25-shot)** | Commonsense Understanding | AI2 Reasoning Challenge |
| **HellaSwag (10-shot)** | Commonsense Understanding | Sentence completion |
| **OpenBookQA (0-shot)** | Commonsense Understanding | Open-book question answering |
| **PIQA (0-shot)** | Commonsense Understanding | Physical intuition |
| **WinoGrande (5-shot)** | Commonsense Understanding | Coreference resolution |
| **RACE (0-shot)** | Reading Comprehension | Reading comprehension |
| **Global MMLU Lite (5-shot)** | Multilingual | Multilingual understanding |
| **MGSM (8-shot)** | Multilingual | Multilingual grade school math |

### Available Task Names (Base Models)

| Task Name | Benchmark |
|-----------|-----------|
| `adlr_mmlu` | MMLU (5-shot) |
| `adlr_mmlu_pro_5_shot_base` | MMLU-Pro (5-shot, CoT) |
| `adlr_agieval_en_cot` | AGIEval-En (CoT) |
| `adlr_humaneval_greedy` | HumanEval (greedy) |
| `adlr_mbpp_sanitized_3_shot_greedy` | MBPP Sanitized (3-shot, greedy) |
| `adlr_gsm8k_cot_8_shot` | GSM8K (8-shot) |
| `adlr_minerva_math_nemo_4_shot` | MATH (4-shot) |
| `adlr_math_500_4_shot_sampled` | MATH-500 (4-shot) |
| `adlr_arc_challenge_llama_25_shot` | ARC-Challenge (25-shot) |
| `hellaswag` | HellaSwag (10-shot) |
| `openbookqa` | OpenBookQA (0-shot) |
| `piqa` | PIQA (0-shot) |
| `adlr_race` | RACE (0-shot) |
| `adlr_winogrande_5_shot` | WinoGrande (5-shot) |
| `adlr_global_mmlu_lite_5_shot` | Global MMLU Lite (5-shot) |
| `adlr_mgsm_native_cot_8_shot` | MGSM (8-shot) |

### Running Base Model Evaluations

Base model configs use local vLLM deployment instead of an API endpoint:

```bash
cd packages/nemo-evaluator-launcher/examples/nemotron

# Run NVIDIA Nemotron base model evaluation
nemo-evaluator-launcher run --config local_nvidia-nemotron-3-nano-30b-a3b-base.yaml

# Run Qwen3 base model evaluation
nemo-evaluator-launcher run --config local_qwen3-30b-a3b-base.yaml
```

### Base Model Configuration Details

#### Deployment (vLLM)

The base models are deployed locally using vLLM:

```yaml
deployment:
  image: vllm/vllm-openai:v0.12.0
  hf_model_handle: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
  tensor_parallel_size: 1
  data_parallel_size: 1
  extra_args: "--max-model-len 262144 --mamba_ssm_cache_dtype float32 --no-enable-prefix-caching"
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
nemo-evaluator-launcher run --config local_nvidia-nemotron-3-nano-30b-a3b-base.yaml -t adlr_mmlu

# Run only coding benchmarks
nemo-evaluator-launcher run --config local_nvidia-nemotron-3-nano-30b-a3b-base.yaml -t adlr_humaneval_greedy -t adlr_mbpp_sanitized_3_shot_greedy

# Run math benchmarks
nemo-evaluator-launcher run --config local_nvidia-nemotron-3-nano-30b-a3b-base.yaml -t adlr_gsm8k_cot_8_shot -t adlr_minerva_math_nemo_4_shot -t adlr_math_500_4_shot_sampled
```

### Quick Testing (Base Models)

For quick configuration testing, uncomment `limit_samples` in the config file or override via CLI:

```bash
nemo-evaluator-launcher run \
  --config local_nvidia-nemotron-3-nano-30b-a3b-base.yaml \
  -o evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

> **Warning:** Always run full evaluations (without `limit_samples`) for actual benchmark results. Results from test runs with limited samples should never be used to compare models.

---

## License

This configuration and tutorial are provided under the Apache License 2.0. See the main repository LICENSE file for details.

