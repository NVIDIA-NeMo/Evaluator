# NVIDIA Nemotron 3 Nano 30B A3B — Reproducing Model Card Evaluation Results

This tutorial demonstrates how to reproduce the evaluation results for the **NVIDIA Nemotron 3 Nano 30B A3B** model using the NeMo Evaluator Launcher.

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

---

## Prerequisites

### 1. Install the NeMo Evaluator Launcher

```bash
pip install nemo-evaluator-launcher
```

Or install from source:

```bash
cd packages/nemo-evaluator-launcher
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

### Option 1: Using the Config File Directly

Navigate to the examples directory and run:

```bash
cd packages/nemo-evaluator-launcher/examples/nemotron

nemo-evaluator-launcher run --config local_nvidia_nemotron_3_nano_30b_a3b.yaml
```

### Option 2: Using Config Path from Anywhere

```bash
nemo-evaluator-launcher run --config /path/to/examples/nemotron/local_nvidia_nemotron_3_nano_30b_a3b.yaml
```

### Option 3: Dry Run (Preview Configuration)

To preview the configuration without running the evaluation:

```bash
nemo-evaluator-launcher run --config nemotron/local_nvidia_nemotron_3_nano_30b_a3b.yaml --dry-run
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
    model_id: nvidia/nemotron-nano-3-30b-a3b
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

## License

This configuration and tutorial are provided under the Apache License 2.0. See the main repository LICENSE file for details.

