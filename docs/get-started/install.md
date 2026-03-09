# Installation

## Requirements

- Python 3.10+
- An OpenAI-compatible model endpoint (NVIDIA API Catalog, vLLM, NIM, etc.)

## Install from Source

```bash
git clone https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nemo-evaluator-next.git
cd nemo-evaluator-next
pip install -e ".[scoring]"
```

## Install Extras

| Extra | Command | What it adds |
|-------|---------|-------------|
| `scoring` | `pip install -e ".[scoring]"` | sympy for symbolic math comparison |
| `stats` | `pip install -e ".[stats]"` | scipy for confidence intervals and Mann-Whitney U p-values in regression |
| `pi` | `pip install -e ".[pi]"` | Prime Intellect `verifiers` integration |
| `skills` | `pip install -e ".[skills]"` | NeMo Skills benchmark integration |
| `harnesses` | `pip install -e ".[harnesses]"` | lm-evaluation-harness tasks |
| `all` | `pip install -e ".[all]"` | Everything above |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, all extras |

## Verify Installation

```bash
nel --version
nel list
```

Expected output:

```
nemo-evaluator 0.5.0

Available environments:
  drop, gpqa, gsm8k, healthbench, humaneval, math500,
  mgsm, mmlu, mmlu_pro, simpleqa, triviaqa
```

## Docker

```bash
docker build -t nemo-evaluator .
docker run nemo-evaluator nel list
```

## Next Steps

Proceed to the {doc}`quickstart` to run your first evaluation.
