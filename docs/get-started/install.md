# Installation

## Requirements

- Python 3.10+
- An OpenAI-compatible model endpoint (NVIDIA API Catalog, vLLM, TGI, etc.)

## Install from Source

```bash
git clone https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nemo-evaluator-next.git
cd nemo-evaluator-next
pip install -e ".[scoring]"
```

## Optional Extras

| Extra | Command | What it adds |
|-------|---------|-------------|
| `scoring` | `pip install -e ".[scoring]"` | sympy for symbolic math comparison |
| `stats` | `pip install -e ".[stats]"` | scipy for normal confidence intervals |
| `ray` | `pip install -e ".[ray]"` | Ray distributed evaluation |
| `pi` | `pip install -e ".[pi]"` | Prime Intellect `verifiers` integration |
| `all` | `pip install -e ".[all]"` | Everything above |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, all extras |

## Verify Installation

```bash
nel --version
nel list-environments
```

Expected output:

```
Available environments:
  - gsm8k
  - triviaqa
```

## Docker

```bash
docker build -t nemo-evaluator .
docker run nemo-evaluator list-environments
```

## Next Steps

Proceed to the {doc}`quickstart` to run your first evaluation.
