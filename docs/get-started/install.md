# Installation

## Requirements

- Python 3.10+
- An OpenAI-compatible model endpoint (NVIDIA API Catalog, vLLM, NIM, etc.)

## Install from Source

```bash
git clone https://github.com/NVIDIA-NeMo/Evaluator.git
cd Evaluator
pip install -e ".[scoring]"
```

## Install Extras

| Extra | Command | What it adds |
|-------|---------|-------------|
| `scoring` | `pip install -e ".[scoring]"` | sympy for symbolic math comparison |
| `stats` | `pip install -e ".[stats]"` | scipy for confidence intervals, McNemar significance testing, and regression analysis |
| `skills` | `pip install -e ".[skills]"` | NeMo Skills benchmark integration |
| `harbor` | `pip install -e ".[harbor]"` | Harbor agent integration (OpenHands, Terminus-2, etc.) |
| ~~`proxy`~~ | Removed | Adapter proxy is now built-in (no extra install needed) |
| `inspect` | `pip install -e ".[inspect]"` | Inspect AI log export (`inspect_ai`-compatible `EvalLog` files) |
| `ray` | `pip install -e ".[ray]"` | Ray distributed launcher |
| `harnesses` | `pip install -e ".[harnesses]"` | lm-evaluation-harness tasks |
| `export` | `pip install -e ".[export]"` | WandB and MLflow experiment tracker export |
| `docs` | `pip install -e ".[docs]"` | Sphinx, NVIDIA theme, mermaid for building docs |
| `all` | `pip install -e ".[all]"` | Everything above |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, all extras |

## Verify Installation

```bash
nel --version
nel list
```

Expected output:

```
nemo-evaluator 0.12.0

Available environments:
  drop, gpqa, gsm8k, healthbench, humaneval, math500,
  mgsm, mmlu, mmlu_pro, pinchbench, simpleqa,
  swebench-multilingual, swebench-verified, triviaqa, xstest
```

## Docker

```bash
docker build -t nemo-evaluator .
docker run nemo-evaluator nel list
```

## Next Steps

Proceed to the {doc}`quickstart` to run your first evaluation.
