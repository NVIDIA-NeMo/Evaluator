# Conversational LLM Evaluations in Minutes with NVIDIA NeMo Evaluator Agent Skills

Running LLM evaluations should not require manually drafting long and complex YAML files. For developers, configuration overhead often becomes the bottleneck. The new **nel-assistant** agent skill enables natural language configuration of production-ready evaluations.

Built on the [NVIDIA NeMo Evaluator library](https://github.com/NVIDIA-NeMo/Evaluator), it allows developers to configure, run, and monitor evaluations directly within Cursor, or any other preferred agentic development tool. All through interaction with the agent and not manually creating YAML files or shell commands.

---

## The Problem: Configuration Overhead

Running a single LLM evaluation means making dozens of interconnected decisions:

- **Execution:** Local Docker or SLURM cluster?
- **Deployment:** vLLM, SGLang, NVIDIA NIM, NVIDIA TensorRT-LLM, or external endpoint? How many nodes?
- **Model:** What temperature? What context length? Does it use reasoning tokens?
- **Benchmarks:** Tau2-Bench, MTEB, GSM8K, AIME, GPQA, LiveCodeBench, RULER, more? All of the above?
- **Export:** Local files, CSV, Weights & Biases, or MLflow?

Each choice spawns sub-choices. Using vLLM? Configure tensor parallelism. Running reasoning models? Parse thinking tokens. Multi-node SLURM? Set up HAProxy load balancing. The result is an overhead of working through complex YAML configs that are easy to get wrong and hard to debug.

```yaml
# A typical evaluation config - 200+ lines, easy to misconfigure
execution:
  backend: slurm
  params:
    account: ???  # What's my account name again?
    partition: ???
    time_limit: "24:00:00"
deployment:
  backend: vllm
  params:
    tensor_parallel_size: ???  # 2? 4? 8?
    max_model_len: ???  # Model card says... 32K? 128K?

model:
  sampling_params:
    temperature: ???  # Was it 0.6 or 0.8?
    top_p: ???

# ... 150 more lines ...
```

This configuration burden turns a simple evaluation into unnecessary complexity.

---

## The Solution: Agent Skills for LLM Evaluation

Agent skills have become the standard way to give agents deep domain expertise. A "skill" packages instructions, scripts, and resources into portable folders that any compatible agent can discover and use.

In the coding assistants space, the ecosystem has exploded covering everything from Git workflows to API integrations. The **nel-assistant** is an agent skill that transforms Cursor, Claude Code, Codex, and other agentic developer tools into an evaluation expert. Instead of manual YAML authoring, you have a conversation. The skill researches model cards to find optimal temperatures, top_p values, and context lengths, and identifies the optimal tensor parallelism for your specific GPU setup.

**One conversation. Zero manual YAML debugging. Production-ready config.**

```
You: "Evaluate NVIDIA Nemotron-3-Nano-30B-A3B on standard benchmarks
      using vLLM locally. Export to Weights & Biases."

Agent: I'll help you set that up. First, let me check your environment...
       ✓ NeMo Evaluator 26.01 detected
       I'll need 5 quick answers to build your config:
       1. Execution: local (detected from your request)
       2. Deployment: vllm (confirmed)
       3. Export: wandb (confirmed)
       4. Model type: chat or base? [chat]
       5. Benchmark category: standard (confirmed)

       Now let me research the model card...
       ✓ Found: temperature=0.6, top_p=0.95, context=128K
       ✓ Optimal TP=8 for your 2x H100 setup

       Generated config: Nemotron-3-Nano-30B-A3B.yaml
       Ready to run!
```

---

## How It Works: Three Phases

### Phase 1: Configure

The skill starts by asking five targeted questions to establish context:

1. **Execution environment:** Local or SLURM?
2. **Deployment backend:** vLLM, SGLang, NVIDIA NIM, NVIDIA TensorRT-LLM, or external?
3. **Export destination:** None, MLflow, or Weights & Biases?
4. **Model type:** Base, chat, or reasoning?
5. **Benchmark categories:** Standard, code, math, safety, or multilingual?

From these answers, it calls:

```bash
nel skills build-config \
  --execution local \
  --deployment vllm \
  --model-type chat \
  --benchmarks standard
```

This deep-merges modular YAML templates into tested, schema-compliant fragments that compose into structurally valid configs and minimizes syntax errors. With the skill alongside, the agent never generates free-form YAML, eliminating syntax errors.

Next, the agent automatically analyzes the model card and applies optimal configuration parameters.

Give the agent a HuggingFace handle `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` or checkpoint path, and it uses WebSearch to extract:

- **Sampling params:** Temperature, top_p
- **Hardware logic:** Optimal TP/DP settings based on your GPU count
- **Reasoning config:** System prompts, payload modifiers (e.g., `enable_thinking` for o1-style models)
- **Context length:** Max model length for vLLM `--max-model-len`

Developers no longer need to search through model cards to find the right settings. The agent reads the model details and applies the correct parameters automatically.

Without the skill, this usually means jumping between Hugging Face, blog posts, and documentation. It takes time and breaks focus. With the skill, the setup happens in seconds.

### Phase 2: Validate and Refine

The skill identifies the remaining `???` values in the YAML:

- **SLURM details:** Account names, partition names, time limits
- **Export URIs:** WandB project names, MLflow tracking URIs
- **API keys:** Environment variables for deployments

You can interactively:

- **Add/remove tasks:** Browse `nel ls tasks` and pick exactly what you want
- **Override per-task settings:** "Use temperature=0 for HumanEval but 0.7 for MMLU"
- **Configure advanced scaling:** For >120B models, set up data-parallel multi-node with HAProxy load balancing
- **Add reasoning interceptors:** Strip `<think>` tokens, cache reasoning traces

### Phase 3: Run and Monitor

The agent proposes a three-tier staged rollout: **Dry run**, **Smoke test**, and **Full run**.

```bash
# 1. Dry run - validate without execution
nel run --config nemotron-3-nano.yaml --dry-run

# 2. Smoke test - 10 samples per task
nel run --config nemotron-3-nano.yaml \
  -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10

# 3. Full run
nel run --config nemotron-3-nano.yaml
```

Once submitted, progress can be monitored directly in Cursor using commands for status, detailed metrics, and live logs. You never leave your coding environment!

```
> Please, check the evaluation progress.

# Agent runs: nel status -nemotron-3-nano-20260212-143022 && nel info ...

Status: RUNNING
Progress: 3/8 tasks completed
- ✓ mmlu: 65.2% accuracy (5 hours)
- ✓ hellaswag: 78.4% accuracy (2 hours)
- ✓ arc_challenge: 53.8% accuracy (1 hour)
- ⏳ truthfulqa_mc2: 45% complete...
- ⏳ winogrande: In queue
- ⏳ gsm8k: In queue
- ⏳ humaneval: In queue
- ⏳ mbpp: In queue
```

---

## Technical Details

### Template-Based Generation

Instead of generating YAML from scratch, nel-assistant merges modular templates for execution, deployment, benchmarks, and exports. This deep merge ensures structural validity.

### Model Card Extraction Pipeline

1. Cursor or your agentic IDE fetches the HuggingFace model card via web search.
2. Extraction via regex identifies parameters and chat templates.
3. Hardware logic calculates optimal TP/DP based on model size and available GPU memory.
4. Reasoning detection checks for keywords like "reasoning" or "chain-of-thought."
5. Values are injected directly into the config YAML.

Generic LLMs hallucinate YAML syntax. They mix incompatible backends. They invent flags that don't exist.

Instead of generating YAML from scratch, `nel skills build-config` merges modular templates:

```
templates/
├── execution/
│   ├── local.yaml          # Docker execution
│   └── slurm.yaml          # SLURM execution
├── deployment/
│   ├── vllm.yaml           # vLLM backend
│   ├── sglang.yaml         # SGLang backend
│   └── nim.yaml            # NVIDIA NIM
├── benchmarks/
│   ├── reasoning.yaml      # GPQA-D, HellaSwag, SciCode, MATH, AIME
│   └── agentic.yaml        # TerminalBench, SWE-Bench
│   ├── longcontext.yaml    # AA-LCR, RULER
│   ├── instruction.yaml    # IFBench, ArenaHard
│   ├── multi-lingual.yaml  # MMLU-ProX, WMT24++
└── export/
    ├── wandb.yaml          # W&B integration
    └── mlflow.yaml         # MLflow integration
```

**Deep merge = structural validity.** You can't produce invalid YAML when you're composing pre-validated fragments.

The nel-assistant uses `build-config` to merge tested templates. Every config is structurally valid by construction. The agent composes YAML like a type-safe compiler, not a text generator.

---

## Configuration Should Not Be a Bottleneck

LLM evaluation already involves important decisions — selecting benchmarks, interpreting results, and comparing models. Configuration should support that process, not slow it down.

The nel-assistant skill makes it invisible. You describe what you want in natural language, and the agent handles the rest: researching model cards, generating configs, validating setups, staging rollouts, and monitoring progress.

**No more manually created complex YAML files. No more hunting through documentation. No more syntax errors.**

Just: *"Evaluate this model on these benchmarks."*

---

## Resources

- **GitHub:** [NVIDIA NeMo Evaluator Library](https://github.com/NVIDIA-NeMo/Evaluator)
- **Skill:** [nel-assistant](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/.claude/skills/nel-assistant/SKILL.md)

The nel-assistant skill is open-source and ships with NVIDIA NeMo Evaluator 26.01+.
Contributions welcome on GitHub!
