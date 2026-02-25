# Using the BYOB Skill with Coding Agents

The BYOB skill is an interactive workflow that walks you through creating a custom benchmark from scratch. It reads your data, picks a scorer, generates the benchmark file, smoke tests it, and compiles — all within a single conversation. It works with any AI coding agent that supports custom instructions.

| Agent | How the skill is loaded | How to invoke |
|-------|------------------------|---------------|
| **Claude Code** | Slash command from `.claude/commands/byob.md` | Type `/byob` in any session within the NeMo Evaluator repo |
| **Codex** | Agent instructions from `examples/byob/AGENTS.md` | Ask Codex to "create a BYOB benchmark" in the repo |

## Prerequisites

- An AI coding agent ([Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex](https://openai.com/index/codex/), or similar)
- NeMo Evaluator installed: `pip install -e .` from `packages/nemo-evaluator`
- A dataset file (JSONL, CSV, JSON array, or HuggingFace `hf://` URI)

## How to invoke

### Claude Code

In any Claude Code session within the NeMo Evaluator repo, type:

```
/byob
```

You can optionally provide a description upfront to skip the opening questions:

```
/byob I have a yes/no QA dataset at /data/qa.jsonl with fields passage, question, answer
```

### Codex

Open a Codex session in the NeMo Evaluator repo. Codex automatically picks up the instructions from `examples/byob/AGENTS.md`. Ask it to create a benchmark in natural language:

```
Create a BYOB benchmark for my yes/no QA dataset at /data/qa.jsonl with fields passage, question, answer
```

### Other coding agents

Any coding agent that can read files and run shell commands can follow the BYOB workflow. Point the agent at `examples/byob/AGENTS.md` for the full instruction set, or at the templates in `examples/byob/templates/` as starting points.

## The 5-step workflow

The skill guides you through five steps, showing progress as `[Step N/5: Name]`.

### Step 1 — Understand

The agent asks what kind of benchmark you want. You describe it in plain language.

```
┌─────────────────────────────────────────────────────────────┐
│ You: Create a BYOB benchmark                                │
│                                                             │
│ Agent: Welcome to BYOB! I'll help you create a custom       │
│ evaluation benchmark in 5 steps.                            │
│                                                             │
│ [Step 1/5: Understand]                                      │
│ What kind of benchmark do you want to create? For example:  │
│ - "AIME 2025 math problems"                                 │
│ - "safety benchmark from my CSV at data.csv"                │
│ - "yes/no QA over passages"                                 │
│                                                             │
│ You: I have a yes/no QA dataset. The model reads a passage  │
│ and answers yes or no. My data is at ./qa_data.jsonl        │
└─────────────────────────────────────────────────────────────┘
```

### Step 2 — Data

The agent reads your file, shows the first 3 rows, confirms the schema, and converts to JSONL if needed.

```
┌─────────────────────────────────────────────────────────────┐
│ Agent: [Step 2/5: Data]                                     │
│ Let me read your dataset.                                   │
│                                                             │
│ Found 500 rows with fields:                                 │
│   - passage (str)                                           │
│   - question (str)                                          │
│   - answer (str)                                            │
│                                                             │
│ First 3 rows:                                               │
│ 1. passage: "Persian, also known as Farsi..."               │
│    question: "do iran and afghanistan speak the same..."    │
│    answer: "true"                                           │
│ 2. passage: "Ghost is a 1990 American romantic..."          │
│    question: "is the movie ghost based on a true story"     │
│    answer: "false"                                          │
│ 3. passage: "The Amazon River in South America..."          │
│    question: "is the amazon river the largest river..."     │
│    answer: "true"                                           │
│                                                             │
│ Target field (ground truth): "answer"                       │
│ Does this look correct?                                     │
└─────────────────────────────────────────────────────────────┘
```

### Step 3 — Prompt

The agent generates a prompt template with `{field}` placeholders matching your dataset.

```
┌─────────────────────────────────────────────────────────────┐
│ Agent: [Step 3/5: Prompt]                                   │
│ Here's the prompt template I'll use:                        │
│                                                             │
│   Answer yes or no.                                         │
│                                                             │
│   {passage}                                                 │
│                                                             │
│   {question}                                                │
│   Answer:                                                   │
│                                                             │
│ Each {field} will be filled from your dataset rows.         │
│ Want to adjust the wording?                                 │
└─────────────────────────────────────────────────────────────┘
```

### Step 4 — Score

The agent picks a built-in scorer if possible, otherwise generates a custom one. It always smoke tests with synthetic inputs before proceeding.

```
┌─────────────────────────────────────────────────────────────┐
│ Agent: [Step 4/5: Score]                                    │
│ For yes/no QA, I'll write a custom scorer that extracts     │
│ yes/no from the model response and compares to the target.  │
│                                                             │
│ Smoke testing scorer...                                     │
│  ✓ scorer(ScorerInput("Yes, correct", "true", {})) → T     │
│  ✓ scorer(ScorerInput("No", "false", {}))           → T    │
│  ✓ scorer(ScorerInput("I'm not sure", "true", {}))  → F    │
│ All smoke tests passed.                                     │
└─────────────────────────────────────────────────────────────┘
```

### Step 5 — Ship

The agent writes the benchmark file, compiles it, and gives you the run command.

```
┌─────────────────────────────────────────────────────────────┐
│ Agent: [Step 5/5: Ship]                                     │
│ Writing benchmark to ./my_qa_benchmark.py:                  │
│                                                             │
│   from nemo_evaluator.contrib.byob import (                 │
│       benchmark, scorer, ScorerInput,                       │
│   )                                                         │
│                                                             │
│   @benchmark(                                               │
│       name="my-qa",                                         │
│       dataset="/abs/path/to/qa_data.jsonl",                 │
│       prompt="Answer yes or no.\n\n{passage}\n\n..."        │
│       target_field="answer",                                │
│   )                                                         │
│   @scorer                                                   │
│   def qa_scorer(sample: ScorerInput) -> dict:               │
│       ...                                                   │
│                                                             │
│ Compiling...                                                │
│ ✓ Compiled and installed byob_my_qa                         │
│                                                             │
│ To run:                                                     │
│   nemo-evaluator run_eval \                                 │
│     --eval_type byob_my_qa.my-qa \                          │
│     --target.api_endpoint.url http://localhost:8000 \        │
│     --target.api_endpoint.model_id my-model                 │
└─────────────────────────────────────────────────────────────┘
```

## Shortcut: provide everything upfront

If you already know your data, target field, and scoring approach, give it all in one shot. The agent skips the questions and generates directly:

**Claude Code:**
```
/byob I have a multiple-choice dataset at /data/mmlu.jsonl with fields question, A, B, C, D and target field "answer" containing a letter A-D. Score by extracting the first letter from the response.
```

**Codex / other agents:**
```
Create a BYOB benchmark from my multiple-choice dataset at /data/mmlu.jsonl with fields question, A, B, C, D and target field "answer" containing a letter A-D. Score by extracting the first letter from the response.
```

## What the skill produces

After the workflow completes you'll have:

| Artifact | Location |
|----------|----------|
| Benchmark source file | `./<name>_benchmark.py` in your working directory |
| Compiled package | `~/.nemo-evaluator/byob_packages/byob_<name>/` |
| Dataset copy | Frozen inside the compiled package |
| Config | `config.yaml` generated inside the package |
| pip install | Auto-installed via `pip install -e` (immediately discoverable) |

## Examples of things you can ask

| Request | What happens |
|---------|-------------|
| `/byob` (Claude Code) or "Create a BYOB benchmark" (Codex) | Opens the guided 5-step wizard |
| "I have a CSV at data.csv" | Starts at step 2 — reads and converts the CSV |
| "AIME 2025 math problems" | Recognizes math type, uses number-extraction scorer |
| "safety benchmark, data at prompts.jsonl" | Uses a safety prompt pattern (direct pass-through) |
| "exact match QA, data at qa.jsonl, target is answer" | Picks built-in `exact_match`, skips most questions |
| "Judge-based evaluation of summaries" | Uses `judge_score()` with `likert_5` template |

## Available built-in scorers

The skill prefers built-in scorers when possible. These are chosen automatically based on your description:

| Scorer | When the skill picks it |
|--------|------------------------|
| `exact_match` | You say "exact match", "string equality", etc. |
| `contains` | You say "target appears in response", "substring" |
| `f1_token` | You say "partial credit", "token overlap", "F1" |
| `regex_match` | You say "regex", "pattern match" |
| `bleu` | You say "BLEU score", "translation quality" |
| `rouge` | You say "ROUGE score", "summarization quality" |
| `retrieval_metrics` | You say "retrieval quality", "RAG evaluation", "precision@k" |
| `judge_score()` | You say "subjective", "judge", "quality rating", "LLM-as-judge" |
| Custom | Math extraction, letter extraction, yes/no, or any described custom logic |

## Troubleshooting

### Skill not found (Claude Code)

Make sure you're running Claude Code from within the NeMo Evaluator repo (or a parent that contains it). The skill is registered at:

```
packages/nemo-evaluator/.claude/commands/byob.md
```

### Agent doesn't know BYOB workflow (Codex / other agents)

Point the agent to the instruction file:

```
Read examples/byob/AGENTS.md and follow the BYOB workflow to create a benchmark.
```

Alternatively, point it at a template as a starting point:

```
Use examples/byob/templates/math_reasoning.py as a reference to create a benchmark for my dataset.
```

### Compilation fails

The agent will attempt up to 2 auto-recovery cycles. If it still fails, it asks you for help. Common issues:

- **"No benchmarks found"** — Missing `@benchmark` or `@scorer` decorator, or wrong decorator order
- **"KeyError: '{field}'"** — Prompt placeholder doesn't match a field in the dataset
- **"Module not found: nemo_evaluator"** — Run `pip install -e packages/nemo-evaluator`

### Scorer smoke test fails

If the scorer doesn't return a dict with bool/float values, the agent stops and shows the error. Fix the scorer function and tell the agent to retry.
