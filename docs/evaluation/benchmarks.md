# Built-in Benchmarks

All 11 built-in benchmarks are defined with `@benchmark` + `@scorer` in `src/nemo_evaluator/benchmarks/`.

## Quick Reference

| Benchmark | Command | Scoring | Type |
|-----------|---------|---------|------|
| MMLU | `nel run --env mmlu` | `multichoice_regex` | Multichoice (4-way) |
| MMLU-Pro | `nel run --env mmlu_pro` | `multichoice_regex` | Multichoice (10-way) |
| MATH-500 | `nel run --env math500` | `answer_line` | Math |
| GPQA Diamond | `nel run --env gpqa` | `multichoice_regex` | Multichoice (shuffled) |
| GSM8K | `nel run --env gsm8k` | `numeric_match` | Math reasoning |
| DROP | `nel run --env drop` | `fuzzy_match` | Reading comprehension |
| MGSM | `nel run --env mgsm` | `numeric_match` | Multilingual math |
| TriviaQA | `nel run --env triviaqa` | `fuzzy_match` | Factual QA |
| HumanEval | `nel run --env humaneval` | `code_sandbox` | Code generation (Docker) |
| SimpleQA | `nel run --env simpleqa` | `needs_judge` | Factuality (LLM judge) |
| HealthBench | `nel run --env healthbench` | `needs_judge` | Health (LLM judge) |

## Extended Environments

Beyond the 11 built-in benchmarks, NEL resolves additional environment types via URI schemes and namespace prefixes:

| Syntax | Source | Example |
|--------|--------|---------|
| `nel run --env <name>` | Built-in registry | `nel run --env mmlu` |
| `nel run --env lm-eval/<task>` | lm-evaluation-harness | `nel run --env lm-eval/aime25` |
| `nel run --env skills://<name>` | NeMo Skills | `nel run --env skills://mmlu-pro` |
| `nel run --env gym://<host:port>` | Remote Gym server | `nel run --env gym://localhost:9090` |
| `nel run --env pi://<name>` | Prime Intellect verifiers | `nel run --env pi://simpleqa` |

## Benchmark Details

### MMLU

Massive Multitask Language Understanding -- 14K 4-choice questions across 57 subjects.

- **Dataset:** `hf://cais/mmlu?config=all&split=test`
- **Scorer:** `multichoice_regex` -- extracts letter (A-D) from "Answer: X" pattern
- **`prepare_row`:** Unpacks `choices` list into `A`, `B`, `C`, `D` fields; maps numeric `answer` to letter

### MMLU-Pro

10-choice variant of MMLU with harder distractors.

- **Dataset:** `hf://TIGER-Lab/MMLU-Pro?split=test`
- **Scorer:** `multichoice_regex` with extended pattern `[A-J]`
- **`prepare_row`:** Pads choices to 10 slots

### MATH-500

500 competition-level math problems.

- **Dataset:** `hf://HuggingFaceH4/MATH-500?split=test`
- **Scorer:** `answer_line` -- extracts answer after "Answer:" and normalizes

### GPQA Diamond

Graduate-level science questions with shuffled answer choices.

- **Dataset:** `hf://Idavidrein/gpqa?config=gpqa_diamond&split=train`
- **Scorer:** `multichoice_regex`
- **`prepare_row`:** Shuffles choices with seeded RNG to prevent position bias

### GSM8K

1,319 grade-school math problems requiring multi-step reasoning.

- **Dataset:** `hf://openai/gsm8k?split=test`
- **Scorer:** `numeric_match` -- extracts last number from response

### DROP

Reading comprehension with discrete reasoning (counting, sorting, arithmetic).

- **Dataset:** `hf://ucinlp/drop?split=validation`
- **Scorer:** `fuzzy_match` with answer aliases

### MGSM

Multilingual GSM8K -- math problems in 10 languages.

- **Dataset:** `hf://juletxara/mgsm?split=test`
- **Scorer:** `numeric_match`

### TriviaQA

Trivia questions with multiple acceptable answer aliases.

- **Dataset:** `hf://trivia_qa?config=rc.nocontext&split=validation`
- **Scorer:** `fuzzy_match` -- normalized substring matching against aliases

### HumanEval

164 Python function-completion problems with test suites.

- **Dataset:** `hf://openai/openai_humaneval?split=test`
- **Scorer:** `code_sandbox` -- extracts code from markdown fences, runs in Docker with network isolation, memory limits, and timeouts
- **Requires:** Docker daemon

### SimpleQA

Short-form factuality questions requiring LLM-as-judge scoring.

- **Dataset:** `hf://basicv8vc/SimpleQA?split=test`
- **Scorer:** `needs_judge` -- flags samples for post-processing by the judge pipeline

### HealthBench

Medical accuracy questions requiring LLM-as-judge scoring.

- **Dataset:** `hf://openai/HealthBench?split=test`
- **Scorer:** `needs_judge`

## Adding a New Benchmark

See the {doc}`../tutorials/byob` tutorial for a complete walkthrough of the `@benchmark` + `@scorer` API.
