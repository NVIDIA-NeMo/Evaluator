(eval-benchmarks)=

# Benchmark Catalog

Comprehensive reference of available evaluation harnesses, benchmarks, and tasks supported in NeMo Eval.

## Overview

NeMo Eval integrates with multiple evaluation frameworks through the NVIDIA Eval Factory ecosystem. Each framework provides specialized benchmarks for different aspects of LLM evaluation.

## Core Evaluation Harnesses

### lm-evaluation-harness (Pre-installed)

The foundational evaluation framework providing academic benchmarks and reasoning tasks.

**Source**: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  
**Package**: `nvidia-lm-eval` (pre-installed in NeMo Framework container)

#### Available Tasks

**Academic Benchmarks**:
- `mmlu` - Massive Multitask Language Understanding
- `mmlu_pro` - Enhanced MMLU with harder questions
- `mmlu_redux` - Refined MMLU dataset
- `arc_challenge` - AI2 Reasoning Challenge
- `hellaswag` - Commonsense reasoning
- `truthfulqa` - Truthfulness assessment

**Reasoning Tasks**:
- `gsm8k` - Grade school math word problems
- `bbh` - Big-Bench Hard reasoning tasks
- `commonsense_qa` - Common sense reasoning
- `winogrande` - Pronoun resolution reasoning

**Multilingual Benchmarks**:
- `arc_multilingual` - ARC in multiple languages
- `hellaswag_multilingual` - HellaSwag multilingual version
- `mgsm` - Multilingual grade school math

**Chat/Instruction Tasks**:
- `mmlu_instruct` - MMLU with instruction formatting
- `mmlu_pro_instruct` - MMLU Pro for instruction-tuned models
- `ifeval` - Instruction following evaluation
- `gpqa_diamond_cot` - Graduate-level Q&A with chain-of-thought

#### Endpoint Requirements

| Task Category | Endpoint Type | Special Requirements |
|---------------|---------------|---------------------|
| Academic benchmarks | `completions` | None |
| Chat benchmarks | `chat` | Instruction-tuned model |
| Log-probability tasks | `completions` | Tokenizer configuration |

## Optional Evaluation Harnesses

### simple-evals

Streamlined evaluation workflows for common benchmarks.

**Installation**: `pip install nvidia-simple-evals`  
**Focus**: Simplified evaluation with standardized metrics

**Key Benchmarks**:
- `mmlu` - Simplified MMLU implementation
- `humaneval` - Code generation evaluation
- `math` - Mathematical reasoning
- `drop` - Reading comprehension with arithmetic

### BigCode

Specialized evaluation for code generation and programming tasks.

**Installation**: `pip install nvidia-bigcode`  
**Focus**: Programming ability assessment

**Key Benchmarks**:
- `humaneval` - Python programming problems
- `mbpp` - Mostly Basic Python Problems
- `apps` - Algorithmic programming problems
- `code_contests` - Competitive programming

### BFCL (Berkeley Function Calling Leaderboard)

Function calling and tool use evaluation.

**Installation**: `pip install nvidia-bfcl`  
**Focus**: API usage and function calling capabilities

**Key Benchmarks**:
- `simple` - Basic function calling
- `multiple` - Multiple function calls
- `parallel` - Parallel function execution
- `irrelevant` - Handling irrelevant function calls

### safety-harness

AI safety and alignment testing framework.

**Installation**: `pip install nvidia-safety-harness`  
**Focus**: Safety, bias, and alignment assessment

**Key Benchmarks**:
- `toxicity` - Toxic content generation
- `bias` - Demographic bias evaluation
- `fairness` - Fairness across groups
- `privacy` - Privacy leakage detection

### garak

LLM vulnerability scanning and red teaming.

**Installation**: `pip install nvidia-garak`  
**Focus**: Security vulnerabilities and attack resistance

**Key Benchmarks**:
- `prompt_injection` - Prompt injection attacks
- `jailbreaking` - Safety filter bypassing
- `hallucination` - Factual accuracy assessment
- `leakage` - Information leakage detection

## Task Configuration Reference

### Discovering Available Tasks

List all available evaluation tasks in your environment:

```python
from nemo_eval.utils.base import list_available_evaluations

# Get all available tasks
available_tasks = list_available_evaluations()
print(available_tasks)
```

### Task Naming Conventions

When multiple frameworks provide the same task name, use the full specification:

```python
# Ambiguous (will cause error if multiple frameworks installed)
task = "mmlu"

# Explicit framework specification
task = "lm-evaluation-harness.mmlu"
task = "simple-evals.mmlu"
```

### Common Configuration Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `limit_samples` | Number of samples to evaluate | `100`, `0.1` (10%) |
| `parallelism` | Concurrent requests | `1`, `4`, `8` |
| `temperature` | Sampling temperature | `0.0`, `1.0` |
| `top_p` | Nucleus sampling | `1.0`, `0.9` |
| `max_tokens` | Response length limit | `256`, `1024` |

## Evaluation Requirements by Category

### Log-Probability Tasks

**Required Configuration**:
```python
params = ConfigParams(
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface"
    }
)
```

**Applicable Tasks**: `arc_challenge`, `hellaswag`, `truthfulqa`, `winogrande`

### Chat/Instruction Tasks

**Required Configuration**:
- Chat endpoint (`/v1/chat/completions/`)
- Instruction-tuned model
- Proper chat template configuration

**Applicable Tasks**: `mmlu_instruct`, `ifeval`, `gpqa_diamond_cot`

### Code Generation Tasks

**Required Configuration**:
- High `max_tokens` limit (512-2048)
- Low temperature (0.0-0.2) for deterministic output
- Code execution environment (for some benchmarks)

**Applicable Tasks**: `humaneval`, `mbpp`, `apps`

### Gated Dataset Requirements

Some benchmarks require Hugging Face authentication:

**Setup**:
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or authenticate via CLI
huggingface-cli login
```

**Gated Benchmarks**: `gpqa`, some safety-harness tasks

## Performance Considerations

### Evaluation Speed Optimization

1. **Parallel Requests**: Increase `parallelism` for faster evaluation
2. **Sample Limiting**: Use `limit_samples` for quick testing
3. **Batch Size**: Optimize model `max_batch_size` for throughput
4. **Multi-Instance**: Use Ray Serve for concurrent model replicas

### Resource Requirements

| Benchmark Category | GPU Memory | Evaluation Time | Special Requirements |
|-------------------|------------|-----------------|---------------------|
| Academic (MMLU) | Low | Fast | None |
| Code Generation | Medium | Slow | Code execution |
| Safety Testing | Low | Medium | Content filtering |
| Multilingual | Medium | Medium | Multilingual tokenizer |

## Custom Benchmark Integration

For benchmarks not available in standard harnesses:

1. **Identify Base Framework**: Choose appropriate evaluation harness
2. **Task Definition**: Create task configuration following harness patterns
3. **Data Preparation**: Format dataset according to harness requirements
4. **Metric Implementation**: Define evaluation metrics and scoring
5. **Validation**: Test with known baseline models

## Troubleshooting Common Issues

### Framework Not Found
```bash
# Install missing framework
pip install nvidia-<framework-name>

# Reload evaluation registry
python -c "import importlib; import core_evals; importlib.reload(core_evals)"
```

### Task Conflicts
```python
# Use explicit framework specification
task = "lm-evaluation-harness.task_name"
```

### Authentication Issues
```bash
# Verify HuggingFace token
huggingface-cli whoami

# Set token for current session
export HF_TOKEN=your_token
```

### Performance Issues
- Reduce `parallelism` if hitting rate limits
- Increase `max_batch_size` for better throughput
- Use `limit_samples` for quick validation
- Consider Ray Serve for multi-instance acceleration

## Next Steps

- **Getting Started**: Try the [MMLU Tutorial](../tutorials/mmlu.ipynb)
- **Custom Tasks**: Learn [Custom Task Configuration](custom-tasks.md)
- **Advanced Evaluation**: Explore [Log-Probability Methods](logprobs.md)
- **Performance**: Review [Deployment Options](../deployment/index.md)
