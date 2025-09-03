(evaluation-overview)=

# Model Evaluation

Comprehensive guide to LLM evaluation concepts, methodologies, and configuration in the NeMo Eval framework.

## Overview

Model evaluation in NeMo Eval encompasses various approaches to assess LLM performance across different dimensions: accuracy, reasoning, safety, coding ability, and domain expertise. This section provides conceptual guidance and reference materials for configuring and understanding evaluations.

## Before You Start

Before configuring evaluations, ensure you have:

- **Deployed Model**: A model deployed via [PyTriton](../deployment/pytriton.md) or [Ray Serve](../deployment/ray-serve.md)
- **Evaluation Framework**: Required evaluation packages installed (see [Benchmarks](benchmarks.md))
- **Authentication**: Hugging Face token for gated datasets (when required)
- **Tokenizer Access**: Model tokenizer for log-probability evaluations

---

## Evaluation Methodologies

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run Evaluations
:link: run-evals/index
:link-type: doc
Step-by-step guides for different evaluation scenarios: text generation, log-probability, code generation, safety testing, and function calling.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Parameters
:link: eval-parameters
:link-type: ref
Comprehensive reference for evaluation configuration parameters, optimization patterns, and framework-specific settings.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Task Configuration
:link: eval-custom-tasks
:link-type: ref
Learn how to configure evaluations for tasks without pre-defined configurations using custom benchmark definitions.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Benchmark Catalog
:link: eval-benchmarks
:link-type: ref
Explore available evaluation harnesses, benchmarks, and their specific use cases and requirements.
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: eval-troubleshooting
:link-type: ref
Resolve common evaluation issues, debug configuration problems, and optimize evaluation performance.
:::

::::

## Core Evaluation Concepts

### Evaluation Types

**Text Generation Evaluation**: Models generate responses to prompts, which are then assessed for correctness, quality, or adherence to instructions.

**Log-Probability Evaluation**: Models assign probabilities to different text continuations, allowing assessment of confidence and likelihood across multiple choices.

**Multi-Choice Evaluation**: Models select the best answer from predefined options, typically used for academic benchmarks and reasoning tasks.

### Evaluation Endpoints

NeMo Eval supports evaluation through OpenAI-compatible API endpoints:

- **Completions Endpoint** (`/v1/completions/`): Direct text completion without chat formatting
- **Chat Endpoint** (`/v1/chat/completions/`): Conversational interface with role-based message formatting

### Evaluation Metrics

Different benchmarks use various metrics:

- **Accuracy**: Percentage of correct answers
- **Perplexity**: Measure of model uncertainty  
- **Pass@k**: Percentage of problems solved in k attempts (coding tasks)
- **BLEU/ROUGE**: Text similarity scores
- **Safety Scores**: Alignment and safety assessment metrics
