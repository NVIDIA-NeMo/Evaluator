(evaluation-model)=
# Evaluation Model

This page introduces NeMo Eval's core evaluation model: evaluation types, endpoint formats, and core metrics.

## Evaluation Types

- **Text Generation**: Models generate responses to prompts, assessed for correctness or quality.
- **Log Probability**: Models assign probabilities to continuations for multiple-choice style scoring.
- **Multiple Choice**: Models pick the best answer from predefined options.

## Endpoints

NeMo Eval targets OpenAI-compatible APIs:

- **Completions**: `/v1/completions/` — direct text completion without chat formatting.
- **Chat**: `/v1/chat/completions/` — conversational interface with role-based messages.

## Core Metrics

- **Accuracy**
- **Perplexity**
- **Pass@k** (code)
- **BLEU/ROUGE** (text similarity)
- **Safety Scores**

For hands-on guides, refer to {ref}`eval-run`.


