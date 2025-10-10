(evaluation-model)=

# Evaluation Model

NeMo Evaluator provides evaluation approaches and endpoint compatibility for comprehensive AI model assessment.

## Evaluation Approaches

NeMo Evaluator supports several evaluation approaches through containerized harnesses:

- **Text Generation**: Models generate responses to prompts, assessed for correctness or quality against reference answers or rubrics.
- **Log Probability**: Models assign probabilities to token sequences, enabling confidence measurement without text generation. Effective for choice-based tasks and base model evaluation.
- **Code Generation**: Models generate code from natural language descriptions, evaluated for correctness through test execution.
- **Function Calling**: Models generate structured outputs for tool use and API interaction scenarios.
- **Safety & Security**: Evaluation against adversarial prompts and safety benchmarks to test model alignment and robustness.

One or more evaluation harnesses implement each approach. To discover available tasks for each approach, use `nemo-evaluator-launcher ls tasks`.

## Endpoint Compatibility

NeMo Evaluator targets OpenAI-compatible API endpoints. The platform supports the following endpoint types:

- **`completions`**: Direct text completion without chat formatting (`/v1/completions`). Used for base models and academic benchmarks.
- **`chat`**: Conversational interface with role-based messages (`/v1/chat/completions`). Used for instruction-tuned and chat models.
- **`vlm`**: Vision-language model endpoints supporting image inputs.
- **`embedding`**: Embedding generation endpoints for retrieval evaluation.

Each evaluation task specifies which endpoint types it supports. Verify compatibility using `nemo-evaluator-launcher ls tasks`.

## Metrics

Individual evaluation harnesses define metrics that vary by task. Common metric types include:

- **Accuracy metrics**: Exact match, normalized accuracy, F1 scores
- **Generative metrics**: BLEU, ROUGE, code execution pass rates
- **Probability metrics**: Perplexity, log-likelihood scores
- **Safety metrics**: Refusal rates, toxicity scores, vulnerability detection

The platform returns results in a standardized schema regardless of the source harness. To see metrics for a specific task, refer to {ref}`eval-benchmarks` or run an evaluation and inspect the results.

For hands-on guides, refer to {ref}`eval-run`.
