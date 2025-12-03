(eval-run)=

# Evaluation Techniques

Follow step-by-step guides for different evaluation scenarios and methodologies in NeMo Evaluator.

## Before You Start

Ensure you have:

1. Completed the initial getting started guides for {ref}`gs-install` and {ref}`gs-quickstart`.
2. Have your endpoint and API key ready or prepared for the checkpoint you wish to deploy.
3. Prepared your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) for accessing gated datasets.


## Evaluations

Select an evaluation type tailored to your model capabilities.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Text Generation
:link: text-gen
:link-type: ref
Measure model performance through natural language generation for academic benchmarks, reasoning tasks, and general knowledge assessment.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Log-Probability
:link: logprobs
:link-type: ref
Assess model confidence and uncertainty using log-probabilities for multiple-choice scenarios without text generation.
:::

:::{grid-item-card} {octicon}`comment;1.5em;sd-mr-1` Reasoning
:link: reasoning
:link-type: ref
Control the thinking budget and post-process the responses to extract the reasoning content and the final answer
:::


::::

<!-- TODO: add once ready
:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: code-generation
:link-type: ref
Measure programming capabilities through code generation, completion, and algorithmic problem solving.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Function Calling
:link: function-calling
:link-type: ref
Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors.
::: -->


<!-- TODO: add once ready
Code Generation <code-generation>
Function Calling <function-calling> -->


:::{toctree}
:hidden:
Text Generation <text-gen>
Log Probability <logprobs>
Reasoning <reasoning>
:::
