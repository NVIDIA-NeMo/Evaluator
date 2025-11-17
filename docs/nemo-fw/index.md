(nemo-fw-overview)=

# About NeMo Framework Integration

Specialized guides for evaluating models trained or deployed using NeMo Framework. These pages cover advanced evaluation scenarios, including log-probability-based evaluation, custom task configuration, and extending the evaluation environment with additional NVIDIA Eval Factory packages.

## Before You Start

Before using these guides, ensure you have:

1. **NeMo Framework environment**: Docker image or local installation
2. **Deployed model**: NeMo checkpoint deployed and accessible
3. **NeMo Eval installed**: Core evaluation packages (`nvidia-lm-eval` is pre-installed in NeMo Framework Docker)

---

## Evaluation Methods

Choose your evaluation approach based on model type and task requirements.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Log-Probability Evaluation
:link: logprobs
:link-type: doc

Evaluate base models using log-probabilities for multiple-choice tasks without text generation.

+++
{bdg-secondary}`Base Models` {bdg-secondary}`Multiple Choice`
:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Custom Task Configuration
:link: custom-task
:link-type: doc

Run evaluations on tasks without pre-defined configs by specifying harness and task names directly.

+++
{bdg-secondary}`Advanced` {bdg-secondary}`Custom Tasks`
:::

::::

## Extend Evaluation Capabilities

Add specialized evaluation frameworks and packages to assess additional model capabilities.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` On-Demand Evaluation Packages
:link: optional-eval-package
:link-type: doc

Install and use additional NVIDIA Eval Factory packages (BFCL, garak, BigCode, simple-evals, safety-harness).

+++
{bdg-secondary}`Extension` {bdg-secondary}`Specialized`
:::

::::
