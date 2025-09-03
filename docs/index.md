(template-home)=

# {{ product_name }} Documentation

Welcome to the {{ product_name_short }} documentation.

## Introduction to {{ product_name_short }}

New to {{ product_name_short }}? Start here to get up and running with your first model deployment and evaluation.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` About {{ product_name_short }}
:link: get-started-overview
:link-type: ref
Learn what {{ product_name_short }} is, its key capabilities, and who should use it for LLM evaluation.
:::

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` Key Features
:link: about-key-features
:link-type: ref
<!-- TBD -->
:::

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
<!-- TBD -->
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started-overview
:link-type: ref
Install {{ product_name_short }} and run your first model evaluation in just a few minutes.
:::

::::

## Evaluation Workflows

Explore the main capabilities of {{ product_name_short }} for model deployment and evaluation.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Model Deployment  
:link: deployment/index
:link-type: ref
Deploy NeMo models using PyTriton or Ray Serve for high-performance inference and evaluation.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Model Evaluation
:link: evaluation/index
:link-type: ref
Evaluate models using various benchmarks, harnesses, and evaluation techniques.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Development
:link: development/index
:link-type: ref
Contribute to {{ product_name_short }}, extend functionality, and access API documentation.
:::

::::

:::{toctree}
:hidden:
Home <self>
:::

:::{toctree}
:caption: About
:hidden:

Overview <about/index>
Key Features <about/key-features>
Concepts <about/concepts/index>
Release Notes <about/release-notes/index>
:::

:::{toctree}
:caption: Get Started
:hidden:

About Getting Started <get-started/index>
Install Eval <get-started/install>
Quickstart <get-started/quickstart>
:::

:::{toctree}
:caption: Tutorials
:hidden:

About Tutorials <tutorials/index>
:::

:::{toctree}
:caption: Evaluation
:hidden:

About Model Evaluation <evaluation/index>
Run Evals <evaluation/run-evals/index>
Custom Task Configuration <evaluation/custom-tasks>
Benchmark Catalog <evaluation/benchmarks>
Troubleshooting <evaluation/troubleshooting>
:::

:::{toctree}
:caption: Deployment
:hidden:

About Model Deployment <deployment/index>
PyTriton Backend <deployment/pytriton>
Ray Serve Deployment <deployment/ray-serve>
Evaluation Adapters <deployment/adapters>
:::

:::{toctree}
:caption: References
:hidden:

References <references/index>
Eval Parameters <evaluation/parameters>
API Documentation <apidocs/index.rst>
:::
