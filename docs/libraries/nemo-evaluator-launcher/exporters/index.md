# Exporters

Exporters move evaluation results and artifacts from completed runs to external destinations for analysis, sharing, and reporting. They provide flexible options for integrating evaluation results into your existing workflows and tools.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` Overview
:link: overview
:link-type: doc

Understand how exporters work, their common patterns, and configuration options for moving evaluation results.
:::

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Local Files
:link: local
:link-type: doc

Export results and artifacts to local or network file systems for custom analysis and archival.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Weights & Biases
:link: wandb
:link-type: doc

Track metrics, artifacts, and run metadata in W&B for comprehensive experiment management.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` MLflow
:link: mlflow
:link-type: doc

Export metrics and artifacts to MLflow Tracking Server for centralized ML lifecycle management.
:::

:::{grid-item-card} {octicon}`table;1.5em;sd-mr-1` Google Sheets
:link: gsheets
:link-type: doc

Export metrics to Google Sheets for easy sharing, reporting, and collaborative analysis.
:::

::::

## Choosing an Exporter

Select exporters based on your analysis and reporting needs:

- **Local**: Perfect for custom analysis, file-based workflows, and development
- **W&B**: Ideal for ML teams using Weights & Biases for experiment tracking
- **MLflow**: Best for organizations with MLflow-based ML platforms
- **Google Sheets**: Great for business reporting and collaborative metric tracking

Multiple exporters can be configured simultaneously to support different stakeholder needs and workflow integration points.
