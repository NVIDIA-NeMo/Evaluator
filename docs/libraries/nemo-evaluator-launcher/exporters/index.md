# Exporters

Exporters move evaluation results and artifacts from completed runs to external destinations for analysis, sharing, and reporting. They provide flexible options for integrating evaluation results into your existing workflows and tools.

## How to Set an Exporter

::::{tab-set}

:::{tab-item} CLI

```bash
nv-eval export <id1> [<id2> ...] \
  --dest <local|gsheets|wandb|mlflow|leaderboard> \
  [options]
```

:::

:::{tab-item} Python

```python
from nemo_evaluator_launcher.api.functional import export_results

export_results(
    ["8abcd123"], 
    dest="local", 
    config={
        "format": "json", 
        "output_dir": "./out"
    }
)
```

:::

::::

## Choosing an Exporter

Select exporters based on your analysis and reporting needs:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

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

Multiple exporters can be configured simultaneously to support different stakeholder needs and workflow integration points.

## Add Your Own Exporter

It's straightforward to add a custom exporter to fit your tools:
- Define destination-specific configuration (credentials, endpoints, paths)
- Implement metric selection and artifact upload logic
- Ensure idempotency (e.g., `skip_existing`) and good error messages
- Expose a CLI/Python entry point consistent with other exporters

:::{toctree}
:caption: Exporters
:hidden:

Local Files <local>
Weights & Biases <wandb>
MLflow <mlflow>
Google Sheets <gsheets>
:::
