# Interceptors

Configure request/response interceptors to add logging, caching, and custom processing to your evaluations.

## Core Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`log;1.5em;sd-mr-1` Request & Response Logging
:link: logging
:link-type: doc

Log incoming requests and outgoing responses for debugging and analysis.
:::

:::{grid-item-card} {octicon}`cache;1.5em;sd-mr-1` Caching
:link: caching
:link-type: doc

Cache requests and responses to improve performance and reduce API calls.
:::

::::

## Specialized Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`comment;1.5em;sd-mr-1` System Messages
:link: system-messages
:link-type: doc

Modify system messages and prompts in requests.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Payload Modification
:link: payload-modification
:link-type: doc

Add, remove, or modify request parameters.
:::

:::{grid-item-card} {octicon}`brain;1.5em;sd-mr-1` Reasoning
:link: reasoning
:link-type: doc

Handle reasoning tokens and track reasoning metrics.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Progress Tracking
:link: progress-tracking
:link-type: doc

Track evaluation progress and status updates.
:::

::::

## Post-Evaluation Processing

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`report;1.5em;sd-mr-1` Post-Evaluation Hooks
:link: post-evaluation-hooks
:link-type: doc

Run additional processing, reporting, or cleanup after evaluations complete.
:::

::::

:::{toctree}
:caption: Interceptors
:hidden:

Request & Response Logging <logging>
Caching <caching>
System Messages <system-messages>
Payload Modification <payload-modification>
Reasoning <reasoning>
Progress Tracking <progress-tracking>
Post-Evaluation Hooks <post-evaluation-hooks>
:::
