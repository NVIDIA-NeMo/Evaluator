
(deployment-adapters-recipes)=
# Recipes

Practical, focused examples for common adapter scenarios.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Reasoning Cleanup
:link: adapters-recipe-reasoning
:link-type: ref
Strip intermediate reasoning tokens before scoring.
:::

:::{grid-item-card} Custom System Prompt (Chat)
:link: adapters-recipe-system-prompt
:link-type: ref
Enforce a standard system prompt for chat endpoints.
:::

:::{grid-item-card} Request Parameter Modification
:link: adapters-recipe-response-shaping
:link-type: ref
Standardize request parameters across endpoint providers.
:::

:::{grid-item-card} Logging Caps
:link: adapters-recipe-logging
:link-type: ref
Control logging volume for requests and responses.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Reasoning Cleanup <reasoning-cleanup>
Custom System Prompt (Chat) <custom-system-prompt>
Request Parameter Modification <response-shaping>
Logging Caps <logging-caps>
```

