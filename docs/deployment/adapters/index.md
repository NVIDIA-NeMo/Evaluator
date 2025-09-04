<!-- markdownlint-disable MD041 -->
(adapters)=

# Evaluation Adapters

Evaluation adapters provide a flexible mechanism for intercepting and modifying requests/responses between the evaluation harness and the model endpoint. This allows for custom processing, logging, and transformation of data during the evaluation process.

## Architecture

Evaluation adapters run as a local reverse proxy that applies chained interceptors to requests and responses between your evaluation client and the upstream model endpoint.

```{mermaid}
sequenceDiagram
    participant Client as Evaluation Client
    participant AS as AdapterServer<br/>(Local Reverse Proxy)
    participant RI as Request Interceptors<br/>(intcptr_1 → intcptr_N)
    participant Model as Upstream Model<br/>Endpoint
    participant RespI as Response Interceptors<br/>(intcptr'_1 → intcptr'_M)

    Client->>AS: Send request to localhost:PORT
    AS->>RI: Inject flask.Request into request interceptor chain
    RI->>AS: Return processed request
    AS->>Model: Forward request to upstream endpoint
    Model->>AS: Return HTTP response
    AS->>RespI: Pass response to response interceptor chain
    RespI->>AS: Return finalized response
    AS->>Client: Return processed response for metric computation
```

Flow summary:

1. Client sends the request to the AdapterServer (localhost:PORT).
2. Request interceptors can verify and transform headers and payloads.
3. AdapterServer forwards the request to the upstream model endpoint.
4. The model executes inference and returns an HTTP response.
5. Response interceptors can extract data, reformat output, and log as configured.
6. AdapterServer returns the final response to the client.
7. The evaluation uses the processed response to compute metrics.

## Topics

Explore the following pages to use and configure adapters.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Usage
:link: adapters-usage
:link-type: ref
Learn how to enable adapters and pass `AdapterConfig` to `evaluate`.
:::

:::{grid-item-card} Recipes
:link: deployment-adapters-recipes
:link-type: ref
Reasoning cleanup, system prompt override, response shaping, logging caps.
:::

::::{grid-item-card} Configuration
:link: adapters-configuration
:link-type: ref
View available `AdapterConfig` options and defaults.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Usage <usage>
Architecture <architecture>
Recipes <recipes/index>
Configuration <configuration>
```
