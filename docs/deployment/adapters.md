(adapters)=

# Evaluation Adapters

Evaluation adapters provide a flexible mechanism for intercepting and modifying requests/responses between the evaluation harness and the model endpoint. This allows for custom processing, logging, and transformation of data during the evaluation process.

## Usage Example

To enable the adapters, pass the  `AdapterConfig` class to the `evaluate`  method of the `src/nemo_eval/api.py`.
The example below configures the adapter for reasoning:

```python
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, AdapterConfig
target_config = EvaluationTarget(api_endpoint={"url": completions_url, "type": "completions"})
eval_config = EvaluationConfig(
    type="mmlu",
    params={"limit_samples": 1},
    output_dir=f"{WORKSPACE}/mmlu",
)
# Configure adapter for reasoning
adapter_config = AdapterConfig(
    api_url=target_config.api_endpoint.url,
    use_reasoning=True,
    end_reasoning_token="</think>",
    custom_system_prompt="You are a helpful assistant that thinks step by step.",
    max_logged_requests=5,
    max_logged_responses=5
)

results = evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config,
    adapter_cfg=adapter_config,
)
```

## Architecture

The adapter system uses a chain of interceptors that process requests and responses in sequence. Here's the high-level architecture:

```
         ┌───────────────────────┐
         │                       │
         │ NVIDIA Eval Factory   │
         │                       │
         └───▲──────┬────────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:<free port>)                     │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                                                    │  │   │                      │
         │   │ └────────────────────────────────────────────────────┘  │   │                      │
         │   │                                                         │   │  upstream endpoint   │
         │   │                                                         │   │   with actual model  │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │ ┌─┼──────────────────────────────────────────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘
```

## Adapter System: Request/Response Flow

The **AdapterServer** acts as a local reverse proxy, mediating communication between the **NVIDIA Eval Factory** (client) and the **upstream model endpoint**. It employs a pipeline of interceptors to enable pre-processing of outgoing requests and post-processing of incoming responses. This architecture allows for modular insertion of custom logic without modifying the core client or model service.

```{mermaid}
sequenceDiagram
    participant Client as NVIDIA Eval Factory<br/>(Client)
    participant AS as AdapterServer<br/>(Local Reverse Proxy)
    participant RI as Request Interceptors<br/>(Chain: intcptr_1 → intcptr_N)
    participant Model as Upstream Model<br/>Endpoint
    participant RespI as Response Interceptors<br/>(Chain: intcptr'_1 → intcptr'_M)

    Note over Client, RespI: Evaluation Request Processing Flow

    Client->>AS: 1-2. Initiate evaluation request<br/>(routes to localhost:<port>)
    
    AS->>AS: 3. Receive inbound HTTP request
    
    AS->>RI: 4. Inject flask.Request into<br/>request interceptor chain
    
    Note over RI: Sequential processing:<br/>• Header manipulation<br/>• Payload transformation<br/>• Request validation<br/>• Authentication/authorization<br/>• Logging
    
    RI->>AS: Processed request<br/>(potentially mutated)
    
    AS->>Model: 5. Dispatch request to<br/>upstream endpoint
    
    Model->>Model: 6. Execute inference logic<br/>and generate response
    
    Model->>AS: 7. Return HTTP response
    
    AS->>RespI: 8. Pass requests.Response to<br/>response interceptor chain
    
    Note over RespI: Sequential processing:<br/>• Data extraction<br/>• Output reformatting<br/>• Caching strategies<br/>• Custom logging
    
    RespI->>AS: 9. Finalized processed<br/>requests.Response
    
    AS->>Client: 10-11. Transmit final response<br/>for metric computation & analysis
```

Here's a detailed breakdown of the interaction flow:

1.  **Client Request Initiation**:
    * The **NVIDIA Eval Factory** (client) initiates an evaluation sequence by dispatching a request destined for a model.

2.  **Request Routing to AdapterServer**:
    * The Eval Factory directs its request not to the target model endpoint directly, but to the locally running **AdapterServer**. This server listens on a designated port (e.g., `localhost:<port>`) and functions as the primary interface for the interception pipeline.

3.  **Inbound Request Reception**:
    * The **AdapterServer** receives the inbound HTTP request from the Eval Factory.

4.  **Request Interceptor Chain Processing**:
    * Upon receipt, the request (instantiated as a `flask.Request` object) is injected into a configurable **chain of RequestInterceptors** (`intcptr_1` → `intcptr_2` → ... → `intcptr_N`).
    * Each interceptor sequentially processes the `flask.Request` object. This design enables developers to implement diverse functionalities such as header manipulation, payload transformation, request validation, authentication/authorization handshakes, or detailed logging before the request is forwarded to the subsequent interceptor or the upstream service.

5.  **Dispatch to Upstream Model Endpoint**:
    * Once the `flask.Request` has successfully traversed the entire request interceptor chain (i.e., after processing by `intcptr_N`), the **AdapterServer** dispatches the (potentially mutated) request to the designated **upstream endpoint** hosting the target AI model.

6.  **Model Inference and Response Generation**:
    * The **upstream model endpoint** executes its inference logic based on the received request and generates a corresponding HTTP response.

7.  **Response Routing to AdapterServer**:
    * The model's HTTP response is routed back to the **AdapterServer**.

8.  **Response Interceptor Chain Processing**:
    * The incoming response (typically a `requests.Response` object) is then passed through a **Chain of ResponseInterceptors**. The flow is: upstream response → `intcptr'_1` → `intcptr'_2` → ... → `intcptr'_M`.
    * This chain processes the `requests.Response` sequentially. Each interceptor can inspect, modify, or augment the response, facilitating tasks such as data extraction from the payload, reformatting output, implementing caching strategies, or custom logging before it's relayed to the client.

9.  **Finalized Response**:
    * After traversal through the entire response interceptor chain (i.e., after processing by `intcptr'_M`), the `requests.Response` object represents the final, processed data to be returned to the client.

10. **Response Transmission to Client**:
    * The **AdapterServer** transmits this final `requests.Response` back to the initial caller, the NVIDIA Eval Factory.

11. **Client Consumes Processed Response**:
    * The **NVIDIA Eval Factory** receives the processed response. This data can then be consumed for metric computation, results aggregation, or further analysis within the evaluation framework.

## Configuration

The adapter system is configured using the `AdapterConfig` class with the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_url` | `str` |  | URL of the api endpoint (same as `target_config.api_endpoint.url`) that will be proxied |
| `local_port` | `Optional[int]` | `None` | Local port to use for the adapter server. If `None`, a free port will be chosen automatically |
| `use_reasoning` | `bool` | `False` | Whether to use the clean-reasoning-tokens adapter |
| `end_reasoning_token` | `str` | `"</think>"` | Token that signifies the end of reasoning output |
| `custom_system_prompt` | `Optional[str]` | `None` | A custom system prompt to replace the original one (if not None), **only for chat endpoints** |
| `max_logged_responses` | `Optional[int]` | `5` | Maximum number of responses to log. Set to 0 to disable. If None, all will be logged |
| `max_logged_requests` | `Optional[int]` | `5` | Maximum number of requests to log. Set to 0 to disable. If None, all will be logged |

