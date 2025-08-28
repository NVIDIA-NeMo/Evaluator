# Evaluation Adapters

Evaluation adapters provide a flexible mechanism for intercepting and modifying requests/responses between the evaluation harness and the model endpoint. This allows for custom processing, logging, and transformation of data during the evaluation process.

## Usage Example

To enable the adapters, set the `adapter_config` field for the `ApiEndpoint` class.
The example below configures the adapter for a reasoning model:

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import EvaluationConfig, EvaluationTarget, AdapterConfig
adapter_config = AdapterConfig(
        interceptors=[
            # strip reasoning tokens from the response
            dict(
                name="reasoning",
                config={"end_reasoning_token": "</think>"},
            ),
            # set custom system prompt to enable reasoning
            dict(
                name="system_message",
                config={"system_message": "Detailed thinking on"}
            ),
            # log 5 request-response pair for verifying if model behaves as expected
            dict(
                name="request_logging",
                config={"max_requests": 5}
            ),
            dict(
                name="response_logging",
                config={"max_responses": 5}
            ),
        ]
    )

target_config = EvaluationTarget(
    api_endpoint={
        "url": chat_url,
        "model_id": "megatron_model",
        "type": "chat",
        "adapter_config": adapter_config
    }
)
eval_config = EvaluationConfig(
    type="mmlu_instruct",
    params={"limit_samples": 1, "temperature": 0.6, "top_p": 0.95},
    output_dir=f"{WORKSPACE}/mmlu",
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

The adapter system is configured using the `AdapterConfig` and the following interceptors:

| Name | Description |
|------|-------------|
| request_logging | Logs incoming requests to files for debugging and analysis. |
| response_logging | Logs outgoing responses to files. |
| caching | Caches requests and responses to improve performance and reduce API calls. |
| system_message | Modifies the system message in requests. |
| payload_modifier | Modifies request parameters by adding, removing or replacing |
| reasoning | Handles reasoning tokens in responses and tracks reasoning metrics. |
| progress_tracking | Tracks evaluation progress by counting the number of samples sent to the server |
