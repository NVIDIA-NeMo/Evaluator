# Adapter Context

Domain language for adapter proxy and interceptor behavior around model service traffic. This context extends the common evaluation language for code under `src/nemo_evaluator/adapters/`.

## Language

**Adapter Proxy**:
Local HTTP middleware attached to a model **Service** that intercepts solver/agent-to-service traffic before and after upstream model calls.
_Avoid_: Proxy, LiteLLM proxy, model service

**Interceptor**:
One configured component in an **Adapter Proxy** pipeline that runs at a declared stage to inspect, modify, forward, or directly answer one model-service HTTP exchange.
_Avoid_: Callback, hook, middleware

**Adapter Request**:
The interceptable model-service HTTP request object inside the **Adapter Proxy**, after the proxy has parsed the incoming request and before an upstream response exists.
_Avoid_: HTTP request, prompt, service call

**Adapter Response**:
The interceptable model-service HTTP response object inside the **Adapter Proxy**, either returned by the upstream endpoint or produced directly by an **Interceptor**.
_Avoid_: HTTP response, completion, solution

**Direct Adapter Response**:
An **Adapter Response** produced by a **Forwarding Stage** **Interceptor** before the **Endpoint Interceptor** performs an upstream model-service call.
_Avoid_: Short-circuit response, mock response, cached response

**Adapter Pipeline**:
The ordered chain inside an **Adapter Proxy** that runs configured **Interceptors** plus the built-in upstream forwarding step for one **Adapter Request**.
_Avoid_: Callback list, middleware stack

**Request Stage**:
The **Adapter Pipeline** phase where **Interceptors** may inspect or modify an **Adapter Request**, but cannot produce an **Adapter Response**.
_Avoid_: Pre-hook, request callback

**Forwarding Stage**:
The **Adapter Pipeline** phase where an **Interceptor** either forwards the **Adapter Request** upstream or directly produces an **Adapter Response**.
_Avoid_: Request-to-response stage, endpoint phase, short-circuit stage

**Response Stage**:
The **Adapter Pipeline** phase where **Interceptors** may inspect or modify an **Adapter Response** after one exists.
_Avoid_: Post-hook, response callback

**Endpoint Interceptor**:
The built-in **Forwarding Stage** **Interceptor** that performs the upstream model-service HTTP call and turns the upstream result into an **Adapter Response**.
_Avoid_: Endpoint, service endpoint, model service

**Interceptor Context**:
Per-request shared state carried through the **Adapter Pipeline**, including a request id and adapter-only metadata such as session id, cache key, or cache hit status.
_Avoid_: Context, metadata, eval metadata

**Adapter Session**:
A logical sequence of model-service calls identified by the `/s/<id>` path prefix on **Adapter Proxy** requests, used by **Interceptors** that need per-session state such as turn counting or session-scoped cache keys.
_Avoid_: Session, solver session, trajectory session

**Proxy Config**:
The `services.<name>.proxy` YAML/runtime configuration that enables and configures an **Adapter Proxy** for one model **Service**, including configured **Interceptors**, upstream timeout/retry settings, extra body fields, extra headers, and verbosity.
_Avoid_: Adapter Proxy, service config, benchmark config

## Relationships

- A **Solver** or agent calls a model **Service** through an **Adapter Proxy** when that service has adapter proxy behavior enabled.
- An **Adapter Proxy** belongs to one model **Service**.
- An **Adapter Proxy** may inspect or modify request and response traffic, but it is not the **Service** itself.
- An **Adapter Proxy** does not own seeding, verification, scoring, aggregation, or release decisions.
- An **Adapter Proxy** owns one **Adapter Pipeline** for one model **Service**.
- An **Adapter Pipeline** runs zero or more configured **Interceptors** plus the built-in **Endpoint Interceptor**.
- An **Interceptor** is scoped to service traffic; it is not an evaluation-loop callback.
- An **Adapter Request** represents one in-flight model-service call inside an **Adapter Proxy**.
- An **Adapter Response** completes one **Adapter Request**.
- An **Adapter Response** is not a **Solution**; it is service traffic that a **Solver** may use while producing a **Solution**.
- **Request Stage** **Interceptors** run in order in an **Adapter Pipeline**.
- **Forwarding Stage** **Interceptors** run after the **Request Stage** and before the **Response Stage**.
- A **Forwarding Stage** **Interceptor** may produce a **Direct Adapter Response** without invoking the **Endpoint Interceptor** for that **Adapter Request**.
- **Response Stage** **Interceptors** run in reverse order after an **Adapter Response** exists.
- The **Endpoint Interceptor** is inserted by the **Adapter Proxy**; it is not the upstream **Service** itself.
- Each **Adapter Request** and **Adapter Response** carries one **Interceptor Context** for the current model-service exchange.
- **Interceptor Context** is in-flight adapter state, not persisted evaluation metadata.
- An **Adapter Session** may span multiple **Adapter Requests**.
- The **Adapter Proxy** strips the `/s/<id>` prefix before the **Endpoint Interceptor** forwards the request upstream.
- **Adapter Session** identity is stored in **Interceptor Context** for **Interceptors** that need it.
- A **Proxy Config** belongs to one model **Service** configuration.
- A **Proxy Config** configures an **Adapter Proxy**, but it is not the running **Adapter Proxy**.

## Flagged ambiguities

- "proxy" was used broadly for local HTTP middleware and generic networking infrastructure; resolved: **Adapter Proxy** is the local service traffic middleware in `src/nemo_evaluator/adapters/`.
- "LiteLLM proxy" appears in older docs for adapter behavior; resolved: use **Adapter Proxy** for the built-in interceptor proxy.
- "callback" was used for older proxy integration points; resolved: **Interceptor** is the configured pipeline component in the current adapter proxy.
- "hook" is reserved for explicit lifecycle hooks such as `PostEvalHook`; resolved: do not use it for request/response traffic interception.
- "request" and "response" can mean Starlette objects, raw HTTP traffic, or model payloads; resolved: **Adapter Request** and **Adapter Response** name the internal proxy data objects.
- "callback list" underspecifies ordering and short-circuit behavior; resolved: **Adapter Pipeline** is the ordered chain that owns those semantics.
- `REQUEST_TO_RESPONSE` is the implementation enum value; resolved: use **Forwarding Stage** in domain docs when describing the phase that either forwards upstream or returns an **Adapter Response** directly.
- "endpoint" can mean a route, URL, or service endpoint; resolved: **Endpoint Interceptor** names the built-in forwarding interceptor inside the **Adapter Pipeline**.
- "short-circuit response" is implementation-flavored; resolved: **Direct Adapter Response** names a response produced by the **Adapter Pipeline** without an upstream service call.
- "context" is overloaded across docs, config, multiprocessing, and domain state; resolved: **Interceptor Context** names only the per-request state carried through the **Adapter Pipeline**.
- "session" is used for solver and trajectory concerns elsewhere; resolved: **Adapter Session** names only the `/s/<id>` adapter proxy traffic grouping.
- "proxy config" and "adapter proxy" were easy to blur; resolved: **Proxy Config** is the service config section, while **Adapter Proxy** is the running local middleware.
