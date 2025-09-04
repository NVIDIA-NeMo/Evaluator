(adapters-concepts)=
# Adapters

Adapters are a local reverse proxy that apply chained interceptors to requests and responses between the evaluation client and the upstream model endpoint.

## Architecture

Sequence overview:

1. Client sends request to the AdapterServer (localhost:PORT).
2. Request interceptors verify/transform headers and payloads.
3. Adapter forwards to the upstream model endpoint.
4. Model returns HTTP response.
5. Response interceptors extract/reshape/log as configured.
6. Adapter returns processed response for metric computation.

Common patterns: reasoning cleanup, system prompt standardization, response shaping, logging caps.

For usage and recipes, see {ref}`adapters`.


