(deployment-concepts)=
# Deployment Concepts

High-level guidance for choosing backends and understanding where adapters fit.

## Backend Selection

| Use Case | Recommended Backend | Key Benefits |
|----------|-------------------|--------------|
| Production deployment | PyTriton | High performance, multi-node support |
| Accelerated evaluation | Ray Serve | Multi-instance, horizontal scaling |
| Custom processing | Adapters | Request/response transformation |

## Adapter Positioning

Adapters sit between the evaluation client and the upstream model endpoint, enabling request/response interception without changes to either side.

## Scaling Modes

- Multi-node model parallelism (PyTriton)
- Multi-instance horizontal scaling (Ray Serve)

For deployment steps, refer to {ref}`deployment-overview`.


