(deployment-concepts)=
# Deployment Concepts

High-level guidance for choosing backends and understanding where adapters fit.

## Backend Selection

| Use Case | Recommended Backend | Key Benefits |
|----------|-------------------|--------------|
| Production deployment | vLLM | High performance, multi-GPU support |
| Custom processing | Adapters | Request/response transformation |

## Adapter Positioning

Adapters sit between the evaluation client and the upstream model endpoint, enabling request/response interception without changes to either side.

## Scaling Modes

- Multi-GPU tensor and pipeline parallelism (vLLM)

For deployment steps, refer to {ref}`deployment-overview`.


