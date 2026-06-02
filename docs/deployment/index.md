# Deployment Guide

## Deployment Matrix

| Environment | Sharding | Live Serve | Regression Gate | Effort |
|-------------|----------|------------|-----------------|--------|
| {doc}`Local <local>` | Manual via env vars | `nel serve` | `nel compare` | Minimal |
| {doc}`Docker Compose <docker>` | Per-container env vars | docker-compose service | Script | Low |
| {doc}`SLURM <slurm>` | `nel eval run` with executor config | `nel serve` | sbatch chain | Medium |
| {doc}`Kubernetes <kubernetes>` | Indexed Job | Deployment + Service | CI pipeline | Medium |
| {doc}`Ray <ray>` | `@ray.remote` tasks | N/A | Script | Medium |
| {doc}`GitLab CI <ci-regression>` | CI variables per job | N/A | `regression:check` stage | Low |

## Which deployment to choose?

```{mermaid}
flowchart TD
    START["How big is your eval?"] --> SIZE{"> 1000 problems<br/>or > 4 repeats?"}
    SIZE -->|No| LOCAL["Local CLI<br/>nel eval run"]
    SIZE -->|Yes| INFRA{"What infra?"}

    INFRA --> HPC["HPC cluster"]
    INFRA --> CLOUD["Cloud / K8s"]
    INFRA --> RAYC["Ray cluster"]

    HPC --> SLURM["SLURM<br/>nel eval run config.yaml"]
    CLOUD --> K8S["K8s Indexed Job"]
    RAYC --> RAY["Ray launcher"]

    LOCAL --> CI{"Need regression gate?"}
    CI -->|Yes| GITLAB["GitLab CI"]
    CI -->|No| DONE["Done"]
```
