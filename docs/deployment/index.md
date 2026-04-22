# Deployment Guide

```{toctree}
:maxdepth: 1

local
docker
slurm
kubernetes
ray
ci-regression
```

## Deployment Matrix

| Environment | Sharding | Live Serve | Regression Gate | Effort |
|-------------|----------|------------|-----------------|--------|
| Local | Manual via env vars | `nel serve` | `nel compare` | Minimal |
| Docker Compose | Per-container env vars | docker-compose service | Script | Low |
| SLURM | `nel eval run` with executor config | `nel serve` | sbatch chain | Medium |
| Kubernetes | Indexed Job | Deployment + Service | CI pipeline | Medium |
| Ray | `@ray.remote` tasks | N/A | Script | Medium |
| GitLab CI | CI variables per job | N/A | `regression:check` stage | Low |

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
