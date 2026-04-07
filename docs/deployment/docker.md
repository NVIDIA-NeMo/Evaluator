# Docker Deployment

## Build the image

```bash
docker build -t nemo-evaluator .
```

The Dockerfile installs the package with `[scoring]` extras and sets `ENTRYPOINT ["nel"]`.

## Run a single evaluation

```bash
docker run --rm \
    -e NEMO_API_KEY=$NEMO_API_KEY \
    -v $(pwd)/results:/app/eval_results \
    nemo-evaluator eval run --bench gsm8k --repeats 2
```

## Docker Compose

The `deploy/docker-compose.yaml` defines four service profiles:

### Serve + remote eval

```bash
docker compose -f deploy/docker-compose.yaml up serve eval-remote
```

```{mermaid}
flowchart LR
    A["eval-remote container"] -->|"HTTP"| B["serve container<br/>gsm8k:9090"]
    A -->|"HTTP"| C["Model API"]
```

### Local eval (no server)

```bash
docker compose -f deploy/docker-compose.yaml run eval-local
```

### Sharded evaluation

```bash
for i in 0 1 2 3; do
    NEL_SHARD_IDX=$i NEL_TOTAL_SHARDS=4 \
        docker compose -f deploy/docker-compose.yaml run -d eval-shard
done

# Merge after all complete
docker compose run eval-local nel eval merge /app/eval_results
```

## Health checks

The `serve` service includes:

```yaml
healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
    interval: 10s
    retries: 3
```
