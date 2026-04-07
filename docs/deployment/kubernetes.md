# Kubernetes Deployment

## Single evaluation job

```bash
kubectl apply -f deploy/k8s/eval-job.yaml
```

**Manifest:** `deploy/k8s/eval-job.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nel-eval-gsm8k
spec:
  template:
    spec:
      containers:
        - name: eval
          image: nemo-evaluator:latest
          args: ["eval", "run", "--bench", "gsm8k", "--repeats", "4",
                 "--output-dir", "/data/results", "--no-progress"]
          env:
            - name: NEMO_API_KEY
              valueFrom:
                secretKeyRef:
                  name: nemo-api
                  key: api_key
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: eval-data
      restartPolicy: Never
```

## Distributed evaluation (Indexed Job)

```{mermaid}
flowchart TB
    IJ["Indexed Job<br/>completions=8"] --> P0["Pod 0<br/>NEL_SHARD_IDX=0"]
    IJ --> P1["Pod 1<br/>NEL_SHARD_IDX=1"]
    IJ --> P7["Pod 7<br/>NEL_SHARD_IDX=7"]
    P0 --> PVC["Shared PVC<br/>shard_0/ ... shard_7/"]
    P1 --> PVC
    P7 --> PVC
    PVC --> MJ["Merge Job"]
    MJ --> RESULT["merged/eval-*.json"]
```

Apply `deploy/k8s/eval-indexed-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nel-eval-sharded
spec:
  completions: 8
  parallelism: 8
  completionMode: Indexed
  template:
    spec:
      containers:
        - name: eval
          image: nemo-evaluator:latest
          args: ["eval", "run", "--bench", "gsm8k", "--repeats", "8",
                 "--output-dir", "/data/shard_$(NEL_SHARD_IDX)", "--no-progress"]
          env:
            - name: NEL_SHARD_IDX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: NEL_TOTAL_SHARDS
              value: "8"
```

Then merge:

```bash
kubectl apply -f deploy/k8s/eval-merge.yaml
```

## Persistent serving

For Gym training integration:

```bash
kubectl apply -f deploy/k8s/serve-deployment.yaml
```

Creates a `Deployment` + `ClusterIP Service` at `nel-serve.default.svc:9090`.

Includes readiness and liveness probes:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 5
  periodSeconds: 10
livenessProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 15
  periodSeconds: 30
```

From Gym training pods:

```yaml
resource_servers:
  nemo_evaluator:
    endpoint: http://nel-serve.default.svc:9090
```
