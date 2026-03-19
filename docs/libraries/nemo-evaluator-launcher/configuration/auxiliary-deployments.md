(auxiliary-deployments)=

# Auxiliary Deployments

Auxiliary deployments let you run additional model servers alongside the primary deployment within a single launcher job. This is essential for evaluation workflows that require multiple endpoints — for example, **LLM-as-a-judge** benchmarks where a second model scores the primary model's outputs.

:::{note}
Auxiliary deployments are currently supported by the **Slurm executor** only.
:::

:::{important}
LLM-as-a-judge evaluations (such as AIME) require an evaluation container image
that supports judge configuration. Verify that your evaluation container version
includes judge endpoint support before running auxiliary deployment workflows.
:::

## When to Use Auxiliary Deployments

Use auxiliary deployments when your evaluation benchmark needs access to a model endpoint beyond the primary model under test:

- **LLM-as-a-judge**: A judge model scores or grades the primary model's responses (e.g., AIME, GPQA with judge scoring)
- **Reward models**: A reward or preference model evaluates output quality
- **Rerankers**: A reranking model reorders candidate responses
- **Custom pipelines**: Any evaluation that requires multiple model endpoints

Without auxiliary deployments, you would need to deploy these additional models separately and manage their lifecycle manually. The launcher handles all of this automatically.

## Configuration

Auxiliary deployments are defined in the top-level `auxiliary_deployments` section of your config. Each key is a deployment name (e.g., `judge`, `reranker`) that identifies the auxiliary.

:::{note}
Each auxiliary deployment requires **its own dedicated Slurm node(s)**, separate from the primary deployment nodes. The total node count in the Slurm job equals `execution.num_nodes` (primary) plus the sum of all auxiliary `num_nodes`. This may be extended in the future.
:::

### Fields

Auxiliary deployments use the same core fields as the primary {ref}`deployment configuration <configuration-overview>` (`type`, `image`, `port`, `served_model_name`, `endpoints`, `env_vars`, `checkpoint_path`, `cache_path`, `extra_args`, `pre_cmd`, `command`, `command_template`, and the vLLM-specific parallelism/memory fields). See the {ref}`deployment docs <deployment-vllm>` for detailed descriptions of these shared fields.

The **required** fields for each auxiliary are: `type`, `image`, `port`, `served_model_name`, and `endpoints` (must include `health`).

Additionally, auxiliary deployments support these **auxiliary-specific** fields:

| Field | Default | Description |
|---|---|---|
| `num_nodes` | `1` | Number of dedicated Slurm nodes for this auxiliary deployment |
| `num_instances` | `1` | Independent server instances (enables HAProxy load balancing when > 1; see {ref}`aux-multi-instance`) |
| `env_prefix` | `<NAME>.upper()` | Prefix for generated environment variables (e.g., `JUDGE` for a `judge` deployment) |

:::{tip}
Auxiliary deployments **support Hydra `defaults:` composition**. Use the `@` package directive to apply deployment defaults to an auxiliary:

```yaml
defaults:
  - execution: slurm/default
  - deployment: vllm                                  # primary
  - deployment/nim@auxiliary_deployments.reranker      # auxiliary inherits NIM defaults
  - _self_
```

This reduces duplication — the auxiliary inherits all default fields from the deployment config group, and you only need to override what differs (e.g., `image`, `served_model_name`, `port`).
:::

### Minimal Example

A minimal working configuration using Hydra defaults composition for both the primary and auxiliary deployments:

```yaml
defaults:
  - execution: slurm/default
  - deployment: vllm                                  # primary deployment defaults
  - deployment/vllm@auxiliary_deployments.judge        # judge inherits vLLM defaults
  - _self_

execution:
  hostname: my-cluster.com
  account: my-account
  output_dir: /shared/results

deployment:
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 8

# Only override fields that differ from the vLLM defaults
auxiliary_deployments:
  judge:
    hf_model_handle: Qwen/Qwen2.5-72B-Instruct
    served_model_name: Qwen/Qwen2.5-72B-Instruct
    port: 8001  # Different port to avoid collision with primary
    num_nodes: 1

evaluation:
  tasks:
    - name: AIME_2025
```

:::{note}
Without Hydra defaults composition, you must specify all required fields directly in the auxiliary section. See the {ref}`complete working example <aux-working-example>` for both approaches.
:::

## How It Works

When auxiliary deployments are configured, the launcher:

1. **Allocates additional Slurm nodes** — the total `#SBATCH --nodes` equals `execution.num_nodes` (primary) plus the sum of all auxiliary `num_nodes`.

2. **Splits the node list** — primary model nodes are allocated first, then each auxiliary deployment receives its dedicated nodes.

3. **Starts auxiliary servers** — each auxiliary gets its own `srun` command with the specified container image, mounts, and environment variables.

4. **Waits for readiness** — the launcher polls each auxiliary's health endpoint before starting the evaluation.

5. **Exports endpoint URLs** — for each non-health endpoint, the launcher exports environment variables following the pattern:

   ```
   <ENV_PREFIX>_<ENDPOINT_NAME>_URL=http://<node>:<port><path>
   <ENV_PREFIX>_MODEL_ID=<served_model_name>
   ```

   For a `judge` deployment with `env_prefix: JUDGE` (the default, derived from the key name) and a `chat` endpoint:
   ```
   JUDGE_CHAT_URL=http://judge-node:8001/v1/chat/completions
   JUDGE_MODEL_ID=meta-llama/Llama-3.1-70B-Instruct
   ```

6. **Resolves config references** — in `evaluation.nemo_evaluator_config`, you can use `auxiliary_deployments.<name>.<field>` as placeholder values. The launcher resolves these before passing the config to the evaluation container:

   - `auxiliary_deployments.judge.model_id` → literal value of `served_model_name`
   - `auxiliary_deployments.judge.chat_url` → shell variable `${JUDGE_CHAT_URL}` (resolved at runtime)

7. **Cleans up** — after the evaluation completes, the launcher terminates all auxiliary servers.

## Referencing Auxiliary Endpoints in Evaluation Config

The evaluation harness needs to know the judge endpoint URL and model ID. Use the `auxiliary_deployments.<name>.<field>` reference syntax in your `nemo_evaluator_config`:

```yaml
evaluation:
  nemo_evaluator_config:
    config:
      params:
        extra:
          judge:
            url: auxiliary_deployments.judge.chat_url
            model_id: auxiliary_deployments.judge.model_id
```

The launcher resolves `model_id` references to literal values and endpoint URL references to shell variables that are expanded at runtime. This works because the launcher:

1. Parses the YAML config
2. Replaces `auxiliary_deployments.<name>.model_id` with the `served_model_name` value
3. Replaces `auxiliary_deployments.<name>.<endpoint>_url` with `${<PREFIX>_<ENDPOINT>_URL}` shell variables
4. Uses `envsubst` at runtime to inject the actual URLs

(aux-disabling)=
## Disabling an Auxiliary Deployment

Set `type: none` to skip a deployment without removing the config:

```yaml
auxiliary_deployments:
  judge:
    type: none  # Skipped — no judge deployed
```

When `type: none` is set, the launcher skips all infrastructure for that auxiliary (no nodes allocated, no `srun`, no health check, no env vars exported). This is useful for switching between a self-hosted judge and an external judge API without restructuring your config.

## Volume Mounts

Mount additional host directories into auxiliary deployment containers using `execution.mounts.auxiliary`:

```yaml
execution:
  mounts:
    auxiliary:
      judge:
        /shared/cache/huggingface: /root/.cache/huggingface
```

The key under `auxiliary` must match the auxiliary deployment name.

(aux-multi-instance)=
## Multi-Instance Deployments

For higher throughput, run multiple instances of an auxiliary deployment with HAProxy load balancing:

```yaml
defaults:
  - deployment/vllm@auxiliary_deployments.judge
  # ... other defaults ...
  - _self_

auxiliary_deployments:
  judge:
    served_model_name: judge-model
    port: 8001
    num_nodes: 4       # Total nodes for this auxiliary
    num_instances: 2   # 2 instances of 2 nodes each
```

When `num_instances > 1`:
- `num_nodes` must be divisible by `num_instances`
- HAProxy is automatically configured to load-balance across instances
- The evaluation container connects through the proxy (proxy ports start at 5010)

## Reserved Environment Prefixes

The following `env_prefix` values are reserved and cannot be used for auxiliary deployments:

- `SLURM`
- `PRIMARY`
- `MODEL`
- `SERVER`

(aux-working-example)=
## Complete Working Example

See [`examples/auxiliary-endpoint/slurm_vllm_judge.yaml`](../../../../packages/nemo-evaluator-launcher/examples/auxiliary-endpoint/slurm_vllm_judge.yaml) for a complete configuration that deploys a vLLM judge model for AIME LLM-as-a-judge evaluation.

```bash
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/auxiliary-endpoint/slurm_vllm_judge.yaml \
    -o execution.hostname=my-cluster.com \
    -o execution.account=my-account \
    -o execution.output_dir=/shared/results \
    -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10  # Test run
```
