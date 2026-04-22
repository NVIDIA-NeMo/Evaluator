# Config Composition

NEL supports Hydra-style config composition: reusable fragments, base-config
inheritance, and self-referencing interpolation — with zero external
dependencies and full backward compatibility.

**Existing flat YAML configs continue to work unchanged.** Composition is
opt-in via the `defaults:` key.

## Quick example

Instead of duplicating 100+ lines across variant configs, inherit from a base:

```yaml
# 24a_miniswe.yaml — 15 lines instead of 106
defaults:
  - _base_: 24_slurm_swebench_verified_super

benchmarks:
  - name: harbor://swebench-verified@1.0
    max_concurrent: 30
    solver:
      agent: mini-swe-agent
      agent_kwargs: { max_iterations: 30 }
    sandbox: { concurrency: 30 }

output:
  dir: ./eval_results/swebench-verified-miniswe
```

## `defaults:` list

Add a `defaults:` key at the top of your config. Each entry loads and merges a
config fragment or base config:

```yaml
defaults:
  - services/nemotron_fp8_vllm     # loads conf/services/nemotron_fp8_vllm.yaml
  - clusters/slurm_oci_8gpu        # loads conf/clusters/slurm_oci_8gpu.yaml
  - sandboxes/ecs_swebench         # loads conf/sandboxes/ecs_swebench.yaml
  - _self_                         # where *this* config sits in merge order

benchmarks:
  - name: harbor://swebench-verified@1.0
    solver: { type: harbor, service: nemotron }
```

`_self_` marks where the main config body sits. Default (when omitted): last,
meaning your config has the highest priority.

## Config groups

The directory name in a fragment path maps to the top-level config key:

| Fragment path              | Merges into  |
|----------------------------|--------------|
| `services/nemotron_fp8`    | `services:`  |
| `clusters/slurm_8gpu`      | `cluster:`   |
| `sandboxes/ecs_swebench`  | `sandboxes:` |
| `output/lustre`            | `output:`    |

> **Note:** `clusters/` (plural) maps to `cluster:` (singular). All other group
> names match their key directly.

### Fragment file structure

A fragment file contains the content for its section, not the wrapping key:

```yaml
# conf/clusters/slurm_oci_8gpu.yaml
type: slurm
hostname: slurm-login.example.com
account: your-slurm-account
walltime: "04:00:00"
node_pools:
  gpu:
    partition: batch
    gres: "gpu:8"
```

For `services/`, include the service name as a key (since multiple services can
coexist):

```yaml
# conf/services/nemotron_fp8_vllm.yaml
nemotron:
  type: vllm
  model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
  # ... full config ...
```

## Base-config inheritance

Use `_base_:` to inherit from another complete config:

```yaml
defaults:
  - _base_: 24_slurm_swebench_verified_super

# Only specify what differs
benchmarks:
  - name: harbor://swebench-verified@1.0
    solver:
      agent: mini-swe-agent
output:
  dir: /path/to/different/output
```

The base config is loaded and recursively composed (including its own
`defaults:`), then the current config's body is deep-merged on top.

Chains work too: grandparent → parent → child.

## Merge semantics

- **Dicts**: recursive deep merge (later values win on conflict)
- **Lists**: replace entirely (no concatenation)
- **Scalars**: later value wins
- **`null`**: explicitly deletes a key from the base

```yaml
defaults:
  - _base_: base_config

output:
  extra_field: null     # removes extra_field inherited from base
  dir: /new/output      # overrides base's dir
```

## Self-referencing interpolation

Reference other values in the merged config with `${.path.to.value}`:

```yaml
services:
  model:
    type: vllm
    model: nvidia/Llama-3.1-70B

output:
  dir: ./results/${.services.model.model}    # → ./results/nvidia/Llama-3.1-70B
```

The leading dot distinguishes self-references from environment variables:
- `${.services.model.model}` → self-reference (resolved during composition)
- `${HF_TOKEN}` → env var (resolved later by `_expand_env`)

### Resolution order

1. **Composition** — `defaults:` fragments are merged and `_base_:` inheritance applied
2. **Self-refs** — `${.path}` references resolved against the merged dict
3. **CLI overrides** — `-O key=value` applied
4. **Pydantic validation** — schema checked, types coerced
5. **Env-var expansion** — `${HF_TOKEN}` resolved at runtime (not during composition)

This means self-refs see the fully-merged config but *not* CLI overrides.
Env vars are intentionally deferred so secrets stay out of serialized configs.

## Search path

Fragments are resolved in order (first match wins):

1. `conf/` relative to the config file
2. `~/.config/nemo-evaluator/conf/`
3. Package built-in defaults (shipped with NEL)

For `_base_:` references, the directory containing the referencing config is
searched first.

File extensions `.yaml` and `.yml` are tried automatically.

## CLI overrides

`-O key=value` overrides are applied **after** composition, so they always win:

```bash
nel eval run config.yaml -O output.dir=/tmp/test -O benchmarks.0.max_problems=5
```

## Creating your own fragment library

```
my-project/
  configs/
    conf/
      services/
        nemotron_fp8.yaml
        llama_70b.yaml
      clusters/
        slurm_8gpu.yaml
        slurm_4gpu.yaml
      sandboxes/
        ecs_swebench.yaml
    swebench_openhands.yaml       # uses defaults: to compose
    swebench_miniswe.yaml         # _base_: swebench_openhands
```

Or share team-wide fragments via `~/.config/nemo-evaluator/conf/`.

## Error handling

- **Missing fragment**: `FileNotFoundError` with the fragment name and all
  search paths that were tried
- **Circular reference**: `ValueError` if config A references B which
  references A
- **Invalid entry**: `ValueError` for unrecognized `defaults:` entries
