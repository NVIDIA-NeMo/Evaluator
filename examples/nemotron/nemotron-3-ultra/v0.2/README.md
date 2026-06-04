# Nemotron 3 Ultra 550B A55B — v0.2 evaluation recipes

Evaluation configs for the Nemotron 3 Ultra model-card benchmarks on
`nemo-evaluator-launcher` 0.2.6. Run against any OpenAI-compatible endpoint (NVCF,
vLLM, SGLang), or self-host the model.

This stack has three kinds of config:

- **Instruct suite** — one monolithic config, `local_nemotron-3-ultra-550b-a55b.yaml`;
  run the whole suite or a single benchmark with `-t <task>`.
- **Gym agentic** — one config per benchmark, `local_nemotron-3-ultra-550b-a55b_<bench>.yaml`
  (LiveCodeBench, Tau2, GDPVal).
- **Base suite** — `local_nemotron-3-ultra-550b-a55b-base.yaml` (local vLLM).

For the cross-generation overview and the full benchmark table, see
[`../reproducibility.md`](../reproducibility.md).

---

## Common setup

### 1. Install `nemo-evaluator-launcher`

```bash
python3 -m venv .venv
.venv/bin/pip install "nemo-evaluator-launcher[all]==0.2.6"
```

### 2. Configure credentials & endpoints

Copy `.env.example` to `.env` and fill in the env vars. Each benchmark below lists
exactly which `.env` keys and (for Gym agentic) `_endpoints.yaml` blocks to set.

### 3. Policy model

Required by every benchmark — the model under test:

| `.env` key | Config block | Set / recommended model |
|---|---|---|
| `INFERENCE_API_KEY` | `_endpoints.yaml` → `target.api_endpoint` (instruct + Gym agentic) | endpoint `url` + `model_id` — your model under test |
| `HF_TOKEN` | `.env` (`evaluation.env_vars`) | HuggingFace token (dataset access) |

The **base suite** is the exception — it deploys its own local vLLM
(`deployment: vllm`) and needs no policy API key, only `HF_TOKEN`.

Default instruct params (set in the config): `temperature 1.0`, `top_p 0.95`,
`parallelism 1`, `request_timeout 3600`, `max_retries 10`, thinking on. To self-host,
point `target.api_endpoint.url` at your vLLM endpoint; full serve commands (BF16 / FP8
/ NVFP4 — parsers, TP, env vars) are in the header of
`local_nemotron-3-ultra-550b-a55b.yaml`.

### 4. Judge / simulator endpoints

Benchmarks that use a judge or simulator read its endpoint from the config (instruct)
or `_endpoints.yaml` (Gym agentic); see per-benchmark below for which key/block to
fill. Judges are self-supplied: fill each judge `url`/`model_id` placeholder.

### 5. Bind mounts (`extra_docker_args`)

The cache mounts are optional but recommended (they cache datasets/wheels across
runs):

```yaml
execution:
  extra_docker_args: >-
    -v /home/you/.cache/huggingface:/cache/huggingface
    -v /home/you/.cache/uv:/cache/uv
```

---

## Running

Do the [Common setup](#common-setup) once (install, `.env`, `_endpoints.yaml`, mounts),
then:

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env             # full instruct suite
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env --dry-run   # preview
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_gpqa  # one instruct benchmark
nel run --config local_nemotron-3-ultra-550b-a55b-base.yaml --env-file .env        # base suite (local vLLM, BF16-Base, TP=8)
```

Gym agentic benchmarks are run from their own config file (see below).

### Output

Results are written to each config's output dir (`artifacts/<task>/results.json`);
the Gym recipes default to `${HOME}/nel-results/<benchmark>/`. Override with
`-o "execution.output_dir=/absolute/path"`. For `resume_from_cache` to survive
retries, `output_dir` must be an absolute path on stable storage (not tmpfs).

---

## Benchmarks

### Instruct suite

One monolithic config — `local_nemotron-3-ultra-550b-a55b.yaml` — with one benchmark
per section below. Select a single benchmark with `-t <task>` (note the mixed
`nemo_skills.` prefix on some tasks); every command shares this config and differs only
in `-t`.

Every benchmark needs the [policy model](#3-policy-model). Judge-based benchmarks also
need a judge — its **API key** comes from `.env` (add it there; these judge keys are not
in `.env.example`) and its **`url`/`model_id`** are filled inline in that task's
`extra.judge` block in the config. Fill both before running.

#### GPQA (Diamond)

- **Endpoints / keys:** policy model only.
- **Recommended:** `num_repeats: 8`; `++prompt_config=eval/aai/mcq-4choices`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_gpqa
```

#### HLE

- **Endpoints / keys:** `JUDGE_API_KEY` in `.env`; fill the judge `url` in the
  `nemo_skills.ns_hle_aa` task's `extra.judge` block (`model_id` defaults to **gpt-4o**).
- **Recommended:** `num_repeats: 1`; `hle_strict_judge: true`; `++server.enable_soft_fail=True`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t nemo_skills.ns_hle_aa
```

#### MMLU-Pro

- **Endpoints / keys:** policy model only.
- **Recommended:** `num_repeats: 1`; 10-choice boxed (`++prompt_config=eval/aai/mcq-10choices-boxed`).

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_mmlu_pro
```

#### MMLU-Pro X

- **Endpoints / keys:** policy model only.
- **Recommended:** `num_repeats: 1`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_mmlu_prox
```

#### SciCode

- **Endpoints / keys:** policy model only.
- **Recommended:** `num_repeats: 8`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_scicode
```

#### CritPt

- **Endpoints / keys:** `ARTIFICIAL_ANALYSIS_API_KEY` in `.env` — Artificial Analysis grades the runs.
- **Recommended:** `num_repeats: 5`. AA grades fixed 70-problem batches (no sub-sampling).

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t nemo_skills.ns_critpt
```

#### IFBench

- **Endpoints / keys:** policy model only.
- **Recommended:** `num_repeats: 8`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t nemo_skills.ns_ifbench
```

#### AA-LCR

- **Endpoints / keys:** `QWEN_API_KEY` in `.env` (mapped to `JUDGE_API_KEY` for this task);
  fill the judge `url`/`model_id` in the `ns_aa_lcr` task's `extra.judge` block — used a
  **non-reasoning `qwen/qwen3-235b-a22b`**.
- **Recommended:** `num_repeats: 16`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_aa_lcr
```

#### AA-Omniscience

- **Endpoints / keys:** `GEMINI_API_KEY` in `.env` (mapped to `JUDGE_API_KEY` for this task);
  fill the judge `url` in the `nemo_skills.ns_omniscience` task's `extra.judge` block
  (`model_id` defaults to **`gemini-3-flash-preview`**).
- **Recommended:** `num_repeats: 10`; `++parse_reasoning=False`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t nemo_skills.ns_omniscience
```

#### IMO-AnswerBench

- **Endpoints / keys:** `JUDGE_API_KEY` in `.env` — used **gpt-4o**.
- **Recommended:** `num_repeats: 5`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t nemo_skills.ns_imo_answerbench
```

#### Multi-Challenge

- **Endpoints / keys:** `JUDGE_API_KEY` in `.env` — the config also exposes it as
  `OPENAI_API_KEY` inside the container, so no separate var is needed.
- **Recommended:** `num_repeats: 8`; `attempts: 1`; `seed: 42`.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t multi-challenge
```

#### WMT24++

- **Endpoints / keys:** policy model only — COMET scores locally, no judge endpoint.
- **Prepare:** download [XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL), mount it,
  and set `comet.model_path` in the `ns_wmt24pp_comet` task. COMET runs via
  `--judge_step_fn` (nemo-skills 26.05 ignores `--judge_type=comet`).
- **Recommended:** suite default repeats.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b.yaml --env-file .env -t ns_wmt24pp_comet
```

### LiveCodeBench v6 Cascade

Code generation, executed and graded against test cases. No external judge — only
the common policy endpoint.

```bash
nel run --config local_nemotron-3-ultra-550b-a55b_livecodebench.yaml --env-file .env
```

### Tau2 (TauBench V3)

Tool-use conversations across telecom / airline / retail, with a simulated customer.

| `.env` key | `_endpoints.yaml` block | Set / recommended model |
|---|---|---|
| `TAU2_SIMULATOR_API_KEY` | `tau2_simulator` | simulator `base_url` + `model` — used **GPT-5.2** (Tau2-Bench recommends Qwen; we used GPT-5.2 for reported runs) |

```bash
nel run --config local_nemotron-3-ultra-550b-a55b_tau2.yaml --env-file .env
```

### GDPVal (Stirrup agent)

An agent produces office/PDF deliverables in a sandboxed code-exec environment;
a pairwise/rubric judge scores them. This is the heaviest benchmark — it needs an
Apptainer SIF and extra bind mounts. Keep thinking mode on (non-thinking loses ~86%
of pairwise judgements).

| `.env` key | `_endpoints.yaml` block | Set / recommended model |
|---|---|---|
| `GDPVAL_JUDGE_API_KEY` | `gdpval_judge` | judge `base_url` + `model` — used **Gemini 3.1 Pro** |
| `TAVILY_API_KEY` | — | agent web-search API key |

**Build the SIF once:**

```bash
apptainer build python-3.12.gdpval.sif \
  https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/2502893977e9e9af84adc1fa8d38c9314208d3ee/responses_api_agents/stirrup_agent/containers/gdpval.def
```

Apptainer runs the SIF sandbox via setuid Apptainer under `--privileged` — the
config ships this, and it's
[Apptainer's recommendation for running inside Docker](https://apptainer.org/docs/admin/main/installation.html#running-inside-docker):

```yaml
execution:
  extra_docker_args: >-
    --privileged
    -v /you/python-3.12.gdpval.sif:/gdpval/sif/python-3.12.gdpval.sif
    # ...other mounts
```

To run without `--privileged`, see [Appendix: rootless Apptainer](#appendix--rootless-apptainer-without---privileged).

**Running without the Apptainer sandbox (fallback, not recommended).** Running the
sandbox is **strongly recommended**. If you can't (e.g. Apptainer/`--privileged` is
unavailable on your host), you can run GDPVal unsandboxed: omit the SIF bind mount
and remove the `GDPVAL_CONTAINER_PATH` env var from the config (it defaults to
`null`, so the agent executes generated code locally). `--privileged` and
`/dev/fuse` are then unnecessary too.

> ⚠️ **Disclaimer:** this runs the agent's generated code directly in the eval
> container with no sandbox isolation, and the execution environment differs from
> the reference sandbox — **results may not be comparable to sandboxed scores.** Use
> only for smoke-testing the pipeline, not for reported numbers.

**Scoring mode.** GDPVal is primarily run in `comparison` mode (pairwise vs a
baseline's deliverables). It's a two-step flow: first produce baseline deliverables
in `rubric` mode, then run Ultra in `comparison` mode against them.

Step 1 — baseline deliverables (a model of your choice, `rubric` mode). In the
config, set the baseline model in `_endpoints.yaml` `target.api_endpoint` and keep
`gdpval.reward_mode: rubric`:

```yaml
# _endpoints.yaml
target:
  api_endpoint:
    url: <baseline-endpoint-url>
    model_id: <baseline-model>

# gdpval config
gdpval:
  reward_mode: rubric
```

Then run with a distinct output dir (deliverables land in `.../gdpval/deliverables`):

```bash
nel run --config local_nemotron-3-ultra-550b-a55b_gdpval.yaml --env-file .env \
  -o execution.output_dir=$HOME/nel-results/gdpval-baseline
```

Step 2 — Ultra in `comparison` mode. In the config, set the scoring block:

```yaml
gdpval:
  reward_mode: comparison
  reference_elo: <baseline-elo>   # ELO of the baseline model
```

and point the reference mount in `extra_docker_args` at the baseline deliverables
(`/gdpval/refs/test_ref` is fixed — don't change it):

```yaml
execution:
  extra_docker_args: >-
    --privileged
    -v $HOME/nel-results/gdpval-baseline/gdpval/deliverables:/gdpval/refs/test_ref
    # ...other mounts (SIF, caches)
```

Then run:

```bash
nel run --config local_nemotron-3-ultra-550b-a55b_gdpval.yaml --env-file .env
```

### Base suite

Pretraining benchmarks (MMLU, GSM8K, HumanEval, RULER, … — 21 tasks) run against a
local vLLM endpoint (BF16-Base, TP=8):

```bash
nel run --config local_nemotron-3-ultra-550b-a55b-base.yaml --env-file .env
```

---

## Monitoring & customization

```bash
nel status ; nel logs <job-id>                                                              # status / logs
nel run --config <cfg> -o execution.output_dir=/abs/path                                    # override output
nel run --config <cfg> -o evaluation.nemo_evaluator_config.config.params.limit_samples=10   # smoke test only — never for reported numbers
```

## Troubleshooting

- **Keys not found** — export them or use `.env`; names must match exactly.
- **Timeouts** — `request_timeout` is 360 / 3600 / 36000s (base / instruct / agentic); on rate limits lower `parallelism` / `max_concurrent`.
- **Judge** — judge `url` must be reachable and its key valid.
- **Gated HF datasets** — accept the terms on the Hub; `HF_TOKEN` must have access.
- **Gym** — recipes repair the image's per-server venvs at runtime; if a server won't start, check the pinned Gym commit and `ray==2.49.2`.

## Expected results

Comparable to the model card; expect slight variation (`temperature > 0` with
`num_repeats` consensus), and LLM-judged scores depend on the judge you supply. Never
report `limit_samples` runs.

---

## Appendix — rootless Apptainer (without `--privileged`)

For GDPVal, if `--privileged` is not allowed on your host. Requires a one-time host
setting:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

Then install Apptainer as unprivileged static binaries (no setuid), run the agent as
a **non-root** user, and replace `--privileged` in `extra_docker_args` with:

```yaml
    --security-opt seccomp=unconfined
    --security-opt systempaths=unconfined
    --device /dev/fuse
```

## License

Apache 2.0 — see the repository `LICENSE`.
