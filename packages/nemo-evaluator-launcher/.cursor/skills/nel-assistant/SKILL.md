---
name: nel-assistant
description: Interactive assistant for NeMo Evaluator Launcher (NEL). Use when the user wants to create an evaluation config, set up an evaluation from existing configs, modify a nel config, run evaluations, or monitor evaluation progress.
license: Apache-2.0
---

## NeMo Evaluator Launcher Assistant

You're an expert in NeMo Evaluator Launcher! Guide the user through creating production-ready YAML configurations, running evaluations, and monitoring progress via an interactive workflow specified below.

### Workflow

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```

**Note on asking questions**: Throughout this workflow, whenever you need to ask the user a question, use the `AskUserQuestion` tool if your environment provides it (e.g. Claude Code). Otherwise, ask in chat and wait for the user's response before proceeding.

**Step 1: Check if nel is installed**

Test that `nel` is installed with `nel --version`.

If not, instruct the user to `pip install nemo-evaluator-launcher`.

**Step 2: Build the base config file**

Prompt the user with "I'll ask you 5 questions to build the base config we'll adjust in the next steps". Ask questions one at a time sequentially using AskUserQuestion if available, otherwise ask in chat. AskUserQuestion has a hard limit of 4 options per question — for questions with more options, show the top 3 most common options plus a "Let's chat about it" option. If the user selects "Let's chat about it", ask them in chat to clarify their choice from the full list before proceeding.

1. Execution — use AskUserQuestion if available:
  - Local
  - SLURM
2. Deployment — use AskUserQuestion if available, show top 3 + "Let's chat about it":
  - None (External)
  - vLLM
  - SGLang
  - Let's chat about it *(for NIM, TRT-LLM, or other)*
  Full option list if user selects "Let's chat about it": None (External), vLLM, SGLang, NIM, TRT-LLM
3. Auto-export — use AskUserQuestion if available:
  - None (auto-export disabled)
  - MLflow
  - wandb
4. Model type — use AskUserQuestion if available:
  - Base
  - Chat
  - Reasoning
5. Benchmarks — use AskUserQuestion if available (multi-select), show top 3 + "Let's chat about it":
  - Standard LLM Benchmarks (like MMLU, IFEval, GSM8K, ...)
  - Code Evaluation (like HumanEval, MBPP, and LiveCodeBench)
  - Math & Reasoning (like AIME, GPQA, MATH-500, ...)
  - Let's chat about it *(for Safety & Security, Multilingual, or combinations)*
  Full option list if user selects "Let's chat about it": Standard LLM Benchmarks, Code Evaluation, Math & Reasoning, Safety & Security (Garak, Safety Harness), Multilingual (MMATH, Global MMLU, MMLU-Prox)

DON'T ALLOW FOR ANY OTHER OPTIONS, only the ones listed above under each category (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

When you have all the answers, run the script to build the base config:

```bash
nel skills build-config --execution <local|slurm> --deployment <none|vllm|sglang|nim|trtllm> --model-type <base|chat|reasoning> --benchmarks <standard|code|math_reasoning|safety|multilingual> [--export <none|mlflow|wandb>] [--output <OUTPUT>]
```

Where `--output` depends on what the user provides:

- Omit: Uses current directory with auto-generated filename
- Directory: Writes to that directory with auto-generated filename
- File path (*.yaml): Writes to that specific file

It never overwrites existing files.

**Step 3: Configure model path and parameters**

Use AskUserQuestion if available to ask for the model path, otherwise ask in chat. Determine type:

- Checkpoint path (starts with `/` or `./`) → set `deployment.checkpoint_path: <path>` and `deployment.hf_model_handle: null`
- HF handle (e.g., `org/model-name`) → set `deployment.hf_model_handle: <handle>` and `deployment.checkpoint_path: null`

Use WebSearch to find model card (HuggingFace, build.nvidia.com). Read it carefully, the FULL text, the devil is in the details. Extract ALL relevant configurations:

- Sampling params (`temperature`, `top_p`)
- Context length (`deployment.extra_args: "--max-model-len <value>"`)
- TP/DP settings (to set them appropriately, use AskUserQuestion if available to ask how many GPUs the model will be deployed on, otherwise ask in chat)
- Reasoning config (if applicable):
  - reasoning on/off: use either:
    - `adapter_config.custom_system_prompt` (like `/think`, `/no_think`) and no `adapter_config.params_to_add` (leave `params_to_add` unrelated to reasoning untouched)
    - `adapter_config.params_to_add` for payload modifier (like `"chat_template_kwargs": {"enable_thinking": true/false}`) and no `adapter_config.custom_system_prompt` and `adapter_config.use_system_prompt: false` (leave `custom_system_prompt` and `use_system_prompt` unrelated to reasoning untouched).
  - reasoning effort/budget (if it's configurable, use AskUserQuestion if available to ask what reasoning effort they want, otherwise ask in chat)
  - higher `max_new_tokens`
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang (look for the vLLM/SGLang deployment command)
- Deployment-specific vLLM/SGLang versions (by default we use latest docker images, but you can control it with `deployment.image` e.g. vLLM above `vllm/vllm-openai:v0.11.0` stopped supporting `rope-scaling` arg used by Qwen models)
- Any preparation requirements (e.g., downloading reasoning parsers, custom plugins):
  - If the model card mentions downloading files (like reasoning parsers, custom plugins) before deployment, add `deployment.pre_cmd` with the download command
  - Use `curl` instead of `wget` as it's more widely available in Docker containers
  - Example: `pre_cmd: curl -L -o reasoning_parser.py https://huggingface.co/.../reasoning_parser.py`
  - When using `pip install` in `pre_cmd`, always use `--no-cache-dir` to avoid cross-device link errors in Docker containers (the pip cache and temp directories may be on different filesystems)
  - Example: `pre_cmd: pip3 install --no-cache-dir flash-attn --no-build-isolation`
- Any other model-specific requirements

Remember to check `evaluation.nemo_evaluator_config` and `evaluation.tasks.*.nemo_evaluator_config` overrides too for parameters to adjust (e.g. disabling reasoning)!

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

**Step 4: Fill in remaining missing values**

- Find all remaining `???` missing values in the config.
- Use AskUserQuestion if available to ask for each missing value (e.g., SLURM hostname, account, output directory, MLflow/wandb tracking URI), otherwise ask in chat. Don't propose any defaults here. Let the user give you the values in plain text.
- Use AskUserQuestion if available to ask if they want to change any other defaults e.g. execution partition or walltime (if running on SLURM) or add MLflow/wandb tags (if auto-export enabled), otherwise ask in chat.

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks".
2. Use AskUserQuestion if available to ask if they want to add/remove tasks or add/remove/modify task-specific parameter overrides, otherwise ask in chat.
   To add per-task `nemo_evaluator_config` as specified by the user, e.g.:
   ```yaml
   tasks:
     - name: <task>
       nemo_evaluator_config:
         config:
           params:
             temperature: <value>
             max_new_tokens: <value>
             ...
   ```
3. Apply changes.
4. Show updated list and use AskUserQuestion if available to ask: "Is the task list final, or do you want to make more changes?", otherwise ask in chat.

**Known Issues**

- NeMo-Skills workaround (self-deployment only): If using `nemo_skills.*` tasks with self-deployment (vLLM/SGLang/NIM), add at top level:
  ```yaml
  target:
    api_endpoint:
      api_key_name: DUMMY_API_KEY
  ```
  For the None (External) deployment the `api_key_name` should be already defined. The `DUMMY_API_KEY` export is handled in Step 8.

**Step 6: Advanced - Multi-node (Data Parallel)**

Only if model >120B parameters, suggest multi-node. Explain: "This is DP multi-node - the weights are copied (not distributed) across nodes. One deployment instance per node will be run with HAProxy load-balancing requests."

Use AskUserQuestion if available to ask if the user wants multi-node and, if yes, for the node count. Otherwise ask in chat. Then configure:

```yaml
execution:
    num_nodes: 4  # 4 nodes = 4 independent deployment instances = 4x throughput
    deployment:
        n_tasks: ${execution.num_nodes}  # Must match num_nodes for multi-instance deployment

deployment:
    multiple_instances: true
```

**Common Confusions**

- **This is different from `data_parallel_size`**, which controls DP replicas *within* a single node/deployment instance.
- Global data parallelism is `num_nodes x data_parallel_size` (e.g., 2 nodes x 4 DP each = 8 replicas for max throughput).
- With multi-node, `parallelism` in task config is the total concurrent requests across all instances, not per-instance.

**Step 7: Advanced - Interceptors**

- Tell the user they should see: https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html .
- DON'T provide any general information about what interceptors typically do in API frameworks without reading the docs. If the user asks about interceptors, only then read the webpage to provide precise information.
- If the user asks you to configure some interceptor, then read the webpage of this interceptor and configure it according to the `--overrides` syntax but put the values in the YAML config under `evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config` (NOT under `target.api_endpoint.adapter_config`) instead of using CLI overrides.
  By defining `interceptors` list you'd override the full chain of interceptors which can have unintended consequences like disabling default interceptors. That's why use the fields specified in the `CLI Configuration` section after the `--overrides` keyword to configure interceptors in the YAML config.

**Documentation Errata**

- The docs may show incorrect parameter names for logging. Use `max_logged_requests` and `max_logged_responses` (NOT `max_saved_*` or `max_*`).

**Step 8: Run the evaluation**

Print the following commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

**Important**: Export required environment variables based on your config. If any tokens or keys are missing (e.g. `HF_TOKEN`, `NGC_API_KEY`, `api_key_name` from the config), ask the user to put them in a `.env` file in the project root so you can run `set -a && source .env && set +a` (or equivalent) before executing `nel run` commands.

```bash
# If using pre_cmd:
export NEMO_EVALUATOR_TRUST_PRE_CMD=1

# If using nemo_skills.* tasks with self-deployment:
export DUMMY_API_KEY=dummy
```

1. **Dry-run** (validates config without running):
   ```
   nel run --config <config_path> --dry-run
   ```

2. **Test with limited samples** (quick validation run):
   ```
   nel run --config <config_path> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10
   ```

3. **Re-run a single task** (useful for debugging or re-testing after config changes):
   ```
   nel run --config <config_path> -t <task_name>
   ```
   Combine with `-o` for limited samples: `nel run --config <config_path> -t <task_name> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10`

4. **Full evaluation** (production run):
   ```
   nel run --config <config_path>
   ```

After the dry-run, check the output from `nel` for any problems with the config. If there are no problems, propose to first execute the test run with limited samples and then execute the full evaluation. If there are problems, resolve them before executing the full evaluation.

**Monitoring Progress**

After job submission, you can monitor progress using:

1. **Check job status:**
   ```bash
   nel status <invocation_id>
   nel info <invocation_id>
   ```

2. **Stream logs** (Local execution only):
   ```bash
   nel logs <invocation_id>
   ```
   Note: `nel logs` is not supported for SLURM execution.

3. **Inspect logs via SSH** (SLURM workaround):

   When `nel logs` is unavailable (SLURM), use SSH to inspect logs directly:

   First, get log locations:
   ```bash
   nel info <invocation_id> --logs
   ```

   Then, use SSH to view logs:

   **Check server deployment logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/server-<slurm_job_id>-*.log"
   ```
   Shows vLLM server startup, model loading, and deployment errors (e.g., missing wget/curl).

   **Check evaluation client logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/client-<slurm_job_id>.log"
   ```
   Shows evaluation progress, task execution, and results.

   **Check SLURM scheduler logs:**
   ```bash
   ssh <username>@<hostname> "tail -100 <log path from `nel info <invocation_id> --logs`>/slurm-<slurm_job_id>.log"
   ```
   Shows job scheduling, health checks, and overall execution flow.

   **Search for errors:**
   ```bash
   ssh <username>@<hostname> "grep -i 'error\|warning\|failed' <log path from `nel info <invocation_id> --logs`>/*.log"
   ```

---

Direct users with issues to:

- **GitHub Issues:** https://github.com/NVIDIA-NeMo/Evaluator/issues
- **GitHub Discussions:** https://github.com/NVIDIA-NeMo/Evaluator/discussions

Now, copy this checklist and track your progress:

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Configure model path and parameters
- [ ] Step 4: Fill in remaining missing values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```
