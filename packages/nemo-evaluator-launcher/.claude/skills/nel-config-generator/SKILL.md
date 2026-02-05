---
name: nel-config-generator
description: Interactively generate evaluation configuration YAML files for NeMo Evaluator Launcher (NEL). Use when the user wants to create an evaluation config, set up an evaluation from existing configs, or modify a nel config.
license: Apache-2.0
---

## NeMo Evaluator Launcher Config Generator

You're an expert in generating NeMo Evaluator Launcher configs! Guide the user through creating production-ready YAML configurations via an interactive workflow specified below:

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Fill in the missing values
- [ ] Step 4: Configure model path and parameters
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```

**Step 1: Check if nel is installed**

Test that `nel` is installed with `nel --version`.

If not, instruct the user to `pip install nemo-evaluator-launcher`.

**Step 2: Build the base config file**

Prompt the user with "I'll ask you 5 questions to build the base config we'll adjust in the next steps". Guide the user through the 5 questions using AskUserQuestion:

1. Execution:
  - Local
  - SLURM
2. Deployment:
  - None (External)
  - vLLM
  - SGLang
  - NIM
  - TRT-LLM
3. Auto-export:
  - None (auto-export disabled)
  - MLflow
  - wandb
4. Model type
  - Base
  - Chat
  - Reasoning
5. Benchmarks:
  - Standard LLM Benchmarks (like MMLU, IFEval, GSM8K, ...)
  - Code Evaluation (like HumanEval, MBPP, and LiveCodeBench)
  - Math & Reasoning (like AIME, GPQA, MATH-500, ...)
  - Safety & Security (like Garak and Safety Harness)
  - Multilingual (like MMATH, Global MMLU, MMLU-Prox)

Don't provide any other options, only the ones listed above under each category (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

When you have all the answers, run exactly the script to build the base config:

```
python scripts/build_config.py --execution <local|slurm> --deployment <none|vllm|sglang|nim|trtllm> --model-type <base|chat|reasoning> --benchmarks <standard|code|math_reasoning|safety|multilingual> [--export <none|mlflow|wandb>] [--output <OUTPUT>] --validate
```

Where `--output` depends on what the user provides:

- Omit: Uses current directory with auto-generated filename
- Directory: Writes to that directory with auto-generated filename
- File path (*.yaml): Writes to that specific file

It never overwrites existing files.

**Step 3: Fill in the missing values**

- Find all the `???` missing values in the base config created in the Step 2.
- Ask the user what to enter in place of the missing values.

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python scripts/verify_config.py <config_path>`

**Step 4: Configure model path and parameters**

Ask for model path. Determine type:

- Checkpoint path (starts with `/` or `./`) → set `deployment.checkpoint_path`
- HF handle (e.g., `org/model-name`) → set `deployment.hf_model_handle`

Use WebSearch to find model card (HuggingFace, build.nvidia.com). Extract ANY relevant configuration:

- Sampling params (`temperature`, `top_p`)
- Context length → `deployment.extra_args: "--max-model-len <value>"`
- TP/DP settings
- Reasoning config (if applicable):
  - custom system prompts (`/think`, `/no_think`)
  - `process_reasoning_traces: true`, add it if the deployment DOESN'T have reasoning parser that already processes the reasoning traces
  - `params_to_add` for payload modifier (like `"chat_template_kwargs": {"thinking": true}`),
  - higher `max_new_tokens`
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang
- Any other model-specific requirements

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python scripts/verify_config.py <config_path>`

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks" and ask if they want to add/remove tasks.
2. Apply changes.
3. **Verify**: `python scripts/verify_config.py <config_path>`.
4. Show updated list and ask: "Is the task list final, or do you want to make more changes?"

After task list is confirmed, ask if user wants task-specific parameter overrides. If yes, add per-task `nemo_evaluator_config` as specified by the user, e.g.:

```yaml
tasks:
  - name: <task>
    nemo_evaluator_config:
      config:
        params:
          temperature: <value>
          max_new_tokens: <value>
```

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python scripts/verify_config.py <config_path>`

**Step 6: Advanced - Multi-node (Data Parallel)**

If model >100B, suggest multi-node. Explain: "This is DP multi-node - the weights are copied (not distributed) across nodes. One deployment instance per node will be run with HAProxy load-balancing requests."

Ask if user wants multi-node. If yes, ask for node count and configure:
- `execution.num_nodes: <count>`
- `execution.deployment.n_tasks: ${execution.num_nodes}`
- `deployment.multiple_instances: true`

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python scripts/verify_config.py <config_path>`

**Step 7: Advanced - Interceptors**

Tell the user they should see: https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html and continue.

**Step 8: Run the evaluation**

Print the following three commands for the user to execute in order to confirm the config works as expected before the full run:

1. **Dry-run** (validates config without running):
   ```
   nel run --config <config_path> --dry-run
   ```

2. **Test with limited samples** (quick validation run):
   ```
   nel run --config <config_path> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10
   ```

3. **Full evaluation** (production run):
   ```
   nel run --config <config_path>
   ```

Ask the user to execute these commands in order to confirm the configuration is doing what they expect before the full run.

---

Direct users with issues to:

- **GitHub Issues:** https://github.com/NVIDIA-NeMo/Evaluator/issues
- **GitHub Discussions:** https://github.com/NVIDIA-NeMo/Evaluator/discussions

Now, copy this checklist and track your progress:

```
Config Generation Progress:
- [ ] Step 1: Check if nel is installed
- [ ] Step 2: Build the base config file
- [ ] Step 3: Fill in the missing values
- [ ] Step 4: Configure model path and parameters
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Advanced - Multi-node (Data Parallel)
- [ ] Step 7: Advanced - Interceptors
- [ ] Step 8: Run the evaluation
```

