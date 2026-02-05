---
name: nel-config-generator
description: Interactively generate evaluation configuration YAML files for NeMo Evaluator Launcher (NEL). Use when the user wants to create an evaluation config, set up an evaluation from existing configs, or modify a nel config.
license: Apache-2.0
---

## NeMo Evaluator Launcher Config Generator

You're an expert in generating NeMo Evaluator Launcher configs! Guide the user through creating production-ready YAML configurations via an interactive workflow specified below.

### Running Skill Scripts

This skill includes helper scripts in its `scripts/` directory. First, use `Glob` to find `nel-config-generator/SKILL.md` and determine the skill directory path (referred to as `SKILL_DIR` below). Then run scripts using absolute paths:

```bash
python <SKILL_DIR>/scripts/<script-name>.py <args>
```

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
  Allow for multiple choices in this question.
  1. Standard LLM Benchmarks (like MMLU, IFEval, GSM8K, ...)
  2. Code Evaluation (like HumanEval, MBPP, and LiveCodeBench)
  3. Math & Reasoning (like AIME, GPQA, MATH-500, ...)
  4. Safety & Security (like Garak and Safety Harness)
  5. Multilingual (like MMATH, Global MMLU, MMLU-Prox)

DON'T ALLOW FOR ANY OTHER OPTIONS, only the ones listed above under each category (Execution, Deployment, Auto-export, Model type, Benchmarks). YOU HAVE TO GATHER THE ANSWERS for the 5 questions before you can build the base config.

When you have all the answers, run the script to build the base config:

```bash
python <SKILL_DIR>/scripts/build_config.py --execution <local|slurm> --deployment <none|vllm|sglang|nim|trtllm> --model-type <base|chat|reasoning> --benchmarks <standard|code|math_reasoning|safety|multilingual> [--export <none|mlflow|wandb>] [--output <OUTPUT>] --validate
```

Where `--output` depends on what the user provides:

- Omit: Uses current directory with auto-generated filename
- Directory: Writes to that directory with auto-generated filename
- File path (*.yaml): Writes to that specific file

It never overwrites existing files.

**Step 3: Configure model path and parameters**

Ask for model path. Determine type:

- Checkpoint path (starts with `/` or `./`) → set `deployment.checkpoint_path`
- HF handle (e.g., `org/model-name`) → set `deployment.hf_model_handle`

Use WebSearch to find model card (HuggingFace, build.nvidia.com). Extract ANY relevant configuration:

- Sampling params (`temperature`, `top_p`)
- Context length → `deployment.extra_args: "--max-model-len <value>"`
- TP/DP settings (to set them appropriately, AskUserQuestion on how many GPUs the model will be deployed)
- Reasoning config (if applicable):
  - custom system prompts (`/think`, `/no_think`)
  - `params_to_add` for payload modifier (like `"chat_template_kwargs": {"thinking": true}`),
  - reasoning effort (if it's configurable, AskUserQuestion what reasoning effort they want)
  - higher `max_new_tokens`
  - etc.
- Deployment-specific `extra_args` for vLLM/SGLang
- Any other model-specific requirements

Present findings, explain each setting, ask user to confirm or adjust. If no model card found, ask user directly for the above configurations.

Skip verification here - missing values will be filled in Step 4.

**Step 4: Fill in remaining missing values**

- Find all remaining `???` missing values in the config.
- Ask the user only for values that couldn't be auto-discovered from the model card (e.g., SLURM hostname, account, output directory, MLflow/wandb tracking URI and tags).

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python <SKILL_DIR>/scripts/verify_config.py <config_path>`

**Step 5: Confirm tasks (iterative)**

Show tasks in the current config. Loop until the user confirms the task list is final:

1. Tell the user: "Run `nel ls tasks` to see all available tasks" and ask if they want to add/remove tasks.
2. Apply changes.
3. **Verify**: `python <SKILL_DIR>/scripts/verify_config.py <config_path>`.
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

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python <SKILL_DIR>/scripts/verify_config.py <config_path>`

**Step 6: Advanced - Multi-node (Data Parallel)**

If model >120B, suggest multi-node. Explain: "This is DP multi-node - the weights are copied (not distributed) across nodes. One deployment instance per node will be run with HAProxy load-balancing requests."

Ask if user wants multi-node. If yes, ask for node count and configure:

```yaml
execution:
    num_nodes: 4  # 4 nodes = 4 independent deployment instances = 4x throughput
    deployment:
        n_tasks: ${execution.num_nodes}  # Must match num_nodes for multi-instance deployment

deployment:
    multiple_instances: true
```

Common confusion:

- **This is different from `data_parallel_size`**, which controls DP replicas *within* a single node/deployment instance.
- Global data parallelism is `num_nodes x data_parallel_size` (e.g., 2 nodes x 4 DP each = 8 replicas for max throughput).
- With multi-node, `parallelism` in task config is the total concurrent requests across all instances, not per-instance.

YOU MUST VERIFY THE CONFIG BEFORE GOING TO THE NEXT STEP. RESOLVE ANY ISSUES WITH THE CONFIG BEFORE GOING TO THE NEXT STEP. RUN: `python <SKILL_DIR>/scripts/verify_config.py <config_path>`

**Step 7: Advanced - Interceptors**

Tell the user they should see: https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html and continue.

**Step 8: Run the evaluation**

Print the following three commands to the user. Propose to execute them in order to confirm the config works as expected before the full run.

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

After the dry-run, check the output from `nel` for any problems with the config. If there are no problems, propose to first execute the test run with limited samples and then execute the full evaluation. If there are problems, resolve them before executing the full evaluation.

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
