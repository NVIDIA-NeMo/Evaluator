---
orphan: true
---

(create-framework-definition-file)=

# Tutorial: Create a Framework Definition File

Learn by building a complete FDF for a simple evaluation framework.

**What you'll build**: An FDF that wraps a hypothetical CLI tool called `domain-eval`

**Time**: 20 minutes

**Prerequisites**:

- Python evaluation framework with a CLI
- Basic YAML knowledge
- Understanding of your framework's parameters

## What You're Creating

By the end, you'll have integrated your evaluation framework with {{ product_name_short }}, allowing users to run:

```bash
nemo-evaluator run_eval \
  --eval_type domain_specific_task \
  --model_id meta/llama-3.1-8b-instruct \
  --model_url https://integrate.api.nvidia.com/v1/chat/completions \
  --model_type chat
```

---

## Step 1: Understand Your Framework

First, document your framework's CLI interface. For our example `domain-eval`:

```bash
# How your CLI currently works
domain-eval \
  --model-name gpt-4 \
  --api-url https://api.example.com/v1/chat/completions \
  --task medical_qa \
  --temperature 0.0 \
  --output-dir ./results
```

**Action**: Write down your own framework's command structure.

---

## Step 2: Create the Directory Structure

```bash
mkdir -p my-framework/core_evals/domain_eval
cd my-framework/core_evals/domain_eval
touch framework.yml output.py __init__.py
```

**Why this structure?** {{ product_name_short }} discovers frameworks by scanning `core_evals/` directories.

---

## Step 3: Add Framework Identification

Create `framework.yml` and start with the identification section:

```yaml
framework:
  name: domain-eval                    # Lowercase, hyphenated
  pkg_name: domain_eval                # Python package name
  full_name: Domain Evaluation Framework
  description: Evaluates models on domain-specific medical and legal tasks
  url: https://github.com/example/domain-eval
```

**Why these fields?**

- `name`: Used in CLI commands (`--framework domain-eval`)
- `pkg_name`: Used for Python imports
- `full_name`: Shows in documentation
- `url`: Links users to your source code

**Test**: This minimal FDF should now be discoverable (but not runnable yet).

---

## Step 4: Map CLI Parameters to Template Variables

Now map your CLI to {{ product_name_short }}'s configuration structure:

| Your CLI Flag | Maps To | FDF Template Variable |
|---------------|---------|----------------------|
| `--model-name` | Model ID | `{{target.api_endpoint.model_id}}` |
| `--api-url` | Endpoint URL | `{{target.api_endpoint.url}}` |
| `--task` | Task name | `{{config.params.task}}` |
| `--temperature` | Temperature | `{{config.params.temperature}}` |
| `--output-dir` | Output path | `{{config.output_dir}}` |

**Action**: Create this mapping for your own framework.

---

## Step 5: Write the Command Template

Add the `defaults` section with your command template:

```yaml
defaults:
  command: >-
    {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
    domain-eval 
      --model-name {{target.api_endpoint.model_id}}
      --api-url {{target.api_endpoint.url}}
      --task {{config.params.task}}
      --temperature {{config.params.temperature}}
      --output-dir {{config.output_dir}}
```

**Understanding the template**:

- `{% if ... %}`: Conditional - exports API key if provided
- `{{variable}}`: Placeholder filled with actual values at runtime
- Line breaks are optional (using `>-` makes it readable)

**Common pattern**: Export environment variables before the command runs.

---

## Step 6: Define Default Parameters

Add default configuration values:

```yaml
defaults:
  command: >-
    # ... command from previous step ...
  
  config:
    params:
      temperature: 0.0           # Deterministic by default
      max_new_tokens: 1024       # Token limit
      parallelism: 10            # Concurrent requests
      max_retries: 5             # API retry attempts
      request_timeout: 60        # Seconds
  
  target:
    api_endpoint:
      type: chat                 # Default to chat endpoint
```

**Why defaults?** Users can run evaluations without specifying every parameter.

---

## Step 7: Define Your Evaluation Tasks

Add the specific tasks your framework supports:

```yaml
evaluations:
  - name: medical_qa
    description: Medical question answering evaluation
    defaults:
      config:
        type: medical_qa
        supported_endpoint_types:
          - chat
        params:
          task: medical_qa       # Passed to --task flag

  - name: legal_reasoning
    description: Legal reasoning and case analysis
    defaults:
      config:
        type: legal_reasoning
        supported_endpoint_types:
          - chat
          - completions          # Supports both endpoint types
        params:
          task: legal_reasoning
          temperature: 0.0       # Override for deterministic reasoning
```

**Key points**:

- Each evaluation has a unique `name` (used in CLI)
- `supported_endpoint_types` declares API compatibility
- Task-specific `params` override framework defaults

---

## Step 8: Create the Output Parser

Create `output.py` to parse your framework's results:

```python
def parse_output(output_dir: str) -> dict:
    """Parse evaluation results from your framework's output format."""
    import json
    from pathlib import Path
    
    # Adapt this to your framework's output format
    results_file = Path(output_dir) / "results.json"
    
    with open(results_file) as f:
        raw_results = json.load(f)
    
    # Convert to {{ product_name_short }} format
    return {
        "tasks": {
            "medical_qa": {
                "name": "medical_qa",
                "metrics": {
                    "accuracy": raw_results["accuracy"],
                    "f1_score": raw_results["f1"]
                }
            }
        }
    }
```

**What this does**: Translates your framework's output format into {{ product_name_short }}'s standard schema.

---

## Step 9: Test Your FDF

Install your framework package and test:

```bash
# From your-framework/ directory
pip install -e .

# List available evaluations (should show your tasks)
eval-factory list_evals --framework domain-eval

# Run a test evaluation
nemo-evaluator run_eval \
  --eval_type medical_qa \
  --model_id gpt-3.5-turbo \
  --model_url https://api.openai.com/v1/chat/completions \
  --model_type chat \
  --api_key_name OPENAI_API_KEY \
  --output_dir ./test_results \
  --overrides "config.params.limit_samples=5"
```

**Expected output**: Your CLI should execute with substituted parameters.

---

## Step 10: Add Conditional Logic (Advanced)

Make parameters optional with Jinja2 conditionals:

```yaml
defaults:
  command: >-
    domain-eval 
      --model-name {{target.api_endpoint.model_id}}
      --api-url {{target.api_endpoint.url}}
      {% if config.params.task is not none %}--task {{config.params.task}}{% endif %}
      {% if config.params.temperature is not none %}--temperature {{config.params.temperature}}{% endif %}
      {% if config.params.limit_samples is not none %}--num-samples {{config.params.limit_samples}}{% endif %}
      --output-dir {{config.output_dir}}
```

**When to use conditionals**: For optional flags that shouldn't appear if not specified.

---

## Complete Example

Here's your full `framework.yml`:

```yaml
framework:
  name: domain-eval
  pkg_name: domain_eval
  full_name: Domain Evaluation Framework
  description: Evaluates models on domain-specific tasks
  url: https://github.com/example/domain-eval

defaults:
  command: >-
    {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
    domain-eval 
      --model-name {{target.api_endpoint.model_id}}
      --api-url {{target.api_endpoint.url}}
      --task {{config.params.task}}
      --temperature {{config.params.temperature}}
      --output-dir {{config.output_dir}}
  
  config:
    params:
      temperature: 0.0
      max_new_tokens: 1024
      parallelism: 10
      max_retries: 5
      request_timeout: 60
  
  target:
    api_endpoint:
      type: chat

evaluations:
  - name: medical_qa
    description: Medical question answering
    defaults:
      config:
        type: medical_qa
        supported_endpoint_types:
          - chat
        params:
          task: medical_qa
  
  - name: legal_reasoning
    description: Legal reasoning tasks
    defaults:
      config:
        type: legal_reasoning
        supported_endpoint_types:
          - chat
          - completions
        params:
          task: legal_reasoning
```

---

## Next Steps

**Dive deeper into FDF features**: {ref}`framework-definition-file`

**Learn about advanced templating**: {ref}`advanced-features`

**Share your framework**: Package and distribute via PyPI

**Troubleshooting**: {ref}`fdf-troubleshooting`

---

## Common Patterns

### Pattern 1: Framework with Custom CLI Flags

```yaml
command: >-
  my-eval --custom-flag {{config.params.extra.custom_value}}
```

Use `extra` dict for framework-specific parameters.

### Pattern 2: Multiple Output Files

```yaml
command: >-
  my-eval --results {{config.output_dir}}/results.json
          --logs {{config.output_dir}}/logs.txt
```

Organize outputs in subdirectories using `output_dir`.

### Pattern 3: Environment Variable Setup

```yaml
command: >-
  export HF_TOKEN=${{target.api_endpoint.api_key}} && 
  export TOKENIZERS_PARALLELISM=false && 
  my-eval ...
```

Set environment variables before execution.

---

## Summary

You've learned how to:

✅ Create the FDF directory structure  
✅ Map your CLI to template variables  
✅ Write Jinja2 command templates  
✅ Define default parameters  
✅ Declare evaluation tasks  
✅ Create output parsers  
✅ Test your integration  

**Your framework is now integrated with {{ product_name_short }}!**

