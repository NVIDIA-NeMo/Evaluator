# Documentation Cleanup Recommendations

**Date:** October 1, 2025  
**Scope:** NeMo Evaluator Library Documentation - Unsubstantiated Content Review

---

## Content Requiring Removal or Revision

### 1. Generic Best Practices (cli.md:311-341)

**Current Content:**
```markdown
### 4. **Performance**
- Use appropriate sample sizes for testing
- Enable caching through adapter configuration
- Monitor resource usage

### 5. **Security**
- Store API keys in environment variables
- Use secure communication channels
- Validate all inputs and configurations
```

**Issues:**
- ❌ "Use appropriate sample sizes" - No guidance on what's appropriate
- ❌ "Monitor resource usage" - No monitoring tool provided
- ❌ "Use secure communication channels" - Generic security advice
- ❌ "Validate all inputs and configurations" - Not enforced by code

**Recommended Action:**

**Option A - Remove entirely**
Delete the Performance and Security sections that don't provide actionable, code-backed guidance.

**Option B - Replace with specific, actionable items**
```markdown
### 4. **Performance**
- Limit samples during development using `--overrides 'config.params.limit_samples=10'`
- Enable caching to reuse responses: `--overrides 'target.api_endpoint.adapter_config.use_caching=True'`
- Use `--dry_run` to validate configuration before running full evaluations

### 5. **Security**
- Pass API keys via environment variables using `--api_key_name` parameter
  ```bash
  export MY_API_KEY=your_key_here
  eval-factory run_eval --api_key_name MY_API_KEY ...
  ```
- API keys are never logged or stored in configuration files
```

---

### 2. Hardware Prerequisites (containers/index.md:84-87)

**Current Content:**
```markdown
### Prerequisites

- Docker or NVIDIA Container Toolkit
- NVIDIA GPU (for GPU-accelerated evaluation)
- Sufficient disk space for models and datasets
```

**Issues:**
- ❌ GPU requirement not enforced in code
- ❌ "Sufficient disk space" is vague and unquantified
- ❌ These are assumptions, not code-validated requirements

**Recommended Action:**

**Replace with:**
```markdown
### Prerequisites

- Docker or container runtime
- NVIDIA Container Toolkit (if using GPU-accelerated containers)
- Network access to model API endpoints

**Note:** Hardware requirements vary by evaluation type and are determined by the evaluation container you choose, not the NeMo Evaluator core library.
```

---

### 3. Vague Use Cases (caching.md:143-149)

**Current Content:**
```markdown
## Use Cases

- **Development Iterations**: Cache responses during prompt engineering and testing
- **Cost Optimization**: Reduce API costs for repeated evaluations
- **Reproducible Results**: Ensure identical responses for identical requests
- **Performance**: Speed up re-runs of large evaluations
- **Offline Analysis**: Enable offline analysis of cached evaluation data
```

**Issues:**
- ⚠️ Marketing language without concrete examples
- ⚠️ No quantification of benefits
- ⚠️ "Cost optimization" and "speed up" are unsubstantiated claims

**Recommended Action:**

**Replace with concrete examples:**
```markdown
## Use Cases

### Development Iterations
Cache responses during prompt engineering to avoid repeated API calls:
```python
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./dev_cache",
                "reuse_cached_responses": True
            }
        )
    ]
)
```

### Reproducible Results
Ensure identical responses for identical requests by caching and reusing responses:
- Same request parameters + same cache = identical response
- Useful for comparing evaluation runs across different times

### Offline Analysis
Save requests and responses for later analysis:
```python
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="caching",
            config={
                "save_requests": True,
                "save_responses": True,
                "cache_dir": "./analysis_data"
            }
        )
    ]
)
```
Cached data is stored in JSON format in `cache_dir` for post-processing.
```

---

### 4. Generic Configuration Management Advice (cli.md:313-316)

**Current Content:**
```markdown
### 1. **Configuration Management**
- Use YAML configuration files for complex setups
- Use environment variables for sensitive data
- Validate configurations before running evaluations
```

**Issues:**
- ⚠️ "Validate configurations" - No validation tool provided
- ⚠️ Generic advice applicable to any software

**Recommended Action:**

**Replace with specific examples:**
```markdown
### 1. **Configuration Management**

**Complex Setups:**
Store reusable configurations in YAML files:
```yaml
# config.yaml
config:
  type: mmlu_pro
  params:
    limit_samples: 100
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
```

**Validate Before Running:**
Use `--dry_run` to preview configuration without executing:
```bash
eval-factory run_eval \
  --run_config config.yaml \
  --output_dir ./results \
  --dry_run
```

**Sensitive Data:**
Pass API keys via environment variables:
```bash
export MY_API_KEY=nvapi-xxx
eval-factory run_eval --api_key_name MY_API_KEY ...
```
```

---

## Content to Keep (Validated Against Code)

### ✅ Substantiated Best Practices

1. **Use `--dry_run` flag** (cli.md:320)
   - Real CLI flag verified in entrypoint.py:79-83

2. **Enable caching through adapter configuration** (cli.md:330)
   - Real feature with CachingInterceptor implementation

3. **Use dot notation for parameter overrides** (cli.md:319)
   - Implemented via deep_update utility in core/utils.py

4. **Use YAML configuration files** (cli.md:314)
   - Supported via --run_config parameter

5. **Store API keys in environment variables** (cli.md:334)
   - Supported via --api_key_name parameter and ApiEndpoint.api_key field

---

## Summary

### Remove Entirely:
- Generic security advice without code backing
- Hardware requirements without verification
- Performance monitoring advice without tools
- Validation advice without implementation

### Replace with Specific Examples:
- Abstract "use cases" → concrete code examples
- "Appropriate sample sizes" → specific parameter examples
- Generic advice → feature-specific guidance

### Keep As-Is:
- All recommendations tied to actual CLI flags, configuration options, or implemented features
- Examples that demonstrate real functionality

---

## Implementation Priority

**High Priority (User Confusion):**
1. Remove/fix generic security and validation advice
2. Replace vague use cases with concrete examples

**Medium Priority (Clarity):**
3. Fix hardware prerequisites section
4. Update best practices to be feature-specific

**Low Priority (Nice to Have):**
5. Add quantified examples where possible
6. Expand code examples for clarity

---

## Validation Principle

**Keep if:**
- ✅ Backed by source code implementation
- ✅ Demonstrates actual CLI flags or API features
- ✅ Provides specific, testable examples

**Remove if:**
- ❌ Generic software engineering advice
- ❌ Unsubstantiated performance or cost claims
- ❌ Hardware/infrastructure recommendations not in code
- ❌ "Should" statements without implementation backing

