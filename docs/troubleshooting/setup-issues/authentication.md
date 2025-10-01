(authentication)=

# Authentication

Solutions for HuggingFace token issues and dataset access permissions.

## Common Authentication Issues

### Problem: `401 Unauthorized` for Gated Datasets

**Solution**:

```bash
# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Or authenticate using CLI
huggingface-cli login

# Verify authentication
huggingface-cli whoami
```

**In Python**:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

### Problem: `403 Forbidden` for Specific Datasets

**Solution**:

1. Request access to the gated dataset on HuggingFace
2. Wait for approval from dataset maintainers
3. Ensure your token has the required permissions

## Datasets Requiring Authentication

The following datasets require `HF_TOKEN` and access approval:

- **GPQA Diamond** (and variants): [Request access](https://huggingface.co/datasets/Idavidrein/gpqa)
- **Aegis v2**: Required for safety evaluation tasks
- **HLE**: Human-like evaluation tasks

:::{note}
Most standard benchmarks (MMLU, HellaSwag, ARC, etc.) do not require authentication.
:::
