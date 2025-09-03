(authentication)=

# Authentication

Solutions for HuggingFace token issues, dataset access permissions, and gated model problems.

## Common Authentication Issues

### ❌ Problem: `401 Unauthorized` for gated datasets

**Solution**:

```bash
# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Or authenticate via CLI
huggingface-cli login

# Verify authentication
huggingface-cli whoami
```

**In Python**:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

### ❌ Problem: `403 Forbidden` for specific datasets

**Solution**:

1. Request access to the gated dataset on HuggingFace
2. Wait for approval from dataset maintainers
3. Ensure your token has the required permissions

## Best Practices

### Token Management

1. **Store tokens securely**: Never commit tokens to version control
2. **Use environment variables**: Set `HF_TOKEN` in your shell profile
3. **Verify permissions**: Check that your token has access to required datasets
4. **Token scope**: Ensure your token has appropriate read permissions

### Troubleshooting Access Issues

```python
# Test token validity
import requests

headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
response = requests.get("https://huggingface.co/api/whoami", headers=headers)

if response.status_code == 200:
    print("Token is valid")
    print(f"User: {response.json()['name']}")
else:
    print(f"Token issue: {response.status_code}")
```

### Common Gated Datasets

These datasets typically require access approval:

- MMLU (some versions)
- HellaSwag
- TruthfulQA
- Various safety evaluation datasets

Request access through the HuggingFace web interface before attempting to use them in evaluations.
