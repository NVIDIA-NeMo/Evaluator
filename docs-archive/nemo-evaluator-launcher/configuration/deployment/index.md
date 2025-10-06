# Deployment Configuration

Deployment configurations define how to provision and host model endpoints for evaluation.

## Supported Types

- **[vLLM](vllm.md)**: High-performance LLM serving with optimized attention
- **[SGLang](sglang.md)**: Structured generation with efficient memory usage  
- **[NIM](nim.md)**: NVIDIA-optimized inference microservices
- **[Generic](generic.md)**: Custom server deployment with flexible configuration
- **[None](none.md)**: Use existing endpoints (no deployment)

## Quick Reference

```yaml
deployment:
  type: vllm  # or sglang, nim, none
  # ... deployment-specific settings
```

**When to use:**
- **vLLM**: General-purpose LLM serving
- **SGLang**: General-purpose LLM serving
- **NIM**: NVIDIA hardware optimized deployments
- **Generic**: Custom servers not covered by built-in configs
- **None**: Existing endpoints

## Custom Server Integration

**Need to deploy a server not covered by built-in configs?**

**Quick integration**: Use [Generic deployment](generic.md) for any Docker-based server with OpenAI-compatible API.

**Advanced integration**: Create custom deployment template in `configs/deployment/` for reusable configurations.

## Configuration Files

See all available deployment configurations: [Deployment Configs](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment)
