# Launcher Issues

Troubleshooting guide for NeMo Evaluator Launcher-specific problems including configuration validation, job management, and multi-backend execution issues.

## Configuration Issues

###  Configuration Validation Errors

**Problem**: Configuration fails validation before execution

**Solution**: Use dry-run to validate configuration:

```bash
# Validate configuration without running
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

**Common Issues**:

1. **Missing Required Fields**:
```
Error: Missing required field 'execution.output_dir'
```
**Fix**: Add output directory to config or override:
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=./results
```

2. **Invalid Task Names**:
```
Error: Unknown task 'invalid_task'. Available tasks: hellaswag, arc_challenge, ...
```
**Fix**: List available tasks and use correct names:
```bash
nv-eval ls tasks
```

3. **Configuration Conflicts**:
```
Error: Cannot specify both 'api_key' and 'api_key_name' in target.api_endpoint
```
**Fix**: Use only one authentication method in configuration.

###  Hydra Configuration Errors

**Problem**: Hydra fails to resolve configuration composition

**Common Errors**:
```
MissingConfigException: Cannot find primary config 'missing_config'
```

**Solutions**:

1. **Verify Config Directory**:
```bash
# List available configs
ls examples/
# Ensure config file exists
ls examples/local_llama_3_1_8b_instruct.yaml
```

2. **Check Config Composition**:
```yaml
# Verify defaults section in config file
defaults:
  - execution: local
  - deployment: none
  - _self_
```

3. **Use Absolute Paths**:
```bash
nv-eval run --config-dir /absolute/path/to/configs --config-name my_config
```

## Job Management Issues

###  Job Status Problems

**Problem**: Cannot check job status or jobs appear stuck

**Diagnosis**:
```bash
# Check job status
nv-eval status <invocation_id>

# List all runs
nv-eval ls runs

# Check specific job
nv-eval status <job_id>
```

**Common Issues**:

1. **Invalid Invocation ID**:
```
Error: Invocation 'abc123' not found
```
**Fix**: Use correct invocation ID from run output or list recent runs:
```bash
nv-eval ls runs
```

2. **Stale Job Database**:
**Fix**: Check execution database location and permissions:
```bash
# Default database location varies by executor
# Local: ./nemo_evaluator_launcher.db
# Check permissions and disk space
ls -la nemo_evaluator_launcher.db
```

###  Job Termination Issues

**Problem**: Cannot kill running jobs

**Solutions**:
```bash
# Kill entire invocation
nv-eval kill <invocation_id>

# Kill specific job
nv-eval kill <job_id>

# Force kill (if supported by executor)
nv-eval kill <invocation_id> --force
```

**Executor-Specific Issues**:

- **Local**: Jobs run in Docker containers - ensure Docker daemon is running
- **Slurm**: Check Slurm queue status with `squeue`
- **Lepton**: Verify Lepton workspace connectivity

## Multi-Backend Execution Issues

###  Local Executor Problems

**Problem**: Docker-related execution failures

**Common Issues**:
1. **Docker Not Running**:
```
Error: Cannot connect to Docker daemon
```
**Fix**: Start Docker daemon:
```bash
# macOS/Windows: Start Docker Desktop
# Linux: 
sudo systemctl start docker
```

2. **Container Pull Failures**:
```
Error: Failed to pull container image
```
**Fix**: Check network connectivity and container registry access:
```bash
docker pull nvcr.io/nvidia/nemo:24.01
```

3. **GPU Access Issues**:
```
Error: NVIDIA runtime not found
```
**Fix**: Install nvidia-container-toolkit and restart Docker.

###  Slurm Executor Problems

**Problem**: Jobs fail to submit to Slurm cluster

**Diagnosis**:
```bash
# Check Slurm cluster status
sinfo
squeue -u $USER

# Check partition availability
sinfo -p <partition_name>
```

**Common Issues**:
1. **Invalid Partition**:
```
Error: Invalid partition name 'gpu'
```
**Fix**: Use correct partition name:
```bash
# List available partitions
sinfo -s
```

2. **Resource Unavailable**:
```
Error: Insufficient resources for job
```
**Fix**: Adjust resource requirements:
```yaml
execution:
  nodes: 1
  gpus_per_node: 2  # Reduce from 8
  time_limit: "2:00:00"  # Reduce time limit
```

###  Lepton Executor Problems

**Problem**: Lepton deployment or execution failures

**Diagnosis**:
```bash
# Check Lepton authentication
lep workspace list

# Test connection
lep deployment list
```

**Common Issues**:
1. **Authentication Failure**:
```
Error: Invalid Lepton credentials
```
**Fix**: Re-authenticate with Lepton:
```bash
lep login -c <workspace_name>:<your_token>
```

2. **Deployment Timeout**:
```
Error: Deployment failed to reach Ready state
```
**Fix**: Check Lepton workspace capacity and try smaller model or fewer replicas.

## Export Issues

###  Export Failures

**Problem**: Results export fails to destination

**Diagnosis**:
```bash
# List completed runs
nv-eval ls runs --status completed

# Try export with verbose logging
nv-eval export <invocation_id> --dest local --format json -v
```

**Common Issues**:

1. **Missing Dependencies**:
```
Error: MLflow not installed
```
**Fix**: Install required exporter dependencies:
```bash
pip install nemo-evaluator-launcher[mlflow]
```

2. **Authentication Issues**:
```
Error: Invalid W&B credentials
```
**Fix**: Configure authentication for export destination:
```bash
# W&B
wandb login

# Google Sheets
# Provide service account credentials file
```

3. **Network Issues**:
```
Error: Connection timeout to MLflow server
```
**Fix**: Verify network connectivity and server availability:
```bash
curl -I http://mlflow-server:5000/health
```

## Performance Issues

###  Slow Execution

**Problem**: Evaluations run slower than expected

**Optimization Strategies**:

1. **Increase Parallelism**:
```yaml
evaluation:
  overrides:
    config.params.parallelism: 32  # Increase from default
```

2. **Optimize Resource Allocation**:
```yaml
execution:
  gpus_per_node: 8  # Use all available GPUs
deployment:
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

3. **Use Test Samples for Development**:
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```

###  Resource Exhaustion

**Problem**: Out of memory or disk space errors

**Solutions**:

1. **Monitor Resources**:
```bash
# GPU memory
nvidia-smi

# Disk space
df -h

# System memory
free -h
```

2. **Reduce Resource Usage**:
```yaml
evaluation:
  overrides:
    config.params.max_new_tokens: 512  # Reduce from 2048
    config.params.parallelism: 4       # Reduce concurrent requests
```

## Getting Help

### Debug Information Collection

When reporting launcher issues, include:

1. **Configuration Details**:
```bash
# Show resolved configuration
nv-eval run --config-dir examples --config-name <config> --dry-run
```

2. **System Information**:
```bash
# Launcher version
nv-eval --version

# System info
python --version
docker --version  # For local executor
sinfo             # For Slurm executor
lep workspace list # For Lepton executor
```

3. **Job Information**:
```bash
# Job status
nv-eval status <invocation_id>

# Recent runs
nv-eval ls runs
```

4. **Log Files**:
- Local executor: Check Docker container logs
- Slurm executor: Check job output files in output directory
- Lepton executor: Check Lepton job logs

### Common Resolution Steps

1. **Validate Configuration**: Always use `--dry-run` first
2. **Check Dependencies**: Ensure all required packages installed
3. **Verify Connectivity**: Test endpoint accessibility
4. **Monitor Resources**: Check available compute resources
5. **Review Logs**: Examine detailed error messages and stack traces

For complex issues, consider using the [Python API](../../libraries/nemo-evaluator-launcher/api) for more granular control and debugging capabilities.
