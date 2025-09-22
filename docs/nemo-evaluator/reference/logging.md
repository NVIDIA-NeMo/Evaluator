# Logging Configuration

This document describes how to configure and use logging in the NVIDIA NeMo Evaluator framework.

## Log Levels

Set these environment variables for logging configuration:

```bash
# Set log level (INFO, DEBUG, WARN, ERROR, CRITICAL)
export NEMO_EVALUATOR_LOG_DIR=DEBUG
```

| Level | Description | Use Case |
|-------|-------------|----------|
| `INFO` | General information | Normal operation logs |
| `DEBUG` | Detailed debugging | Development and troubleshooting |
| `WARN` | Warning messages | Potential issues |
| `ERROR` | Error messages | Problems that need attention |
| `CRITICAL` | Critical errors | Severe problems requiring immediate action |

## Log Output

## Console Output
Logs appear in the console (stderr) with color coding:
- **Green**: INFO messages
- **Yellow**: WARNING messages
- **Red**: ERROR messages
- **Red Background**: CRITICAL messages
- **Grey**: DEBUG messages


## Custom Log Directory
Specify a custom log directory using the `NEMO_EVALUATOR_LOG_DIR` environment variable:

```bash
# Set custom log directory
export NEMO_EVALUATOR_LOG_DIR=/path/to/logs/

# Run evaluation (logs will be written to the specified directory)
eval-factory run_eval ...
```

If `NEMO_EVALUATOR_LOG_DIR` is not set, logs will only appear in the console (stderr) and no files will be created.


## Using Logging Interceptors

NeMo Evaluator also supports dedicated interceptors for request & response logging. Add logging to your adapter configuration:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "request_logging"
          config:
            log_request_body: true
            log_request_headers: true
        - name: "response_logging"
          config:
            log_response_body: true
            log_response_headers: true
```

or use CLI overrides:
```bash
--overrides "target.api_endpoint.adapter_config.use_request_logging=true,target.api_endpoint.adapter_config.use_response_logging=true"
```

## Request Tracking

Each request automatically gets a unique UUID that appears in all related log messages. This helps trace requests through the system.

## Troubleshooting

## No Logs Appearing
- Check that logging interceptors are enabled in your configuration
- Verify log level with `NEMO_EVALUATOR_LOG_LEVEL=INFO`

## Missing DEBUG Logs
- Set `NEMO_EVALUATOR_LOG_LEVEL=DEBUG`

## Logs Not Going to Files
- Check directory permissions
- Verify log directory path

## Debug Mode
```bash
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
```

## Examples

## Basic Logging
```bash
# Enable DEBUG logging
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Run evaluation with logging
eval-factory run_eval --eval_type mmlu_pro --model_id gpt-4 ...
```

## Custom Log Directory
```bash
# Specify custom log location using environment variable
export NEMO_EVALUATOR_LOG_DIR=./my_logs/

# Run evaluation with logging to custom directory
eval-factory run_eval --eval_type mmlu_pro ...
```

## Environment Verification
```bash
echo "NEMO_EVALUATOR_LOG_LEVEL: $NEMO_EVALUATOR_LOG_LEVEL"
echo "NEMO_EVALUATOR_LOG_DIR: $NEMO_EVALUATOR_LOG_DIR"
```
