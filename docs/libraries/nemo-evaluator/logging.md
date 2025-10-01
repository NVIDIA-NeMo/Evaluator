(nemo-evaluator-logging)=

# Logging Configuration

This document describes how to configure and use logging in the NVIDIA NeMo Evaluator framework.

## Log Levels

Set these environment variables for logging configuration:

```bash
# Set log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
export LOG_LEVEL=DEBUG
# or (legacy, still supported)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
```

| Level | Description | Use Case |
|-------|-------------|----------|
| `INFO` | General information | Normal operation logs |
| `DEBUG` | Detailed debugging | Development and troubleshooting |
| `WARNING` | Warning messages | Potential issues |
| `ERROR` | Error messages | Problems that need attention |
| `CRITICAL` | Critical errors | Severe problems requiring immediate action |

## Log Output

### Console Output

Logs appear in the console (stderr) with color coding:

- **Green**: INFO messages
- **Yellow**: WARNING messages
- **Red**: ERROR messages
- **Red background**: CRITICAL messages
- **Gray**: DEBUG messages

### Custom Log Directory

Specify a custom log directory using the `NEMO_EVALUATOR_LOG_DIR` environment variable:

```bash
# Set custom log directory
export NEMO_EVALUATOR_LOG_DIR=/path/to/logs/

# Run evaluation (logs will be written to the specified directory)
eval-factory run_eval ...
```

If `NEMO_EVALUATOR_LOG_DIR` is not set, logs appear in the console (stderr) without file output.

## Using Logging Interceptors

NeMo Evaluator supports dedicated interceptors for request and response logging. Add logging to your adapter configuration:

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

## Request Tracking

Each request automatically gets a unique UUID that appears in all related log messages. This helps trace requests through the system.

## Troubleshooting

### No logs appearing

- Enable logging interceptors in your configuration
- Verify log level with `LOG_LEVEL=INFO` or `NEMO_EVALUATOR_LOG_LEVEL=INFO`

### Missing DEBUG logs

- Set `LOG_LEVEL=DEBUG` or `NEMO_EVALUATOR_LOG_LEVEL=DEBUG`

### Logs not going to files

- Check directory permissions
- Verify log directory path with `NEMO_EVALUATOR_LOG_DIR`

### Debug mode

```bash
export LOG_LEVEL=DEBUG
```

## Examples

### Basic logging

```bash
# Enable DEBUG logging
export LOG_LEVEL=DEBUG

# Run evaluation with logging
eval-factory run_eval --eval_type mmlu_pro --model_id gpt-4 ...
```

### Custom log directory

```bash
# Specify custom log location using environment variable
export NEMO_EVALUATOR_LOG_DIR=./my_logs/

# Run evaluation with logging to custom directory
eval-factory run_eval --eval_type mmlu_pro ...
```

### Environment verification

```bash
echo "LOG_LEVEL: $LOG_LEVEL"
echo "NEMO_EVALUATOR_LOG_DIR: $NEMO_EVALUATOR_LOG_DIR"
```
