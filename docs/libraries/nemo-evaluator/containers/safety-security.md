# Safety & Security Containers

Containers specialized for evaluating AI model safety, security, and robustness against various threats and biases.

---

## Garak Container

**NGC Catalog**: [garak](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)

Container for security and robustness evaluation of AI models.

**Use Cases:**
- Security testing
- Adversarial attack evaluation
- Robustness assessment
- Safety evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/garak:25.07.1
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | `150` |
| `temperature` | `0.1` |
| `top_p` | `0.7` |
| `parallelism` | `32` |
| `probes` | `None` |

**Key Features:**
- Automated security testing
- Vulnerability detection
- Prompt injection testing
- Adversarial robustness evaluation
- Comprehensive security reporting

**Security Test Categories:**
- Prompt Injection Attacks
- Data Extraction Attempts
- Jailbreak Techniques
- Adversarial Prompts
- Social Engineering Tests

---

## Safety Harness Container

**NGC Catalog**: [safety-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)

Container for comprehensive safety evaluation of AI models.

**Use Cases:**
- Safety alignment evaluation
- Harmful content detection
- Bias and fairness assessment
- Ethical AI evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/safety-harness:25.07.3
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `6144` |
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `parallelism` | `8` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `judge` | `{'url': None, 'model_id': None, 'api_key': None, 'parallelism': 32, 'request_timeout': 60, 'max_retries': 16}` |

**Key Features:**
- Comprehensive safety benchmarks
- Bias detection and measurement
- Harmful content classification
- Ethical alignment assessment
- Detailed safety reporting

**Safety Evaluation Areas:**
- Bias and Fairness
- Harmful Content Generation
- Toxicity Detection
- Hate Speech Identification
- Ethical Decision Making
- Social Impact Assessment
