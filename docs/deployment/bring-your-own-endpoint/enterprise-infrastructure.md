(bring-your-own-endpoint-enterprise)=

# Enterprise Infrastructure

Integrate NeMo Evaluator with existing enterprise infrastructure, including Kubernetes clusters, MLOps pipelines, and custom deployment platforms. This approach provides maximum control and integration with organizational systems.

## Overview

Enterprise integration involves:
- Deploying models within existing infrastructure
- Integrating with organizational security and compliance systems
- Leveraging existing MLOps and DevOps workflows
- Maintaining enterprise-grade reliability and monitoring

## Kubernetes Integration

### Basic Kubernetes Deployment

Deploy models using Kubernetes for container orchestration:

```yaml
# k8s-model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-model-serving
  namespace: ml-models
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-model-serving
  template:
    metadata:
      labels:
        app: llama-model-serving
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:v0.5.0
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3.1-8B-Instruct"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.9"
        - name: MAX_MODEL_LEN
          value: "4096"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 300
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
      nodeSelector:
        gpu-type: "a100"
      tolerations:
      - key: "gpu-node"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: llama-model-service
  namespace: ml-models
spec:
  selector:
    app: llama-model-serving
  ports:
  - port: 8080
    targetPort: 8080
    name: api
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llama-model-ingress
  namespace: ml-models
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - llama-api.company.com
    secretName: llama-api-tls
  rules:
  - host: llama-api.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llama-model-service
            port:
              number: 8080
```

### Advanced Kubernetes Configuration

#### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-model-hpa
  namespace: ml-models
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

#### ConfigMap for Model Configuration

```yaml
# model-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-models
data:
  config.yaml: |
    model:
      name: "meta-llama/Llama-3.1-8B-Instruct"
      max_model_len: 4096
      gpu_memory_utilization: 0.9
      tensor_parallel_size: 1
    
    server:
      host: "0.0.0.0"
      port: 8080
      max_num_seqs: 256
      
    logging:
      level: "INFO"
      format: "json"
      
  prometheus.yaml: |
    metrics:
      enabled: true
      port: 9090
      path: "/metrics"
```

#### Secret Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
  namespace: ml-models
type: Opaque
data:
  # Base64 encoded values
  huggingface-token: <base64-encoded-token>
  api-key: <base64-encoded-api-key>
  
---
apiVersion: v1
kind: Secret
metadata:
  name: registry-secret
  namespace: ml-models
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-encoded-docker-config>
```

### Using Kubernetes Deployment with NeMo Evaluator

#### Configuration

```yaml
# config/kubernetes_deployment.yaml
deployment:
  type: none  # Model already deployed in K8s

target:
  api_endpoint:
    url: https://llama-api.company.com/v1/chat/completions
    model_id: llama-3.1-8b-instruct
    
    # Use service account token for authentication
    api_key: ${K8S_SERVICE_TOKEN}
    
    adapter_config:
      # Enterprise logging
      use_request_logging: true
      use_response_logging: true
      logging_dir: "/shared/logs/evaluations"
      
      # Compliance tracking
      add_request_id: true
      track_user_sessions: true

execution:
  backend: local

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
  
  # Enterprise metadata
  metadata:
    environment: "production"
    cost_center: "ml-research"
    project: "model-evaluation"
```

#### Command-Line Usage

```bash
# Get service endpoint
ENDPOINT=$(kubectl get ingress llama-model-ingress -n ml-models -o jsonpath='{.spec.rules[0].host}')

# Run evaluation
nemo-evaluator-launcher run \
    --config-dir configs \
    --config-name kubernetes_deployment \
    -o target.api_endpoint.url=https://${ENDPOINT}/v1/chat/completions
```

## MLOps Pipeline Integration

### MLflow Integration

Integrate with MLflow for experiment tracking and model management:

```python
# mlflow_integration.py
import mlflow
import mlflow.sklearn
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget
)

class MLflowEvaluationPipeline:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def evaluate_registered_model(self, model_name, model_version, eval_config):
        # Get model endpoint from MLflow
        model_uri = f"models:/{model_name}/{model_version}"
        
        with mlflow.start_run():
            # Log model information
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("evaluation_tasks", eval_config.type)
            
            # Get model endpoint (assuming it's deployed)
            endpoint_url = self._get_model_endpoint(model_name, model_version)
            
            # Configure evaluation
            api_endpoint = ApiEndpoint(
                url=endpoint_url,
                model_id=f"{model_name}-v{model_version}"
            )
            target = EvaluationTarget(api_endpoint=api_endpoint)
            
            # Run evaluation
            results = evaluate(target_cfg=target, eval_cfg=eval_config)
            
            # Log results
            for metric_name, metric_value in results.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts
            mlflow.log_artifacts(eval_config.output_dir)
            
            return results
    
    def _get_model_endpoint(self, model_name, model_version):
        # Get endpoint from model registry metadata
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(model_name, model_version)
        
        # Assuming endpoint is stored in model description or tags
        return model_version_details.tags.get("endpoint_url")

# Usage
pipeline = MLflowEvaluationPipeline(
    tracking_uri="https://mlflow.company.com",
    experiment_name="model-evaluations"
)

eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params={"limit_samples": 100}
)

results = pipeline.evaluate_registered_model(
    model_name="llama-3.1-8b-instruct",
    model_version="1",
    eval_config=eval_config
)
```

### Kubeflow Pipelines Integration

Create Kubeflow pipelines for automated evaluation workflows:

```python
# kubeflow_pipeline.py
from kfp import dsl, components
from kfp.v2 import compiler
from kfp.v2.dsl import component, pipeline

@component(
    base_image="python:3.10",
    packages_to_install=["nemo-evaluator-launcher"]
)
def deploy_model(
    model_path: str,
    deployment_type: str,
    resource_requests: dict
) -> str:
    """Deploy model and return endpoint URL"""
    import subprocess
    import json
    
    # Deploy model using launcher
    cmd = [
        "nemo-evaluator-launcher", "run",
        "--config-dir", "configs",
        "--config-name", "kubeflow_deployment",
        "-o", f"deployment.model_path={model_path}",
        "-o", f"deployment.type={deployment_type}",
        "--dry-run"  # Get deployment info without running
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    config = json.loads(result.stdout)
    
    return config["target"]["api_endpoint"]["url"]

@component(
    base_image="python:3.10",
    packages_to_install=["nemo-evaluator-launcher"]
)
def run_evaluation(
    endpoint_url: str,
    model_id: str,
    tasks: list,
    output_path: str
) -> dict:
    """Run evaluation and return results"""
    import subprocess
    import json
    
    # Run evaluation
    cmd = [
        "nemo-evaluator-launcher", "run",
        "--config-dir", "configs", 
        "--config-name", "kubeflow_evaluation",
        "-o", f"target.api_endpoint.url={endpoint_url}",
        "-o", f"target.api_endpoint.model_id={model_id}",
        "-o", f"evaluation.tasks={json.dumps(tasks)}",
        "-o", f"evaluation.output_dir={output_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    with open(f"{output_path}/results.json") as f:
        results = json.load(f)
    
    return results

@pipeline(name="model-evaluation-pipeline")
def evaluation_pipeline(
    model_path: str = "/models/llama-3.1-8b",
    deployment_type: str = "vllm",
    evaluation_tasks: list = ["mmlu_pro", "gsm8k"],
    output_path: str = "/shared/results"
):
    # Deploy model
    deploy_task = deploy_model(
        model_path=model_path,
        deployment_type=deployment_type,
        resource_requests={"nvidia.com/gpu": "1"}
    )
    
    # Run evaluation
    eval_task = run_evaluation(
        endpoint_url=deploy_task.output,
        model_id="deployed-model",
        tasks=evaluation_tasks,
        output_path=output_path
    )
    
    # Cleanup could be added here
    
    return eval_task.output

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=evaluation_pipeline,
    package_path="evaluation_pipeline.yaml"
)
```

### Airflow Integration

Create Airflow DAGs for scheduled evaluations:

```python
# airflow_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_evaluation_pipeline',
    default_args=default_args,
    description='Automated model evaluation pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'evaluation'],
)

# Check model availability
check_model = BashOperator(
    task_id='check_model_availability',
    bash_command='curl -f https://llama-api.company.com/health',
    dag=dag,
)

# Run evaluation on Kubernetes
run_evaluation = KubernetesPodOperator(
    task_id='run_model_evaluation',
    name='model-evaluation-pod',
    namespace='ml-jobs',
    image='nemo-evaluator:latest',
    cmds=['nemo-evaluator-launcher'],
    arguments=[
        'run',
        '--config-dir', 'configs',
        '--config-name', 'production_evaluation',
        '-o', 'evaluation.output_dir=/shared/results/{{ ds }}'
    ],
    env_vars={
        'API_ENDPOINT': 'https://llama-api.company.com/v1/chat/completions',
        'MODEL_ID': 'llama-3.1-8b-instruct',
        'EVALUATION_DATE': '{{ ds }}'
    },
    resources={
        'requests': {'memory': '4Gi', 'cpu': '2'},
        'limits': {'memory': '8Gi', 'cpu': '4'}
    },
    dag=dag,
)

# Process results
def process_results(**context):
    import json
    import pandas as pd
    
    # Load results
    results_path = f"/shared/results/{context['ds']}/results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    # Create summary
    summary = {
        'date': context['ds'],
        'model': 'llama-3.1-8b-instruct',
        'mmlu_score': results['mmlu_pro']['accuracy'],
        'gsm8k_score': results['gsm8k']['accuracy'],
    }
    
    # Save to database or send to monitoring system
    # ... implementation details ...
    
    return summary

process_results_task = PythonOperator(
    task_id='process_evaluation_results',
    python_callable=process_results,
    dag=dag,
)

# Set task dependencies
check_model >> run_evaluation >> process_results_task
```

## Security and Compliance

### Enterprise Authentication

#### OAuth 2.0 / OIDC Integration

```python
# oauth_client.py
import aiohttp
import asyncio
from authlib.integrations.httpx_client import AsyncOAuth2Client

class EnterpriseAuthClient:
    def __init__(self, client_id, client_secret, token_endpoint, api_base_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_endpoint = token_endpoint
        self.api_base_url = api_base_url
        self.access_token = None
        
    async def get_access_token(self):
        """Get OAuth 2.0 access token"""
        async with AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret
        ) as client:
            token = await client.fetch_token(
                self.token_endpoint,
                grant_type='client_credentials'
            )
            self.access_token = token['access_token']
            return self.access_token
    
    async def make_authenticated_request(self, endpoint, payload):
        """Make API request with OAuth token"""
        if not self.access_token:
            await self.get_access_token()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}{endpoint}",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 401:
                    # Token expired, refresh and retry
                    await self.get_access_token()
                    headers['Authorization'] = f'Bearer {self.access_token}'
                    async with session.post(
                        f"{self.api_base_url}{endpoint}",
                        json=payload,
                        headers=headers
                    ) as retry_response:
                        return await retry_response.json()
                
                return await response.json()

# Usage with NeMo Evaluator
auth_client = EnterpriseAuthClient(
    client_id=os.getenv('OAUTH_CLIENT_ID'),
    client_secret=os.getenv('OAUTH_CLIENT_SECRET'),
    token_endpoint='https://auth.company.com/oauth/token',
    api_base_url='https://llama-api.company.com'
)

# Custom adapter for OAuth authentication
class OAuthAdapter:
    def __init__(self, auth_client):
        self.auth_client = auth_client
    
    async def process_request(self, request):
        # Add OAuth token to request
        token = await self.auth_client.get_access_token()
        request.headers['Authorization'] = f'Bearer {token}'
        return request
```

#### LDAP/Active Directory Integration

```python
# ldap_auth.py
import ldap3
from ldap3 import Server, Connection, ALL

class LDAPAuthenticator:
    def __init__(self, server_url, base_dn, service_account_dn, service_password):
        self.server = Server(server_url, get_info=ALL)
        self.base_dn = base_dn
        self.service_conn = Connection(
            self.server,
            service_account_dn,
            service_password,
            auto_bind=True
        )
    
    def authenticate_user(self, username, password):
        """Authenticate user against LDAP/AD"""
        # Search for user
        search_filter = f"(sAMAccountName={username})"
        self.service_conn.search(
            self.base_dn,
            search_filter,
            attributes=['cn', 'memberOf', 'mail']
        )
        
        if not self.service_conn.entries:
            return None
        
        user_entry = self.service_conn.entries[0]
        user_dn = user_entry.entry_dn
        
        # Verify password
        user_conn = Connection(self.server, user_dn, password)
        if user_conn.bind():
            # Extract user info and groups
            groups = [str(group) for group in user_entry.memberOf]
            return {
                'username': username,
                'name': str(user_entry.cn),
                'email': str(user_entry.mail),
                'groups': groups
            }
        
        return None
    
    def check_authorization(self, user_info, required_groups):
        """Check if user has required group membership"""
        user_groups = set(user_info.get('groups', []))
        required_groups = set(required_groups)
        
        return bool(user_groups.intersection(required_groups))

# Integration with evaluation
ldap_auth = LDAPAuthenticator(
    server_url='ldaps://dc.company.com',
    base_dn='DC=company,DC=com',
    service_account_dn='CN=service-account,OU=Service Accounts,DC=company,DC=com',
    service_password=os.getenv('LDAP_SERVICE_PASSWORD')
)

def authorize_evaluation(username, password):
    user_info = ldap_auth.authenticate_user(username, password)
    if not user_info:
        raise Exception("Authentication failed")
    
    # Check if user can run evaluations
    required_groups = ['CN=ML-Researchers,OU=Groups,DC=company,DC=com']
    if not ldap_auth.check_authorization(user_info, required_groups):
        raise Exception("Insufficient privileges for model evaluation")
    
    return user_info
```

### Audit Logging

```python
# audit_logger.py
import json
import logging
import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_file_path, syslog_server=None):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Syslog handler for SIEM integration
        if syslog_server:
            syslog_handler = logging.handlers.SysLogHandler(address=syslog_server)
            syslog_formatter = logging.Formatter(
                'NeMoEvaluator: %(message)s'
            )
            syslog_handler.setFormatter(syslog_formatter)
            self.logger.addHandler(syslog_handler)
    
    def log_evaluation_start(self, user_id: str, model_id: str, tasks: list, 
                           metadata: Dict[str, Any] = None):
        """Log evaluation start event"""
        event = {
            'event_type': 'evaluation_start',
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'user_id': user_id,
            'model_id': model_id,
            'tasks': tasks,
            'metadata': metadata or {}
        }
        self.logger.info(json.dumps(event))
    
    def log_api_request(self, user_id: str, endpoint: str, request_data: Dict[str, Any],
                       response_status: int, response_time: float):
        """Log API request for compliance"""
        event = {
            'event_type': 'api_request',
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'user_id': user_id,
            'endpoint': endpoint,
            'request_size': len(json.dumps(request_data)),
            'response_status': response_status,
            'response_time_ms': response_time * 1000,
            'request_hash': hashlib.sha256(json.dumps(request_data).encode()).hexdigest()[:16]
        }
        self.logger.info(json.dumps(event))
    
    def log_data_access(self, user_id: str, data_type: str, data_classification: str,
                       access_reason: str):
        """Log data access for compliance"""
        event = {
            'event_type': 'data_access',
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'data_classification': data_classification,
            'access_reason': access_reason
        }
        self.logger.info(json.dumps(event))

# Integration with NeMo Evaluator
audit_logger = AuditLogger(
    log_file_path='/var/log/nemo-evaluator/audit.log',
    syslog_server=('siem.company.com', 514)
)

class AuditingAdapter:
    def __init__(self, audit_logger, user_id):
        self.audit_logger = audit_logger
        self.user_id = user_id
    
    def pre_evaluation(self, config):
        self.audit_logger.log_evaluation_start(
            user_id=self.user_id,
            model_id=config.target.api_endpoint.model_id,
            tasks=[task.name for task in config.evaluation.tasks],
            metadata={'environment': 'production'}
        )
    
    def pre_request(self, request):
        # Log sensitive data access
        if any(keyword in str(request).lower() for keyword in ['personal', 'confidential']):
            self.audit_logger.log_data_access(
                user_id=self.user_id,
                data_type='model_request',
                data_classification='confidential',
                access_reason='model_evaluation'
            )
```

### Data Loss Prevention (DLP)

```python
# dlp_scanner.py
import re
from typing import List, Dict, Any

class DLPScanner:
    def __init__(self):
        # Define patterns for sensitive data
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'api_key': re.compile(r'[A-Za-z0-9]{32,}'),
        }
    
    def scan_text(self, text: str) -> Dict[str, List[str]]:
        """Scan text for sensitive data patterns"""
        findings = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                findings[pattern_name] = matches
        
        return findings
    
    def sanitize_text(self, text: str) -> str:
        """Remove or mask sensitive data"""
        sanitized = text
        
        for pattern_name, pattern in self.patterns.items():
            if pattern_name == 'email':
                sanitized = pattern.sub('[EMAIL_REDACTED]', sanitized)
            elif pattern_name == 'ssn':
                sanitized = pattern.sub('[SSN_REDACTED]', sanitized)
            elif pattern_name == 'credit_card':
                sanitized = pattern.sub('[CARD_REDACTED]', sanitized)
            elif pattern_name == 'api_key':
                sanitized = pattern.sub('[KEY_REDACTED]', sanitized)
            else:
                sanitized = pattern.sub(f'[{pattern_name.upper()}_REDACTED]', sanitized)
        
        return sanitized

class DLPAdapter:
    def __init__(self, dlp_scanner, alert_callback=None):
        self.dlp_scanner = dlp_scanner
        self.alert_callback = alert_callback
    
    def process_request(self, request):
        """Scan and sanitize request data"""
        request_text = json.dumps(request)
        findings = self.dlp_scanner.scan_text(request_text)
        
        if findings:
            # Alert security team
            if self.alert_callback:
                self.alert_callback(f"Sensitive data detected in request: {findings}")
            
            # Sanitize request
            sanitized_text = self.dlp_scanner.sanitize_text(request_text)
            return json.loads(sanitized_text)
        
        return request
    
    def process_response(self, response):
        """Scan response for data leakage"""
        response_text = json.dumps(response)
        findings = self.dlp_scanner.scan_text(response_text)
        
        if findings:
            # Alert and potentially block response
            if self.alert_callback:
                self.alert_callback(f"Potential data leakage in response: {findings}")
            
            # Option to sanitize or block response
            return self.dlp_scanner.sanitize_text(response_text)
        
        return response

# Usage
dlp_scanner = DLPScanner()
dlp_adapter = DLPAdapter(
    dlp_scanner=dlp_scanner,
    alert_callback=lambda msg: print(f"DLP ALERT: {msg}")
)
```

## Monitoring and Observability

### Prometheus Integration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "model_evaluation_rules.yml"
    
    scrape_configs:
      - job_name: 'model-serving'
        static_configs:
          - targets: ['llama-model-service.ml-models:9090']
        scrape_interval: 10s
        metrics_path: /metrics
        
      - job_name: 'evaluation-jobs'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names: ['ml-jobs']
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: nemo-evaluator
            
  model_evaluation_rules.yml: |
    groups:
      - name: model_evaluation_alerts
        rules:
          - alert: ModelServingDown
            expr: up{job="model-serving"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "Model serving endpoint is down"
              description: "Model serving has been down for more than 1 minute"
              
          - alert: HighEvaluationLatency
            expr: histogram_quantile(0.95, rate(model_request_duration_seconds_bucket[5m])) > 10
            for: 2m
            labels:
              severity: warning
            annotations:
              summary: "High evaluation latency detected"
              description: "95th percentile latency is above 10 seconds"
              
          - alert: EvaluationJobFailed
            expr: increase(evaluation_jobs_failed_total[5m]) > 0
            for: 0m
            labels:
              severity: warning
            annotations:
              summary: "Evaluation job failed"
              description: "At least one evaluation job has failed"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "NeMo Evaluator Enterprise Dashboard",
    "panels": [
      {
        "title": "Model Serving Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"model-serving\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Evaluation Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(evaluation_jobs_success_total[5m]) / rate(evaluation_jobs_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      }
    ]
  }
}
```

### Custom Metrics Collection

```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading

class EvaluationMetrics:
    def __init__(self, port=9090):
        # Metrics
        self.request_count = Counter(
            'evaluation_requests_total',
            'Total evaluation requests',
            ['model', 'task', 'status']
        )
        
        self.request_duration = Histogram(
            'evaluation_request_duration_seconds',
            'Evaluation request duration',
            ['model', 'task']
        )
        
        self.active_evaluations = Gauge(
            'evaluation_active_jobs',
            'Number of active evaluation jobs',
            ['model']
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy_score',
            'Model accuracy scores',
            ['model', 'task']
        )
        
        # Start metrics server
        start_http_server(port)
    
    def record_evaluation_start(self, model, task):
        self.active_evaluations.labels(model=model).inc()
        return time.time()
    
    def record_evaluation_complete(self, model, task, start_time, status, accuracy=None):
        duration = time.time() - start_time
        
        self.request_count.labels(model=model, task=task, status=status).inc()
        self.request_duration.labels(model=model, task=task).observe(duration)
        self.active_evaluations.labels(model=model).dec()
        
        if accuracy is not None:
            self.model_accuracy.labels(model=model, task=task).set(accuracy)

# Integration with evaluation
metrics = EvaluationMetrics()

class MetricsAdapter:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.start_times = {}
    
    def pre_evaluation(self, config):
        model_id = config.target.api_endpoint.model_id
        for task in config.evaluation.tasks:
            key = f"{model_id}_{task.name}"
            self.start_times[key] = self.metrics.record_evaluation_start(model_id, task.name)
    
    def post_evaluation(self, config, results):
        model_id = config.target.api_endpoint.model_id
        for task_name, task_results in results.items():
            key = f"{model_id}_{task_name}"
            if key in self.start_times:
                self.metrics.record_evaluation_complete(
                    model=model_id,
                    task=task_name,
                    start_time=self.start_times[key],
                    status='success',
                    accuracy=task_results.get('accuracy')
                )
```

## Cost Management and Chargeback

### Cost Allocation

```python
# cost_allocator.py
import json
from datetime import datetime, timedelta
from typing import Dict, List

class CostAllocator:
    def __init__(self, cost_rates: Dict[str, float]):
        """
        cost_rates: Dictionary mapping resource types to hourly costs
        e.g., {'gpu.a100': 3.0, 'cpu.core': 0.05, 'memory.gb': 0.01}
        """
        self.cost_rates = cost_rates
        self.usage_records = []
    
    def record_usage(self, user_id: str, project: str, resource_type: str, 
                    quantity: float, duration_hours: float, metadata: Dict = None):
        """Record resource usage for cost allocation"""
        cost = self.cost_rates.get(resource_type, 0) * quantity * duration_hours
        
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'project': project,
            'resource_type': resource_type,
            'quantity': quantity,
            'duration_hours': duration_hours,
            'cost': cost,
            'metadata': metadata or {}
        }
        
        self.usage_records.append(record)
        return record
    
    def generate_cost_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate cost report for a date range"""
        filtered_records = [
            record for record in self.usage_records
            if start_date <= datetime.fromisoformat(record['timestamp']) <= end_date
        ]
        
        # Aggregate by project
        project_costs = {}
        user_costs = {}
        
        for record in filtered_records:
            project = record['project']
            user = record['user_id']
            cost = record['cost']
            
            if project not in project_costs:
                project_costs[project] = {'total_cost': 0, 'resources': {}}
            project_costs[project]['total_cost'] += cost
            
            resource_type = record['resource_type']
            if resource_type not in project_costs[project]['resources']:
                project_costs[project]['resources'][resource_type] = 0
            project_costs[project]['resources'][resource_type] += cost
            
            if user not in user_costs:
                user_costs[user] = 0
            user_costs[user] += cost
        
        return {
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'total_cost': sum(record['cost'] for record in filtered_records),
            'project_breakdown': project_costs,
            'user_breakdown': user_costs,
            'detailed_records': filtered_records
        }

# Integration with evaluation
cost_allocator = CostAllocator({
    'gpu.a100': 3.0,      # $3/hour per A100 GPU
    'gpu.v100': 2.0,      # $2/hour per V100 GPU
    'cpu.core': 0.05,     # $0.05/hour per CPU core
    'memory.gb': 0.01,    # $0.01/hour per GB RAM
    'storage.gb': 0.0001  # $0.0001/hour per GB storage
})

class CostTrackingAdapter:
    def __init__(self, cost_allocator, user_id, project):
        self.cost_allocator = cost_allocator
        self.user_id = user_id
        self.project = project
        self.start_time = None
        self.resources = {}
    
    def pre_evaluation(self, config):
        self.start_time = datetime.utcnow()
        
        # Estimate resource usage based on deployment config
        if hasattr(config, 'deployment'):
            if config.deployment.type == 'vllm':
                self.resources['gpu.a100'] = getattr(config.deployment, 'tensor_parallel_size', 1)
                self.resources['cpu.core'] = 4  # Estimated
                self.resources['memory.gb'] = 16  # Estimated
    
    def post_evaluation(self, config, results):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds() / 3600  # hours
            
            for resource_type, quantity in self.resources.items():
                self.cost_allocator.record_usage(
                    user_id=self.user_id,
                    project=self.project,
                    resource_type=resource_type,
                    quantity=quantity,
                    duration_hours=duration,
                    metadata={
                        'evaluation_tasks': list(results.keys()),
                        'model_id': config.target.api_endpoint.model_id
                    }
                )
```

## Troubleshooting Enterprise Deployments

### Debugging Kubernetes Issues

```bash
# Check pod status
kubectl get pods -n ml-models -l app=llama-model-serving

# View pod logs
kubectl logs -n ml-models deployment/llama-model-serving -f

# Describe pod for events
kubectl describe pod -n ml-models <pod-name>

# Check resource usage
kubectl top pods -n ml-models

# Access pod shell for debugging
kubectl exec -it -n ml-models <pod-name> -- /bin/bash

# Check service endpoints
kubectl get endpoints -n ml-models llama-model-service

# Test service connectivity
kubectl run debug --image=curlimages/curl -it --rm --restart=Never -- \
  curl -v http://llama-model-service.ml-models:8080/health
```

### Network Connectivity Issues

```python
# network_diagnostics.py
import asyncio
import aiohttp
import socket
import time

class NetworkDiagnostics:
    @staticmethod
    async def check_endpoint_health(url, timeout=30):
        """Check if endpoint is healthy"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(f"{url}/health") as response:
                    return {
                        'status': 'healthy' if response.status == 200 else 'unhealthy',
                        'status_code': response.status,
                        'response_time': response.headers.get('X-Response-Time'),
                        'content': await response.text()
                    }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def check_dns_resolution(hostname):
        """Check DNS resolution"""
        try:
            ip = socket.gethostbyname(hostname)
            return {'status': 'success', 'ip': ip}
        except socket.gaierror as e:
            return {'status': 'error', 'error': str(e)}
    
    @staticmethod
    def check_port_connectivity(host, port, timeout=5):
        """Check if port is reachable"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            result = sock.connect_ex((host, port))
            sock.close()
            return {'status': 'open' if result == 0 else 'closed', 'result_code': result}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Usage
async def diagnose_connectivity():
    diagnostics = NetworkDiagnostics()
    
    # Check endpoint health
    health = await diagnostics.check_endpoint_health('https://llama-api.company.com')
    print(f"Health check: {health}")
    
    # Check DNS
    dns = diagnostics.check_dns_resolution('llama-api.company.com')
    print(f"DNS resolution: {dns}")
    
    # Check port connectivity
    port = diagnostics.check_port_connectivity('llama-api.company.com', 443)
    print(f"Port connectivity: {port}")

# Run diagnostics
asyncio.run(diagnose_connectivity())
```

### Performance Troubleshooting

```python
# performance_profiler.py
import time
import psutil
import GPUtil
from contextlib import contextmanager

class PerformanceProfiler:
    def __init__(self):
        self.metrics = []
    
    @contextmanager
    def profile_evaluation(self, description=""):
        """Context manager for profiling evaluation performance"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent
        
        try:
            gpus = GPUtil.getGPUs()
            start_gpu_util = [gpu.load * 100 for gpu in gpus]
            start_gpu_memory = [gpu.memoryUtil * 100 for gpu in gpus]
        except:
            start_gpu_util = []
            start_gpu_memory = []
        
        yield
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        try:
            gpus = GPUtil.getGPUs()
            end_gpu_util = [gpu.load * 100 for gpu in gpus]
            end_gpu_memory = [gpu.memoryUtil * 100 for gpu in gpus]
        except:
            end_gpu_util = []
            end_gpu_memory = []
        
        metrics = {
            'description': description,
            'duration': end_time - start_time,
            'cpu_usage': {'start': start_cpu, 'end': end_cpu},
            'memory_usage': {'start': start_memory, 'end': end_memory},
            'gpu_utilization': {'start': start_gpu_util, 'end': end_gpu_util},
            'gpu_memory': {'start': start_gpu_memory, 'end': end_gpu_memory},
            'timestamp': time.time()
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def get_performance_report(self):
        """Generate performance report"""
        if not self.metrics:
            return "No performance data collected"
        
        total_duration = sum(m['duration'] for m in self.metrics)
        avg_cpu = sum(m['cpu_usage']['end'] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m['memory_usage']['end'] for m in self.metrics) / len(self.metrics)
        
        report = f"""
Performance Report:
==================
Total Evaluations: {len(self.metrics)}
Total Duration: {total_duration:.2f}s
Average Duration: {total_duration/len(self.metrics):.2f}s
Average CPU Usage: {avg_cpu:.1f}%
Average Memory Usage: {avg_memory:.1f}%
"""
        
        if self.metrics[0]['gpu_utilization']['end']:
            avg_gpu_util = sum(sum(m['gpu_utilization']['end']) for m in self.metrics) / len(self.metrics)
            avg_gpu_memory = sum(sum(m['gpu_memory']['end']) for m in self.metrics) / len(self.metrics)
            report += f"""
Average GPU Utilization: {avg_gpu_util:.1f}%
Average GPU Memory: {avg_gpu_memory:.1f}%
"""
        
        return report

# Usage
profiler = PerformanceProfiler()

with profiler.profile_evaluation("MMLU Pro evaluation"):
    # Run your evaluation here
    results = evaluate(target_cfg=target, eval_cfg=config)

print(profiler.get_performance_report())
```

## Best Practices

### Infrastructure
- **Use infrastructure as code**: Terraform, Helm charts for reproducible deployments
- **Implement proper resource limits**: Prevent resource contention and ensure fair sharing
- **Design for high availability**: Multi-zone deployments with proper failover
- **Plan for disaster recovery**: Backup strategies and recovery procedures

### Security
- **Follow principle of least privilege**: Minimal required permissions for each component
- **Implement defense in depth**: Multiple layers of security controls
- **Regular security audits**: Automated and manual security assessments
- **Keep systems updated**: Regular patching and vulnerability management

### Operations
- **Comprehensive monitoring**: Metrics, logs, and distributed tracing
- **Automated alerting**: Proactive notification of issues and anomalies
- **Runbook documentation**: Clear procedures for common operational tasks
- **Regular testing**: Disaster recovery drills and chaos engineering

### Compliance
- **Document everything**: Maintain audit trails and documentation
- **Regular compliance checks**: Automated compliance validation
- **Data governance**: Clear data handling and retention policies
- **Access controls**: Regular review and rotation of access credentials

## Next Steps

- **Scale deployment**: Implement multi-region and multi-cloud strategies
- **Advanced security**: Implement zero-trust architecture and advanced threat detection
- **Cost optimization**: Implement FinOps practices and cost optimization strategies
- **Automation**: Build fully automated CI/CD pipelines for model deployment and evaluation
