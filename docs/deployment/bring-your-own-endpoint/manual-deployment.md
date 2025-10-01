(bring-your-own-endpoint-manual)=

# Manual Deployment

Deploy models yourself using popular serving frameworks, then point NeMo Evaluator to your endpoints. This approach gives you full control over deployment infrastructure and serving configuration.

## Overview

Manual deployment involves:
- Setting up model serving using frameworks like PyTriton, Ray Serve, vLLM, or custom solutions
- Configuring OpenAI-compatible endpoints
- Managing infrastructure, scaling, and monitoring yourself
- Using either the launcher or core library to run evaluations against your endpoints

## Serving Frameworks

### vLLM Serving

High-performance inference with optimized attention mechanisms and continuous batching.

#### Installation and Setup

```bash
# Install vLLM
pip install vllm

# For specific GPU support
pip install vllm[cuda]  # For CUDA
pip install vllm[rocm]  # For ROCm
```

#### Basic Deployment

```bash
# Deploy with OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8080 \
    --served-model-name llama-3.1-8b \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

#### Advanced vLLM Configuration

```bash
# Multi-GPU deployment with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --port 8080 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --served-model-name llama-3.1-70b \
    --dtype bfloat16 \
    --max-num-seqs 256 \
    --swap-space 4
```

#### vLLM with Custom Configuration

```python
# custom_vllm_server.py
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse

def main():
    # Custom model loading
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="bfloat16",
        # Custom configurations
        enforce_eager=True,
        disable_custom_all_reduce=False,
    )
    
    # Start OpenAI-compatible server
    run_server(
        model=llm,
        host="0.0.0.0",
        port=8080,
        served_model_name="custom-llama-3.1-8b"
    )

if __name__ == "__main__":
    main()
```

### TensorRT-LLM Serving

NVIDIA's optimized inference engine for maximum performance.

#### Installation

```bash
# Install TensorRT-LLM (requires specific CUDA version)
pip install tensorrt-llm

# Or use Docker
docker pull nvcr.io/nvidia/tensorrt:24.07-py3
```

#### Model Conversion and Deployment

```bash
# Convert model to TensorRT format
python convert_checkpoint.py \
    --model_dir /path/to/llama-3.1-8b \
    --output_dir /path/to/trt_engines/llama-3.1-8b \
    --dtype bfloat16

# Build TensorRT engine
trtllm-build \
    --checkpoint_dir /path/to/trt_engines/llama-3.1-8b \
    --output_dir /path/to/trt_engines/llama-3.1-8b/1-gpu \
    --gemm_plugin bfloat16 \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 1024

# Deploy with Triton server
python3 scripts/launch_triton_server.py \
    --model_repo /path/to/model_repository \
    --world_size 1
```

### Hugging Face Text Generation Inference

Production-ready inference server from Hugging Face.

#### Docker Deployment

```bash
# Deploy with TGI Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --num-shard 1 \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 4096
```

#### Custom TGI Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    ports:
      - "8080:80"
    environment:
      - MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
      - NUM_SHARD=2
      - MAX_INPUT_LENGTH=4096
      - MAX_TOTAL_TOKENS=8192
      - QUANTIZE=bitsandbytes-nf4
    volumes:
      - ./models:/data
      - ./cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Custom FastAPI Server

Build your own OpenAI-compatible server with FastAPI.

#### Basic Implementation

```python
# custom_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import time
import uuid

app = FastAPI()

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Request/Response models
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": inputs.input_ids.shape[1] + len(tokenizer.encode(generated_text))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Convert chat messages to prompt
    prompt = ""
    for message in request.messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"User: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"
    
    prompt += "Assistant: "
    
    # Use completion endpoint logic
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop
    )
    
    return await create_completion(completion_request)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

#### Running Custom Server

```bash
# Install dependencies
pip install fastapi uvicorn transformers torch

# Run server
python custom_server.py

# Or with uvicorn directly
uvicorn custom_server:app --host 0.0.0.0 --port 8080 --workers 1
```

## Using Manual Deployments with NeMo Evaluator

### With Launcher

Once your manual deployment is running, use the launcher to evaluate:

```bash
# Basic evaluation against manual deployment
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://localhost:8080/v1/completions \
    -o target.api_endpoint.model_id=your-model-name \
    -o deployment.type=none  # No launcher deployment needed
```

#### Configuration File Approach

```yaml
# config/manual_deployment.yaml
deployment:
  type: none  # No deployment by launcher

target:
  api_endpoint:
    url: http://localhost:8080/v1/completions
    model_id: llama-3.1-8b
    # Optional authentication
    api_key: ${API_KEY}
    
    # Optional adapter configuration
    adapter_config:
      use_caching: true
      caching_dir: ./cache
      use_request_logging: true
      max_logged_requests: 10

execution:
  backend: local  # Run evaluation locally

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 100
    - name: gsm8k
      params:
        limit_samples: 50
```

### With Core Library

Direct API usage for manual deployments:

```python
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget
)

# Configure your manual deployment endpoint
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions",
    model_id="llama-3.1-8b",
    # Optional authentication
    api_key="your-api-key"
)

target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="results",
    params={
        "limit_samples": 100,
        "parallelism": 4
    }
)

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(f"Results: {results.metrics}")
```

## Load Balancing and High Availability

### NGINX Load Balancer

```nginx
# nginx.conf
upstream model_servers {
    server localhost:8080 weight=1 max_fails=3 fail_timeout=30s;
    server localhost:8081 weight=1 max_fails=3 fail_timeout=30s;
    server localhost:8082 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name model-api.local;

    location /v1/ {
        proxy_pass http://model_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://model_servers;
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }
}
```

### HAProxy Configuration

```haproxy
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 300000ms
    timeout server 300000ms
    option httplog

frontend model_api
    bind *:80
    default_backend model_servers

backend model_servers
    balance roundrobin
    option httpchk GET /health
    server model1 localhost:8080 check inter 10s
    server model2 localhost:8081 check inter 10s
    server model3 localhost:8082 check inter 10s
```

## Monitoring and Observability

### Prometheus Metrics

```python
# Add to your custom server
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
GENERATION_TOKENS = Histogram('model_generation_tokens', 'Generated tokens per request')

@app.middleware("http")
async def add_prometheus_metrics(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    try:
        # Test model inference
        test_input = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            model.generate(test_input.input_ids, max_new_tokens=1)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_used": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }, 503
```

## Performance Optimization

### Batch Processing

```python
# Implement batching in custom server
from asyncio import Queue
import asyncio
from typing import List

class BatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size=8, max_wait_time=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = Queue()
        self.processing = False
    
    async def add_request(self, request, response_future):
        await self.queue.put((request, response_future))
        if not self.processing:
            asyncio.create_task(self.process_batch())
    
    async def process_batch(self):
        self.processing = True
        batch = []
        
        # Collect requests for batch
        start_time = time.time()
        while len(batch) < self.max_batch_size:
            try:
                timeout = max(0, self.max_wait_time - (time.time() - start_time))
                request, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                batch.append((request, future))
            except asyncio.TimeoutError:
                break
        
        if batch:
            await self.process_requests(batch)
        
        self.processing = False
    
    async def process_requests(self, batch):
        # Process batch of requests together
        prompts = [req.prompt for req, _ in batch]
        
        # Batch tokenization
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        
        # Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=batch[0][0].max_tokens,
                temperature=batch[0][0].temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Return results
        for i, (request, future) in enumerate(batch):
            generated_text = self.tokenizer.decode(
                outputs[i][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            future.set_result(generated_text)
```

### Memory Management

```python
# Memory-efficient model loading
import gc

def load_model_with_memory_management():
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load with specific dtype and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        device_map="auto",           # Automatic device placement
        low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
        offload_folder="./offload",  # Offload to disk if needed
    )
    
    return model

# Periodic cleanup
async def periodic_cleanup():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        torch.cuda.empty_cache()
        gc.collect()
```

## Security Considerations

### API Authentication

```python
# Add authentication to custom server
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = os.getenv("API_TOKEN")
    if not expected_token or credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, token: str = Depends(verify_token)):
    # Your completion logic here
    pass
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/completions")
@limiter.limit("10/minute")  # 10 requests per minute
async def create_completion(request: Request, completion_request: CompletionRequest):
    # Your completion logic here
    pass
```

## Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Monitor GPU memory
def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        print(f"GPU Memory - Allocated: {allocated/1e9:.2f}GB, Cached: {cached/1e9:.2f}GB")

# Clear memory when needed
torch.cuda.empty_cache()
```

**Slow Inference:**
```python
# Profile inference
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your inference code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Connection Issues:**
```bash
# Test endpoint connectivity
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "prompt": "Hello", "max_tokens": 10}'

# Check if port is listening
netstat -tulpn | grep 8080
```

## Best Practices

### Deployment
- **Use containerization**: Docker/Podman for consistent environments
- **Implement health checks**: Proper health and readiness endpoints
- **Monitor resources**: CPU, GPU, memory usage tracking
- **Plan for scaling**: Design for horizontal scaling from the start

### Performance
- **Optimize batch sizes**: Balance latency vs throughput
- **Use appropriate precision**: bfloat16/float16 for memory efficiency
- **Implement caching**: Cache model outputs for repeated requests
- **Profile regularly**: Identify and address performance bottlenecks

### Operations
- **Centralized logging**: Structured logging for debugging and monitoring
- **Implement graceful shutdown**: Handle termination signals properly
- **Version your models**: Track model versions and configurations
- **Backup configurations**: Version control your deployment configs

## Next Steps

- **Advanced serving**: Explore [PyTriton](pytriton.md) and [Ray Serve](ray-serve.md) for production deployments
- **Hosted alternatives**: Compare with [hosted services](hosted-services.md) for operational simplicity
- **Enterprise integration**: Scale to [enterprise infrastructure](enterprise-infrastructure.md)
- **Monitoring setup**: Implement comprehensive observability and alerting
