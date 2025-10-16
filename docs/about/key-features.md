
(about-key-features)=

# Key Features

NeMo Evaluator SDK delivers comprehensive AI model evaluation through a dual-library architecture that scales from local development to enterprise production. Experience container-first reproducibility, multi-backend execution, and comprehensive set of state-of-the-art benchmarks.

##  **Unified Orchestration (NeMo Evaluator Launcher)**

### Multi-Backend Execution
Run evaluations anywhere with unified configuration and monitoring:

- **Local Execution**: Docker-based evaluation on your workstation
- **HPC Clusters**: Slurm integration for large-scale parallel evaluation  
- **Cloud Platforms**: Lepton AI and custom cloud backend support
- **Hybrid Workflows**: Mix local development with cloud production

```bash
# Single command, multiple backends
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name slurm_llama_3_1_8b_instruct  
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name lepton_vllm_llama_3_1_8b_instruct
```

### Evaluation Benchmarks & Tasks
Access comprehensive benchmark suite with single CLI:

```bash
# Discover available benchmarks
nemo-evaluator-launcher ls tasks

# Run academic benchmarks
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o 'evaluation.tasks=["mmlu_pro", "gsm8k", "arc_challenge"]'

# Run safety evaluation
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o 'evaluation.tasks=["aegis_v2", "garak"]'
```

### Built-in Result Export
First-class integration with MLOps platforms:

```bash
# Export to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets
nemo-evaluator-launcher export <invocation_id> --dest gsheets
```

##  **Core Evaluation Engine (NeMo Evaluator Core)**

### Container-First Architecture
Pre-built NGC containers guarantee reproducible results across environments:

## NGC Container Catalog

```{list-table}
:header-rows: 1
:widths: 20 25 15 15 25

* - Container
  - Description
  - NGC Catalog
  - Latest Tag
  - Key Benchmarks
* - **agentic_eval**
  - Agentic AI evaluation framework
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)
  - {{ docker_compose_latest }}
  - agentic_eval_answer_accuracy, agentic_eval_goal_accuracy_with_reference, agentic_eval_goal_accuracy_without_reference, agentic_eval_topic_adherence, agentic_eval_tool_call_accuracy
* - **bfcl**
  - Function calling evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)
  - {{ docker_compose_latest }}
  - bfclv2, bfclv2_ast, bfclv2_ast_prompting, bfclv3, bfclv3_ast, bfclv3_ast_prompting
* - **bigcode-evaluation-harness**
  - Code generation evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)
  - {{ docker_compose_latest }}
  - humaneval, humaneval_instruct, humanevalplus, mbpp, mbppplus, mbppplus_nemo, multiple-cpp, multiple-cs, multiple-d, multiple-go, multiple-java, multiple-jl, multiple-js, multiple-lua, multiple-php, multiple-pl, multiple-py, multiple-r, multiple-rb, multiple-rkt, multiple-rs, multiple-scala, multiple-sh, multiple-swift, multiple-ts
* -  **compute-eval**
  - CUDA code evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/compute-eval)
  - {{ docker_compose_latest }}
  - cccl_problems, combined_problems, cuda_problems 
* - **garak**
  - Security and robustness testing
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)
  - {{ docker_compose_latest }}
  - garak
* - **genai-perf** 
  - GenAI performance benchmarking 
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf)
  - {{ docker_compose_latest }}
  - genai_perf_generation, genai_perf_summarization
* - **helm**
  - Holistic evaluation framework
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)
  - {{ docker_compose_latest }}
  - ci_bench, ehr_sql, head_qa, med_dialog_healthcaremagic, med_dialog_icliniq, medbullets, medcalc_bench, medec, medhallu, medi_qa, medication_qa, mtsamples_procedures, mtsamples_replicate, pubmed_qa, race_based_med
* - **hle**
  - Academic knowledge and problem solving
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)
  - {{ docker_compose_latest }}
  - hle
* - **ifbench**
  - Instruction following evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)
  - {{ docker_compose_latest }}
  - ifbench
* - **livecodebench**
  - Live coding evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)
  - {{ docker_compose_latest }}
  - AA_codegeneration, codeexecution_v2, codeexecution_v2_cot, codegeneration_notfast, codegeneration_release_latest, codegeneration_release_v1, codegeneration_release_v2, codegeneration_release_v3, codegeneration_release_v4, codegeneration_release_v5, codegeneration_release_v6, livecodebench_0724_0125, livecodebench_0824_0225
* - **lm-evaluation-harness**
  - Language model benchmarks
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
  - {{ docker_compose_latest }}
  - adlr_arc_challenge_llama, adlr_gsm8k_fewshot_cot, adlr_humaneval_greedy, adlr_humanevalplus_greedy, adlr_mbpp_sanitized_3shot_greedy, adlr_mbppplus_greedy_sanitized, adlr_minerva_math_nemo, adlr_mmlu_pro_5_shot_base, adlr_race, adlr_truthfulqa_mc2, agieval, arc_challenge, arc_challenge_chat, arc_multilingual, bbh, bbh_instruct, bbq, commonsense_qa, frames_naive, frames_naive_with_links, frames_oracle, global_mmlu, global_mmlu_ar, global_mmlu_bn, global_mmlu_de, global_mmlu_en, global_mmlu_es, global_mmlu_fr, global_mmlu_full, global_mmlu_full_am, global_mmlu_full_ar, global_mmlu_full_bn, global_mmlu_full_cs, global_mmlu_full_de, global_mmlu_full_el, global_mmlu_full_en, global_mmlu_full_es, global_mmlu_full_fa, global_mmlu_full_fil, global_mmlu_full_fr, global_mmlu_full_ha, global_mmlu_full_he, global_mmlu_full_hi, global_mmlu_full_id, global_mmlu_full_ig, global_mmlu_full_it, global_mmlu_full_ja, global_mmlu_full_ko, global_mmlu_full_ky, global_mmlu_full_lt, global_mmlu_full_mg, global_mmlu_full_ms, global_mmlu_full_ne, global_mmlu_full_nl, global_mmlu_full_ny, global_mmlu_full_pl, global_mmlu_full_pt, global_mmlu_full_ro, global_mmlu_full_ru, global_mmlu_full_si, global_mmlu_full_sn, global_mmlu_full_so, global_mmlu_full_sr, global_mmlu_full_sv, global_mmlu_full_sw, global_mmlu_full_te, global_mmlu_full_tr, global_mmlu_full_uk, global_mmlu_full_vi, global_mmlu_full_yo, global_mmlu_full_zh, global_mmlu_hi, global_mmlu_id, global_mmlu_it, global_mmlu_ja, global_mmlu_ko, global_mmlu_pt, global_mmlu_sw, global_mmlu_yo, global_mmlu_zh, gpqa, gpqa_diamond_cot, gpqa_diamond_cot_5_shot, gsm8k, gsm8k_cot_instruct, gsm8k_cot_llama, gsm8k_cot_zeroshot, gsm8k_cot_zeroshot_llama, hellaswag, hellaswag_multilingual, humaneval_instruct, ifeval, m_mmlu_id_str, mbpp_plus, mgsm, mgsm_cot, mmlu, mmlu_cot_0_shot_chat, mmlu_instruct, mmlu_logits, mmlu_pro, mmlu_pro_instruct, mmlu_prox, mmlu_prox_de, mmlu_prox_es, mmlu_prox_fr, mmlu_prox_it, mmlu_prox_ja, mmlu_redux, mmlu_redux_instruct, musr, openbookqa, piqa, social_iqa, truthfulqa, wikilingua, winogrande
* - **mmath**
  - Multilingual math reasoning
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath)
  - {{ docker_compose_latest }}
  - mmath_ar, mmath_en, mmath_es, mmath_fr, mmath_ja, mmath_ko, mmath_pt, mmath_th, mmath_vi, mmath_zh
* - **mtbench**
  - Multi-turn conversation evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
  - {{ docker_compose_latest }}
  - mtbench, mtbench-cor1
* - **nemo-skills** 
  - Language model benchmarks (science, math, agentic)
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills)
  - {{ docker_compose_latest }}
  - ns_aime2024, ns_aime2025, ns_aime2025_ef, ns_bfcl_v3, ns_gpqa, ns_gpqa_ef, ns_hle, ns_livecodebench, ns_mmlu, ns_mmlu_pro 
* - **rag_retriever_eval**
  - RAG system evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval)
  - {{ docker_compose_latest }}
  - RAG, Retriever
* - **safety-harness**
  - Safety and bias evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
  - {{ docker_compose_latest }}
  - aegis_v2, aegis_v2_ar, aegis_v2_de, aegis_v2_es, aegis_v2_fr, aegis_v2_hi, aegis_v2_ja, aegis_v2_reasoning, aegis_v2_th, aegis_v2_zh-CN, bbq_full, bbq_small, wildguard
* - **scicode**
  - Coding for scientific research
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)
  - {{ docker_compose_latest }}
  - aa_scicode, scicode, scicode_background
* - **simple-evals**
  - Basic evaluation tasks
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
  - {{ docker_compose_latest }}
  - AA_AIME_2024, AA_math_test_500, AIME_2024, AIME_2025, aime_2024_nemo, aime_2025_nemo, gpqa_diamond, gpqa_diamond_aa_v2, gpqa_diamond_aa_v2_llama_4, gpqa_diamond_nemo, gpqa_extended, gpqa_main, humaneval, humanevalplus, math_test_500, math_test_500_nemo, mgsm, mmlu, mmlu_am, mmlu_ar, mmlu_ar-lite, mmlu_bn, mmlu_bn-lite, mmlu_cs, mmlu_de, mmlu_de-lite, mmlu_el, mmlu_en, mmlu_en-lite, mmlu_es, mmlu_es-lite, mmlu_fa, mmlu_fil, mmlu_fr, mmlu_fr-lite, mmlu_ha, mmlu_he, mmlu_hi, mmlu_hi-lite, mmlu_id, mmlu_id-lite, mmlu_ig, mmlu_it, mmlu_it-lite, mmlu_ja, mmlu_ja-lite, mmlu_ko, mmlu_ko-lite, mmlu_ky, mmlu_llama_4, mmlu_lt, mmlu_mg, mmlu_ms, mmlu_my-lite, mmlu_ne, mmlu_nl, mmlu_ny, mmlu_pl, mmlu_pro, mmlu_pro_llama_4, mmlu_pt, mmlu_pt-lite, mmlu_ro, mmlu_ru, mmlu_si, mmlu_sn, mmlu_so, mmlu_sr, mmlu_sv, mmlu_sw, mmlu_sw-lite, mmlu_te, mmlu_tr, mmlu_uk, mmlu_vi, mmlu_yo, mmlu_yo-lite, mmlu_zh-lite, simpleqa
* - **tooltalk**
  - Tool usage evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)
  - {{ docker_compose_latest }}
  - tooltalk
* - **vlmevalkit**
  - Vision-language model evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
  - {{ docker_compose_latest }}
  - ai2d_judge, chartqa, ocrbench, slidevqa
```

```bash
# Pull and run any evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
```

### Advanced Adapter System
Sophisticated request/response processing pipeline with interceptor architecture:

```yaml
# Configure adapter system in YAML configuration
target:
  api_endpoint:
    url: "http://localhost:8080/v1/completions/"
    model_id: "my-model"
    adapter_config:
      interceptors:
        # System message interceptor
        - name: system_message
          config:
            system_message: "You are a helpful AI assistant. Think step by step."
        
        # Request logging interceptor
        - name: request_logging
          config:
            max_requests: 1000
        
        # Caching interceptor
        - name: caching
          config:
            cache_dir: "./evaluation_cache"
        
        # Communication with http://localhost:8080/v1/completions/
        -name: endpoint

        # Reasoning interceptor
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
        
        # Response logging interceptor
        - name: response_logging
          config:
            max_responses: 1000
        
        # Progress tracking interceptor
        - name: progress_tracking
```

### Programmatic API
Full Python API for integration into ML pipelines:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, EvaluationTarget

# Configure and run evaluation programmatically
result = evaluate(
    eval_cfg=EvaluationConfig(type="mmlu_pro", output_dir="./results"),
    target_cfg=EvaluationTarget(api_endpoint=endpoint_config)
)
```

##  **Container Direct Access**

### NGC Container Catalog
Direct access to specialized evaluation containers through [NGC Catalog](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=label%3A%22NSPECT-JL1B-TVGU%22):

```bash
# Academic benchmarks
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# Code generation evaluation  
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}

# Safety and security testing
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}

# Vision-language model evaluation
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/vlmevalkit:{{ docker_compose_latest }}
```

### Reproducible Evaluation Environments
Every container provides:
- **Fixed dependencies**: Locked versions for consistent results
- **Pre-configured frameworks**: Ready-to-run evaluation harnesses  
- **Isolated execution**: No dependency conflicts between evaluations
- **Version tracking**: Tagged releases for exact reproducibility

##  **Enterprise Features**

### Multi-Backend Scalability
Scale from laptop to datacenter with unified configuration:

- **Local Development**: Quick iteration with Docker
- **HPC Clusters**: Slurm integration for large-scale evaluation
- **Cloud Platforms**: Lepton AI and custom backend support
- **Hybrid Workflows**: Seamless transition between environments

### Advanced Configuration Management
Hydra-based configuration with full reproducibility:

```yaml
# Evaluation configuration with overrides
evaluation:
  tasks:
    - name: mmlu_pro
      overrides:
        config.params.limit_samples: 1000
    - name: gsm8k
      overrides:
        config.params.temperature: 0.0

execution:
  output_dir: results

target:
  api_endpoint:
    url: https://my-model-endpoint.com/v1/chat/completions
    model_id: my-custom-model
```

##  **OpenAI API Compatibility**

### Universal Model Support
NeMo Evaluator supports OpenAI-compatible API endpoints:

- **Hosted Models**: NVIDIA Build, OpenAI, Anthropic, Cohere
- **Self-Hosted**: vLLM, TRT-LLM, NeMo Framework
- **Custom Endpoints**: Any service implementing OpenAI API spec

The platform supports the following endpoint types:

- **`completions`**: Direct text completion without chat formatting (`/v1/completions`). Used for base models and academic benchmarks.
- **`chat`**: Conversational interface with role-based messages (`/v1/chat/completions`). Used for instruction-tuned and chat models.
- **`vlm`**: Vision-language model endpoints supporting image inputs.
- **`embedding`**: Embedding generation endpoints for retrieval evaluation.

### Endpoint Type Support
Support for diverse evaluation endpoint types through the evaluation configuration:

```yaml
# Text generation evaluation (chat endpoint)
target:
  api_endpoint:
    type: chat
    url: https://api.example.com/v1/chat/completions

# Log-probability evaluation (completions endpoint)
target:
  api_endpoint:
    type: completions
    url: https://api.example.com/v1/completions

# Vision-language evaluation (vlm endpoint)
target:
  api_endpoint:
    type: vlm
    url: https://api.example.com/v1/chat/completions

# Retrival evaluation (embedding endpoint)
target:
  api_endpoint:
    type: embedding
    url: https://api.example.com/v1/embeddings

```

##  **Extensibility and Customization**

### Custom Framework Support
Add your own evaluation frameworks using Framework Definition Files:

```yaml
# custom_framework.yml
framework:
  name: my_custom_eval
  description: Custom evaluation for domain-specific tasks
  
defaults:
  command: >-
    python custom_eval.py --model {{target.api_endpoint.model_id}}
    --task {{config.params.task}} --output {{config.output_dir}}
    
evaluations:
  - name: domain_specific_task
    description: Evaluate domain-specific capabilities
    defaults:
      config:
        params:
          task: domain_task
          temperature: 0.0
```

### Advanced Interceptor Configuration
Fine-tune request/response processing with the adapter system through YAML configuration:

```yaml
# Production-ready adapter configuration in framework YAML
target:
  api_endpoint:
    url: "https://production-api.com/v1/completions"
    model_id: "production-model"
    adapter_config:
      log_failed_requests: true
      interceptors:
        # System message interceptor
        - name: system_message
          config:
            system_message: "You are an expert AI assistant specialized in this domain."
        
        # Request logging interceptor
        - name: request_logging
          config:
            max_requests: 5000
        
        # Caching interceptor
        - name: caching
          config:
            cache_dir: "./production_cache"
        
        # Reasoning interceptor
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
        
        # Response logging interceptor
        - name: response_logging
          config:
            max_responses: 5000
        
        # Progress tracking interceptor
        - name: progress_tracking
          config:
            progress_tracking_url: "http://monitoring.internal:3828/progress"
```

##  **Security and Safety**

### Comprehensive Safety Evaluation
Built-in safety assessment through specialized containers:

```bash
# Run safety evaluation suite
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["aegis_v2", "garak"]'
```

**Safety Containers Available:**
- **safety-harness**: Content safety evaluation using NemoGuard judge models
- **garak**: Security vulnerability scanning and prompt injection detection  
- **agentic_eval**: Tool usage and planning evaluation for agentic AI systems

##  **Monitoring and Observability**

### Real-Time Progress Tracking
Monitor evaluation progress across all backends:

```bash
# Check evaluation status
nemo-evaluator-launcher status <invocation_id>

# Kill running evaluations
nemo-evaluator-launcher kill <invocation_id>
```

### Result Export and Analysis
Export evaluation results to MLOps platforms for downstream analysis:

```bash
# Export to MLflow for experiment tracking
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases for visualization
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets for sharing
nemo-evaluator-launcher export <invocation_id> --dest gsheets
```
