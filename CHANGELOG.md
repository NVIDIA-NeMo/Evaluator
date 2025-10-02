# Changelog

## NVIDIA NeMo-Eval 0.1.0

* Evaluation for Automodel with vllm OAI deployment and nvidia-lm-eval as the eval harness
* Support for Logprob benchmarks with Ray
* Use evaluation APIs from nvidia-eval-commons

### Known Issues

* Very low flexible-extract score with GSM8k due to lack of stop word support in MegatronLLMDeployable. However this does not impact strict-match score
  
## NVIDIA NeMo-Eval 0.1.0a0

* Enable logprobs benchmarks with nvidia-lm-eval  
* Support for new harnesses:  
  * BFCL  
  * BigCode  
  * Simple-evals  
  * Safety-harness  
  * Garak  
* Single node multi instance/DP evaluation with Ray
