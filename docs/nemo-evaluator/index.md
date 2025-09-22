# NeMo Evaluator: The Universal Platform for LLM Evaluation

NeMo Evaluator is an open-source evaluation engine. It provides standardized, reproducible AI model evaluation through a containerized architecture and adapter system. It enables you to run evaluations across multiple specialized evaluation harnesses (17+ containers including LM-Eval, HELM, MT-Bench, and others) against any OpenAI-compatible model API. The platform's core strength lies in its interceptor-based adapter architecture. This architecture standardizes request/response flow and optional logging/caching layers. It also includes a collection of ready-to-use evaluation containers published through NVIDIA's NGC catalog.

[Container Reference](./reference/containers.md) | [Using Containers](./workflows/using-containers.md) | [CLI Reference](./reference/cli.md) | [Configuration Guide](./reference/configuring-interceptors.md) | [Python API](./workflows/python-api.md)

---

The architecture is as follows:


         ┌─────────────────────┐
         │                     │
         │  NeMo Evaluator     │
         │  harness            │
         └───▲──────┬──────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:3825)                            │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                                                    │  │   │                      │
         │   │ └─────────────────────-──────────────────────────────┘  │   │   deployed           │
         │   │                                                         │   │   OAI-compatible     │
         │   │                                                         │   │   model endpoint     │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │ ┌─┼──────────────────────────────────────────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘

Interceptors are pieces of independent logic. They are designed to be easy to add separately.

NeMo Evaluator is the core, open-source evaluation engine. It powers standardized, reproducible AI model evaluation across benchmarks. It provides the adapter/interceptor architecture, evaluation workflows, and ready-to-use evaluation containers. These components ensure consistent results across environments and over time.

## How It Differs from the Launcher

- **nemo-evaluator**: Core evaluation engine, adapter system, and evaluation containers. This component focuses on correctness, repeatability, and benchmark definitions.
- **nemo-evaluator-launcher**: Orchestration layer on top of the core engine. It adds a unified CLI, multi-backend execution (local/Slurm/hosted), job monitoring, and exporters. Refer to the [NeMo Evaluator Launcher documentation](../nemo-evaluator-launcher/index.md).

## Key Capabilities

- **Adapter/Interceptor Architecture**: Standardizes how requests and responses flow to your endpoint (OpenAI-compatible) and through optional logging/caching layers
- **Benchmarks and Containers**: Curated evaluation harnesses packaged as reproducible containers
  - Browse available containers: [Container Reference](./reference/containers.md)
- **Flexible Configuration**: Fully resolved configs per run enable exact replays and comparisons
- **Metrics and Artifacts**: Consistent result schemas and artifact layouts for downstream analysis

## Architecture Overview

- Targets an OpenAI-compatible endpoint for the model under test.
- Applies optional interceptors (request/response logging, caching, and others).
- Executes benchmark tasks using the corresponding containerized framework.
- Produces metrics, logs, and artifacts in a standard directory structure.

## Using the Core Library

- **Python API**: Programmatic access to core evaluation functionality
  - API reference: [API Reference](./reference/api.md)
- **Containers**: Run evaluations using the published containers for each framework
  - Container reference: [Container Reference](./reference/containers.md)

For end-to-end CLI and multi-backend orchestration, use the [NeMo Evaluator Launcher](../nemo-evaluator-launcher/index.md).

## Extending

Add your own benchmark or framework by defining its configuration and interfaces:

- Extension guide: [Framework Definition File](./extending/framework-definition-file.md).

## Next Steps

- Read the architecture details and glossary in the main docs.
- Explore containers and pick the benchmarks you need: [Container Reference](./reference/containers.md).
- If you want a turnkey CLI, start with the [NeMo Evaluator Launcher Tutorial](../nemo-evaluator-launcher/tutorial.md).

## NVIDIA NGC Containers

NeMo Evaluator provides pre-built evaluation containers through the NVIDIA NGC catalog:

| Container | Description | NGC Catalog | Latest Tag | Supported Benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **agentic_eval** | Agentic AI evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) | `25.08.1` | agentic_eval_answer_accuracy, agentic_eval_goal_accuracy_with_reference, agentic_eval_goal_accuracy_without_reference, agentic_eval_topic_adherence, agentic_eval_tool_call_accuracy |
| **bfcl** | Function calling | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) | `25.08.1` | bfclv2, bfclv2_ast, bfclv2_ast_prompting, bfclv3, bfclv3_ast, bfclv3_ast_prompting |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.08.1` | humaneval, humaneval_instruct, humanevalplus, mbpp, mbppplus, mbppplus_nemo, multiple-cpp, multiple-cs, multiple-d, multiple-go, multiple-java, multiple-jl, multiple-js, multiple-lua, multiple-php, multiple-pl, multiple-py, multiple-r, multiple-rb, multiple-rkt, multiple-rs, multiple-scala, multiple-sh, multiple-swift, multiple-ts |
| **garak** | Safety and vulnerability testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.08.1` | garak |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.08.1` | ci_bench, ehr_sql, head_qa, med_dialog_healthcaremagic, med_dialog_icliniq, medbullets, medcalc_bench, medec, medhallu, medi_qa, medication_qa, mtsamples_procedures, mtsamples_replicate, pubmed_qa, race_based_med |
| **hle** | Academic knowledge and problem solving | Link: TBD | `25.08.1` | hle |
| **ifbench** | Instruction following | Link: TBD | `25.08.1` | ifbench |
| **livecodebench** | Coding | Link: TBD | `25.08.1` | AA_codegeneration, codeexecution_v2, codeexecution_v2_cot, codegeneration_notfast, codegeneration_release_latest, codegeneration_release_v1, codegeneration_release_v2, codegeneration_release_v3, codegeneration_release_v4, codegeneration_release_v5, codegeneration_release_v6, livecodebench_0724_0125, livecodebench_0824_0225 |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.08.1` | adlr_arc_challenge_llama, adlr_gsm8k_fewshot_cot, adlr_humaneval_greedy, adlr_humanevalplus_greedy, adlr_mbpp_sanitized_3shot_greedy, adlr_mbppplus_greedy_sanitized, adlr_minerva_math_nemo, adlr_mmlu_pro_5_shot_base, adlr_race, adlr_truthfulqa_mc2, agieval, arc_challenge, arc_challenge_chat, arc_multilingual, bbh, bbh_instruct, bbq, commonsense_qa, frames_naive, frames_naive_with_links, frames_oracle, global_mmlu, global_mmlu_ar, global_mmlu_bn, global_mmlu_de, global_mmlu_en, global_mmlu_es, global_mmlu_fr, global_mmlu_full, global_mmlu_full_am, global_mmlu_full_ar, global_mmlu_full_bn, global_mmlu_full_cs, global_mmlu_full_de, global_mmlu_full_el, global_mmlu_full_en, global_mmlu_full_es, global_mmlu_full_fa, global_mmlu_full_fil, global_mmlu_full_fr, global_mmlu_full_ha, global_mmlu_full_he, global_mmlu_full_hi, global_mmlu_full_id, global_mmlu_full_ig, global_mmlu_full_it, global_mmlu_full_ja, global_mmlu_full_ko, global_mmlu_full_ky, global_mmlu_full_lt, global_mmlu_full_mg, global_mmlu_full_ms, global_mmlu_full_ne, global_mmlu_full_nl, global_mmlu_full_ny, global_mmlu_full_pl, global_mmlu_full_pt, global_mmlu_full_ro, global_mmlu_full_ru, global_mmlu_full_si, global_mmlu_full_sn, global_mmlu_full_so, global_mmlu_full_sr, global_mmlu_full_sv, global_mmlu_full_sw, global_mmlu_full_te, global_mmlu_full_tr, global_mmlu_full_uk, global_mmlu_full_vi, global_mmlu_full_yo, global_mmlu_full_zh, global_mmlu_hi, global_mmlu_id, global_mmlu_it, global_mmlu_ja, global_mmlu_ko, global_mmlu_pt, global_mmlu_sw, global_mmlu_yo, global_mmlu_zh, gpqa, gpqa_diamond_cot, gpqa_diamond_cot_5_shot, gsm8k, gsm8k_cot_instruct, gsm8k_cot_llama, gsm8k_cot_zeroshot, gsm8k_cot_zeroshot_llama, hellaswag, hellaswag_multilingual, humaneval_instruct, ifeval, m_mmlu_id_str, mbpp_plus, mgsm, mgsm_cot, mmlu, mmlu_cot_0_shot_chat, mmlu_instruct, mmlu_logits, mmlu_pro, mmlu_pro_instruct, mmlu_prox, mmlu_prox_de, mmlu_prox_es, mmlu_prox_fr, mmlu_prox_it, mmlu_prox_ja, mmlu_redux, mmlu_redux_instruct, musr, openbookqa, piqa, social_iqa, truthfulqa, wikilingua, winogrande |
| **mmath** | Multilingual math reasoning | Link: TBD | `25.08.1` | mmath_ar, mmath_en, mmath_es, mmath_fr, mmath_ja, mmath_ko, mmath_pt, mmath_th, mmath_vi, mmath_zh |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.08.1` | mtbench, mtbench-cor1 |
| **rag_retriever_eval** | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) | `25.08.1` | RAG, Retriever |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.08.1` | aegis_v2, aegis_v2_ar, aegis_v2_de, aegis_v2_es, aegis_v2_fr, aegis_v2_hi, aegis_v2_ja, aegis_v2_reasoning, aegis_v2_th, aegis_v2_zh-CN, bbq_full, bbq_small, wildguard |
| **scicode** | Coding for scientific research | Link: TBD | `25.08.1` | aa_scicode, scicode, scicode_background |
| **simple-evals** | Common evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.08.1` | AA_AIME_2024, AA_math_test_500, AIME_2024, AIME_2025, aime_2024_nemo, aime_2025_nemo, gpqa_diamond, gpqa_diamond_aa_v2, gpqa_diamond_aa_v2_llama_4, gpqa_diamond_nemo, gpqa_extended, gpqa_main, humaneval, humanevalplus, math_test_500, math_test_500_nemo, mgsm, mmlu, mmlu_am, mmlu_ar, mmlu_ar-lite, mmlu_bn, mmlu_bn-lite, mmlu_cs, mmlu_de, mmlu_de-lite, mmlu_el, mmlu_en, mmlu_en-lite, mmlu_es, mmlu_es-lite, mmlu_fa, mmlu_fil, mmlu_fr, mmlu_fr-lite, mmlu_ha, mmlu_he, mmlu_hi, mmlu_hi-lite, mmlu_id, mmlu_id-lite, mmlu_ig, mmlu_it, mmlu_it-lite, mmlu_ja, mmlu_ja-lite, mmlu_ko, mmlu_ko-lite, mmlu_ky, mmlu_llama_4, mmlu_lt, mmlu_mg, mmlu_ms, mmlu_my-lite, mmlu_ne, mmlu_nl, mmlu_ny, mmlu_pl, mmlu_pro, mmlu_pro_llama_4, mmlu_pt, mmlu_pt-lite, mmlu_ro, mmlu_ru, mmlu_si, mmlu_sn, mmlu_so, mmlu_sr, mmlu_sv, mmlu_sw, mmlu_sw-lite, mmlu_te, mmlu_tr, mmlu_uk, mmlu_vi, mmlu_yo, mmlu_yo-lite, mmlu_zh-lite, simpleqa |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.08.1` | tooltalk |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.08.1` | ai2d_judge, chartqa, ocrbench, slidevqa |

## Container Usage

```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:<tag>

# Example: Pull agentic evaluation container
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:25.08.1

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/eval-factory/<container-name>:<tag>
```

## Disclaimer

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
