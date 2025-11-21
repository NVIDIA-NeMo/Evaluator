
```{list-table}
:header-rows: 1
:widths: 20 25 15 15 25

* - Container
  - Description
  - NGC Catalog
  - Latest Tag
  - Key Benchmarks
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
  - aci_bench, ehr_sql, head_qa, med_dialog_healthcaremagic, med_dialog_icliniq, medbullets, medcalc_bench, medec, medhallu, medi_qa, medication_qa, mtsamples_procedures, mtsamples_replicate, pubmed_qa, race_based_med
* - **hle**
  - Academic knowledge and problem solving
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)
  - {{ docker_compose_latest }}
  - hle, hle_aa_v2
* - **ifbench**
  - Instruction following evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)
  - {{ docker_compose_latest }}
  - ifbench, ifbench_aa_v2
* - **livecodebench**
  - Live coding evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)
  - {{ docker_compose_latest }}
  - AA_codegeneration, codeexecution_v2, codeexecution_v2_cot, codegeneration_notfast, codegeneration_release_latest, codegeneration_release_v1, codegeneration_release_v2, codegeneration_release_v3, codegeneration_release_v4, codegeneration_release_v5, codegeneration_release_v6, livecodebench_0724_0125, livecodebench_0824_0225, livecodebench_aa_v2, testoutputprediction
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
  - ns_aime2024, ns_aime2025, ns_bfcl_v3, ns_gpqa, ns_hle, ns_livecodebench, ns_mmlu, ns_mmlu_pro
* - **profbench**
  - Professional domains in Business and Scientific Research
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench)
  - {{ docker_compose_latest }}
  - llm_judge, report_generation
* - **safety-harness**
  - Safety and bias evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
  - {{ docker_compose_latest }}
  - aegis_v2, aegis_v2_reasoning, wildguard
* - **scicode**
  - Coding for scientific research
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)
  - {{ docker_compose_latest }}
  - aa_scicode, scicode, scicode_aa_v2, scicode_background
* - **simple-evals**
  - Basic evaluation tasks
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
  - {{ docker_compose_latest }}
  - AA_AIME_2024, AA_math_test_500, AIME_2024, AIME_2025, AIME_2025_aa_v2, aime_2024_nemo, aime_2025_nemo, browsecomp, gpqa_diamond, gpqa_diamond_aa_v2, gpqa_diamond_aa_v2_llama_4, gpqa_diamond_nemo, gpqa_extended, gpqa_main, healthbench, healthbench_consensus, healthbench_hard, humaneval, humanevalplus, math_test_500, math_test_500_nemo, mgsm, mgsm_aa_v2, mmlu, mmlu_am, mmlu_ar, mmlu_ar-lite, mmlu_bn, mmlu_bn-lite, mmlu_cs, mmlu_de, mmlu_de-lite, mmlu_el, mmlu_en, mmlu_en-lite, mmlu_es, mmlu_es-lite, mmlu_fa, mmlu_fil, mmlu_fr, mmlu_fr-lite, mmlu_ha, mmlu_he, mmlu_hi, mmlu_hi-lite, mmlu_id, mmlu_id-lite, mmlu_ig, mmlu_it, mmlu_it-lite, mmlu_ja, mmlu_ja-lite, mmlu_ko, mmlu_ko-lite, mmlu_ky, mmlu_llama_4, mmlu_lt, mmlu_mg, mmlu_ms, mmlu_my-lite, mmlu_ne, mmlu_nl, mmlu_ny, mmlu_pl, mmlu_pro, mmlu_pro_aa_v2, mmlu_pro_llama_4, mmlu_pt, mmlu_pt-lite, mmlu_ro, mmlu_ru, mmlu_si, mmlu_sn, mmlu_so, mmlu_sr, mmlu_sv, mmlu_sw, mmlu_sw-lite, mmlu_te, mmlu_tr, mmlu_uk, mmlu_vi, mmlu_yo, mmlu_yo-lite, mmlu_zh-lite, simpleqa
* - **tooltalk**
  - Tool usage evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)
  - {{ docker_compose_latest }}
  - tooltalk
* - **vlmevalkit**
  - Vision-language model evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
  - {{ docker_compose_latest }}
  - ai2d_judge, chartqa, mathvista-mini, mmmu_judge, ocrbench, slidevqa
```