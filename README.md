# NeMo Evaluator SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/downloads/)
[![Tests](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Evaluator/actions/workflows/cicd-main.yml)
[![codecov](https://codecov.io/github/NVIDIA-NeMo/Evaluator/graph/badge.svg)](https://codecov.io/github/NVIDIA-NeMo/Evaluator)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![nemo-evaluator PyPI version](https://img.shields.io/pypi/v/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator.svg)](https://pypi.org/project/nemo-evaluator/)
[![nemo-evaluator-launcher PyPI version](https://img.shields.io/pypi/v/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![nemo-evaluator-launcher PyPI downloads](https://img.shields.io/pypi/dm/nemo-evaluator-launcher.svg)](https://pypi.org/project/nemo-evaluator-launcher/)
[![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)](#)

## [📖 Documentation](https://docs.nvidia.com/nemo/evaluator/latest/)

NeMo Evaluator SDK is an open-source platform for robust, reproducible, and scalable evaluation of Large Language Models. It enables you to run hundreds of benchmarks across popular evaluation harnesses against any OpenAI-compatible model API. Evaluations execute in open-source Docker containers for auditable and trustworthy results. The platform's containerized architecture allows for the rapid integration of public benchmarks and private datasets.

NeMo Evaluator SDK is built on four core principles to provide a reliable and versatile evaluation experience:

- **Reproducibility by Default**: All configurations, random seeds, and software provenance are captured automatically for auditable and repeatable evaluations.
- **Scale Anywhere**: Run evaluations from a local machine to a Slurm cluster or cloud-native backends like Lepton AI without changing your workflow.
- **State-of-the-Art Benchmarking**: Access a comprehensive suite of over 100 benchmarks from 18 popular open-source evaluation harnesses. See the full list of [Supported benchmarks and evaluation harnesses](#-supported-benchmarks-and-evaluation-harnesses).
- **Extensible and Customizable**: Integrate new evaluation harnesses, add custom benchmarks with proprietary data, and define custom result exporters for existing MLOps tooling.

## ⚙️ How It Works: Launcher and Core Engine

The platform consists of two main components:

- **`nemo-evaluator` ([The Evaluation Core Engine](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/index.html))**: A Python library that manages the interaction between an evaluation harness and the model being tested.
- **`nemo-evaluator-launcher` ([The CLI and Orchestration](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/index.html))**: The primary user interface and orchestration layer. It handles configuration, selects the execution environment, and launches the appropriate container to run the evaluation.

Most users typically interact with `nemo-evaluator-launcher`, which serves as a universal gateway to different benchmarks and harnesses. However, it is also possible to interact directly with `nemo-evaluator` by following this [guide](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/workflows/cli.html).


## 📊 Supported Benchmarks and Evaluation Harnesses

NeMo Evaluator Launcher provides pre-built evaluation containers for different evaluation harnesses through the NVIDIA NGC catalog. Each harness supports a variety of benchmarks, which can then be called via `nemo-evaluator`. This table provides a list of benchmark names per harness. A more detailed list of task names can be found in the [list of NGC containers](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/containers/index.html).

| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **bfcl** | Function calling | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) | `26.01` | BFCL v2 and v3 |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `26.01` | MBPP, MBPP-Plus, HumanEval, HumanEval+, Multiple (cpp, cs, d, go, java, jl, js, lua, php, pl, py, r, rb, rkt, rs, scala, sh, swift, ts) |
| **compute-eval** | CUDA code evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/compute-eval) | `26.01` | CCCL, Combined Problems, CUDA |
| **CoDec** | Contamination detection | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/contamination-detection) | `26.01` | CODEC, MTEB |
| **garak** | Safety and vulnerability testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `26.01` | Garak |
| **genai-perf** | GenAI performance benchmarking | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf) | `26.01` | GenAI Perf Generation & Summarization |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `26.01` | MedHelm |
| **hle** | Academic knowledge and problem solving | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle) | `26.01` | HLE |
| **ifbench** | Instruction following | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench) | `26.01` | IFBench |
| **livecodebench** | Coding | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench) | `26.01` | LiveCodeBench (v1-v6, 0724_0125, 0824_0225) |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `26.01` | ARC Challenge (also multilingual), GSM8K, HumanEval, HumanEval+, MBPP, MBPP+, MINERVA Math, RACE, AGIEval, BBH, BBQ, CSQA, Frames, Global MMLU, GPQA-D, HellaSwag (also multilingual), IFEval, MGSM, MMLU, MMLU-Pro, MMLU-ProX (de, es, fr, it, ja), MMLU-Redux, MUSR, OpenbookQA, Piqa, Social IQa, TruthfulQA, WikiLingua, WinoGrande |
| **long-context-eval** | Long context evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/long-context-eval) | `26.01` | Long Context Evaluation |
| **mmath** | Multilingual math reasoning | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath) | `26.01` | EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `26.01` | MT-Bench |
| **nemo-skills** | Language model benchmarks (science, math, agentic)  | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo-skills) | `26.01` | AIME 24 & 25, BFCL_v3, GPQA, HLE, LiveCodeBench, MMLU, MMLU-Pro |
| **profbench** | Professional domains in Business and Scientific Research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench) | `26.01` | ProfBench |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `26.01` | Aegis v2, WildGuard |
| **scicode** | Coding for scientific research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode) | `26.01` | SciCode |
| **simple-evals** | Common evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `26.01` | GPQA-D, MATH-500, AIME 24 & 25, HumanEval, HumanEval+, MGSM, MMLU (also multilingual), MMLU-Pro, MMLU-lite (AR, BN, DE, EN, ES, FR, HI, ID, IT, JA, KO, MY, PT, SW, YO, ZH), SimpleQA, BrowseComp, HealthBench |
| **tau2-bench** | TAU2 benchmark evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tau2-bench) | `26.01` | TAU2-Bench |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `26.01` | ToolTalk |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `26.01` | AI2D, ChartQA, MMMU, MathVista-MINI, OCRBench, SlideVQA |

<!-- BEGIN AUTOGENERATION -->
<!-- mapping toml checksum: sha256:881e6d1de31824c9e77a3e13c0a9ab988d6bab7cc9fab5b298ef1e5b1bdf1af9 -->
<!--
| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |
|-----------|-------------|-------------|------------| ------------|
| **bfcl** | The Berkeley Function Calling Leaderboard V3 (also called Berkeley Tool Calling Leaderboard V3) evaluates the LLM's ability to call functions (aka tools) accurately. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl?version=25.11) | `25.11` | bfclv3, bfclv3_ast, bfclv3_ast_prompting, bfclv2, bfclv2_ast, bfclv2_ast_prompting |
| **bigcode-evaluation-harness** | A framework for the evaluation of autoregressive code generation language models. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness?version=25.11) | `25.11` | humaneval, humaneval_instruct, humanevalplus, mbpp, mbppplus, mbppplus_nemo, multiple-py, multiple-sh, multiple-cpp, multiple-cs, multiple-d, multiple-go, multiple-java, multiple-js, multiple-jl, multiple-lua, multiple-pl, multiple-php, multiple-r, multiple-rkt, multiple-rb, multiple-rs, multiple-scala, multiple-swift, multiple-ts |
| **garak** | Garak is an LLM vulnerability scanner. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak?version=25.11) | `25.11` | garak |
| **genai_perf_eval** | GenAI Perf is a tool to evaluate the performance of LLM endpoints. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf?version=25.11) | `25.11` | genai_perf_summarization, genai_perf_generation |
| **helm** | A framework for evaluating large language models in medical applications across various healthcare tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm?version=25.11) | `25.11` | medcalc_bench, medec, head_qa, medbullets, pubmed_qa, ehr_sql, race_based_med, medhallu, mtsamples_replicate, aci_bench, mtsamples_procedures, medication_qa, med_dialog_healthcaremagic, med_dialog_icliniq, medi_qa |
| **hle** | Humanity's Last Exam (HLE) is a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark of its kind with broad subject coverage. Humanity's Last Exam consists of 3,000 questions across dozens of subjects, including mathematics, humanities, and the natural sciences. HLE is developed globally by subject-matter experts and consists of multiple-choice and short-answer questions suitable for automated grading. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle?version=25.11) | `25.11` | hle, hle_aa_v2 |
| **ifbench** | IFBench is a new, challenging benchmark for precise instruction following. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench?version=25.11) | `25.11` | ifbench, ifbench_aa_v2 |
| **livecodebench** | Holistic and Contamination Free Evaluation of Large Language Models for Code. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench?version=25.11) | `25.11` | codegeneration_release_latest, codegeneration_release_v1, codegeneration_release_v2, codegeneration_release_v3, codegeneration_release_v4, codegeneration_release_v5, codegeneration_release_v6, codegeneration_notfast, testoutputprediction, codeexecution_v2, codeexecution_v2_cot, livecodebench_0724_0125, livecodebench_aa_v2, livecodebench_0824_0225 |
| **lm-evaluation-harness** | This project provides a unified framework to test generative language models on a large number of different evaluation tasks. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness?version=25.11) | `25.11` | mmlu, mmlu_instruct, mmlu_cot_0_shot_chat, ifeval, mmlu_pro, mmlu_pro_instruct, mmlu_redux, mmlu_redux_instruct, m_mmlu_id_str, gsm8k, gsm8k_cot_instruct, gsm8k_cot_zeroshot, gsm8k_cot_llama, gsm8k_cot_zeroshot_llama, humaneval_instruct, mbpp_plus, mgsm, mgsm_cot, wikilingua, winogrande, arc_challenge, arc_challenge_chat, hellaswag, truthfulqa, bbh, bbh_instruct, musr, gpqa, gpqa_diamond_cot, frames_naive, frames_naive_with_links, frames_oracle, commonsense_qa, openbookqa, mmlu_logits, piqa, social_iqa, adlr_agieval_en_cot, adlr_math_500_4_shot_sampled, adlr_race, adlr_truthfulqa_mc2, adlr_arc_challenge_llama_25_shot, adlr_gpqa_diamond_cot_5_shot, adlr_mmlu, adlr_mmlu_pro_5_shot_base, adlr_minerva_math_nemo_4_shot, adlr_gsm8k_cot_8_shot, adlr_humaneval_greedy, adlr_humaneval_sampled, adlr_mbpp_sanitized_3_shot_greedy, adlr_mbpp_sanitized_3_shot_sampled, adlr_global_mmlu_lite_5_shot, adlr_mgsm_native_cot_8_shot, adlr_commonsense_qa_7_shot, adlr_winogrande_5_shot, bbq, arc_multilingual, hellaswag_multilingual, mmlu_prox, mmlu_prox_fr, mmlu_prox_de, mmlu_prox_it, mmlu_prox_ja, mmlu_prox_es, global_mmlu_full, global_mmlu_full_am, global_mmlu_full_ar, global_mmlu_full_bn, global_mmlu_full_cs, global_mmlu_full_de, global_mmlu_full_el, global_mmlu_full_en, global_mmlu_full_es, global_mmlu_full_fa, global_mmlu_full_fil, global_mmlu_full_fr, global_mmlu_full_ha, global_mmlu_full_he, global_mmlu_full_hi, global_mmlu_full_id, global_mmlu_full_ig, global_mmlu_full_it, global_mmlu_full_ja, global_mmlu_full_ko, global_mmlu_full_ky, global_mmlu_full_lt, global_mmlu_full_mg, global_mmlu_full_ms, global_mmlu_full_ne, global_mmlu_full_nl, global_mmlu_full_ny, global_mmlu_full_pl, global_mmlu_full_pt, global_mmlu_full_ro, global_mmlu_full_ru, global_mmlu_full_si, global_mmlu_full_sn, global_mmlu_full_so, global_mmlu_full_sr, global_mmlu_full_sv, global_mmlu_full_sw, global_mmlu_full_te, global_mmlu_full_tr, global_mmlu_full_uk, global_mmlu_full_vi, global_mmlu_full_yo, global_mmlu_full_zh, global_mmlu, global_mmlu_ar, global_mmlu_bn, global_mmlu_de, global_mmlu_en, global_mmlu_es, global_mmlu_fr, global_mmlu_hi, global_mmlu_id, global_mmlu_it, global_mmlu_ja, global_mmlu_ko, global_mmlu_pt, global_mmlu_sw, global_mmlu_yo, global_mmlu_zh, agieval |
| **mmath** | MMATH is a new benchmark specifically designed for multilingual complex reasoning. It comprises 374 carefully selected math problems from high-quality sources, including AIME, CNMO, and MATH-500, and covers ten typologically and geographically diverse languages. Each problem is translated and validated through a rigorous pipeline that combines frontier LLMs with human verification, ensuring semantic consistency. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath?version=25.11) | `25.11` | mmath_en, mmath_zh, mmath_ar, mmath_es, mmath_fr, mmath_ja, mmath_ko, mmath_pt, mmath_th, mmath_vi |
| **mtbench** | MT-bench is designed to test multi-turn conversation and instruction-following ability, covering common use cases and focusing on challenging questions to differentiate models. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench?version=25.11) | `25.11` | mtbench, mtbench-cor1 |
| **nemo_skills** | NeMo Skills - a project to improve skills of LLMs | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills?version=25.11) | `25.11` | ns_aime2024, ns_aime2025, ns_gpqa, ns_bfcl_v3, ns_bfcl_v4, ns_livecodebench, ns_hle, ns_ruler, ns_mmlu, ns_mmlu_pro, ns_scicode, ns_aa_lcr, ns_ifbench |
| **profbench** | Professional domain benchmark for evaluating LLMs on Physics PhD, Chemistry PhD, Finance MBA, and Consulting MBA tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench?version=25.11) | `25.11` | report_generation, llm_judge |
| **safety_eval** | Harness for Safety evaluations | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness?version=25.11) | `25.11` | aegis_v2, aegis_v2_reasoning, wildguard |
| **scicode** | SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode?version=25.11) | `25.11` | scicode, scicode_background, scicode_aa_v2 |
| **simple_evals** | simple-evals - a lightweight library for evaluating language models. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals?version=25.11) | `25.11` | AIME_2025, AIME_2024, AA_AIME_2024, AA_math_test_500, math_test_500, mgsm, humaneval, humanevalplus, mmlu_pro, mmlu_am, mmlu_ar, mmlu_bn, mmlu_cs, mmlu_de, mmlu_el, mmlu_en, mmlu_es, mmlu_fa, mmlu_fil, mmlu_fr, mmlu_ha, mmlu_he, mmlu_hi, mmlu_id, mmlu_ig, mmlu_it, mmlu_ja, mmlu_ko, mmlu_ky, mmlu_lt, mmlu_mg, mmlu_ms, mmlu_ne, mmlu_nl, mmlu_ny, mmlu_pl, mmlu_pt, mmlu_ro, mmlu_ru, mmlu_si, mmlu_sn, mmlu_so, mmlu_sr, mmlu_sv, mmlu_sw, mmlu_te, mmlu_tr, mmlu_uk, mmlu_vi, mmlu_yo, mmlu_ar-lite, mmlu_bn-lite, mmlu_de-lite, mmlu_en-lite, mmlu_es-lite, mmlu_fr-lite, mmlu_hi-lite, mmlu_id-lite, mmlu_it-lite, mmlu_ja-lite, mmlu_ko-lite, mmlu_my-lite, mmlu_pt-lite, mmlu_sw-lite, mmlu_yo-lite, mmlu_zh-lite, mmlu, gpqa_diamond, gpqa_extended, gpqa_main, simpleqa, aime_2025_nemo, aime_2024_nemo, math_test_500_nemo, gpqa_diamond_nemo, gpqa_diamond_aa_v2_llama_4, gpqa_diamond_aa_v2, AIME_2025_aa_v2, mgsm_aa_v2, mmlu_pro_aa_v2, mmlu_llama_4, mmlu_pro_llama_4, healthbench, healthbench_consensus, healthbench_hard, browsecomp |
| **tooltalk** | ToolTalk is designed to evaluate tool-augmented LLMs as a chatbot. ToolTalk contains a handcrafted dataset of 28 easy conversations and 50 hard conversations. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk?version=25.11) | `25.11` | tooltalk |
| **vlmevalkit** | VLMEvalKit is an open-source evaluation toolkit of large vision-language models (LVLMs). It enables one-command evaluation of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt generation-based evaluation for all LVLMs, and provide the evaluation results obtained with both exact matching and LLM-based answer extraction. | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit?version=25.11) | `25.11` | ai2d_judge, chartqa, mathvista-mini, mmmu_judge, ocrbench, ocr_reasoning, slidevqa |
-->
<!-- END AUTOGENERATION -->

## 🚀 Quickstart

Get your first evaluation result in minutes. This guide uses your local machine to run a small benchmark against an OpenAI API-compatible endpoint.

### 1. Install the Launcher

The launcher is the only package required to get started.

```bash
pip install nemo-evaluator-launcher
```

### 2. Set Up Your Model Endpoint

NeMo Evaluator works with any model that exposes an OpenAI-compatible endpoint. For this quickstart, we will use the OpenAI API.

**What is an OpenAI-compatible endpoint?** A server that exposes /v1/chat/completions and /v1/completions endpoints, matching the OpenAI API specification.

**Options for model endpoints:**

- **Hosted endpoints** (fastest): Use ready-to-use hosted models from providers like [build.nvidia.com](https://build.nvidia.com) that expose OpenAI-compatible APIs with no hosting required.
- **Self-hosted options**: Host your own models using tools like NVIDIA NIM, vLLM, or TensorRT-LLM for full control over your evaluation environment.
- **Models trained with NeMo framework**: Host your models trained with NeMo framework by deploying them as OpenAI-compatible endpoints using [NeMo Export-Deploy](https://github.com/nvidia-nemo/export-deploy/tree/main). More detailed user guide [here](https://github.com/nvidia-nemo/evaluator/tree/main/docs/nemo-fw).

<!-- TODO(martas): uncomment once publish -->
<!-- For detailed setup instructions including self-hosted configurations, see the [tutorials](https://docs.nvidia.com/nemo/evaluator/latest/tutorials/). -->

**Getting an NGC API Key for build.nvidia.com:**

To use out-of-the-box build.nvidia.com APIs, you need an API key:

1. Register an account at [build.nvidia.com](https://build.nvidia.com).
2. In the Setup menu under Keys/Secrets, generate an API key.
3. Set the environment variable by executing `export NGC_API_KEY=<YOUR_API_KEY>`.

### 3. Run Your First Evaluation

Run a small evaluation on your local machine. The launcher automatically pulls the correct container and executes the benchmark. The list of benchmarks is directly configured in the YAML file.

**Configuration Examples**: Explore ready-to-use configuration files in [`packages/nemo-evaluator-launcher/examples/`](./packages/nemo-evaluator-launcher/examples/) for local, Lepton, and Slurm deployments with various model hosting options (vLLM, NIM, hosted endpoints).

Once you have the example configuration file, either by cloning this repository or downloading one directly such as `local_nvidia_nemotron_nano_9b_v2.yaml`, you can run the following command:


```bash
nemo-evaluator-launcher run --config packages/nemo-evaluator-launcher/examples/local_nvidia_nemotron_nano_9b_v2.yaml -o execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

After running this command, you will see a `job_id`, which can be used to track the job and its results. All logs will be available in your `<YOUR_OUTPUT_LOCAL_DIR>`.

### 4. Check Your Results

Results, logs, and run configurations are saved locally. Inspect the status of the evaluation job by using the corresponding `job_id`:

```bash
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

## 🤝 Contribution Guide

We welcome community contributions. Please see our [Contribution Guide](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) for instructions on submitting pull requests, reporting issues, and suggesting features.


## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE) file for details.


## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions)
- **Documentation**: [NeMo Evaluator Documentation](https://docs.nvidia.com/nemo/evaluator/latest/)


## 🐛 Known issues

- `nel ls` might require docker authenthication and currently does not support fetching credentials from known password management systems such as MacOS's Keychain or GNOME Keyring.