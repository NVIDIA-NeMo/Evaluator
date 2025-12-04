# lm-evaluation-harness

This page contains all evaluation tasks for the **lm-evaluation-harness** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [adlr_agieval_en_cot](#lm-evaluation-harness-adlr-agieval-en-cot)
  - ADLR version of the AGIEval-EN-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_arc_challenge_llama_25_shot](#lm-evaluation-harness-adlr-arc-challenge-llama-25-shot)
  - ARC-Challenge-Llama version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_commonsense_qa_7_shot](#lm-evaluation-harness-adlr-commonsense-qa-7-shot)
  - CommonsenseQA version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_global_mmlu_lite_5_shot](#lm-evaluation-harness-adlr-global-mmlu-lite-5-shot)
  - Global-MMLU subset (8 languages - es, de, fr, zh, it, ja, pt, ko) used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_gpqa_diamond_cot_5_shot](#lm-evaluation-harness-adlr-gpqa-diamond-cot-5-shot)
  - ADLR version of the GPQA-Diamond-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_gsm8k_cot_8_shot](#lm-evaluation-harness-adlr-gsm8k-cot-8-shot)
  - GSM8K-CoT version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_humaneval_greedy](#lm-evaluation-harness-adlr-humaneval-greedy)
  - HumanEval Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_humaneval_sampled](#lm-evaluation-harness-adlr-humaneval-sampled)
  - HumanEval Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_math_500_4_shot_sampled](#lm-evaluation-harness-adlr-math-500-4-shot-sampled)
  - MATH-500 Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_mbpp_sanitized_3_shot_greedy](#lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-greedy)
  - MBPP Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_mbpp_sanitized_3_shot_sampled](#lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-sampled)
  - MBPP Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_mgsm_native_cot_8_shot](#lm-evaluation-harness-adlr-mgsm-native-cot-8-shot)
  - MGSM native CoT subset (6 languages - es, de, fr, zh, ja, ru) used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_minerva_math_nemo_4_shot](#lm-evaluation-harness-adlr-minerva-math-nemo-4-shot)
  - Minerva-Math version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_mmlu](#lm-evaluation-harness-adlr-mmlu)
  - MMLU version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_mmlu_pro_5_shot_base](#lm-evaluation-harness-adlr-mmlu-pro-5-shot-base)
  - MMLU-Pro 5-shot base version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_race](#lm-evaluation-harness-adlr-race)
  - RACE version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_truthfulqa_mc2](#lm-evaluation-harness-adlr-truthfulqa-mc2)
  - TruthfulQA-MC2 version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [adlr_winogrande_5_shot](#lm-evaluation-harness-adlr-winogrande-5-shot)
  - Winogrande version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [agieval](#lm-evaluation-harness-agieval)
  - AGIEval - A Human-Centric Benchmark for Evaluating Foundation Models
* - [arc_challenge](#lm-evaluation-harness-arc-challenge)
  - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.
* - [arc_challenge_chat](#lm-evaluation-harness-arc-challenge-chat)
  - - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [arc_multilingual](#lm-evaluation-harness-arc-multilingual)
  - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.
* - [bbh](#lm-evaluation-harness-bbh)
  - The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.
* - [bbh_instruct](#lm-evaluation-harness-bbh-instruct)
  - The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.
* - [bbq](#lm-evaluation-harness-bbq)
  - The BBQ (Bias Benchmark for QA) is a benchmark designed to measure social biases in question answering systems. It contains ambiguous questions spanning 9 categories - disability, gender, nationality, physical appearance, race/ethnicity, religion, sexual orientation, socioeconomic status, and age.
* - [commonsense_qa](#lm-evaluation-harness-commonsense-qa)
  - - CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. - It contains 12,102 questions with one correct answer and four distractor answers.
* - [frames_naive](#lm-evaluation-harness-frames-naive)
  - Frames Naive uses the prompt as input without additional context
* - [frames_naive_with_links](#lm-evaluation-harness-frames-naive-with-links)
  - Frames Naive with Links provides the prompt and relevant Wikipedia article links
* - [frames_oracle](#lm-evaluation-harness-frames-oracle)
  - Frames Oracle (long context) provides prompts and relevant text from curated and processed Wikipedia articles from "parasail-ai/frames-benchmark-wikipedia".
* - [global_mmlu](#lm-evaluation-harness-global-mmlu)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_ar](#lm-evaluation-harness-global-mmlu-ar)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_bn](#lm-evaluation-harness-global-mmlu-bn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_de](#lm-evaluation-harness-global-mmlu-de)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_en](#lm-evaluation-harness-global-mmlu-en)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_es](#lm-evaluation-harness-global-mmlu-es)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_fr](#lm-evaluation-harness-global-mmlu-fr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full](#lm-evaluation-harness-global-mmlu-full)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_am](#lm-evaluation-harness-global-mmlu-full-am)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ar](#lm-evaluation-harness-global-mmlu-full-ar)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_bn](#lm-evaluation-harness-global-mmlu-full-bn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_cs](#lm-evaluation-harness-global-mmlu-full-cs)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_de](#lm-evaluation-harness-global-mmlu-full-de)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_el](#lm-evaluation-harness-global-mmlu-full-el)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_en](#lm-evaluation-harness-global-mmlu-full-en)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_es](#lm-evaluation-harness-global-mmlu-full-es)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_fa](#lm-evaluation-harness-global-mmlu-full-fa)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_fil](#lm-evaluation-harness-global-mmlu-full-fil)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_fr](#lm-evaluation-harness-global-mmlu-full-fr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ha](#lm-evaluation-harness-global-mmlu-full-ha)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_he](#lm-evaluation-harness-global-mmlu-full-he)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_hi](#lm-evaluation-harness-global-mmlu-full-hi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_id](#lm-evaluation-harness-global-mmlu-full-id)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ig](#lm-evaluation-harness-global-mmlu-full-ig)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_it](#lm-evaluation-harness-global-mmlu-full-it)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ja](#lm-evaluation-harness-global-mmlu-full-ja)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ko](#lm-evaluation-harness-global-mmlu-full-ko)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ky](#lm-evaluation-harness-global-mmlu-full-ky)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_lt](#lm-evaluation-harness-global-mmlu-full-lt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_mg](#lm-evaluation-harness-global-mmlu-full-mg)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ms](#lm-evaluation-harness-global-mmlu-full-ms)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ne](#lm-evaluation-harness-global-mmlu-full-ne)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_nl](#lm-evaluation-harness-global-mmlu-full-nl)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ny](#lm-evaluation-harness-global-mmlu-full-ny)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_pl](#lm-evaluation-harness-global-mmlu-full-pl)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_pt](#lm-evaluation-harness-global-mmlu-full-pt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ro](#lm-evaluation-harness-global-mmlu-full-ro)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_ru](#lm-evaluation-harness-global-mmlu-full-ru)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_si](#lm-evaluation-harness-global-mmlu-full-si)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_sn](#lm-evaluation-harness-global-mmlu-full-sn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_so](#lm-evaluation-harness-global-mmlu-full-so)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_sr](#lm-evaluation-harness-global-mmlu-full-sr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_sv](#lm-evaluation-harness-global-mmlu-full-sv)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_sw](#lm-evaluation-harness-global-mmlu-full-sw)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_te](#lm-evaluation-harness-global-mmlu-full-te)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_tr](#lm-evaluation-harness-global-mmlu-full-tr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_uk](#lm-evaluation-harness-global-mmlu-full-uk)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_vi](#lm-evaluation-harness-global-mmlu-full-vi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_yo](#lm-evaluation-harness-global-mmlu-full-yo)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_full_zh](#lm-evaluation-harness-global-mmlu-full-zh)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_hi](#lm-evaluation-harness-global-mmlu-hi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_id](#lm-evaluation-harness-global-mmlu-id)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_it](#lm-evaluation-harness-global-mmlu-it)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_ja](#lm-evaluation-harness-global-mmlu-ja)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_ko](#lm-evaluation-harness-global-mmlu-ko)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_pt](#lm-evaluation-harness-global-mmlu-pt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_sw](#lm-evaluation-harness-global-mmlu-sw)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_yo](#lm-evaluation-harness-global-mmlu-yo)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [global_mmlu_zh](#lm-evaluation-harness-global-mmlu-zh)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [gpqa](#lm-evaluation-harness-gpqa)
  - The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.
* - [gpqa_diamond_cot](#lm-evaluation-harness-gpqa-diamond-cot)
  - The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.
* - [gsm8k](#lm-evaluation-harness-gsm8k)
  - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems.
* - [gsm8k_cot_instruct](#lm-evaluation-harness-gsm8k-cot-instruct)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation with custom instructions.
* - [gsm8k_cot_llama](#lm-evaluation-harness-gsm8k-cot-llama)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought evaluation - implementation taken from llama.
* - [gsm8k_cot_zeroshot](#lm-evaluation-harness-gsm8k-cot-zeroshot)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation.
* - [gsm8k_cot_zeroshot_llama](#lm-evaluation-harness-gsm8k-cot-zeroshot-llama)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation - implementation taken from llama.
* - [hellaswag](#lm-evaluation-harness-hellaswag)
  - The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.
* - [hellaswag_multilingual](#lm-evaluation-harness-hellaswag-multilingual)
  - The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.
* - [humaneval_instruct](#lm-evaluation-harness-humaneval-instruct)
  - - The HumanEval benchmark measures functional correctness for synthesizing programs from docstrings. - Implementation taken from llama.
* - [ifeval](#lm-evaluation-harness-ifeval)
  - IFEval is a dataset designed to test a model's ability to follow explicit instructions, such as "include keyword x" or "use format y." The focus is on the model's adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.
* - [m_mmlu_id_str](#lm-evaluation-harness-m-mmlu-id-str)
  - - The MMLU (Massive Multitask Language Understanding) benchmark translated to Indonesian. - This variant uses the Indonesian version of the MMLU tasks with string-based evaluation.
* - [mbpp_plus](#lm-evaluation-harness-mbpp-plus)
  - MBPP EvalPlus is an extension of the MBPP benchmark that explores the limits of the current generation of large language models for program synthesis in general purpose programming languages.
* - [mgsm](#lm-evaluation-harness-mgsm)
  - The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.
* - [mgsm_cot](#lm-evaluation-harness-mgsm-cot)
  - The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.
* - [mmlu](#lm-evaluation-harness-mmlu)
  - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities.
* - [mmlu_cot_0_shot_chat](#lm-evaluation-harness-mmlu-cot-0-shot-chat)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to chain-of-thought zero-shot evaluation.
* - [mmlu_instruct](#lm-evaluation-harness-mmlu-instruct)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to zero-shot evaluation and instructs the model to produce a single letter response.
* - [mmlu_logits](#lm-evaluation-harness-mmlu-logits)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. - It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant uses the logits of the model to evaluate the accuracy.
* - [mmlu_pro](#lm-evaluation-harness-mmlu-pro)
  - MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.
* - [mmlu_pro_instruct](#lm-evaluation-harness-mmlu-pro-instruct)
  - - MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [mmlu_prox](#lm-evaluation-harness-mmlu-prox)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation
* - [mmlu_prox_de](#lm-evaluation-harness-mmlu-prox-de)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (German dataset)
* - [mmlu_prox_es](#lm-evaluation-harness-mmlu-prox-es)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Spanish dataset)
* - [mmlu_prox_fr](#lm-evaluation-harness-mmlu-prox-fr)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (French dataset)
* - [mmlu_prox_it](#lm-evaluation-harness-mmlu-prox-it)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Italian dataset)
* - [mmlu_prox_ja](#lm-evaluation-harness-mmlu-prox-ja)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Japanese dataset)
* - [mmlu_redux](#lm-evaluation-harness-mmlu-redux)
  - MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects.
* - [mmlu_redux_instruct](#lm-evaluation-harness-mmlu-redux-instruct)
  - - MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [musr](#lm-evaluation-harness-musr)
  - The MuSR (Multistep Soft Reasoning) benchmark evaluates the reasoning capabilities of large language models through complex, multistep tasks specified in natural language narratives. It introduces sophisticated natural language and complex reasoning challenges to test the limits of chain-of-thought prompting.
* - [openbookqa](#lm-evaluation-harness-openbookqa)
  - - OpenBookQA is a question-answering dataset modeled after open book exams for assessing human understanding of a subject. - It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small "book" of 1,326 core science facts and the application of these facts to novel situations. - For training, the dataset includes a mapping from each question to the core science fact it was designed to probe. - Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. - The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.
* - [piqa](#lm-evaluation-harness-piqa)
  - - Physical Interaction: Question Answering (PIQA) is a physical commonsense
  reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
  the physical knowledge of existing models. To what extent are current approaches
  actually learning about the world?
* - [social_iqa](#lm-evaluation-harness-social-iqa)
  - - Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear").
* - [truthfulqa](#lm-evaluation-harness-truthfulqa)
  - The TruthfulQA benchmark measures the truthfulness of language models in generating answers to questions. It consists of 817 questions across 38 categories, such as health, law, finance, and politics, designed to test whether models can avoid generating false answers that mimic common human misconceptions.
* - [wikilingua](#lm-evaluation-harness-wikilingua)
  - The WikiLingua benchmark is a large-scale, multilingual dataset designed for evaluating cross-lingual abstractive summarization systems. It includes approximately 770,000 article-summary pairs in 18 languages, extracted from WikiHow, with gold-standard alignments created by matching images used to describe each how-to step in an article.
* - [winogrande](#lm-evaluation-harness-winogrande)
  - WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires commonsense reasoning.
```

(lm-evaluation-harness-adlr-agieval-en-cot)=
## adlr_agieval_en_cot

ADLR version of the AGIEval-EN-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_agieval_en_cot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_agieval_en_cot
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: adlr_agieval_en_cot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-arc-challenge-llama-25-shot)=
## adlr_arc_challenge_llama_25_shot

ARC-Challenge-Llama version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_arc_challenge_llama_25_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_arc_challenge_llama
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 25
  supported_endpoint_types:
  - completions
  type: adlr_arc_challenge_llama_25_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-commonsense-qa-7-shot)=
## adlr_commonsense_qa_7_shot

CommonsenseQA version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_commonsense_qa_7_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: commonsense_qa
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 7
  supported_endpoint_types:
  - completions
  type: adlr_commonsense_qa_7_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-global-mmlu-lite-5-shot)=
## adlr_global_mmlu_lite_5_shot

Global-MMLU subset (8 languages - es, de, fr, zh, it, ja, pt, ko) used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_global_mmlu_lite_5_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_global_mmlu
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: adlr_global_mmlu_lite_5_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-gpqa-diamond-cot-5-shot)=
## adlr_gpqa_diamond_cot_5_shot

ADLR version of the GPQA-Diamond-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_gpqa_diamond_cot_5_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_gpqa_diamond_cot_5_shot
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: adlr_gpqa_diamond_cot_5_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-gsm8k-cot-8-shot)=
## adlr_gsm8k_cot_8_shot

GSM8K-CoT version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_gsm8k_cot_8_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_gsm8k_fewshot_cot
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 8
  supported_endpoint_types:
  - completions
  type: adlr_gsm8k_cot_8_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-humaneval-greedy)=
## adlr_humaneval_greedy

HumanEval Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_humaneval_greedy`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_humaneval_greedy
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: adlr_humaneval_greedy
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-humaneval-sampled)=
## adlr_humaneval_sampled

HumanEval Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_humaneval_sampled`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_humaneval_sampled
    temperature: 0.6
    request_timeout: 30
    top_p: 0.95
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: adlr_humaneval_sampled
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-math-500-4-shot-sampled)=
## adlr_math_500_4_shot_sampled

MATH-500 Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_math_500_4_shot_sampled`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_math_500_4_shot_sampled
    temperature: 0.7
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 4
  supported_endpoint_types:
  - completions
  type: adlr_math_500_4_shot_sampled
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-greedy)=
## adlr_mbpp_sanitized_3_shot_greedy

MBPP Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_mbpp_sanitized_3_shot_greedy`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_mbpp_sanitized_3_shot_greedy
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 3
  supported_endpoint_types:
  - completions
  type: adlr_mbpp_sanitized_3_shot_greedy
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-mbpp-sanitized-3-shot-sampled)=
## adlr_mbpp_sanitized_3_shot_sampled

MBPP Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_mbpp_sanitized_3_shot_sampled`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_mbpp_sanitized_3shot_sampled
    temperature: 0.6
    request_timeout: 30
    top_p: 0.95
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 3
  supported_endpoint_types:
  - completions
  type: adlr_mbpp_sanitized_3_shot_sampled
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-mgsm-native-cot-8-shot)=
## adlr_mgsm_native_cot_8_shot

MGSM native CoT subset (6 languages - es, de, fr, zh, ja, ru) used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_mgsm_native_cot_8_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_mgsm_native_cot_8_shot
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 8
  supported_endpoint_types:
  - completions
  type: adlr_mgsm_native_cot_8_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-minerva-math-nemo-4-shot)=
## adlr_minerva_math_nemo_4_shot

Minerva-Math version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_minerva_math_nemo_4_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_minerva_math_nemo
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 4
  supported_endpoint_types:
  - completions
  type: adlr_minerva_math_nemo_4_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-mmlu)=
## adlr_mmlu

MMLU version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_mmlu`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_str
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
      args: --trust_remote_code
  supported_endpoint_types:
  - completions
  type: adlr_mmlu
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-mmlu-pro-5-shot-base)=
## adlr_mmlu_pro_5_shot_base

MMLU-Pro 5-shot base version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_mmlu_pro_5_shot_base`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_mmlu_pro_5_shot_base
    temperature: 0.0
    request_timeout: 30
    top_p: 1.0e-05
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: adlr_mmlu_pro_5_shot_base
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-race)=
## adlr_race

RACE version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_race`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_race
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: adlr_race
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-truthfulqa-mc2)=
## adlr_truthfulqa_mc2

TruthfulQA-MC2 version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_truthfulqa_mc2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: adlr_truthfulqa_mc2
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: adlr_truthfulqa_mc2
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-adlr-winogrande-5-shot)=
## adlr_winogrande_5_shot

Winogrande version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `adlr_winogrande_5_shot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: winogrande
    temperature: 1.0
    request_timeout: 30
    top_p: 1.0
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: adlr_winogrande_5_shot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-agieval)=
## agieval

AGIEval - A Human-Centric Benchmark for Evaluating Foundation Models

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `agieval`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: agieval
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: agieval
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-arc-challenge)=
## arc_challenge

The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `arc_challenge`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: arc_challenge
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: arc_challenge
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-arc-challenge-chat)=
## arc_challenge_chat

- The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `arc_challenge_chat`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: arc_challenge_chat
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
  supported_endpoint_types:
  - chat
  type: arc_challenge_chat
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-arc-multilingual)=
## arc_multilingual

The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `arc_multilingual`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: arc_multilingual
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: arc_multilingual
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-bbh)=
## bbh

The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `bbh`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: leaderboard_bbh
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: bbh
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-bbh-instruct)=
## bbh_instruct

The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `bbh_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: bbh_zeroshot
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: bbh_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-bbq)=
## bbq

The BBQ (Bias Benchmark for QA) is a benchmark designed to measure social biases in question answering systems. It contains ambiguous questions spanning 9 categories - disability, gender, nationality, physical appearance, race/ethnicity, religion, sexual orientation, socioeconomic status, and age.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `bbq`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: bbq_generate
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: bbq
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-commonsense-qa)=
## commonsense_qa

- CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. - It contains 12,102 questions with one correct answer and four distractor answers.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `commonsense_qa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: commonsense_qa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 7
  supported_endpoint_types:
  - completions
  type: commonsense_qa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-frames-naive)=
## frames_naive

Frames Naive uses the prompt as input without additional context

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `frames_naive`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: frames_naive
    temperature: 0.0
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: frames_naive
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-frames-naive-with-links)=
## frames_naive_with_links

Frames Naive with Links provides the prompt and relevant Wikipedia article links

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `frames_naive_with_links`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: frames_naive_with_links
    temperature: 0.0
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: frames_naive_with_links
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-frames-oracle)=
## frames_oracle

Frames Oracle (long context) provides prompts and relevant text from curated and processed Wikipedia articles from "parasail-ai/frames-benchmark-wikipedia".

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `frames_oracle`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: frames_oracle
    temperature: 0.0
    request_timeout: 1000
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: frames_oracle
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu)=
## global_mmlu

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-ar)=
## global_mmlu_ar

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_ar`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_ar
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_ar
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-bn)=
## global_mmlu_bn

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_bn`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_bn
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_bn
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-de)=
## global_mmlu_de

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_de`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_de
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_de
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-en)=
## global_mmlu_en

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_en`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_en
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_en
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-es)=
## global_mmlu_es

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_es`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_es
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_es
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-fr)=
## global_mmlu_fr

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_fr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_fr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_fr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full)=
## global_mmlu_full

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-am)=
## global_mmlu_full_am

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_am`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_am
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_am
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ar)=
## global_mmlu_full_ar

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ar`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ar
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ar
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-bn)=
## global_mmlu_full_bn

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_bn`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_bn
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_bn
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-cs)=
## global_mmlu_full_cs

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_cs`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_cs
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_cs
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-de)=
## global_mmlu_full_de

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_de`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_de
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_de
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-el)=
## global_mmlu_full_el

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_el`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_el
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_el
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-en)=
## global_mmlu_full_en

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_en`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_en
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_en
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-es)=
## global_mmlu_full_es

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_es`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_es
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_es
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-fa)=
## global_mmlu_full_fa

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_fa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_fa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_fa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-fil)=
## global_mmlu_full_fil

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_fil`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_fil
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_fil
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-fr)=
## global_mmlu_full_fr

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_fr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_fr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_fr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ha)=
## global_mmlu_full_ha

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ha`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ha
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ha
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-he)=
## global_mmlu_full_he

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_he`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_he
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_he
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-hi)=
## global_mmlu_full_hi

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_hi`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_hi
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_hi
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-id)=
## global_mmlu_full_id

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_id`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_id
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_id
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ig)=
## global_mmlu_full_ig

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ig`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ig
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ig
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-it)=
## global_mmlu_full_it

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_it`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_it
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_it
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ja)=
## global_mmlu_full_ja

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ja`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ja
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ja
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ko)=
## global_mmlu_full_ko

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ko`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ko
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ko
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ky)=
## global_mmlu_full_ky

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ky`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ky
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ky
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-lt)=
## global_mmlu_full_lt

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_lt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_lt
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_lt
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-mg)=
## global_mmlu_full_mg

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_mg`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_mg
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_mg
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ms)=
## global_mmlu_full_ms

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ms`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ms
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ms
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ne)=
## global_mmlu_full_ne

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ne`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ne
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ne
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-nl)=
## global_mmlu_full_nl

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_nl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_nl
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_nl
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ny)=
## global_mmlu_full_ny

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ny`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ny
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ny
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-pl)=
## global_mmlu_full_pl

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_pl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_pl
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_pl
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-pt)=
## global_mmlu_full_pt

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_pt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_pt
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_pt
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ro)=
## global_mmlu_full_ro

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ro`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ro
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ro
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-ru)=
## global_mmlu_full_ru

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_ru`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_ru
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_ru
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-si)=
## global_mmlu_full_si

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_si`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_si
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_si
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-sn)=
## global_mmlu_full_sn

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_sn`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_sn
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_sn
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-so)=
## global_mmlu_full_so

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_so`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_so
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_so
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-sr)=
## global_mmlu_full_sr

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_sr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_sr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_sr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-sv)=
## global_mmlu_full_sv

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_sv`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_sv
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_sv
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-sw)=
## global_mmlu_full_sw

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_sw`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_sw
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_sw
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-te)=
## global_mmlu_full_te

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_te`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_te
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_te
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-tr)=
## global_mmlu_full_tr

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_tr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_tr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_tr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-uk)=
## global_mmlu_full_uk

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_uk`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_uk
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_uk
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-vi)=
## global_mmlu_full_vi

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_vi`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_vi
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_vi
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-yo)=
## global_mmlu_full_yo

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_yo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_yo
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_yo
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-full-zh)=
## global_mmlu_full_zh

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_full_zh`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_full_zh
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_full_zh
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-hi)=
## global_mmlu_hi

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_hi`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_hi
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_hi
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-id)=
## global_mmlu_id

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_id`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_id
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_id
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-it)=
## global_mmlu_it

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_it`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_it
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_it
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-ja)=
## global_mmlu_ja

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_ja`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_ja
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_ja
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-ko)=
## global_mmlu_ko

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_ko`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_ko
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_ko
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-pt)=
## global_mmlu_pt

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_pt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_pt
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_pt
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-sw)=
## global_mmlu_sw

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_sw`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_sw
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_sw
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-yo)=
## global_mmlu_yo

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_yo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_yo
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_yo
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-global-mmlu-zh)=
## global_mmlu_zh

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `global_mmlu_zh`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: global_mmlu_zh
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: global_mmlu_zh
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gpqa)=
## gpqa

The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gpqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: leaderboard_gpqa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: gpqa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gpqa-diamond-cot)=
## gpqa_diamond_cot

The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gpqa_diamond_cot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: gpqa_diamond_cot_zeroshot
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: gpqa_diamond_cot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gsm8k)=
## gsm8k

The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gsm8k`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: gsm8k
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: gsm8k
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gsm8k-cot-instruct)=
## gsm8k_cot_instruct

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation with custom instructions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gsm8k_cot_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: gsm8k_zeroshot_cot
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --add_instruction
  supported_endpoint_types:
  - chat
  type: gsm8k_cot_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gsm8k-cot-llama)=
## gsm8k_cot_llama

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought evaluation - implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gsm8k_cot_llama`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: gsm8k_cot_llama
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: gsm8k_cot_llama
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gsm8k-cot-zeroshot)=
## gsm8k_cot_zeroshot

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gsm8k_cot_zeroshot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: gsm8k_cot_zeroshot
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: gsm8k_cot_zeroshot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-gsm8k-cot-zeroshot-llama)=
## gsm8k_cot_zeroshot_llama

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation - implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `gsm8k_cot_zeroshot_llama`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: gsm8k_cot_llama
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
  supported_endpoint_types:
  - chat
  type: gsm8k_cot_zeroshot_llama
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-hellaswag)=
## hellaswag

The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `hellaswag`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: hellaswag
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 10
  supported_endpoint_types:
  - completions
  type: hellaswag
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-hellaswag-multilingual)=
## hellaswag_multilingual

The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `hellaswag_multilingual`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: hellaswag_multilingual
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 10
  supported_endpoint_types:
  - completions
  type: hellaswag_multilingual
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-humaneval-instruct)=
## humaneval_instruct

- The HumanEval benchmark measures functional correctness for synthesizing programs from docstrings. - Implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `humaneval_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: humaneval_instruct
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: humaneval_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-ifeval)=
## ifeval

IFEval is a dataset designed to test a model's ability to follow explicit instructions, such as "include keyword x" or "use format y." The focus is on the model's adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `ifeval`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: ifeval
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  type: ifeval
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-m-mmlu-id-str)=
## m_mmlu_id_str

- The MMLU (Massive Multitask Language Understanding) benchmark translated to Indonesian. - This variant uses the Indonesian version of the MMLU tasks with string-based evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `m_mmlu_id_str`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: m_mmlu_id_str
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --trust_remote_code
  supported_endpoint_types:
  - chat
  - completions
  type: m_mmlu_id_str
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mbpp-plus)=
## mbpp_plus

MBPP EvalPlus is an extension of the MBPP benchmark that explores the limits of the current generation of large language models for program synthesis in general purpose programming languages.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mbpp_plus`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mbpp_plus
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mbpp_plus
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mgsm)=
## mgsm

The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mgsm`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mgsm_direct
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: mgsm
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mgsm-cot)=
## mgsm_cot

The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mgsm_cot`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: mgsm_cot_native
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
  supported_endpoint_types:
  - chat
  - completions
  type: mgsm_cot
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu)=
## mmlu

The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_str
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
      args: --trust_remote_code
  supported_endpoint_types:
  - completions
  type: mmlu
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-cot-0-shot-chat)=
## mmlu_cot_0_shot_chat

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to chain-of-thought zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_cot_0_shot_chat`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_cot_0_shot_chat
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
  supported_endpoint_types:
  - chat
  type: mmlu_cot_0_shot_chat
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-instruct)=
## mmlu_instruct

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to zero-shot evaluation and instructs the model to produce a single letter response.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_str
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --trust_remote_code --add_instruction
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-logits)=
## mmlu_logits

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. - It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant uses the logits of the model to evaluate the accuracy.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_logits`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: mmlu_logits
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-pro)=
## mmlu_pro

MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_pro`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_pro
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_pro
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-pro-instruct)=
## mmlu_pro_instruct

- MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_pro_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: mmlu_pro
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
  supported_endpoint_types:
  - chat
  type: mmlu_pro_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox)=
## mmlu_prox

A Multilingual Benchmark for Advanced Large Language Model Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox-de)=
## mmlu_prox_de

A Multilingual Benchmark for Advanced Large Language Model Evaluation (German dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox_de`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox_de
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox_de
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox-es)=
## mmlu_prox_es

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Spanish dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox_es`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox_es
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox_es
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox-fr)=
## mmlu_prox_fr

A Multilingual Benchmark for Advanced Large Language Model Evaluation (French dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox_fr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox_fr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox_fr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox-it)=
## mmlu_prox_it

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Italian dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox_it`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox_it
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox_it
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-prox-ja)=
## mmlu_prox_ja

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Japanese dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_prox_ja`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_prox_ja
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - chat
  - completions
  type: mmlu_prox_ja
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-redux)=
## mmlu_redux

MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_redux`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: mmlu_redux
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: mmlu_redux
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-mmlu-redux-instruct)=
## mmlu_redux_instruct

- MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `mmlu_redux_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_new_tokens: 8192
    max_retries: 5
    parallelism: 10
    task: mmlu_redux
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --add_instruction
  supported_endpoint_types:
  - chat
  type: mmlu_redux_instruct
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-musr)=
## musr

The MuSR (Multistep Soft Reasoning) benchmark evaluates the reasoning capabilities of large language models through complex, multistep tasks specified in natural language narratives. It introduces sophisticated natural language and complex reasoning challenges to test the limits of chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `musr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: leaderboard_musr
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: musr
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-openbookqa)=
## openbookqa

- OpenBookQA is a question-answering dataset modeled after open book exams for assessing human understanding of a subject. - It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small "book" of 1,326 core science facts and the application of these facts to novel situations. - For training, the dataset includes a mapping from each question to the core science fact it was designed to probe. - Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. - The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `openbookqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: openbookqa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: openbookqa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-piqa)=
## piqa

- Physical Interaction: Question Answering (PIQA) is a physical commonsense
  reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
  the physical knowledge of existing models. To what extent are current approaches
  actually learning about the world?

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `piqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: piqa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  supported_endpoint_types:
  - completions
  type: piqa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-social-iqa)=
## social_iqa

- Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear").

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `social_iqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: social_iqa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
  supported_endpoint_types:
  - completions
  type: social_iqa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-truthfulqa)=
## truthfulqa

The TruthfulQA benchmark measures the truthfulness of language models in generating answers to questions. It consists of 817 questions across 38 categories, such as health, law, finance, and politics, designed to test whether models can avoid generating false answers that mimic common human misconceptions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `truthfulqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: truthfulqa
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
  type: truthfulqa
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-wikilingua)=
## wikilingua

The WikiLingua benchmark is a large-scale, multilingual dataset designed for evaluating cross-lingual abstractive summarization systems. It includes approximately 770,000 article-summary pairs in 18 languages, extracted from WikiHow, with gold-standard alignments created by matching images used to describe each how-to step in an article.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `wikilingua`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: wikilingua
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
  supported_endpoint_types:
  - chat
  type: wikilingua
target:
  api_endpoint:
    stream: false
```

</details>


(lm-evaluation-harness-winogrande)=
## winogrande

WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires commonsense reasoning.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:db398f277ac73d5c4c0167f9b5e7299c625bceba1da68cf02dd460cb4d8fc60b
```

**Task Type:** `winogrande`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
framework_name: lm-evaluation-harness
pkg_name: lm_evaluation_harness
config:
  params:
    max_retries: 5
    parallelism: 10
    task: winogrande
    temperature: 1.0e-07
    request_timeout: 30
    top_p: 0.9999999
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
  supported_endpoint_types:
  - completions
  type: winogrande
target:
  api_endpoint:
    stream: false
```

</details>

