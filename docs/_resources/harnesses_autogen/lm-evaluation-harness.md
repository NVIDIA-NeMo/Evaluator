# lm-evaluation-harness

This page contains all evaluation tasks for the **lm-evaluation-harness** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [ADLR AGIEval-EN-CoT](#lm-evaluation-harness-adlr-agieval-en-cot)
  - ADLR version of the AGIEval-EN-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR ARC-Challenge-Llama](#lm-evaluation-harness-adlr-arc-challenge-llama)
  - ARC-Challenge-Llama version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR CommonsenseQA](#lm-evaluation-harness-adlr-commonsenseqa)
  - CommonsenseQA version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR GPQA-Diamond-CoT](#lm-evaluation-harness-adlr-gpqa-diamond-cot)
  - ADLR version of the GPQA-Diamond-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR GSM8K-CoT](#lm-evaluation-harness-adlr-gsm8k-cot)
  - GSM8K-CoT version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR Global-MMLU](#lm-evaluation-harness-adlr-global-mmlu)
  - Global-MMLU subset (8 languages - es, de, fr, zh, it, ja, pt, ko) used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR HumanEval Greedy](#lm-evaluation-harness-adlr-humaneval-greedy)
  - HumanEval Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR HumanEval Sampled](#lm-evaluation-harness-adlr-humaneval-sampled)
  - HumanEval Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MATH-500 Sampled](#lm-evaluation-harness-adlr-math-500-sampled)
  - MATH-500 Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MBPP Greedy](#lm-evaluation-harness-adlr-mbpp-greedy)
  - MBPP Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MBPP Sampled](#lm-evaluation-harness-adlr-mbpp-sampled)
  - MBPP Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MGSM-CoT](#lm-evaluation-harness-adlr-mgsm-cot)
  - MGSM native CoT subset (6 languages - es, de, fr, zh, ja, ru) used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MMLU](#lm-evaluation-harness-adlr-mmlu)
  - MMLU version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR MMLU-Pro](#lm-evaluation-harness-adlr-mmlu-pro)
  - MMLU-Pro 5-shot base version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR Minerva-Math](#lm-evaluation-harness-adlr-minerva-math)
  - Minerva-Math version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR RACE](#lm-evaluation-harness-adlr-race)
  - RACE version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR TruthfulQA-MC2](#lm-evaluation-harness-adlr-truthfulqa-mc2)
  - TruthfulQA-MC2 version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [ADLR Winogrande](#lm-evaluation-harness-adlr-winogrande)
  - Winogrande version used by NVIDIA Applied Deep Learning Research team (ADLR).
* - [AGIEval](#lm-evaluation-harness-agieval)
  - AGIEval - A Human-Centric Benchmark for Evaluating Foundation Models
* - [ARC Challenge](#lm-evaluation-harness-arc-challenge)
  - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.
* - [ARC Challenge-instruct](#lm-evaluation-harness-arc-challenge-instruct)
  - - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [ARC Multilingual](#lm-evaluation-harness-arc-multilingual)
  - The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.
* - [BBQ](#lm-evaluation-harness-bbq)
  - The BBQ (Bias Benchmark for QA) is a benchmark designed to measure social biases in question answering systems. It contains ambiguous questions spanning 9 categories - disability, gender, nationality, physical appearance, race/ethnicity, religion, sexual orientation, socioeconomic status, and age.
* - [BIG-Bench Hard](#lm-evaluation-harness-big-bench-hard)
  - The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.
* - [BIG-Bench Hard-instruct](#lm-evaluation-harness-big-bench-hard-instruct)
  - The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.
* - [CommonsenseQA](#lm-evaluation-harness-commonsenseqa)
  - - CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. - It contains 12,102 questions with one correct answer and four distractor answers.
* - [Frames Naive](#lm-evaluation-harness-frames-naive)
  - Frames Naive uses the prompt as input without additional context
* - [Frames Naive with Links](#lm-evaluation-harness-frames-naive-with-links)
  - Frames Naive with Links provides the prompt and relevant Wikipedia article links
* - [Frames Oracle](#lm-evaluation-harness-frames-oracle)
  - Frames Oracle (long context) provides prompts and relevant text from curated and processed Wikipedia articles from "parasail-ai/frames-benchmark-wikipedia".
* - [GPQA](#lm-evaluation-harness-gpqa)
  - The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.
* - [GPQA-Diamond-CoT](#lm-evaluation-harness-gpqa-diamond-cot)
  - The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.
* - [GSM8K](#lm-evaluation-harness-gsm8k)
  - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems.
* - [GSM8K-cot-llama](#lm-evaluation-harness-gsm8k-cot-llama)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought evaluation - implementation taken from llama.
* - [GSM8K-cot-zeroshot](#lm-evaluation-harness-gsm8k-cot-zeroshot)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation.
* - [GSM8K-cot-zeroshot-llama](#lm-evaluation-harness-gsm8k-cot-zeroshot-llama)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation - implementation taken from llama.
* - [GSM8K-instruct](#lm-evaluation-harness-gsm8k-instruct)
  - - The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation with custom instructions.
* - [Global-MMLU](#lm-evaluation-harness-global-mmlu)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-AR](#lm-evaluation-harness-global-mmlu-ar)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-BN](#lm-evaluation-harness-global-mmlu-bn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-DE](#lm-evaluation-harness-global-mmlu-de)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-EN](#lm-evaluation-harness-global-mmlu-en)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-ES](#lm-evaluation-harness-global-mmlu-es)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-FR](#lm-evaluation-harness-global-mmlu-fr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full](#lm-evaluation-harness-global-mmlu-full)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-AM](#lm-evaluation-harness-global-mmlu-full-am)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-AR](#lm-evaluation-harness-global-mmlu-full-ar)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-BN](#lm-evaluation-harness-global-mmlu-full-bn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-CS](#lm-evaluation-harness-global-mmlu-full-cs)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-DE](#lm-evaluation-harness-global-mmlu-full-de)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-EL](#lm-evaluation-harness-global-mmlu-full-el)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-EN](#lm-evaluation-harness-global-mmlu-full-en)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-ES](#lm-evaluation-harness-global-mmlu-full-es)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-FA](#lm-evaluation-harness-global-mmlu-full-fa)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-FIL](#lm-evaluation-harness-global-mmlu-full-fil)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-FR](#lm-evaluation-harness-global-mmlu-full-fr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-HA](#lm-evaluation-harness-global-mmlu-full-ha)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-HE](#lm-evaluation-harness-global-mmlu-full-he)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-HI](#lm-evaluation-harness-global-mmlu-full-hi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-ID](#lm-evaluation-harness-global-mmlu-full-id)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-IG](#lm-evaluation-harness-global-mmlu-full-ig)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-IT](#lm-evaluation-harness-global-mmlu-full-it)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-JA](#lm-evaluation-harness-global-mmlu-full-ja)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-KO](#lm-evaluation-harness-global-mmlu-full-ko)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-KY](#lm-evaluation-harness-global-mmlu-full-ky)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-LT](#lm-evaluation-harness-global-mmlu-full-lt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-MG](#lm-evaluation-harness-global-mmlu-full-mg)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-MS](#lm-evaluation-harness-global-mmlu-full-ms)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-NE](#lm-evaluation-harness-global-mmlu-full-ne)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-NL](#lm-evaluation-harness-global-mmlu-full-nl)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-NY](#lm-evaluation-harness-global-mmlu-full-ny)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-PL](#lm-evaluation-harness-global-mmlu-full-pl)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-PT](#lm-evaluation-harness-global-mmlu-full-pt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-RO](#lm-evaluation-harness-global-mmlu-full-ro)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-RU](#lm-evaluation-harness-global-mmlu-full-ru)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SI](#lm-evaluation-harness-global-mmlu-full-si)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SN](#lm-evaluation-harness-global-mmlu-full-sn)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SO](#lm-evaluation-harness-global-mmlu-full-so)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SR](#lm-evaluation-harness-global-mmlu-full-sr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SV](#lm-evaluation-harness-global-mmlu-full-sv)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-SW](#lm-evaluation-harness-global-mmlu-full-sw)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-TE](#lm-evaluation-harness-global-mmlu-full-te)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-TR](#lm-evaluation-harness-global-mmlu-full-tr)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-UK](#lm-evaluation-harness-global-mmlu-full-uk)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-VI](#lm-evaluation-harness-global-mmlu-full-vi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-YO](#lm-evaluation-harness-global-mmlu-full-yo)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-Full-ZH](#lm-evaluation-harness-global-mmlu-full-zh)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-HI](#lm-evaluation-harness-global-mmlu-hi)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-ID](#lm-evaluation-harness-global-mmlu-id)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-IT](#lm-evaluation-harness-global-mmlu-it)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-JA](#lm-evaluation-harness-global-mmlu-ja)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-KO](#lm-evaluation-harness-global-mmlu-ko)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-PT](#lm-evaluation-harness-global-mmlu-pt)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-SW](#lm-evaluation-harness-global-mmlu-sw)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-YO](#lm-evaluation-harness-global-mmlu-yo)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [Global-MMLU-ZH](#lm-evaluation-harness-global-mmlu-zh)
  - Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
* - [HellaSwag](#lm-evaluation-harness-hellaswag)
  - The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.
* - [HellaSwag Multilingual](#lm-evaluation-harness-hellaswag-multilingual)
  - The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.
* - [HumanEval-instruct](#lm-evaluation-harness-humaneval-instruct)
  - - The HumanEval benchmark measures functional correctness for synthesizing programs from docstrings. - Implementation taken from llama.
* - [IFEval](#lm-evaluation-harness-ifeval)
  - IFEval is a dataset designed to test a model's ability to follow explicit instructions, such as "include keyword x" or "use format y." The focus is on the model's adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.
* - [MBPP EvalPlus](#lm-evaluation-harness-mbpp-evalplus)
  - MBPP EvalPlus is an extension of the MBPP benchmark that explores the limits of the current generation of large language models for program synthesis in general purpose programming languages.
* - [MGSM](#lm-evaluation-harness-mgsm)
  - The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.
* - [MGSM-CoT](#lm-evaluation-harness-mgsm-cot)
  - The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.
* - [MMLU](#lm-evaluation-harness-mmlu)
  - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities.
* - [MMLU-Logits](#lm-evaluation-harness-mmlu-logits)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. - It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant uses the logits of the model to evaluate the accuracy.
* - [MMLU-Pro](#lm-evaluation-harness-mmlu-pro)
  - MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.
* - [MMLU-Pro-instruct](#lm-evaluation-harness-mmlu-pro-instruct)
  - - MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [MMLU-ProX](#lm-evaluation-harness-mmlu-prox)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation
* - [MMLU-ProX-French](#lm-evaluation-harness-mmlu-prox-french)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (French dataset)
* - [MMLU-ProX-German](#lm-evaluation-harness-mmlu-prox-german)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (German dataset)
* - [MMLU-ProX-Italian](#lm-evaluation-harness-mmlu-prox-italian)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Italian dataset)
* - [MMLU-ProX-Japanese](#lm-evaluation-harness-mmlu-prox-japanese)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Japanese dataset)
* - [MMLU-ProX-Spanish](#lm-evaluation-harness-mmlu-prox-spanish)
  - A Multilingual Benchmark for Advanced Large Language Model Evaluation (Spanish dataset)
* - [MMLU-Redux](#lm-evaluation-harness-mmlu-redux)
  - MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects.
* - [MMLU-Redux-instruct](#lm-evaluation-harness-mmlu-redux-instruct)
  - - MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects. - This variant applies a chat template and defaults to zero-shot evaluation.
* - [MMLU-instruct](#lm-evaluation-harness-mmlu-instruct)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to zero-shot evaluation and instructs the model to produce a single letter response.
* - [MMLU-tigerlab](#lm-evaluation-harness-mmlu-tigerlab)
  - - The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to chain-of-thought zero-shot evaluation.
* - [MuSR](#lm-evaluation-harness-musr)
  - The MuSR (Multistep Soft Reasoning) benchmark evaluates the reasoning capabilities of large language models through complex, multistep tasks specified in natural language narratives. It introduces sophisticated natural language and complex reasoning challenges to test the limits of chain-of-thought prompting.
* - [OpenBookQA](#lm-evaluation-harness-openbookqa)
  - - OpenBookQA is a question-answering dataset modeled after open book exams for assessing human understanding of a subject. - It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small "book" of 1,326 core science facts and the application of these facts to novel situations. - For training, the dataset includes a mapping from each question to the core science fact it was designed to probe. - Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. - The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.
* - [PIQA](#lm-evaluation-harness-piqa)
  - - Physical Interaction: Question Answering (PIQA) is a physical commonsense
  reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
  the physical knowledge of existing models. To what extent are current approaches
  actually learning about the world?
* - [Social IQA](#lm-evaluation-harness-social-iqa)
  - - Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear").
* - [Truthful QA](#lm-evaluation-harness-truthful-qa)
  - The TruthfulQA benchmark measures the truthfulness of language models in generating answers to questions. It consists of 817 questions across 38 categories, such as health, law, finance, and politics, designed to test whether models can avoid generating false answers that mimic common human misconceptions.
* - [WikiLingua](#lm-evaluation-harness-wikilingua)
  - The WikiLingua benchmark is a large-scale, multilingual dataset designed for evaluating cross-lingual abstractive summarization systems. It includes approximately 770,000 article-summary pairs in 18 languages, extracted from WikiHow, with gold-standard alignments created by matching images used to describe each how-to step in an article.
* - [indonesian_mmlu](#lm-evaluation-harness-indonesian-mmlu)
  - - The MMLU (Massive Multitask Language Understanding) benchmark translated to Indonesian. - This variant uses the Indonesian version of the MMLU tasks with string-based evaluation.
* - [winogrande](#lm-evaluation-harness-winogrande)
  - WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires commonsense reasoning.
```

(lm-evaluation-harness-adlr-agieval-en-cot)=
## ADLR AGIEval-EN-CoT

ADLR version of the AGIEval-EN-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: adlr_agieval_en_cot
  type: adlr_agieval_en_cot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-arc-challenge-llama)=
## ADLR ARC-Challenge-Llama

ARC-Challenge-Llama version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 25
    task: adlr_arc_challenge_llama
  type: adlr_arc_challenge_llama_25_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-commonsenseqa)=
## ADLR CommonsenseQA

CommonsenseQA version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 7
    task: commonsense_qa
  type: adlr_commonsense_qa_7_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-gpqa-diamond-cot)=
## ADLR GPQA-Diamond-CoT

ADLR version of the GPQA-Diamond-CoT benchmark used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: adlr_gpqa_diamond_cot_5_shot
  type: adlr_gpqa_diamond_cot_5_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-gsm8k-cot)=
## ADLR GSM8K-CoT

GSM8K-CoT version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 8
    task: adlr_gsm8k_fewshot_cot
  type: adlr_gsm8k_cot_8_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-global-mmlu)=
## ADLR Global-MMLU

Global-MMLU subset (8 languages - es, de, fr, zh, it, ja, pt, ko) used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: adlr_global_mmlu
  type: adlr_global_mmlu_lite_5_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-humaneval-greedy)=
## ADLR HumanEval Greedy

HumanEval Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: adlr_humaneval_greedy
  type: adlr_humaneval_greedy
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-humaneval-sampled)=
## ADLR HumanEval Sampled

HumanEval Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.6
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: adlr_humaneval_sampled
  type: adlr_humaneval_sampled
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-math-500-sampled)=
## ADLR MATH-500 Sampled

MATH-500 Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.7
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 4
    task: adlr_math_500_4_shot_sampled
  type: adlr_math_500_4_shot_sampled
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-mbpp-greedy)=
## ADLR MBPP Greedy

MBPP Greedy version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 3
    task: adlr_mbpp_sanitized_3_shot_greedy
  type: adlr_mbpp_sanitized_3_shot_greedy
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-mbpp-sampled)=
## ADLR MBPP Sampled

MBPP Sampled version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.6
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 3
    task: adlr_mbpp_sanitized_3shot_sampled
  type: adlr_mbpp_sanitized_3_shot_sampled
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-mgsm-cot)=
## ADLR MGSM-CoT

MGSM native CoT subset (6 languages - es, de, fr, zh, ja, ru) used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 8
    task: adlr_mgsm_native_cot_8_shot
  type: adlr_mgsm_native_cot_8_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-mmlu)=
## ADLR MMLU

MMLU version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
      args: --trust_remote_code
    task: mmlu_str
  type: adlr_mmlu
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-mmlu-pro)=
## ADLR MMLU-Pro

MMLU-Pro 5-shot base version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: adlr_mmlu_pro_5_shot_base
  type: adlr_mmlu_pro_5_shot_base
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-minerva-math)=
## ADLR Minerva-Math

Minerva-Math version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 4
    task: adlr_minerva_math_nemo
  type: adlr_minerva_math_nemo_4_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-race)=
## ADLR RACE

RACE version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: adlr_race
  type: adlr_race
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-truthfulqa-mc2)=
## ADLR TruthfulQA-MC2

TruthfulQA-MC2 version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: adlr_truthfulqa_mc2
  type: adlr_truthfulqa_mc2
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-adlr-winogrande)=
## ADLR Winogrande

Winogrande version used by NVIDIA Applied Deep Learning Research team (ADLR).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0
    top_p: 1.0
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: winogrande
  type: adlr_winogrande_5_shot
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-agieval)=
## AGIEval

AGIEval - A Human-Centric Benchmark for Evaluating Foundation Models

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: agieval
  type: agieval
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-arc-challenge)=
## ARC Challenge

The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: arc_challenge
  type: arc_challenge
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-arc-challenge-instruct)=
## ARC Challenge-instruct

- The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
    task: arc_challenge_chat
  type: arc_challenge_chat
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-arc-multilingual)=
## ARC Multilingual

The ARC dataset consists of 7,787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options). The questions are sorted into a Challenge Set of 2,590 "hard" questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: arc_multilingual
  type: arc_multilingual
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-bbq)=
## BBQ

The BBQ (Bias Benchmark for QA) is a benchmark designed to measure social biases in question answering systems. It contains ambiguous questions spanning 9 categories - disability, gender, nationality, physical appearance, race/ethnicity, religion, sexual orientation, socioeconomic status, and age.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: bbq_generate
  type: bbq
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-big-bench-hard)=
## BIG-Bench Hard

The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: leaderboard_bbh
  type: bbh
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-big-bench-hard-instruct)=
## BIG-Bench Hard-instruct

The BIG-Bench Hard (BBH) benchmark is a part of the BIG-Bench evaluation suite, focusing on 23 particularly difficult tasks that current language models struggle with. These tasks require complex, multi-step reasoning, and the benchmark evaluates models using few-shot learning and chain-of-thought prompting techniques.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: bbh_zeroshot
  type: bbh_instruct
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-commonsenseqa)=
## CommonsenseQA

- CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. - It contains 12,102 questions with one correct answer and four distractor answers.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 7
    task: commonsense_qa
  type: commonsense_qa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-frames-naive)=
## Frames Naive

Frames Naive uses the prompt as input without additional context

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.0
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: frames_naive
  type: frames_naive
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-frames-naive-with-links)=
## Frames Naive with Links

Frames Naive with Links provides the prompt and relevant Wikipedia article links

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.0
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: frames_naive_with_links
  type: frames_naive_with_links
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-frames-oracle)=
## Frames Oracle

Frames Oracle (long context) provides prompts and relevant text from curated and processed Wikipedia articles from "parasail-ai/frames-benchmark-wikipedia".

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.0
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: frames_oracle
    timeout: 1000
  type: frames_oracle
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gpqa)=
## GPQA

The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: leaderboard_gpqa
  type: gpqa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gpqa-diamond-cot)=
## GPQA-Diamond-CoT

The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. It is designed to be extremely difficult for both humans and AI, ensuring that questions cannot be easily answered using web searches.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: gpqa_diamond_cot_zeroshot
  type: gpqa_diamond_cot
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gsm8k)=
## GSM8K

The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: gsm8k
  type: gsm8k
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gsm8k-cot-llama)=
## GSM8K-cot-llama

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought evaluation - implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: gsm8k_cot_llama
  type: gsm8k_cot_llama
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gsm8k-cot-zeroshot)=
## GSM8K-cot-zeroshot

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: gsm8k_cot_zeroshot
  type: gsm8k_cot_zeroshot
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gsm8k-cot-zeroshot-llama)=
## GSM8K-cot-zeroshot-llama

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation - implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
    task: gsm8k_cot_llama
  type: gsm8k_cot_zeroshot_llama
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-gsm8k-instruct)=
## GSM8K-instruct

- The GSM8K benchmark evaluates the arithmetic reasoning of large language models using 1,319 grade school math word problems. - This variant defaults to chain-of-thought zero-shot evaluation with custom instructions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --add_instruction
    task: gsm8k_zeroshot_cot
  type: gsm8k_cot_instruct
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu)=
## Global-MMLU

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu
  type: global_mmlu
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-ar)=
## Global-MMLU-AR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_ar
  type: global_mmlu_ar
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-bn)=
## Global-MMLU-BN

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_bn
  type: global_mmlu_bn
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-de)=
## Global-MMLU-DE

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_de
  type: global_mmlu_de
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-en)=
## Global-MMLU-EN

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_en
  type: global_mmlu_en
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-es)=
## Global-MMLU-ES

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_es
  type: global_mmlu_es
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-fr)=
## Global-MMLU-FR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_fr
  type: global_mmlu_fr
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full)=
## Global-MMLU-Full

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full
  type: global_mmlu_full
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-am)=
## Global-MMLU-Full-AM

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_am
  type: global_mmlu_full_am
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ar)=
## Global-MMLU-Full-AR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ar
  type: global_mmlu_full_ar
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-bn)=
## Global-MMLU-Full-BN

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_bn
  type: global_mmlu_full_bn
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-cs)=
## Global-MMLU-Full-CS

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_cs
  type: global_mmlu_full_cs
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-de)=
## Global-MMLU-Full-DE

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_de
  type: global_mmlu_full_de
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-el)=
## Global-MMLU-Full-EL

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_el
  type: global_mmlu_full_el
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-en)=
## Global-MMLU-Full-EN

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_en
  type: global_mmlu_full_en
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-es)=
## Global-MMLU-Full-ES

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_es
  type: global_mmlu_full_es
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-fa)=
## Global-MMLU-Full-FA

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_fa
  type: global_mmlu_full_fa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-fil)=
## Global-MMLU-Full-FIL

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_fil
  type: global_mmlu_full_fil
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-fr)=
## Global-MMLU-Full-FR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_fr
  type: global_mmlu_full_fr
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ha)=
## Global-MMLU-Full-HA

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ha
  type: global_mmlu_full_ha
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-he)=
## Global-MMLU-Full-HE

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_he
  type: global_mmlu_full_he
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-hi)=
## Global-MMLU-Full-HI

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_hi
  type: global_mmlu_full_hi
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-id)=
## Global-MMLU-Full-ID

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_id
  type: global_mmlu_full_id
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ig)=
## Global-MMLU-Full-IG

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ig
  type: global_mmlu_full_ig
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-it)=
## Global-MMLU-Full-IT

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_it
  type: global_mmlu_full_it
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ja)=
## Global-MMLU-Full-JA

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ja
  type: global_mmlu_full_ja
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ko)=
## Global-MMLU-Full-KO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ko
  type: global_mmlu_full_ko
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ky)=
## Global-MMLU-Full-KY

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ky
  type: global_mmlu_full_ky
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-lt)=
## Global-MMLU-Full-LT

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_lt
  type: global_mmlu_full_lt
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-mg)=
## Global-MMLU-Full-MG

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_mg
  type: global_mmlu_full_mg
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ms)=
## Global-MMLU-Full-MS

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ms
  type: global_mmlu_full_ms
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ne)=
## Global-MMLU-Full-NE

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ne
  type: global_mmlu_full_ne
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-nl)=
## Global-MMLU-Full-NL

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_nl
  type: global_mmlu_full_nl
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ny)=
## Global-MMLU-Full-NY

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ny
  type: global_mmlu_full_ny
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-pl)=
## Global-MMLU-Full-PL

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_pl
  type: global_mmlu_full_pl
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-pt)=
## Global-MMLU-Full-PT

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_pt
  type: global_mmlu_full_pt
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ro)=
## Global-MMLU-Full-RO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ro
  type: global_mmlu_full_ro
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-ru)=
## Global-MMLU-Full-RU

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_ru
  type: global_mmlu_full_ru
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-si)=
## Global-MMLU-Full-SI

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_si
  type: global_mmlu_full_si
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-sn)=
## Global-MMLU-Full-SN

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_sn
  type: global_mmlu_full_sn
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-so)=
## Global-MMLU-Full-SO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_so
  type: global_mmlu_full_so
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-sr)=
## Global-MMLU-Full-SR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_sr
  type: global_mmlu_full_sr
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-sv)=
## Global-MMLU-Full-SV

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_sv
  type: global_mmlu_full_sv
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-sw)=
## Global-MMLU-Full-SW

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_sw
  type: global_mmlu_full_sw
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-te)=
## Global-MMLU-Full-TE

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_te
  type: global_mmlu_full_te
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-tr)=
## Global-MMLU-Full-TR

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_tr
  type: global_mmlu_full_tr
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-uk)=
## Global-MMLU-Full-UK

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_uk
  type: global_mmlu_full_uk
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-vi)=
## Global-MMLU-Full-VI

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_vi
  type: global_mmlu_full_vi
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-yo)=
## Global-MMLU-Full-YO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_yo
  type: global_mmlu_full_yo
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-full-zh)=
## Global-MMLU-Full-ZH

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_full_zh
  type: global_mmlu_full_zh
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-hi)=
## Global-MMLU-HI

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_hi
  type: global_mmlu_hi
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-id)=
## Global-MMLU-ID

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_id
  type: global_mmlu_id
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-it)=
## Global-MMLU-IT

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_it
  type: global_mmlu_it
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-ja)=
## Global-MMLU-JA

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_ja
  type: global_mmlu_ja
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-ko)=
## Global-MMLU-KO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_ko
  type: global_mmlu_ko
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-pt)=
## Global-MMLU-PT

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_pt
  type: global_mmlu_pt
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-sw)=
## Global-MMLU-SW

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_sw
  type: global_mmlu_sw
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-yo)=
## Global-MMLU-YO

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_yo
  type: global_mmlu_yo
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-global-mmlu-zh)=
## Global-MMLU-ZH

Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: global_mmlu_zh
  type: global_mmlu_zh
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-hellaswag)=
## HellaSwag

The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 10
    task: hellaswag
  type: hellaswag
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-hellaswag-multilingual)=
## HellaSwag Multilingual

The HellaSwag benchmark tests a language model's commonsense reasoning by having it choose the most logical ending for a given story.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 10
    task: hellaswag_multilingual
  type: hellaswag_multilingual
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-humaneval-instruct)=
## HumanEval-instruct

- The HumanEval benchmark measures functional correctness for synthesizing programs from docstrings. - Implementation taken from llama.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: humaneval_instruct
  type: humaneval_instruct
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-ifeval)=
## IFEval

IFEval is a dataset designed to test a model's ability to follow explicit instructions, such as "include keyword x" or "use format y." The focus is on the model's adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: ifeval
  type: ifeval
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mbpp-evalplus)=
## MBPP EvalPlus

MBPP EvalPlus is an extension of the MBPP benchmark that explores the limits of the current generation of large language models for program synthesis in general purpose programming languages.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mbpp_plus
  type: mbpp_plus
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mgsm)=
## MGSM

The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mgsm_direct
  type: mgsm
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mgsm-cot)=
## MGSM-CoT

The Multilingual Grade School Math (MGSM) benchmark evaluates the reasoning abilities of large language models in multilingual settings. It consists of 250 grade-school math problems from the GSM8K dataset, translated into ten diverse languages, and tests models using chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
    task: mgsm_cot_native
  type: mgsm_cot
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu)=
## MMLU

The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
      args: --trust_remote_code
    task: mmlu_str
  type: mmlu
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-logits)=
## MMLU-Logits

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. - It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant uses the logits of the model to evaluate the accuracy.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: mmlu
  type: mmlu_logits
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-pro)=
## MMLU-Pro

MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: mmlu_pro
  type: mmlu_pro
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-pro-instruct)=
## MMLU-Pro-instruct

- MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
    task: mmlu_pro
  type: mmlu_pro_instruct
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox)=
## MMLU-ProX

A Multilingual Benchmark for Advanced Large Language Model Evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox
  type: mmlu_prox
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox-french)=
## MMLU-ProX-French

A Multilingual Benchmark for Advanced Large Language Model Evaluation (French dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox_fr
  type: mmlu_prox_fr
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox-german)=
## MMLU-ProX-German

A Multilingual Benchmark for Advanced Large Language Model Evaluation (German dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox_de
  type: mmlu_prox_de
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox-italian)=
## MMLU-ProX-Italian

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Italian dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox_it
  type: mmlu_prox_it
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox-japanese)=
## MMLU-ProX-Japanese

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Japanese dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox_ja
  type: mmlu_prox_ja
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-prox-spanish)=
## MMLU-ProX-Spanish

A Multilingual Benchmark for Advanced Large Language Model Evaluation (Spanish dataset)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_prox_es
  type: mmlu_prox_es
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-redux)=
## MMLU-Redux

MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: mmlu_redux
  type: mmlu_redux
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-redux-instruct)=
## MMLU-Redux-instruct

- MMLU-Redux is a subset of 3,000 manually re-annotated questions across 30 MMLU subjects. - This variant applies a chat template and defaults to zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 8192
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --add_instruction
    task: mmlu_redux
  type: mmlu_redux_instruct
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-instruct)=
## MMLU-instruct

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to zero-shot evaluation and instructs the model to produce a single letter response.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --trust_remote_code --add_instruction
    task: mmlu_str
  type: mmlu_instruct
  supported_endpoint_types:
  - chat
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-mmlu-tigerlab)=
## MMLU-tigerlab

- The MMLU (Massive Multitask Language Understanding) benchmark is designed to measure the knowledge acquired during pretraining by evaluating models in zero-shot and few-shot settings. It covers 57 subjects across various fields, testing both world knowledge and problem-solving abilities. - This variant defaults to chain-of-thought zero-shot evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
    task: mmlu_cot_0_shot_chat
  type: mmlu_cot_0_shot_chat
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-musr)=
## MuSR

The MuSR (Multistep Soft Reasoning) benchmark evaluates the reasoning capabilities of large language models through complex, multistep tasks specified in natural language narratives. It introduces sophisticated natural language and complex reasoning challenges to test the limits of chain-of-thought prompting.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: leaderboard_musr
  type: musr
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-openbookqa)=
## OpenBookQA

- OpenBookQA is a question-answering dataset modeled after open book exams for assessing human understanding of a subject. - It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small "book" of 1,326 core science facts and the application of these facts to novel situations. - For training, the dataset includes a mapping from each question to the core science fact it was designed to probe. - Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. - The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: openbookqa
  type: openbookqa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-piqa)=
## PIQA

- Physical Interaction: Question Answering (PIQA) is a physical commonsense
  reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
  the physical knowledge of existing models. To what extent are current approaches
  actually learning about the world?

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: piqa
  type: piqa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-social-iqa)=
## Social IQA

- Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear").

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
    task: social_iqa
  type: social_iqa
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-truthful-qa)=
## Truthful QA

The TruthfulQA benchmark measures the truthfulness of language models in generating answers to questions. It consists of 817 questions across 38 categories, such as health, law, finance, and politics, designed to test whether models can avoid generating false answers that mimic common human misconceptions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
    task: truthfulqa
  type: truthfulqa
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-wikilingua)=
## WikiLingua

The WikiLingua benchmark is a large-scale, multilingual dataset designed for evaluating cross-lingual abstractive summarization systems. It includes approximately 770,000 article-summary pairs in 18 languages, extracted from WikiHow, with gold-standard alignments created by matching images used to describe each how-to step in an article.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      args: --trust_remote_code
    task: wikilingua
  type: wikilingua
  supported_endpoint_types:
  - chat
target:
  api_endpoint:
    stream: false

```

</details>


(lm-evaluation-harness-indonesian-mmlu)=
## indonesian_mmlu

- The MMLU (Massive Multitask Language Understanding) benchmark translated to Indonesian. - This variant uses the Indonesian version of the MMLU tasks with string-based evaluation.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `lm-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 0
      args: --trust_remote_code
    task: m_mmlu_id_str
  type: m_mmlu_id_str
  supported_endpoint_types:
  - chat
  - completions
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/lm-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:1612de5cfdd7aeb6a9a380d0fcf4e2504ddd7dcee6da9b23f1167fe541dbed64
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} lm-eval --tasks {{config.params.task}}{% if config.params.extra.num_fewshot is defined %} --num_fewshot {{ config.params.extra.num_fewshot }}{% endif %} --model {% if target.api_endpoint.type == "completions" %}local-completions{% elif target.api_endpoint.type == "chat" %}local-chat-completions{% endif %} --model_args "base_url={{target.api_endpoint.url}},model={{target.api_endpoint.model_id}},tokenized_requests={{config.params.extra.tokenized_requests}},{% if config.params.extra.tokenizer is not none %}tokenizer={{config.params.extra.tokenizer}}{% endif %},tokenizer_backend={{config.params.extra.tokenizer_backend}},num_concurrent={{config.params.parallelism}},timeout={{ config.params.request_timeout }},max_retries={{ config.params.max_retries }},stream={{ target.api_endpoint.stream }}" --log_samples --output_path {{config.output_dir}} --use_cache {{config.output_dir}}/lm_cache {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} {% if target.api_endpoint.type == "chat" %}--fewshot_as_multiturn --apply_chat_template {% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.temperature is not none or config.params.top_p is not none or config.params.max_new_tokens is not none %}--gen_kwargs="{% if config.params.temperature is not none %}temperature={{ config.params.temperature }}{% endif %}{% if config.params.top_p is not none %},top_p={{ config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %},max_gen_toks={{ config.params.max_new_tokens }}{% endif %}"{% endif %} {% if config.params.extra.downsampling_ratio is not none %}--downsampling_ratio {{ config.params.extra.downsampling_ratio }}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: null
    temperature: 1.0e-07
    top_p: 0.9999999
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      tokenizer: null
      tokenizer_backend: None
      downsampling_ratio: null
      tokenized_requests: false
      num_fewshot: 5
    task: winogrande
  type: winogrande
  supported_endpoint_types:
  - completions
target:
  api_endpoint:
    stream: false

```

</details>

