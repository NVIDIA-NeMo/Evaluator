# Run the evaluation that uses log-probabilities

- [Run the evaluation that uses log-probabilities](#run-the-evaluation-that-uses-log-probabilities)
  - [Introduction](#introduction)
  - [Evaluate NeMo 2 checkpoint with arc_challenge](#evaluate-nemo-2-checkpoint-with-arc_challenge)


## Introduction

The most typical approach to LLM evaluation is by providing the model with a question or instruction and assesing the quality of its response.
However, an alternative approach is to use the perplexity of the input provided to the model.

Perplexity quantifies the "surprise" or "uncertainty" an LLM exhibits when processing a given text sequence.
It is derived directly from the log-probabilities assigned by the LLM to each token in the input.

In this approach to evaluation, the LLM is provided with both question and answer combined into a single text.
Then the perplexity of the tokens belonging to the answer is calculated.
This allows to asses how likely it is that an LLM would return the given answer for the given question.
It is often used with multiple-choice questions, and the answer with the lowest perplexity is treated as the one selected by the model.

Such approach is especially useful when evaluating base (pre-trained) models as it elliminates the need for instruction following and doesn't require the model to adhere to a specified output format.

## Evaluate NeMo 2 checkpoint with arc_challenge

In this example we will use the `arc_challenge` task from `nvidia-lm-eval`.
The NeMo Framework docker image comes with `nvidia-lm-eval` package pre-installed.
If you are running this example in a different environment, install the evaluation package:
```bash
pip install nvidia-lm-eval==25.6
```

First, you need to deploy your model:
```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:linenos:
```

```bash
python deploy.py &
```
The server will return log-probabilities of the tokens if it recieves `logprob=<int>` parameters as a part of the request.
When paired with `echo=true`, the model will return the input as a part of its response, together with its associated log-probabilities. 

This happens under the hood when we launch the evaluation on `arc_challenge`.
You can do it with the following code:

```{literalinclude} ../scripts/snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Note that you must provide a path to the tokenizer:
```
        "extra": {
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
        },
```

This is necessary for tokenizing the input on the client side and selecting only the probabilities associated with the input part correspondig to the answer.

This example uses only 10 samples.
To run the evaluation on the whole dataset, remove the `"limit_samples"` paramter.
