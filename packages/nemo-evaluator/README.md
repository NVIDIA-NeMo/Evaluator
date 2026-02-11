# NeMo Evaluator

For complete documentation, please see: [docs/nemo-evaluator/index.md](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/docs/nemo-evaluator/index.md)

## Custom Benchmarks with BYOB

Create custom evaluation benchmarks in ~12 lines of Python using the BYOB (Bring Your Own Benchmark) framework:

```python
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="my-qa", dataset="data.jsonl", prompt="Q: {question}\nA:", target_field="answer")
@scorer
def check(response: str, target: str, metadata: dict) -> dict:
    return {"correct": target.lower() in response.lower()}
```

```bash
# Compile and run
nemo-evaluator-byob my_benchmark.py
nemo-evaluator run_eval --eval_type byob_my_qa.my-qa --model_url http://localhost:8000 --model_id my-model
```

See the [BYOB quickstart guide](examples/byob/README.md) for full documentation, built-in scorers, native mode, and examples.
