# Run the evaluation that uses log-probabilities

1. Deploy your model:

```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:linenos:
```

Run the deployment in the background:
```bash
python deploy.py &
```

2. Run the evaluation

The NeMo Framework docker image comes with `nvidia-lm-eval` pre-installed.
If you are running this example in a different environment, install the evaluation package:
```bash
pip install nvidia-lm-eval==25.6
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```
