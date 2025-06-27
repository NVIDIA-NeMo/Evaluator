# Adding on-demand evaluation packages

To run evaluations, do the following:

1. Deploy your model:

```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:linenos:
```

Run the deployment in the background:
```bash
python deploy.py &
```

2. Export the required environment variables. 

3. Run the evalution of your choice.

---

### Evaluate `bfcl`

First, install the evaluation package:
```{literalinclude} ../scripts/snippets/bfcl.py
:language: bash
:lines: 1
:linenos:
```

Export the required variables:
```{literalinclude} ../scripts/snippets/bfcl.py
:language: bash
:start-after: "## Export the required variables"
:end-before: "## Run the evaluation"
:linenos:
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/bfcl.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

---

### Evaluate `garak`

First, install the evaluation package:
```{literalinclude} ../scripts/snippets/garak.py
:language: bash
:lines: 1
:linenos:
```

Export the required variables:
```{literalinclude} ../scripts/snippets/garak.py
:language: bash
:start-after: "## Export the required variables"
:end-before: "## Run the evaluation"
:linenos:
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/garak.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

---

### Evaluate `bigcode`

First, install the evaluation package:
```{literalinclude} ../scripts/snippets/bigcode.py
:language: bash
:lines: 1
:linenos:
```

Export the required variables:
```{literalinclude} ../scripts/snippets/bigcode.py
:language: bash
:start-after: "## Export the required variables"
:end-before: "## Run the evaluation"
:linenos:
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/bigcode.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

---

### Evaluate `simple-evals`

First, install the evaluation package:
```{literalinclude} ../scripts/snippets/simple_evals.py
:language: bash
:lines: 1
:linenos:
```

Export the required variables:
```{literalinclude} ../scripts/snippets/simple_evals.py
:language: bash
:start-after: "## Export the required variables"
:end-before: "## Run the evaluation"
:linenos:
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/simple_evals.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

---

### Evaluate `safety`

First, install the evaluation package:
```{literalinclude} ../scripts/snippets/safety.py
:language: bash
:lines: 1
:linenos:
```

Export the required variables:
```{literalinclude} ../scripts/snippets/safety.py
:language: bash
:start-after: "## Export the required variables"
:end-before: "## Run the evaluation"
:linenos:
```

Run the evaluation:
```{literalinclude} ../scripts/snippets/safety.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

---
