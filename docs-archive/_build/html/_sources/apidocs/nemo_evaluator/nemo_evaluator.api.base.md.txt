# {py:mod}`nemo_evaluator.api.base`

```{py:module} nemo_evaluator.api.base
```

```{autodoc2-docstring} nemo_evaluator.api.base
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalFrameworkBase <nemo_evaluator.api.base.EvalFrameworkBase>`
  - ```{autodoc2-docstring} nemo_evaluator.api.base.EvalFrameworkBase
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} EvalFrameworkBase
:canonical: nemo_evaluator.api.base.EvalFrameworkBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_evaluator.api.base.EvalFrameworkBase
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} parse_output(output_dir: str) -> nemo_evaluator.api.api_dataclasses.EvaluationResult
:canonical: nemo_evaluator.api.base.EvalFrameworkBase.parse_output
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} nemo_evaluator.api.base.EvalFrameworkBase.parse_output
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} framework_def() -> pathlib.Path
:canonical: nemo_evaluator.api.base.EvalFrameworkBase.framework_def
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} nemo_evaluator.api.base.EvalFrameworkBase.framework_def
:parser: docs.autodoc2_docstrings_parser
```

````

`````
