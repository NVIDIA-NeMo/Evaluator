# {py:mod}`nemo_evaluator.core.evaluate`

```{py:module} nemo_evaluator.core.evaluate
```

```{autodoc2-docstring} nemo_evaluator.core.evaluate
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_output <nemo_evaluator.core.evaluate.parse_output>`
  - ```{autodoc2-docstring} nemo_evaluator.core.evaluate.parse_output
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`evaluate <nemo_evaluator.core.evaluate.evaluate>`
  - ```{autodoc2-docstring} nemo_evaluator.core.evaluate.evaluate
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.core.evaluate.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.core.evaluate.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_evaluator.core.evaluate.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.core.evaluate.logger
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} parse_output(evaluation: nemo_evaluator.api.api_dataclasses.Evaluation) -> nemo_evaluator.api.api_dataclasses.EvaluationResult
:canonical: nemo_evaluator.core.evaluate.parse_output

```{autodoc2-docstring} nemo_evaluator.core.evaluate.parse_output
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} evaluate(eval_cfg: nemo_evaluator.api.api_dataclasses.EvaluationConfig, target_cfg: nemo_evaluator.api.api_dataclasses.EvaluationTarget) -> nemo_evaluator.api.api_dataclasses.EvaluationResult
:canonical: nemo_evaluator.core.evaluate.evaluate

```{autodoc2-docstring} nemo_evaluator.core.evaluate.evaluate
:parser: docs.autodoc2_docstrings_parser
```
````
