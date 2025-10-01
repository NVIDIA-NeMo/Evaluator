# {py:mod}`nemo_evaluator.logging.context`

```{py:module} nemo_evaluator.logging.context
```

```{autodoc2-docstring} nemo_evaluator.logging.context
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`bind_request_id <nemo_evaluator.logging.context.bind_request_id>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.context.bind_request_id
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`request_context <nemo_evaluator.logging.context.request_context>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.context.request_context
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_current_request_id <nemo_evaluator.logging.context.get_current_request_id>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.context.get_current_request_id
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_bound_logger <nemo_evaluator.logging.context.get_bound_logger>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.context.get_bound_logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:function} bind_request_id(request_id: typing.Optional[str] = None) -> str
:canonical: nemo_evaluator.logging.context.bind_request_id

```{autodoc2-docstring} nemo_evaluator.logging.context.bind_request_id
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} request_context(request_id: typing.Optional[str] = None)
:canonical: nemo_evaluator.logging.context.request_context

```{autodoc2-docstring} nemo_evaluator.logging.context.request_context
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_current_request_id() -> typing.Optional[str]
:canonical: nemo_evaluator.logging.context.get_current_request_id

```{autodoc2-docstring} nemo_evaluator.logging.context.get_current_request_id
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_bound_logger(request_id: typing.Optional[str] = None, logger_name: str = None)
:canonical: nemo_evaluator.logging.context.get_bound_logger

```{autodoc2-docstring} nemo_evaluator.logging.context.get_bound_logger
:parser: docs.autodoc2_docstrings_parser
```
````
