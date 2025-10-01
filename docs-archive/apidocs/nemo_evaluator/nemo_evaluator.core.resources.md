# {py:mod}`nemo_evaluator.core.resources`

```{py:module} nemo_evaluator.core.resources
```

```{autodoc2-docstring} nemo_evaluator.core.resources
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_token_usage_from_cache_db <nemo_evaluator.core.resources.get_token_usage_from_cache_db>`
  - ```{autodoc2-docstring} nemo_evaluator.core.resources.get_token_usage_from_cache_db
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_token_usage_from_cache <nemo_evaluator.core.resources.get_token_usage_from_cache>`
  - ```{autodoc2-docstring} nemo_evaluator.core.resources.get_token_usage_from_cache
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`aggregate_runtime_metrics <nemo_evaluator.core.resources.aggregate_runtime_metrics>`
  - ```{autodoc2-docstring} nemo_evaluator.core.resources.aggregate_runtime_metrics
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_update_persistent_metrics <nemo_evaluator.core.resources._update_persistent_metrics>`
  - ```{autodoc2-docstring} nemo_evaluator.core.resources._update_persistent_metrics
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`monitor_memory_usage <nemo_evaluator.core.resources.monitor_memory_usage>`
  - ```{autodoc2-docstring} nemo_evaluator.core.resources.monitor_memory_usage
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:function} get_token_usage_from_cache_db(cache_db_path: str | pathlib.Path) -> dict
:canonical: nemo_evaluator.core.resources.get_token_usage_from_cache_db

```{autodoc2-docstring} nemo_evaluator.core.resources.get_token_usage_from_cache_db
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_token_usage_from_cache(cache_dir: str) -> dict
:canonical: nemo_evaluator.core.resources.get_token_usage_from_cache

```{autodoc2-docstring} nemo_evaluator.core.resources.get_token_usage_from_cache
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} aggregate_runtime_metrics(output_dir: str) -> dict[str, typing.Any]
:canonical: nemo_evaluator.core.resources.aggregate_runtime_metrics

```{autodoc2-docstring} nemo_evaluator.core.resources.aggregate_runtime_metrics
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} _update_persistent_metrics(output_dir: str, start_time: float, peak_memory: int, peak_tree_memory: int, run_id: str) -> None
:canonical: nemo_evaluator.core.resources._update_persistent_metrics

```{autodoc2-docstring} nemo_evaluator.core.resources._update_persistent_metrics
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} monitor_memory_usage(func, *args, interval_ms, cache_dir: str | None = None, output_dir: str | None = None, **kwargs) -> tuple[nemo_evaluator.api.api_dataclasses.EvaluationResult, dict[str, typing.Any]]
:canonical: nemo_evaluator.core.resources.monitor_memory_usage

```{autodoc2-docstring} nemo_evaluator.core.resources.monitor_memory_usage
:parser: docs.autodoc2_docstrings_parser
```
````
