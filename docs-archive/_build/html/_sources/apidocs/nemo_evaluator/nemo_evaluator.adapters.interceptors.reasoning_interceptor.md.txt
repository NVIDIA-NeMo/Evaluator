# {py:mod}`nemo_evaluator.adapters.interceptors.reasoning_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.reasoning_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResponseReasoningInterceptor <nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} ResponseReasoningInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`, {py:obj}`nemo_evaluator.adapters.types.PostEvalHook`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} end_reasoning_token
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.end_reasoning_token
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.end_reasoning_token
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} start_reasoning_token
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.start_reasoning_token
:type: str | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.start_reasoning_token
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} add_reasoning
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.add_reasoning
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.add_reasoning
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} migrate_reasoning_content
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.migrate_reasoning_content
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.migrate_reasoning_content
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enable_reasoning_tracking
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.enable_reasoning_tracking
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.enable_reasoning_tracking
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} include_if_not_finished
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.include_if_not_finished
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.include_if_not_finished
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stats_file_saving_interval
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.stats_file_saving_interval
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.stats_file_saving_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enable_caching
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.enable_caching
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.enable_caching
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} cache_dir
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.cache_dir
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.cache_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} logging_aggregated_stats_interval
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.logging_aggregated_stats_interval
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.Params.logging_aggregated_stats_interval
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} end_reasoning_token
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.end_reasoning_token
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.end_reasoning_token
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} start_reasoning_token
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.start_reasoning_token
:type: str | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.start_reasoning_token
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} add_reasoning
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.add_reasoning
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.add_reasoning
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} migrate_reasoning_content
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.migrate_reasoning_content
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.migrate_reasoning_content
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enable_reasoning_tracking
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.enable_reasoning_tracking
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.enable_reasoning_tracking
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} include_if_not_finished
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.include_if_not_finished
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.include_if_not_finished
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enable_caching
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.enable_caching
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.enable_caching
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} cache_dir
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.cache_dir
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.cache_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _request_stats_cache
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._request_stats_cache
:type: typing.Optional[nemo_evaluator.adapters.caching.diskcaching.Cache]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._request_stats_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stats_file_saving_interval
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.stats_file_saving_interval
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.stats_file_saving_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} logging_aggregated_stats_interval
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.logging_aggregated_stats_interval
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.logging_aggregated_stats_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _load_and_aggregate_cached_stats() -> None
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._load_and_aggregate_cached_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._load_and_aggregate_cached_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _cache_reasoning_stats(request_id: str, stats: dict[str, any], context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._cache_reasoning_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._cache_reasoning_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _update_reasoning_stats(reasoning_info: dict) -> None
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._update_reasoning_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._update_reasoning_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _process_reasoning_message(msg: dict, usage: dict = None) -> tuple[dict, dict]
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._process_reasoning_message

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._process_reasoning_message
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _strip_reasoning(text: str) -> str
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._strip_reasoning

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._strip_reasoning
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _migrate_reasoning_content(msg: dict)
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._migrate_reasoning_content

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._migrate_reasoning_content
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.intercept_response

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_stats_to_file(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._save_stats_to_file

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor._save_stats_to_file
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} post_eval_hook(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.post_eval_hook

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.reasoning_interceptor.ResponseReasoningInterceptor.post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

``````
