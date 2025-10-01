# {py:mod}`nemo_evaluator.adapters.interceptors.response_stats_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.response_stats_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResponseStatsInterceptor <nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} ResponseStatsInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`, {py:obj}`nemo_evaluator.adapters.types.PostEvalHook`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} collect_token_stats
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_token_stats
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_token_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} collect_finish_reasons
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_finish_reasons
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_finish_reasons
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} collect_tool_calls
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_tool_calls
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.collect_tool_calls
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stats_file_saving_interval
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.stats_file_saving_interval
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.stats_file_saving_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} save_individuals
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.save_individuals
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.save_individuals
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} cache_dir
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.cache_dir
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.cache_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} logging_aggregated_stats_interval
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.logging_aggregated_stats_interval
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.Params.logging_aggregated_stats_interval
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:method} _load_aggregated_cached_stats() -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._load_aggregated_cached_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._load_aggregated_cached_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _calculate_inference_time(run_data: dict) -> float
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._calculate_inference_time

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._calculate_inference_time
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _update_basic_stats(resp: nemo_evaluator.adapters.types.AdapterResponse, current_time: float) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_basic_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_basic_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _update_running_stats(stat_name: str, value: float) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_running_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_running_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _update_time_tracking(current_time: float) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_time_tracking

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_time_tracking
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _update_response_stats(individual_stats: dict[str, any]) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_response_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._update_response_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _add_basic_response_stats(adapter_response, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._add_basic_response_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._add_basic_response_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _extract_detailed_response_stats(response_data: dict) -> dict
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._extract_detailed_response_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._extract_detailed_response_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _cache_request_stats(request_id: str, stats: dict[str, any]) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._cache_request_stats

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._cache_request_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.intercept_response

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_stats_to_file(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_stats_to_file

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_stats_to_file
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_run_ids_info() -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_run_ids_info

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_run_ids_info
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_aggregated_stats_to_cache() -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_aggregated_stats_to_cache

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_aggregated_stats_to_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _load_interceptor_state() -> dict
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._load_interceptor_state

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._load_interceptor_state
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_interceptor_state(state: dict) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_interceptor_state

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor._save_interceptor_state
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} post_eval_hook(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.post_eval_hook

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.response_stats_interceptor.ResponseStatsInterceptor.post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

``````
