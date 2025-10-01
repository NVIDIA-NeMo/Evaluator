# {py:mod}`nemo_evaluator.adapters.interceptors.caching_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.caching_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CachingInterceptor <nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} CachingInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.RequestToResponseInterceptor`, {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} cache_dir
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.cache_dir
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.cache_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} reuse_cached_responses
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.reuse_cached_responses
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.reuse_cached_responses
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} save_requests
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.save_requests
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.save_requests
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} save_responses
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.save_responses
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.save_responses
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_saved_requests
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.max_saved_requests
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.max_saved_requests
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_saved_responses
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.max_saved_responses
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.Params.max_saved_responses
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} responses_cache
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.responses_cache
:type: nemo_evaluator.adapters.caching.diskcaching.Cache
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.responses_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} requests_cache
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.requests_cache
:type: nemo_evaluator.adapters.caching.diskcaching.Cache
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.requests_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} headers_cache
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.headers_cache
:type: nemo_evaluator.adapters.caching.diskcaching.Cache
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.headers_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _generate_cache_key(data: typing.Any) -> str
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._generate_cache_key
:staticmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._generate_cache_key
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_from_cache(cache_key: str) -> tuple[typing.Any, typing.Any] | None
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._get_from_cache

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._get_from_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _save_to_cache(cache_key: str, content: typing.Any, headers: typing.Any) -> None
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._save_to_cache

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor._save_to_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_request(req: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest | nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.intercept_request

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.intercept_request
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.intercept_response

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.caching_interceptor.CachingInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

``````
