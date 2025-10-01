# {py:mod}`nemo_evaluator.adapters.types`

```{py:module} nemo_evaluator.adapters.types
```

```{autodoc2-docstring} nemo_evaluator.adapters.types
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdapterGlobalContext <nemo_evaluator.adapters.types.AdapterGlobalContext>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterGlobalContext
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`AdapterRequestContext <nemo_evaluator.adapters.types.AdapterRequestContext>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequestContext
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`AdapterRequest <nemo_evaluator.adapters.types.AdapterRequest>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequest
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`AdapterResponse <nemo_evaluator.adapters.types.AdapterResponse>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterResponse
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`RequestInterceptor <nemo_evaluator.adapters.types.RequestInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`RequestToResponseInterceptor <nemo_evaluator.adapters.types.RequestToResponseInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestToResponseInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`ResponseInterceptor <nemo_evaluator.adapters.types.ResponseInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.ResponseInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`PostEvalHook <nemo_evaluator.adapters.types.PostEvalHook>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.types.PostEvalHook
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} AdapterGlobalContext
:canonical: nemo_evaluator.adapters.types.AdapterGlobalContext

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterGlobalContext
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} output_dir
:canonical: nemo_evaluator.adapters.types.AdapterGlobalContext.output_dir
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterGlobalContext.output_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} url
:canonical: nemo_evaluator.adapters.types.AdapterGlobalContext.url
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterGlobalContext.url
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} metrics_path
:canonical: nemo_evaluator.adapters.types.AdapterGlobalContext.metrics_path
:type: pathlib.Path

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterGlobalContext.metrics_path
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} AdapterRequestContext
:canonical: nemo_evaluator.adapters.types.AdapterRequestContext

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequestContext
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} cache_hit
:canonical: nemo_evaluator.adapters.types.AdapterRequestContext.cache_hit
:type: bool
:value: >
   False

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequestContext.cache_hit
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} cache_key
:canonical: nemo_evaluator.adapters.types.AdapterRequestContext.cache_key
:type: str | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequestContext.cache_key
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} request_id
:canonical: nemo_evaluator.adapters.types.AdapterRequestContext.request_id
:type: str | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequestContext.request_id
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} AdapterRequest
:canonical: nemo_evaluator.adapters.types.AdapterRequest

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequest
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} r
:canonical: nemo_evaluator.adapters.types.AdapterRequest.r
:type: flask.Request
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequest.r
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} rctx
:canonical: nemo_evaluator.adapters.types.AdapterRequest.rctx
:type: nemo_evaluator.adapters.types.AdapterRequestContext
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterRequest.rctx
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} AdapterResponse
:canonical: nemo_evaluator.adapters.types.AdapterResponse

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterResponse
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} r
:canonical: nemo_evaluator.adapters.types.AdapterResponse.r
:type: requests.Response
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterResponse.r
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} rctx
:canonical: nemo_evaluator.adapters.types.AdapterResponse.rctx
:type: nemo_evaluator.adapters.types.AdapterRequestContext
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterResponse.rctx
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} latency_ms
:canonical: nemo_evaluator.adapters.types.AdapterResponse.latency_ms
:type: float | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.types.AdapterResponse.latency_ms
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:exception} FatalErrorException()
:canonical: nemo_evaluator.adapters.types.FatalErrorException

Bases: {py:obj}`Exception`

```{autodoc2-docstring} nemo_evaluator.adapters.types.FatalErrorException
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.types.FatalErrorException.__init__
:parser: docs.autodoc2_docstrings_parser
```

````

`````{py:class} RequestInterceptor
:canonical: nemo_evaluator.adapters.types.RequestInterceptor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestInterceptor
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} intercept_request(req: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest
:canonical: nemo_evaluator.adapters.types.RequestInterceptor.intercept_request
:abstractmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestInterceptor.intercept_request
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} RequestToResponseInterceptor
:canonical: nemo_evaluator.adapters.types.RequestToResponseInterceptor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestToResponseInterceptor
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} intercept_request(req: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest | nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.types.RequestToResponseInterceptor.intercept_request
:abstractmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.types.RequestToResponseInterceptor.intercept_request
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} ResponseInterceptor
:canonical: nemo_evaluator.adapters.types.ResponseInterceptor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_evaluator.adapters.types.ResponseInterceptor
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.types.ResponseInterceptor.intercept_response
:abstractmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.types.ResponseInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} PostEvalHook
:canonical: nemo_evaluator.adapters.types.PostEvalHook

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_evaluator.adapters.types.PostEvalHook
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} post_eval_hook(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.types.PostEvalHook.post_eval_hook
:abstractmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.types.PostEvalHook.post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

`````
