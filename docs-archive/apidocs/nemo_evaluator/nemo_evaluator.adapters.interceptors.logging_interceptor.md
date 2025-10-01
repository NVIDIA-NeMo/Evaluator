# {py:mod}`nemo_evaluator.adapters.interceptors.logging_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.logging_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RequestLoggingInterceptor <nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`ResponseLoggingInterceptor <nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_safe_headers <nemo_evaluator.adapters.interceptors.logging_interceptor._get_safe_headers>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor._get_safe_headers
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:function} _get_safe_headers(headers: dict[str, str]) -> dict[str, str]
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor._get_safe_headers

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor._get_safe_headers
:parser: docs.autodoc2_docstrings_parser
```
````

``````{py:class} RequestLoggingInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.RequestInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} log_request_body
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.log_request_body
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.log_request_body
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} log_request_headers
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.log_request_headers
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.log_request_headers
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_requests
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.max_requests
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.Params.max_requests
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} log_request_body
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.log_request_body
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.log_request_body
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} log_request_headers
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.log_request_headers
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.log_request_headers
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_requests
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.max_requests
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.max_requests
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _request_count
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor._request_count
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor._request_count
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_request(ar: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.intercept_request

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.RequestLoggingInterceptor.intercept_request
:parser: docs.autodoc2_docstrings_parser
```

````

``````

``````{py:class} ResponseLoggingInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} log_response_body
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.log_response_body
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.log_response_body
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} log_response_headers
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.log_response_headers
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.log_response_headers
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_responses
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.max_responses
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.Params.max_responses
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} log_response_body
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.log_response_body
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.log_response_body
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} log_response_headers
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.log_response_headers
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.log_response_headers
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_responses
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.max_responses
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.max_responses
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _response_count
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor._response_count
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor._response_count
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.intercept_response

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.logging_interceptor.ResponseLoggingInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

``````
