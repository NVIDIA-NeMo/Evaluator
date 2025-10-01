# {py:mod}`nemo_evaluator.adapters.interceptors.raise_client_error_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RaiseClientErrorInterceptor <nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} RaiseClientErrorInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} exclude_status_codes
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.exclude_status_codes
:type: typing.Optional[typing.List[int]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.exclude_status_codes
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_codes
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_codes
:type: typing.Optional[typing.List[int]] | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_codes
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_code_range_start
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_code_range_start
:type: typing.Optional[int] | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_code_range_start
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_code_range_end
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_code_range_end
:type: typing.Optional[int] | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.Params.status_code_range_end
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} exclude_status_codes
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.exclude_status_codes
:type: typing.List[int] | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.exclude_status_codes
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_codes
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_codes
:type: typing.List[int] | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_codes
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_code_range_start
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_code_range_start
:type: int | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_code_range_start
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} status_code_range_end
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_code_range_end
:type: int | None
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.status_code_range_end
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _format_exception(response: requests.Response, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.FatalErrorException
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor._format_exception

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor._format_exception
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _handle_client_error(response: requests.Response, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> requests.Response
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor._handle_client_error

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor._handle_client_error
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(resp: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.intercept_response

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.raise_client_error_interceptor.RaiseClientErrorInterceptor.intercept_response
:parser: docs.autodoc2_docstrings_parser
```

````

``````
