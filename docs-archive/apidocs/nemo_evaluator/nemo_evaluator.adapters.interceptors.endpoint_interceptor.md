# {py:mod}`nemo_evaluator.adapters.interceptors.endpoint_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.endpoint_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EndpointInterceptor <nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} EndpointInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.RequestToResponseInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_request(ar: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.intercept_request

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.endpoint_interceptor.EndpointInterceptor.intercept_request
:parser: docs.autodoc2_docstrings_parser
```

````

`````
