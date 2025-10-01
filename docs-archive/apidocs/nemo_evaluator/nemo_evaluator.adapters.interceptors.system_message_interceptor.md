# {py:mod}`nemo_evaluator.adapters.interceptors.system_message_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.system_message_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SystemMessageInterceptor <nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} SystemMessageInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.RequestInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} system_message
:canonical: nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.Params.system_message
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.Params.system_message
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} system_message
:canonical: nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.system_message
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.system_message
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_request(ar: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest
:canonical: nemo_evaluator.adapters.interceptors.system_message_interceptor.SystemMessageInterceptor.intercept_request

````

``````
