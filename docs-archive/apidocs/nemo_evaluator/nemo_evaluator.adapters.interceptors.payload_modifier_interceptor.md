# {py:mod}`nemo_evaluator.adapters.interceptors.payload_modifier_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PayloadParamsModifierInterceptor <nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} PayloadParamsModifierInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.RequestInterceptor`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} params_to_remove
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_remove
:type: typing.Optional[typing.List[str]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_remove
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} params_to_add
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_add
:type: typing.Optional[typing.Dict[str, typing.Any]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_add
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} params_to_rename
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_rename
:type: typing.Optional[typing.Dict[str, str]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.Params.params_to_rename
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} _params_to_remove
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_remove
:type: typing.List[str]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_remove
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _params_to_add
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_add
:type: typing.Dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_add
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _params_to_rename
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_rename
:type: typing.Dict[str, str]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor._params_to_rename
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_request(ar: nemo_evaluator.adapters.types.AdapterRequest, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterRequest | nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.payload_modifier_interceptor.PayloadParamsModifierInterceptor.intercept_request

````

``````
