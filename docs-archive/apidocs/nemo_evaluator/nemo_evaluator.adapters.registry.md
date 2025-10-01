# {py:mod}`nemo_evaluator.adapters.registry`

```{py:module} nemo_evaluator.adapters.registry
```

```{autodoc2-docstring} nemo_evaluator.adapters.registry
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InterceptorMetadata <nemo_evaluator.adapters.registry.InterceptorMetadata>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`InterceptorRegistry <nemo_evaluator.adapters.registry.InterceptorRegistry>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.adapters.registry.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.registry.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_evaluator.adapters.registry.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.registry.logger
:parser: docs.autodoc2_docstrings_parser
```

````

`````{py:class} InterceptorMetadata
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} name
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.name
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} description
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.description
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.description
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} interceptor_class
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.interceptor_class
:type: typing.Type[nemo_evaluator.adapters.types.RequestInterceptor | nemo_evaluator.adapters.types.ResponseInterceptor | nemo_evaluator.adapters.types.RequestToResponseInterceptor | nemo_evaluator.adapters.types.PostEvalHook]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.interceptor_class
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} init_schema
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.init_schema
:type: typing.Optional[typing.Type[pydantic.BaseModel]]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.init_schema
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} supports_request_interception() -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.supports_request_interception

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.supports_request_interception
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} supports_response_interception() -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.supports_response_interception

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.supports_response_interception
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} supports_request_to_response_interception() -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.supports_request_to_response_interception

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.supports_request_to_response_interception
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} supports_post_eval_hook() -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorMetadata.supports_post_eval_hook

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorMetadata.supports_post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} InterceptorRegistry()
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} _instance
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry._instance
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry._instance
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __new__()
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.__new__

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.__new__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_instance() -> nemo_evaluator.adapters.registry.InterceptorRegistry
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.get_instance
:classmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.get_instance
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} register(name: str, interceptor_class: typing.Type[nemo_evaluator.adapters.types.RequestInterceptor | nemo_evaluator.adapters.types.ResponseInterceptor | nemo_evaluator.adapters.types.RequestToResponseInterceptor | nemo_evaluator.adapters.types.PostEvalHook], metadata: nemo_evaluator.adapters.registry.InterceptorMetadata) -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.register

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.register
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_or_create_instance(name: str, config: dict[str, typing.Any]) -> nemo_evaluator.adapters.types.RequestInterceptor | nemo_evaluator.adapters.types.ResponseInterceptor | nemo_evaluator.adapters.types.RequestToResponseInterceptor | nemo_evaluator.adapters.types.PostEvalHook
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry._get_or_create_instance

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry._get_or_create_instance
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} discover_components(modules: typing.Optional[list[str]] = None, dirs: typing.Optional[list[str]] = None) -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.discover_components

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.discover_components
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _discover_from_modules(modules: typing.Optional[list[str]]) -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry._discover_from_modules

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry._discover_from_modules
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _discover_from_directories(dirs: typing.Optional[list[str]]) -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry._discover_from_directories

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry._discover_from_directories
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _should_process_file(py_file: pathlib.Path) -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry._should_process_file

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry._should_process_file
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_all_components() -> dict[str, nemo_evaluator.adapters.registry.InterceptorMetadata]
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.get_all_components

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.get_all_components
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_metadata(name: str) -> typing.Optional[nemo_evaluator.adapters.registry.InterceptorMetadata]
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.get_metadata

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.get_metadata
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_post_eval_hooks() -> dict[str, nemo_evaluator.adapters.registry.InterceptorMetadata]
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.get_post_eval_hooks

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.get_post_eval_hooks
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_interceptors() -> dict[str, nemo_evaluator.adapters.registry.InterceptorMetadata]
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.get_interceptors

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.get_interceptors
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} is_request_interceptor(name: str) -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.is_request_interceptor

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.is_request_interceptor
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} is_response_interceptor(name: str) -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.is_response_interceptor

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.is_response_interceptor
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} is_request_to_response_interceptor(name: str) -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.is_request_to_response_interceptor

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.is_request_to_response_interceptor
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} is_post_eval_hook(name: str) -> bool
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.is_post_eval_hook

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.is_post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} clear_cache() -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.clear_cache

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.clear_cache
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} reset() -> None
:canonical: nemo_evaluator.adapters.registry.InterceptorRegistry.reset

```{autodoc2-docstring} nemo_evaluator.adapters.registry.InterceptorRegistry.reset
:parser: docs.autodoc2_docstrings_parser
```

````

`````
