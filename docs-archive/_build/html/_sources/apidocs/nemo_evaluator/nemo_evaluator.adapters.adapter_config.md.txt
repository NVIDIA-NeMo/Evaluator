# {py:mod}`nemo_evaluator.adapters.adapter_config`

```{py:module} nemo_evaluator.adapters.adapter_config
```

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DiscoveryConfig <nemo_evaluator.adapters.adapter_config.DiscoveryConfig>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.DiscoveryConfig
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`InterceptorConfig <nemo_evaluator.adapters.adapter_config.InterceptorConfig>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`PostEvalHookConfig <nemo_evaluator.adapters.adapter_config.PostEvalHookConfig>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`AdapterConfig <nemo_evaluator.adapters.adapter_config.AdapterConfig>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} DiscoveryConfig(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.adapter_config.DiscoveryConfig

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.DiscoveryConfig
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.DiscoveryConfig.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} modules
:canonical: nemo_evaluator.adapters.adapter_config.DiscoveryConfig.modules
:type: list[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.DiscoveryConfig.modules
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} dirs
:canonical: nemo_evaluator.adapters.adapter_config.DiscoveryConfig.dirs
:type: list[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.DiscoveryConfig.dirs
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} InterceptorConfig(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.adapter_config.InterceptorConfig

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} name
:canonical: nemo_evaluator.adapters.adapter_config.InterceptorConfig.name
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig.name
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enabled
:canonical: nemo_evaluator.adapters.adapter_config.InterceptorConfig.enabled
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig.enabled
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} config
:canonical: nemo_evaluator.adapters.adapter_config.InterceptorConfig.config
:type: dict[str, typing.Any]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.InterceptorConfig.config
:parser: docs.autodoc2_docstrings_parser
```

````

`````

``````{py:class} PostEvalHookConfig(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} name
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.name
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.name
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} enabled
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.enabled
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.enabled
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} config
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.config
:type: dict[str, typing.Any]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.config
:parser: docs.autodoc2_docstrings_parser
```

````

`````{py:class} Config
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.Config

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.Config
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} use_enum_values
:canonical: nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.Config.use_enum_values
:value: >
   True

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.PostEvalHookConfig.Config.use_enum_values
:parser: docs.autodoc2_docstrings_parser
```

````

`````

``````

`````{py:class} AdapterConfig(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} discovery
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.discovery
:type: nemo_evaluator.adapters.adapter_config.DiscoveryConfig
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.discovery
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} interceptors
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.interceptors
:type: list[nemo_evaluator.adapters.adapter_config.InterceptorConfig]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.interceptors
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} post_eval_hooks
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.post_eval_hooks
:type: list[nemo_evaluator.adapters.adapter_config.PostEvalHookConfig]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.post_eval_hooks
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} endpoint_type
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.endpoint_type
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.endpoint_type
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} caching_dir
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.caching_dir
:type: str | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.caching_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} generate_html_report
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.generate_html_report
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.generate_html_report
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} log_failed_requests
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.log_failed_requests
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.log_failed_requests
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} tracking_requests_stats
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.tracking_requests_stats
:type: bool
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.tracking_requests_stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} html_report_size
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.html_report_size
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.html_report_size
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_legacy_defaults() -> dict[str, typing.Any]
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.get_legacy_defaults
:classmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.get_legacy_defaults
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_validated_config(run_config: dict[str, typing.Any]) -> AdapterConfig | None
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.get_validated_config
:classmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.get_validated_config
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_default_output_dir(legacy_config: dict[str, typing.Any], run_config: dict[str, typing.Any] | None = None) -> str | None
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig._get_default_output_dir
:staticmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig._get_default_output_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_default_cache_dir(legacy_config: dict[str, typing.Any], run_config: dict[str, typing.Any] | None = None, subdir: str = 'cache') -> str
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig._get_default_cache_dir
:staticmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig._get_default_cache_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} from_legacy_config(legacy_config: dict[str, typing.Any], run_config: dict[str, typing.Any] | None = None) -> nemo_evaluator.adapters.adapter_config.AdapterConfig
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.from_legacy_config
:classmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.from_legacy_config
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_interceptor_configs() -> dict[str, dict[str, typing.Any]]
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.get_interceptor_configs

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.get_interceptor_configs
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_post_eval_hook_configs() -> dict[str, dict[str, typing.Any]]
:canonical: nemo_evaluator.adapters.adapter_config.AdapterConfig.get_post_eval_hook_configs

```{autodoc2-docstring} nemo_evaluator.adapters.adapter_config.AdapterConfig.get_post_eval_hook_configs
:parser: docs.autodoc2_docstrings_parser
```

````

`````
