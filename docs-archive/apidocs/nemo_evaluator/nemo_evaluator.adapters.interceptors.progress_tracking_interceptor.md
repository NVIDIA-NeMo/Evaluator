# {py:mod}`nemo_evaluator.adapters.interceptors.progress_tracking_interceptor`

```{py:module} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProgressTrackingInterceptor <nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

``````{py:class} ProgressTrackingInterceptor(params: Params)
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor

Bases: {py:obj}`nemo_evaluator.adapters.types.ResponseInterceptor`, {py:obj}`nemo_evaluator.adapters.types.PostEvalHook`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.__init__
:parser: docs.autodoc2_docstrings_parser
```

`````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params

Bases: {py:obj}`nemo_evaluator.logging.BaseLoggingParams`

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} progress_tracking_url
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.progress_tracking_url
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.progress_tracking_url
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} progress_tracking_interval
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.progress_tracking_interval
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.progress_tracking_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} request_method
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.request_method
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.request_method
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} output_dir
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.output_dir
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.Params.output_dir
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:attribute} progress_tracking_url
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_tracking_url
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_tracking_url
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} progress_tracking_interval
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_tracking_interval
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_tracking_interval
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} progress_filepath
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_filepath
:type: typing.Optional[pathlib.Path]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.progress_filepath
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} request_method
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.request_method
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.request_method
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _initialize_samples_processed() -> int
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._initialize_samples_processed

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._initialize_samples_processed
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _write_progress(num_samples: int)
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._write_progress

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._write_progress
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _send_progress(num_samples: int) -> requests.Response
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._send_progress

```{autodoc2-docstring} nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor._send_progress
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} intercept_response(ar: nemo_evaluator.adapters.types.AdapterResponse, context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> nemo_evaluator.adapters.types.AdapterResponse
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.intercept_response

````

````{py:method} post_eval_hook(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.interceptors.progress_tracking_interceptor.ProgressTrackingInterceptor.post_eval_hook

````

``````
