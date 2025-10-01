# {py:mod}`nemo_evaluator.adapters.reports.post_eval_report_hook`

```{py:module} nemo_evaluator.adapters.reports.post_eval_report_hook
```

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PostEvalReportHook <nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

```````{py:class} PostEvalReportHook(params: Params)
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook

Bases: {py:obj}`nemo_evaluator.adapters.types.PostEvalHook`

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.__init__
:parser: docs.autodoc2_docstrings_parser
```

``````{py:class} Params(/, **data: typing.Any)
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} report_types
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.report_types
:type: typing.List[typing.Literal[html, json]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.report_types
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} html_report_size
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.html_report_size
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.html_report_size
:parser: docs.autodoc2_docstrings_parser
```

````

`````{py:class} Config
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.Config

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.Config
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} use_enum_values
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.Config.use_enum_values
:value: >
   True

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.Params.Config.use_enum_values
:parser: docs.autodoc2_docstrings_parser
```

````

`````

``````

````{py:method} _tojson_utf8(data: typing.Any) -> str
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._tojson_utf8

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._tojson_utf8
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_request_content(request_data: typing.Any) -> typing.Any
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._get_request_content

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._get_request_content
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _get_response_content(response_data: typing.Any) -> typing.Any
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._get_response_content

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._get_response_content
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _collect_entries(cache_dir: pathlib.Path, api_url: str) -> list
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._collect_entries

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._collect_entries
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _generate_html_report(entries: list, output_path: pathlib.Path) -> None
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._generate_html_report

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._generate_html_report
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _generate_json_report(entries: list, output_path: pathlib.Path) -> None
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._generate_json_report

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook._generate_json_report
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} post_eval_hook(context: nemo_evaluator.adapters.types.AdapterGlobalContext) -> None
:canonical: nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.post_eval_hook

```{autodoc2-docstring} nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook.post_eval_hook
:parser: docs.autodoc2_docstrings_parser
```

````

```````
