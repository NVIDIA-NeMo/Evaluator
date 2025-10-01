# {py:mod}`nemo_evaluator.core.input`

```{py:module} nemo_evaluator.core.input
```

```{autodoc2-docstring} nemo_evaluator.core.input
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_run_config <nemo_evaluator.core.input.load_run_config>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.load_run_config
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`parse_cli_args <nemo_evaluator.core.input.parse_cli_args>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.parse_cli_args
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`parse_override_params <nemo_evaluator.core.input.parse_override_params>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.parse_override_params
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_framework_evaluations <nemo_evaluator.core.input.get_framework_evaluations>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.get_framework_evaluations
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_get_framework_evaluations <nemo_evaluator.core.input._get_framework_evaluations>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input._get_framework_evaluations
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`merge_dicts <nemo_evaluator.core.input.merge_dicts>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.merge_dicts
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_available_evaluations <nemo_evaluator.core.input.get_available_evaluations>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.get_available_evaluations
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`check_task_invocation <nemo_evaluator.core.input.check_task_invocation>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.check_task_invocation
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`check_required_default_missing <nemo_evaluator.core.input.check_required_default_missing>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.check_required_default_missing
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`check_adapter_config <nemo_evaluator.core.input.check_adapter_config>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.check_adapter_config
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_evaluation <nemo_evaluator.core.input.get_evaluation>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.get_evaluation
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`check_type_compatibility <nemo_evaluator.core.input.check_type_compatibility>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.check_type_compatibility
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`prepare_output_directory <nemo_evaluator.core.input.prepare_output_directory>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.prepare_output_directory
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`validate_configuration <nemo_evaluator.core.input.validate_configuration>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.validate_configuration
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.core.input.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.core.input.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_evaluator.core.input.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.core.input.logger
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} load_run_config(yaml_file: str) -> dict
:canonical: nemo_evaluator.core.input.load_run_config

```{autodoc2-docstring} nemo_evaluator.core.input.load_run_config
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} parse_cli_args(args) -> dict
:canonical: nemo_evaluator.core.input.parse_cli_args

```{autodoc2-docstring} nemo_evaluator.core.input.parse_cli_args
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} parse_override_params(override_params_str: typing.Optional[str] = None) -> dict
:canonical: nemo_evaluator.core.input.parse_override_params

```{autodoc2-docstring} nemo_evaluator.core.input.parse_override_params
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_framework_evaluations(filepath: str) -> tuple[str, dict, dict[str, nemo_evaluator.api.api_dataclasses.Evaluation]]
:canonical: nemo_evaluator.core.input.get_framework_evaluations

```{autodoc2-docstring} nemo_evaluator.core.input.get_framework_evaluations
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} _get_framework_evaluations(def_file: str) -> tuple[dict[str, dict[str, nemo_evaluator.api.api_dataclasses.Evaluation]], dict[str, dict], dict[str, nemo_evaluator.api.api_dataclasses.Evaluation]]
:canonical: nemo_evaluator.core.input._get_framework_evaluations

```{autodoc2-docstring} nemo_evaluator.core.input._get_framework_evaluations
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} merge_dicts(dict1, dict2)
:canonical: nemo_evaluator.core.input.merge_dicts

```{autodoc2-docstring} nemo_evaluator.core.input.merge_dicts
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_available_evaluations() -> tuple[dict[str, dict[str, nemo_evaluator.api.api_dataclasses.Evaluation]], dict[str, nemo_evaluator.api.api_dataclasses.Evaluation], dict]
:canonical: nemo_evaluator.core.input.get_available_evaluations

```{autodoc2-docstring} nemo_evaluator.core.input.get_available_evaluations
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} check_task_invocation(run_config: dict)
:canonical: nemo_evaluator.core.input.check_task_invocation

```{autodoc2-docstring} nemo_evaluator.core.input.check_task_invocation
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} check_required_default_missing(run_config: dict)
:canonical: nemo_evaluator.core.input.check_required_default_missing

```{autodoc2-docstring} nemo_evaluator.core.input.check_required_default_missing
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} check_adapter_config(run_config)
:canonical: nemo_evaluator.core.input.check_adapter_config

```{autodoc2-docstring} nemo_evaluator.core.input.check_adapter_config
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_evaluation(evaluation_config: nemo_evaluator.api.api_dataclasses.EvaluationConfig, target_config: nemo_evaluator.api.api_dataclasses.EvaluationTarget) -> nemo_evaluator.api.api_dataclasses.Evaluation
:canonical: nemo_evaluator.core.input.get_evaluation

```{autodoc2-docstring} nemo_evaluator.core.input.get_evaluation
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} check_type_compatibility(evaluation: nemo_evaluator.api.api_dataclasses.Evaluation)
:canonical: nemo_evaluator.core.input.check_type_compatibility

```{autodoc2-docstring} nemo_evaluator.core.input.check_type_compatibility
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} prepare_output_directory(evaluation: nemo_evaluator.api.api_dataclasses.Evaluation)
:canonical: nemo_evaluator.core.input.prepare_output_directory

```{autodoc2-docstring} nemo_evaluator.core.input.prepare_output_directory
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} validate_configuration(run_config: dict) -> nemo_evaluator.api.api_dataclasses.Evaluation
:canonical: nemo_evaluator.core.input.validate_configuration

```{autodoc2-docstring} nemo_evaluator.core.input.validate_configuration
:parser: docs.autodoc2_docstrings_parser
```
````
