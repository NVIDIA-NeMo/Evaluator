# {py:mod}`nemo_evaluator.core.utils`

```{py:module} nemo_evaluator.core.utils
```

```{autodoc2-docstring} nemo_evaluator.core.utils
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`deep_update <nemo_evaluator.core.utils.deep_update>`
  - ```{autodoc2-docstring} nemo_evaluator.core.utils.deep_update
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`dotlist_to_dict <nemo_evaluator.core.utils.dotlist_to_dict>`
  - ```{autodoc2-docstring} nemo_evaluator.core.utils.dotlist_to_dict
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`run_command <nemo_evaluator.core.utils.run_command>`
  - ```{autodoc2-docstring} nemo_evaluator.core.utils.run_command
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.core.utils.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.core.utils.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`KeyType <nemo_evaluator.core.utils.KeyType>`
  - ```{autodoc2-docstring} nemo_evaluator.core.utils.KeyType
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_evaluator.core.utils.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.core.utils.logger
:parser: docs.autodoc2_docstrings_parser
```

````

```{py:exception} MisconfigurationError()
:canonical: nemo_evaluator.core.utils.MisconfigurationError

Bases: {py:obj}`Exception`

```

````{py:data} KeyType
:canonical: nemo_evaluator.core.utils.KeyType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} nemo_evaluator.core.utils.KeyType
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} deep_update(mapping: dict[nemo_evaluator.core.utils.KeyType, typing.Any], *updating_mappings: dict[nemo_evaluator.core.utils.KeyType, typing.Any], skip_nones: bool = False) -> dict[nemo_evaluator.core.utils.KeyType, typing.Any]
:canonical: nemo_evaluator.core.utils.deep_update

```{autodoc2-docstring} nemo_evaluator.core.utils.deep_update
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} dotlist_to_dict(dotlist: list[str]) -> dict
:canonical: nemo_evaluator.core.utils.dotlist_to_dict

```{autodoc2-docstring} nemo_evaluator.core.utils.dotlist_to_dict
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} run_command(command, cwd=None, verbose=False, propagate_errors=False)
:canonical: nemo_evaluator.core.utils.run_command

```{autodoc2-docstring} nemo_evaluator.core.utils.run_command
:parser: docs.autodoc2_docstrings_parser
```
````
