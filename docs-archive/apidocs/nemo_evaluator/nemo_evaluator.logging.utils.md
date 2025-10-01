# {py:mod}`nemo_evaluator.logging.utils`

```{py:module} nemo_evaluator.logging.utils
```

```{autodoc2-docstring} nemo_evaluator.logging.utils
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MainConsoleRenderer <nemo_evaluator.logging.utils.MainConsoleRenderer>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_ensure_log_dir <nemo_evaluator.logging.utils._ensure_log_dir>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils._ensure_log_dir
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_get_env_log_dir <nemo_evaluator.logging.utils._get_env_log_dir>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils._get_env_log_dir
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_get_env_log_level <nemo_evaluator.logging.utils._get_env_log_level>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils._get_env_log_level
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`custom_timestamper <nemo_evaluator.logging.utils.custom_timestamper>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils.custom_timestamper
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_configure_structlog <nemo_evaluator.logging.utils._configure_structlog>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils._configure_structlog
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`configure_logging <nemo_evaluator.logging.utils.configure_logging>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils.configure_logging
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`get_logger <nemo_evaluator.logging.utils.get_logger>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils.get_logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.logging.utils.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.logging.utils.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:function} _ensure_log_dir(log_dir: str = None) -> pathlib.Path
:canonical: nemo_evaluator.logging.utils._ensure_log_dir

```{autodoc2-docstring} nemo_evaluator.logging.utils._ensure_log_dir
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} _get_env_log_dir() -> str | None
:canonical: nemo_evaluator.logging.utils._get_env_log_dir

```{autodoc2-docstring} nemo_evaluator.logging.utils._get_env_log_dir
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} _get_env_log_level() -> str
:canonical: nemo_evaluator.logging.utils._get_env_log_level

```{autodoc2-docstring} nemo_evaluator.logging.utils._get_env_log_level
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} custom_timestamper(_, __, event_dict)
:canonical: nemo_evaluator.logging.utils.custom_timestamper

```{autodoc2-docstring} nemo_evaluator.logging.utils.custom_timestamper
:parser: docs.autodoc2_docstrings_parser
```
````

`````{py:class} MainConsoleRenderer(colors: bool = True)
:canonical: nemo_evaluator.logging.utils.MainConsoleRenderer

```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} LEVEL_MAP
:canonical: nemo_evaluator.logging.utils.MainConsoleRenderer.LEVEL_MAP
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer.LEVEL_MAP
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} RESET
:canonical: nemo_evaluator.logging.utils.MainConsoleRenderer.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer.RESET
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __call__(logger, method_name, event_dict)
:canonical: nemo_evaluator.logging.utils.MainConsoleRenderer.__call__

```{autodoc2-docstring} nemo_evaluator.logging.utils.MainConsoleRenderer.__call__
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:function} _configure_structlog(log_dir: str = None) -> None
:canonical: nemo_evaluator.logging.utils._configure_structlog

```{autodoc2-docstring} nemo_evaluator.logging.utils._configure_structlog
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} configure_logging(log_dir: str = None) -> None
:canonical: nemo_evaluator.logging.utils.configure_logging

```{autodoc2-docstring} nemo_evaluator.logging.utils.configure_logging
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} get_logger(name: str = None) -> structlog.BoundLogger
:canonical: nemo_evaluator.logging.utils.get_logger

```{autodoc2-docstring} nemo_evaluator.logging.utils.get_logger
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:data} logger
:canonical: nemo_evaluator.logging.utils.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.logging.utils.logger
:parser: docs.autodoc2_docstrings_parser
```

````
