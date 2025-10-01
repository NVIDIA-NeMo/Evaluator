# {py:mod}`nemo_evaluator.adapters.server`

```{py:module} nemo_evaluator.adapters.server
```

```{autodoc2-docstring} nemo_evaluator.adapters.server
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdapterServerProcess <nemo_evaluator.adapters.server.AdapterServerProcess>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServerProcess
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`AdapterServer <nemo_evaluator.adapters.server.AdapterServer>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_setup_file_logging <nemo_evaluator.adapters.server._setup_file_logging>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server._setup_file_logging
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`is_port_open <nemo_evaluator.adapters.server.is_port_open>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.is_port_open
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`wait_for_server <nemo_evaluator.adapters.server.wait_for_server>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.wait_for_server
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`_run_adapter_server <nemo_evaluator.adapters.server._run_adapter_server>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server._run_adapter_server
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`spawn_adapter_server <nemo_evaluator.adapters.server.spawn_adapter_server>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.spawn_adapter_server
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_evaluator.adapters.server.logger>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.server.logger
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_evaluator.adapters.server.logger
:value: >
   'get_logger(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.server.logger
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} _setup_file_logging() -> None
:canonical: nemo_evaluator.adapters.server._setup_file_logging

```{autodoc2-docstring} nemo_evaluator.adapters.server._setup_file_logging
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} is_port_open(host: str, port: int, timeout: float = 0.5) -> bool
:canonical: nemo_evaluator.adapters.server.is_port_open

```{autodoc2-docstring} nemo_evaluator.adapters.server.is_port_open
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} wait_for_server(host: str, port: int, max_wait: float = 10, interval: float = 0.2) -> bool
:canonical: nemo_evaluator.adapters.server.wait_for_server

```{autodoc2-docstring} nemo_evaluator.adapters.server.wait_for_server
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} _run_adapter_server(api_url: str, output_dir: str, adapter_config: nemo_evaluator.adapters.adapter_config.AdapterConfig) -> None
:canonical: nemo_evaluator.adapters.server._run_adapter_server

```{autodoc2-docstring} nemo_evaluator.adapters.server._run_adapter_server
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} spawn_adapter_server(api_url: str, output_dir: str, adapter_config: nemo_evaluator.adapters.adapter_config.AdapterConfig) -> multiprocessing.context.SpawnProcess | None
:canonical: nemo_evaluator.adapters.server.spawn_adapter_server

```{autodoc2-docstring} nemo_evaluator.adapters.server.spawn_adapter_server
:parser: docs.autodoc2_docstrings_parser
```
````

`````{py:class} AdapterServerProcess(evaluation: nemo_evaluator.api.api_dataclasses.Evaluation)
:canonical: nemo_evaluator.adapters.server.AdapterServerProcess

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServerProcess
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServerProcess.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} __enter__()
:canonical: nemo_evaluator.adapters.server.AdapterServerProcess.__enter__

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServerProcess.__enter__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __exit__(type, value, traceback)
:canonical: nemo_evaluator.adapters.server.AdapterServerProcess.__exit__

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServerProcess.__exit__
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} AdapterServer(api_url: str, output_dir: str, adapter_config: nemo_evaluator.adapters.adapter_config.AdapterConfig)
:canonical: nemo_evaluator.adapters.server.AdapterServer

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} DEFAULT_ADAPTER_HOST
:canonical: nemo_evaluator.adapters.server.AdapterServer.DEFAULT_ADAPTER_HOST
:type: str
:value: >
   'localhost'

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.DEFAULT_ADAPTER_HOST
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} DEFAULT_ADAPTER_PORT
:canonical: nemo_evaluator.adapters.server.AdapterServer.DEFAULT_ADAPTER_PORT
:type: int
:value: >
   3825

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.DEFAULT_ADAPTER_PORT
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _validate_and_build_chains() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._validate_and_build_chains

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._validate_and_build_chains
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _validate_adapter_chain_definition() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._validate_adapter_chain_definition

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._validate_adapter_chain_definition
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _validate_interceptor_order() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._validate_interceptor_order

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._validate_interceptor_order
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _build_interceptor_chains() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._build_interceptor_chains

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._build_interceptor_chains
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _build_post_eval_hooks() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._build_post_eval_hooks

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._build_post_eval_hooks
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} run() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer.run

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.run
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _EXCLUDED_HEADERS
:canonical: nemo_evaluator.adapters.server.AdapterServer._EXCLUDED_HEADERS
:value: >
   ['content-encoding', 'content-length', 'transfer-encoding', 'connection', 'keep-alive', 'proxy-authe...

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._EXCLUDED_HEADERS
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _process_response_headers(response: requests.Response) -> typing.List[tuple[str, str]]
:canonical: nemo_evaluator.adapters.server.AdapterServer._process_response_headers
:classmethod:

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._process_response_headers
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _handler(path: str) -> flask.Response
:canonical: nemo_evaluator.adapters.server.AdapterServer._handler

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._handler
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _log_failed_request(status_code: int, error_message: str, current_request=None) -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer._log_failed_request

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._log_failed_request
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _run_post_eval_hooks_handler() -> flask.Response
:canonical: nemo_evaluator.adapters.server.AdapterServer._run_post_eval_hooks_handler

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer._run_post_eval_hooks_handler
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} run_post_eval_hooks() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer.run_post_eval_hooks

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.run_post_eval_hooks
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} generate_report() -> None
:canonical: nemo_evaluator.adapters.server.AdapterServer.generate_report

```{autodoc2-docstring} nemo_evaluator.adapters.server.AdapterServer.generate_report
:parser: docs.autodoc2_docstrings_parser
```

````

`````
