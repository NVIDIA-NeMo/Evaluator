# {py:mod}`nemo_eval.utils.base`

```{py:module} nemo_eval.utils.base
```

```{autodoc2-docstring} nemo_eval.utils.base
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_endpoint <nemo_eval.utils.base.check_endpoint>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.check_endpoint
    :summary:
    ```
* - {py:obj}`check_health <nemo_eval.utils.base.check_health>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.check_health
    :summary:
    ```
* - {py:obj}`find_framework <nemo_eval.utils.base.find_framework>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.find_framework
    :summary:
    ```
* - {py:obj}`list_available_evaluations <nemo_eval.utils.base.list_available_evaluations>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.list_available_evaluations
    :summary:
    ```
* - {py:obj}`wait_for_fastapi_server <nemo_eval.utils.base.wait_for_fastapi_server>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.wait_for_fastapi_server
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_eval.utils.base.logger>`
  - ```{autodoc2-docstring} nemo_eval.utils.base.logger
    :summary:
    ```
````

### API

````{py:function} check_endpoint(endpoint_url: str, endpoint_type: str, model_name: str, max_retries: int = 600, retry_interval: int = 2) -> bool
:canonical: nemo_eval.utils.base.check_endpoint

```{autodoc2-docstring} nemo_eval.utils.base.check_endpoint
```
````

````{py:function} check_health(health_url: str, max_retries: int = 600, retry_interval: int = 2) -> bool
:canonical: nemo_eval.utils.base.check_health

```{autodoc2-docstring} nemo_eval.utils.base.check_health
```
````

````{py:function} find_framework(eval_task: str) -> str
:canonical: nemo_eval.utils.base.find_framework

```{autodoc2-docstring} nemo_eval.utils.base.find_framework
```
````

````{py:function} list_available_evaluations() -> dict[str, list[str]]
:canonical: nemo_eval.utils.base.list_available_evaluations

```{autodoc2-docstring} nemo_eval.utils.base.list_available_evaluations
```
````

````{py:data} logger
:canonical: nemo_eval.utils.base.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_eval.utils.base.logger
```

````

````{py:function} wait_for_fastapi_server(base_url: str = 'http://0.0.0.0:8080', model_name: str = 'megatron_model', max_retries: int = 600, retry_interval: int = 10)
:canonical: nemo_eval.utils.base.wait_for_fastapi_server

```{autodoc2-docstring} nemo_eval.utils.base.wait_for_fastapi_server
```
````
