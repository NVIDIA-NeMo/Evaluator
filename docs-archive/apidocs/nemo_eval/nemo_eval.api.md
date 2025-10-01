# {py:mod}`nemo_eval.api`

```{py:module} nemo_eval.api
```

```{autodoc2-docstring} nemo_eval.api
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`deploy <nemo_eval.api.deploy>`
  - ```{autodoc2-docstring} nemo_eval.api.deploy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AnyPath <nemo_eval.api.AnyPath>`
  - ```{autodoc2-docstring} nemo_eval.api.AnyPath
    :summary:
    ```
* - {py:obj}`logger <nemo_eval.api.logger>`
  - ```{autodoc2-docstring} nemo_eval.api.logger
    :summary:
    ```
````

### API

````{py:data} AnyPath
:canonical: nemo_eval.api.AnyPath
:value: >
   None

```{autodoc2-docstring} nemo_eval.api.AnyPath
```

````

````{py:function} deploy(nemo_checkpoint: Optional[nemo_eval.api.AnyPath] = None, hf_model_id_path: Optional[nemo_eval.api.AnyPath] = None, serving_backend: str = 'pytriton', model_name: str = 'megatron_model', server_port: int = 8080, server_address: str = '0.0.0.0', triton_address: str = '0.0.0.0', triton_port: int = 8000, num_gpus: int = 1, num_nodes: int = 1, tensor_parallelism_size: int = 1, pipeline_parallelism_size: int = 1, context_parallel_size: int = 1, expert_model_parallel_size: int = 1, max_input_len: int = 4096, max_batch_size: int = 8, enable_flash_decode: bool = True, enable_cuda_graphs: bool = True, legacy_ckpt: bool = False, use_vllm_backend: bool = True, num_replicas: int = 1, num_cpus: Optional[int] = None, include_dashboard: bool = True)
:canonical: nemo_eval.api.deploy

```{autodoc2-docstring} nemo_eval.api.deploy
```
````

````{py:data} logger
:canonical: nemo_eval.api.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_eval.api.logger
```

````
