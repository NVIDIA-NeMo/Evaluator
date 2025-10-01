# {py:mod}`nemo_evaluator.core.registry`

```{py:module} nemo_evaluator.core.registry
```

```{autodoc2-docstring} nemo_evaluator.core.registry
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalFramworkRegistry <nemo_evaluator.core.registry.EvalFramworkRegistry>`
  - ```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_global_registry <nemo_evaluator.core.registry.get_global_registry>`
  - ```{autodoc2-docstring} nemo_evaluator.core.registry.get_global_registry
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`register_framework <nemo_evaluator.core.registry.register_framework>`
  - ```{autodoc2-docstring} nemo_evaluator.core.registry.register_framework
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_REGISTRY <nemo_evaluator.core.registry._REGISTRY>`
  - ```{autodoc2-docstring} nemo_evaluator.core.registry._REGISTRY
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} EvalFramworkRegistry
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} _instance
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry._instance
:type: typing.ClassVar[EvalFramworkRegistry | None]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry._instance
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} _entries
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry._entries
:type: list[typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]]
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry._entries
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __new__() -> nemo_evaluator.core.registry.EvalFramworkRegistry
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry.__new__

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry.__new__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} register(framework_entry_class: typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]) -> typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry.register

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry.register
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get_entries() -> list[typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]]
:canonical: nemo_evaluator.core.registry.EvalFramworkRegistry.get_entries

```{autodoc2-docstring} nemo_evaluator.core.registry.EvalFramworkRegistry.get_entries
:parser: docs.autodoc2_docstrings_parser
```

````

`````

````{py:data} _REGISTRY
:canonical: nemo_evaluator.core.registry._REGISTRY
:value: >
   'EvalFramworkRegistry(...)'

```{autodoc2-docstring} nemo_evaluator.core.registry._REGISTRY
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} get_global_registry() -> nemo_evaluator.core.registry.EvalFramworkRegistry
:canonical: nemo_evaluator.core.registry.get_global_registry

```{autodoc2-docstring} nemo_evaluator.core.registry.get_global_registry
:parser: docs.autodoc2_docstrings_parser
```
````

````{py:function} register_framework(user_framework: typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]) -> typing.Type[nemo_evaluator.api.base.EvalFrameworkBase]
:canonical: nemo_evaluator.core.registry.register_framework

```{autodoc2-docstring} nemo_evaluator.core.registry.register_framework
:parser: docs.autodoc2_docstrings_parser
```
````
