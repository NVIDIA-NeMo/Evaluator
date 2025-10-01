# {py:mod}`nemo_evaluator.api.api_dataclasses`

```{py:module} nemo_evaluator.api.api_dataclasses
```

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EndpointType <nemo_evaluator.api.api_dataclasses.EndpointType>`
  -
* - {py:obj}`ApiEndpoint <nemo_evaluator.api.api_dataclasses.ApiEndpoint>`
  -
* - {py:obj}`EvaluationTarget <nemo_evaluator.api.api_dataclasses.EvaluationTarget>`
  -
* - {py:obj}`ConfigParams <nemo_evaluator.api.api_dataclasses.ConfigParams>`
  -
* - {py:obj}`EvaluationConfig <nemo_evaluator.api.api_dataclasses.EvaluationConfig>`
  -
* - {py:obj}`Evaluation <nemo_evaluator.api.api_dataclasses.Evaluation>`
  -
* - {py:obj}`ScoreStats <nemo_evaluator.api.api_dataclasses.ScoreStats>`
  - ```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`Score <nemo_evaluator.api.api_dataclasses.Score>`
  -
* - {py:obj}`MetricResult <nemo_evaluator.api.api_dataclasses.MetricResult>`
  -
* - {py:obj}`TaskResult <nemo_evaluator.api.api_dataclasses.TaskResult>`
  -
* - {py:obj}`GroupResult <nemo_evaluator.api.api_dataclasses.GroupResult>`
  - ```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.GroupResult
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`EvaluationResult <nemo_evaluator.api.api_dataclasses.EvaluationResult>`
  -
````

### API

`````{py:class} EndpointType()
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

````{py:attribute} UNDEFINED
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType.UNDEFINED
:value: >
   'undefined'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EndpointType.UNDEFINED
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} CHAT
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType.CHAT
:value: >
   'chat'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EndpointType.CHAT
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} COMPLETIONS
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType.COMPLETIONS
:value: >
   'completions'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EndpointType.COMPLETIONS
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} VLM
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType.VLM
:value: >
   'vlm'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EndpointType.VLM
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} EMBEDDING
:canonical: nemo_evaluator.api.api_dataclasses.EndpointType.EMBEDDING
:value: >
   'embedding'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EndpointType.EMBEDDING
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} ApiEndpoint(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} model_config
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.model_config
:value: >
   'ConfigDict(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.model_config
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} api_key
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.api_key
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.api_key
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} model_id
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.model_id
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.model_id
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stream
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.stream
:type: typing.Optional[bool]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.stream
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} type
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.type
:type: typing.Optional[nemo_evaluator.api.api_dataclasses.EndpointType]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.type
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} url
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.url
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.url
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} adapter_config
:canonical: nemo_evaluator.api.api_dataclasses.ApiEndpoint.adapter_config
:type: typing.Optional[nemo_evaluator.adapters.adapter_config.AdapterConfig]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ApiEndpoint.adapter_config
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} EvaluationTarget(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationTarget

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} api_endpoint
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationTarget.api_endpoint
:type: typing.Optional[nemo_evaluator.api.api_dataclasses.ApiEndpoint]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationTarget.api_endpoint
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} ConfigParams(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} limit_samples
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.limit_samples
:type: typing.Optional[int | float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.limit_samples
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_new_tokens
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.max_new_tokens
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.max_new_tokens
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max_retries
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.max_retries
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.max_retries
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} parallelism
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.parallelism
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.parallelism
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} task
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.task
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.task
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} temperature
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.temperature
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.temperature
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} request_timeout
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.request_timeout
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.request_timeout
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} top_p
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.top_p
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.top_p
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} extra
:canonical: nemo_evaluator.api.api_dataclasses.ConfigParams.extra
:type: typing.Optional[typing.Dict[str, typing.Any]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ConfigParams.extra
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} EvaluationConfig(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationConfig

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} output_dir
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationConfig.output_dir
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationConfig.output_dir
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} params
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationConfig.params
:type: typing.Optional[nemo_evaluator.api.api_dataclasses.ConfigParams]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationConfig.params
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} supported_endpoint_types
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationConfig.supported_endpoint_types
:type: typing.Optional[list[str]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationConfig.supported_endpoint_types
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} type
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationConfig.type
:type: typing.Optional[str]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationConfig.type
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} Evaluation(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} command
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.command
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.command
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} framework_name
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.framework_name
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.framework_name
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} pkg_name
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.pkg_name
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.pkg_name
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} config
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.config
:type: nemo_evaluator.api.api_dataclasses.EvaluationConfig
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.config
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} target
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.target
:type: nemo_evaluator.api.api_dataclasses.EvaluationTarget
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.target
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} render_command()
:canonical: nemo_evaluator.api.api_dataclasses.Evaluation.render_command

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Evaluation.render_command
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} ScoreStats(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} count
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.count
:type: typing.Optional[int]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.count
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} sum
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.sum
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.sum
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} sum_squared
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.sum_squared
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.sum_squared
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} min
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.min
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.min
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} max
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.max
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.max
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} mean
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.mean
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.mean
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} variance
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.variance
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.variance
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stddev
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.stddev
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.stddev
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stderr
:canonical: nemo_evaluator.api.api_dataclasses.ScoreStats.stderr
:type: typing.Optional[float]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.ScoreStats.stderr
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} Score(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.Score

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} value
:canonical: nemo_evaluator.api.api_dataclasses.Score.value
:type: float
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Score.value
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} stats
:canonical: nemo_evaluator.api.api_dataclasses.Score.stats
:type: nemo_evaluator.api.api_dataclasses.ScoreStats
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.Score.stats
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} MetricResult(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.MetricResult

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} scores
:canonical: nemo_evaluator.api.api_dataclasses.MetricResult.scores
:type: typing.Dict[str, nemo_evaluator.api.api_dataclasses.Score]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.MetricResult.scores
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} TaskResult(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.TaskResult

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} metrics
:canonical: nemo_evaluator.api.api_dataclasses.TaskResult.metrics
:type: typing.Dict[str, nemo_evaluator.api.api_dataclasses.MetricResult]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.TaskResult.metrics
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} GroupResult(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.GroupResult

Bases: {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.GroupResult
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.GroupResult.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:attribute} groups
:canonical: nemo_evaluator.api.api_dataclasses.GroupResult.groups
:type: typing.Optional[typing.Dict[str, nemo_evaluator.api.api_dataclasses.GroupResult]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.GroupResult.groups
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} metrics
:canonical: nemo_evaluator.api.api_dataclasses.GroupResult.metrics
:type: typing.Dict[str, nemo_evaluator.api.api_dataclasses.MetricResult]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.GroupResult.metrics
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} EvaluationResult(/, **data: typing.Any)
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationResult

Bases: {py:obj}`pydantic.BaseModel`

````{py:attribute} tasks
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationResult.tasks
:type: typing.Optional[typing.Dict[str, nemo_evaluator.api.api_dataclasses.TaskResult]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationResult.tasks
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:attribute} groups
:canonical: nemo_evaluator.api.api_dataclasses.EvaluationResult.groups
:type: typing.Optional[typing.Dict[str, nemo_evaluator.api.api_dataclasses.GroupResult]]
:value: >
   'Field(...)'

```{autodoc2-docstring} nemo_evaluator.api.api_dataclasses.EvaluationResult.groups
:parser: docs.autodoc2_docstrings_parser
```

````

`````
