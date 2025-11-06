.. _modelling-inout:

``nemo_evaluator.api.api_dataclasses``
======================================


NeMo Evaluator Core operates on strictly defined input and output which are modelled through pydantic dataclasses. Whether you use Python API or CLI, the reference below serves as a map of configuration options and output format.  




.. currentmodule:: nemo_evaluator.api.api_dataclasses

Modeling Target
---------------
.. autosummary::
    :nosignatures:
    :recursive:

    ApiEndpoint
    EndpointType
    EvaluationTarget

Modeling Evaluation
---------------
.. autosummary::
    :nosignatures:
    :recursive:

    EvaluationConfig
    ConfigParams


Modeling Result
---------------

.. autosummary::
    :nosignatures:
    :recursive:


    EvaluationResult
    GroupResult
    MetricResult
    Score
    ScoreStats
    TaskResult


.. automodule:: nemo_evaluator.api.api_dataclasses
    :members:
    :undoc-members:


