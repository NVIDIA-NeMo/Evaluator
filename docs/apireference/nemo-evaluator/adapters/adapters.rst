nemo_evaluator.adapters
=======================



Interfaces
--------------
.. currentmodule:: nemo_evaluator.adapters.types
.. autosummary::
    :nosignatures:
    :recursive:

    RequestInterceptor
    ResponseInterceptor
    RequestToResponseInterceptor
    PostEvalHook


Configuration
--------------
.. currentmodule:: nemo_evaluator.adapters.adapter_config
.. autosummary::
    :nosignatures:
    :recursive:

    DiscoveryConfig
    InterceptorConfig
    PostEvalHookConfig
    AdapterConfig

.. .. automodule:: nemo_evaluator.adapters.adapter_config
..     :members:
..     :undoc-members:


Interceptors
-------------
.. currentmodule:: nemo_evaluator.adapters
.. autosummary::
    :nosignatures:
    :recursive:

    Caching
    EndpointInterceptor
    PayloadParamsModifierInterceptr,
    ProgressTrackingInterceptor
    RaiseClientErrorInterceptor
    RequestLoggingInterceptor
    ResponseLoggingInterceptor
    ResponseReasoningInterceptor
    ResponseStatsInterceptor
    SystemMessageInterceptor


.. .. automodule:: nemo_evaluator.adapters
..     :members:
..     :undoc-members:

.. toctree::
    :hidden:
    
    adapter-config
    interceptors
