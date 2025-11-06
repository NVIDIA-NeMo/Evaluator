``nemo_evaluator.adapters``
=======================

Interceptors and PostEvalHooks are important part of NeMo Evaluator SDK. They expand functionality of each harness, providing a standardized way of enabling features in your evaluation runs.  
Behind each interceptor and post-eval-hook stands a specific class that implements its logic. 
However, these classes are referenced only to provide their configuration options which are reflected in ``Params`` of each class and to point from which classes one should inherit. 

From the usage perspective, one should always use configuration classes (see :ref:`configuration`) to add them to evaluations. No interceptor should be directly instantiated.

Interceptors are defined in a chain. They go under ``target.api_endpoint.adapter_config`` and can be defined as follow::

    adapter_config = AdapterConfig(
        interceptors=[
            InterceptorConfig(
                name="system_message",
                enabled=True,
                config={"system_message": "You are a helpful assistant."}
            ),
            InterceptorConfig(name="request_logging", enabled=True),
            InterceptorConfig(
                name="caching",
                enabled=True,
                config={"cache_dir": "./cache", "reuse_cached_responses": True}
            ),
            InterceptorConfig(name="reasoning", enabled=True),
            InterceptorConfig(name="endpoint")
        ]
    )
.. _configuration:

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
.. currentmodule:: nemo_evaluator.adapters.interceptors
.. autosummary::
    :nosignatures:
    :recursive:

    CachingInterceptor
    EndpointInterceptor
    PayloadParamsModifierInterceptor
    ProgressTrackingInterceptor
    RaiseClientErrorInterceptor
    RequestLoggingInterceptor
    ResponseLoggingInterceptor
    ResponseReasoningInterceptor
    ResponseStatsInterceptor
    SystemMessageInterceptor

PostEvalHooks
-------------
.. currentmodule:: nemo_evaluator.adapters.reports
.. autosummary::
    :nosignatures:
    :recursive:

    PostEvalReportHook

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


.. .. automodule:: nemo_evaluator.adapters
..     :members:
..     :undoc-members:

.. toctree::
    :hidden:
    
    adapter-config
    interceptors
    types
