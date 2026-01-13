``nemo_evaluator.api``
======================================

The central point of evaluation is ``evaluate()`` function that takes standarized input and returns standarized output. See :ref:`modelling-inout` to learn how to instantiate standardized input and consume standardized output. Below is an example of how one might configure and run evaluation via Python API::

    from nemo_evaluator.core.evaluate import evaluate
    from nemo_evaluator.api.api_dataclasses import (
        EvaluationConfig, 
        EvaluationTarget, 
        ConfigParams,
        ApiEndpoint
    )

    # Create evaluation configuration
    eval_config = EvaluationConfig(
        type="simple_evals.mmlu_pro",
        output_dir="./results", 
        params=ConfigParams(
            limit_samples=100,
            temperature=0.1
        )
    )

    # Create target configuration
    target_config = EvaluationTarget(
        api_endpoint=ApiEndpoint(
            url="https://integrate.api.NVIDIA.com/v1/chat/completions",
            model_id="meta/llama-3.2-3b-instruct",
            type="chat",
            api_key="MY_API_KEY" # Name of the environment variable that stores api_key
        )
    )

    # Run evaluation
    result = evaluate(eval_config, target_config)

.. automodule:: nemo_evaluator.api
    :members:
    :undoc-members:
    :member-order: bysource


.. toctree::
    :hidden:
    
    api-dataclasses
    nemo-evaluator.adapters <../adapters/adapters>


