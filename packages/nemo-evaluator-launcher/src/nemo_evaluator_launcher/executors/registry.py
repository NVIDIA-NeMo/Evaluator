_EXECUTOR_REGISTRY = {}


def register_executor(executor_name: str):
    def wrapper(executor_cls):
        _EXECUTOR_REGISTRY[executor_name] = executor_cls
        return executor_cls

    return wrapper


def get_executor(executor_name: str):
    if executor_name not in _EXECUTOR_REGISTRY:
        raise ValueError(
            f"Executor {executor_name} not found. Available executors: {list(_EXECUTOR_REGISTRY.keys())}"
        )
    return _EXECUTOR_REGISTRY[executor_name]
