from typing import Dict, Type

_EXPORTER_REGISTRY: Dict[str, Type] = {}


def register_exporter(name: str):
    def wrapper(cls):
        _EXPORTER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_exporter(name: str):
    if name not in _EXPORTER_REGISTRY:
        raise ValueError(
            f"Unknown exporter: {name}. Available: {list(_EXPORTER_REGISTRY.keys())}"
        )
    return _EXPORTER_REGISTRY[name]


def available_exporters() -> list[str]:
    return list(_EXPORTER_REGISTRY.keys())
