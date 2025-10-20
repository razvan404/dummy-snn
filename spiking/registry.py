from collections import defaultdict
from typing import Any


class __Registry:
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, module: str, name: str):
        def decorator(model_cls):
            cls._registry[module][name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(
        cls, module: str, name: str | None = None, kwargs: dict[str, Any] = None
    ):
        if name is None:
            module_splits = module.split(".")
            module, name = ".".join(module_splits[:-1]), module_splits[-1]

        if module not in cls._registry:
            raise NotImplementedError(f"Module {module} is not registered")

        if name not in cls._registry[module]:
            raise NotImplementedError(
                f"Class {name} is not registered in module {module}"
            )

        return cls._registry[module][name](**(kwargs or {}))


registry = __Registry()
