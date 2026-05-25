# src/utils/misc.py
from typing import Any
from typing import Mapping

import copy

from types import MappingProxyType

def freeze_dict(d: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Deep-freeze a mapping into an immutable structure.

    - Returns a fresh object every call (no caching).
    - Recursively freezes nested dicts.
    - Converts lists/tuples to tuples.
    - Leaves primitive values as-is.

    This prevents accidental mutation + prevents shared-state bugs in configs.
    """

    def _freeze(x: Any) -> Any:
        if isinstance(x, dict):
            # recursively freeze nested dicts
            return MappingProxyType({k: _freeze(v) for k, v in x.items()})
        if isinstance(x, (list, tuple)):
            return tuple(_freeze(v) for v in x)
        return x

    # ensure we return a fresh mappingproxy each call
    return _freeze(copy.deepcopy(dict(d)))

