"""quant-studio kernel registry: every module in this package exposing SPECS is merged here."""

import importlib
import pkgutil
import sys

from .common import QuantSpec, TORCH_DTYPES

QUANT_TYPES: dict[str, QuantSpec] = {}
for _m in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    if _m == "common" or _m.endswith("_common") or _m.endswith("_tables"):
        continue
    try:
        _mod = importlib.import_module(f".{_m}", __package__)
    except Exception as e:  # a broken kernel module should not take down the others
        print(f"warning: kernel module {_m} failed to import: {e}", file=sys.stderr)
        continue
    QUANT_TYPES.update(getattr(_mod, "SPECS", {}))
