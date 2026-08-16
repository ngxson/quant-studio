"""quant-studio kernel registry: every module in this package exposing SPECS is merged here."""

import importlib
import pkgutil
import sys
from pathlib import Path

# the kernels need the llama.cpp fork's gguf-py; prefer it over any installed gguf
if "gguf" not in sys.modules:
    _root = Path(__file__).resolve().parent.parent
    for _p in (_root / "gguf-py", _root / "llama.cpp" / "gguf-py", _root.parent / "llama.cpp" / "gguf-py"):
        if _p.is_dir():
            sys.path.insert(0, str(_p))
            break

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
