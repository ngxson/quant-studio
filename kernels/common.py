"""Shared pieces for quant-studio kernels."""

from __future__ import annotations

import torch
from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tld
    HAS_TRITON = True
except ImportError:  # CPU/MPS-only environments
    triton = tl = tld = None
    HAS_TRITON = False

F32_TINY = torch.finfo(torch.float32).tiny

TORCH_DTYPES = {
    GGMLQuantizationType.F32: torch.float32,
    GGMLQuantizationType.F16: torch.float16,
    GGMLQuantizationType.BF16: torch.bfloat16,
}


class QuantSpec:
    def __init__(self, ggml_type: GGMLQuantizationType, ftype: LlamaFileType,
                 bytes_per_elem: int, needs_imatrix: bool, make_kernel):
        self.ggml_type = ggml_type
        self.ftype = ftype
        self.bytes_per_elem = bytes_per_elem  # rough full-pipeline working set per element
        self.needs_imatrix = needs_imatrix
        self.make_kernel = make_kernel        # (device, qw_tensor|None) -> fn(x)->uint8


_compiled: dict[str, object] = {}


def jit(name: str, fn):
    """Process-wide torch.compile cache for shape-stable inner functions (CUDA paths)."""
    if name not in _compiled:
        _compiled[name] = torch.compile(fn, dynamic=True)
    return _compiled[name]
