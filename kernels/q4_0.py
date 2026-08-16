"""q4_0 quantization: bit-exact port of quantize_row_q4_0_ref() from ggml-quants.c."""

from __future__ import annotations

import torch
from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec

QK4_0 = 32


def block_scales(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """b: (n_blocks, 32); returns (d, inv_d) per block, both (n_blocks, 1)."""
    # signed value with the largest magnitude in each block
    amax_idx = b.abs().argmax(dim=1, keepdim=True)
    max_val = b.gather(1, amax_idx)

    d = max_val / -8.0
    inv_d = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    return d, inv_d


def pack_q4_0(d: torch.Tensor, q: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """d: (n_blocks, 1) float32, q: (n_blocks, 32) uint8; returns (n, k//32 * 18) uint8."""
    nb = k // QK4_0
    out = torch.empty((n * nb, 2 + QK4_0 // 2), dtype=torch.uint8, device=q.device)
    out[:, :2] = d.to(torch.float16).view(torch.uint8)          # fp16 scale, LE
    out[:, 2:] = q[:, : QK4_0 // 2] + q[:, QK4_0 // 2 :] * 16   # lo | hi << 4
    return out.reshape(n, nb * (2 + QK4_0 // 2))


def quantize_q4_0(x: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, k) float32, k % 32 == 0; returns (n_rows, k//32 * 18) uint8 on the same device."""
    n, k = x.shape
    b = x.reshape(n * k // QK4_0, QK4_0)
    d, inv_d = block_scales(b)

    # C ref: (int8_t)(v*id + 8.5f); the value is >= 0.5 so trunc == floor
    q = (b * inv_d + 8.5).floor().clamp_(0, 15).to(torch.uint8)
    return pack_q4_0(d, q, n, k)


def _make_kernel(device: torch.device, qw) -> callable:
    return quantize_q4_0


SPECS = {
    "q4_0": QuantSpec(GGMLQuantizationType.Q4_0, LlamaFileType.MOSTLY_Q4_0,
                      16, False, _make_kernel),
}
