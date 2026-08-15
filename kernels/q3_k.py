"""Q3_K quantization: port of quantize_row_q3_K_ref() / quantize_row_q3_K_impl() from ggml-quants.c.

Matches the quantize_q3_K() dispatch: without an imatrix the _ref path (make_q3_quants
per 16-group, 6-bit scales), with one the _impl path (make_qx_quants per 16-group plus
make_qx_quants over the 16 sub-block scales).
The search loops live in qx_common (Triton on CUDA, eager torch on CPU/MPS).
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .qx_common import absmax_first, make_q3_quants, make_qx_quants, seq_sum

QK_K = 256
BLOCK_BYTES = QK_K // 8 + QK_K // 4 + 12 + 2  # hmask + 2-bit quants + 6-bit scales + fp16 d = 110


def quantize_q3_k(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 110) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    xsb = x.reshape(ns, QK_K)
    xg = xsb.reshape(ns * 16, 16)

    if qw is None:
        scales, L = make_q3_quants(xg, 4)
        sc16 = scales.reshape(ns, 16)
        _, max_scale = absmax_first(sc16)
        has = max_scale != 0
        iscale = -32.0 / torch.where(has, max_scale, torch.ones_like(max_scale))
        # the C code narrows nearest_int() to int8 before the clamp
        l = torch.round(iscale[:, None] * sc16).long()
        l = (((l & 0xFF) ^ 0x80) - 0x80).clamp_(-32, 31) + 32
        ls6 = torch.where(has[:, None], l, torch.zeros_like(l))
        d16 = torch.where(has, 1.0 / iscale, torch.zeros_like(iscale)).to(torch.float16)
    else:
        sigma2 = 2.0 * seq_sum(xsb * xsb) / QK_K
        qw_sb = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        w = (qw_sb * torch.sqrt(sigma2[:, None] + xsb * xsb)).reshape(ns * 16, 16)
        scales, L = make_qx_quants(xg, w, 4)
        sc16 = scales.reshape(ns, 16)
        sw = seq_sum(w.reshape(ns, 16, 16))
        d_blk, ls = make_qx_quants(sc16, sw, 32)
        ls6 = ls.long()
        d16 = d_blk.to(torch.float16)

    # requantize with the 6-bit scales; d == 0 keeps the levels from the group fit like the C code
    dq = d16.to(torch.float32)[:, None] * (ls6 - 32).float()
    use = dq != 0
    dsafe = torch.where(use, dq, torch.ones_like(dq))
    x16 = xsb.reshape(ns, 16, 16)
    lq = torch.round(x16 / dsafe[..., None]).clamp_(-4.0, 3.0) + 4.0
    Lfin = torch.where(use[..., None], lq, L.reshape(ns, 16, 16)).reshape(ns, QK_K).to(torch.int64)

    # hmask bit b of byte l covers value 32*b + l; qs packs quarters of each 128-half
    hb = Lfin > 3
    L2 = torch.where(hb, Lfin - 4, Lfin)
    bits = torch.arange(8, device=x.device)
    hmask = (hb.reshape(ns, 8, 32).long() << bits[None, :, None]).sum(1).to(torch.uint8)
    q4 = L2.reshape(ns, 2, 4, 32)
    qs = (q4[:, :, 0] | (q4[:, :, 1] << 2) | (q4[:, :, 2] << 4) | (q4[:, :, 3] << 6))
    qs = qs.reshape(ns, QK_K // 4).to(torch.uint8)

    # split 6-bit scales: low nibbles pair (j, j+8), high 2-bit parts land in bytes 8..11
    lo = ls6 & 0xF
    hi = ls6 >> 4
    b07 = lo[:, :8] | (lo[:, 8:] << 4)
    b8b = hi[:, 0:4] | (hi[:, 4:8] << 2) | (hi[:, 8:12] << 4) | (hi[:, 12:16] << 6)
    scb = torch.cat([b07, b8b], dim=1).to(torch.uint8)

    out = torch.cat([hmask, qs, scb, d16[:, None].view(torch.uint8)], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_q3_k(x, qw)


SPECS = {
    "q3_k": QuantSpec(GGMLQuantizationType.Q3_K, LlamaFileType.MOSTLY_Q3_K_M,
                      24, False, _make_kernel, uses_imatrix=True),
}
