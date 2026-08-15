"""Q6_K quantization: port of quantize_row_q6_K_ref() / quantize_row_q6_K_impl() from ggml-quants.c.

Matches the quantize_q6_K() dispatch: make_qx_quants per 16-group with w = x*x (no imatrix)
or the raw imatrix weights, int8 scales picked from the max |scale|, requantize with 6-bit levels.
The search loop lives in qx_common (Triton on CUDA, eager torch on CPU/MPS).
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .qx_common import GROUP_MAX_EPS, absmax_first, make_qx_quants

QK_K = 256
BLOCK_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # ql + qh + int8 scales + fp16 d = 210


def quantize_q6_k(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 210) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    xsb = x.reshape(ns, QK_K)
    xg = xsb.reshape(ns * 16, 16)

    if qw is None:
        w = xg * xg
    else:
        w = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns * 16, 16)
    scales, L = make_qx_quants(xg, w, 32)

    sc16 = scales.reshape(ns, 16)
    amax, max_scale = absmax_first(sc16)
    live = amax >= GROUP_MAX_EPS  # dead superblocks zero out entirely in the C code
    iscale = -128.0 / torch.where(live, max_scale, torch.ones_like(max_scale))
    d16 = torch.where(live, 1.0 / iscale, torch.zeros_like(iscale)).to(torch.float16)
    # the C code stores MIN(127, nearest_int()) into int8; emulate the narrowing
    sc8 = torch.round(iscale[:, None] * sc16).clamp_(max=127.0).long()
    sc8 = ((sc8 & 0xFF) ^ 0x80) - 0x80

    # requantize with the int8 scales; d == 0 keeps the levels from the group fit like the C code
    dq = d16.to(torch.float32)[:, None] * sc8.float()
    use = dq != 0
    dsafe = torch.where(use, dq, torch.ones_like(dq))
    x16 = xsb.reshape(ns, 16, 16)
    lq = torch.round(x16 / dsafe[..., None]).clamp_(-32.0, 31.0) + 32.0
    Lfin = torch.where(use[..., None], lq, L.reshape(ns, 16, 16)).reshape(ns, QK_K).to(torch.int64)
    Lfin = torch.where(live[:, None], Lfin, torch.zeros_like(Lfin))
    sc8 = torch.where(live[:, None], sc8, torch.zeros_like(sc8))

    # pack quarters of each 128-half: ql holds low nibbles (q1|q3, q2|q4), qh the high 2 bits
    q4 = Lfin.reshape(ns, 2, 4, 32)
    q1, q2, q3, qq4 = q4[:, :, 0], q4[:, :, 1], q4[:, :, 2], q4[:, :, 3]
    ql = torch.cat([(q1 & 0xF) | ((q3 & 0xF) << 4), (q2 & 0xF) | ((qq4 & 0xF) << 4)], dim=-1)
    ql = ql.reshape(ns, QK_K // 2).to(torch.uint8)
    qh = ((q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((qq4 >> 4) << 6))
    qh = qh.reshape(ns, QK_K // 4).to(torch.uint8)
    scb = sc8.to(torch.int8).view(torch.uint8)

    out = torch.cat([ql, qh, scb, d16[:, None].view(torch.uint8)], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_q6_k(x, qw)


SPECS = {
    "q6_k": QuantSpec(GGMLQuantizationType.Q6_K, LlamaFileType.MOSTLY_Q6_K,
                      24, False, _make_kernel, uses_imatrix=True),
}
