"""Q2_K quantization: port of quantize_row_q2_K_ref() / quantize_row_q2_K_impl() from ggml-quants.c.

Matches the quantize_q2_K() dispatch: without an imatrix the _ref path (make_qkx2_quants with
use_mad), with one the _impl path (make_qkx3_quants weights + make_qp_quants over the 16 group
scales/mins). The search loops live in kq_common (Triton on CUDA, eager torch on CPU/MPS).
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .kq_common import make_qkx_quants, make_qp_quants, qkx_steps, sdiv

QK_K = 256
BLOCK_BYTES = QK_K // 16 + QK_K // 4 + 2 + 2  # 4-bit scales/mins + 2-bit quants + fp16 d/dmin = 84

_STEPS_REF = qkx_steps(-0.5, 0.1, 15, 3)
_STEPS_IMPL = qkx_steps(-0.9, 0.05, 36, 3)


def quantize_q2_k(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 84) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    xsb = x.reshape(ns, QK_K)
    xg = xsb.reshape(ns * 16, 16)

    if qw is None:
        scales, mins, L = make_qkx_quants(xg, xg.abs(), _STEPS_REF, 3, use_mad=True)
        sc16 = scales.reshape(ns, 16)
        mn16 = mins.reshape(ns, 16)
        max_scale = sc16.max(-1).values
        max_min = mn16.max(-1).values
        pos_s = max_scale > 0
        pos_m = max_min > 0
        inv_scale = torch.where(pos_s, sdiv(15.0, max_scale), torch.zeros_like(max_scale))
        inv_min = torch.where(pos_m, sdiv(15.0, max_min), torch.zeros_like(max_min))
        # the C code stores nearest_int() straight into the uint8 scales byte, no MIN()
        ls = torch.round(inv_scale[:, None] * sc16).long() & 0xFF
        lm = torch.round(inv_min[:, None] * mn16).long() & 0xFF
        # C keeps d/dmin at +0 unless the max is strictly positive (avoids -0.0 from the max)
        zero = torch.zeros_like(max_scale)
        d16 = torch.where(pos_s, max_scale / 15.0, zero).to(torch.float16)
        m16 = torch.where(pos_m, max_min / 15.0, zero).to(torch.float16)
    else:
        sigma2 = (xsb * xsb).sum(-1) / QK_K
        qw_sb = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        w = (qw_sb * torch.sqrt(sigma2[:, None] + xsb * xsb)).reshape(ns * 16, 16)
        scales, mins, L = make_qkx_quants(xg, w, _STEPS_IMPL, 3)
        sw = w.sum(-1).reshape(ns, 16)
        d_blk, ls = make_qp_quants(scales.reshape(ns, 16), sw, 15)
        m_blk, lm = make_qp_quants(mins.reshape(ns, 16), sw, 15)
        ls = ls.long() & 0xFF
        lm = lm.long() & 0xFF
        d16 = d_blk.to(torch.float16)
        m16 = m_blk.to(torch.float16)

    scb = (ls | (lm << 4)) & 0xFF

    # requantize with the 4-bit group scales; d == 0 keeps the levels from make_qkx like the C code
    dg = d16.to(torch.float32)[:, None] * (scb & 0xF).float()
    mg = m16.to(torch.float32)[:, None] * (scb >> 4).float()
    use = dg != 0
    dsafe = torch.where(use, dg, torch.ones_like(dg))
    x16 = xsb.reshape(ns, 16, 16)
    lq = torch.round((x16 + mg[..., None]) / dsafe[..., None]).clamp_(0.0, 3.0)
    Lfin = torch.where(use[..., None], lq, L.reshape(ns, 16, 16)).to(torch.int64)

    # pack: per 128-value half, byte l holds levels l, l+32, l+64, l+96
    q4 = Lfin.reshape(ns, 2, 4, 32)
    qs = q4[:, :, 0] | (q4[:, :, 1] << 2) | (q4[:, :, 2] << 4) | (q4[:, :, 3] << 6)
    qs = qs.reshape(ns, QK_K // 4).to(torch.uint8)

    out = torch.cat([scb.to(torch.uint8), qs,
                     d16[:, None].view(torch.uint8), m16[:, None].view(torch.uint8)], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_q2_k(x, qw)


SPECS = {
    "q2_k": QuantSpec(GGMLQuantizationType.Q2_K, LlamaFileType.MOSTLY_Q2_K,
                      24, False, _make_kernel, uses_imatrix=True),
}
