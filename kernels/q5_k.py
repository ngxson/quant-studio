"""Q5_K quantization: port of quantize_row_q5_K_ref() / quantize_row_q5_K_impl() from ggml-quants.c.

Matches the quantize_q5_K() dispatch: without an imatrix the _ref path (make_qkx2_quants),
with one the _impl path (make_qkx3_quants weights + make_qp_quants over the 8 sub-block scales/mins).
The search loops live in kq_common (Triton on CUDA, eager torch on CPU/MPS).
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .kq_common import make_qkx_quants, make_qp_quants, qkx_steps

QK_K = 256
BLOCK_BYTES = 2 + 2 + 12 + QK_K // 8 + QK_K // 2  # fp16 d/dmin + 6-bit scales + qh + qs = 176

_STEPS_REF = qkx_steps(-0.5, 0.1, 15, 31)
_STEPS_IMPL = qkx_steps(-0.9, 0.05, 36, 31)


def quantize_q5_k(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 176) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    xsb = x.reshape(ns, QK_K)
    xg = xsb.reshape(ns * 8, 32)

    if qw is None:
        av_x = torch.sqrt((xg * xg).sum(-1) / 32)
        w = av_x[:, None] + xg.abs()
        scales, mins, L = make_qkx_quants(xg, w, _STEPS_REF, 31)
        sc8 = scales.reshape(ns, 8)
        mn8 = mins.reshape(ns, 8)
        max_scale = sc8.max(-1).values
        max_min = mn8.max(-1).values
        inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
        inv_min = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
        # the C code narrows nearest_int() to uint8 before MIN(63, ...)
        ls = (torch.round(inv_scale[:, None] * sc8).long() & 0xFF).clamp(max=63)
        lm = (torch.round(inv_min[:, None] * mn8).long() & 0xFF).clamp(max=63)
        d16 = (max_scale / 63.0).to(torch.float16)
        m16 = (max_min / 63.0).to(torch.float16)
    else:
        sigma2 = 2.0 * (xsb * xsb).sum(-1) / QK_K
        qw_sb = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        w = (qw_sb * torch.sqrt(sigma2[:, None] + xsb * xsb)).reshape(ns * 8, 32)
        scales, mins, L = make_qkx_quants(xg, w, _STEPS_IMPL, 31)
        sw = w.sum(-1).reshape(ns, 8)
        d_blk, ls = make_qp_quants(scales.reshape(ns, 8), sw, 63)
        m_blk, lm = make_qp_quants(mins.reshape(ns, 8), sw, 63)
        ls = ls.long() & 0xFF
        lm = lm.long() & 0xFF
        d16 = d_blk.to(torch.float16)
        m16 = m_blk.to(torch.float16)

    # requantize with the 6-bit block scales; d == 0 keeps the levels from make_qkx like the C code
    dg = d16.to(torch.float32)[:, None] * (ls & 63).float()
    mg = m16.to(torch.float32)[:, None] * (lm & 63).float()
    use = dg != 0
    dsafe = torch.where(use, dg, torch.ones_like(dg))
    x8 = xsb.reshape(ns, 8, 32)
    lq = torch.round((x8 + mg[..., None]) / dsafe[..., None]).clamp_(0.0, 31.0)
    Lfin = torch.where(use[..., None], lq, L.reshape(ns, 8, 32)).to(torch.int64)

    # pack: qs pairs sub-blocks (2j, 2j+1) as lo4 | hi4 << 4; qh[j] holds bit 4 of sub-block s at bit s
    q2 = Lfin.reshape(ns, 4, 2, 32)
    qs = ((q2[:, :, 0, :] & 0xF) | ((q2[:, :, 1, :] & 0xF) << 4)).to(torch.uint8).reshape(ns, QK_K // 2)
    hb = Lfin >> 4
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    qh = (hb << shifts[None, :, None]).sum(dim=1).to(torch.uint8)
    b03 = ls[:, :4] | ((ls[:, 4:] >> 4) << 6)
    b47 = lm[:, :4] | ((lm[:, 4:] >> 4) << 6)
    b8b = (ls[:, 4:] & 0xF) | ((lm[:, 4:] & 0xF) << 4)
    scb = torch.cat([b03, b47, b8b], dim=1).to(torch.uint8)

    out = torch.cat([d16[:, None].view(torch.uint8), m16[:, None].view(torch.uint8), scb, qh, qs], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_q5_k(x, qw)


SPECS = {
    "q5_k": QuantSpec(GGMLQuantizationType.Q5_K, LlamaFileType.MOSTLY_Q5_K_M,
                      24, False, _make_kernel, uses_imatrix=True),
}
