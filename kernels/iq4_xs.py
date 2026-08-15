"""IQ4_XS quantization: port of quantize_iq4_xs() from ggml-quants.c.

Every superblock goes through quantize_row_iq4_nl_impl(super_block_size=256, block_size=32, kvalues_iq4nl, ntry=7):
a per-32-block least-squares scale search over the nonlinear LUT (best_index_int8), then the block scales
quantized to 6 bits (scales_l nibbles + scales_h 2-bit pairs) and the levels reprojected with the rounded scales.
The scale search runs in a Triton kernel on CUDA and eager torch elsewhere; reprojection and packing are shared torch ops.
Reductions are tree-ordered, so near-tie accepts can differ from the CPU reference at ULP level.
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import HAS_TRITON, QuantSpec, tl, triton

QK_K = 256
BLOCK_BYTES = 2 + 2 + QK_K // 64 + QK_K // 2  # fp16 d + scales_h + scales_l + 4-bit quants = 136
GROUP_MAX_EPS = 1e-15

KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

_vals_cache: dict[str, torch.Tensor] = {}


def _values_for(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _vals_cache:
        _vals_cache[key] = torch.tensor(KVALUES_IQ4NL, dtype=torch.float32, device=device)
    return _vals_cache[key]


def _best_index(al: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """best_index_int8(): bracket the ascending LUT, then the C midpoint compare."""
    mu = torch.searchsorted(vals, al.contiguous(), right=True).clamp_(1, 15)
    lo, hi = vals[mu - 1], vals[mu]
    return torch.where(al - lo < hi - al, mu - 1, mu)


def _seq_sum(t: torch.Tensor) -> torch.Tensor:
    """Left-to-right f32 sum over the last dim.
    Constant blocks make the accept test an exact tie, so the order must match C on every device."""
    cols = t.unbind(-1)
    acc = cols[0]
    for c in cols[1:]:
        acc = acc + c
    return acc


def _search_scales_eager(xb: torch.Tensor, w: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """Per-block scale of quantize_row_iq4_nl_impl: initial LS fit + ntry=7 sweep.
    xb, w: (G, 32); returns (G,) f32. Accepted d/best feed later steps, so the sweep is sequential."""
    imax = xb.abs().argmax(-1, keepdim=True)
    mx = xb.gather(1, imax).squeeze(1)               # signed value with the largest magnitude
    live = mx.abs() >= GROUP_MAX_EPS
    mxs = torch.where(live, mx, torch.ones_like(mx))

    d0 = -mxs / vals[0]
    q = vals[_best_index((1.0 / d0)[:, None] * xb, vals)]
    sumqx = _seq_sum(w * q * xb)
    sumq2 = _seq_sum(w * q * q)
    ok = sumq2 > 0
    d = torch.where(ok, sumqx / torch.where(ok, sumq2, torch.ones_like(sumq2)), torch.zeros_like(sumq2))
    best = d * sumqx
    for itry in range(-7, 8):
        idv = float(itry + KVALUES_IQ4NL[0]) / mxs
        q = vals[_best_index(idv[:, None] * xb, vals)]
        sumqx = _seq_sum(w * q * xb)
        sumq2 = _seq_sum(w * q * q)
        acc = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nd = sumqx / torch.where(acc, sumq2, torch.ones_like(sumq2))
        d = torch.where(acc, nd, d)
        best = torch.where(acc, nd * sumqx, best)
    return torch.where(live, d, torch.zeros_like(d))


# ---------------------------------------------------------------------------
# CUDA: Triton scale search
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _lut_project(al, v_ptr):
        """best_index_int8() -> LUT value: count vals <= al, then the C midpoint compare."""
        cnt = (al >= tl.load(v_ptr)).to(tl.int32)
        for i in tl.static_range(1, 16):
            cnt += (al >= tl.load(v_ptr + i)).to(tl.int32)
        mu = tl.minimum(tl.maximum(cnt, 1), 15)
        lo = tl.load(v_ptr + mu - 1)
        hi = tl.load(v_ptr + mu)
        return tl.where(al - lo < hi - al, lo, hi)

    @triton.jit
    def _seq_sums(tqx, tq2, BLOCK: tl.constexpr):
        """Left-to-right f32 sums over the 32 lanes.
        Constant blocks make the accept test an exact tie, so the order must match C on every device."""
        sumqx = tl.zeros((BLOCK,), tl.float32)
        sumq2 = tl.zeros((BLOCK,), tl.float32)
        for i in tl.static_range(32):
            lane = tl.arange(0, 32)[None, :] == i
            sumqx += tl.sum(tl.where(lane, tqx, 0.0), axis=1)
            sumq2 += tl.sum(tl.where(lane, tq2, 0.0), axis=1)
        return sumqx, sumq2

    @triton.jit
    def _scale_kernel(x_ptr, w_ptr, v_ptr, scale_ptr, G, BLOCK: tl.constexpr):
        """Per-block scale search, one lane-row per 32-value block.
        Accepted d/best feed later steps, so the sweep stays sequential."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        ax = tl.abs(x)
        imax = tl.argmax(ax, axis=1)                 # first max, like the C scan
        mx = tl.sum(tl.where(tl.arange(0, 32)[None, :] == imax[:, None], x, 0.0), axis=1)
        live = tl.max(ax, axis=1) >= 1e-15
        mxs = tl.where(live, mx, 1.0)

        d0 = -mxs / tl.load(v_ptr)                   # values[0] = -127
        q = _lut_project((1.0 / d0)[:, None] * x, v_ptr)
        sumqx, sumq2 = _seq_sums(w * q * x, w * q * q, BLOCK)
        ok = sumq2 > 0
        d = tl.where(ok, sumqx / tl.where(ok, sumq2, 1.0), 0.0)
        best = d * sumqx
        for itry in tl.static_range(-7, 8):
            idv = (itry - 127.0) / mxs               # (itry + values[0])/max
            q = _lut_project(idv[:, None] * x, v_ptr)
            sumqx, sumq2 = _seq_sums(w * q * x, w * q * q, BLOCK)
            acc = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            nd = sumqx / tl.where(acc, sumq2, 1.0)
            d = tl.where(acc, nd, d)
            best = tl.where(acc, nd * sumqx, best)
        tl.store(scale_ptr + g, tl.where(live, d, 0.0), mask=m)

    def _search_scales_triton(xb: torch.Tensor, w: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        G = xb.shape[0]
        scale = torch.empty(G, dtype=torch.float32, device=xb.device)
        BLOCK = 64
        _scale_kernel[((G + BLOCK - 1) // BLOCK,)](
            xb.contiguous(), w.contiguous(), vals, scale, G, BLOCK=BLOCK, num_warps=4)
        return scale


def quantize_iq4_xs(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 136) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    dev = x.device
    xsb = x.reshape(ns, QK_K)
    xb = xsb.reshape(ns * 8, 32)
    if qw is None:
        w = xb * xb
    else:
        sigma2 = 2.0 * (xsb * xsb).sum(-1) / QK_K
        qw_sb = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        w = (qw_sb * torch.sqrt(sigma2[:, None] + xsb * xsb)).reshape(ns * 8, 32)

    vals = _values_for(dev)
    search = _search_scales_triton if dev.type == "cuda" and HAS_TRITON else _search_scales_eager
    scales = search(xb, w, vals).reshape(ns, 8)

    # d = -max_scale/32; 6-bit block scales; reproject the levels with the rounded scales
    imax = scales.abs().argmax(-1, keepdim=True)     # first max abs, like the C scan
    max_scale = scales.gather(1, imax).squeeze(1)
    d = -max_scale / 32.0
    idq = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    ls = torch.round(idq[:, None] * scales).clamp_(-32.0, 31.0)
    dl = d[:, None] * ls
    idl = torch.where(dl != 0, 1.0 / dl, torch.zeros_like(dl))
    L = _best_index(idl[:, :, None] * xsb.reshape(ns, 8, 32), vals)

    # pack: scales_l nibble pairs, scales_h 2-bit pairs in a LE uint16, qs lo | hi << 4 per 16
    lsq = (ls + 32.0).to(torch.int64)                # 0..63
    scales_l = ((lsq[:, 0::2] & 0xF) | ((lsq[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    hbits = ((lsq >> 4) << (2 * torch.arange(8, device=dev))).sum(-1)
    scales_h = torch.stack([hbits & 0xFF, hbits >> 8], dim=1).to(torch.uint8)
    L2 = L.reshape(ns, 8, 2, 16)
    qs = (L2[:, :, 0, :] | (L2[:, :, 1, :] << 4)).to(torch.uint8).reshape(ns, QK_K // 2)

    d16 = d.to(torch.float16)
    out = torch.cat([d16[:, None].view(torch.uint8), scales_h, scales_l, qs], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_iq4_xs(x, qw)


SPECS = {
    "iq4_xs": QuantSpec(GGMLQuantizationType.IQ4_XS, LlamaFileType.MOSTLY_IQ4_XS,
                        24, False, _make_kernel, uses_imatrix=True),
}
