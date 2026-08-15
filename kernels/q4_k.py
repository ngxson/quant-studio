"""Q4_K quantization: port of quantize_row_q4_K_ref() / quantize_row_q4_K_impl() from ggml-quants.c.

Matches the quantize_q4_K() dispatch: without an imatrix the _ref path (make_qkx2_quants),
with one the _impl path (make_qkx3_quants weights + make_qp_quants over the 8 sub-block scales/mins).
Reductions are tree-ordered, so near-tie candidate accepts can differ from the CPU reference at ULP level, same as the iq2 port.
On CUDA the two search loops are Triton kernels that keep the whole candidate sweep in registers; CPU/MPS use the eager torch port.
"""

from __future__ import annotations

import numpy as np
import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import HAS_TRITON, F32_TINY, QuantSpec, tl, tld, triton

QK_K = 256
GROUP_MAX_EPS = 1e-15
BLOCK_BYTES = 2 + 2 + 12 + QK_K // 2  # fp16 d + fp16 dmin + 6-bit scales + 4-bit quants = 144

# (rmin, rdelta, nstep) for the two make_qkx variants
_QKX_PARAMS = {"ref": (-1.0, 0.1, 20), "impl": (-0.9, 0.05, 36)}


def _qkx_steps(variant: str) -> np.ndarray:
    """Candidate iscale numerators (rmin + rdelta*is + nmax), rounded per step like the C fp32 chain."""
    rmin, rdelta, nstep = _QKX_PARAMS[variant]
    return np.array([np.float32(rmin) + np.float32(rdelta) * np.float32(s) + np.float32(15)
                     for s in range(nstep + 1)], dtype=np.float32)


def _qp_steps() -> np.ndarray:
    """make_qp_quants candidate numerators (0.1*is + nmax) for is in -4..4, is != 0."""
    return np.array([np.float32(0.1) * np.float32(s) + np.float32(63)
                     for s in range(-4, 5) if s != 0], dtype=np.float32)


# ---------------------------------------------------------------------------
# CUDA: Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _qkx_kernel(x_ptr, w_ptr, c_ptr, scale_ptr, min_ptr, l_ptr, G,
                    NSTEP: tl.constexpr, BLOCK: tl.constexpr):
        """make_qkx2/qkx3_quants(n=32, nmax=15, use_mad=false), one lane-row per group.
        An accepted min feeds the following candidate steps, so the sweep stays sequential."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        mn = tl.minimum(tl.min(x, axis=1), 0.0)
        mx = tl.max(x, axis=1)
        live = mx != mn
        sum_w = tl.sum(w, axis=1)
        sum_x = tl.sum(w * x, axis=1)

        iscale = 15.0 / tl.where(live, mx - mn, 1.0)
        scale = 1.0 / iscale
        L = tl.maximum(tl.minimum(tld.rint(iscale[:, None] * (x - mn[:, None])), 15.0), 0.0)
        diff = scale[:, None] * L + mn[:, None] - x
        best = tl.sum(w * diff * diff, axis=1)

        for s in tl.static_range(0, NSTEP + 1):
            c = tl.load(c_ptr + s)
            isc = c / tl.where(live, mx - mn, 1.0)
            La = tl.maximum(tl.minimum(tld.rint(isc[:, None] * (x - mn[:, None])), 15.0), 0.0)
            sum_l = tl.sum(w * La, axis=1)
            sum_l2 = tl.sum(w * La * La, axis=1)
            sum_xl = tl.sum(w * La * x, axis=1)
            D = sum_w * sum_l2 - sum_l * sum_l
            ok = D > 0
            ts = (sum_w * sum_xl - sum_x * sum_l) / tl.where(ok, D, 1.0)
            tm = (sum_l2 * sum_x - sum_l * sum_xl) / tl.where(ok, D, 1.0)
            pos = tm > 0
            ts = tl.where(pos, sum_xl / tl.where(sum_l2 > 0, sum_l2, 1.0), ts)
            tm = tl.where(pos, 0.0, tm)
            diff2 = ts[:, None] * La + tm[:, None] - x
            cur = tl.sum(w * diff2 * diff2, axis=1)
            acc = ok & (cur < best) & live
            scale = tl.where(acc, ts, scale)
            mn = tl.where(acc, tm, mn)
            best = tl.where(acc, cur, best)
            L = tl.where(acc[:, None], La, L)

        tl.store(scale_ptr + g, tl.where(live, scale, 0.0), mask=m)
        tl.store(min_ptr + g, -mn, mask=m)
        tl.store(l_ptr + offs, tl.where(live[:, None], L, 0.0), mask=m[:, None])

    @triton.jit
    def _qp8_kernel(x_ptr, w_ptr, c_ptr, eps_ptr, scale_ptr, l_ptr, B, BLOCK: tl.constexpr):
        """make_qp_quants(n=8, nmax=63) over the per-superblock scale/min vectors."""
        b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = b < B
        offs = b[:, None] * 8 + tl.arange(0, 8)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        mx = tl.max(x, axis=1)
        dead = mx < tl.load(eps_ptr)
        mxs = tl.where(dead, 1.0, mx)
        iscale = 63.0 / mxs
        L0 = tld.rint(iscale[:, None] * x)               # first pass: no clamp in C
        df = x - (1.0 / iscale)[:, None] * L0
        best = tl.sum(w * df * df, axis=1)
        for s in tl.static_range(0, 8):
            c = tl.load(c_ptr + s)
            isc = c / mxs
            l = tl.minimum(tld.rint(isc[:, None] * x), 63.0)
            df2 = x - (1.0 / isc)[:, None] * l
            mse = tl.sum(w * df2 * df2, axis=1)
            better = mse < best
            best = tl.where(better, mse, best)
            iscale = tl.where(better, isc, iscale)

        L = tl.minimum(tld.rint(iscale[:, None] * x), 63.0)
        sumlx = tl.sum(w * x * L, axis=1)
        suml2 = tl.sum(w * L * L, axis=1)
        for _p in range(5):
            for i in tl.static_range(8):
                lane = tl.arange(0, 8)[None, :] == i
                xi = tl.sum(tl.where(lane, x, 0.0), axis=1)
                wi = tl.sum(tl.where(lane, w, 0.0), axis=1)
                Li = tl.sum(tl.where(lane, L, 0.0), axis=1)
                slx = sumlx - wi * xi * Li
                sl2 = suml2 - wi * Li * Li
                ok = (slx > 0) & (sl2 > 0)
                newl = tl.minimum(tld.rint(xi * sl2 / tl.where(ok, slx, 1.0)), 63.0)
                slx2 = slx + wi * xi * newl
                sl22 = sl2 + wi * newl * newl
                commit = ok & (newl != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
                L = tl.where(commit[:, None] & lane, newl[:, None], L)
                sumlx = tl.where(commit, slx2, sumlx)
                suml2 = tl.where(commit, sl22, suml2)

        scale = tl.where(suml2 > 0, sumlx / tl.where(suml2 > 0, suml2, 1.0), 0.0)
        tl.store(scale_ptr + b, tl.where(dead, 0.0, scale), mask=m)
        tl.store(l_ptr + offs, tl.where(dead[:, None], 0.0, L), mask=m[:, None])

    class _Consts:
        def __init__(self, device: torch.device):
            self.qkx = {v: torch.from_numpy(_qkx_steps(v)).to(device) for v in _QKX_PARAMS}
            self.qp = torch.from_numpy(_qp_steps()).to(device)
            self.eps = torch.tensor([GROUP_MAX_EPS], dtype=torch.float32, device=device)

    _consts_cache: dict[str, _Consts] = {}

    def _consts_for(device: torch.device) -> _Consts:
        key = str(device)
        if key not in _consts_cache:
            _consts_cache[key] = _Consts(device)
        return _consts_cache[key]

    def _qkx_triton(x: torch.Tensor, w: torch.Tensor, variant: str):
        G = x.shape[0]
        C = _consts_for(x.device)
        scale = torch.empty(G, dtype=torch.float32, device=x.device)
        the_min = torch.empty(G, dtype=torch.float32, device=x.device)
        L = torch.empty(G, 32, dtype=torch.float32, device=x.device)
        BLOCK = 64
        _qkx_kernel[((G + BLOCK - 1) // BLOCK,)](
            x, w, C.qkx[variant], scale, the_min, L, G,
            NSTEP=_QKX_PARAMS[variant][2], BLOCK=BLOCK, num_warps=4)
        return scale, the_min, L

    def _qp_triton(x: torch.Tensor, w: torch.Tensor):
        B = x.shape[0]
        C = _consts_for(x.device)
        scale = torch.empty(B, dtype=torch.float32, device=x.device)
        L = torch.empty(B, 8, dtype=torch.float32, device=x.device)
        BLOCK = 128
        _qp8_kernel[((B + BLOCK - 1) // BLOCK,)](
            x.contiguous(), w.contiguous(), C.qp, C.eps, scale, L, B,
            BLOCK=BLOCK, num_warps=4)
        return scale, L


# ---------------------------------------------------------------------------
# eager fallback (CPU / MPS) - the reference implementation
# ---------------------------------------------------------------------------


def _make_qkx_quants(x: torch.Tensor, w: torch.Tensor, variant: str):
    """make_qkx2/qkx3_quants(n=32, nmax=15, use_mad=false): weighted [min, scale] grid fit.
    x, w: (G, 32); returns (scale, the_min, L) with L in 0..15.
    An accepted min feeds the following candidate steps, so the sweep is sequential like the C loop."""
    nmax = 15.0
    mn = x.min(-1).values.clamp(max=0.0)
    mx = x.max(-1).values
    degen = mx == mn
    one = torch.ones_like(mx)
    sum_w = w.sum(-1)
    sum_x = (w * x).sum(-1)

    iscale = nmax / torch.where(degen, one, mx - mn)
    scale = 1.0 / iscale
    L = torch.round(iscale[:, None] * (x - mn[:, None])).clamp_(0.0, nmax)
    diff = scale[:, None] * L + mn[:, None] - x
    best = (w * diff * diff).sum(-1)

    for c in _qkx_steps(variant):
        isc = float(c) / torch.where(degen, one, mx - mn)
        Laux = torch.round(isc[:, None] * (x - mn[:, None])).clamp_(0.0, nmax)
        sum_l = (w * Laux).sum(-1)
        sum_l2 = (w * Laux * Laux).sum(-1)
        sum_xl = (w * Laux * x).sum(-1)
        D = sum_w * sum_l2 - sum_l * sum_l
        Dsafe = torch.where(D > 0, D, one)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Dsafe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        pos = this_min > 0
        this_scale = torch.where(pos, sum_xl / sum_l2.clamp(min=F32_TINY), this_scale)
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)
        diff = this_scale[:, None] * Laux + this_min[:, None] - x
        cur = (w * diff * diff).sum(-1)
        acc = (D > 0) & (cur < best) & ~degen
        scale = torch.where(acc, this_scale, scale)
        mn = torch.where(acc, this_min, mn)
        best = torch.where(acc, cur, best)
        L = torch.where(acc[:, None], Laux, L)

    scale = torch.where(degen, torch.zeros_like(scale), scale)
    L = torch.where(degen[:, None], torch.zeros_like(L), L)
    return scale, -mn, L


def _make_qp_quants(x: torch.Tensor, w: torch.Tensor):
    """make_qp_quants(n=8, nmax=63): nonnegative grid fit for the superblock scale/min vectors.
    x, w: (B, 8); returns (scale, L)."""
    nmax = 63.0
    mx = x.max(-1).values
    dead = mx < GROUP_MAX_EPS
    mxs = torch.where(dead, torch.ones_like(mx), mx)

    iscale = nmax / mxs
    L0 = torch.round(iscale[:, None] * x)                # first pass: no clamp in C
    best_mse = (w * (x - (1.0 / iscale)[:, None] * L0) ** 2).sum(-1)
    for c in _qp_steps():
        isc = float(c) / mxs
        l = torch.clamp(torch.round(isc[:, None] * x), max=nmax)
        mse = (w * (x - (1.0 / isc)[:, None] * l) ** 2).sum(-1)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        iscale = torch.where(better, isc, iscale)

    L = torch.clamp(torch.round(iscale[:, None] * x), max=nmax)
    sumlx = (w * x * L).sum(-1)
    suml2 = (w * L * L).sum(-1)
    # extra passes after convergence are no-ops, so the C early-break does not change results
    xs, ws, Ls = x.unbind(1), w.unbind(1), list(L.unbind(1))
    for _ in range(5):
        for i in range(8):
            xi, wi, Li = xs[i], ws[i], Ls[i]
            slx = sumlx - wi * xi * Li
            sl2 = suml2 - wi * Li * Li
            ok = (slx > 0) & (sl2 > 0)
            newl = torch.clamp(torch.round(xi * sl2 / torch.where(ok, slx, torch.ones_like(slx))),
                               max=nmax)
            slx2 = slx + wi * xi * newl
            sl22 = sl2 + wi * newl * newl
            commit = ok & (newl != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            Ls[i] = torch.where(commit, newl, Li)
            sumlx = torch.where(commit, slx2, sumlx)
            suml2 = torch.where(commit, sl22, suml2)
    L = torch.stack(Ls, dim=1)
    scale = torch.where(suml2 > 0, sumlx / suml2.clamp(min=F32_TINY), torch.zeros_like(sumlx))
    scale = torch.where(dead, torch.zeros_like(scale), scale)
    L = torch.where(dead[:, None], torch.zeros_like(L), L)
    return scale, L


# ---------------------------------------------------------------------------
# shared orchestration
# ---------------------------------------------------------------------------


def quantize_q4_k(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 144) uint8 on the same device."""
    n, k = x.shape
    ns = n * k // QK_K
    fast = x.device.type == "cuda" and HAS_TRITON
    qkx = _qkx_triton if fast else _make_qkx_quants
    qp = _qp_triton if fast else _make_qp_quants
    xsb = x.reshape(ns, QK_K)
    xg = xsb.reshape(ns * 8, 32)

    if qw is None:
        av_x = torch.sqrt((xg * xg).sum(-1) / 32)
        w = av_x[:, None] + xg.abs()
        scales, mins, L = qkx(xg, w, "ref")
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
        scales, mins, L = qkx(xg, w, "impl")
        sw = w.sum(-1).reshape(ns, 8)
        d_blk, ls = qp(scales.reshape(ns, 8), sw)
        m_blk, lm = qp(mins.reshape(ns, 8), sw)
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
    lq = torch.round((x8 + mg[..., None]) / dsafe[..., None]).clamp_(0.0, 15.0)
    Lfin = torch.where(use[..., None], lq, L.reshape(ns, 8, 32)).to(torch.int64)

    # pack: qs pairs sub-blocks (2j, 2j+1) as lo | hi << 4; scales/mins use the split 6-bit layout
    q2 = Lfin.reshape(ns, 4, 2, 32)
    qs = (q2[:, :, 0, :] | (q2[:, :, 1, :] << 4)).to(torch.uint8).reshape(ns, QK_K // 2)
    b03 = ls[:, :4] | ((ls[:, 4:] >> 4) << 6)
    b47 = lm[:, :4] | ((lm[:, 4:] >> 4) << 6)
    b8b = (ls[:, 4:] & 0xF) | ((lm[:, 4:] & 0xF) << 4)
    scb = torch.cat([b03, b47, b8b], dim=1).to(torch.uint8)

    out = torch.cat([d16[:, None].view(torch.uint8), m16[:, None].view(torch.uint8), scb, qs], dim=1)
    return out.reshape(n, k // QK_K * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    return lambda x: quantize_q4_k(x, qw)


SPECS = {
    "q4_k": QuantSpec(GGMLQuantizationType.Q4_K, LlamaFileType.MOSTLY_Q4_K_M,
                      24, False, _make_kernel, uses_imatrix=True),
}
