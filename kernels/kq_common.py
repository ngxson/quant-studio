"""Shared search routines for the k-quant kernels (Q2_K .. Q6_K).

make_qkx_quants ports make_qkx2/qkx3_quants() and make_qp_quants ports make_qp_quants()
from ggml-quants.c, parameterized by group width and nmax.
Each has an eager torch version (CPU/MPS) and a Triton version (CUDA) that keeps the
whole candidate sweep in registers.
Reductions are tree-ordered, so near-tie candidate accepts can differ from the CPU
reference at ULP level.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import HAS_TRITON, F32_TINY, tl, tld, triton

GROUP_MAX_EPS = 1e-15


def qkx_steps(rmin: float, rdelta: float, nstep: int, nmax: int) -> np.ndarray:
    """Candidate iscale numerators (rmin + rdelta*is + nmax), rounded per step like the C fp32 chain."""
    return np.array([np.float32(rmin) + np.float32(rdelta) * np.float32(s) + np.float32(nmax)
                     for s in range(nstep + 1)], dtype=np.float32)


def qp_steps(nmax: int) -> np.ndarray:
    """make_qp_quants candidate numerators (0.1*is + nmax) for is in -4..4, is != 0."""
    return np.array([np.float32(0.1) * np.float32(s) + np.float32(nmax)
                     for s in range(-4, 5) if s != 0], dtype=np.float32)


# ---------------------------------------------------------------------------
# eager (CPU / MPS) - the reference implementation
# ---------------------------------------------------------------------------


def tree_sum(v: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair tree sum over the last dim (power of two).
    Fixed association shared with the Triton _tree_sum, so CPU and CUDA sums match bit-exactly."""
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    return v.squeeze(-1)


def sdiv(v: float, t: torch.Tensor) -> torch.Tensor:
    """True division scalar/tensor; python-scalar/tensor multiplies by the reciprocal instead."""
    return torch.scalar_tensor(v, dtype=t.dtype, device=t.device) / t


def make_qkx_quants_eager(x: torch.Tensor, w: torch.Tensor, steps: np.ndarray, nmax: int,
                          use_mad: bool = False):
    """make_qkx2/qkx3_quants(): weighted [min, scale] grid fit.
    x, w: (G, n); returns (scale, the_min, L) with L in 0..nmax.
    use_mad switches the error metric from w*diff^2 to w*|diff| like the C flag.
    An accepted min feeds the following candidate steps, so the sweep is sequential like the C loop."""
    fmax = float(nmax)
    mn = x.min(-1).values.clamp(max=0.0)
    mx = x.max(-1).values
    degen = mx == mn
    one = torch.ones_like(mx)
    sum_w = tree_sum(w)
    sum_x = tree_sum(w * x)

    iscale = sdiv(fmax, torch.where(degen, one, mx - mn))
    scale = 1.0 / iscale
    L = torch.round(iscale[:, None] * (x - mn[:, None])).clamp_(0.0, fmax)
    diff = scale[:, None] * L + mn[:, None] - x
    best = tree_sum(w * (diff.abs() if use_mad else diff * diff))

    for c in steps:
        isc = sdiv(float(c), torch.where(degen, one, mx - mn))
        Laux = torch.round(isc[:, None] * (x - mn[:, None])).clamp_(0.0, fmax)
        sum_l = tree_sum(w * Laux)
        sum_l2 = tree_sum(w * Laux * Laux)
        sum_xl = tree_sum(w * Laux * x)
        D = sum_w * sum_l2 - sum_l * sum_l
        Dsafe = torch.where(D > 0, D, one)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Dsafe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        pos = this_min > 0
        this_scale = torch.where(pos, sum_xl / sum_l2.clamp(min=F32_TINY), this_scale)
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)
        diff = this_scale[:, None] * Laux + this_min[:, None] - x
        cur = tree_sum(w * (diff.abs() if use_mad else diff * diff))
        acc = (D > 0) & (cur < best) & ~degen
        scale = torch.where(acc, this_scale, scale)
        mn = torch.where(acc, this_min, mn)
        best = torch.where(acc, cur, best)
        L = torch.where(acc[:, None], Laux, L)

    scale = torch.where(degen, torch.zeros_like(scale), scale)
    L = torch.where(degen[:, None], torch.zeros_like(L), L)
    # the_min = -min, but keep zero at +0.0: C compares mins with `> 0`, so -0.0 acts as +0.0
    the_min = torch.where(mn == 0, torch.zeros_like(mn), -mn)
    return scale, the_min, L


def make_qp_quants_eager(x: torch.Tensor, w: torch.Tensor, nmax: int):
    """make_qp_quants(): nonnegative grid fit for the superblock scale/min vectors.
    x, w: (B, n); returns (scale, L)."""
    fmax = float(nmax)
    mx = x.max(-1).values
    dead = mx < GROUP_MAX_EPS
    mxs = torch.where(dead, torch.ones_like(mx), mx)

    iscale = fmax / mxs
    L0 = torch.round(iscale[:, None] * x)                # first pass: no clamp in C
    best_mse = (w * (x - (1.0 / iscale)[:, None] * L0) ** 2).sum(-1)
    for c in qp_steps(nmax):
        isc = float(c) / mxs
        l = torch.clamp(torch.round(isc[:, None] * x), max=fmax)
        mse = (w * (x - (1.0 / isc)[:, None] * l) ** 2).sum(-1)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        iscale = torch.where(better, isc, iscale)

    L = torch.clamp(torch.round(iscale[:, None] * x), max=fmax)
    sumlx = (w * x * L).sum(-1)
    suml2 = (w * L * L).sum(-1)
    # extra passes after convergence are no-ops, so the C early-break does not change results
    xs, ws, Ls = x.unbind(1), w.unbind(1), list(L.unbind(1))
    for _ in range(5):
        for i in range(x.shape[1]):
            xi, wi, Li = xs[i], ws[i], Ls[i]
            slx = sumlx - wi * xi * Li
            sl2 = suml2 - wi * Li * Li
            ok = (slx > 0) & (sl2 > 0)
            newl = torch.clamp(torch.round(xi * sl2 / torch.where(ok, slx, torch.ones_like(slx))),
                               max=fmax)
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
# CUDA: Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _pair_add(v, BLOCK: tl.constexpr, M: tl.constexpr):
        a, b = tl.split(tl.reshape(v, (BLOCK, M, 2)))
        return tld.add_rn(a, b)

    @triton.jit
    def _tree_sum(v, BLOCK: tl.constexpr, NL: tl.constexpr):
        """Adjacent-pair tree sum over axis 1 (NL a power of two, 8..64).
        Same association as the eager tree_sum; add_rn blocks ptxas fma contraction,
        so CPU and CUDA sums match bit-exactly."""
        tl.static_assert(NL == 8 or NL == 16 or NL == 32 or NL == 64)
        if NL == 64:
            v = _pair_add(v, BLOCK, 32)
        if NL >= 32:
            v = _pair_add(v, BLOCK, 16)
        if NL >= 16:
            v = _pair_add(v, BLOCK, 8)
        v = _pair_add(v, BLOCK, 4)
        v = _pair_add(v, BLOCK, 2)
        v = _pair_add(v, BLOCK, 1)
        return tl.reshape(v, (BLOCK,))

    @triton.jit
    def _qkx_kernel(x_ptr, w_ptr, c_ptr, scale_ptr, min_ptr, l_ptr, G,
                    NL: tl.constexpr, NMAX: tl.constexpr, NSTEP: tl.constexpr,
                    USE_MAD: tl.constexpr, BLOCK: tl.constexpr):
        """make_qkx2/qkx3_quants(), one lane-row per group.
        USE_MAD switches the error metric from w*diff^2 to w*|diff| like the C flag.
        An accepted min feeds the following candidate steps, so the sweep stays sequential."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        offs = g[:, None] * NL + tl.arange(0, NL)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        mn = tl.minimum(tl.min(x, axis=1), 0.0)
        mx = tl.max(x, axis=1)
        live = mx != mn
        sum_w = _tree_sum(w, BLOCK, NL)
        sum_x = _tree_sum(w * x, BLOCK, NL)

        # div_rn forces ieee division (default f32 div is approximate);
        # add_rn/sub_rn on the mul+add sites keeps ptxas from fusing them into fma
        one = tl.full((1,), 1.0, tl.float32)
        iscale = tld.div_rn(NMAX * one, tl.where(live, mx - mn, 1.0))
        scale = tld.div_rn(one, iscale)
        L = tl.maximum(tl.minimum(tld.rint(iscale[:, None] * (x - mn[:, None])), NMAX), 0.0)
        diff = tld.add_rn(scale[:, None] * L, mn[:, None]) - x
        if USE_MAD:
            best = _tree_sum(w * tl.abs(diff), BLOCK, NL)
        else:
            best = _tree_sum(w * diff * diff, BLOCK, NL)

        for s in tl.static_range(0, NSTEP + 1):
            c = tl.load(c_ptr + s)
            isc = tld.div_rn(c * one, tl.where(live, mx - mn, 1.0))
            La = tl.maximum(tl.minimum(tld.rint(isc[:, None] * (x - mn[:, None])), NMAX), 0.0)
            sum_l = _tree_sum(w * La, BLOCK, NL)
            sum_l2 = _tree_sum(w * La * La, BLOCK, NL)
            sum_xl = _tree_sum(w * La * x, BLOCK, NL)
            D = tld.sub_rn(sum_w * sum_l2, sum_l * sum_l)
            ok = D > 0
            Dsafe = tl.where(ok, D, 1.0)
            ts = tld.div_rn(tld.sub_rn(sum_w * sum_xl, sum_x * sum_l), Dsafe)
            tm = tld.div_rn(tld.sub_rn(sum_l2 * sum_x, sum_l * sum_xl), Dsafe)
            pos = tm > 0
            ts = tl.where(pos, tld.div_rn(sum_xl, tl.where(sum_l2 > 0, sum_l2, 1.0)), ts)
            tm = tl.where(pos, 0.0, tm)
            diff2 = tld.add_rn(ts[:, None] * La, tm[:, None]) - x
            if USE_MAD:
                cur = _tree_sum(w * tl.abs(diff2), BLOCK, NL)
            else:
                cur = _tree_sum(w * diff2 * diff2, BLOCK, NL)
            acc = ok & (cur < best) & live
            scale = tl.where(acc, ts, scale)
            mn = tl.where(acc, tm, mn)
            best = tl.where(acc, cur, best)
            L = tl.where(acc[:, None], La, L)

        tl.store(scale_ptr + g, tl.where(live, scale, 0.0), mask=m)
        tl.store(min_ptr + g, tl.where(mn == 0, 0.0, -mn), mask=m)
        tl.store(l_ptr + offs, tl.where(live[:, None], L, 0.0), mask=m[:, None])

    @triton.jit
    def _qp_kernel(x_ptr, w_ptr, c_ptr, eps_ptr, scale_ptr, l_ptr, B,
                   NL: tl.constexpr, NMAX: tl.constexpr, BLOCK: tl.constexpr):
        """make_qp_quants() over the per-superblock scale/min vectors."""
        b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = b < B
        offs = b[:, None] * NL + tl.arange(0, NL)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        mx = tl.max(x, axis=1)
        dead = mx < tl.load(eps_ptr)
        mxs = tl.where(dead, 1.0, mx)
        iscale = NMAX / mxs
        L0 = tld.rint(iscale[:, None] * x)               # first pass: no clamp in C
        df = x - (1.0 / iscale)[:, None] * L0
        best = tl.sum(w * df * df, axis=1)
        for s in tl.static_range(0, 8):
            c = tl.load(c_ptr + s)
            isc = c / mxs
            l = tl.minimum(tld.rint(isc[:, None] * x), NMAX)
            df2 = x - (1.0 / isc)[:, None] * l
            mse = tl.sum(w * df2 * df2, axis=1)
            better = mse < best
            best = tl.where(better, mse, best)
            iscale = tl.where(better, isc, iscale)

        L = tl.minimum(tld.rint(iscale[:, None] * x), NMAX)
        sumlx = tl.sum(w * x * L, axis=1)
        suml2 = tl.sum(w * L * L, axis=1)
        for _p in range(5):
            for i in tl.static_range(NL):
                lane = tl.arange(0, NL)[None, :] == i
                xi = tl.sum(tl.where(lane, x, 0.0), axis=1)
                wi = tl.sum(tl.where(lane, w, 0.0), axis=1)
                Li = tl.sum(tl.where(lane, L, 0.0), axis=1)
                slx = sumlx - wi * xi * Li
                sl2 = suml2 - wi * Li * Li
                ok = (slx > 0) & (sl2 > 0)
                newl = tl.minimum(tld.rint(xi * sl2 / tl.where(ok, slx, 1.0)), NMAX)
                slx2 = slx + wi * xi * newl
                sl22 = sl2 + wi * newl * newl
                commit = ok & (newl != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
                L = tl.where(commit[:, None] & lane, newl[:, None], L)
                sumlx = tl.where(commit, slx2, sumlx)
                suml2 = tl.where(commit, sl22, suml2)

        scale = tl.where(suml2 > 0, sumlx / tl.where(suml2 > 0, suml2, 1.0), 0.0)
        tl.store(scale_ptr + b, tl.where(dead, 0.0, scale), mask=m)
        tl.store(l_ptr + offs, tl.where(dead[:, None], 0.0, L), mask=m[:, None])

    _const_cache: dict = {}

    def _const_for(device: torch.device, arr: np.ndarray) -> torch.Tensor:
        key = (str(device), arr.tobytes())
        if key not in _const_cache:
            _const_cache[key] = torch.from_numpy(arr).to(device)
        return _const_cache[key]

    _EPS_ARR = np.array([GROUP_MAX_EPS], dtype=np.float32)

    def make_qkx_quants_triton(x: torch.Tensor, w: torch.Tensor, steps: np.ndarray, nmax: int,
                               use_mad: bool = False):
        G, n = x.shape
        c = _const_for(x.device, steps)
        scale = torch.empty(G, dtype=torch.float32, device=x.device)
        the_min = torch.empty(G, dtype=torch.float32, device=x.device)
        L = torch.empty(G, n, dtype=torch.float32, device=x.device)
        BLOCK = 64
        _qkx_kernel[((G + BLOCK - 1) // BLOCK,)](
            x.contiguous(), w.contiguous(), c, scale, the_min, L, G,
            NL=n, NMAX=float(nmax), NSTEP=len(steps) - 1, USE_MAD=use_mad,
            BLOCK=BLOCK, num_warps=4)
        return scale, the_min, L

    def make_qp_quants_triton(x: torch.Tensor, w: torch.Tensor, nmax: int):
        B, n = x.shape
        c = _const_for(x.device, qp_steps(nmax))
        eps = _const_for(x.device, _EPS_ARR)
        scale = torch.empty(B, dtype=torch.float32, device=x.device)
        L = torch.empty(B, n, dtype=torch.float32, device=x.device)
        BLOCK = 128
        _qp_kernel[((B + BLOCK - 1) // BLOCK,)](
            x.contiguous(), w.contiguous(), c, eps, scale, L, B,
            NL=n, NMAX=float(nmax), BLOCK=BLOCK, num_warps=4)
        return scale, L


def use_triton(x: torch.Tensor) -> bool:
    return x.device.type == "cuda" and HAS_TRITON


def make_qkx_quants(x, w, steps, nmax, use_mad=False):
    fn = make_qkx_quants_triton if use_triton(x) else make_qkx_quants_eager
    return fn(x, w, steps, nmax, use_mad)


def make_qp_quants(x, w, nmax):
    fn = make_qp_quants_triton if use_triton(x) else make_qp_quants_eager
    return fn(x, w, nmax)
