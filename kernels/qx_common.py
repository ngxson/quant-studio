"""Shared search routines for the sign-symmetric k-quants (Q3_K, Q6_K).

make_qx_quants ports make_qx_quants(rmse_type=1, with/without qw) and make_q3_quants
ports make_q3_quants(do_rmse=true) from ggml-quants.c, parameterized by nmax.
Each has an eager torch version (CPU/MPS) and a Triton version (CUDA).
Sums accumulate in the sequential C order and Triton divides with div_rn, so the
eager and Triton paths stay bit-identical to each other on identical inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import HAS_TRITON, tl, tld, triton

GROUP_MAX_EPS = 1e-15


def qx_steps(nmax: int) -> np.ndarray:
    """Candidate iscale numerators (nmax + 0.1*is) for is in -9..9, is != 0, rounded like the C fp32 chain."""
    return np.array([np.float32(nmax) + np.float32(0.1) * np.float32(s)
                     for s in range(-9, 10) if s != 0], dtype=np.float32)


def absmax_first(x: torch.Tensor):
    """Per-row value with the largest |x|, first hit winning like the strict C scan.
    Returns (amax, max)."""
    ax = x.abs()
    amax = ax.max(-1).values
    ar = torch.arange(x.shape[-1], device=x.device)
    first = torch.where(ax == amax[:, None], ar, x.shape[-1]).min(-1).values
    mx = x.gather(-1, first[:, None]).squeeze(-1)
    return amax, mx


def seq_sum(t: torch.Tensor) -> torch.Tensor:
    """Strict left-to-right sum over the last dim, matching the sequential C accumulation.
    Also keeps CPU and CUDA bit-identical, unlike the tree-ordered .sum()."""
    parts = t.unbind(-1)
    s = parts[0]
    for p in parts[1:]:
        s = s + p
    return s


# ---------------------------------------------------------------------------
# eager (CPU / MPS) - the reference implementation
# ---------------------------------------------------------------------------


def make_qx_quants_eager(x: torch.Tensor, w: torch.Tensor, nmax: int):
    """make_qx_quants(rmse_type=1): signed grid fit with an 18-candidate iscale sweep.
    x, w: (G, n); returns (scale, L) with L in 0..2*nmax-1 (0 for all-zero groups).
    An accept updates best, which gates later candidates, so the sweep stays sequential."""
    fmax = float(nmax)
    amax, mx = absmax_first(x)
    live = amax >= GROUP_MAX_EPS
    one = torch.ones_like(mx)
    mxs = torch.where(live, mx, one)

    iscale = -fmax / mxs
    L = torch.round(iscale[:, None] * x).clamp_(-fmax, fmax - 1.0)
    sumlx = seq_sum(w * x * L)
    suml2 = seq_sum(w * L * L)
    scale = torch.where(suml2 != 0, sumlx / torch.where(suml2 != 0, suml2, one),
                        torch.zeros_like(sumlx))
    best = scale * sumlx
    for c in qx_steps(nmax):
        isc = -float(c) / mxs
        La = torch.round(isc[:, None] * x).clamp_(-fmax, fmax - 1.0)
        slx = seq_sum(w * x * La)
        sl2 = seq_sum(w * La * La)
        acc = (sl2 > 0) & (slx * slx > best * sl2)
        cand = slx / torch.where(acc, sl2, one)
        scale = torch.where(acc, cand, scale)
        best = torch.where(acc, cand * slx, best)
        L = torch.where(acc[:, None], La, L)

    scale = torch.where(live, scale, torch.zeros_like(scale))
    L = torch.where(live[:, None], L + fmax, torch.zeros_like(L))
    return scale, L


def make_q3_quants_eager(x: torch.Tensor, nmax: int):
    """make_q3_quants(do_rmse=true): signed fit with 5 greedy per-lane improvement passes.
    x: (G, n); returns (scale, L) with L in 0..2*nmax-1 (0 for all-zero groups).
    Commits chain through sumlx/suml2, so the lane loop stays sequential like the C code."""
    fmax = float(nmax)
    w = x * x
    amax, mx = absmax_first(x)
    live = amax >= GROUP_MAX_EPS
    one = torch.ones_like(mx)
    mxs = torch.where(live, mx, one)

    iscale = -fmax / mxs
    L0 = torch.round(iscale[:, None] * x).clamp_(-fmax, fmax - 1.0)
    sumlx = seq_sum(w * x * L0)
    suml2 = seq_sum(w * L0 * L0)
    # extra passes after convergence are no-ops, so the C early-break does not change results
    xs, ws, Ls = x.unbind(1), w.unbind(1), list(L0.unbind(1))
    for _ in range(5):
        for i in range(x.shape[1]):
            xi, wi, Li = xs[i], ws[i], Ls[i]
            slx = sumlx - wi * xi * Li
            ok = slx > 0
            sl2 = suml2 - wi * Li * Li
            newl = torch.round(xi * sl2 / torch.where(ok, slx, one)).clamp_(-fmax, fmax - 1.0)
            slx2 = slx + wi * xi * newl
            sl22 = sl2 + wi * newl * newl
            commit = ok & (newl != Li) & (sl22 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            Ls[i] = torch.where(commit, newl, Li)
            sumlx = torch.where(commit, slx2, sumlx)
            suml2 = torch.where(commit, sl22, suml2)

    L = torch.stack(Ls, dim=1)
    scale = torch.where(suml2 > 0, sumlx / torch.where(suml2 > 0, suml2, one),
                        torch.zeros_like(sumlx))
    scale = torch.where(live, scale, torch.zeros_like(scale))
    L = torch.where(live[:, None], L + fmax, torch.zeros_like(L))
    return scale, L


# ---------------------------------------------------------------------------
# CUDA: Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _div(a, b):
        """IEEE-rounded f32 division; the / operator lowers to approximate div.full.f32.
        a broadcasts against b so scalar numerators promote to tensors for the extern call."""
        return tld.div_rn(a + tl.zeros_like(b), b)

    @triton.jit
    def _seq_sum2(t1, t2, NL: tl.constexpr):
        """Strict left-to-right lane sums of t1 and t2, matching the sequential C accumulation."""
        lane = tl.arange(0, NL)[None, :]
        s1 = tl.sum(tl.where(lane == 0, t1, 0.0), axis=1)
        s2 = tl.sum(tl.where(lane == 0, t2, 0.0), axis=1)
        for i in tl.static_range(1, NL):
            s1 += tl.sum(tl.where(lane == i, t1, 0.0), axis=1)
            s2 += tl.sum(tl.where(lane == i, t2, 0.0), axis=1)
        return s1, s2

    @triton.jit
    def _qx_kernel(x_ptr, w_ptr, c_ptr, scale_ptr, l_ptr, G,
                   NL: tl.constexpr, NMAX: tl.constexpr, NSTEP: tl.constexpr,
                   BLOCK: tl.constexpr):
        """make_qx_quants(rmse_type=1), one lane-row per group."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        lane = tl.arange(0, NL)[None, :]
        offs = g[:, None] * NL + lane
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        ax = tl.abs(x)
        amax = tl.max(ax, axis=1)
        first = tl.min(tl.where(ax == amax[:, None], lane, NL), axis=1)
        mx = tl.sum(tl.where(lane == first[:, None], x, 0.0), axis=1)
        live = amax >= 1e-15
        mxs = tl.where(live, mx, 1.0)

        iscale = _div(-NMAX, mxs)
        L = tl.maximum(tl.minimum(tld.rint(iscale[:, None] * x), NMAX - 1), -NMAX)
        sumlx, suml2 = _seq_sum2(w * x * L, w * L * L, NL)
        scale = tl.where(suml2 != 0, _div(sumlx, tl.where(suml2 != 0, suml2, 1.0)), 0.0)
        best = scale * sumlx
        for s in tl.static_range(0, NSTEP):
            c = tl.load(c_ptr + s)
            isc = _div(-c, mxs)
            La = tl.maximum(tl.minimum(tld.rint(isc[:, None] * x), NMAX - 1), -NMAX)
            slx, sl2 = _seq_sum2(w * x * La, w * La * La, NL)
            acc = (sl2 > 0) & (slx * slx > best * sl2)
            cand = _div(slx, tl.where(acc, sl2, 1.0))
            scale = tl.where(acc, cand, scale)
            best = tl.where(acc, cand * slx, best)
            L = tl.where(acc[:, None], La, L)

        tl.store(scale_ptr + g, tl.where(live, scale, 0.0), mask=m)
        tl.store(l_ptr + offs, tl.where(live[:, None], L + NMAX, 0.0), mask=m[:, None])

    @triton.jit
    def _q3_kernel(x_ptr, scale_ptr, l_ptr, G,
                   NL: tl.constexpr, NMAX: tl.constexpr, BLOCK: tl.constexpr):
        """make_q3_quants(do_rmse=true), one lane-row per group.
        Commits chain through sumlx/suml2, so the lane loop stays sequential."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        lane = tl.arange(0, NL)[None, :]
        offs = g[:, None] * NL + lane
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = x * x

        ax = tl.abs(x)
        amax = tl.max(ax, axis=1)
        first = tl.min(tl.where(ax == amax[:, None], lane, NL), axis=1)
        mx = tl.sum(tl.where(lane == first[:, None], x, 0.0), axis=1)
        live = amax >= 1e-15
        mxs = tl.where(live, mx, 1.0)

        iscale = _div(-NMAX, mxs)
        L = tl.maximum(tl.minimum(tld.rint(iscale[:, None] * x), NMAX - 1), -NMAX)
        sumlx, suml2 = _seq_sum2(w * x * L, w * L * L, NL)
        for _p in range(5):
            for i in tl.static_range(NL):
                li = lane == i
                xi = tl.sum(tl.where(li, x, 0.0), axis=1)
                wi = xi * xi
                Li = tl.sum(tl.where(li, L, 0.0), axis=1)
                slx = sumlx - wi * xi * Li
                ok = slx > 0
                sl2 = suml2 - wi * Li * Li
                newl = tl.maximum(tl.minimum(tld.rint(_div(xi * sl2, tl.where(ok, slx, 1.0))),
                                             NMAX - 1), -NMAX)
                slx2 = slx + wi * xi * newl
                sl22 = sl2 + wi * newl * newl
                commit = ok & (newl != Li) & (sl22 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
                L = tl.where(commit[:, None] & li, newl[:, None], L)
                sumlx = tl.where(commit, slx2, sumlx)
                suml2 = tl.where(commit, sl22, suml2)

        scale = tl.where(suml2 > 0, _div(sumlx, tl.where(suml2 > 0, suml2, 1.0)), 0.0)
        tl.store(scale_ptr + g, tl.where(live, scale, 0.0), mask=m)
        tl.store(l_ptr + offs, tl.where(live[:, None], L + NMAX, 0.0), mask=m[:, None])

    _const_cache: dict = {}

    def _const_for(device: torch.device, arr: np.ndarray) -> torch.Tensor:
        key = (str(device), arr.tobytes())
        if key not in _const_cache:
            _const_cache[key] = torch.from_numpy(arr).to(device)
        return _const_cache[key]

    def make_qx_quants_triton(x: torch.Tensor, w: torch.Tensor, nmax: int):
        G, n = x.shape
        steps = qx_steps(nmax)
        c = _const_for(x.device, steps)
        scale = torch.empty(G, dtype=torch.float32, device=x.device)
        L = torch.empty(G, n, dtype=torch.float32, device=x.device)
        BLOCK = 64
        _qx_kernel[((G + BLOCK - 1) // BLOCK,)](
            x.contiguous(), w.contiguous(), c, scale, L, G,
            NL=n, NMAX=float(nmax), NSTEP=len(steps), BLOCK=BLOCK, num_warps=4)
        return scale, L

    def make_q3_quants_triton(x: torch.Tensor, nmax: int):
        G, n = x.shape
        scale = torch.empty(G, dtype=torch.float32, device=x.device)
        L = torch.empty(G, n, dtype=torch.float32, device=x.device)
        BLOCK = 64
        _q3_kernel[((G + BLOCK - 1) // BLOCK,)](
            x.contiguous(), scale, L, G,
            NL=n, NMAX=float(nmax), BLOCK=BLOCK, num_warps=4)
        return scale, L


def use_triton(x: torch.Tensor) -> bool:
    return x.device.type == "cuda" and HAS_TRITON


def make_qx_quants(x, w, nmax):
    fn = make_qx_quants_triton if use_triton(x) else make_qx_quants_eager
    return fn(x, w, nmax)


def make_q3_quants(x, nmax):
    fn = make_q3_quants_triton if use_triton(x) else make_q3_quants_eager
    return fn(x, nmax)
