"""IQ2_XXS quantization for quant-studio: a torch/Triton port of quantize_row_iq2_xxs_impl() from ggml-quants.c.

The E8-lattice tables mirror iq2xs_init_impl(): a neighbour list holds every grid point within the two smallest distinct squared distances of a code, in the same (distance, grid index) order the C qsort produces.
On-grid codes get a single-entry list with themselves, equivalent to the C "kmap hit -> keep the code" path.
The search follows the C code step for step, but reductions are tree-ordered, so an occasional ULP-level tie-break can differ from the CPU reference.

On CUDA the hot parts are two Triton kernels plus torch.compile'd glue.
Real-data candidate lists are bimodal: about half the codes are on-grid (one candidate), the rest have 40-150.
So on-grid subgroups resolve with a kmap lookup and the search kernel runs only over a compacted index of off-grid subgroups.
CPU/MPS fall back to the plain torch implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import HAS_TRITON, F32_TINY, QuantSpec, jit, tl, tld, triton
from .iq2_tables import KGRID_2BIT_256

QK_K = 256
KMAP_SIZE = 43692
GROUP_MAX_EPS = 1e-15
BLOCK_BYTES = 2 + QK_K // 4  # fp16 d + 32 uint16 = 66
_TINY = F32_TINY


class Tables:
    def __init__(self, device: torch.device):
        kgrid = np.asarray(KGRID_2BIT_256, dtype=np.int64)          # (256,) codes
        shifts = 2 * np.arange(8)
        grid_l = (kgrid[:, None] >> shifts) & 3                     # (256,8) levels 0..3
        grid_v = 2 * grid_l + 1                                     # values 1,3,5,7

        kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
        kmap[kgrid] = np.arange(256)

        missing = np.nonzero(kmap < 0)[0]
        pos = 2 * ((missing[:, None] >> shifts) & 3) + 1            # (M,8)
        d2 = ((pos[:, None, :] - grid_v[None, :, :]) ** 2).sum(-1)  # (M,256)
        order = np.argsort(d2 * 512 + np.arange(256), axis=1)       # (d2, idx) order
        d2s = np.take_along_axis(d2, order, axis=1)
        d0 = d2s[:, :1]
        has2 = (d2s > d0).any(1)
        second = np.where(has2, d2s[np.arange(len(missing)), (d2s > d0).argmax(1)], d0[:, 0])
        counts = (d2s <= second[:, None]).sum(1)                    # prefix mask by row
        maxn = int(counts.max())
        maxn_pad = (maxn + 3) // 4 * 4  # rows 8B-aligned for int64 vector loads

        # candidate lists indexed by code; on-grid codes list only themselves
        neigh = np.full((KMAP_SIZE, maxn_pad), -1, dtype=np.int16)
        neigh[missing, :maxn] = np.where(np.arange(maxn)[None, :] < counts[:, None],
                                         order[:, :maxn], -1).astype(np.int16)
        neigh[kgrid, 0] = np.arange(256, dtype=np.int16)
        nlen = np.zeros(KMAP_SIZE, dtype=np.int16)
        nlen[missing] = counts.astype(np.int16)
        nlen[kgrid] = 1

        dev = device
        self.codes = torch.from_numpy(kgrid.astype(np.int16)).to(dev)       # (256,)
        self.grid_f = torch.from_numpy(grid_v.astype(np.float32)).to(dev)   # (256,8)
        self.grid_l = torch.from_numpy((grid_v - 1) // 2).to(dev)           # (256,8) 0..3
        self.kmap = torch.from_numpy(kmap).to(dev)
        self.neigh = torch.from_numpy(neigh).to(dev)                        # (43692,maxn_pad)
        self.neigh64 = torch.from_numpy(neigh.view(np.int64)
                                        .reshape(KMAP_SIZE, maxn_pad // 4)).to(dev)
        self.nlen = torch.from_numpy(nlen).to(dev)
        self.shifts2 = (2 * torch.arange(8, device=dev)).long()
        self.arange4 = torch.arange(4, device=dev).long()
        # length buckets for the eager path, so short lists do not pay for the padded max
        self.buckets = tuple(w for w in (8, 16, 48, maxn) if w <= maxn)


_tables_cache: dict[str, Tables] = {}


def tables_for(device: torch.device) -> Tables:
    key = str(device)
    if key not in _tables_cache:
        _tables_cache[key] = Tables(device)
    return _tables_cache[key]


def _finalize(q2lo: torch.Tensor, q2hi: torch.Tensor, scales_g: torch.Tensor,
              n: int, k: int, dev: torch.device) -> torch.Tensor:
    """Superblock packing: d = max_scale/31, 4-bit sub-scales in the top nibble, serialize to LE blocks."""
    ns = n * k // QK_K
    sc = scales_g.reshape(ns, QK_K // 32)
    max_scale = sc.max(-1).values
    has = max_scale > 0
    d = max_scale / 31.0
    d16 = torch.where(has, d, torch.zeros_like(d)).to(torch.float16)
    inv_d = 1.0 / torch.where(has, d, torch.ones_like(d))
    lnib = torch.clamp(torch.round(0.5 * (inv_d[:, None] * sc - 1.0)), 0, 15).long()
    q2hi = q2hi.reshape(ns, -1) | (lnib << 28)
    q2lo = q2lo.reshape(ns, -1)
    q2lo = torch.where(has[:, None], q2lo, torch.zeros_like(q2lo))
    q2hi = torch.where(has[:, None], q2hi, torch.zeros_like(q2hi))

    words = torch.stack([q2lo, q2hi], dim=-1).reshape(ns, 2 * (QK_K // 32))
    shifts8 = (8 * torch.arange(4, device=dev)).long()
    qbytes = ((words[..., None] >> shifts8) & 255).to(torch.uint8).reshape(ns, QK_K // 4)
    dbytes = d16[:, None].contiguous().view(torch.uint8)
    out = torch.cat([dbytes, qbytes], dim=1)             # (ns, 66)
    return out.reshape(n, (k // QK_K) * BLOCK_BYTES)


# ---------------------------------------------------------------------------
# CUDA: Triton kernels + torch.compile'd glue
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _qp_kernel(x_ptr, w_ptr, out_ptr, G, BLOCK: tl.constexpr):
        """make_qp_quants(n=32, nmax=4), one lane-row per group."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        offs = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + offs, mask=m[:, None], other=0.0)

        mx = tl.max(x, axis=1)
        dead = mx < 1e-15
        mxs = tl.where(dead, 1.0, mx)
        iscale = 4.0 / mxs
        L = tld.rint(iscale[:, None] * x)                # first pass: no clamp in C
        df = x - (1.0 / iscale)[:, None] * L
        best_mse = tl.sum(w * df * df, axis=1)
        for s in tl.static_range(-4, 5):
            if s != 0:
                isc = tl.full((1,), 0.1 * s + 4.0, tl.float32) / mxs
                l = tl.minimum(tld.rint(isc[:, None] * x), 4.0)
                df2 = x - (1.0 / isc)[:, None] * l
                mse = tl.sum(w * df2 * df2, axis=1)
                better = mse < best_mse
                best_mse = tl.where(better, mse, best_mse)
                iscale = tl.where(better, isc, iscale)

        L = tl.minimum(tld.rint(iscale[:, None] * x), 4.0)
        sumlx = tl.sum(w * x * L, axis=1)
        suml2 = tl.sum(w * L * L, axis=1)
        for _p in range(5):
            for i in tl.static_range(32):
                lane = tl.arange(0, 32)[None, :] == i
                xi = tl.sum(tl.where(lane, x, 0.0), axis=1)
                wi = tl.sum(tl.where(lane, w, 0.0), axis=1)
                Li = tl.sum(tl.where(lane, L, 0.0), axis=1)
                slx = sumlx - wi * xi * Li
                sl2 = suml2 - wi * Li * Li
                ok = (slx > 0) & (sl2 > 0)
                newl = tl.minimum(tld.rint(xi * sl2 / tl.where(ok, slx, 1.0)), 4.0)
                slx2 = slx + wi * xi * newl
                sl22 = sl2 + wi * newl * newl
                commit = ok & (newl != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
                L = tl.where(commit[:, None] & lane, newl[:, None], L)
                sumlx = tl.where(commit, slx2, sumlx)
                suml2 = tl.where(commit, sl22, suml2)

        scale = tl.where(suml2 > 0,
                         sumlx / tl.maximum(suml2, tl.full((1,), 1e-38, tl.float32)), 0.0)
        tl.store(out_ptr + g, tl.where(dead, 0.0, scale), mask=m)

    @triton.jit
    def _search_kernel(idx_ptr, u_ptr, xv_ptr, wv_ptr, sc_ptr, neigh64_ptr, nlen_ptr,
                       code_ptr, win_ptr, N, M, W4: tl.constexpr, BLOCK: tl.constexpr):
        """iq2_find_best_neighbour over a compacted list of off-grid (step, subgroup) problems.
        Candidates load four at a time as one int64; grid values decode from the 512B code table.
        M = subgroups per step; sc_ptr holds (steps, M//4) per-group scales."""
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = r < N
        a8 = tl.arange(0, 8)[None, :]
        p = tl.load(idx_ptr + r, mask=m, other=0)        # flat (step, subgroup) ids
        step = p // M
        sub = p - step * M
        off8 = sub[:, None] * 8 + a8
        xv = tl.load(xv_ptr + off8, mask=m[:, None], other=0.0)
        wv = tl.load(wv_ptr + off8, mask=m[:, None], other=0.0)
        sc = tl.load(sc_ptr + step * (M // 4) + (sub >> 2), mask=m, other=1.0)
        u = tl.load(u_ptr + p, mask=m, other=0)
        lens = tl.where(m, tl.load(nlen_ptr + u).to(tl.int32), 0)
        maxlen = tl.max(lens)
        best = tl.full((BLOCK,), float("inf"), tl.float32)
        bestc = tl.zeros((BLOCK,), dtype=tl.int32)
        for t4 in range(0, maxlen, 4):
            q4 = tl.load(neigh64_ptr + u * W4 + t4 // 4, mask=t4 < lens, other=-1)
            for j in tl.static_range(4):
                cand = ((q4 >> (16 * j)) & 0xFFFF).to(tl.int32)
                valid = (cand < 256) & (t4 + j < lens)   # 0xFFFF == the -1 padding
                code = tl.load(code_ptr + tl.where(valid, cand, 0)).to(tl.int32)
                gv = (2 * ((code[:, None] >> (2 * a8)) & 3) + 1).to(tl.float32)
                df = sc[:, None] * gv - xv
                d2 = tl.sum(wv * df * df, axis=1)
                take = valid & (d2 < best)               # strict <, keeps first
                best = tl.where(take, d2, best)
                bestc = tl.where(take, cand, bestc)
        tl.store(win_ptr + p, bestc, mask=m)

    def _qp_triton(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        G = x.shape[0]
        out = torch.empty(G, dtype=torch.float32, device=x.device)
        BLOCK = 64
        _qp_kernel[((G + BLOCK - 1) // BLOCK,)](
            x, w, out, G, BLOCK=BLOCK, num_warps=4)
        return out

    def _search_off_grid(u_all: torch.Tensor, xv_m, wv_m, sc_all, T: Tables) -> torch.Tensor:
        """Winning grid index per (step, subgroup) problem.
        On-grid codes resolve via kmap; off-grid problems go through one batched search launch."""
        steps = u_all.shape[0]
        M = u_all[0].numel()
        u = u_all.reshape(-1)
        win = T.kmap[u.long()].to(torch.int32)           # >= 0 iff on-grid
        off = (win < 0).nonzero(as_tuple=True)[0].to(torch.int32)
        if off.numel():
            BLOCK = 128
            _search_kernel[((off.numel() + BLOCK - 1) // BLOCK,)](
                off, u, xv_m, wv_m, sc_all.contiguous(), T.neigh64, T.nlen, T.codes,
                win, off.numel(), M, W4=T.neigh64.shape[1], BLOCK=BLOCK, num_warps=4)
        return win.reshape(steps, M // 4, 4)

    def _sweep_codes(xval, inv, sh):
        """Codes of the tentative levels clamp(round((id*x - 1)/2), 0, 2)."""
        lev = torch.clamp(torch.round(0.5 * (inv[:, None] * xval - 1.0)), 0, 2).to(torch.int32)
        return (lev.reshape(-1, 4, 8) << sh).sum(-1, dtype=torch.int32)

    def _prep_fn(xbl, qw_bl, bit):
        """Importance weights + sign-parity fold; returns per-group (w, waux, xval) and sign bits."""
        sigma2 = (xbl * xbl).sum(-1) / QK_K
        w = qw_bl * torch.sqrt(sigma2[:, None] + xbl * xbl)
        x8 = xbl.reshape(-1, 4, 8)
        w8 = w.reshape(-1, 4, 8)
        neg = x8 < 0
        xval8 = x8.abs()
        s = (neg.long() * bit).sum(-1)
        odd = (neg.sum(-1) % 2) == 1
        imin = (w8 * x8 * x8).argmin(-1)                 # first min, like the C scan
        flip = torch.nn.functional.one_hot(imin, 8).bool() & odd[:, :, None]
        xval8 = torch.where(flip, -xval8, xval8)
        s = s ^ ((1 << imin) * odd.long())
        G = x8.shape[0]
        return (w.reshape(G, 32), torch.sqrt(w).reshape(G, 32),
                xval8.reshape(G, 32), s & 127)

    def _final_fn(w, xval, win4, Lc, scale, proj, active, block_signs, codes, sh2, ar4):
        """Final least-squares refit, sign flip for negative scales, packed [indices | signs] words."""
        lv = ((codes[win4.long()].long().unsqueeze(-1) >> sh2) & 3).float()
        q = (2.0 * lv + 1.0).reshape(w.shape)
        sumqx = (w * xval * q).sum(-1)
        sumq2 = (w * q * q).sum(-1)
        refit = proj & (sumq2 > 0)
        scale = torch.where(refit, sumqx / sumq2.clamp(min=_TINY), scale)
        Lc = torch.where(proj[:, None], win4, Lc)
        negs = active & (scale < 0)
        scale = torch.where(negs, -scale, scale)
        block_signs = torch.where(negs[:, None], (~block_signs) & 127, block_signs)
        q2lo = (Lc.long() << (8 * ar4)).sum(-1)
        q2hi = (block_signs << (7 * ar4)).sum(-1)
        zero = torch.zeros_like(q2lo)
        return (torch.where(active, q2lo, zero), torch.where(active, q2hi, zero),
                torch.where(active, scale, torch.zeros_like(scale)))

    def _post_win(w, xval, win4, scale, best, Lc, active, codes, sh2):
        """Decode winner levels + least-squares scale + accept test."""
        lv = ((codes[win4.long()].long().unsqueeze(-1) >> sh2) & 3).float()
        q = (2.0 * lv + 1.0).reshape(w.shape)
        sumqx = (w * xval * q).sum(-1)
        sumq2 = (w * q * q).sum(-1)
        acc = active & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        newscale = sumqx / sumq2.clamp(min=_TINY)
        return (torch.where(acc, newscale, scale),
                torch.where(acc, newscale * sumqx, best),
                torch.where(acc[:, None], win4, Lc))

    def _quantize_cuda(x: torch.Tensor, qw: torch.Tensor, T: Tables) -> torch.Tensor:
        n, k = x.shape
        ns = n * k // QK_K
        dev = x.device
        G = ns * (QK_K // 32)
        xbl = x.reshape(ns, QK_K)
        qw_bl = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        bit = (1 << torch.arange(8, device=dev)).long()
        w, waux, xval, block_signs = jit("iq2_prep", _prep_fn)(xbl, qw_bl, bit)
        xv_m = xval.reshape(-1, 8)                       # (M,8) subgroup views
        wv_m = waux.reshape(-1, 8)

        # initial scale (skip all-zero groups)
        gmax = xval.max(-1).values
        active = gmax >= GROUP_MAX_EPS
        scale0 = _qp_triton(xval, w)                     # C passes kMaxQ+1 = 4
        eff_max = scale0 * 3.0
        active &= eff_max > 0
        eff_safe = torch.where(active, eff_max, torch.ones_like(eff_max))

        codes_fn = jit("iq2_sweep_codes", _sweep_codes)
        post_fn = jit("iq2_post_win", _post_win)
        sh32 = (2 * torch.arange(8, device=dev, dtype=torch.int32))

        # the 13 candidate scales depend only on eff_max, so all searches batch into one launch
        invs = [(5.0 + 0.1 * step) / eff_safe for step in range(-6, 7)]
        u_all = torch.stack([codes_fn(xval, inv, sh32) for inv in invs])
        sc_all = torch.stack([1.0 / inv for inv in invs])
        win_all = _search_off_grid(u_all, xv_m, wv_m, sc_all, T)

        scale = scale0.clone()
        best = torch.zeros_like(scale)
        Lc = torch.zeros(G, 4, dtype=torch.int32, device=dev)
        for st in range(13):
            scale, best, Lc = post_fn(w, xval, win_all[st], scale, best, Lc, active,
                                      T.codes, T.shifts2)

        # final projection with the chosen scale + refit + sign flip + packing
        proj = active & (scale > 0)
        inv = 1.0 / torch.where(proj, scale, torch.ones_like(scale))
        u = codes_fn(xval, inv, sh32)
        win = _search_off_grid(u[None], xv_m, wv_m, scale[None], T)[0]
        q2lo, q2hi, scales_g = jit("iq2_final", _final_fn)(
            w, xval, win, Lc, scale, proj, active, block_signs,
            T.codes, T.shifts2, T.arange4)
        return _finalize(q2lo, q2hi, scales_g, n, k, dev)

# ---------------------------------------------------------------------------
# eager fallback (CPU / MPS) - the reference implementation
# ---------------------------------------------------------------------------


def _sweep_pre(xval, inv):
    """Tentative levels for a candidate scale: clamp(round((id*x - 1)/2), 0, 2)."""
    return torch.clamp(torch.round(0.5 * (inv[:, None] * xval - 1.0)), 0, 2).long()


def _sweep_post(w, xval, Laux, scale, best, L, active):
    """Weighted least-squares scale for the projected levels + accept test."""
    q = (2 * Laux + 1).float()
    sumqx = (w * xval * q).sum(-1)
    sumq2 = (w * q * q).sum(-1)
    acc = active & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
    newscale = sumqx / sumq2.clamp(min=_TINY)
    return (torch.where(acc, newscale, scale),
            torch.where(acc, newscale * sumqx, best),
            torch.where(acc[:, None], Laux, L))


def _project_bucketed(Lq, xval8, waux8, scale_g, T: Tables):
    """Eager-mode projection: same search, bucketed by candidate-list length."""
    G = Lq.shape[0]
    u = (Lq << T.shifts2).sum(-1).reshape(-1)
    xv = xval8.reshape(-1, 8)
    wv = waux8.reshape(-1, 8)
    sc = scale_g[:, None].expand(G, 4).reshape(-1)
    lens = T.nlen[u]
    gi = torch.empty_like(u)
    prev = 0
    for width in T.buckets:
        sel = ((lens > prev) & (lens <= width)).nonzero(as_tuple=True)[0]
        prev = width
        if sel.numel() == 0:
            continue
        sub = max(1, (1 << 23) // width)
        for s0 in range(0, sel.numel(), sub):
            idx = sel[s0 : s0 + sub]
            cand = T.neigh[u[idx], :width].long()
            gv = T.grid_f[cand.clamp(min=0)]
            diff = sc[idx, None, None] * gv - xv[idx][:, None, :]
            d2 = (wv[idx][:, None, :] * diff * diff).sum(-1)
            d2.masked_fill_(cand < 0, float("inf"))
            j = d2.argmin(-1)                            # first min == C's strict <
            gi[idx] = cand.gather(1, j[:, None]).squeeze(1)
    return T.grid_l[gi.reshape(G, 4)]


def _make_qp_quants(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Port of make_qp_quants(n=32, nmax=4). x, w: (G, 32) f32; returns (G,)."""
    mx = x.max(-1).values
    dead = mx < GROUP_MAX_EPS
    mxs = torch.where(dead, torch.ones_like(mx), mx)

    iscale = 4.0 / mxs
    L0 = torch.round(iscale[:, None] * x)                # first pass: no clamp in C
    best_mse = (w * (x - (1.0 / iscale)[:, None] * L0) ** 2).sum(-1)
    for step in range(-4, 5):
        if step == 0:
            continue
        isc = (0.1 * step + 4.0) / mxs
        l = torch.clamp(torch.round(isc[:, None] * x), max=4.0)
        mse = (w * (x - (1.0 / isc)[:, None] * l) ** 2).sum(-1)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        iscale = torch.where(better, isc, iscale)

    L = torch.clamp(torch.round(iscale[:, None] * x), max=4.0)
    sumlx = (w * x * L).sum(-1)
    suml2 = (w * L * L).sum(-1)
    # extra passes after convergence are no-ops, so the C early-break does not change results
    xs, ws, Ls = x.unbind(1), w.unbind(1), list(L.unbind(1))
    for _ in range(5):
        for i in range(32):
            xi, wi, Li = xs[i], ws[i], Ls[i]
            slx = sumlx - wi * xi * Li
            sl2 = suml2 - wi * Li * Li
            ok = (slx > 0) & (sl2 > 0)
            newl = torch.clamp(torch.round(xi * sl2 / torch.where(ok, slx, torch.ones_like(slx))),
                               max=4.0)
            slx2 = slx + wi * xi * newl
            sl22 = sl2 + wi * newl * newl
            commit = ok & (newl != Li) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            Ls[i] = torch.where(commit, newl, Li)
            sumlx = torch.where(commit, slx2, sumlx)
            suml2 = torch.where(commit, sl22, suml2)

    scale = torch.where(suml2 > 0, sumlx / suml2.clamp(min=_TINY), torch.zeros_like(sumlx))
    return torch.where(dead, torch.zeros_like(scale), scale)


def quantize_iq2_xxs(x: torch.Tensor, qw: torch.Tensor, T: Tables) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights; returns (n_rows, k//256 * 66) uint8."""
    dev = x.device
    if dev.type == "cuda" and HAS_TRITON:
        return _quantize_cuda(x, qw, T)

    n, k = x.shape
    ns = n * k // QK_K
    xbl = x.reshape(ns, QK_K)
    sigma2 = (xbl * xbl).sum(-1) / QK_K
    qw_bl = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
    weight = qw_bl * torch.sqrt(sigma2[:, None] + xbl * xbl)

    G = ns * (QK_K // 32)
    xb = xbl.reshape(G, 32)
    w = weight.reshape(G, 32)
    waux = torch.sqrt(w)

    # --- signs: fold into |x|, forcing even sign-parity per 8 values ---
    x8 = xb.reshape(G, 4, 8)
    neg = x8 < 0
    xval8 = x8.abs()
    bit = (1 << torch.arange(8, device=dev)).long()
    s = (neg.long() * bit).sum(-1)                       # (G,4)
    odd = (neg.sum(-1) % 2) == 1
    imin = (w.reshape(G, 4, 8) * x8 * x8).argmin(-1)     # first min, like the C scan
    flip = torch.nn.functional.one_hot(imin, 8).bool() & odd[:, :, None]
    xval8 = torch.where(flip, -xval8, xval8)
    s = s ^ ((1 << imin) * odd.long())
    block_signs = s & 127                                # (G,4)
    xval = xval8.reshape(G, 32)
    waux8 = waux.reshape(G, 4, 8)

    def proj_levels(inv, sc):
        Lq = _sweep_pre(xval, inv).reshape(G, 4, 8)
        return _project_bucketed(Lq, xval8, waux8, sc, T).reshape(G, 32).to(torch.int8)

    # --- initial scale (skip all-zero groups) ---
    gmax = xval.max(-1).values
    active = gmax >= GROUP_MAX_EPS
    scale0 = _make_qp_quants(xval, w)                    # C passes kMaxQ+1 = 4
    eff_max = scale0 * 3.0
    active &= eff_max > 0
    eff_safe = torch.where(active, eff_max, torch.ones_like(eff_max))

    # --- 13-candidate scale sweep with grid projection ---
    scale = scale0.clone()
    best = torch.zeros_like(scale)
    L = torch.zeros(G, 32, dtype=torch.int8, device=dev)
    for step in range(-6, 7):
        inv = (5.0 + 0.1 * step) / eff_safe              # 2*kMaxQ-1 = 5
        Laux = proj_levels(inv, 1.0 / inv)
        scale, best, L = _sweep_post(w, xval, Laux, scale, best, L, active)

    # --- final projection with the chosen scale + least-squares refit ---
    proj = active & (scale > 0)
    inv = 1.0 / torch.where(proj, scale, torch.ones_like(scale))
    Lp = proj_levels(inv, scale)
    q = (2 * Lp + 1).float()
    sumqx = (w * xval * q).sum(-1)
    sumq2 = (w * q * q).sum(-1)
    refit = proj & (sumq2 > 0)
    scale = torch.where(refit, sumqx / sumq2.clamp(min=_TINY), scale)
    L = torch.where(proj[:, None], Lp, L)

    # negative least-squares scale: flip the scale and all the signs
    negs = active & (scale < 0)
    scale = torch.where(negs, -scale, scale)
    block_signs = torch.where(negs[:, None], (~block_signs) & 127, block_signs)

    # --- pack: 32-bit words per group [grid indices | signs] ---
    u = (L.long().reshape(G, 4, 8) << T.shifts2).sum(-1)  # (G,4) on-grid codes
    gi = T.kmap[u].clamp(min=0)
    q2lo = (gi << (8 * T.arange4)).sum(-1)               # (G,)
    q2hi = (block_signs << (7 * T.arange4)).sum(-1)      # (G,)
    scales_g = torch.where(active, scale, torch.zeros_like(scale))
    q2lo = torch.where(active, q2lo, torch.zeros_like(q2lo))
    q2hi = torch.where(active, q2hi, torch.zeros_like(q2hi))
    return _finalize(q2lo, q2hi, scales_g, n, k, dev)


def _make_kernel(device: torch.device, qw: torch.Tensor) -> callable:
    T = tables_for(device)
    return lambda x: quantize_iq2_xxs(x, qw, T)


SPECS = {
    "iq2_xxs": QuantSpec(GGMLQuantizationType.IQ2_XXS, LlamaFileType.MOSTLY_IQ2_XXS,
                         64, True, _make_kernel),
}
