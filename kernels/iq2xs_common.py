"""Shared machinery for the IQ2_XS / IQ2_S ports of quantize_row_iq2_xs_impl() and quantize_row_iq2_s_impl() from ggml-quants.c.

Tables mirror iq2xs_init_impl(): for the 512-entry grid (IQ2_XS) a neighbour list holds every grid point within the two smallest distinct squared distances (nwant=2); for the 1024-entry grid (IQ2_S) only the smallest (nwant=1), in the (distance, grid index) order the C qsort produces.
Both types share the same per-16 search: a 19-step candidate scale sweep with grid projection, an off-grid-only reprojection at the accepted scale, and a final least-squares refit.
On CUDA the projection runs as one batched Triton kernel over the compacted off-grid subgroups; on-grid subgroups resolve with a kmap lookup. CPU/MPS use a bucketed eager search.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import HAS_TRITON, F32_TINY, tl, tld, triton
from .iq2xs_tables import KGRID_2BIT_512, KGRID_2BIT_1024

QK_K = 256
KMAP_SIZE = 43692
_TINY = F32_TINY

# id = (2*kMaxQ-1 + is*0.1f)/max for is in -9..9, folded in fp32 like the C code
SWEEP_CS = [float(np.float32(5.0) + np.float32(s) * np.float32(0.1)) for s in range(-9, 10)]

_steps_cache: dict[str, torch.Tensor] = {}


def _steps_for(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _steps_cache:
        _steps_cache[key] = torch.tensor(SWEEP_CS, dtype=torch.float32, device=device)
    return _steps_cache[key]


def tree_sum(t: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair tree sum over the last dim (power of two); the Triton sweep uses the
    same association, so the accept chain matches between CPU and CUDA."""
    while t.shape[-1] > 1:
        t = t[..., 0::2] + t[..., 1::2]
    return t[..., 0]


class Tables:
    def __init__(self, device: torch.device, kgrid_list: list[int], nwant: int):
        grid_size = len(kgrid_list)
        kgrid = np.asarray(kgrid_list, dtype=np.int64)
        shifts = 2 * np.arange(8)
        grid_l = (kgrid[:, None] >> shifts) & 3                     # (grid,8) levels
        grid_v = 2 * grid_l + 1

        kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
        kmap[kgrid] = np.arange(grid_size)

        missing = np.nonzero(kmap < 0)[0]
        M = len(missing)
        pos = (2 * ((missing[:, None] >> shifts) & 3) + 1).astype(np.int32)
        gv = grid_v.astype(np.int32)
        order = np.empty((M, grid_size), dtype=np.int16)
        counts = np.empty(M, dtype=np.int64)
        for i0 in range(0, M, 4096):                                # chunked to bound memory
            p = pos[i0:i0 + 4096]
            d2 = ((p[:, None, :] - gv[None, :, :]) ** 2).sum(-1)    # (c,grid)
            o = np.argsort(d2 * grid_size + np.arange(grid_size), axis=1)  # (d2, idx) order
            d2s = np.take_along_axis(d2, o, axis=1)
            d0 = d2s[:, :1]
            if nwant == 1:
                second = d0[:, 0]
            else:
                has2 = (d2s > d0).any(1)
                second = np.where(has2, d2s[np.arange(len(p)), (d2s > d0).argmax(1)], d0[:, 0])
            counts[i0:i0 + 4096] = (d2s <= second[:, None]).sum(1)
            order[i0:i0 + 4096] = o.astype(np.int16)
        maxn = int(counts.max())
        maxn_pad = (maxn + 3) // 4 * 4  # rows 8B-aligned for int64 vector loads

        # candidate lists indexed by code; on-grid codes list only themselves
        neigh = np.full((KMAP_SIZE, maxn_pad), -1, dtype=np.int16)
        neigh[missing, :maxn] = np.where(np.arange(maxn)[None, :] < counts[:, None],
                                         order[:, :maxn], -1).astype(np.int16)
        neigh[kgrid, 0] = np.arange(grid_size, dtype=np.int16)
        nlen = np.zeros(KMAP_SIZE, dtype=np.int16)
        nlen[missing] = counts.astype(np.int16)
        nlen[kgrid] = 1

        dev = device
        self.grid_size = grid_size
        self.codes = torch.from_numpy(kgrid.astype(np.int16)).to(dev)       # (grid,)
        self.grid_f = torch.from_numpy(grid_v.astype(np.float32)).to(dev)   # (grid,8)
        self.kmap = torch.from_numpy(kmap).to(dev)
        self.neigh = torch.from_numpy(neigh).to(dev)
        self.neigh64 = torch.from_numpy(neigh.view(np.int64)
                                        .reshape(KMAP_SIZE, maxn_pad // 4)).to(dev)
        self.nlen = torch.from_numpy(nlen).to(dev)
        self.shifts2 = (2 * torch.arange(8, device=dev)).long()
        self.code0 = int(kmap[0]) if kmap[0] >= 0 else 0
        # length buckets for the eager path, so short lists do not pay for the padded max
        self.buckets = tuple(dict.fromkeys(w for w in (8, 16, 48, maxn) if w <= maxn))


_tables_cache: dict[tuple[str, str], Tables] = {}


def tables_for(kind: str, device: torch.device) -> Tables:
    key = (kind, str(device))
    if key not in _tables_cache:
        grid, nwant = (KGRID_2BIT_512, 2) if kind == "xs" else (KGRID_2BIT_1024, 1)
        _tables_cache[key] = Tables(device, grid, nwant)
    return _tables_cache[key]


if HAS_TRITON:

    @triton.jit
    def _pair_add(v, BLOCK: tl.constexpr, M: tl.constexpr):
        a, b = tl.split(tl.reshape(v, (BLOCK, M, 2)))
        return tld.add_rn(a, b)

    @triton.jit
    def _tree16(v, BLOCK: tl.constexpr):
        """Adjacent-pair tree sum over 16 lanes; add_rn blocks fma contraction, matching tree_sum."""
        v = _pair_add(v, BLOCK, 8)
        v = _pair_add(v, BLOCK, 4)
        v = _pair_add(v, BLOCK, 2)
        v = _pair_add(v, BLOCK, 1)
        return tl.reshape(v, (BLOCK,))

    @triton.jit
    def _codes_kernel(x_ptr, gsafe_ptr, cs_ptr, u_ptr, sc_ptr, G,
                      S: tl.constexpr, BLOCK: tl.constexpr):
        """Tentative codes + scales for every sweep step in one pass over xval.
        Divisions mirror the eager ops: c/gsafe is reciprocal-multiply, 1/inv is a true division."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        a16 = tl.arange(0, 16)[None, :]
        x = tl.load(x_ptr + g[:, None] * 16 + a16, mask=m[:, None], other=0.0)
        one = tl.full((BLOCK,), 1.0, tl.float32)
        r = tld.div_rn(one, tl.load(gsafe_ptr + g, mask=m, other=1.0))
        sh = 2 * (a16 % 8)
        half = a16 < 8
        for s in tl.static_range(0, S):
            c = tl.load(cs_ptr + s)
            inv = c * r
            lev = tl.maximum(tl.minimum(tld.rint(0.5 * (inv[:, None] * x - 1.0)), 2.0), 0.0)
            bits = lev.to(tl.int32) << sh
            tl.store(u_ptr + (s * G + g) * 2 + 0, tl.sum(tl.where(half, bits, 0), axis=1), mask=m)
            tl.store(u_ptr + (s * G + g) * 2 + 1, tl.sum(tl.where(half, 0, bits), axis=1), mask=m)
            tl.store(sc_ptr + s * G + g, tld.div_rn(one, inv), mask=m)

    @triton.jit
    def _sweep_kernel(x_ptr, w_ptr, act_ptr, scale0_ptr, win_ptr, og_ptr, code_ptr,
                      scale_out, win_out, og_out, G,
                      CODE0: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
        """Fused accept chain over the S sweep steps: decode the winners from the code
        table and carry (scale, best, win, og) in registers instead of eager passes."""
        g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = g < G
        a16 = tl.arange(0, 16)[None, :]
        x = tl.load(x_ptr + g[:, None] * 16 + a16, mask=m[:, None], other=0.0)
        w = tl.load(w_ptr + g[:, None] * 16 + a16, mask=m[:, None], other=0.0)
        act = tl.load(act_ptr + g, mask=m, other=0) != 0
        scale = tl.load(scale0_ptr + g, mask=m, other=0.0)
        best = tl.zeros((BLOCK,), tl.float32)
        a2 = tl.arange(0, 2)[None, :]
        win = tl.full((BLOCK, 2), CODE0, tl.int32)
        og = tl.full((BLOCK, 2), 1, tl.int32)
        tiny = tl.full((1,), 1.1754943508222875e-38, tl.float32)
        sh = 2 * (a16 % 8)
        hi = (a16 >= 8).to(tl.int32)
        for s in tl.static_range(0, S):
            wi = tl.load(win_ptr + (s * G + g[:, None]) * 2 + hi, mask=m[:, None], other=0)
            code = tl.load(code_ptr + wi, mask=m[:, None], other=0).to(tl.int32)
            q = (2 * ((code >> sh) & 3) + 1).to(tl.float32)
            sumqx = _tree16(w * x * q, BLOCK)
            sumq2 = _tree16(w * q * q, BLOCK)
            acc = act & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            ns = tld.div_rn(sumqx, tl.maximum(sumq2, tiny))
            scale = tl.where(acc, ns, scale)
            best = tl.where(acc, ns * sumqx, best)
            w2 = tl.load(win_ptr + (s * G + g[:, None]) * 2 + a2, mask=m[:, None], other=0)
            o2 = tl.load(og_ptr + (s * G + g[:, None]) * 2 + a2, mask=m[:, None], other=1)
            win = tl.where(acc[:, None], w2, win)
            og = tl.where(acc[:, None], o2.to(tl.int32), og)
        tl.store(scale_out + g, scale, mask=m)
        tl.store(win_out + g[:, None] * 2 + a2, win, mask=m[:, None])
        tl.store(og_out + g[:, None] * 2 + a2, og, mask=m[:, None])

    @triton.jit
    def _search_kernel(idx_ptr, u_ptr, xv_ptr, wv_ptr, sc_ptr, neigh64_ptr, nlen_ptr,
                       code_ptr, win_ptr, N, M, GRID: tl.constexpr, W4: tl.constexpr,
                       BLOCK: tl.constexpr):
        """iq2_find_best_neighbour over a compacted list of off-grid (step, subgroup) problems.
        Candidates load four at a time as one int64; grid values decode from the code table.
        M = subgroups per step; sc_ptr holds (steps, M//2) per-group scales."""
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = r < N
        a8 = tl.arange(0, 8)[None, :]
        p = tl.load(idx_ptr + r, mask=m, other=0)        # flat (step, subgroup) ids
        step = p // M
        sub = p - step * M
        off8 = sub[:, None] * 8 + a8
        xv = tl.load(xv_ptr + off8, mask=m[:, None], other=0.0)
        wv = tl.load(wv_ptr + off8, mask=m[:, None], other=0.0)
        sc = tl.load(sc_ptr + step * (M // 2) + (sub >> 1), mask=m, other=1.0)
        u = tl.load(u_ptr + p, mask=m, other=0)
        lens = tl.where(m, tl.load(nlen_ptr + u).to(tl.int32), 0)
        maxlen = tl.max(lens)
        best = tl.full((BLOCK,), float("inf"), tl.float32)
        bestc = tl.zeros((BLOCK,), dtype=tl.int32)
        for t4 in range(0, maxlen, 4):
            q4 = tl.load(neigh64_ptr + u * W4 + t4 // 4, mask=t4 < lens, other=-1)
            for j in tl.static_range(4):
                cand = ((q4 >> (16 * j)) & 0xFFFF).to(tl.int32)
                valid = (cand < GRID) & (t4 + j < lens)  # 0xFFFF == the -1 padding
                code = tl.load(code_ptr + tl.where(valid, cand, 0)).to(tl.int32)
                gv = (2 * ((code[:, None] >> (2 * a8)) & 3) + 1).to(tl.float32)
                df = sc[:, None] * gv - xv
                d2 = tl.sum(wv * df * df, axis=1)
                take = valid & (d2 < best)               # strict <, keeps first
                best = tl.where(take, d2, best)
                bestc = tl.where(take, cand, bestc)
        tl.store(win_ptr + p, bestc, mask=m)


def _search_eager(u, xv, wv, sc, T: Tables) -> torch.Tensor:
    """Best neighbour per off-grid problem, bucketed by candidate-list length."""
    lens = T.nlen[u.long()]
    gi = torch.empty_like(u, dtype=torch.int32)
    prev = 0
    for width in T.buckets:
        sel = ((lens > prev) & (lens <= width)).nonzero(as_tuple=True)[0]
        prev = width
        if sel.numel() == 0:
            continue
        sub = max(1, (1 << 23) // width)
        for s0 in range(0, sel.numel(), sub):
            idx = sel[s0 : s0 + sub]
            cand = T.neigh[u[idx].long(), :width].long()
            gv = T.grid_f[cand.clamp(min=0)]
            diff = sc[idx, None, None] * gv - xv[idx][:, None, :]
            d2 = (wv[idx][:, None, :] * diff * diff).sum(-1)
            d2.masked_fill_(cand < 0, float("inf"))
            j = d2.argmin(-1)                            # first min == C's strict <
            gi[idx] = cand.gather(1, j[:, None]).squeeze(1).to(torch.int32)
    return gi


def search_codes(u_all, xv, wv, sc_all, T: Tables, sel=None):
    """Winner grid index per (step, subgroup) problem plus the kmap-hit flag.
    u_all: (S,G,2) codes; xv, wv: (2G,8); sc_all: (S,G) per-group scales."""
    S, G, _ = u_all.shape
    M = 2 * G
    u = u_all.reshape(-1).to(torch.int32)
    win = T.kmap[u.long()].to(torch.int32)               # >= 0 iff on-grid
    og = (win >= 0).reshape(S, G, 2)
    offm = win < 0
    if sel is not None:
        offm &= sel.reshape(-1)
    off = offm.nonzero(as_tuple=True)[0]
    if off.numel():
        sc_all = sc_all.contiguous()
        if u.device.type == "cuda" and HAS_TRITON:
            off32 = off.to(torch.int32)
            BLOCK = 128
            _search_kernel[((off32.numel() + BLOCK - 1) // BLOCK,)](
                off32, u, xv, wv, sc_all, T.neigh64, T.nlen, T.codes, win,
                off32.numel(), M, GRID=T.grid_size, W4=T.neigh64.shape[1],
                BLOCK=BLOCK, num_warps=4)
        else:
            step = off // M
            sub = off - step * M
            win[off] = _search_eager(u[off], xv[sub], wv[sub],
                                     sc_all.reshape(S, G)[step, sub >> 1], T)
    return win.reshape(S, G, 2), og


def _codes(xval, inv, T: Tables):
    """Codes of the tentative levels clamp(round((id*x - 1)/2), 0, 2), per subgroup of 8."""
    lev = torch.clamp(torch.round(0.5 * (inv[:, None] * xval - 1.0)), 0, 2).long()
    return (lev.reshape(-1, 2, 8) << T.shifts2).sum(-1)  # (G,2)


def scale_search(xval, w, waux, active, T: Tables):
    """The shared per-16 search: sweep + accept chain + off-grid reprojection + refit.
    xval, w, waux: (G,16); returns (scale (G,), win (G,2) grid indices)."""
    G = xval.shape[0]
    dev = xval.device
    gmax = xval.max(-1).values
    gsafe = torch.where(active, gmax, torch.ones_like(gmax))
    xv = xval.reshape(-1, 8)
    wv = waux.reshape(-1, 8)

    # the 19 candidate scales depend only on gmax, so all searches batch into one launch
    S = len(SWEEP_CS)
    if dev.type == "cuda" and HAS_TRITON:
        u_all = torch.empty(S, G, 2, dtype=torch.int32, device=dev)
        sc_all = torch.empty(S, G, dtype=torch.float32, device=dev)
        BLOCK = 128
        grid = ((G + BLOCK - 1) // BLOCK,)
        _codes_kernel[grid](xval.contiguous(), gsafe, _steps_for(dev), u_all, sc_all, G,
                            S=S, BLOCK=BLOCK, num_warps=4)
        win_all, og_all = search_codes(u_all, xv, wv, sc_all, T)
        scale0 = torch.where(active, gmax / 5.0, torch.zeros_like(gmax))
        scale = torch.empty_like(scale0)
        win = torch.empty(G, 2, dtype=torch.int32, device=dev)
        og8 = torch.empty(G, 2, dtype=torch.int32, device=dev)
        _sweep_kernel[grid](xval.contiguous(), w.contiguous(), active.to(torch.uint8),
                            scale0, win_all.contiguous(), og_all.to(torch.uint8).contiguous(),
                            T.codes, scale, win, og8, G,
                            CODE0=T.code0, S=S, BLOCK=BLOCK, num_warps=4)
        og = og8 != 0
    else:
        invs = [c / gsafe for c in SWEEP_CS]
        u_all = torch.stack([_codes(xval, inv, T) for inv in invs])
        sc_all = torch.stack([1.0 / inv for inv in invs])
        win_all, og_all = search_codes(u_all, xv, wv, sc_all, T)

        scale = torch.where(active, gmax / 5.0, torch.zeros_like(gmax))
        best = torch.zeros_like(scale)
        win = torch.full((G, 2), T.code0, dtype=torch.int32, device=dev)
        og = torch.ones(G, 2, dtype=torch.bool, device=dev)
        for st in range(S):
            q = T.grid_f[win_all[st].long()].reshape(G, 16)
            sumqx = tree_sum(w * xval * q)
            sumq2 = tree_sum(w * q * q)
            acc = active & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            newscale = sumqx / sumq2.clamp(min=_TINY)
            scale = torch.where(acc, newscale, scale)
            best = torch.where(acc, newscale * sumqx, best)
            win = torch.where(acc[:, None], win_all[st], win)
            og = torch.where(acc[:, None], og_all[st], og)

    # re-project only the subgroups that ended off-grid, then refit the scale
    need = active & (scale > 0) & ~og.all(-1)
    inv = 1.0 / torch.where(scale > 0, scale, torch.ones_like(scale))
    u2 = _codes(xval, inv, T)
    selm = need[:, None] & ~og
    if selm.any():
        win2, _ = search_codes(u2[None], xv, wv, scale[None], T, sel=selm.reshape(1, -1))
        win = torch.where(selm, win2[0], win)
        q = T.grid_f[win.long()].reshape(G, 16)
        sumqx = tree_sum(w * xval * q)
        sumq2 = tree_sum(w * q * q)
        scale = torch.where(need & (sumq2 > 0), sumqx / sumq2.clamp(min=_TINY), scale)
    return scale, win.long()


def pack_scales(scales_g, ns: int):
    """d = max_scale/31 and the per-16 4-bit scale pairs shared by both block layouts."""
    sc = scales_g.reshape(ns, QK_K // 16)
    max_scale = sc.max(-1).values
    has = max_scale > 0
    d = max_scale / 31.0
    inv_d = 1.0 / torch.where(has, d, torch.ones_like(d))
    l = torch.clamp(torch.round(0.5 * (inv_d[:, None] * sc - 1.0)), 0, 15).long()
    l = torch.where(has[:, None], l, torch.zeros_like(l))
    sbytes = (l[:, 0::2] | (l[:, 1::2] << 4)).to(torch.uint8)      # (ns, 8)
    return has, d, sbytes
