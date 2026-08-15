"""Shared pieces for the IQ3 kernels: ports of iq3xs_init_impl() and iq3_find_best_neighbour() from ggml-quants.c.

The 3-bit grids hold 4-value codes (levels 0..7, points 2l+1 in 1..15) over a 4096-entry kmap.
A neighbour list holds every grid point within the nwant smallest distinct squared distances of an off-grid code (nwant = 2 for the 256 grid, 3 for the 512 grid), in the C qsort (distance, grid index) order.
On-grid codes resolve via the kmap; the off-grid subgroups go through one batched search: a Triton kernel on CUDA, a bucketed eager search on CPU/MPS.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import HAS_TRITON, tl, triton
from .iq3_tables import KGRID_3BIT_256, KGRID_3BIT_512

QK_K = 256
KMAP_SIZE = 4096


class Tables:
    def __init__(self, device: torch.device, grid_size: int):
        kgrid = np.asarray(KGRID_3BIT_256 if grid_size == 256 else KGRID_3BIT_512, dtype=np.int64)
        nwant = 2 if grid_size == 256 else 3
        shifts = 3 * np.arange(4)
        grid_l = (kgrid[:, None] >> shifts) & 7                     # (gs,4) levels 0..7
        grid_v = 2 * grid_l + 1                                     # values 1,3,...,15

        kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
        kmap[kgrid] = np.arange(grid_size)

        missing = np.nonzero(kmap < 0)[0]
        pos = 2 * ((missing[:, None] >> shifts) & 7) + 1            # (M,4)
        d2 = ((pos[:, None, :] - grid_v[None, :, :]) ** 2).sum(-1)  # (M,gs)
        order = np.argsort(d2 * 4096 + np.arange(grid_size), axis=1)  # (d2, idx) order
        d2s = np.take_along_axis(d2, order, axis=1)
        # keep points within the nwant smallest distinct distances, like the C nhave walk
        ndist = np.cumsum(np.pad(d2s[:, 1:] > d2s[:, :-1], ((0, 0), (1, 0))), axis=1)
        counts = (ndist < nwant).sum(1)
        maxn = int(counts.max())
        maxn_pad = (maxn + 3) // 4 * 4  # rows 8B-aligned for int64 vector loads

        neigh = np.full((KMAP_SIZE, maxn_pad), -1, dtype=np.int16)
        neigh[missing, :maxn] = np.where(np.arange(maxn)[None, :] < counts[:, None],
                                         order[:, :maxn], -1).astype(np.int16)
        nlen = np.zeros(KMAP_SIZE, dtype=np.int16)
        nlen[missing] = counts.astype(np.int16)

        dev = device
        self.grid_size = grid_size
        self.codes = torch.from_numpy(kgrid.astype(np.int32)).to(dev)        # (gs,)
        self.grid_f = torch.from_numpy(grid_v.astype(np.float32)).to(dev)    # (gs,4)
        self.kmap = torch.from_numpy(kmap).to(dev)
        self.neigh = torch.from_numpy(neigh).to(dev)                         # (4096,maxn_pad)
        self.neigh64 = torch.from_numpy(neigh.view(np.int64)
                                        .reshape(KMAP_SIZE, maxn_pad // 4)).to(dev)
        self.nlen = torch.from_numpy(nlen).to(dev)
        self.shifts3 = (3 * torch.arange(4, device=dev)).long()
        # length buckets for the eager path, so short lists do not pay for the padded max
        self.buckets = tuple(w for w in (8, 16, 48, maxn) if w <= maxn)


def tree_sum32(t: torch.Tensor) -> torch.Tensor:
    """Pairwise tree sum over the last dim (32); explicit order keeps CPU and CUDA bit-identical."""
    for _ in range(5):
        t = t[..., 0::2] + t[..., 1::2]
    return t[..., 0]


_tables_cache: dict[tuple[str, int], Tables] = {}


def tables_for(device: torch.device, grid_size: int) -> Tables:
    key = (str(device), grid_size)
    if key not in _tables_cache:
        _tables_cache[key] = Tables(device, grid_size)
    return _tables_cache[key]


if HAS_TRITON:

    @triton.jit
    def _search_kernel(idx_ptr, u_ptr, xv_ptr, wv_ptr, sc_ptr, neigh64_ptr, nlen_ptr,
                       code_ptr, win_ptr, N, M, GS: tl.constexpr, W4: tl.constexpr,
                       BLOCK: tl.constexpr):
        """iq3_find_best_neighbour over a compacted list of off-grid (step, subgroup) problems.
        Candidates load four at a time as one int64; grid values decode from the code table.
        M = subgroups per step; sc_ptr holds (steps, M//8) per-group scales."""
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = r < N
        p = tl.load(idx_ptr + r, mask=m, other=0)        # flat (step, subgroup) ids
        step = p // M
        sub = p - step * M
        xv0 = tl.load(xv_ptr + sub * 4 + 0, mask=m, other=0.0)
        xv1 = tl.load(xv_ptr + sub * 4 + 1, mask=m, other=0.0)
        xv2 = tl.load(xv_ptr + sub * 4 + 2, mask=m, other=0.0)
        xv3 = tl.load(xv_ptr + sub * 4 + 3, mask=m, other=0.0)
        wv0 = tl.load(wv_ptr + sub * 4 + 0, mask=m, other=0.0)
        wv1 = tl.load(wv_ptr + sub * 4 + 1, mask=m, other=0.0)
        wv2 = tl.load(wv_ptr + sub * 4 + 2, mask=m, other=0.0)
        wv3 = tl.load(wv_ptr + sub * 4 + 3, mask=m, other=0.0)
        sc = tl.load(sc_ptr + step * (M // 8) + (sub >> 3), mask=m, other=1.0)
        u = tl.load(u_ptr + p, mask=m, other=0)
        lens = tl.where(m, tl.load(nlen_ptr + u).to(tl.int32), 0)
        maxlen = tl.max(lens)
        best = tl.full((BLOCK,), float("inf"), tl.float32)
        bestc = tl.zeros((BLOCK,), dtype=tl.int32)
        for t4 in range(0, maxlen, 4):
            q4 = tl.load(neigh64_ptr + u * W4 + t4 // 4, mask=t4 < lens, other=-1)
            for j in tl.static_range(4):
                cand = ((q4 >> (16 * j)) & 0xFFFF).to(tl.int32)
                valid = (cand < GS) & (t4 + j < lens)    # 0xFFFF == the -1 padding
                code = tl.load(code_ptr + tl.where(valid, cand, 0))
                df0 = sc * (2 * (code & 7) + 1).to(tl.float32) - xv0
                df1 = sc * (2 * ((code >> 3) & 7) + 1).to(tl.float32) - xv1
                df2 = sc * (2 * ((code >> 6) & 7) + 1).to(tl.float32) - xv2
                df3 = sc * (2 * ((code >> 9) & 7) + 1).to(tl.float32) - xv3
                # C add order, matching the eager path
                d2 = ((wv0 * df0 * df0 + wv1 * df1 * df1) + wv2 * df2 * df2) + wv3 * df3 * df3
                take = valid & (d2 < best)               # strict <, keeps first
                best = tl.where(take, d2, best)
                bestc = tl.where(take, cand, bestc)
        tl.store(win_ptr + p, bestc, mask=m)


def _search_eager(off: torch.Tensor, u: torch.Tensor, xv, wv, sc_all, win, T: Tables, M: int):
    """Bucketed-by-length eager search for the off-grid problems in `off`."""
    sub = off % M
    step = off // M
    uo = u[off].long()
    sc = sc_all.reshape(-1)[step * (M // 8) + (sub >> 3)]
    xvo = xv[sub]
    wvo = wv[sub]
    lens = T.nlen[uo]
    prev = 0
    for width in T.buckets:
        sel = ((lens > prev) & (lens <= width)).nonzero(as_tuple=True)[0]
        prev = width
        if sel.numel() == 0:
            continue
        chunk = max(1, (1 << 23) // width)
        for s0 in range(0, sel.numel(), chunk):
            idx = sel[s0 : s0 + chunk]
            cand = T.neigh[uo[idx], :width].long()
            gv = T.grid_f[cand.clamp(min=0)]
            diff = sc[idx, None, None] * gv - xvo[idx][:, None, :]
            t = wvo[idx][:, None, :] * diff * diff
            d2 = ((t[..., 0] + t[..., 1]) + t[..., 2]) + t[..., 3]  # C add order
            d2.masked_fill_(cand < 0, float("inf"))
            j = d2.argmin(-1)                            # first min == C's strict <
            win[off[idx]] = cand.gather(1, j[:, None]).squeeze(1).to(torch.int32)


def project(u_all: torch.Tensor, xv: torch.Tensor, wv: torch.Tensor,
            sc_all: torch.Tensor, T: Tables) -> torch.Tensor:
    """Winning grid index per (step, subgroup) problem.
    u_all: (S, M) clamped 12-bit codes; xv, wv: (M, 4); sc_all: (S, M//8) per-group scales."""
    S, M = u_all.shape
    u = u_all.reshape(-1)
    win = T.kmap[u.long()].to(torch.int32)               # >= 0 iff on-grid
    off = (win < 0).nonzero(as_tuple=True)[0]
    if off.numel():
        if u.device.type == "cuda" and HAS_TRITON:
            off32 = off.to(torch.int32)
            BLOCK = 128
            _search_kernel[((off32.numel() + BLOCK - 1) // BLOCK,)](
                off32, u, xv.contiguous(), wv.contiguous(), sc_all.contiguous(),
                T.neigh64, T.nlen, T.codes, win, off32.numel(), M,
                GS=T.grid_size, W4=T.neigh64.shape[1], BLOCK=BLOCK, num_warps=4)
        else:
            _search_eager(off, u, xv, wv, sc_all, win, T, M)
    return win.reshape(S, M)
