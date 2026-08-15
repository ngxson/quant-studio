"""IQ3_S quantization for quant-studio: a torch/Triton port of quantize_row_iq3_s_impl(block_size=32) from ggml-quants.c.

Uses the 512-entry grid (qh holds the 9th index bit) with explicit sign bytes, no parity fold.
Unlike iq3_xxs, the post-sweep reprojection replaces every subgroup, and skipped all-zero groups do not advance the C qs/signs write pointers, so live groups compact toward the front of the block.
Grid tables and the neighbour search live in iq3_common; reductions are tree-ordered, so an occasional ULP-level tie-break can differ from the CPU reference.
"""

from __future__ import annotations

import numpy as np
import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import F32_TINY, QuantSpec
from .iq3_common import QK_K, project, tables_for, tree_sum32

BLOCK_BYTES = 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 + QK_K // 64  # 110
_TINY = F32_TINY
# candidate iscale numerators (2*kMaxQ-1 + is*0.2f), rounded per step like the C fp32 chain
_STEPS = [float(np.float32(15.0) + np.float32(s) * np.float32(0.2)) for s in range(-9, 10)]


def _codes(xval: torch.Tensor, inv: torch.Tensor, sh3: torch.Tensor) -> torch.Tensor:
    """12-bit codes of the tentative levels clamp(round((id*x - 1)/2), 0, 7)."""
    lev = torch.clamp(torch.round(0.5 * (inv[:, None] * xval - 1.0)), 0, 7).long()
    return (lev.reshape(-1, 8, 4) << sh3).sum(-1)        # (G,8)


def quantize_iq3_s(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 110) uint8 on the same device."""
    dev = x.device
    T = tables_for(dev, 512)
    n, k = x.shape
    ns = n * k // QK_K
    G = ns * (QK_K // 32)
    xbl = x.reshape(ns, QK_K)
    if qw is None:
        weight = xbl * xbl
    else:
        sigma2 = 2.0 * (xbl * xbl).sum(-1) / QK_K
        qw_bl = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
        weight = qw_bl * torch.sqrt(sigma2[:, None] + xbl * xbl)
    w = weight.reshape(G, 32)
    waux = torch.sqrt(w)
    xb = xbl.reshape(G, 32)

    # signs are kept explicitly, no parity fold
    x8 = xb.reshape(G, 4, 8)
    neg = x8 < 0
    bit = (1 << torch.arange(8, device=dev)).long()
    block_signs = (neg.long() * bit).sum(-1)             # (G,4) full 8-bit masks
    xval = x8.abs().reshape(G, 32)

    gmax = xval.max(-1).values
    active = gmax > 0
    gsafe = torch.where(active, gmax, torch.ones_like(gmax))
    xv = xval.reshape(-1, 4)                             # (M,4) subgroup views
    wv = waux.reshape(-1, 4)

    # the 19 candidate scales depend only on the group max, so all searches batch into one launch
    invs = [c / gsafe for c in _STEPS]
    u_all = torch.stack([_codes(xval, inv, T.shifts3).reshape(-1) for inv in invs]).to(torch.int32)
    sc_all = torch.stack([1.0 / inv for inv in invs])
    win_all = project(u_all, xv, wv, sc_all, T)          # (19, G*8)
    on_all = T.kmap[u_all.long()] >= 0

    scale = gmax / 15.0
    best = torch.zeros_like(scale)
    Wg = torch.zeros(G, 8, dtype=torch.int32, device=dev)  # grid entry 0 == all-zero levels
    ongrid = torch.zeros(G, 8, dtype=torch.bool, device=dev)
    for st in range(len(_STEPS)):
        win = win_all[st].reshape(G, 8)
        q = T.grid_f[win.long()].reshape(G, 32)
        sumqx = tree_sum32(w * xval * q)
        sumq2 = tree_sum32(w * q * q)
        acc = active & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        newscale = sumqx / sumq2.clamp(min=_TINY)
        scale = torch.where(acc, newscale, scale)
        best = torch.where(acc, newscale * sumqx, best)
        Wg = torch.where(acc[:, None], win, Wg)
        ongrid = torch.where(acc[:, None], on_all[st].reshape(G, 8), ongrid)

    # every subgroup reprojects with the accepted scale, then one least-squares refit
    redo = active & ongrid.logical_not().any(-1) & (scale > 0)
    inv2 = 1.0 / torch.where(redo, scale, torch.ones_like(scale))
    u2 = _codes(xval, inv2, T.shifts3).reshape(1, -1).to(torch.int32)
    win2 = project(u2, xv, wv, scale.reshape(1, G), T).reshape(G, 8)
    Wg = torch.where(redo[:, None], win2, Wg)
    q = T.grid_f[Wg.long()].reshape(G, 32)
    sumqx = tree_sum32(w * xval * q)
    sumq2 = tree_sum32(w * q * q)
    refit = redo & (sumq2 > 0)
    scale = torch.where(refit, sumqx / sumq2.clamp(min=_TINY), scale)

    # negative least-squares scale: flip the scale and all the signs
    negs = active & (scale < 0)
    scale = torch.where(negs, -scale, scale)
    block_signs = torch.where(negs[:, None], (~block_signs) & 255, block_signs)

    gi = Wg.long().reshape(ns, 8, 8)
    act = active.reshape(ns, 8)
    ar8 = torch.arange(8, device=dev).long()
    qs_g = gi & 255                                      # (ns,8,8) index bytes per group
    qh_g = ((gi >> 8) << ar8).sum(-1)                    # (ns,8) 9th-bit byte per group
    sg_g = block_signs.reshape(ns, 8, 4)

    # skipped groups do not advance the C qs/signs pointers, so live groups compact to the front
    slot = torch.cumsum(act.long(), dim=1) - 1
    slot = torch.where(act, slot, torch.full_like(slot, 8))  # dummy slot 8 is discarded
    qs_buf = torch.zeros(ns, 9, 8, dtype=torch.long, device=dev)
    qs_buf.scatter_(1, slot[:, :, None].expand(-1, -1, 8), qs_g)
    sg_buf = torch.zeros(ns, 9, 4, dtype=torch.long, device=dev)
    sg_buf.scatter_(1, slot[:, :, None].expand(-1, -1, 4), sg_g)
    qs_bytes = qs_buf[:, :8].reshape(ns, QK_K // 4)
    sg_bytes = sg_buf[:, :8].reshape(ns, QK_K // 8)
    qh_bytes = torch.where(act, qh_g, torch.zeros_like(qh_g))  # qh keeps absolute group slots

    # d = max_scale/31 (stored with the C fudge factor), 4-bit scale pairs
    scales_g = torch.where(active, scale, torch.zeros_like(scale))
    scb = scales_g.reshape(ns, QK_K // 32)
    max_scale = scb.max(-1).values
    has = max_scale > 0
    d = max_scale / 31.0
    d16 = torch.where(has, d * 1.033, torch.zeros_like(d)).to(torch.float16)
    inv_d = 1.0 / torch.where(has, d, torch.ones_like(d))
    lnib = torch.clamp(torch.round(0.5 * (inv_d[:, None] * scb - 1.0)), 0, 15).long()
    lnib = torch.where(has[:, None], lnib, torch.zeros_like(lnib))
    lp = lnib.reshape(ns, 4, 2)
    sc_bytes = lp[:, :, 0] | (lp[:, :, 1] << 4)

    dbytes = d16[:, None].contiguous().view(torch.uint8)
    out = torch.cat([dbytes, qs_bytes.to(torch.uint8), qh_bytes.to(torch.uint8),
                     sg_bytes.to(torch.uint8), sc_bytes.to(torch.uint8)], dim=1)  # (ns, 110)
    return out.reshape(n, (k // QK_K) * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    tables_for(device, 512)
    return lambda x: quantize_iq3_s(x, qw)


SPECS = {
    "iq3_s": QuantSpec(GGMLQuantizationType.IQ3_S, LlamaFileType.MOSTLY_IQ3_S,
                       64, False, _make_kernel, uses_imatrix=True),
}
