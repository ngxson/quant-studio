"""IQ3_XXS quantization for quant-studio: a torch/Triton port of quantize_row_iq3_xxs_impl(256) from ggml-quants.c.

Grid tables and the neighbour search live in iq3_common (iq3xs_init_impl / iq3_find_best_neighbour ports).
The search follows the C code step for step, but reductions are tree-ordered, so an occasional ULP-level tie-break can differ from the CPU reference.
"""

from __future__ import annotations

import numpy as np
import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import F32_TINY, HAS_TRITON, QuantSpec
from .iq3_common import QK_K, project, sweep_fused, tables_for, tree_sum32

GROUP_MAX_EPS = 1e-8  # GROUP_MAX_EPS_IQ3_XXS
BLOCK_BYTES = 2 + 3 * QK_K // 8  # fp16 d + 64 index bytes + 8 sign/scale words = 98
_TINY = F32_TINY
# candidate iscale numerators (2*kMaxQ-1 + is*0.2f), rounded per step like the C fp32 chain
_STEPS = [float(np.float32(15.0) + np.float32(s) * np.float32(0.2)) for s in range(-15, 16)]


def _codes(xval: torch.Tensor, inv: torch.Tensor, sh3: torch.Tensor) -> torch.Tensor:
    """12-bit codes of the tentative levels clamp(round((id*x - 1)/2), 0, 7)."""
    lev = torch.clamp(torch.round(0.5 * (inv[:, None] * xval - 1.0)), 0, 7).long()
    return (lev.reshape(-1, 8, 4) << sh3).sum(-1)        # (G,8)


def quantize_iq3_xxs(x: torch.Tensor, qw: torch.Tensor | None) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights or None.
    Returns (n_rows, k//256 * 98) uint8 on the same device."""
    dev = x.device
    T = tables_for(dev, 256)
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

    # signs: fold into |x|, forcing even sign-parity per 8 values
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

    gmax = xval.max(-1).values
    active = gmax >= GROUP_MAX_EPS
    gsafe = torch.where(active, gmax, torch.ones_like(gmax))
    xv = xval.reshape(-1, 4)                             # (M,4) subgroup views
    wv = waux.reshape(-1, 4)

    # the 31 candidate scales depend only on the group max, so all searches batch into one launch
    if dev.type == "cuda" and HAS_TRITON:
        scale, Wg, ongrid = sweep_fused(xval, w, xv, wv, gsafe, active, gmax / 15.0,
                                        _STEPS, T, og_init=1)
    else:
        invs = [c / gsafe for c in _STEPS]
        u_all = torch.stack([_codes(xval, inv, T.shifts3).reshape(-1) for inv in invs]).to(torch.int32)
        sc_all = torch.stack([1.0 / inv for inv in invs])
        win_all, on_all = project(u_all, xv, wv, sc_all, T)  # (31, G*8)

        scale = gmax / 15.0
        best = torch.zeros_like(scale)
        Wg = torch.zeros(G, 8, dtype=torch.int32, device=dev)  # grid entry 0 == all-zero levels
        ongrid = torch.ones(G, 8, dtype=torch.bool, device=dev)
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

    # only off-grid subgroups reproject with the accepted scale, then one least-squares refit
    redo = active & ongrid.logical_not().any(-1) & (scale > 0)
    inv2 = 1.0 / torch.where(redo, scale, torch.ones_like(scale))
    u2 = _codes(xval, inv2, T.shifts3).reshape(1, -1).to(torch.int32)
    win2 = project(u2, xv, wv, scale.reshape(1, G), T)[0].reshape(G, 8)
    Wg = torch.where(redo[:, None] & ~ongrid, win2, Wg)
    q = T.grid_f[Wg.long()].reshape(G, 32)
    sumqx = tree_sum32(w * xval * q)
    sumq2 = tree_sum32(w * q * q)
    refit = redo & (sumq2 > 0)
    scale = torch.where(refit, sumqx / sumq2.clamp(min=_TINY), scale)

    # negative least-squares scale: flip the scale and all the signs
    negs = active & (scale < 0)
    scale = torch.where(negs, -scale, scale)
    block_signs = torch.where(negs[:, None], (~block_signs) & 127, block_signs)

    ar4 = torch.arange(4, device=dev).long()
    qidx = torch.where(active[:, None], Wg.long(), torch.zeros_like(Wg, dtype=torch.long))
    sw = (block_signs << (7 * ar4)).sum(-1)              # (G,) 7-bit signs x4
    sw = torch.where(active, sw, torch.zeros_like(sw))
    scales_g = torch.where(active, scale, torch.zeros_like(scale))

    # superblock: d = max_scale/31 (stored with the C fudge factor), 4-bit scales in the top nibble
    sc = scales_g.reshape(ns, QK_K // 32)
    max_scale = sc.max(-1).values
    has = max_scale > 0
    d = max_scale / 31.0
    d16 = torch.where(has, d * 1.0125, torch.zeros_like(d)).to(torch.float16)
    inv_d = 1.0 / torch.where(has, d, torch.ones_like(d))
    lnib = torch.clamp(torch.round(0.5 * (inv_d[:, None] * sc - 1.0)), 0, 15).long()
    words = sw.reshape(ns, QK_K // 32) | (lnib << 28)
    words = torch.where(has[:, None], words, torch.zeros_like(words))
    qidx = torch.where(has[:, None], qidx.reshape(ns, QK_K // 4), torch.zeros_like(qidx).reshape(ns, QK_K // 4))

    wbytes = ((words[..., None] >> (8 * ar4)) & 255).to(torch.uint8).reshape(ns, QK_K // 8)
    dbytes = d16[:, None].contiguous().view(torch.uint8)
    out = torch.cat([dbytes, qidx.to(torch.uint8), wbytes], dim=1)  # (ns, 98)
    return out.reshape(n, (k // QK_K) * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw) -> callable:
    tables_for(device, 256)
    return lambda x: quantize_iq3_xxs(x, qw)


SPECS = {
    # ggml can quantize iq3_xxs without an imatrix, but llama-quantize refuses; match it
    "iq3_xxs": QuantSpec(GGMLQuantizationType.IQ3_XXS, LlamaFileType.MOSTLY_IQ3_XXS,
                         64, True, _make_kernel),
}
