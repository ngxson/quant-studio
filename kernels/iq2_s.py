"""IQ2_S quantization for quant-studio: a torch/Triton port of quantize_row_iq2_s_impl() from ggml-quants.c.
Uses the 1024-entry E8 grid with nwant=1 neighbour lists; signs store explicitly per byte (no parity fold).
Blocks hold fp16 d (scaled by 0.9875 at store), 32 grid-index low bytes, 32 sign bytes, qh with the 2 high grid-index bits per subgroup, and per-16 4-bit scales.
The search core lives in iq2xs_common; this module adds the sign split and the block packing.
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .iq2xs_common import QK_K, pack_scales, scale_search, tables_for

GROUP_MAX_EPS_IQ2_S = 1e-8
BLOCK_BYTES = 2 + QK_K // 4 + QK_K // 16  # fp16 d + qs[64] + qh[8] + scales[8] = 82


def quantize_iq2_s(x: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights; returns (n_rows, k//256 * 82) uint8."""
    n, k = x.shape
    dev = x.device
    T = tables_for("s", dev)
    ns = n * k // QK_K
    xbl = x.reshape(ns, QK_K)
    sigma2 = 2.0 * (xbl * xbl).sum(-1) / QK_K
    qw_bl = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
    weight = qw_bl * torch.sqrt(sigma2[:, None] + xbl * xbl)

    G = ns * (QK_K // 16)
    w = weight.reshape(G, 16)
    waux = torch.sqrt(w)

    x8 = xbl.reshape(G, 2, 8)
    bit = (1 << torch.arange(8, device=dev)).long()
    signs = ((x8 < 0).long() * bit).sum(-1)              # (G,2) full 8-bit signs
    xval = x8.abs().reshape(G, 16)

    active = xval.max(-1).values >= GROUP_MAX_EPS_IQ2_S
    scale, win = scale_search(xval, w, waux, active, T)

    # negative least-squares scale: flip the scale and all the signs
    negs = active & (scale < 0)
    scale = torch.where(negs, -scale, scale)
    signs = torch.where(negs[:, None], (~signs) & 255, signs)

    win = torch.where(active[:, None], win, torch.zeros_like(win))
    signs = torch.where(active[:, None], signs, torch.zeros_like(signs))
    scales_g = torch.where(active, scale, torch.zeros_like(scale))

    has, d, sbytes = pack_scales(scales_g, ns)
    d16 = torch.where(has, d * 0.9875, torch.zeros_like(d)).to(torch.float16)
    win = win.reshape(ns, QK_K // 8)                     # 32 subgroups per block
    lobytes = (win & 255).to(torch.uint8)
    qh = ((win >> 8).reshape(ns, QK_K // 32, 4) << (2 * torch.arange(4, device=dev))).sum(-1)
    qhbytes = qh.to(torch.uint8)
    signbytes = signs.reshape(ns, QK_K // 8).to(torch.uint8)
    dbytes = d16[:, None].contiguous().view(torch.uint8)
    out = torch.cat([dbytes, lobytes, signbytes, qhbytes, sbytes], dim=1)  # (ns, 82)
    return out.reshape(n, (k // QK_K) * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw: torch.Tensor) -> callable:
    tables_for("s", device)
    return lambda x: quantize_iq2_s(x, qw)


SPECS = {
    "iq2_s": QuantSpec(GGMLQuantizationType.IQ2_S, LlamaFileType.MOSTLY_IQ2_S,
                       64, True, _make_kernel),
}
