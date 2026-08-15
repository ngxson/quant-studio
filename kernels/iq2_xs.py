"""IQ2_XS quantization for quant-studio: a torch/Triton port of quantize_row_iq2_xs_impl() from ggml-quants.c.
Uses the 512-entry E8 grid with nwant=2 neighbour lists; blocks hold fp16 d, 32 uint16 words [9-bit grid index | 7-bit signs], and per-16 4-bit scales.
The search core lives in iq2xs_common; this module adds the sign-parity fold and the block packing.
"""

from __future__ import annotations

import torch

from gguf import GGMLQuantizationType
from gguf.constants import LlamaFileType

from .common import QuantSpec
from .iq2xs_common import QK_K, pack_scales, scale_search, tables_for

GROUP_MAX_EPS = 1e-15
BLOCK_BYTES = 2 + QK_K // 4 + QK_K // 32  # fp16 d + 32 uint16 + 8 scale bytes = 74


def quantize_iq2_xs(x: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, k) f32, k % 256 == 0; qw: (k,) f32 imatrix weights; returns (n_rows, k//256 * 74) uint8."""
    n, k = x.shape
    dev = x.device
    T = tables_for("xs", dev)
    ns = n * k // QK_K
    xbl = x.reshape(ns, QK_K)
    sigma2 = (xbl * xbl).sum(-1) / QK_K
    qw_bl = qw.reshape(1, k // QK_K, QK_K).expand(n, -1, -1).reshape(ns, QK_K)
    weight = qw_bl * torch.sqrt(sigma2[:, None] + xbl * xbl)

    G = ns * (QK_K // 16)
    w = weight.reshape(G, 16)
    waux = torch.sqrt(w)

    # signs fold into |x|, forcing even sign-parity per 8 values
    x8 = xbl.reshape(G, 2, 8)
    neg = x8 < 0
    xval8 = x8.abs()
    bit = (1 << torch.arange(8, device=dev)).long()
    s = (neg.long() * bit).sum(-1)                       # (G,2)
    odd = (neg.sum(-1) % 2) == 1
    imin = (w.reshape(G, 2, 8) * x8 * x8).argmin(-1)     # first min, like the C scan
    flip = torch.nn.functional.one_hot(imin, 8).bool() & odd[:, :, None]
    xval8 = torch.where(flip, -xval8, xval8)
    s = s ^ ((1 << imin) * odd.long())
    signs = s & 127
    xval = xval8.reshape(G, 16)

    active = xval.max(-1).values >= GROUP_MAX_EPS
    scale, win = scale_search(xval, w, waux, active, T)

    # negative least-squares scale: flip the scale and all the signs
    negs = active & (scale < 0)
    scale = torch.where(negs, -scale, scale)
    signs = torch.where(negs[:, None], (~signs) & 127, signs)

    q2 = win | (signs << 9)                              # (G,2) uint16 words
    q2 = torch.where(active[:, None], q2, torch.zeros_like(q2))
    scales_g = torch.where(active, scale, torch.zeros_like(scale))

    has, d, sbytes = pack_scales(scales_g, ns)
    d16 = torch.where(has, d, torch.zeros_like(d)).to(torch.float16)
    q2 = q2.reshape(ns, QK_K // 8)
    q2 = torch.where(has[:, None], q2, torch.zeros_like(q2))
    shifts8 = (8 * torch.arange(2, device=dev)).long()
    qbytes = ((q2[..., None] >> shifts8) & 255).to(torch.uint8).reshape(ns, QK_K // 4)
    dbytes = d16[:, None].contiguous().view(torch.uint8)
    out = torch.cat([dbytes, qbytes, sbytes], dim=1)     # (ns, 74)
    return out.reshape(n, (k // QK_K) * BLOCK_BYTES)


def _make_kernel(device: torch.device, qw: torch.Tensor) -> callable:
    tables_for("xs", device)
    return lambda x: quantize_iq2_xs(x, qw)


SPECS = {
    "iq2_xs": QuantSpec(GGMLQuantizationType.IQ2_XS, LlamaFileType.MOSTLY_IQ2_XS,
                        64, True, _make_kernel),
}
