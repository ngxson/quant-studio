"""AdaRound rounding optimization for Q4_0, driven by --ada.

Ref: Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization".
Each weight learns a soft round-down/up variable; the loss is the layer output
reconstruction error tr(dW G dW^T), where G = X^T X is the Gram matrix of the
input activations collected by cpp/gmatrix.cpp. Rows of W are independent under
this loss, so the chunked streaming pipeline still applies.
"""

from __future__ import annotations

import numpy as np
import torch

from .q4_0 import QK4_0, block_scales, pack_q4_0

GAMMA, ZETA = -0.1, 1.1  # rectified sigmoid bounds
BETA_HI, BETA_LO = 20.0, 2.0


class GramFile:
    """Lazy access to a gmatrix GGUF: weight name -> (d, d) Gram matrix."""

    def __init__(self, path):
        from gguf import GGUFReader
        self.reader = GGUFReader(path)  # keeps the mmap alive
        self.grams = {t.name[: -len(".gram")]: t for t in self.reader.tensors if t.name.endswith(".gram")}
        self.counts = {t.name[: -len(".counts")]: t for t in self.reader.tensors if t.name.endswith(".counts")}

    def __contains__(self, name: str) -> bool:
        return name in self.grams

    def __len__(self) -> int:
        return len(self.grams)

    def load(self, name: str, device: torch.device) -> torch.Tensor:
        t = self.grams[name]
        d = int(t.shape[0])
        G = torch.from_numpy(np.array(t.data, dtype=np.float32).reshape(d, d)).to(device)
        G = (G + G.T) / 2.0  # the backend matmul is not exactly symmetric
        cnt = float(np.array(self.counts[name].data)[0]) if name in self.counts else 0.0
        if cnt > 0:
            G /= cnt
        return G


def _rectified_sigmoid(alpha: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def _rec_loss(q: torch.Tensor, d: torch.Tensor, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    w_hat = ((q - 8.0) * d).reshape(x.shape)
    diff = w_hat - x
    return ((diff @ G) * diff).sum()


def quantize_q4_0_ada(x: torch.Tensor, G: torch.Tensor, iters: int = 500, lr: float = 1e-1,
                      reg: float = 5.0, warmup: float = 0.2) -> torch.Tensor:
    """Same contract as quantize_q4_0, but rounding is optimized against G."""
    n, k = x.shape
    b = x.reshape(n * k // QK4_0, QK4_0)

    with torch.no_grad():
        d, inv_d = block_scales(b)
        t = b * inv_d + 8.0
        q_floor = t.floor()
        q_rtn = (t + 0.5).floor().clamp(0, 15)
        l_rtn = _rec_loss(q_rtn, d, x, G)

        frac = (t - q_floor).clamp(1e-4, 1.0 - 1e-4)
        p = (frac - GAMMA) / (ZETA - GAMMA)
        alpha = torch.log(p / (1.0 - p))

    if iters > 0 and l_rtn > 0:
        alpha.requires_grad_(True)
        opt = torch.optim.Adam([alpha], lr=lr)
        lam = reg * float(l_rtn) / alpha.numel()
        n_warm = int(iters * warmup)
        for it in range(iters):
            h = _rectified_sigmoid(alpha)
            q = (q_floor + h).clamp(0.0, 15.0)
            loss = _rec_loss(q, d, x, G)
            if it >= n_warm:
                beta = BETA_HI - (BETA_HI - BETA_LO) * (it - n_warm) / max(1, iters - n_warm - 1)
                loss = loss + lam * (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        h = _rectified_sigmoid(alpha)
        q = (q_floor + (h > 0.5)).clamp(0, 15).to(torch.uint8)
    return pack_q4_0(d, q, n, k)


def make_q4_0_kernel(G: torch.Tensor, iters: int = 500) -> callable:
    return lambda x: quantize_q4_0_ada(x, G, iters=iters)
