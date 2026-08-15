#!/usr/bin/env python3
"""GPU-vs-CPU parity check for one quant type, on random data. Run on the CUDA box:
  ~/quant-studio/venv/bin/python gpu_check.py <module> <fn> [--imat]
e.g.: gpu_check.py q5_k quantize_q5_k --imat
The kernel fn must have signature fn(x, qw) with qw=None allowed unless --imat-only.
Prints block agreement between the CUDA path and the eager CPU path; expect >= 97%."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _p in (ROOT / "gguf-py", ROOT / ".." / "llama.cpp" / "gguf-py"):
    if _p.is_dir():
        sys.path.insert(0, str(_p.resolve()))
        break

import numpy as np
import torch

mod_name, fn_name = sys.argv[1], sys.argv[2]
use_imat = "--imat" in sys.argv
mod = importlib.import_module(f"kernels.{mod_name}")
fn = getattr(mod, fn_name)

torch.manual_seed(7)
ok = True
for rows, k in ((512, 256), (256, 512), (37, 1024)):
    x = torch.randn(rows, k) * 0.05 * (0.2 + torch.rand(rows, 1) * 3.0)
    x[0] = 0.0
    x[1] = 0.5
    x[2] = -0.5
    x[3] = 1e-6
    x[3, 7] = 8.0
    qw = (torch.rand(k) * 4.0 + 0.05) if use_imat else None
    cpu = fn(x, qw).numpy()
    gpu = fn(x.cuda(), None if qw is None else qw.cuda()).cpu().numpy()
    bb = cpu.shape[1] * rows // (rows * k // 256)  # bytes per 256-elem superblock
    a = cpu.reshape(-1, bb)
    b = gpu.reshape(-1, bb)
    same = float((a == b).all(1).mean())
    ok &= same >= 0.97
    print(f"rows={rows} k={k} imat={use_imat}: gpu-vs-cpu identical blocks {same * 100:.2f}%")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
