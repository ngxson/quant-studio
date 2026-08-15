#!/usr/bin/env python3
"""Correctness harness: quantize the synthetic model with llama-quantize (ground truth)
and with quant-studio (CPU), then compare per tensor.

Usage: python tmp/harness.py q2_k [q5_k ...]   (or "all")
Pass criteria per quantized tensor: same type, >= 90% bit-identical blocks,
dequantized weighted-MSE ratio vs llama-quantize <= 1.005, passthroughs byte-exact.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "gguf-py", ROOT / ".." / "llama.cpp" / "gguf-py"):
    if _p.is_dir():
        sys.path.insert(0, str(_p.resolve()))
        break

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as GT
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

HERE = ROOT / "tmp"
LLAMA_QUANTIZE = os.environ.get("LLAMA_QUANTIZE") or next(
    (str(c) for c in (ROOT / ".." / "llama.cpp" / "build" / "bin" / "llama-quantize",
                      Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize") if c.exists()),
    "llama-quantize")

# qtype -> (llama-quantize type name, ggml tensor type, modes)
TYPES = {
    "q2_k":    ("Q2_K",    GT.Q2_K,    ("imat", "noimat")),
    "q3_k":    ("Q3_K",    GT.Q3_K,    ("imat", "noimat")),
    "q4_k":    ("Q4_K",    GT.Q4_K,    ("imat", "noimat")),
    "q5_k":    ("Q5_K",    GT.Q5_K,    ("imat", "noimat")),
    "q6_k":    ("Q6_K",    GT.Q6_K,    ("imat", "noimat")),
    "iq2_xxs": ("IQ2_XXS", GT.IQ2_XXS, ("imat",)),
    "iq2_xs":  ("IQ2_XS",  GT.IQ2_XS,  ("imat",)),
    # llama-quantize maps ftype IQ2_S -> tensor IQ2_XS; tensor IQ2_S comes from ftype IQ2_M
    "iq2_s":   ("IQ2_M",   GT.IQ2_S,   ("imat",)),
    # llama-quantize refuses iq3_xxs without an imatrix, so only the imat mode has ground truth
    "iq3_xxs": ("IQ3_XXS", GT.IQ3_XXS, ("imat",)),
    "iq3_s":   ("IQ3_S",   GT.IQ3_S,   ("imat", "noimat")),
    "iq4_xs":  ("IQ4_XS",  GT.IQ4_XS,  ("imat", "noimat")),
}


def ensure_synth() -> None:
    if not (HERE / "synth.gguf").exists() or not (HERE / "synth-imatrix.gguf").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "dev_make_synth.py")], check=True)


def run_pair(qtype: str, mode: str) -> tuple[Path, Path]:
    lq_name, _, _ = TYPES[qtype]
    ref = HERE / f"synth-ref-{qtype}-{mode}.gguf"
    got = HERE / f"synth-out-{qtype}-{mode}.gguf"
    cmd = [LLAMA_QUANTIZE]
    if mode == "imat":
        cmd += ["--imatrix", str(HERE / "synth-imatrix.gguf")]
    cmd += ["--pure", str(HERE / "synth.gguf"), str(ref), lq_name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"llama-quantize failed for {qtype} {mode}")

    cmd = [sys.executable, str(ROOT / "quant-studio.py"), str(HERE / "synth.gguf"),
           str(got), qtype, "--mem", "1M", "--device", "cpu"]
    if mode == "imat":
        cmd += ["--imatrix", str(HERE / "synth-imatrix.gguf")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"quant-studio failed for {qtype} {mode}")
    return ref, got


def compare(qtype: str, mode: str, ref_path: Path, got_path: Path) -> bool:
    _, gt, _ = TYPES[qtype]
    block_elems, block_bytes = GGML_QUANT_SIZES[gt]
    ref = GGUFReader(ref_path)
    got = {t.name: t for t in GGUFReader(got_path).tensors}
    src = {t.name: t for t in GGUFReader(HERE / "synth.gguf").tensors}
    ok = True
    for tr in ref.tensors:
        tg = got[tr.name]
        tag = f"  {qtype}/{mode} {tr.name:28s}"
        if tg.tensor_type != tr.tensor_type:
            print(f"{tag} FAIL type {tr.tensor_type.name} != {tg.tensor_type.name}")
            ok = False
            continue
        a = np.asarray(tr.data).reshape(-1).view(np.uint8)
        b = np.asarray(tg.data).reshape(-1).view(np.uint8)
        if tr.tensor_type != gt:
            if not np.array_equal(a, b):
                print(f"{tag} FAIL passthrough bytes differ")
                ok = False
            continue
        blocks_a = a.reshape(-1, block_bytes)
        blocks_b = b.reshape(-1, block_bytes)
        same = float((blocks_a == blocks_b).all(1).mean())

        import torch
        k = int(tr.shape[0])
        xf = torch.from_numpy(np.array(src[tr.name].data)).view(torch.bfloat16).float().numpy().reshape(-1, k)
        dq_ref = dequantize(np.asarray(tr.data), gt).reshape(-1, k)
        dq_got = dequantize(np.asarray(tg.data), gt).reshape(-1, k)
        e_ref = float(((dq_ref - xf) ** 2).mean())
        e_got = float(((dq_got - xf) ** 2).mean())
        ratio = e_got / e_ref if e_ref > 0 else (1.0 if e_got == 0 else float("inf"))
        good = same >= 0.90 and ratio <= 1.005 and np.isfinite(dq_got).all()
        ok &= good
        print(f"{tag} {'ok  ' if good else 'FAIL'} blocks {same * 100:6.2f}%  mse ratio {ratio:.6f}")
    return ok


def main() -> None:
    args = sys.argv[1:] or ["all"]
    qtypes = list(TYPES) if args == ["all"] else args
    ensure_synth()
    failed = []
    for qtype in qtypes:
        for mode in TYPES[qtype][2]:
            try:
                ref, got = run_pair(qtype, mode)
                if not compare(qtype, mode, ref, got):
                    failed.append(f"{qtype}/{mode}")
            except Exception as e:
                print(f"  {qtype}/{mode} ERROR: {e}")
                failed.append(f"{qtype}/{mode}")
    print("\n" + ("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
