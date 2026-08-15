#!/usr/bin/env python3
"""End-to-end test: build a synthetic GGUF, quantize it with quant-studio,
and verify bit-exact parity with gguf-py's reference q4_0 implementation.

Run: python tests/test_q4_0.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".." / "llama.cpp" / "gguf-py"))

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import quantize as ref_quantize

rng = np.random.default_rng(42)


def build_input(path: Path) -> dict[str, np.ndarray]:
    """Write a small GGUF with a mix of tensor types. Returns the f32 view of
    every tensor that should end up quantized, keyed by name."""
    w = GGUFWriter(path, arch="llama")
    w.add_context_length(2048)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_bool("test.flag", True)
    w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    expect_q4: dict[str, np.ndarray] = {}

    # f32 2D - big enough that --mem 64K forces several chunks
    a = rng.standard_normal((200, 128)).astype(np.float32)
    w.add_tensor("blk.0.attn_q.weight", a)
    expect_q4["blk.0.attn_q.weight"] = a

    # f16 2D
    b = rng.standard_normal((96, 64)).astype(np.float16)
    w.add_tensor("blk.0.ffn_up.weight", b)
    expect_q4["blk.0.ffn_up.weight"] = b.astype(np.float32)

    # bf16 2D - stored as raw bytes with an explicit BF16 dtype
    c16 = torch.from_numpy(rng.standard_normal((80, 96)).astype(np.float32)).to(torch.bfloat16)
    c_bytes = c16.view(torch.uint8).numpy()
    w.add_tensor("blk.0.attn_k.weight", c_bytes, raw_dtype=GGMLQuantizationType.BF16)
    expect_q4["blk.0.attn_k.weight"] = c16.to(torch.float32).numpy()

    # f16 3D (experts) - rows are the last numpy axis
    d = rng.standard_normal((4, 16, 64)).astype(np.float16)
    w.add_tensor("blk.0.ffn_gate_exps.weight", d)
    expect_q4["blk.0.ffn_gate_exps.weight"] = d.astype(np.float32)

    # must all be passed through untouched:
    w.add_tensor("blk.0.attn_norm.weight", rng.standard_normal(128).astype(np.float32))
    w.add_tensor("blk.0.odd.weight", rng.standard_normal((10, 40)).astype(np.float32))  # 40 % 32 != 0
    w.add_tensor("rope_freqs.weight", rng.standard_normal(32).astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return expect_q4


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="quant-studio-test-"))
    src, dst = tmp / "in.gguf", tmp / "out.gguf"
    expect_q4 = build_input(src)

    # tiny --mem so every 2D tensor is processed in multiple row chunks
    subprocess.run(
        [sys.executable, str(ROOT / "quant-studio.py"), str(src), str(dst), "q4_0", "--mem", "64K"],
        check=True,
    )

    r_in = GGUFReader(src)
    r_out = GGUFReader(dst)
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(("  ok  " if cond else "  FAIL") + f"  {msg}")
        failures += 0 if cond else 1

    # metadata preserved + file_type/quant_version updated
    check(r_out.get_field("general.architecture").contents() == "llama", "arch preserved")
    check(r_out.get_field("llama.context_length").contents() == 2048, "uint32 kv preserved")
    check(abs(r_out.get_field("llama.attention.layer_norm_rms_epsilon").contents() - 1e-5) < 1e-12,
          "float kv preserved")
    check(r_out.get_field("test.flag").contents() is True, "bool kv preserved")
    check(r_out.get_field("tokenizer.ggml.tokens").contents() == ["<s>", "</s>", "hello"],
          "string array kv preserved")
    check(r_out.get_field("general.file_type").contents() == int(gguf.LlamaFileType.MOSTLY_Q4_0),
          "file_type == MOSTLY_Q4_0")
    check(r_out.get_field("general.quantization_version").contents() == gguf.GGML_QUANT_VERSION,
          "quantization_version set")

    out_tensors = {t.name: t for t in r_out.tensors}
    check([t.name for t in r_out.tensors] == [t.name for t in r_in.tensors], "tensor order preserved")

    for t_in in r_in.tensors:
        t_out = out_tensors[t_in.name]
        check(list(t_out.shape) == list(t_in.shape), f"{t_in.name}: logical shape preserved")
        if t_in.name in expect_q4:
            check(t_out.tensor_type == GGMLQuantizationType.Q4_0, f"{t_in.name}: type is Q4_0")
            ref = ref_quantize(expect_q4[t_in.name], GGMLQuantizationType.Q4_0)
            got = t_out.data
            check(got.nbytes == ref.nbytes and np.array_equal(got.reshape(-1), ref.reshape(-1)),
                  f"{t_in.name}: bit-exact vs gguf-py reference")
        else:
            check(t_out.tensor_type == t_in.tensor_type, f"{t_in.name}: type unchanged")
            check(np.array_equal(t_out.data.reshape(-1).view(np.uint8),
                                 t_in.data.reshape(-1).view(np.uint8)),
                  f"{t_in.name}: bytes identical")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURES'} (files in {tmp})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
