#!/usr/bin/env python3
"""Build a tiny synthetic llama-arch GGUF + matching imatrix GGUF that llama-quantize
accepts, for correctness testing of quant-studio kernels without real weights.
Writes tmp/synth.gguf and tmp/synth-imatrix.gguf."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "gguf-py", ROOT / ".." / "llama.cpp" / "gguf-py"):
    if _p.is_dir():
        sys.path.insert(0, str(_p.resolve()))
        break

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType, GGUFWriter

HERE = ROOT / "tmp"
rng = np.random.default_rng(1234)

N_EMBD = 256
N_FF = 512
N_LAYER = 2
N_VOCAB = 512


def bf16_tensor(w: GGUFWriter, name: str, a: np.ndarray) -> None:
    t = torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    w.add_tensor(name, t.view(torch.uint8).numpy(), raw_dtype=GGMLQuantizationType.BF16)


def weight(rows: int, cols: int, scale: float = 0.05) -> np.ndarray:
    # per-row varying magnitude so scales/mins get realistic spread
    a = rng.standard_normal((rows, cols)) * scale
    a *= (0.2 + rng.random((rows, 1)) * 3.0)
    return a.astype(np.float32)


def main() -> None:
    quantizable: list[tuple[str, int]] = []  # (name, k)

    w = GGUFWriter(HERE / "synth.gguf", arch="llama")
    w.add_block_count(N_LAYER)
    w.add_context_length(512)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(8)
    w.add_head_count_kv(8)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(N_EMBD // 8)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(N_VOCAB)])
    w.add_array("tokenizer.ggml.scores", [0.0] * N_VOCAB)
    w.add_array("tokenizer.ggml.token_type", [1] * N_VOCAB)
    w.add_tokenizer_model("llama")

    def add(name: str, rows: int, cols: int) -> np.ndarray:
        a = weight(rows, cols)
        quantizable.append((name, cols))
        return a

    emb = add("token_embd.weight", N_VOCAB, N_EMBD)
    bf16_tensor(w, "token_embd.weight", emb)

    for i in range(N_LAYER):
        p = f"blk.{i}."
        for nm, rows, cols in ((p + "attn_q.weight", N_EMBD, N_EMBD),
                               (p + "attn_k.weight", N_EMBD, N_EMBD),
                               (p + "attn_v.weight", N_EMBD, N_EMBD),
                               (p + "attn_output.weight", N_EMBD, N_EMBD),
                               (p + "ffn_gate.weight", N_FF, N_EMBD),
                               (p + "ffn_up.weight", N_FF, N_EMBD),
                               (p + "ffn_down.weight", N_EMBD, N_FF)):
            a = add(nm, rows, cols)
            if nm == "blk.0.attn_q.weight":
                # adversarial rows: zeros, constants, outlier, mixed-magnitude blocks
                a[0, :] = 0.0
                a[1, :] = 0.5
                a[2, :] = -0.5
                a[3, :] = 1e-6
                a[3, 7] = 8.0
                a[4, :128] = 0.25
                a[5, :] = np.abs(a[5, :])   # all-positive row
                a[6, :] = -np.abs(a[6, :])  # all-negative row
            bf16_tensor(w, nm, a)
        w.add_tensor(p + "attn_norm.weight", weight(1, N_EMBD)[0])
        w.add_tensor(p + "ffn_norm.weight", weight(1, N_EMBD)[0])

    w.add_tensor("output_norm.weight", weight(1, N_EMBD)[0])
    out = add("output.weight", N_VOCAB, N_EMBD)
    bf16_tensor(w, "output.weight", out)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # matching imatrix: positive per-column activations for every quantizable tensor
    wi = GGUFWriter(HERE / "synth-imatrix.gguf", arch="imatrix")
    wi.add_string("general.type", "imatrix")
    wi.add_array("imatrix.datasets", ["synthetic"])
    wi.add_uint32("imatrix.chunk_count", 10)
    wi.add_uint32("imatrix.chunk_size", 512)
    for name, k in quantizable:
        vals = (rng.random(k).astype(np.float32) * 4.0 + 0.05) * 100.0
        wi.add_tensor(name + ".in_sum2", vals)
        wi.add_tensor(name + ".counts", np.array([100.0], dtype=np.float32))
    wi.write_header_to_file()
    wi.write_kv_data_to_file()
    wi.write_tensors_to_file()
    wi.close()
    print(f"wrote synth.gguf ({len(quantizable)} quantizable tensors) + synth-imatrix.gguf")


if __name__ == "__main__":
    main()
