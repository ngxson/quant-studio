#!/usr/bin/env python3
"""quant-studio: GPU-accelerated GGUF quantizer.

Streams row chunks from a mmap'd source GGUF, quantizes them on the GPU (CUDA / MPS / CPU fallback) and appends the packed blocks to the output file, so peak memory stays within --mem for any model size.
On CUDA the pipeline is double-buffered with pinned staging: the mmap->RAM copy of the next chunk overlaps the GPU work of the current one, and file writes happen on a separate thread after a CUDA event fires.

Usage:
    python quant-studio.py in.gguf out.gguf q4_0 --mem 4G
    python quant-studio.py in.gguf out.gguf iq2_xxs --mem 4G --imatrix imatrix.gguf --token-embedding-type q4_0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# use a local gguf-py copy if present, else the llama.cpp checkout next to this repo
for _p in (Path(__file__).parent / "gguf-py",
           Path(__file__).parent / ".." / "llama.cpp" / "gguf-py"):
    if _p.is_dir():
        sys.path.insert(0, str(_p.resolve()))
        break

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGML_QUANT_VERSION, LlamaFileType

from kernels import QUANT_TYPES, TORCH_DTYPES

# name-based exclusions from llama-quant.cpp (subset seen so far, extend as needed)
EXCLUDE_SUBSTRINGS = (
    "_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight",
    "altup", "laurel", "per_layer_model_proj", "ssm_conv1d",
    "shortconv.conv.weight", "indexer.k_proj.weight", "indexer.q_proj.weight",
    "time_mix_", "attn_rel_b.weight",
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse_mem(s: str) -> int:
    s = s.strip().upper()
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if s and s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_quantizable(tensor: gguf.ReaderTensor) -> bool:
    """Simplified eligibility rules from llama-quant.cpp; block-size compatibility is checked per target type."""
    name = tensor.name
    return (
        name.endswith("weight")
        and not any(sub in name for sub in EXCLUDE_SUBSTRINGS)
        and len(tensor.shape) >= 2
        and tensor.tensor_type in TORCH_DTYPES
    )


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    """Load a GGUF imatrix: per tensor, mean squared input activation per column (in_sum2 / counts)."""
    r = GGUFReader(path)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[: -len(".in_sum2")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
        elif t.name.endswith(".counts"):
            counts[t.name[: -len(".counts")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
    out = {}
    for name, s in sums.items():
        c = counts.get(name)
        if c is None:
            continue
        e = s.reshape(c.size, -1)
        w = np.zeros_like(e)
        np.divide(e, c[:, None], out=w, where=c[:, None] > 0)
        out[name] = w.reshape(-1)
    return out


def fmt_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:8.1f} {unit}" if unit != "B" else f"{n:8d} B  "
        n /= 1024
    return f"{n} B"


# ---------------------------------------------------------------------------
# output writer thread
# ---------------------------------------------------------------------------


class OutputWriter:
    """Serializes all output-file writes on one worker thread so the main thread can keep feeding the GPU.
    Submission order == write order."""

    def __init__(self, fout, alignment: int):
        self.fout = fout
        self.alignment = alignment
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futs: deque = deque()

    def submit(self, fn):
        while self.futs and self.futs[0].done():
            self.futs.popleft().result()  # surface any write error
        fut = self.pool.submit(fn)
        self.futs.append(fut)
        return fut

    def pad_to_alignment(self):
        def _pad():
            pos = self.fout.tell()
            pad = GGUFWriter.ggml_pad(pos, self.alignment) - pos
            if pad:
                self.fout.write(bytes(pad))
        self.submit(_pad)

    def flush(self):
        while self.futs:
            self.futs.popleft().result()


# ---------------------------------------------------------------------------
# quantize paths
# ---------------------------------------------------------------------------


def tensor_as_rows(tensor: gguf.ReaderTensor) -> np.ndarray:
    """(n_rows, row_bytes) uint8 view of the source mmap, higher dims folded in."""
    return tensor.data.reshape(-1, tensor.data.shape[-1]).view(np.uint8)


def out_row_bytes_for(k: int, ggml_type: GGMLQuantizationType) -> int:
    block, type_size = GGML_QUANT_SIZES[ggml_type]
    return k // block * type_size


class CudaPipeline:
    """Double-buffered mmap -> pinned -> GPU -> pinned -> disk pipeline.

    With two buffer slots the CPU-side copy of chunk i+1 overlaps the GPU work of chunk i.
    File writes wait on a CUDA event inside the OutputWriter thread, never on the main thread.
    The slot state spans tensors, so small tensors do not drain the pipeline.
    """

    def __init__(self, device: torch.device, out: OutputWriter):
        self.device = device
        self.out = out
        self.slot = 0
        self.in_pin: list[torch.Tensor | None] = [None, None]
        self.out_pin: list[torch.Tensor | None] = [None, None]
        self.h2d_events: list[torch.cuda.Event | None] = [None, None]
        self.write_futs: list = [None, None]

    @staticmethod
    def _grow(bufs: list, i: int, nbytes: int) -> torch.Tensor:
        if bufs[i] is None or bufs[i].numel() < nbytes:
            bufs[i] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        return bufs[i][:nbytes]

    def quantize_tensor(self, tensor: gguf.ReaderTensor, quant_fn,
                        dst_type: GGMLQuantizationType, rows_per_chunk: int) -> int:
        src_dtype = TORCH_DTYPES[tensor.tensor_type]
        data = tensor_as_rows(tensor)
        n_rows, in_row_bytes = data.shape
        k = int(tensor.shape[0])
        out_row_bytes = out_row_bytes_for(k, dst_type)
        written = 0

        for i0 in range(0, n_rows, rows_per_chunk):
            n = min(rows_per_chunk, n_rows - i0)
            s = self.slot
            self.slot ^= 1

            # reclaim this slot's buffers from two chunks ago before reuse
            if self.write_futs[s] is not None:
                self.write_futs[s].result()
            if self.h2d_events[s] is not None:
                self.h2d_events[s].synchronize()

            in_pin = self._grow(self.in_pin, s, n * in_row_bytes)
            np.copyto(in_pin.numpy().reshape(n, in_row_bytes), data[i0 : i0 + n])
            x = in_pin.to(self.device, non_blocking=True)
            self.h2d_events[s] = ev_in = torch.cuda.Event()
            ev_in.record()

            x = x.view(src_dtype).reshape(n, k).to(torch.float32)
            packed = quant_fn(x).reshape(-1)

            out_pin = self._grow(self.out_pin, s, n * out_row_bytes)
            out_pin.copy_(packed, non_blocking=True)
            ev_out = torch.cuda.Event()
            ev_out.record()

            out_np = out_pin.numpy()

            def _write(ev=ev_out, buf=out_np):
                ev.synchronize()
                self.out.fout.write(buf.data)

            self.write_futs[s] = self.out.submit(_write)
            written += n * out_row_bytes
        return written


def quantize_tensor_simple(
    tensor: gguf.ReaderTensor, quant_fn, rows_per_chunk: int,
    device: torch.device, out: OutputWriter,
) -> int:
    """Plain chunked path for MPS / CPU."""
    src_dtype = TORCH_DTYPES[tensor.tensor_type]
    data = tensor_as_rows(tensor)
    k = int(tensor.shape[0])
    written = 0
    for i0 in range(0, data.shape[0], rows_per_chunk):
        rows = np.array(data[i0 : i0 + rows_per_chunk])  # copy out of the mmap
        n = rows.shape[0]
        x = torch.from_numpy(rows).view(src_dtype).reshape(n, k).to(device).to(torch.float32)
        packed = quant_fn(x).cpu().numpy()
        out.submit(lambda buf=packed: out.fout.write(buf.data))
        written += packed.nbytes
    return written


def copy_tensor(tensor: gguf.ReaderTensor, mem_bytes: int, out: OutputWriter) -> int:
    """Pass a tensor through untouched; the writer thread reads the mmap."""
    flat = tensor.data.reshape(-1).view(np.uint8)
    step = max(1 << 20, mem_bytes)
    for off in range(0, flat.shape[0], step):
        out.submit(lambda a=flat[off : off + step]: out.fout.write(a.tobytes()))
    return int(flat.shape[0])


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------


def copy_metadata(reader: GGUFReader, writer: GGUFWriter, ftype: LlamaFileType) -> None:
    overrides = {
        gguf.Keys.General.FILE_TYPE: (int(ftype), GGUFValueType.UINT32),
        gguf.Keys.General.QUANTIZATION_VERSION: (GGML_QUANT_VERSION, GGUFValueType.UINT32),
    }
    for field in reader.fields.values():
        # architecture is re-added by GGUFWriter; GGUF.* are virtual fields
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in overrides:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for key, (val, vtype) in overrides.items():
        writer.add_key_value(key, val, vtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU-accelerated GGUF quantizer")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("qtype", choices=sorted(QUANT_TYPES), help="target quantization type")
    ap.add_argument("--mem", default="4G", help="memory budget per chunk, e.g. 4G / 512M")
    ap.add_argument("--device", default="auto", help="torch device (auto/cuda/mps/cpu)")
    ap.add_argument("--imatrix", type=Path, default=None,
                    help="GGUF importance matrix (required for iq2_xxs, used when given for q4_k)")
    ap.add_argument("--token-embedding-type", choices=sorted(QUANT_TYPES), default=None,
                    help="override type for token_embd.weight (like llama-quantize)")
    args = ap.parse_args()

    spec = QUANT_TYPES[args.qtype]
    mem_bytes = parse_mem(args.mem)
    device = pick_device(args.device)
    print(f"quant-studio: {args.input} -> {args.output} [{args.qtype}] "
          f"device={device.type} mem={args.mem}")

    imatrix: dict[str, np.ndarray] = {}
    if args.imatrix is not None:
        imatrix = load_imatrix(args.imatrix)
        print(f"imatrix: {len(imatrix)} entries from {args.imatrix}")
    elif spec.needs_imatrix:
        sys.exit(f"error: {args.qtype} requires --imatrix")

    reader = GGUFReader(args.input, mode="r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        sys.exit("error: input has no general.architecture key")
    arch = arch_field.contents()

    # decide the target type + kernel for every tensor
    plans = []  # (tensor, dst_type|None, make_kernel_args|None, bytes_per_elem)
    for tensor in reader.tensors:
        name = tensor.name
        k = int(tensor.shape[0])
        dst = None
        tspec = None
        if is_quantizable(tensor):
            tspec = spec
            if name == "token_embd.weight" and args.token_embedding_type is not None:
                tspec = QUANT_TYPES[args.token_embedding_type]
            elif name == "token_embd.weight" and spec.needs_imatrix and name not in imatrix:
                sys.exit(f"error: {name} has no imatrix entry; pass "
                         f"--token-embedding-type q4_0 (llama-quantize needs this too)")
            elif spec.needs_imatrix and name not in imatrix:
                sys.exit(f"error: missing imatrix entry for {name} "
                         f"in a very low-bit quantization")
            block = GGML_QUANT_SIZES[tspec.ggml_type][0]
            if k % block != 0:
                if spec.needs_imatrix:
                    sys.exit(f"error: {name} ncols {k} not divisible by {block} "
                             f"(type fallback not implemented)")
                tspec = None  # q4_0 mode: just pass through, matches our old rule
            if tspec is not None:
                dst = tspec.ggml_type
        plans.append((tensor, dst, tspec))

    writer = GGUFWriter(None, arch=arch)
    copy_metadata(reader, writer, spec.ftype)

    # register all tensor infos first (the header needs them before any data)
    for tensor, dst, _tspec in plans:
        if dst is not None:
            k = int(tensor.shape[0])
            row_b = out_row_bytes_for(k, dst)
            nbytes = tensor.n_elements // k * row_b
            byte_shape = (*[int(d) for d in reversed(tensor.shape[1:])], row_b)
            writer.add_tensor_info(tensor.name, byte_shape, np.uint8, nbytes, raw_dtype=dst)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file(path=args.output)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    fout = writer.fout[0]

    out = OutputWriter(fout, writer.data_alignment)
    pipeline = CudaPipeline(device, out) if device.type == "cuda" else None

    total_in = total_out = 0
    t_start = time.time()
    for i, (tensor, dst, tspec) in enumerate(plans):
        shape_str = "x".join(str(int(d)) for d in tensor.shape)
        src = tensor.tensor_type.name
        k = int(tensor.shape[0])

        out.pad_to_alignment()
        if dst is not None:
            qw = None
            if tspec.uses_imatrix and tensor.name in imatrix:
                qw = torch.from_numpy(imatrix[tensor.name][:k].copy()).to(device)
            quant_fn = tspec.make_kernel(device, qw)
            rows_per_chunk = max(1, mem_bytes // (k * tspec.bytes_per_elem))
            if pipeline is not None:
                nbytes = pipeline.quantize_tensor(tensor, quant_fn, dst, rows_per_chunk)
            else:
                nbytes = quantize_tensor_simple(tensor, quant_fn, rows_per_chunk, device, out)
        else:
            nbytes = copy_tensor(tensor, mem_bytes, out)
        total_in += tensor.n_bytes
        total_out += nbytes
        print(f"[{i + 1:4d}/{len(plans)}] {tensor.name:48s} {shape_str:>16s}  "
              f"{src:>5s} -> {dst.name if dst else src:<8s} "
              f"{fmt_size(tensor.n_bytes)} -> {fmt_size(nbytes)}")

    out.pad_to_alignment()
    out.flush()
    writer.close()

    dt = time.time() - t_start
    print(f"done in {dt:.1f}s: {fmt_size(total_in)} -> {fmt_size(total_out)} "
          f"({total_out / max(total_in, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
