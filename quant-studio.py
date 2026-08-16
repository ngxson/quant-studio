#!/usr/bin/env python3
"""quant-studio: GPU-accelerated GGUF quantizer.

Streams row chunks from a mmap'd source GGUF, quantizes them on the GPU (CUDA / MPS / CPU fallback) and appends the packed blocks to the output file, so peak memory stays within --mem for any model size.
On CUDA the pipeline is double-buffered with pinned staging: the mmap->RAM copy of the next chunk overlaps the GPU work of the current one, and file writes happen on a separate thread after a CUDA event fires.

Usage:
    python quant-studio.py in.gguf out.gguf Q4_K_M --mem 4G --imatrix imatrix.gguf
    python quant-studio.py in.gguf out.gguf iq2_xxs --pure --mem 4G --imatrix imatrix.gguf --token-embedding-type q4_0
The scheme mixtures, tensor eligibility rules and fallbacks mirror llama-quantize; see kernels/scheme.py.
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

# use a local gguf-py copy if present, else the llama.cpp submodule or a sibling checkout
for _p in (Path(__file__).parent / "gguf-py",
           Path(__file__).parent / "llama.cpp" / "gguf-py",
           Path(__file__).parent / ".." / "llama.cpp" / "gguf-py"):
    if _p.is_dir():
        sys.path.insert(0, str(_p.resolve()))
        break

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGML_QUANT_VERSION, LlamaFileType

from kernels import QUANT_TYPES, TORCH_DTYPES, adaround, scheme

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


def parse_ggml_type(s: str) -> GGMLQuantizationType:
    try:
        return GGMLQuantizationType[s.upper()]
    except KeyError:
        sys.exit(f"error: unknown ggml type: {s}")


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
    ap = argparse.ArgumentParser(
        description="GPU-accelerated GGUF quantizer, drop-in for llama-quantize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=scheme.format_scheme_table())
    ap.add_argument("input", type=Path, help="source GGUF (F32/F16/BF16)")
    ap.add_argument("output", nargs="?", default=None,
                    help="output GGUF (default: ggml-model-<TYPE>.gguf next to the input)")
    ap.add_argument("qtype", nargs="?", default=None, metavar="type",
                    help="quantization scheme, see the table below")
    ap.add_argument("nthreads", nargs="?", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--mem", default="4G", help="memory budget per chunk, e.g. 4G / 512M")
    ap.add_argument("--device", default="auto", help="torch device (auto/cuda/mps/cpu)")
    ap.add_argument("--imatrix", type=Path, default=None, help="GGUF importance matrix")
    ap.add_argument("--ada", type=Path, default=None, metavar="gmatrix",
                    help="AdaRound: optimize weight rounding against this gmatrix file (see cpp/gmatrix.cpp); Q4_0 tensors only")
    ap.add_argument("--ada-iters", type=int, default=500, help="AdaRound optimization steps per chunk")
    ap.add_argument("--pure", action="store_true",
                    help="disable k-quant mixtures and quantize all tensors to the same type")
    ap.add_argument("--leave-output-tensor", action="store_true",
                    help="leave output.weight un(re)quantized")
    ap.add_argument("--output-tensor-type", default=None, metavar="ggml_type",
                    help="use this ggml_type for the output.weight tensor")
    ap.add_argument("--token-embedding-type", default=None, metavar="ggml_type",
                    help="use this ggml_type for the token embeddings tensor")
    ap.add_argument("--tensor-type", action="append", default=[], metavar="name=ggml_type",
                    help="quantize tensors matching this name regex to this type; repeatable")
    ap.add_argument("--tensor-type-file", type=Path, default=None, metavar="file",
                    help="file with name=ggml_type entries, separated by spaces or newlines")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the per-tensor plan and final size without quantizing")
    # llama-quantize options quant-studio recognizes but does not support
    rej = ap.add_argument_group("rejected llama-quantize options")
    rej.add_argument("--allow-requantize", action="store_true", help=argparse.SUPPRESS)
    rej.add_argument("--include-weights", default=None, help=argparse.SUPPRESS)
    rej.add_argument("--exclude-weights", default=None, help=argparse.SUPPRESS)
    rej.add_argument("--prune-layers", default=None, help=argparse.SUPPRESS)
    rej.add_argument("--keep-split", action="store_true", help=argparse.SUPPRESS)
    rej.add_argument("--override-kv", action="append", default=[], help=argparse.SUPPRESS)
    args = ap.parse_args()

    for flag, val in (("--allow-requantize", args.allow_requantize),
                      ("--include-weights", args.include_weights),
                      ("--exclude-weights", args.exclude_weights),
                      ("--prune-layers", args.prune_layers),
                      ("--keep-split", args.keep_split),
                      ("--override-kv", args.override_kv)):
        if val:
            sys.exit(f"error: {flag} is not supported by quant-studio")

    # llama-quantize positional style: the output name may be omitted
    if args.output is not None and scheme.parse_scheme(args.output) is not None and \
            (args.qtype is None or args.qtype.isdigit()):
        args.nthreads = args.qtype
        args.qtype = args.output
        args.output = None
    if args.qtype is None:
        ap.error("missing quantization type")
    if args.nthreads is not None:
        sys.exit("error: the nthreads argument is not supported by quant-studio (GPU pipeline)")

    opt = scheme.parse_scheme(args.qtype)
    if opt is None:
        sys.exit(f"error: unknown quantization scheme {args.qtype!r}\n\n" + scheme.format_scheme_table())
    if not opt.supported:
        sys.exit(f"error: scheme {opt.name} is not supported by quant-studio\n\n"
                 + scheme.format_scheme_table())
    out_path = Path(args.output) if args.output else args.input.parent / f"ggml-model-{opt.name}.gguf"

    tt_entries = list(args.tensor_type)
    if args.tensor_type_file is not None:
        tt_entries += args.tensor_type_file.read_text().split()
    tt_overrides = []
    for ent in tt_entries:
        pat, _, tname = ent.partition("=")
        if not tname:
            sys.exit(f"error: invalid --tensor-type {ent!r}, expected name=ggml_type")
        tt_overrides.append((pat, parse_ggml_type(tname)))
    tet = parse_ggml_type(args.token_embedding_type) if args.token_embedding_type else None
    ott = parse_ggml_type(args.output_tensor_type) if args.output_tensor_type else None

    mem_bytes = parse_mem(args.mem)
    device = pick_device(args.device)
    print(f"quant-studio: {args.input} -> {out_path} [{opt.name}] "
          f"device={device.type} mem={args.mem}")

    imatrix: dict[str, np.ndarray] = {}
    if args.imatrix is not None:
        imatrix = load_imatrix(args.imatrix)
        print(f"imatrix: {len(imatrix)} entries from {args.imatrix}")

    gram = None
    if args.ada is not None:
        gram = adaround.GramFile(args.ada)
        print(f"gmatrix: {len(gram)} entries from {args.ada}")

    reader = GGUFReader(args.input, mode="r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        sys.exit("error: input has no general.architecture key")
    arch = arch_field.contents()

    # decide the target type + kernel for every tensor with the llama-quant.cpp logic;
    # process and write in llama_model_loader order, which the counter-based rules depend on
    tensors = sorted(reader.tensors, key=lambda t: scheme.weight_name_key(t.name))
    model = scheme.ModelInfo.from_reader(reader)
    qs = scheme.init_state(model, [t.name for t in tensors], bool(imatrix), tt_overrides)
    spec_by_type = {s.ggml_type: s for s in QUANT_TYPES.values()}

    plans = []  # (tensor, dst_type|None, spec|None)
    missing: dict[str, list[str]] = {}
    need_imat: list[str] = []
    for tensor in tensors:
        src = tensor.tensor_type
        dst = src
        if src in TORCH_DTYPES:  # requantizing an already-quantized source is rejected above
            ne = tuple(int(d) for d in tensor.shape)
            try:
                dst = scheme.tensor_get_type(
                    qs, model, tensor.name, ne, src, opt.ftype, pure=args.pure,
                    quantize_output_tensor=not args.leave_output_tensor,
                    token_embedding_type=tet, output_tensor_type=ott)
            except ValueError as e:  # no usable shape fallback: keep the source type
                print(f"warning: {tensor.name}: {e}; keeping {src.name}", file=sys.stderr)
        if dst == src:
            plans.append((tensor, None, None))
            continue
        tspec = spec_by_type.get(dst)
        if tspec is None:
            missing.setdefault(dst.name, []).append(tensor.name)
            plans.append((tensor, None, None))
            continue
        if (scheme.tensor_requires_imatrix(tensor.name, dst, opt.ftype) or tspec.needs_imatrix) \
                and tensor.name not in imatrix:
            need_imat.append(tensor.name)
        plans.append((tensor, dst, tspec))

    if missing:
        detail = "; ".join(f"{t} for {len(names)} tensors (e.g. {names[0]})"
                           for t, names in sorted(missing.items()))
        sys.exit(f"error: this scheme needs kernel types quant-studio does not have yet: {detail}")
    if need_imat:
        uniq = sorted(set(need_imat))
        hint = "" if args.imatrix else " (pass --imatrix)"
        if "token_embd.weight" in uniq:
            hint += " (token_embd can be overridden with --token-embedding-type)"
        sys.exit(f"error: {len(uniq)} tensors require importance matrix data{hint}: "
                 + ", ".join(uniq[:4]) + ("..." if len(uniq) > 4 else ""))

    if args.dry_run:
        total_in = total_out = 0
        for tensor, dst, _tspec in plans:
            k = int(tensor.shape[0])
            nbytes = tensor.n_bytes if dst is None else tensor.n_elements // k * out_row_bytes_for(k, dst)
            total_in += tensor.n_bytes
            total_out += nbytes
            print(f"{tensor.name:48s} {tensor.tensor_type.name:>8s} -> "
                  f"{dst.name if dst else tensor.tensor_type.name:<8s} {fmt_size(nbytes)}")
        print(f"dry run: {fmt_size(total_in)} -> {fmt_size(total_out)} tensor data "
              f"({total_out / max(total_in, 1) * 100:.1f}%)")
        return

    writer = GGUFWriter(None, arch=arch)
    copy_metadata(reader, writer, opt.ftype)

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

    writer.write_header_to_file(path=out_path)
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
        ada = False
        if dst is not None:
            qw = None
            if tspec.uses_imatrix and tensor.name in imatrix:
                qw = torch.from_numpy(imatrix[tensor.name][:k].copy()).to(device)
            if gram is not None and dst == GGMLQuantizationType.Q4_0:
                if tensor.name in gram:
                    quant_fn = adaround.make_q4_0_kernel(gram.load(tensor.name, device), iters=args.ada_iters)
                    ada = True
                else:
                    print(f"warning: {tensor.name}: no gmatrix entry, using plain rounding", file=sys.stderr)
            if not ada:
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
        dst_str = (dst.name if dst else src) + ("+ada" if ada else "")
        print(f"[{i + 1:4d}/{len(plans)}] {tensor.name:48s} {shape_str:>16s}  "
              f"{src:>5s} -> {dst_str:<8s} "
              f"{fmt_size(tensor.n_bytes)} -> {fmt_size(nbytes)}")

    out.pad_to_alignment()
    out.flush()
    writer.close()

    dt = time.time() - t_start
    print(f"done in {dt:.1f}s: {fmt_size(total_in)} -> {fmt_size(total_out)} "
          f"({total_out / max(total_in, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
