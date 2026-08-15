#!/usr/bin/env python3
"""Python adaptation of llama.cpp's quantization scheme logic.

Mirrors QUANT_OPTIONS from tools/quantize/quantize.cpp and the per-tensor type
selection from src/llama-quant.cpp: tensor_allows_quantization(),
tensor_get_category(), llama_tensor_get_type_impl(), tensor_type_fallback(),
llama_ftype_get_default_type() and tensor_requires_imatrix().

Reference: llama.cpp commit 9871d51fd7a6a4fb300100ed236f3962e72fd6ee (2026-08-15).
When upstream moves, diff the files above against that commit and update both
this module and LLAMA_CPP_COMMIT.

A scheme is marked as supported when quant-studio has a kernel for its default
tensor type; a mixed scheme can still ask for extra types on some tensors
(e.g. Q4_K_M uses Q6_K for part of the model), so decide_types() reports the
exact per-tensor plan and the driver can reject a model whose plan needs a
kernel we do not have yet.

Print the scheme table with: python -m kernels.scheme
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto

from gguf import GGMLQuantizationType as GT
from gguf.constants import GGML_QUANT_SIZES, LlamaFileType as FT

LLAMA_CPP_COMMIT = "9871d51fd7a6a4fb300100ed236f3962e72fd6ee"

# ---------------------------------------------------------------------------
# QUANT_OPTIONS mirror (tools/quantize/quantize.cpp)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantOption:
    name: str
    ftype: FT
    desc: str

    @property
    def default_type(self) -> GT:
        return ftype_default_type(self.ftype)

    @property
    def supported(self) -> bool:
        """True when quant-studio has a kernel for the scheme's default tensor type."""
        if self.name == "COPY":
            return False
        dt = self.default_type
        return dt is not None and dt in _studio_tensor_types()


QUANT_OPTIONS: list[QuantOption] = [
    QuantOption("Q1_0",      FT.MOSTLY_Q1_0,      "1.125 bpw quantization"),
    QuantOption("Q2_0",      FT.MOSTLY_Q2_0,      "2.25 bpw quantization (group 64)"),
    QuantOption("Q4_0",      FT.MOSTLY_Q4_0,      "4.34G, +0.4685 ppl @ Llama-3-8B"),
    QuantOption("Q4_1",      FT.MOSTLY_Q4_1,      "4.78G, +0.4511 ppl @ Llama-3-8B"),
    QuantOption("MXFP4_MOE", FT.MOSTLY_MXFP4_MOE, "MXFP4 MoE"),
    QuantOption("Q5_0",      FT.MOSTLY_Q5_0,      "5.21G, +0.1316 ppl @ Llama-3-8B"),
    QuantOption("Q5_1",      FT.MOSTLY_Q5_1,      "5.65G, +0.1062 ppl @ Llama-3-8B"),
    QuantOption("IQ2_XXS",   FT.MOSTLY_IQ2_XXS,   "2.06 bpw quantization"),
    QuantOption("IQ2_XS",    FT.MOSTLY_IQ2_XS,    "2.31 bpw quantization"),
    QuantOption("IQ2_S",     FT.MOSTLY_IQ2_S,     "2.5  bpw quantization"),
    QuantOption("IQ2_M",     FT.MOSTLY_IQ2_M,     "2.7  bpw quantization"),
    QuantOption("IQ1_S",     FT.MOSTLY_IQ1_S,     "1.56 bpw quantization"),
    QuantOption("IQ1_M",     FT.MOSTLY_IQ1_M,     "1.75 bpw quantization"),
    QuantOption("TQ1_0",     FT.MOSTLY_TQ1_0,     "1.69 bpw ternarization"),
    QuantOption("TQ2_0",     FT.MOSTLY_TQ2_0,     "2.06 bpw ternarization"),
    QuantOption("Q2_K",      FT.MOSTLY_Q2_K,      "2.96G, +3.5199 ppl @ Llama-3-8B"),
    QuantOption("Q2_K_S",    FT.MOSTLY_Q2_K_S,    "2.96G, +3.1836 ppl @ Llama-3-8B"),
    QuantOption("IQ3_XXS",   FT.MOSTLY_IQ3_XXS,   "3.06 bpw quantization"),
    QuantOption("IQ3_S",     FT.MOSTLY_IQ3_S,     "3.44 bpw quantization"),
    QuantOption("IQ3_M",     FT.MOSTLY_IQ3_M,     "3.66 bpw quantization mix"),
    QuantOption("Q3_K",      FT.MOSTLY_Q3_K_M,    "alias for Q3_K_M"),
    QuantOption("IQ3_XS",    FT.MOSTLY_IQ3_XS,    "3.3 bpw quantization"),
    QuantOption("Q3_K_S",    FT.MOSTLY_Q3_K_S,    "3.41G, +1.6321 ppl @ Llama-3-8B"),
    QuantOption("Q3_K_M",    FT.MOSTLY_Q3_K_M,    "3.74G, +0.6569 ppl @ Llama-3-8B"),
    QuantOption("Q3_K_L",    FT.MOSTLY_Q3_K_L,    "4.03G, +0.5562 ppl @ Llama-3-8B"),
    QuantOption("IQ4_NL",    FT.MOSTLY_IQ4_NL,    "4.50 bpw non-linear quantization"),
    QuantOption("IQ4_XS",    FT.MOSTLY_IQ4_XS,    "4.25 bpw non-linear quantization"),
    QuantOption("Q4_K",      FT.MOSTLY_Q4_K_M,    "alias for Q4_K_M"),
    QuantOption("Q4_K_S",    FT.MOSTLY_Q4_K_S,    "4.37G, +0.2689 ppl @ Llama-3-8B"),
    QuantOption("Q4_K_M",    FT.MOSTLY_Q4_K_M,    "4.58G, +0.1754 ppl @ Llama-3-8B"),
    QuantOption("Q5_K",      FT.MOSTLY_Q5_K_M,    "alias for Q5_K_M"),
    QuantOption("Q5_K_S",    FT.MOSTLY_Q5_K_S,    "5.21G, +0.1049 ppl @ Llama-3-8B"),
    QuantOption("Q5_K_M",    FT.MOSTLY_Q5_K_M,    "5.33G, +0.0569 ppl @ Llama-3-8B"),
    QuantOption("Q6_K",      FT.MOSTLY_Q6_K,      "6.14G, +0.0217 ppl @ Llama-3-8B"),
    QuantOption("Q8_0",      FT.MOSTLY_Q8_0,      "7.96G, +0.0026 ppl @ Llama-3-8B"),
    QuantOption("F16",       FT.MOSTLY_F16,       "14.00G, +0.0020 ppl @ Mistral-7B"),
    QuantOption("BF16",      FT.MOSTLY_BF16,      "14.00G, -0.0050 ppl @ Mistral-7B"),
    QuantOption("F32",       FT.ALL_F32,          "26.00G              @ 7B"),
    QuantOption("COPY",      FT.ALL_F32,          "only copy tensors, no quantizing"),
]


def _studio_tensor_types() -> set[GT]:
    """Tensor types quant-studio has kernels for, straight from the registry."""
    from . import QUANT_TYPES
    return {spec.ggml_type for spec in QUANT_TYPES.values()}


def parse_scheme(name: str) -> QuantOption | None:
    for opt in QUANT_OPTIONS:
        if opt.name.upper() == name.upper():
            return opt
    return None


# ---------------------------------------------------------------------------
# type helpers
# ---------------------------------------------------------------------------

_NOT_QUANTIZED = {GT.F32, GT.F16, GT.BF16, GT.F64, GT.I8, GT.I16, GT.I32, GT.I64}


def ggml_is_quantized(t: GT) -> bool:
    return t not in _NOT_QUANTIZED


def ggml_blck_size(t: GT) -> int:
    return GGML_QUANT_SIZES[t][0]


def ftype_default_type(ftype: FT) -> GT | None:
    """llama_ftype_get_default_type()."""
    table = {
        FT.MOSTLY_Q4_0: GT.Q4_0,
        FT.MOSTLY_Q4_1: GT.Q4_1,
        FT.MOSTLY_Q5_0: GT.Q5_0,
        FT.MOSTLY_Q5_1: GT.Q5_1,
        FT.MOSTLY_Q8_0: GT.Q8_0,
        FT.MOSTLY_F16: GT.F16,
        FT.MOSTLY_BF16: GT.BF16,
        FT.ALL_F32: GT.F32,
        FT.MOSTLY_Q1_0: GT.Q1_0,
        FT.MOSTLY_Q2_0: GT.Q2_0,
        FT.MOSTLY_MXFP4_MOE: GT.MXFP4,
        FT.MOSTLY_Q2_K_S: GT.Q2_K,
        FT.MOSTLY_Q2_K: GT.Q2_K,
        FT.MOSTLY_IQ3_XS: GT.IQ3_S,
        FT.MOSTLY_Q3_K_S: GT.Q3_K,
        FT.MOSTLY_Q3_K_M: GT.Q3_K,
        FT.MOSTLY_Q3_K_L: GT.Q3_K,
        FT.MOSTLY_Q4_K_S: GT.Q4_K,
        FT.MOSTLY_Q4_K_M: GT.Q4_K,
        FT.MOSTLY_Q5_K_S: GT.Q5_K,
        FT.MOSTLY_Q5_K_M: GT.Q5_K,
        FT.MOSTLY_Q6_K: GT.Q6_K,
        FT.MOSTLY_TQ1_0: GT.TQ1_0,
        FT.MOSTLY_TQ2_0: GT.TQ2_0,
        FT.MOSTLY_IQ2_XXS: GT.IQ2_XXS,
        FT.MOSTLY_IQ2_XS: GT.IQ2_XS,
        FT.MOSTLY_IQ2_S: GT.IQ2_XS,   # sic: the IQ2_S scheme defaults to IQ2_XS tensors
        FT.MOSTLY_IQ2_M: GT.IQ2_S,
        FT.MOSTLY_IQ3_XXS: GT.IQ3_XXS,
        FT.MOSTLY_IQ1_S: GT.IQ1_S,
        FT.MOSTLY_IQ1_M: GT.IQ1_M,
        FT.MOSTLY_IQ4_NL: GT.IQ4_NL,
        FT.MOSTLY_IQ4_XS: GT.IQ4_XS,
        FT.MOSTLY_IQ3_S: GT.IQ3_S,
        FT.MOSTLY_IQ3_M: GT.IQ3_S,
    }
    return table.get(ftype)


# ---------------------------------------------------------------------------
# tensor categories (tensor_get_category)
# ---------------------------------------------------------------------------


class Category(Enum):
    OUTPUT = auto()
    TOKEN_EMBD = auto()
    ATTENTION_QKV = auto()
    ATTENTION_KV_B = auto()
    ATTENTION_V = auto()
    ATTENTION_K = auto()
    ATTENTION_Q = auto()
    ATTENTION_OUTPUT = auto()
    FFN_UP = auto()
    FFN_GATE = auto()
    FFN_DOWN = auto()
    OTHER = auto()


def _is_token_embd(name: str) -> bool:
    return name in ("token_embd.weight", "per_layer_token_embd.weight")


def tensor_get_category(name: str) -> Category:
    if name == "output.weight":
        return Category.OUTPUT
    if _is_token_embd(name):
        return Category.TOKEN_EMBD
    if "attn_qkv.weight" in name:
        return Category.ATTENTION_QKV
    if "attn_kv_b.weight" in name:
        return Category.ATTENTION_KV_B
    if "attn_v.weight" in name:
        return Category.ATTENTION_V
    if "attn_k.weight" in name:
        return Category.ATTENTION_K
    if "attn_q.weight" in name:
        return Category.ATTENTION_Q
    if "attn_output.weight" in name:
        return Category.ATTENTION_OUTPUT
    if "ffn_up" in name:
        return Category.FFN_UP
    if "ffn_gate" in name:
        return Category.FFN_GATE
    if "ffn_down" in name:
        return Category.FFN_DOWN
    return Category.OTHER


def weight_name_key(name: str) -> tuple[int, str]:
    """llama_model_loader's weight_name_comparer: order by blk layer number, then name.
    The loader processes and writes tensors in this order, and the counter-based
    mixture rules depend on it, so the driver must iterate the same way."""
    m = re.match(r"blk\.(\d+)\.", name)
    return (int(m.group(1)) if m else -1, name)


def category_is_attn_v(cat: Category) -> bool:
    """Attention-v-like tensors, more sensitive to quantization."""
    return cat in (Category.ATTENTION_V, Category.ATTENTION_QKV, Category.ATTENTION_KV_B)


# name fragments that must never be quantized (tensor_allows_quantization)
_NEVER_QUANTIZE_SUBSTR = (
    "_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight",
    "altup", "laurel", "per_layer_model_proj",
    "ssm_conv1d", "shortconv.conv.weight",
    "indexer.k_proj.weight", "indexer.q_proj.weight",
    "time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight", "time_mix_w2.weight",
    "time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight",
    "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight",
    "time_mix_g1.weight", "time_mix_g2.weight",
    "time_mix_decay_w1.weight", "time_mix_decay_w2.weight", "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_",
    ".rel_pos", ".patch_embd", ".patch_merger",
    "a.rvq.codebook", "mm.a.code_embd",
)


def tensor_allows_quantization(name: str, n_dims: int, quantize_output_tensor: bool = True,
                               only_copy: bool = False) -> bool:
    """tensor_allows_quantization(): should this tensor be quantized at all?"""
    if only_copy or n_dims < 2:
        return False
    if not name.endswith("weight"):
        return False
    if not quantize_output_tensor and name == "output.weight":
        return False
    # BERT positional embeddings / token types (LLM_TN resolves to these names)
    if name in ("position_embd.weight", "token_types.weight"):
        return False
    return not any(s in name for s in _NEVER_QUANTIZE_SUBSTR)


# ---------------------------------------------------------------------------
# model info + selection state
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    arch: str = ""
    n_layer: int = 0
    n_expert: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    # C reads llama_model.type; upstream derives 70B from llama models with 80 layers
    is_70b: bool = False

    @property
    def n_gqa(self) -> int:
        return self.n_head // self.n_head_kv if self.n_head_kv > 0 else 0

    @classmethod
    def from_reader(cls, reader) -> "ModelInfo":
        """Build from a gguf GGUFReader's metadata."""
        arch = reader.get_field("general.architecture").contents()

        def kv(suffix, default=0):
            f = reader.get_field(f"{arch}.{suffix}")
            if f is None:
                return default
            v = f.contents()
            # per-layer arrays: use the first layer like hparams does for layer 0
            return int(v[0]) if isinstance(v, list) else int(v)

        n_layer = kv("block_count")
        n_head = kv("attention.head_count")
        return cls(
            arch=arch,
            n_layer=n_layer,
            n_expert=kv("expert_count"),
            n_head=n_head,
            n_head_kv=kv("attention.head_count_kv", n_head),
            is_70b=(arch == "llama" and n_layer == 80),
        )


@dataclass
class QuantState:
    """quantize_state_impl: counters that make the selection layer-position aware."""
    n_attention_wv: int = 0
    n_ffn_down: int = 0
    n_ffn_gate: int = 0
    n_ffn_up: int = 0
    i_attention_wv: int = 0
    i_ffn_down: int = 0
    i_ffn_gate: int = 0
    i_ffn_up: int = 0
    n_fallback: int = 0
    has_imatrix: bool = False
    has_tied_embeddings: bool = True  # assume tied until we see output.weight
    tensor_type_patterns: list = field(default_factory=list)  # [(compiled regex, GT)]


def init_state(model: ModelInfo, tensor_names: list[str], has_imatrix: bool,
               tt_overrides: list[tuple[str, GT]] | None = None) -> QuantState:
    """init_quantize_state_counters() + constructor bits."""
    qs = QuantState(has_imatrix=has_imatrix)
    for name in tensor_names:
        if category_is_attn_v(tensor_get_category(name)):
            qs.n_attention_wv += 1
        if name == "output.weight":
            qs.has_tied_embeddings = False
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = model.n_layer
    if tt_overrides:
        qs.tensor_type_patterns = [(re.compile(p), t) for p, t in tt_overrides]
    return qs


# ---------------------------------------------------------------------------
# per-tensor type selection (llama_tensor_get_type and friends)
# ---------------------------------------------------------------------------


def _use_more_bits(i_layer: int, n_layers: int) -> bool:
    return i_layer < n_layers // 8 or i_layer >= 7 * n_layers // 8 or (i_layer - n_layers // 8) % 3 == 2


def _layer_info(i_layer: int, n_layer: int, name: str, n_expert: int) -> tuple[int, int]:
    """MoE expert tensors are not consecutive, so the layer comes from the tensor name."""
    if n_expert > 1:
        m = re.match(r"blk\.(\d+)\.", name)
        if m is None:
            raise ValueError(f"failed to determine layer for tensor {name}")
        i_layer = int(m.group(1))
        if not 0 <= i_layer < n_layer:
            raise ValueError(f"bad layer {i_layer} for tensor {name}")
    return i_layer, n_layer


def tensor_type_fallback(qs: QuantState, name: str, ncols: int, target: GT) -> GT:
    """tensor_type_fallback(): incompatible shapes drop to a smaller-block type."""
    if ncols % ggml_blck_size(target) == 0:
        return target
    qs.n_fallback += 1
    table = {
        GT.IQ1_S: GT.IQ4_NL, GT.IQ1_M: GT.IQ4_NL, GT.IQ2_XXS: GT.IQ4_NL,
        GT.IQ2_XS: GT.IQ4_NL, GT.IQ2_S: GT.IQ4_NL, GT.IQ3_XXS: GT.IQ4_NL,
        GT.IQ3_S: GT.IQ4_NL, GT.IQ4_XS: GT.IQ4_NL,
        GT.Q2_0: GT.Q4_0, GT.Q2_K: GT.Q4_0, GT.Q3_K: GT.Q4_0,
        GT.TQ1_0: GT.Q4_0, GT.TQ2_0: GT.Q4_0,
        GT.Q4_K: GT.Q5_0, GT.Q5_K: GT.Q5_1, GT.Q6_K: GT.Q8_0,
    }
    if target not in table:
        raise ValueError(f"no tensor type fallback is defined for type {target.name}")
    ret = table[target]
    if ncols % ggml_blck_size(ret) != 0:
        ret = GT.F16  # very rare: first dimension not divisible by 32
    print(f"warning: {name}: ncols {ncols} not divisible by {ggml_blck_size(target)} "
          f"(required for {target.name}) -> falling back to {ret.name}", file=sys.stderr)
    return ret


def _get_type_impl(qs: QuantState, model: ModelInfo, new_type: GT, name: str,
                   ne: tuple[int, ...], ftype: FT, category: Category,
                   token_embedding_type: GT | None, output_tensor_type: GT | None) -> GT:
    """llama_tensor_get_type_impl(): the standard mixture logic."""
    arch = model.arch
    n_expert = max(1, model.n_expert)
    nx = ne[0]
    ne2 = ne[2] if len(ne) > 2 else 1

    if category == Category.OUTPUT or (qs.has_tied_embeddings and category == Category.TOKEN_EMBD):
        if output_tensor_type is not None:
            new_type = output_tensor_type
        else:
            qk_k = ggml_blck_size(new_type)
            if ftype == FT.MOSTLY_MXFP4_MOE:
                new_type = GT.Q8_0
            elif arch == "falcon" or nx % qk_k != 0:
                new_type = GT.Q8_0
            elif ftype in (FT.MOSTLY_IQ2_XXS, FT.MOSTLY_IQ2_XS, FT.MOSTLY_IQ3_XXS,
                           FT.MOSTLY_IQ1_S, FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M, FT.MOSTLY_IQ1_M):
                new_type = GT.Q5_K
            elif new_type != GT.Q8_0:
                new_type = GT.Q6_K
    elif ftype == FT.MOSTLY_MXFP4_MOE:
        # MoE tensors -> MXFP4, other tensors -> Q8_0
        new_type = GT.MXFP4 if ne2 > 1 else GT.Q8_0
    elif category == Category.TOKEN_EMBD:
        if token_embedding_type is not None:
            new_type = token_embedding_type
        elif ftype in (FT.MOSTLY_IQ2_XXS, FT.MOSTLY_IQ2_XS, FT.MOSTLY_IQ1_S, FT.MOSTLY_IQ1_M):
            new_type = GT.Q2_K
        elif ftype in (FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M):
            new_type = GT.IQ3_S
        elif ftype == FT.MOSTLY_IQ3_XXS:
            new_type = GT.IQ3_S
        elif ftype in (FT.MOSTLY_TQ1_0, FT.MOSTLY_TQ2_0, FT.MOSTLY_Q2_0):
            new_type = GT.Q4_K
    elif ftype in (FT.MOSTLY_IQ2_XXS, FT.MOSTLY_IQ2_XS, FT.MOSTLY_IQ1_S,
                   FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M, FT.MOSTLY_IQ1_M):
        if category_is_attn_v(category):
            if model.n_gqa >= 4 or model.n_expert >= 4:
                new_type = GT.Q4_K
            else:
                new_type = GT.IQ3_S if ftype in (FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M) else GT.Q2_K
            qs.i_attention_wv += 1
        elif model.n_expert == 8 and category == Category.ATTENTION_K:
            new_type = GT.Q4_K
        elif category == Category.FFN_DOWN:
            if qs.i_ffn_down < qs.n_ffn_down // 8:
                new_type = GT.IQ3_S if ftype in (FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M) else GT.Q2_K
            qs.i_ffn_down += 1
        elif category == Category.ATTENTION_OUTPUT:
            if model.n_expert == 8:
                new_type = GT.Q5_K
            elif ftype in (FT.MOSTLY_IQ1_S, FT.MOSTLY_IQ1_M):
                new_type = GT.IQ2_XXS
            elif ftype in (FT.MOSTLY_IQ2_S, FT.MOSTLY_IQ2_M):
                new_type = GT.IQ3_S
    elif category_is_attn_v(category):
        if ftype == FT.MOSTLY_Q2_K:
            new_type = GT.Q4_K if model.n_gqa >= 4 else GT.Q3_K
        elif ftype == FT.MOSTLY_Q2_K_S and model.n_gqa >= 4:
            new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_IQ3_XXS:
            new_type = GT.Q4_K if model.n_gqa >= 4 else (GT.IQ3_S if not qs.has_imatrix else GT.IQ3_XXS)
        elif ftype in (FT.MOSTLY_IQ3_XS, FT.MOSTLY_IQ3_S) and model.n_gqa >= 4:
            new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_IQ3_M:
            new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_Q3_K_M:
            new_type = GT.Q5_K if qs.i_attention_wv < 2 else GT.Q4_K
        elif ftype == FT.MOSTLY_Q3_K_L:
            new_type = GT.Q5_K
        elif ftype in (FT.MOSTLY_IQ4_NL, FT.MOSTLY_IQ4_XS) and model.n_gqa >= 4:
            new_type = GT.Q5_K
        elif ftype in (FT.MOSTLY_Q4_K_M, FT.MOSTLY_Q5_K_M) and \
                _use_more_bits(qs.i_attention_wv, qs.n_attention_wv):
            new_type = GT.Q6_K
        elif ftype == FT.MOSTLY_Q4_K_S and qs.i_attention_wv < 4:
            new_type = GT.Q5_K
        if model.is_70b:
            # 8 heads share attn_v, so the tensor is 8x smaller than attn_q:
            # more bits here are nearly free
            if new_type in (GT.Q3_K, GT.Q4_K):
                new_type = GT.Q5_K
        if model.n_expert == 8:
            new_type = GT.Q8_0
        qs.i_attention_wv += 1
    elif category == Category.ATTENTION_K:
        if model.n_expert == 8:
            new_type = GT.Q8_0
        elif ftype == FT.MOSTLY_IQ3_XS:
            new_type = GT.IQ3_XXS
        elif ftype == FT.MOSTLY_IQ3_XXS:
            new_type = GT.IQ2_S
    elif category == Category.ATTENTION_Q:
        if ftype == FT.MOSTLY_IQ3_XS:
            new_type = GT.IQ3_XXS
        elif ftype == FT.MOSTLY_IQ3_XXS:
            new_type = GT.IQ2_S
    elif category == Category.FFN_DOWN:
        i_layer, n_layer = _layer_info(qs.i_ffn_down, qs.n_ffn_down, name, n_expert)
        if ftype == FT.MOSTLY_Q2_K:
            new_type = GT.Q3_K
        elif ftype == FT.MOSTLY_Q2_K_S:
            if i_layer < n_layer // 8:
                new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_IQ3_XXS and not qs.has_imatrix:
            new_type = GT.Q4_K if i_layer < n_layer // 8 else GT.Q3_K
        elif ftype == FT.MOSTLY_Q3_K_M:
            if i_layer < n_layer // 16:
                new_type = GT.Q5_K
            elif arch != "falcon" or _use_more_bits(i_layer, n_layer):
                new_type = GT.Q4_K
            else:
                new_type = GT.Q3_K
        elif ftype == FT.MOSTLY_IQ3_M and (i_layer < n_layer // 8 or
                                           (model.n_expert == 8 and _use_more_bits(i_layer, n_layer))):
            new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_Q3_K_L:
            new_type = GT.Q4_K if arch == "falcon" else GT.Q5_K
        elif ftype == FT.MOSTLY_Q4_K_M:
            if arch == "falcon":
                new_type = GT.Q6_K if i_layer < n_layer // 16 else \
                    (GT.Q5_K if _use_more_bits(i_layer, n_layer) else GT.Q4_K)
            elif _use_more_bits(i_layer, n_layer):
                new_type = GT.Q6_K
        elif i_layer < n_layer // 8 and ftype in (FT.MOSTLY_IQ4_NL, FT.MOSTLY_IQ4_XS) and \
                not qs.has_imatrix:
            new_type = GT.Q5_K
        elif ftype == FT.MOSTLY_Q5_K_M and _use_more_bits(i_layer, n_layer):
            new_type = GT.Q6_K
        elif ftype == FT.MOSTLY_Q4_K_S and arch != "falcon" and i_layer < n_layer // 8:
            new_type = GT.Q5_K
        elif ftype in (FT.MOSTLY_Q4_0, FT.MOSTLY_Q5_0) and qs.has_imatrix and i_layer < n_layer // 8:
            # guard against craziness in the first few ffn_down layers
            new_type = GT.Q4_1 if ftype == FT.MOSTLY_Q4_0 else GT.Q5_1
        qs.i_ffn_down += 1
    elif category == Category.ATTENTION_OUTPUT:
        if arch != "falcon":
            if model.n_expert == 8:
                if ftype in (FT.MOSTLY_Q2_K, FT.MOSTLY_IQ3_XS, FT.MOSTLY_IQ3_XXS,
                             FT.MOSTLY_Q3_K_S, FT.MOSTLY_Q3_K_M, FT.MOSTLY_IQ4_NL,
                             FT.MOSTLY_Q4_K_S, FT.MOSTLY_Q4_K_M, FT.MOSTLY_IQ3_S,
                             FT.MOSTLY_IQ3_M, FT.MOSTLY_IQ4_XS):
                    new_type = GT.Q5_K
            elif ftype == FT.MOSTLY_Q2_K:
                new_type = GT.Q3_K
            elif ftype == FT.MOSTLY_IQ3_XXS:
                new_type = GT.IQ3_S
            elif ftype == FT.MOSTLY_Q3_K_M:
                new_type = GT.Q4_K
            elif ftype == FT.MOSTLY_Q3_K_L:
                new_type = GT.Q5_K
            elif ftype == FT.MOSTLY_IQ3_M:
                new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_Q3_K_L:
            new_type = GT.Q4_K
    elif category == Category.ATTENTION_QKV:
        if ftype in (FT.MOSTLY_Q3_K_M, FT.MOSTLY_Q3_K_L, FT.MOSTLY_IQ3_M):
            new_type = GT.Q4_K
        elif ftype == FT.MOSTLY_Q4_K_M:
            new_type = GT.Q5_K
        elif ftype == FT.MOSTLY_Q5_K_M:
            new_type = GT.Q6_K
    elif category == Category.FFN_GATE:
        i_layer, n_layer = _layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name, n_expert)
        if ftype == FT.MOSTLY_IQ3_XS and n_layer // 8 <= i_layer < 7 * n_layer // 8:
            new_type = GT.IQ3_XXS
        qs.i_ffn_gate += 1
    elif category == Category.FFN_UP:
        i_layer, n_layer = _layer_info(qs.i_ffn_up, qs.n_ffn_up, name, n_expert)
        if ftype == FT.MOSTLY_IQ3_XS and n_layer // 8 <= i_layer < 7 * n_layer // 8:
            new_type = GT.IQ3_XXS
        qs.i_ffn_up += 1

    return new_type


def tensor_get_type(qs: QuantState, model: ModelInfo, name: str, ne: tuple[int, ...],
                    src_type: GT, ftype: FT, pure: bool = False,
                    quantize_output_tensor: bool = True, only_copy: bool = False,
                    token_embedding_type: GT | None = None,
                    output_tensor_type: GT | None = None) -> GT:
    """llama_tensor_get_type(): the ggml type this tensor should be quantized to.
    ne is the ggml shape (ne[0] = row length); a passthrough returns src_type.
    Stateful: call in file tensor order so the layer counters advance like the C loop."""
    n_dims = 1
    for i, d in enumerate(ne):  # ggml_n_dims: highest axis with ne > 1, plus one
        if d > 1:
            n_dims = i + 1
    if not tensor_allows_quantization(name, n_dims, quantize_output_tensor, only_copy):
        return src_type
    category = tensor_get_category(name)
    if token_embedding_type is not None and category == Category.TOKEN_EMBD:
        return token_embedding_type
    if output_tensor_type is not None and category == Category.OUTPUT:
        return output_tensor_type

    new_type = ftype_default_type(ftype)
    if new_type is None:
        raise ValueError(f"no default tensor type for ftype {ftype!r}")

    if ggml_is_quantized(new_type):
        manual = False
        for pattern, qtype in qs.tensor_type_patterns:
            if pattern.search(name):
                new_type = qtype
                manual = True
                break
        if not manual and not pure:
            new_type = _get_type_impl(qs, model, new_type, name, ne, ftype, category,
                                      token_embedding_type, output_tensor_type)
        new_type = tensor_type_fallback(qs, name, ne[0], new_type)

    return new_type


def tensor_requires_imatrix(name: str, dst_type: GT, ftype: FT) -> bool:
    """tensor_requires_imatrix(): does quantizing this tensor to dst_type need imatrix data?"""
    if _is_token_embd(name) or name == "output.weight":
        return False
    if dst_type in (GT.IQ3_XXS, GT.IQ2_XXS, GT.IQ2_XS, GT.IQ2_S, GT.IQ1_M, GT.IQ1_S):
        return True
    if dst_type == GT.Q2_K:
        # k-quants do not require imatrix data, except Q2_K tensors inside a Q2_K_S file
        return ftype == FT.MOSTLY_Q2_K_S
    return False


def decide_types(model: ModelInfo, tensors: list[tuple[str, tuple[int, ...], GT]], ftype: FT,
                 has_imatrix: bool = False, pure: bool = False,
                 token_embedding_type: GT | None = None,
                 output_tensor_type: GT | None = None,
                 tt_overrides: list[tuple[str, GT]] | None = None) -> list[GT]:
    """Two-pass plan like llama_model_quantize_impl: init counters over all tensor
    names, then pick a target type per tensor in order. tensors is a list of
    (name, ggml shape ne, source type)."""
    qs = init_state(model, [t[0] for t in tensors], has_imatrix, tt_overrides)
    return [tensor_get_type(qs, model, name, ne, src, ftype, pure=pure,
                            token_embedding_type=token_embedding_type,
                            output_tensor_type=output_tensor_type)
            for name, ne, src in tensors]


def format_scheme_table() -> str:
    """Human-readable scheme list with quant-studio support marks, for the driver's --help/errors."""
    lines = [f"quantization schemes (llama.cpp {LLAMA_CPP_COMMIT[:10]}), "
             f"[x] = supported by quant-studio"]
    for opt in QUANT_OPTIONS:
        mark = "x" if opt.supported else " "
        dt = opt.default_type
        lines.append(f"  [{mark}] {opt.name:10s} default {dt.name if dt is not None else '-':8s} {opt.desc}")
    return "\n".join(lines)


if __name__ == "__main__":  # debug aid only; quant-studio.py is the entrypoint
    print(format_scheme_table())
