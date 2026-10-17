#!/usr/bin/env python3
"""Generate 7 to 8B models with random weights and real tokenizers, for
docs/benchmarks.md.

Run from the repository root with

    pip download --no-deps -d WHEELS llama-models mistral-common dashscope
    python3 -P tools/gen_big_models.py WHEELS OUTDIR [llama3 mistral qwen2]

Needs NumPy, sentencepiece, the `gguf` package and llama-cpp-python. For
each model named (all three by default) it writes OUTDIR/NAME-rand-q4km.gguf,
quantized to Q4_K_M by llama.cpp from an F16 file it deletes afterwards
(the F16 file of an 8B model takes 16 GB of disk while it exists).

The shapes and hyperparameters are those of the released models:

    llama3   Llama 3 8B      32 layers, embedding 4096, 32/8 heads, FFN 14336,
                             vocabulary 128256, rope theta 500000
    mistral  Mistral 7B v0.1 32 layers, embedding 4096, 32/8 heads, FFN 14336,
                             vocabulary 32000
    qwen2    Qwen2.5 7B      28 layers, embedding 3584, 28/4 heads, FFN 18944,
                             vocabulary 152064, rope theta 1000000, biases on
                             the query, key and value projections

so file sizes and the speed of a forward pass are those of the real models.
The weights are random normal numbers; the machines the benchmarks were run
on cannot download from Hugging Face. The tokenizers are the real ones,
read from the Python packages that ship them: Llama 3's tiktoken ranks
from Meta's llama-models, Mistral's SentencePiece model from
mistral-common, and Qwen's tiktoken ranks from dashscope. The BPE merges
are recovered from the ranks the way Hugging Face's converter does.
"""

import base64
import ctypes
import glob
import os
import sys
import zipfile

import numpy as np
import sentencepiece
from gguf import GGMLQuantizationType, GGUFWriter

import llama_cpp

MODELS = {
    "llama3": dict(arch="llama", layers=32, embed=4096, heads=32, kv=8, ffn=14336, vocab=128256,
                   theta=500000.0, eps=1e-5, ctx=8192, bias=False),
    "mistral": dict(arch="llama", layers=32, embed=4096, heads=32, kv=8, ffn=14336, vocab=32000,
                    theta=10000.0, eps=1e-5, ctx=32768, bias=False),
    "qwen2": dict(arch="qwen2", layers=28, embed=3584, heads=28, kv=4, ffn=18944, vocab=152064,
                  theta=1000000.0, eps=1e-6, ctx=32768, bias=True),
}

rng = np.random.default_rng(2026)


def wheel_file(wheels, package, member):
    paths = glob.glob(os.path.join(wheels, package + "-*.whl"))
    if not paths:
        raise SystemExit(f"no {package} wheel in {wheels}")
    with zipfile.ZipFile(sorted(paths)[-1]) as z:
        return z.read(member)


def byte_chars():
    """GPT-2's map from bytes to the printable characters of the vocabulary."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tiktoken_vocab(data):
    """Tokens and merges, in GGUF's byte-mapped form, from tiktoken ranks."""
    ranks = {}
    for line in data.decode().splitlines():
        if line:
            tok, r = line.split()
            ranks[base64.b64decode(tok)] = int(r)
    bc = byte_chars()
    mapped = lambda b: "".join(bc[x] for x in b)

    def bpe(token, max_rank):
        parts = [bytes([b]) for b in token]
        while True:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best_rank is None or best_rank >= max_rank:
                return parts
            parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2:]

    tokens = [None] * len(ranks)
    merges = []
    for tok, r in ranks.items():
        tokens[r] = mapped(tok)
        if len(tok) > 1:
            parts = bpe(tok, r)
            if len(parts) == 2:
                merges.append((r, mapped(parts[0]) + " " + mapped(parts[1])))
    merges.sort()
    return tokens, [m for _, m in merges]


def add_tokenizer(w, name, cfg, wheels):
    V = cfg["vocab"]
    if name == "mistral":
        sp = sentencepiece.SentencePieceProcessor()
        sp.LoadFromSerializedProto(wheel_file(wheels, "mistral_common", "mistral_common/data/tokenizer.model.v1"))
        tokens, scores, types = [], [], []
        for i in range(sp.GetPieceSize()):
            tokens.append(sp.IdToPiece(i))
            scores.append(sp.GetScore(i))
            types.append(2 if sp.IsUnknown(i) else 3 if sp.IsControl(i) else 6 if sp.IsByte(i)
                         else 5 if sp.IsUnused(i) else 1)
        assert len(tokens) == V
        w.add_tokenizer_model("llama")
        w.add_tokenizer_pre("default")
        w.add_token_list(tokens)
        w.add_token_scores(scores)
        w.add_token_types(types)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_unk_token_id(0)
        w.add_add_bos_token(True)
        w.add_add_space_prefix(True)
        return
    if name == "llama3":
        tokens, merges = tiktoken_vocab(wheel_file(wheels, "llama_models", "llama_models/llama3/tokenizer.model"))
        named = {0: "<|begin_of_text|>", 1: "<|end_of_text|>", 6: "<|start_header_id|>",
                 7: "<|end_header_id|>", 9: "<|eot_id|>"}
        n = len(tokens)
        specials = [named.get(i, f"<|reserved_special_token_{i}|>") for i in range(V - n)]
        bos, eos, pre, add_bos = n, n + 1, "llama-bpe", True
    else:
        tokens, merges = tiktoken_vocab(wheel_file(wheels, "dashscope", "dashscope/resources/qwen.tiktoken"))
        n = len(tokens)
        named = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        specials = [named[i] if i < len(named) else f"[PAD{n + i}]" for i in range(V - n)]
        bos, eos, pre, add_bos = n, n, "qwen2", False
    types = [1] * n + [3] * len(specials)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens + specials)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(bos)
    w.add_eos_token_id(eos)
    w.add_add_bos_token(add_bos)


def shapes(cfg):
    """GGUF tensor names and shapes (rows, columns), in llama.cpp's order."""
    E, D = cfg["embed"], cfg["embed"] // cfg["heads"]
    KV, F, V = cfg["kv"] * D, cfg["ffn"], cfg["vocab"]
    out = [("token_embd.weight", (V, E))]
    for l in range(cfg["layers"]):
        p = f"blk.{l}."
        out += [(p + "attn_norm.weight", (E,)), (p + "attn_q.weight", (E, E))]
        if cfg["bias"]:
            out += [(p + "attn_q.bias", (E,))]
        out += [(p + "attn_k.weight", (KV, E))]
        if cfg["bias"]:
            out += [(p + "attn_k.bias", (KV,))]
        out += [(p + "attn_v.weight", (KV, E))]
        if cfg["bias"]:
            out += [(p + "attn_v.bias", (KV,))]
        out += [
            (p + "attn_output.weight", (E, E)),
            (p + "ffn_norm.weight", (E,)),
            (p + "ffn_gate.weight", (F, E)),
            (p + "ffn_up.weight", (F, E)),
            (p + "ffn_down.weight", (E, F)),
        ]
    out += [("output_norm.weight", (E,)), ("output.weight", (V, E))]
    return out


def weight(name, shape):
    if name.endswith(".bias"):
        return (0.1 * rng.standard_normal(shape, dtype=np.float32)).astype(np.float32)
    if len(shape) == 1:
        return (1 + 0.1 * rng.standard_normal(shape, dtype=np.float32)).astype(np.float32)
    a = rng.standard_normal(shape, dtype=np.float32)
    a *= np.float32(1 / np.sqrt(shape[1]))
    return a.astype(np.float16)


def generate(name, wheels, out):
    cfg = MODELS[name]
    f16_path = os.path.join(out, f"{name}-rand-f16.gguf")
    q4_path = os.path.join(out, f"{name}-rand-q4km.gguf")
    arch = cfg["arch"]
    w = GGUFWriter(f16_path, arch)
    w.add_name(f"{name}-random")
    w.add_context_length(cfg["ctx"])
    w.add_embedding_length(cfg["embed"])
    w.add_block_count(cfg["layers"])
    w.add_feed_forward_length(cfg["ffn"])
    w.add_rope_dimension_count(cfg["embed"] // cfg["heads"])
    w.add_head_count(cfg["heads"])
    w.add_head_count_kv(cfg["kv"])
    w.add_layer_norm_rms_eps(cfg["eps"])
    w.add_rope_freq_base(cfg["theta"])
    w.add_file_type(1)  # F16
    add_tokenizer(w, name, cfg, wheels)

    # The weights are written one tensor at a time: an 8B model in F16 does
    # not fit in memory.
    sh = shapes(cfg)
    for tname, shape in sh:
        dt = np.float32 if len(shape) == 1 else np.float16
        nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
        w.add_tensor_info(tname, shape, np.dtype(dt), nbytes,
                          GGMLQuantizationType.F32 if dt == np.float32 else GGMLQuantizationType.F16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for tname, shape in sh:
        w.write_tensor_data(weight(tname, shape))
    w.close()

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    params.nthread = os.cpu_count() or 1
    if llama_cpp.llama_model_quantize(f16_path.encode(), q4_path.encode(), ctypes.byref(params)) != 0:
        raise SystemExit("llama.cpp failed to quantize the model")
    os.remove(f16_path)
    print(q4_path, os.path.getsize(q4_path), "bytes")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    wheels, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[3:] or list(MODELS):
        if name not in MODELS:
            raise SystemExit(f"unknown model {name!r}; want one of {', '.join(MODELS)}")
        generate(name, wheels, out)


if __name__ == "__main__":
    main()
