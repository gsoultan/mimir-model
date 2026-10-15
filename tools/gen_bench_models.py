#!/usr/bin/env python3
"""Generate TinyLlama-shaped models with random weights for docs/benchmarks.md.

Run from the repository root with

    python3 -P tools/gen_bench_models.py OUTDIR

Needs NumPy, the `gguf` package and llama-cpp-python. It writes

    OUTDIR/tinyllama-rand-f16.gguf          F16, as convert_hf_to_gguf.py writes
    OUTDIR/tinyllama-rand-q4km.gguf         quantized to Q4_K_M by llama.cpp
    OUTDIR/tinyllama-rand-bf16.safetensors  BF16 with Hugging Face tensor names

The shapes and hyperparameters are TinyLlama 1.1B's (22 layers, embedding
2048, 32 query heads over 4 key-value heads, FFN 5632, vocabulary 32000,
context 2048), so file sizes and load times are those of the real model. The
weights are random normal numbers and the vocabulary is made-up pieces: the
benchmark measures loading and the speed of the forward pass, which do not
depend on the values. The real files are not used because the machines the
benchmarks were run on cannot download from Hugging Face.
"""

import ctypes
import json
import os
import sys

import numpy as np
from gguf import GGUFWriter

import llama_cpp

E, LAYERS, HEADS, KV_HEADS, FFN, V, CTX = 2048, 22, 32, 4, 5632, 32000, 2048
D = E // HEADS
THETA, EPS = 10000.0, 1e-5
rng = np.random.default_rng(2024)


def vocabulary():
    tokens = ["<unk>", "<s>", "</s>"] + ["<0x%02X>" % b for b in range(256)]
    types = [2, 3, 3] + [6] * 256
    letters = "etaoinshrdlucmfwypvbgkjqxz"
    seen = set(tokens)
    n = 0
    while len(tokens) < V:
        # Pieces of one to four letters, with and without the word marker,
        # in a fixed order so the file is reproducible.
        s, k = "", n
        while True:
            s += letters[k % 26]
            k //= 26
            if k == 0:
                break
        n += 1
        for p in ("▁" + s, s):
            if p not in seen and len(tokens) < V:
                seen.add(p)
                tokens.append(p)
                types.append(1)
    scores = [0.0] * 259 + [-float(i) for i in range(V - 259)]
    return tokens, scores, types


def shapes():
    """GGUF tensor names and shapes (rows, columns), in llama.cpp's order."""
    out = [("token_embd.weight", (V, E))]
    for l in range(LAYERS):
        p = f"blk.{l}."
        out += [
            (p + "attn_norm.weight", (E,)),
            (p + "attn_q.weight", (E, E)),
            (p + "attn_k.weight", (KV_HEADS * D, E)),
            (p + "attn_v.weight", (KV_HEADS * D, E)),
            (p + "attn_output.weight", (E, E)),
            (p + "ffn_norm.weight", (E,)),
            (p + "ffn_gate.weight", (FFN, E)),
            (p + "ffn_up.weight", (FFN, E)),
            (p + "ffn_down.weight", (E, FFN)),
        ]
    out += [("output_norm.weight", (E,)), ("output.weight", (V, E))]
    return out


HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def hf_name(name):
    if name in HF:
        return HF[name]
    _, l, rest = name.split(".", 2)
    return f"model.layers.{l}.{HF[rest]}"


def weight(shape):
    if len(shape) == 1:
        return (1 + 0.1 * rng.standard_normal(shape, dtype=np.float32)).astype(np.float32)
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(1 / np.sqrt(shape[1]))


def bf16(a):
    b = a.view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype("<u2")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "bench-models"
    os.makedirs(out, exist_ok=True)
    f16_path = os.path.join(out, "tinyllama-rand-f16.gguf")
    q4_path = os.path.join(out, "tinyllama-rand-q4km.gguf")
    st_path = os.path.join(out, "tinyllama-rand-bf16.safetensors")

    tokens, scores, types = vocabulary()
    w = GGUFWriter(f16_path, "llama")
    w.add_name("tinyllama-random")
    w.add_context_length(CTX)
    w.add_embedding_length(E)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(FFN)
    w.add_rope_dimension_count(D)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(EPS)
    w.add_rope_freq_base(THETA)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)
    w.add_add_space_prefix(True)

    header, offset, blobs = {}, 0, []
    for name, shape in shapes():
        a = weight(shape)
        # Norm weights stay F32, as convert_hf_to_gguf.py writes them.
        w.add_tensor(name, a if len(shape) == 1 else a.astype(np.float16))
        b = bf16(a).tobytes()
        header[hf_name(name)] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(st_path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    params.nthread = os.cpu_count() or 1
    if llama_cpp.llama_model_quantize(f16_path.encode(), q4_path.encode(), ctypes.byref(params)) != 0:
        raise SystemExit("llama.cpp failed to quantize the model")
    for p in (f16_path, q4_path, st_path):
        print(p, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
