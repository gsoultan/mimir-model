#!/usr/bin/env python3
"""Generate the tiny Qwen 2 fixture in testdata/qwen2.

Run from the repository root with `python3 -P tools/gen_qwen2_testdata.py`.
Needs NumPy, the `gguf` package and llama-cpp-python, used as in
tools/gen_llama_testdata.py: llama.cpp quantizes the model to Q4_K_M, gives
the expected token ids for a set of strings, and evaluates the prompt to
cross-check the NumPy oracle below.

The model has the qwen2 architecture with random weights: two layers,
embedding 256, four query heads over two key-value heads, biases on the
query, key and value projections, rope over the two halves of each head
(llama.cpp's NEOX mode) with theta 1000000, and a byte-level BPE
vocabulary of the GPT-2 kind (tokenizer.ggml.model "gpt2", pre-tokenizer
"qwen2") with 160 merges trained on a few sentences. expected.json holds
the oracle's float64 logits for a prompt and a greedy continuation.
"""

import base64
import ctypes
import json
import os
import tempfile
from collections import Counter

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize

import llama_cpp

OUT = "testdata/qwen2"
MODEL = os.path.join(OUT, "tiny-q4km.gguf")
rng = np.random.default_rng(12)

E, LAYERS, HEADS, KV_HEADS, FFN, CTX = 256, 2, 4, 2, 256, 64
D = E // HEADS
THETA, EPS = 1000000.0, 1e-6

# ---- vocabulary: a small byte-level BPE, as GPT-2 and Qwen 2 use
CORPUS = (
    "Once upon a time there was a little girl who lived in a village near the forest. "
    "Whenever she went out, the little girl wore a red riding cloak, so everyone in the "
    "village called her Little Red Riding Hood. One morning she said: I will go to see "
    "my grandmother. The sun was shining and the birds were singing in the trees."
)


def byte_chars():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BC = byte_chars()


def train_bpe(text, n_merges):
    words = Counter("".join(BC[b] for b in (" " + w).encode()) for w in text.split(" "))
    seqs = {w: list(w) for w in words}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[a, b] += n
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append((a, b))
        for w in seqs:
            s, out, i = seqs[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    return merges


merges = train_bpe(CORPUS, 160)
tokens = [BC[b] for b in range(256)]
for a, b in merges:
    if a + b not in tokens:
        tokens.append(a + b)
types = [1] * len(tokens)
tokens.append("<|endoftext|>")
types.append(3)
EOT = len(tokens) - 1
V = len(tokens)

# ---- weights
def lin(n, k):
    return (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)


weights = {
    "token_embd.weight": rng.standard_normal((V, E)).astype(np.float32),
    "output_norm.weight": (1 + 0.1 * rng.standard_normal(E)).astype(np.float32),
    "output.weight": lin(V, E) * 4,
}
for l in range(LAYERS):
    p = f"blk.{l}."
    weights.update({
        p + "attn_norm.weight": (1 + 0.1 * rng.standard_normal(E)).astype(np.float32),
        p + "attn_q.weight": lin(E, E) * 2,
        p + "attn_k.weight": lin(KV_HEADS * D, E) * 2,
        p + "attn_v.weight": lin(KV_HEADS * D, E),
        p + "attn_q.bias": (0.5 * rng.standard_normal(E)).astype(np.float32),
        p + "attn_k.bias": (0.5 * rng.standard_normal(KV_HEADS * D)).astype(np.float32),
        p + "attn_v.bias": (0.5 * rng.standard_normal(KV_HEADS * D)).astype(np.float32),
        p + "attn_output.weight": lin(E, E),
        p + "ffn_norm.weight": (1 + 0.1 * rng.standard_normal(E)).astype(np.float32),
        p + "ffn_gate.weight": lin(FFN, E),
        p + "ffn_up.weight": lin(FFN, E),
        p + "ffn_down.weight": lin(E, FFN),
    })

os.makedirs(OUT, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    f32_path = os.path.join(tmp, "tiny-f32.gguf")
    w = GGUFWriter(f32_path, "qwen2")
    w.add_name("tiny-qwen2")
    w.add_context_length(CTX)
    w.add_embedding_length(E)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(FFN)
    w.add_rope_dimension_count(D)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(EPS)
    w.add_rope_freq_base(THETA)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges([a + " " + b for a, b in merges])
    w.add_bos_token_id(EOT)
    w.add_eos_token_id(EOT)
    w.add_add_bos_token(False)
    for name, t in weights.items():
        w.add_tensor(name, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    params.nthread = 1
    if llama_cpp.llama_model_quantize(f32_path.encode(), MODEL.encode(), ctypes.byref(params)) != 0:
        raise SystemExit("llama.cpp failed to quantize the model")

# ---- the oracle: the forward pass in float64, from the quantized file
r = GGUFReader(MODEL)
W = {}
dtypes = {}
for t in r.tensors:
    a = dequantize(np.asarray(t.data), t.tensor_type)
    W[t.name] = a.astype(np.float64).reshape([int(x) for x in reversed(t.shape)])
    dtypes[t.name] = t.tensor_type.name


def rms(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * g


def rope(x, pos):  # x [T, H, D]; llama.cpp NEOX mode rotates the two halves
    inv = THETA ** (-np.arange(0, D, 2) / D)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def forward(ids):
    n = len(ids)
    pos = np.arange(n, dtype=np.float64)
    x = W["token_embd.weight"][ids]
    for l in range(LAYERS):
        p = f"blk.{l}."
        h = rms(x, W[p + "attn_norm.weight"])
        q = rope((h @ W[p + "attn_q.weight"].T + W[p + "attn_q.bias"]).reshape(n, HEADS, D), pos)
        k = rope((h @ W[p + "attn_k.weight"].T + W[p + "attn_k.bias"]).reshape(n, KV_HEADS, D), pos)
        v = (h @ W[p + "attn_v.weight"].T + W[p + "attn_v.bias"]).reshape(n, KV_HEADS, D)
        o = np.empty((n, HEADS, D))
        for hd in range(HEADS):
            kh = hd // (HEADS // KV_HEADS)
            s = q[:, hd] @ k[:, kh].T / np.sqrt(D)
            s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
            e = np.exp(s - s.max(axis=1, keepdims=True))
            o[:, hd] = (e / e.sum(axis=1, keepdims=True)) @ v[:, kh]
        x = x + o.reshape(n, E) @ W[p + "attn_output.weight"].T
        h = rms(x, W[p + "ffn_norm.weight"])
        g = h @ W[p + "ffn_gate.weight"].T
        u = h @ W[p + "ffn_up.weight"].T
        x = x + (g / (1 + np.exp(-g)) * u) @ W[p + "ffn_down.weight"].T
    return rms(x, W["output_norm.weight"]) @ W["output.weight"].T


# ---- llama.cpp: tokenizer cases and a cross-check of the oracle
llm = llama_cpp.Llama(model_path=MODEL, n_ctx=CTX, logits_all=True, verbose=False, seed=0)
TOKENIZE = [
    "",
    "Once upon a time",
    "the little girl wore a red riding cloak",
    "  two  spaces, then\nnew line",
    "Grandmother's café 日本",
    "it's 2024: 12345 birds!\n\n  ok",
    "zebra quixotic",
]
tok_cases = [{"text": s, "ids": llm.tokenize(s.encode(), add_bos=True, special=False)} for s in TOKENIZE]

PROMPT = "Once upon a time there was"
prompt_ids = llm.tokenize(PROMPT.encode(), add_bos=True, special=False)
prompt_logits = forward(prompt_ids)
ids = list(prompt_ids)
steps = []
for _ in range(8):
    last = forward(ids)[-1]
    nxt = int(np.argmax(last))
    steps.append({"next": nxt, "logits": last})
    ids.append(nxt)

llm.reset()
llm.eval(ids[:-1])
cpp = np.array(llm.scores[: len(ids) - 1])
ours = np.concatenate([prompt_logits, np.array([s["logits"] for s in steps[1:]])])
cpp_diff = float(np.abs(cpp - ours).max())
scale = float(np.abs(ours).max())
print(f"llama.cpp vs oracle: max |logit diff| {cpp_diff:.4f} (logits up to {scale:.2f})")
# llama.cpp rounds activations to 8-bit blocks before each quantized matrix
# product, so its logits differ by a few tenths and a greedy pick may differ
# where the oracle's top two logits are that close.
for i, st in enumerate(steps):
    row = ours[len(prompt_ids) - 1 + i]
    pick = int(np.argmax(cpp[len(prompt_ids) - 1 + i]))
    margin = row[st["next"]] - row[pick]
    print(f"step {i}: oracle {st['next']}, llama.cpp {pick}, oracle margin {margin:.3f}")
    if pick != st["next"] and margin > 2 * cpp_diff:
        raise SystemExit("llama.cpp disagrees with the oracle")
if cpp_diff > 0.05 * scale:
    raise SystemExit("llama.cpp disagrees with the oracle")


def b64(a):
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


expected = {
    "model": os.path.basename(MODEL),
    "vocab": V,
    "dtypes": dtypes,
    "tokenize": tok_cases,
    "prompt": PROMPT,
    "prompt_ids": prompt_ids,
    "prompt_logits": b64(prompt_logits),
    "greedy": [s["next"] for s in steps],
    "greedy_logits": [b64(s["logits"]) for s in steps],
    "llama_cpp_max_diff": round(cpp_diff, 6),
}
with open(os.path.join(OUT, "expected.json"), "w") as f:
    json.dump(expected, f, indent=1)
    f.write("\n")
print("wrote", MODEL, os.path.getsize(MODEL), "bytes, vocabulary", V)
