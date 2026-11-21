#!/usr/bin/env python3
"""Generate the tiny Gemma 2, Gemma 3, Qwen 3 and Phi 3 fixtures in testdata.

Run from the repository root with

    python3 -P tools/gen_family_testdata.py gemma2 gemma3 qwen3 phi3

Needs NumPy, the `gguf` package and llama-cpp-python, used as in
tools/gen_llama_testdata.py: llama.cpp quantizes each model to Q4_K_M,
gives the expected token ids for a set of strings, and evaluates the prompt
to cross-check the NumPy oracle below, so the oracle's reading of each
family's conventions is llama.cpp's.

Each model has random weights and two to six layers of embedding 256 and
what sets its family apart:

- gemma2: embeddings scaled by sqrt(256), heads of 128 features (not
  256/4), GELU-gated feed-forward, norms after attention and after the
  feed-forward layer, a sliding window of 8 positions on even layers,
  attention scores soft-capped at 5 and logits at 10, tied embeddings, and
  a SentencePiece vocabulary without the leading space (as Gemma's).
- gemma3: as gemma2 without the soft caps, with RMSNorm on each query and
  key head, six layers of which the sixth is global, local layers rotated
  with base 10000 and global ones with base 1000000 scaled linearly by 8,
  and one key-value head.
- qwen3: Qwen 2's layout without biases, with RMSNorm on each query and key
  head, heads of 96 features, tied embeddings and Qwen 2's byte-level BPE.
  Its attention output, 384 features wide, is no multiple of Q4_K's blocks
  of 256, so llama.cpp falls back to Q5_0 for it, as it does for Gemma 3
  1B's 1152-wide layers.
- phi3: fused query, key and value projections and fused gate and up
  projections, rope over the first 48 of each head's 64 features with
  LongRoPE factors and an attention factor, a context longer than the
  original one so the long factors apply, and a byte-level BPE vocabulary
  with the "dbrx" pre-tokenizer of Phi 4.

expected.json holds, for each, the oracle's float64 logits for a prompt
long enough to cross the sliding window, and a greedy continuation.
"""

import base64
import ctypes
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np
from gguf import GGUFReader, GGUFWriter, RopeScalingType
from gguf.quants import dequantize

import llama_cpp

CORPUS = (
    "Once upon a time there was a little girl who lived in a village near the forest. "
    "Whenever she went out, the little girl wore a red riding cloak, so everyone in the "
    "village called her Little Red Riding Hood. One morning she said: I will go to see "
    "my grandmother. The sun was shining and the birds were singing in the trees."
)
E, FFN, CTX = 256, 256, 64

FAMILIES = {
    "gemma2": dict(layers=2, heads=4, kv=2, d=128, theta=10000.0, eps=1e-6, vocab="spm", tied=True,
                   window=8, every=2, attn_cap=5.0, final_cap=10.0),
    "gemma3": dict(layers=6, heads=4, kv=1, d=128, theta=1000000.0, eps=1e-6, vocab="spm", tied=True,
                   window=8, every=6, local_theta=10000.0, scale=8.0, head_norms=True),
    "qwen3": dict(layers=2, heads=4, kv=2, d=96, theta=1000000.0, eps=1e-6, vocab="bpe", pre="qwen2",
                  tied=True, head_norms=True),
    "phi3": dict(layers=2, heads=4, kv=2, d=64, rot=48, theta=10000.0, eps=1e-5, vocab="bpe", pre="dbrx",
                 tied=False, orig_ctx=32, attn_factor=1.19),
}


# ---- vocabularies
def spm_vocab():
    sp = "▁"
    words = Counter(sp + w for w in CORPUS.split(" "))
    seqs = {w: list(w) for w in words}
    chars = sorted({c for w in words for c in w})
    pieces = []
    for _ in range(160):
        pairs = Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[a, b] += n
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        pieces.append(a + b)
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
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    types = [3, 3, 3, 2, 3, 3]
    scores = [0.0] * len(tokens)
    tokens += ["<0x%02X>" % b for b in range(256)]
    types += [6] * 256
    scores += [0.0] * 256
    for rank, p in enumerate(pieces):
        tokens.append(p)
        types.append(1)
        scores.append(-float(rank))
    for c in chars + [sp]:
        if c not in tokens:
            tokens.append(c)
            types.append(1)
            scores.append(-1000.0)
    return dict(tokens=tokens, types=types, scores=scores)


def byte_chars():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def bpe_vocab():
    bc = byte_chars()
    words = Counter("".join(bc[b] for b in (" " + w).encode()) for w in CORPUS.split(" "))
    seqs = {w: list(w) for w in words}
    merges = []
    for _ in range(160):
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
    tokens = [bc[b] for b in range(256)]
    for a, b in merges:
        if a + b not in tokens:
            tokens.append(a + b)
    types = [1] * len(tokens)
    tokens.append("<|endoftext|>")
    types.append(3)
    return dict(tokens=tokens, types=types, merges=[a + " " + b for a, b in merges])


def generate(arch):
    f = FAMILIES[arch]
    out = os.path.join("testdata", arch)
    model = os.path.join(out, "tiny-q4km.gguf")
    rng = np.random.default_rng(sum(map(ord, arch)))
    L, H, KV, D = f["layers"], f["heads"], f["kv"], f["d"]
    R = f.get("rot", D)
    EPS, THETA = f["eps"], f["theta"]
    gemma = arch.startswith("gemma")
    voc = spm_vocab() if f["vocab"] == "spm" else bpe_vocab()
    V = len(voc["tokens"])

    def lin(n, k):
        return (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)

    def scale(n):
        return (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    weights = {
        # Tied embeddings are also the output layer: a quarter keeps the
        # logits of the size an untied output layer gives them.
        "token_embd.weight": (rng.standard_normal((V, E)) * (0.25 if f["tied"] else 1)).astype(np.float32),
        "output_norm.weight": scale(E),
    }
    if not f["tied"]:
        weights["output.weight"] = lin(V, E) * 4
    if arch == "phi3":
        h = R // 2
        weights["rope_factors_short.weight"] = (1 + rng.random(h)).astype(np.float32)
        weights["rope_factors_long.weight"] = (1 + 3 * rng.random(h)).astype(np.float32)
    for l in range(L):
        p = f"blk.{l}."
        q, k, v = lin(H * D, E) * 2, lin(KV * D, E) * 2, lin(KV * D, E)
        if arch == "phi3":
            weights[p + "attn_qkv.weight"] = np.concatenate([q, k, v])
        else:
            weights.update({p + "attn_q.weight": q, p + "attn_k.weight": k, p + "attn_v.weight": v})
        weights[p + "attn_norm.weight"] = scale(E)
        weights[p + "attn_output.weight"] = lin(E, H * D)
        weights[p + "ffn_norm.weight"] = scale(E)
        if arch == "phi3":
            weights[p + "ffn_up.weight"] = np.concatenate([lin(FFN, E), lin(FFN, E)])
        else:
            weights[p + "ffn_gate.weight"] = lin(FFN, E)
            weights[p + "ffn_up.weight"] = lin(FFN, E)
        weights[p + "ffn_down.weight"] = lin(E, FFN)
        if f.get("head_norms"):
            weights[p + "attn_q_norm.weight"] = scale(D)
            weights[p + "attn_k_norm.weight"] = scale(D)
        if gemma:
            weights[p + "post_attention_norm.weight"] = scale(E)
            weights[p + "post_ffw_norm.weight"] = scale(E)

    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        f32_path = os.path.join(tmp, "tiny-f32.gguf")
        w = GGUFWriter(f32_path, arch)
        # Not named after the family: llama.cpp gives the special tokens of
        # a model named like Phi 3 attributes by name, and this vocabulary
        # lacks some of them.
        w.add_name("tiny")
        w.add_context_length(CTX)
        w.add_embedding_length(E)
        w.add_block_count(L)
        w.add_feed_forward_length(FFN)
        w.add_head_count(H)
        w.add_head_count_kv(KV)
        w.add_key_length(D)
        w.add_value_length(D)
        w.add_rope_dimension_count(R)
        w.add_layer_norm_rms_eps(EPS)
        w.add_rope_freq_base(THETA)
        if "window" in f:
            w.add_sliding_window(f["window"])
        if "attn_cap" in f:
            w.add_attn_logit_softcapping(f["attn_cap"])
            w.add_final_logit_softcapping(f["final_cap"])
        if "scale" in f:
            w.add_rope_scaling_type(RopeScalingType.LINEAR)
            w.add_rope_scaling_factor(f["scale"])
        if arch == "phi3":
            w.add_rope_scaling_attn_factors(f["attn_factor"])
            w.add_rope_scaling_orig_ctx_len(f["orig_ctx"])
        w.add_token_list(voc["tokens"])
        w.add_token_types(voc["types"])
        if f["vocab"] == "spm":
            w.add_tokenizer_model("llama")
            w.add_token_scores(voc["scores"])
            w.add_bos_token_id(2)
            w.add_eos_token_id(1)
            w.add_unk_token_id(3)
            w.add_add_bos_token(True)
            w.add_add_space_prefix(False)
        else:
            w.add_tokenizer_model("gpt2")
            w.add_tokenizer_pre(f["pre"])
            w.add_token_merges(voc["merges"])
            eot = len(voc["tokens"]) - 1
            w.add_bos_token_id(eot)
            w.add_eos_token_id(eot)
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
        if llama_cpp.llama_model_quantize(f32_path.encode(), model.encode(), ctypes.byref(params)) != 0:
            raise SystemExit("llama.cpp failed to quantize the model")

    # ---- the oracle: the forward pass in float64, from the quantized file
    r = GGUFReader(model)
    W, dtypes = {}, {}
    for t in r.tensors:
        a = dequantize(np.asarray(t.data), t.tensor_type)
        W[t.name] = a.astype(np.float64).reshape([int(x) for x in reversed(t.shape)])
        dtypes[t.name] = t.tensor_type.name

    def rms(x, g):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * g

    def local(l):
        return "window" in f and l % f["every"] < f["every"] - 1

    def rope(x, pos, l):  # x [T, H, D]; NEOX: the two halves of the rotated features
        theta, inv, ms = THETA, None, 1.0
        if local(l) and "local_theta" in f:
            theta = f["local_theta"]
        inv = theta ** (-np.arange(0, R, 2) / R)
        if not local(l) and "scale" in f:
            inv = inv / f["scale"]
        if arch == "phi3":
            inv = inv / W["rope_factors_long.weight"]  # CTX > orig_ctx
            ms = f["attn_factor"]
        ang = pos[:, None] * inv[None, :]
        c, s = (np.cos(ang) * ms)[:, None, :], (np.sin(ang) * ms)[:, None, :]
        x1, x2, rest = x[..., : R // 2], x[..., R // 2 : R], x[..., R:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, rest], axis=-1)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    def forward(ids):
        n = len(ids)
        pos = np.arange(n, dtype=np.float64)
        x = W["token_embd.weight"][ids]
        if gemma:
            x = x * np.sqrt(E)
        for l in range(L):
            p = f"blk.{l}."
            hN = rms(x, W[p + "attn_norm.weight"])
            if arch == "phi3":
                qkv = hN @ W[p + "attn_qkv.weight"].T
                qf, kf, vf = qkv[:, : H * D], qkv[:, H * D : (H + KV) * D], qkv[:, (H + KV) * D :]
            else:
                qf, kf, vf = (hN @ W[p + "attn_" + c + ".weight"].T for c in "qkv")
            q, k, v = qf.reshape(n, H, D), kf.reshape(n, KV, D), vf.reshape(n, KV, D)
            if f.get("head_norms"):
                q, k = rms(q, W[p + "attn_q_norm.weight"]), rms(k, W[p + "attn_k_norm.weight"])
            q, k = rope(q, pos, l), rope(k, pos, l)
            visible = np.tril(np.ones((n, n), bool))
            if local(l):
                i, j = np.arange(n)[:, None], np.arange(n)[None, :]
                visible &= i - j < f["window"]
            o = np.empty((n, H, D))
            for hd in range(H):
                kh = hd // (H // KV)
                s = q[:, hd] @ k[:, kh].T / np.sqrt(D)
                if "attn_cap" in f:
                    s = np.tanh(s / f["attn_cap"]) * f["attn_cap"]
                s = np.where(visible, s, -np.inf)
                e = np.exp(s - s.max(axis=1, keepdims=True))
                o[:, hd] = (e / e.sum(axis=1, keepdims=True)) @ v[:, kh]
            a = o.reshape(n, H * D) @ W[p + "attn_output.weight"].T
            if gemma:
                a = rms(a, W[p + "post_attention_norm.weight"])
            x = x + a
            hN = rms(x, W[p + "ffn_norm.weight"])
            if arch == "phi3":
                gu = hN @ W[p + "ffn_up.weight"].T
                g, u = gu[:, :FFN], gu[:, FFN:]
            else:
                g, u = hN @ W[p + "ffn_gate.weight"].T, hN @ W[p + "ffn_up.weight"].T
            act = gelu(g) * u if gemma else g / (1 + np.exp(-g)) * u
            m = act @ W[p + "ffn_down.weight"].T
            if gemma:
                m = rms(m, W[p + "post_ffw_norm.weight"])
            x = x + m
        head = W["output.weight"] if "output.weight" in W else W["token_embd.weight"]
        logits = rms(x, W["output_norm.weight"]) @ head.T
        if "final_cap" in f:
            logits = np.tanh(logits / f["final_cap"]) * f["final_cap"]
        return logits

    # ---- llama.cpp: tokenizer cases and a cross-check of the oracle
    llm = llama_cpp.Llama(model_path=model, n_ctx=CTX, logits_all=True, verbose=False, seed=0)
    texts = [
        "",
        "Once upon a time",
        "the little girl wore a red riding cloak",
        "  two  spaces, then\nnew line",
        "Grandmother's café 日本",
        "it's 2024: 12345 birds!\n\n  ok",
        "zebra quixotic",
    ]
    tok_cases = [{"text": s, "ids": llm.tokenize(s.encode(), add_bos=True, special=False)} for s in texts]

    prompt = "Once upon a time there was a little girl who lived in a village"
    prompt_ids = llm.tokenize(prompt.encode(), add_bos=True, special=False)
    assert "window" not in f or len(prompt_ids) > f["window"] + 2, len(prompt_ids)
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
    big = float(np.abs(ours).max())
    print(f"{arch}: llama.cpp vs oracle: max |logit diff| {cpp_diff:.4f} (logits up to {big:.2f})")
    for i, st in enumerate(steps):
        row = ours[len(prompt_ids) - 1 + i]
        pick = int(np.argmax(cpp[len(prompt_ids) - 1 + i]))
        margin = row[st["next"]] - row[pick]
        print(f"  step {i}: oracle {st['next']}, llama.cpp {pick}, oracle margin {margin:.3f}")
        if pick != st["next"] and margin > 2 * cpp_diff:
            raise SystemExit(f"{arch}: llama.cpp disagrees with the oracle")
    if cpp_diff > 0.05 * big:
        raise SystemExit(f"{arch}: llama.cpp disagrees with the oracle")

    def b64(a):
        return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()

    expected = {
        "model": os.path.basename(model),
        "vocab": V,
        "dtypes": dtypes,
        "tokenize": tok_cases,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "prompt_logits": b64(prompt_logits),
        "greedy": [s["next"] for s in steps],
        "greedy_logits": [b64(s["logits"]) for s in steps],
        "llama_cpp_max_diff": round(cpp_diff, 6),
    }
    with open(os.path.join(out, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
        fh.write("\n")
    print("wrote", model, os.path.getsize(model), "bytes, vocabulary", V)


if __name__ == "__main__":
    for a in sys.argv[1:] or FAMILIES:
        generate(a)
