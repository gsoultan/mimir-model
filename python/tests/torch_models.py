"""Small PyTorch models for the exporter tests and for
tools/gen_pytorch_testdata.py. Each covers one family: a convolutional
classifier, a 1-D convolution, a GPT-style decoder, a Llama-style decoder
and a set of elementwise and integer operations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=2, dilation=2, groups=2)
        self.conv3 = nn.Conv2d(8, 8, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 10)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)  # 16 -> 8
        y = F.hardswish(self.conv2(x))
        z = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = torch.cat([torch.sigmoid(y) * z, torch.tanh(y - z)], dim=1)[:, ::2]
        x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)  # 8 -> 4
        x = F.adaptive_avg_pool2d(x, 2).flatten(1)
        x = self.drop(F.relu6(self.fc1(x)))
        return torch.softmax(self.fc2(x) / 2.0, dim=-1)

    @staticmethod
    def example():
        return (torch.randn(2, 3, 16, 16),), {"x": {0: Dim("batch", max=16)}}


class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 6, 5, padding="same")
        self.conv2 = nn.Conv1d(6, 6, 3, stride=2, groups=3)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        x = F.gelu(self.conv(x))
        x = F.avg_pool2d(self.conv2(x).unsqueeze(2), (1, 2)).squeeze(2)
        return self.head(x.mean(dim=-1)), x.amax(dim=1)

    @staticmethod
    def example():
        return (torch.randn(2, 4, 40),), {"x": {0: Dim("batch", max=8)}}


class GPTBlock(nn.Module):
    def __init__(self, d, heads, max_len, manual):
        super().__init__()
        self.heads, self.manual = heads, manual
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        if self.manual:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            y = F.softmax(att, dim=-1) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class TinyGPT(nn.Module):
    def __init__(self, vocab=64, d=32, heads=4, layers=2, max_len=64):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(GPTBlock(d, heads, max_len, manual=i % 2 == 1) for i in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)

    def forward(self, idx):
        T = idx.size(1)
        pos = torch.arange(0, T, dtype=torch.long)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        return F.linear(self.ln_f(x), self.wte.weight)  # tied head

    @staticmethod
    def example():
        return (torch.randint(0, 64, (2, 7)),), {"idx": {0: Dim("batch", max=8), 1: Dim("seq", max=32)}}


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class LlamaBlock(nn.Module):
    def __init__(self, d, heads, kv_heads, hidden):
        super().__init__()
        self.heads, self.kv_heads, self.hd = heads, kv_heads, d // heads
        self.norm1, self.norm2 = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate, self.up = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, -1))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class TinyLlama(nn.Module):
    def __init__(self, vocab=64, d=32, heads=4, kv_heads=2, hidden=48, layers=2, theta=10000.0):
        super().__init__()
        hd = d // heads
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(LlamaBlock(d, heads, kv_heads, hidden) for _ in range(layers))
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd), persistent=False)

    def forward(self, ids):
        T = ids.shape[1]
        pos = torch.arange(T, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        x = self.embed(ids)
        cos, sin = emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))

    @staticmethod
    def example():
        return (torch.randint(0, 64, (1, 9)),), {"ids": {1: Dim("seq", max=64)}}


class LlamaCacheBlock(LlamaBlock):
    """A LlamaBlock that keeps its keys and values in buffers of max_len
    positions and attends over the whole cache with a mask, as a static
    cache does. Keys are written with index_copy_ and values with index
    assignment, so both lowerings are exercised."""

    def __init__(self, d, heads, kv_heads, hidden, max_len):
        super().__init__(d, heads, kv_heads, hidden)
        shape = (1, kv_heads, max_len, self.hd)
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def forward(self, x, cos, sin, pos, mask):
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        self.k_cache.index_copy_(2, pos, k)
        self.v_cache[:, :, pos] = v
        y = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, attn_mask=mask, enable_gqa=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, -1))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class TinyLlamaKV(TinyLlama):
    """TinyLlama with a key-value cache: forward(ids, pos) runs the tokens
    ids at the positions pos and leaves their keys and values in the cache,
    so a prompt followed by one token per step gives the logits of the full
    forward pass. Its parameters have TinyLlama's names."""

    MAX_LEN = 32

    def __init__(self, vocab=64, d=32, heads=4, kv_heads=2, hidden=48, layers=2, theta=10000.0):
        super().__init__(vocab, d, heads, kv_heads, hidden, layers, theta)
        self.layers = nn.ModuleList(LlamaCacheBlock(d, heads, kv_heads, hidden, self.MAX_LEN) for _ in range(layers))
        self.register_buffer("kpos", torch.arange(self.MAX_LEN), persistent=False)

    def forward(self, ids, pos):
        freqs = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        x = self.embed(ids)
        cos, sin = emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)
        mask = (self.kpos[None, :] <= pos[:, None])[None, None]  # [1, 1, T, MAX_LEN]
        for layer in self.layers:
            x = layer(x, cos, sin, pos, mask)
        return self.lm_head(self.norm(x))

    def reset(self):
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.v_cache.zero_()

    @staticmethod
    def example():
        seq = Dim("seq", max=TinyLlamaKV.MAX_LEN)
        return (torch.randint(0, 64, (1, 5)), torch.arange(5)), {"ids": {1: seq}, "pos": {0: seq}}


class Ops(nn.Module):
    """Elementwise, comparison, integer and reduction operations."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.linspace(-2, 2, 12).view(3, 4))

    def forward(self, x, n):
        a = torch.where(x > 0.5, x, -x) + torch.clamp(x, -1, 1) * 2 - x.abs().sqrt()
        b = torch.maximum(x, x * 0.5) - torch.minimum(x, torch.zeros_like(x))
        c = (x ** 2 + 1).rsqrt() + torch.exp(-x.abs()) + torch.cos(x) + torch.erf(x)
        m = (x >= 0) & ~(x == 1) | (x < -0.5)
        d = torch.var(x, dim=-1, keepdim=True) + x.std(dim=0).sum() + x.mean() + x.amin(dim=-1, keepdim=True)
        q = torch.div(n, 3, rounding_mode="floor") + torch.div(n, -2, rounding_mode="trunc")
        r = torch.cumsum(n, dim=1) * 2 - (n != 2).sum(dim=1, keepdim=True)
        t = self.table * 3 + 1  # folded
        e = torch.log_softmax(x, dim=0) + F.silu(x) * torch.sigmoid(x) + x.float().pow(3)
        s = torch.stack([x[:, 0], x[:, -1]], dim=1)
        return a + b + c + d + e, m, q + r, t, torch.select(x, 1, 2).unsqueeze(-1) * s, x[1:, 1:3].T

    @staticmethod
    def example():
        x = torch.randn(3, 4)
        n = torch.randint(-20, 20, (3, 4))
        return (x, n), None


MODELS = {
    "tiny-cnn": TinyCNN,
    "audio": AudioNet,
    "tiny-gpt": TinyGPT,
    "tiny-llama": TinyLlama,
}


def build(name, seed=0):
    """The model with deterministic random weights, in eval mode. Batch norm
    statistics are randomised so the folding is exercised."""
    torch.manual_seed(seed)
    m = MODELS[name]()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.2)
                mod.running_var.uniform_(0.5, 1.5)
                mod.weight.normal_(1, 0.2)
                mod.bias.normal_(0, 0.1)
    return m.eval()


def build_kv(seed=0):
    """TinyLlamaKV with the weights of build("tiny-llama", seed) and an
    empty cache."""
    m = TinyLlamaKV()
    m.load_state_dict(build("tiny-llama", seed).state_dict())
    return m.eval()
