from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

NGRAM_PAD_OFFSET = 1
HASH_MULTIPLIER = 1000003
HASH_MASK = 0x7FFFFFFFFFFF


@dataclass(frozen=True)
class MicroLMConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 27
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    n_lanes: int = 4
    window: int = 256
    n_sink: int = 16
    rope_base: float = 10000.0
    engram_layers: tuple[int, ...] = (4, 14)
    engram_buckets: int = 1 << 15
    engram_hashes: int = 4
    engram_orders: tuple[int, ...] = (2, 3)
    mlp_block_size: int = 0
    sinkhorn_iters: int = 15
    mhc_alpha_init: float = 8.0
    gate_tau_init: float = 4.0
    gate_beta_init: float = -2.0
    norm_eps: float = 1e-6

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads


def fwht(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    d = shape[-1]
    y = x.reshape(-1, d)
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        y = y.reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).view(shape)


def zc_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class ZCRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return zc_normalize(x, self.eps) * self.weight


class Rope(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.to(torch.float32)[..., None] * self.inv_freq
        cos, sin = angles.cos(), angles.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class AttentionCache:
    def __init__(self, cfg: MicroLMConfig):
        self.n_sink = cfg.n_sink
        self.window = cfg.window
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        if self.k is None:
            self.k, self.v = k_t, v_t
            return
        self.k = torch.cat([self.k, k_t], dim=1)
        self.v = torch.cat([self.v, v_t], dim=1)
        overflow = self.k.shape[1] - (self.n_sink + self.window)
        if overflow > 0:
            keep_sink_k = self.k[:, : self.n_sink]
            keep_sink_v = self.v[:, : self.n_sink]
            self.k = torch.cat([keep_sink_k, self.k[:, self.n_sink + overflow:]], dim=1)
            self.v = torch.cat([keep_sink_v, self.v[:, self.n_sink + overflow:]], dim=1)

    def __len__(self) -> int:
        return 0 if self.k is None else int(self.k.shape[1])


class SlidingSinkAttention(nn.Module):
    def __init__(self, cfg: MicroLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.wq = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(d, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(d, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)
        nn.init.zeros_(self.wo.weight)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
        self.rope = Rope(cfg.head_dim, cfg.rope_base)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.cfg.head_dim).transpose(1, 2)

    def _project(self, x: torch.Tensor):
        q = self._split_heads(self.wq(x), self.cfg.n_heads)
        k = self._split_heads(self.wk(x), self.cfg.n_kv_heads)
        v = self._split_heads(self.wv(x), self.cfg.n_kv_heads)
        return self.q_norm(q), self.k_norm(k), v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        k = k.repeat_interleave(self.cfg.group_size, dim=1)
        v = v.repeat_interleave(self.cfg.group_size, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.cfg.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        b = out.shape[0]
        out = out.transpose(1, 2).reshape(b, -1, self.cfg.n_heads * self.cfg.head_dim)
        return self.wo(out)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.cfg.n_sink + self.cfg.window:
            raise ValueError(
                "training path supports T <= n_sink + window; "
                "longer sequences require the chunked decode path"
            )
        q, k, v = self._project(x)
        positions = torch.arange(t, device=x.device)
        q = self.rope.rotate(q, positions)
        k = self.rope.rotate(k, positions)
        i = torch.arange(t, device=x.device)[:, None]
        j = torch.arange(t, device=x.device)[None, :]
        causal = j <= i
        in_window = j > i - self.cfg.window
        is_sink = j < self.cfg.n_sink
        mask = (causal & (in_window | is_sink))[None, None]
        return self._attend(q, k, v, mask)

    def forward_decode(self, x_t: torch.Tensor, cache: AttentionCache) -> torch.Tensor:
        if x_t.shape[1] != 1:
            raise ValueError(
                "decode path processes exactly one token per step; "
                "multi-token prefill requires per-query positions and a "
                "causal mask (chunked path, out of scope)"
            )
        q, k_t, v_t = self._project(x_t)
        cache.append(k_t.transpose(1, 2).contiguous(), v_t.transpose(1, 2).contiguous())
        k = cache.k.transpose(1, 2)
        v = cache.v.transpose(1, 2)
        cache_len = k.shape[2]
        positions = torch.arange(cache_len, device=x_t.device)
        k = self.rope.rotate(k, positions)
        q = self.rope.rotate(q, positions[-1:])
        return self._attend(q, k, v, None)


def hash_ngrams(ngrams: torch.Tensor, salt: int, buckets: int) -> torch.Tensor:
    multiplier = HASH_MULTIPLIER + 2 * salt + 1
    h = torch.full(ngrams.shape[:-1], salt, dtype=torch.int64, device=ngrams.device)
    for i in range(ngrams.shape[-1]):
        h = ((h ^ ngrams[..., i]) * multiplier) & HASH_MASK
    h = h ^ (h >> 21)
    return h % buckets


def build_ngrams(ids: torch.Tensor, order: int, pad_id: int) -> torch.Tensor:
    padded = F.pad(ids, (order - 1, 0), value=pad_id)
    return padded.unfold(-1, order, 1)


class EngramFusion(nn.Module):
    def __init__(self, cfg: MicroLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tables = nn.ModuleDict()
        for order in cfg.engram_orders:
            table = nn.Embedding(cfg.engram_buckets, 2 * d)
            nn.init.normal_(table.weight, std=0.02)
            with torch.no_grad():
                table.weight[:, d:].zero_()
            self.tables[str(order)] = table
        self.tau = nn.Parameter(
            torch.full((len(cfg.engram_orders),), cfg.gate_tau_init))
        self.beta = nn.Parameter(
            torch.full((len(cfg.engram_orders),), cfg.gate_beta_init))
        self.salts = [1_000_003 * (j + 1) + 17 for j in range(cfg.engram_hashes)]

    def forward(self, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d_model
        x_hat = F.normalize(h, dim=-1)
        pad_id = self.cfg.vocab_size + NGRAM_PAD_OFFSET
        out = torch.zeros_like(h)
        for oi, order in enumerate(self.cfg.engram_orders):
            table = self.tables[str(order)]
            ngrams = build_ngrams(ids, order, pad_id)
            for salt in self.salts:
                idx = hash_ngrams(ngrams, salt, self.cfg.engram_buckets)
                kv = table(idx)
                k_hat = F.normalize(kv[..., :d], dim=-1)
                v = kv[..., d:]
                sim = (x_hat * k_hat).sum(dim=-1)
                gate = torch.sigmoid(self.tau[oi] * sim + self.beta[oi])
                out = out + gate[..., None] * v
        n_slots = len(self.cfg.engram_orders) * self.cfg.engram_hashes
        return h + out / n_slots


class HadamardMLP(nn.Module):
    def __init__(self, cfg: MicroLMConfig):
        super().__init__()
        d = cfg.d_model
        self.d1a = nn.Parameter(torch.ones(d))
        self.d1b = nn.Parameter(torch.ones(d))
        self.d2b = nn.Parameter(torch.ones(d))
        self.d3 = nn.Parameter(torch.zeros(d))
        self.block_size = cfg.mlp_block_size
        if self.block_size:
            n_blocks = d // self.block_size
            eye = torch.eye(self.block_size).expand(n_blocks, -1, -1).clone()
            self.w2a = nn.Parameter(eye)
        else:
            self.d2a = nn.Parameter(torch.ones(d))

    def _inner_a(self, u: torch.Tensor) -> torch.Tensor:
        if not self.block_size:
            return self.d2a * u
        shape = u.shape
        blocks = u.reshape(-1, shape[-1] // self.block_size, self.block_size)
        mixed = torch.einsum("nbi,bij->nbj", blocks, self.w2a)
        return mixed.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_a = F.silu(self._inner_a(fwht(self.d1a * x)))
        branch_b = self.d2b * fwht(self.d1b * x)
        return self.d3 * fwht(branch_a * branch_b)


class MHCLaneRead(nn.Module):
    def __init__(self, cfg: MicroLMConfig):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(cfg.n_lanes, cfg.d_model) * 0.02)
        self.a = nn.Parameter(torch.zeros(cfg.n_lanes))
        uniform_gate = math.log(1.0 / (cfg.n_lanes - 1))
        self.b = nn.Parameter(torch.full((cfg.n_lanes,), uniform_gate))

    def forward(self, lanes: torch.Tensor) -> torch.Tensor:
        x_hat = zc_normalize(lanes.mean(dim=0))
        logits = self.a[:, None, None, None] * torch.einsum(
            "kd,btd->kbt", self.phi, x_hat
        )[..., None] + self.b[:, None, None, None]
        gates = torch.sigmoid(logits)
        return (gates * lanes).sum(dim=0)


class MHCLaneWrite(nn.Module):
    def __init__(self, cfg: MicroLMConfig):
        super().__init__()
        self.sinkhorn_iters = cfg.sinkhorn_iters
        self.A = nn.Parameter(cfg.mhc_alpha_init * torch.eye(cfg.n_lanes))
        self.g = nn.Parameter(torch.ones(cfg.n_lanes))

    def mixing_matrix(self) -> torch.Tensor:
        m = torch.exp(self.A)
        for _ in range(self.sinkhorn_iters):
            m = m / m.sum(dim=1, keepdim=True)
            m = m / m.sum(dim=0, keepdim=True)
        return m

    def forward(self, lanes: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self.mixing_matrix()
        mixed = torch.einsum("kj,jbtd->kbtd", p, lanes)
        return mixed + self.g[:, None, None, None] * y.unsqueeze(0)


class MicroLMLayer(nn.Module):
    def __init__(self, cfg: MicroLMConfig, layer_idx: int):
        super().__init__()
        self.read = MHCLaneRead(cfg)
        self.engram = EngramFusion(cfg) if layer_idx in cfg.engram_layers else None
        self.attn_norm = ZCRMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SlidingSinkAttention(cfg)
        self.mlp_norm = ZCRMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = HadamardMLP(cfg)
        self.write = MHCLaneWrite(cfg)

    def forward(self, lanes: torch.Tensor, ids: torch.Tensor,
                cache: AttentionCache | None = None) -> torch.Tensor:
        x_bar = self.read(lanes)
        h = x_bar
        if self.engram is not None:
            h = self.engram(h, ids)
        if cache is None:
            h = h + self.attn.forward_train(self.attn_norm(h))
        else:
            h = h + self.attn.forward_decode(self.attn_norm(h), cache)
        h = h + self.mlp(self.mlp_norm(h))
        return self.write(lanes, h - x_bar)


class MicroLM(nn.Module):
    def __init__(self, cfg: MicroLMConfig | None = None):
        super().__init__()
        self.cfg = cfg or MicroLMConfig()
        c = self.cfg
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList(
            MicroLMLayer(c, i) for i in range(c.n_layers))
        self.readout_theta = nn.Parameter(torch.zeros(c.n_lanes))
        self.final_norm = ZCRMSNorm(c.d_model, c.norm_eps)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _readout(self, lanes: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.readout_theta, dim=0)
        merged = torch.einsum("k,kbtd->btd", weights, lanes)
        return self.final_norm(merged) @ self.embedding.weight.T * self.logit_scale

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(ids)
        lanes = h.unsqueeze(0).expand(self.cfg.n_lanes, -1, -1, -1).contiguous()
        for layer in self.layers:
            lanes = layer(lanes, ids)
        return self._readout(lanes)

    def init_caches(self) -> list[AttentionCache]:
        return [AttentionCache(self.cfg) for _ in self.layers]

    @torch.no_grad()
    def decode_step(self, ids_so_far: torch.Tensor,
                    caches: list[AttentionCache]) -> torch.Tensor:
        max_order = max(self.cfg.engram_orders)
        ids_ctx = ids_so_far[:, -max_order:]
        h = self.embedding(ids_so_far[:, -1:])
        lanes = h.unsqueeze(0).expand(self.cfg.n_lanes, -1, -1, -1).contiguous()
        for layer, cache in zip(self.layers, caches):
            lanes = self._decode_layer(layer, lanes, ids_ctx, cache)
        return self._readout(lanes)

    def _decode_layer(self, layer: MicroLMLayer, lanes: torch.Tensor,
                      ids_ctx: torch.Tensor, cache: AttentionCache) -> torch.Tensor:
        x_bar = layer.read(lanes)
        h = x_bar
        if layer.engram is not None:
            max_order = max(self.cfg.engram_orders)
            pad_id = self.cfg.vocab_size + NGRAM_PAD_OFFSET
            padded = F.pad(ids_ctx, (max_order - ids_ctx.shape[1], 0), value=pad_id)
            h = self._engram_last_token(layer.engram, h, padded)
        h = h + layer.attn.forward_decode(layer.attn_norm(h), cache)
        h = h + layer.mlp(layer.mlp_norm(h))
        return layer.write(lanes, h - x_bar)

    def _engram_last_token(self, engram: EngramFusion, h: torch.Tensor,
                           padded_ids: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d_model
        x_hat = F.normalize(h, dim=-1)
        out = torch.zeros_like(h)
        for oi, order in enumerate(self.cfg.engram_orders):
            table = engram.tables[str(order)]
            ngram = padded_ids[:, -order:].unsqueeze(1)
            for salt in engram.salts:
                idx = hash_ngrams(ngram, salt, self.cfg.engram_buckets)
                kv = table(idx)
                k_hat = F.normalize(kv[..., :d], dim=-1)
                v = kv[..., d:]
                sim = (x_hat * k_hat).sum(dim=-1)
                gate = torch.sigmoid(engram.tau[oi] * sim + engram.beta[oi])
                out = out + gate[..., None] * v
        n_slots = len(self.cfg.engram_orders) * self.cfg.engram_hashes
        return h + out / n_slots

    def active_parameter_count(self) -> int:
        total = 0
        for name, p in self.named_parameters():
            if ".tables." in name:
                continue
            total += p.numel()
        return total
