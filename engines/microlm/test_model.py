from __future__ import annotations

import math

import pytest
import torch

from model import (
    AttentionCache, EngramFusion, HadamardMLP, MHCLaneRead, MHCLaneWrite,
    MicroLM, MicroLMConfig, SlidingSinkAttention, build_ngrams, fwht,
    hash_ngrams, zc_normalize,
)

SMALL = MicroLMConfig(
    vocab_size=512,
    d_model=128,
    n_layers=6,
    n_heads=4,
    n_kv_heads=2,
    head_dim=32,
    n_lanes=4,
    window=16,
    n_sink=4,
    engram_layers=(1, 3),
    engram_buckets=1 << 10,
    engram_hashes=2,
)


def _perturb(model: MicroLM, std: float = 0.01, seed: int = 7) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(std * torch.randn(p.shape, generator=gen))


def test_fwht_double_application_recovers_input():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 128)
    assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)


def test_fwht_preserves_norm():
    torch.manual_seed(1)
    x = torch.randn(4, 128)
    assert torch.allclose(
        x.norm(dim=-1), fwht(x).norm(dim=-1), atol=1e-5)


def test_model_forward_produces_finite_logits_with_expected_shape():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    ids = torch.randint(0, SMALL.vocab_size, (2, 12))
    logits = model(ids)
    assert logits.shape == (2, 12, SMALL.vocab_size)
    assert torch.isfinite(logits).all()


def test_model_init_is_exact_noop_pipeline():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    ids = torch.randint(0, SMALL.vocab_size, (2, 10))
    logits = model(ids)
    h = model.embedding(ids)
    expected = model.final_norm(h) @ model.embedding.weight.T * model.logit_scale
    assert torch.allclose(logits, expected, atol=1e-5)


def test_residual_stream_norm_stays_bounded_across_layers():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    _perturb(model, std=0.02)
    ids = torch.randint(0, SMALL.vocab_size, (2, 12))
    norms = []
    lanes = model.embedding(ids).unsqueeze(0).expand(
        SMALL.n_lanes, -1, -1, -1).contiguous()
    for layer in model.layers:
        lanes = layer(lanes, ids)
        norms.append(float(lanes.norm()))
    assert max(norms) / min(norms) < 5.0


def test_engram_gate_spans_wide_range_vs_legacy_dead_zone():
    cos = torch.tensor([-1.0, 1.0])
    legacy = torch.sigmoid(cos / math.sqrt(512))
    legacy_span = float(legacy[1] - legacy[0])
    fixed = torch.sigmoid(SMALL.gate_tau_init * cos + SMALL.gate_beta_init)
    fixed_span = float(fixed[1] - fixed[0])
    assert legacy_span < 0.05
    assert fixed_span > 0.5


def test_mixing_matrix_is_doubly_stochastic_and_near_identity():
    write = MHCLaneWrite(SMALL)
    p = write.mixing_matrix()
    ones = torch.ones(SMALL.n_lanes)
    assert torch.allclose(p.sum(dim=0), ones, atol=1e-4)
    assert torch.allclose(p.sum(dim=1), ones, atol=1e-4)
    assert float(p.diagonal().min()) > 0.99


def test_lane_read_at_init_equals_mean_of_lanes():
    torch.manual_seed(0)
    read = MHCLaneRead(SMALL)
    lanes = torch.randn(SMALL.n_lanes, 2, 6, SMALL.d_model)
    assert torch.allclose(read(lanes), lanes.mean(dim=0), atol=1e-5)


def test_decode_matches_training_forward_for_short_sequences():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    _perturb(model, std=0.02)
    model.eval()
    ids = torch.randint(0, SMALL.vocab_size, (1, 10))
    with torch.no_grad():
        ref = model(ids)
    caches = model.init_caches()
    for t in range(ids.shape[1]):
        step_logits = model.decode_step(ids[:, : t + 1], caches)
        assert torch.allclose(step_logits[:, 0], ref[:, t], atol=1e-4), f"t={t}"


def test_decode_cache_stays_bounded_for_long_generation():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    _perturb(model, std=0.02)
    model.eval()
    caches = model.init_caches()
    ids = torch.randint(0, SMALL.vocab_size, (1, 1))
    limit = SMALL.n_sink + SMALL.window
    for _ in range(3 * limit):
        logits = model.decode_step(ids, caches)
        assert torch.isfinite(logits).all()
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    assert all(len(c) <= limit for c in caches)


def test_training_path_rejects_sequences_beyond_window_plus_sinks():
    model = MicroLM(SMALL)
    too_long = SMALL.n_sink + SMALL.window + 1
    ids = torch.randint(0, SMALL.vocab_size, (1, too_long))
    with pytest.raises(ValueError):
        model(ids)


def test_all_parameters_receive_gradient_after_perturbation():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    _perturb(model, std=0.02)
    ids = torch.randint(0, SMALL.vocab_size, (2, 12))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, SMALL.vocab_size),
        ids[:, 1:].reshape(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    zero_grads = [
        n for n, p in model.named_parameters()
        if ".tables." not in n and float(p.grad.abs().max()) == 0.0
    ]
    assert len(zero_grads) <= 2, zero_grads


def test_hadamard_mlp_blockdiag_init_matches_diagonal_variant():
    torch.manual_seed(0)
    cfg_diag = SMALL
    cfg_block = MicroLMConfig(**{**cfg_diag.__dict__, "mlp_block_size": 32})
    mlp_diag = HadamardMLP(cfg_diag)
    mlp_block = HadamardMLP(cfg_block)
    with torch.no_grad():
        for m in (mlp_diag, mlp_block):
            m.d3.fill_(1.0)
        mlp_block.d1a.copy_(mlp_diag.d1a)
        mlp_block.d1b.copy_(mlp_diag.d1b)
        mlp_block.d2b.copy_(mlp_diag.d2b)
    x = torch.randn(3, 7, cfg_diag.d_model)
    assert torch.allclose(mlp_diag(x), mlp_block(x), atol=1e-5)


def test_engram_lookup_is_deterministic_and_position_consistent():
    torch.manual_seed(0)
    engram = EngramFusion(SMALL)
    with torch.no_grad():
        engram.tables["2"].weight.normal_(std=0.05)
        engram.tables["3"].weight.normal_(std=0.05)
    ids = torch.randint(0, SMALL.vocab_size, (1, 8))
    h = torch.randn(1, 8, SMALL.d_model)
    out1 = engram(h, ids)
    out2 = engram(h, ids)
    assert torch.equal(out1, out2)
    h_last = h[:, -1:, :]
    model = MicroLM(SMALL)
    padded = torch.nn.functional.pad(
        ids[:, -3:], (0, 0), value=SMALL.vocab_size + 1)
    single = model._engram_last_token(engram, h_last, padded)
    assert torch.allclose(single[:, 0], out1[:, -1], atol=1e-5)


def test_active_parameter_count_within_budget_on_reference_config():
    model = MicroLM(MicroLMConfig())
    count = model.active_parameter_count()
    assert 20_000_000 < count < 24_000_000, count


def test_model_trains_from_exact_init_without_perturbation():
    torch.manual_seed(0)
    model = MicroLM(SMALL)
    ids = torch.randint(0, SMALL.vocab_size, (2, 12))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, SMALL.vocab_size),
            ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.8 * losses[0], losses
    wo_norm = float(model.layers[0].attn.wo.weight.abs().max())
    d3_norm = float(model.layers[0].mlp.d3.abs().max())
    assert wo_norm > 0.0
    assert d3_norm > 0.0


def test_engram_gate_modulates_output_through_real_module():
    torch.manual_seed(0)
    engram = EngramFusion(SMALL)
    d = SMALL.d_model
    ids = torch.randint(0, SMALL.vocab_size, (1, 4))
    pad_id = SMALL.vocab_size + 1
    direction = torch.zeros(d)
    direction[0] = 1.0
    with torch.no_grad():
        for order in SMALL.engram_orders:
            table = engram.tables[str(order)]
            ngrams = build_ngrams(ids, order, pad_id)
            for salt in engram.salts:
                idx = hash_ngrams(ngrams[:, -1], salt, SMALL.engram_buckets)
                table.weight[idx, :d] = direction
                table.weight[idx, d:] = 1.0
    h_aligned = direction.expand(1, 4, d) * 3.0
    h_opposed = -h_aligned
    out_aligned = (engram(h_aligned, ids) - h_aligned)[:, -1].norm()
    out_opposed = (engram(h_opposed, ids) - h_opposed)[:, -1].norm()
    assert float(out_aligned) > 10.0 * float(out_opposed)


def test_hash_salts_produce_independent_collisions():
    torch.manual_seed(0)
    buckets = SMALL.engram_buckets
    ngrams = torch.randint(0, SMALL.vocab_size, (200_000, 2), dtype=torch.int64)
    engram = EngramFusion(SMALL)
    salt_a, salt_b = engram.salts[0], engram.salts[1]
    idx_a = hash_ngrams(ngrams, salt_a, buckets)
    idx_b = hash_ngrams(ngrams, salt_b, buckets)
    shift = (idx_b - idx_a) % buckets
    assert shift.unique().numel() > buckets // 4
    ref_a = idx_a[0]
    colliders = (idx_a[1:] == ref_a).nonzero().squeeze(-1) + 1
    if colliders.numel() > 0:
        also_b = (idx_b[colliders] == idx_b[0]).float().mean()
        assert float(also_b) < 0.05


def test_decode_attention_is_stationary_after_eviction_with_constant_input():
    torch.manual_seed(0)
    attn = SlidingSinkAttention(SMALL)
    with torch.no_grad():
        attn.wo.weight.normal_(std=0.05)
    cache = AttentionCache(SMALL)
    x = torch.randn(1, 1, SMALL.d_model)
    outs = []
    with torch.no_grad():
        for _ in range(3 * (SMALL.n_sink + SMALL.window)):
            outs.append(attn.forward_decode(x, cache))
    late = torch.cat(outs[-10:], dim=1)
    assert float(late.abs().max()) > 0.0
    assert torch.allclose(late, late[:, :1].expand_as(late), atol=1e-5)


def test_decode_rejects_multi_token_input():
    model = MicroLM(SMALL)
    cache = AttentionCache(SMALL)
    x = torch.randn(1, 3, SMALL.d_model)
    with pytest.raises(ValueError):
        model.layers[0].attn.forward_decode(x, cache)


def test_attention_sink_tokens_remain_visible_beyond_window():
    torch.manual_seed(0)
    cache = AttentionCache(SMALL)
    b, kvh, hd = 1, SMALL.n_kv_heads, SMALL.head_dim
    marker = torch.full((b, SMALL.n_sink, kvh, hd), 9.0)
    cache.append(marker, marker)
    for _ in range(SMALL.window * 2):
        step = torch.randn(b, 1, kvh, hd)
        cache.append(step, step)
    assert len(cache) == SMALL.n_sink + SMALL.window
    assert torch.equal(cache.k[:, : SMALL.n_sink], marker)
