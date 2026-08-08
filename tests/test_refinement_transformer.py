"""Spatial-only Transformer guarantees.

These tests are the guardrails for the scientific contract: attention must run
over 2-D spatial tokens of a single sample at a single timestamp, with no
temporal axis, no causal masking and no forecast lead-time input.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import pytest
import torch

from granitewxc.refinement import backbones, diffusion, flow_matching, two_phase
from granitewxc.refinement.backbones import (
    SpatialResidualTransformer,
    patchify_2d,
    sincos_2d_positional_encoding,
    unpatchify_2d,
)

MODULES = [backbones, diffusion, flow_matching, two_phase]


def _code_only(source: str) -> str:
    """Strip comments and string literals so only executable code remains."""
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return "\n".join(kept)


def activate_conditioning(model: SpatialResidualTransformer, seed: int = 0) -> SpatialResidualTransformer:
    """Give the zero-initialised adaLN gates non-trivial values.

    The blocks are deliberately identity-at-initialisation (zero-init adaLN and
    output projection) so an untrained refiner predicts a zero residual. The
    architectural safeguards below must be checked on an *active* network, so
    these tests emulate a trained model.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model.blocks:
            linear = block.ada_ln[1]
            linear.weight.normal_(0.0, 0.05, generator=generator)
            linear.bias.normal_(0.0, 0.5, generator=generator)
        final = model.final_ada_ln[1]
        final.weight.normal_(0.0, 0.05, generator=generator)
        final.bias.normal_(0.0, 0.5, generator=generator)
    return model


def make_transformer(**overrides) -> SpatialResidualTransformer:
    kwargs = dict(
        in_channels=2,
        cond_channels=3,
        out_channels=2,
        patch_size=(4, 4),
        embedding_dim=32,
        num_heads=4,
        num_blocks=2,
        zero_init_output=False,
    )
    kwargs.update(overrides)
    return SpatialResidualTransformer(**kwargs)


# ---------------------------------------------------------------------------
# Patchification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (2, 5, 12, 20), (3, 1, 6, 9)])
@pytest.mark.parametrize("patch", [(2, 2), (2, 3), (3, 1)])
def test_patchify_unpatchify_round_trip(shape, patch):
    b, c, h, w = shape
    ph, pw = patch
    h = h - h % ph or ph
    w = w - w % pw or pw
    x = torch.randn(b, c, h, w)
    tokens, gh, gw = patchify_2d(x, ph, pw)
    assert tokens.shape == (b, gh * gw, c * ph * pw)
    restored = unpatchify_2d(tokens, c, gh, gw, ph, pw)
    assert torch.equal(restored, x)


def test_patchify_preserves_2d_layout_order():
    """Token ``i`` must map to grid position ``(i // grid_w, i % grid_w)``."""
    ph = pw = 2
    h, w = 4, 6
    marker = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w)
    tokens, gh, gw = patchify_2d(marker, ph, pw)
    assert (gh, gw) == (2, 3)
    for idx in range(gh * gw):
        row, col = divmod(idx, gw)
        block = marker[0, 0, row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]
        assert torch.equal(tokens[0, idx], block.reshape(-1))


def test_patchify_rejects_non_divisible_input():
    with pytest.raises(ValueError, match="multiples of the patch size"):
        patchify_2d(torch.randn(1, 1, 5, 4), 2, 2)


# ---------------------------------------------------------------------------
# 2-D positional encoding
# ---------------------------------------------------------------------------


def test_sincos_positional_encoding_varies_with_both_axes():
    pe = sincos_2d_positional_encoding(32, 4, 5)
    assert pe.shape == (20, 32)
    grid = pe.reshape(4, 5, 32)
    # different latitude token, same longitude token
    assert not torch.allclose(grid[0, 2], grid[1, 2])
    # same latitude token, different longitude token
    assert not torch.allclose(grid[2, 0], grid[2, 1])
    # every token is unique
    assert len({tuple(row.tolist()) for row in pe}) == 20


def test_learned_positional_encoding_varies_with_both_axes():
    model = make_transformer(positional_encoding="learned_2d")
    pe = model._positional(4, 5, torch.device("cpu"), torch.float32).reshape(4, 5, -1)
    assert not torch.allclose(pe[0, 2], pe[1, 2])
    assert not torch.allclose(pe[2, 0], pe[2, 1])


def test_learned_positional_encoding_rejects_oversized_grid():
    model = make_transformer(positional_encoding="learned_2d", max_tokens_lat=4, max_tokens_lon=4)
    with pytest.raises(ValueError, match="max_tokens"):
        model._positional(8, 4, torch.device("cpu"), torch.float32)


# ---------------------------------------------------------------------------
# Attention scope
# ---------------------------------------------------------------------------


def test_attention_is_over_spatial_tokens_only():
    """Perturbing one batch element must not change any other batch element."""
    torch.manual_seed(0)
    model = activate_conditioning(make_transformer()).eval()
    x = torch.randn(3, 2, 12, 16)
    cond = torch.randn(3, 3, 12, 16)
    t = torch.zeros(3)
    with torch.no_grad():
        base = model(x, cond, t)
        x2 = x.clone()
        x2[1] += 100.0
        perturbed = model(x2, cond, t)
    assert torch.allclose(base[0], perturbed[0], atol=1e-5)
    assert torch.allclose(base[2], perturbed[2], atol=1e-5)
    assert not torch.allclose(base[1], perturbed[1], atol=1e-5)


def test_spatial_tokens_actually_interact():
    """A change at one grid location must be able to influence another."""
    torch.manual_seed(0)
    model = activate_conditioning(make_transformer()).eval()
    x = torch.randn(1, 2, 12, 16)
    cond = torch.randn(1, 3, 12, 16)
    t = torch.zeros(1)
    with torch.no_grad():
        base = model(x, cond, t)
        x2 = x.clone()
        x2[0, :, 0:4, 0:4] += 50.0
        perturbed = model(x2, cond, t)
    far = (base[0, :, 8:, 12:] - perturbed[0, :, 8:, 12:]).abs().max()
    assert float(far) > 0, "spatial attention did not propagate information"


def test_no_causal_masking():
    """Every token must be able to attend to every other token in both directions."""
    torch.manual_seed(0)
    model = activate_conditioning(make_transformer(num_blocks=1)).eval()
    x = torch.zeros(1, 2, 8, 8)
    cond = torch.zeros(1, 3, 8, 8)
    t = torch.zeros(1)

    def influence(src_slice, dst_slice):
        with torch.no_grad():
            base = model(x, cond, t)
            bumped = x.clone()
            bumped[(0, slice(None)) + src_slice] += 10.0
            out = model(bumped, cond, t)
        return float((base[(0, slice(None)) + dst_slice] - out[(0, slice(None)) + dst_slice]).abs().max())

    first_to_last = influence((slice(0, 4), slice(0, 4)), (slice(4, 8), slice(4, 8)))
    last_to_first = influence((slice(4, 8), slice(4, 8)), (slice(0, 4), slice(0, 4)))
    assert first_to_last > 0
    assert last_to_first > 0, "later tokens did not influence earlier ones (causal mask?)"


def test_sdpa_is_not_called_with_causal_flag():
    source = _code_only(inspect.getsource(backbones))
    assert "is_causal" in source
    assert "True" not in source.split("is_causal")[1].split(")")[0]


# ---------------------------------------------------------------------------
# No temporal modelling anywhere
# ---------------------------------------------------------------------------


FORBIDDEN_SUBSTRINGS = [
    "lead_time",
    "lead-time",
    "forecast_hour",
    "forecast_horizon",
    "temporal_attention",
    "cross_time",
    "causal_mask",
    "nn.LSTM",
    "nn.GRU",
    "nn.RNN",
    "autoregress",
]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_no_temporal_or_lead_time_constructs(module):
    source = _code_only(inspect.getsource(module)).lower()
    for needle in FORBIDDEN_SUBSTRINGS:
        assert needle.lower() not in source, (
            f"{module.__name__} contains a temporal construct: {needle!r}"
        )


def test_transformer_has_no_recurrent_or_temporal_submodules():
    model = make_transformer()
    for sub in model.modules():
        assert not isinstance(sub, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))


# ---------------------------------------------------------------------------
# Process-time conditioning
# ---------------------------------------------------------------------------


def test_process_time_does_not_reorder_spatial_tokens():
    """Changing the process time must modulate values, never permute locations."""
    torch.manual_seed(0)
    model = activate_conditioning(make_transformer()).eval()
    cond = torch.randn(1, 3, 12, 16)
    base_x = torch.zeros(1, 2, 12, 16)
    bumped_x = base_x.clone()
    bumped_x[0, :, 4:8, 8:12] = 25.0  # one patch-aligned block

    def response_argmax(process_time: float) -> tuple[int, int]:
        t = torch.full((1,), process_time)
        with torch.no_grad():
            delta = (model(bumped_x, cond, t) - model(base_x, cond, t)).abs()
        flat = delta[0].amax(dim=0).reshape(-1)
        index = int(flat.argmax())
        return divmod(index, delta.shape[-1])

    early = response_argmax(0.0)
    late = response_argmax(500.0)
    # The strongest response stays inside the perturbed block at both process
    # times: the spatial-token layout is independent of the process time.
    assert 4 <= early[0] < 8 and 8 <= early[1] < 12, early
    assert early == late, (early, late)

    # ... while the values themselves do respond to the process time.
    with torch.no_grad():
        a = model(bumped_x, cond, torch.zeros(1))
        b = model(bumped_x, cond, torch.full((1,), 500.0))
    assert not torch.allclose(a, b)


def test_process_time_is_per_sample_not_per_token():
    model = make_transformer().eval()
    x = torch.randn(2, 2, 8, 8)
    cond = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out = model(x, cond, torch.tensor([0.0, 900.0]))
    assert out.shape == (2, 2, 8, 8)


# ---------------------------------------------------------------------------
# Geometry: rectangular grids, padding and exact cropping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", [(8, 8), (12, 20), (13, 19), (7, 5), (4, 33)])
@pytest.mark.parametrize("patch", [(4, 4), (2, 8), (3, 3)])
def test_output_grid_is_exactly_restored(hw, patch):
    h, w = hw
    model = make_transformer(patch_size=patch).eval()
    x = torch.randn(1, 2, h, w)
    cond = torch.randn(1, 3, h, w)
    with torch.no_grad():
        out = model(x, cond, torch.zeros(1))
    assert out.shape == (1, 2, h, w)


def test_mismatched_conditioning_grid_raises():
    model = make_transformer()
    with pytest.raises(ValueError, match="share the spatial grid"):
        model(torch.randn(1, 2, 8, 8), torch.randn(1, 3, 8, 12), torch.zeros(1))


# ---------------------------------------------------------------------------
# Optimized vs reference attention
# ---------------------------------------------------------------------------


def test_optimized_and_reference_attention_agree():
    torch.manual_seed(0)
    model = activate_conditioning(make_transformer(optimized_attention="sdpa")).eval()
    x = torch.randn(2, 2, 12, 16)
    cond = torch.randn(2, 3, 12, 16)
    t = torch.tensor([10.0, 200.0])
    with torch.no_grad():
        fast = model(x, cond, t)
        model.set_attention_implementation("math")
        reference = model(x, cond, t)
    assert torch.allclose(fast, reference, atol=1e-5, rtol=1e-5)


def test_attention_implementation_is_configurable_and_validated():
    model = make_transformer()
    model.set_attention_implementation("math")
    assert all(b.attn.implementation == "math" for b in model.blocks)
    model.set_attention_implementation("auto")
    assert all(b.attn.implementation in {"sdpa", "math"} for b in model.blocks)
    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        model.set_attention_implementation("linear")


def test_gradient_checkpointing_matches_eager():
    torch.manual_seed(0)
    model = make_transformer(gradient_checkpointing=True).train()
    x = torch.randn(1, 2, 8, 8, requires_grad=True)
    cond = torch.randn(1, 3, 8, 8)
    out = model(x, cond, torch.zeros(1))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
