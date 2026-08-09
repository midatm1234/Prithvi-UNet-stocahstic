"""Tests for the optional conditional diffusion decoder head.

These tests use small dummy tensors only (no data, no GPU, no Prithvi backbone),
so they run quickly and verify:

* the deterministic head still imports/builds,
* the score-based SDEs produce finite marginals,
* the diffusion training loss is finite and gradients flow,
* the diffusion sampler returns ``[B, output_channels, H, W]`` (even + odd sizes),
* config plumbing (head-type resolver, loss passthrough, config parsing),
* standardized encode/decode round-trips and precipitation non-negativity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from granitewxc.decoders.diffusion_head import (
    ConditionalScoreUNet,
    DiffusionHead,
    DiffusionHeadConfig,
    build_diffusion_head,
)
import granitewxc.decoders.diffusion_head as diffusion_head_module
from granitewxc.decoders.downscaling import ConvEncoderDecoder
from granitewxc.models.cordex_finetune_model import (
    ClimateECCCFinetuneWrapper,
    resolve_head_type,
)
from granitewxc.models.diffusion_loss import (
    DiffusionLossPassthrough,
    JointResidualDiffusionLoss,
    score_matching_loss,
)
from granitewxc.models.diffusion_sampling import get_ddim_sampler
from granitewxc.models.diffusion_sde import VESDE, VPSDE, build_sde, subVPSDE
from granitewxc.models.loss import build_loss_fn


def _small_head_config(**overrides) -> DiffusionHeadConfig:
    params = dict(
        sde="subvpsde",
        num_scales=50,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=16,
        num_sampling_steps=3,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
    )
    params.update(overrides)
    return DiffusionHeadConfig(**params)


# ---------------------------------------------------------------------------
# SDE math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls", [("vpsde", VPSDE), ("subvpsde", subVPSDE), ("vesde", VESDE)])
def test_build_sde_returns_expected_type_and_finite_marginals(name, cls):
    sde = build_sde({"sde": name}, num_scales=100)
    assert isinstance(sde, cls)
    assert sde.N == 100

    x = torch.randn(2, 2, 6, 6)
    t = torch.rand(2)
    mean, std = sde.marginal_prob(x, t)
    assert mean.shape == x.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std >= 0).all()

    prior = sde.prior_sampling((2, 2, 6, 6))
    assert prior.shape == (2, 2, 6, 6)
    assert torch.isfinite(prior).all()


def test_build_sde_rejects_unknown():
    with pytest.raises(ValueError):
        build_sde({"sde": "not-a-real-sde"})


# ---------------------------------------------------------------------------
# Deterministic head still available
# ---------------------------------------------------------------------------
def test_deterministic_conv_head_still_builds_and_runs():
    head = ConvEncoderDecoder(
        in_channels=4,
        channels=8,
        out_channels=2,
        kernel_size=[3],
        scale=[2],
        upsampling_mode="nearest",
    )
    out = head(torch.randn(1, 4, 5, 5))
    assert out.shape[0] == 1 and out.shape[1] == 2


# ---------------------------------------------------------------------------
# Diffusion training loss
# ---------------------------------------------------------------------------
def test_diffusion_training_loss_is_finite_and_backprops():
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=6, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(2, 6, 8, 8)
    target = torch.randn(2, 2, 8, 8)

    loss = head.training_loss(cond, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_clean_x0_objective_matches_exact_epsilon_derived_formula():
    """The stabilized x0 term must implement the documented SNR identity."""
    torch.manual_seed(3)
    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=20)
    target = torch.randn(3, 2, 4, 5)
    cond = torch.zeros(3, 1, 4, 5)
    weight = 0.7
    cap = 3.0

    def zero_score(x, _cond, _t):
        return torch.zeros_like(x)

    generator = torch.Generator().manual_seed(41)
    total, terms = score_matching_loss(
        sde,
        zero_score,
        target,
        cond,
        eps=1e-5,
        generator=generator,
        clean_x0_reconstruction_weight=weight,
        clean_x0_inverse_snr_cap=cap,
        return_terms=True,
    )

    # Replay the exact random t/noise draws and evaluate the formula directly.
    replay = torch.Generator().manual_seed(41)
    t = torch.rand(3, generator=replay).mul(sde.T - 1e-5).add(1e-5)
    noise = torch.empty_like(target).normal_(generator=replay)
    mean, std = sde.marginal_prob(target, t)
    noisy = mean + std[:, None, None, None] * noise
    unit = torch.ones(3, 1, 1, 1)
    alpha = sde.marginal_prob(unit, t)[0][:, 0, 0, 0]
    x0_prediction = noisy / alpha[:, None, None, None]
    snr = alpha.square() / std.square()
    stabilizer = torch.minimum(torch.ones_like(snr), cap * snr)
    expected_x0 = (
        stabilizer
        * (x0_prediction - target).square().flatten(1).mean(dim=1)
    ).mean()
    expected_epsilon = noise.square().flatten(1).mean(dim=1).mean()
    expected_total = expected_epsilon + weight * expected_x0

    assert torch.allclose(terms["score_matching"], expected_epsilon)
    assert torch.allclose(terms["epsilon_mse"], expected_epsilon)
    assert torch.allclose(terms["clean_x0_reconstruction"], expected_x0)
    assert torch.allclose(total, expected_total)


def test_zero_clean_x0_weight_preserves_original_score_loss_exactly():
    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=20)
    target = torch.randn(2, 1, 3, 3)
    cond = torch.zeros(2, 1, 3, 3)

    def zero_score(x, _cond, _t):
        return torch.zeros_like(x)

    legacy = score_matching_loss(
        sde,
        zero_score,
        target,
        cond,
        generator=torch.Generator().manual_seed(9),
    )
    configured, terms = score_matching_loss(
        sde,
        zero_score,
        target,
        cond,
        generator=torch.Generator().manual_seed(9),
        clean_x0_reconstruction_weight=0.0,
        clean_x0_inverse_snr_cap=100.0,
        return_terms=True,
    )
    assert torch.equal(configured, legacy)
    assert terms["clean_x0_reconstruction"].item() == 0.0
    assert terms["clean_x0_weighted"].item() == 0.0


def test_diffusion_forward_alias_matches_training_loss():
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=4, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(1, 4, 6, 6)
    target = torch.randn(1, 2, 6, 6)
    torch.manual_seed(1)
    a = head.training_loss(cond, target)
    torch.manual_seed(1)
    b = head(cond, target)
    assert torch.allclose(a, b)


def test_residual_training_loss_stops_baseline_and_conditioning_gradients():
    torch.manual_seed(0)
    head = DiffusionHead(
        cond_channels=4,
        output_channels=2,
        head_config=_small_head_config(residual_diffusion=True),
    )
    cond = torch.randn(1, 4, 6, 6, requires_grad=True)
    baseline = torch.randn(1, 2, 6, 6, requires_grad=True)
    target = torch.randn(1, 2, 6, 6)

    loss = head.training_loss(cond, target, baseline_std=baseline)
    loss.backward()

    assert cond.grad is None
    assert baseline.grad is None
    score_grads = [
        parameter.grad
        for parameter in head.score_model.parameters()
        if parameter.requires_grad
    ]
    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.abs().sum() > 0
        for grad in score_grads
    )


# ---------------------------------------------------------------------------
# Diffusion sampler shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h,w", [(8, 8), (7, 9)])
def test_diffusion_sampler_returns_target_shape(h, w):
    torch.manual_seed(0)
    head = DiffusionHead(cond_channels=5, output_channels=2, head_config=_small_head_config())
    cond = torch.randn(2, 5, 4, 4)  # low-res conditioning features
    with torch.no_grad():
        sample = head.sample(cond, h, w)
    assert sample.shape == (2, 2, h, w)
    assert torch.isfinite(sample).all()


def test_ode_sampler_returns_target_shape():
    torch.manual_seed(0)
    cfg = _small_head_config(sampling_method="ode", num_sampling_steps=3)
    head = DiffusionHead(cond_channels=4, output_channels=2, head_config=cfg)
    cond = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        sample = head.sample(cond, 8, 8)
    assert sample.shape == (1, 2, 8, 8)
    assert torch.isfinite(sample).all()


def _install_fixed_residual_sampler(monkeypatch, residual):
    def fake_build_sampler(_cfg, _sde, _shape, device):
        def sample(_score_fn, cond, generator=None):
            del generator
            return residual.to(device=device, dtype=cond.dtype).clone()

        return sample

    monkeypatch.setattr(diffusion_head_module, "build_sampler", fake_build_sampler)


def test_residual_application_scale_zero_returns_exact_baseline(monkeypatch):
    baseline = torch.randn(2, 2, 6, 7)

    def forbidden_sampler(*_args, **_kwargs):
        raise AssertionError("alpha=0 must not build or run a diffusion sampler")

    monkeypatch.setattr(diffusion_head_module, "build_sampler", forbidden_sampler)
    head = DiffusionHead(
        cond_channels=3,
        output_channels=2,
        head_config=_small_head_config(
            residual_diffusion=True,
            residual_application_scale=0.0,
            residual_magnitude_guard_multiple=0.0,
        ),
    )

    full, raw = head.sample_components(
        torch.randn(2, 3, 6, 7),
        6,
        7,
        baseline_std=baseline,
    )

    assert torch.equal(full, baseline)
    assert torch.equal(raw, torch.zeros_like(baseline))


def test_nonzero_residual_application_rejects_nonfinite_sample(monkeypatch):
    baseline = torch.zeros(1, 1, 2, 2)
    residual = torch.full_like(baseline, float("nan"))
    _install_fixed_residual_sampler(monkeypatch, residual)
    head = DiffusionHead(
        cond_channels=2,
        output_channels=1,
        head_config=_small_head_config(
            residual_diffusion=True,
            residual_application_scale=1.0,
            residual_magnitude_guard_multiple=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="NaN or infinity"):
        head.sample_components(
            torch.zeros(1, 2, 2, 2),
            2,
            2,
            baseline_std=baseline,
        )


def test_residual_application_scale_preserves_raw_residual_sign(monkeypatch):
    baseline = torch.tensor([[[[3.0, -2.0], [0.5, 4.0]]]])
    residual = torch.tensor([[[[-4.0, 2.0], [8.0, -6.0]]]])
    alpha = 0.25
    _install_fixed_residual_sampler(monkeypatch, residual)
    head = DiffusionHead(
        cond_channels=2,
        output_channels=1,
        head_config=_small_head_config(
            residual_diffusion=True,
            residual_application_scale=alpha,
            residual_magnitude_guard_multiple=0.0,
        ),
    )

    full, raw = head.sample_components(
        torch.zeros(1, 2, 2, 2),
        2,
        2,
        baseline_std=baseline,
    )

    assert torch.equal(raw, residual)
    assert torch.equal(full, baseline + alpha * residual)
    assert full[0, 0, 0, 0] < baseline[0, 0, 0, 0]
    assert full[0, 0, 0, 1] > baseline[0, 0, 0, 1]


def test_replicate_padding_keeps_constant_fields_boundary_invariant():
    torch.manual_seed(17)
    score_model = ConditionalScoreUNet(
        target_channels=2,
        cond_channels=3,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=16,
        dropout=0.0,
        projected_cond_channels=0,
        padding_mode="replicate",
        zero_init_output=False,
    ).eval()
    spatial_convs = [
        module
        for module in score_model.modules()
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3)
    ]
    assert spatial_convs
    assert all(module.padding_mode == "replicate" for module in spatial_convs)

    x = torch.full((2, 2, 9, 11), 0.75)
    cond = torch.full((2, 3, 9, 11), -0.25)
    t = torch.tensor([0.2, 0.7])
    with torch.no_grad():
        output = score_model(x, cond, t)

    per_example_reference = output[..., :1, :1].expand_as(output)
    assert torch.allclose(output, per_example_reference, atol=1e-5, rtol=1e-5)

    # Same learned weights with the legacy padding semantics acquire a spatial
    # edge response from zeros that do not exist in the regional climate field.
    zero_padded = ConditionalScoreUNet(
        target_channels=2,
        cond_channels=3,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=16,
        dropout=0.0,
        projected_cond_channels=0,
        padding_mode="zeros",
        zero_init_output=False,
    ).eval()
    zero_padded.load_state_dict(score_model.state_dict(), strict=True)
    with torch.no_grad():
        zero_output = zero_padded(x, cond, t)
    zero_reference = zero_output[..., 4:5, 5:6].expand_as(zero_output)
    assert not torch.allclose(zero_output, zero_reference, atol=1e-5, rtol=1e-5)


def test_score_output_layer_is_zero_initialized_by_default():
    score_model = ConditionalScoreUNet(
        target_channels=1,
        cond_channels=2,
        base_channels=8,
        channel_multipliers=(1,),
        num_res_blocks=1,
        time_embed_dim=16,
    )
    assert torch.count_nonzero(score_model.out_conv.weight).item() == 0
    assert torch.count_nonzero(score_model.out_conv.bias).item() == 0
    output = score_model(
        torch.randn(1, 1, 5, 6),
        torch.randn(1, 2, 5, 6),
        torch.tensor([0.5]),
    )
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("sde_cls", [VPSDE, subVPSDE])
def test_ddim_oracle_noise_prediction_reconstructs_x0_in_one_step(sde_cls):
    """A perfect epsilon/score prediction must exactly invert the forward marginal.

    Using one reverse step makes this a focused regression test for the DDIM
    signal coefficient. In particular, ``marginal_prob(zeros, t)`` returns a
    zero *mean field* and cannot be used to recover the multiplicative mean
    coefficient.
    """
    torch.manual_seed(11)
    batch, channels, height, width = 2, 1, 3, 4
    sde = sde_cls(beta_min=0.1, beta_max=20.0, N=1)
    x0 = torch.randn(batch, channels, height, width)
    noise = torch.randn_like(x0)
    terminal_t = torch.ones(batch)
    terminal_mean, terminal_std = sde.marginal_prob(x0, terminal_t)
    x_terminal = terminal_mean + terminal_std[:, None, None, None] * noise

    # Fix the prior so the reverse trajectory starts from the exact forward
    # marginal sample whose oracle score is known.
    sde.prior_sampling = lambda shape: x_terminal.clone()
    sampler = get_ddim_sampler(
        sde,
        (channels, height, width),
        eta=0.0,
        eps=1e-3,
        device="cpu",
    )
    cond = torch.zeros(batch, 1, height, width)

    def oracle_score(x, _cond, t):
        mean_t, std_t = sde.marginal_prob(x0, t)
        return -(x - mean_t) / torch.square(std_t[:, None, None, None])

    reconstructed = sampler(oracle_score, cond)
    assert torch.allclose(reconstructed, x0, atol=5e-5, rtol=5e-5)


def test_vpsde_ddim_eta_adds_nonzero_posterior_noise():
    """VP eta must affect the trajectory instead of being clamped to zero."""
    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=2)
    initial = torch.full((1, 1, 2, 2), 0.25)
    sde.prior_sampling = lambda shape: initial.clone()
    cond = torch.zeros(1, 1, 2, 2)

    def zero_score(x, _cond, t):
        return torch.zeros_like(x)

    stochastic_sampler = get_ddim_sampler(
        sde,
        (1, 2, 2),
        eta=1.0,
        eps=0.5,
        device="cpu",
    )
    first = stochastic_sampler(zero_score, cond, generator=torch.Generator().manual_seed(1))
    second = stochastic_sampler(zero_score, cond, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(first, second)

    deterministic_sampler = get_ddim_sampler(
        sde,
        (1, 2, 2),
        eta=0.0,
        eps=0.5,
        device="cpu",
    )
    first_det = deterministic_sampler(zero_score, cond)
    second_det = deterministic_sampler(zero_score, cond)
    assert torch.equal(first_det, second_det)


def test_subvpsde_ddim_rejects_nonzero_eta():
    sde = subVPSDE(beta_min=0.1, beta_max=20.0, N=2)
    with pytest.raises(ValueError, match="only implemented for VPSDE"):
        get_ddim_sampler(sde, (1, 2, 2), eta=0.2, device="cpu")


def test_build_diffusion_head_factory_reads_config():
    config = SimpleNamespace(
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion={
                "sde": "vpsde",
                "num_scales": 200,
                "prediction_type": "epsilon",
                "clean_x0_reconstruction_weight": "1.0",
                "clean_x0_inverse_snr_cap": "100.0",
                "base_channels": 8,
                "channel_multipliers": [1, 2],
                "num_sampling_steps": 4,
                "padding_mode": "reflect",
                "residual_application_scale": "0.25",
                "zero_init_output": False,
            },
        )
    )
    head = build_diffusion_head(config, cond_channels=3, output_channels=2)
    assert isinstance(head, DiffusionHead)
    assert head.cond_channels == 3
    assert head.output_channels == 2
    assert isinstance(head.sde, VPSDE)
    assert head.sde.N == 200
    assert head.cfg.prediction_type == "epsilon"
    assert head.cfg.clean_x0_reconstruction_weight == pytest.approx(1.0)
    assert head.cfg.clean_x0_inverse_snr_cap == pytest.approx(100.0)
    assert head.cfg.padding_mode == "reflect"
    assert head.cfg.residual_application_scale == pytest.approx(0.25)
    assert head.cfg.zero_init_output is False
    spatial_convs = [
        module
        for module in head.score_model.modules()
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3)
    ]
    assert spatial_convs
    assert all(module.padding_mode == "reflect" for module in spatial_convs)
    assert torch.count_nonzero(head.score_model.out_conv.weight).item() > 0


def test_diffusion_safety_config_defaults_fail_closed():
    cfg = DiffusionHeadConfig()
    assert cfg.padding_mode == "zeros"
    assert cfg.noise_conditioning_scale == 999.0
    assert cfg.residual_application_scale == 0.0
    assert cfg.zero_init_output is True


@pytest.mark.parametrize("padding_mode", ["constant", "edge", ""])
def test_diffusion_config_rejects_invalid_padding_mode(padding_mode):
    with pytest.raises(ValueError, match="padding_mode must be one of"):
        DiffusionHeadConfig(padding_mode=padding_mode)


@pytest.mark.parametrize(
    "residual_application_scale",
    [-0.01, 1.01, float("nan"), float("inf")],
)
def test_diffusion_config_rejects_unsafe_residual_application_scale(
    residual_application_scale,
):
    with pytest.raises(ValueError, match="residual_application_scale"):
        DiffusionHeadConfig(
            residual_application_scale=residual_application_scale
        )


def test_diffusion_config_rejects_non_boolean_zero_init_output():
    with pytest.raises(ValueError, match="zero_init_output must be a boolean"):
        DiffusionHeadConfig(zero_init_output="false")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clean_x0_reconstruction_weight", -0.1, "finite and non-negative"),
        ("clean_x0_inverse_snr_cap", 0.0, "finite and positive"),
    ],
)
def test_diffusion_config_rejects_invalid_clean_x0_options(field, value, message):
    with pytest.raises(ValueError, match=message):
        DiffusionHeadConfig(**{field: value})


@pytest.mark.parametrize("prediction_type", ["x0", "v_prediction", "score"])
def test_diffusion_config_rejects_unsupported_prediction_type(prediction_type):
    with pytest.raises(ValueError, match="prediction_type='epsilon'"):
        DiffusionHeadConfig(prediction_type=prediction_type)


@pytest.mark.parametrize(
    ("configured_scale", "expected_scale"),
    [
        (None, 999.0),
        ("0.375", 0.375),
    ],
    ids=["legacy-compatible-default", "explicit-config-value"],
)
def test_noise_conditioning_scale_reaches_score_model(configured_scale, expected_scale):
    """Default and parsed scales must reach the time embedding exactly once."""
    diffusion_config = {
        "sde": "vpsde",
        "base_channels": 8,
        "channel_multipliers": [1],
        "num_res_blocks": 1,
        "time_embed_dim": 16,
        "projected_cond_channels": 4,
    }
    if configured_scale is not None:
        diffusion_config["noise_conditioning_scale"] = configured_scale
    config = SimpleNamespace(model=SimpleNamespace(diffusion=diffusion_config))
    head_config = DiffusionHeadConfig.from_config(config)
    assert head_config.noise_conditioning_scale == pytest.approx(expected_scale)

    class LabelSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.labels = None

        def forward(self, x, cond, labels):
            self.labels = labels.detach().clone()
            return torch.zeros_like(x)

    head = DiffusionHead(cond_channels=2, output_channels=1, head_config=head_config)
    spy = LabelSpy()
    head.score_model = spy
    x = torch.zeros(2, 1, 2, 2)
    cond = torch.zeros(2, 2, 2, 2)
    t = torch.tensor([0.125, 0.75])
    with torch.no_grad():
        score = head.score_fn(x, cond, t)

    assert torch.isfinite(score).all()
    assert spy.labels is not None
    assert torch.equal(spy.labels, t * expected_scale)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["diffusion", "diffusion_head", "sde", "score", "score_sde"])
def test_resolve_head_type_diffusion_aliases(value):
    config = SimpleNamespace(model=SimpleNamespace(head_type=value))
    assert resolve_head_type(config) == "diffusion"


def test_resolve_head_type_decoder_type_alias_and_default():
    assert resolve_head_type(SimpleNamespace(model=SimpleNamespace(decoder_type="diffusion"))) == "diffusion"
    assert resolve_head_type(SimpleNamespace(model=SimpleNamespace())) == "deterministic"
    assert resolve_head_type(SimpleNamespace()) == "deterministic"
    assert resolve_head_type(None) == "deterministic"


def test_build_loss_fn_returns_diffusion_passthrough():
    config = SimpleNamespace(model=SimpleNamespace(head_type="diffusion"))
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert isinstance(loss_fn, DiffusionLossPassthrough)

    scalar = torch.tensor(1.25, requires_grad=True)
    assert loss_fn(scalar, {"y": torch.zeros(1, 2, 4, 4)}) is scalar
    assert "type" in loss_fn.describe()


def test_build_loss_fn_routes_residual_diffusion_to_joint_weighted_loss():
    config = SimpleNamespace(
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion={"residual_diffusion": True},
        ),
        loss={
            "base": "rmse",
            "deterministic_weight": 0.5,
            "diffusion": {"weight": 3.0},
        },
    )
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert isinstance(loss_fn, JointResidualDiffusionLoss)

    baseline = torch.ones(1, 2, 2, 2, requires_grad=True)
    diffusion = torch.tensor(2.0, requires_grad=True)
    batch = {"y": torch.zeros_like(baseline)}
    total = loss_fn(
        {
            "baseline_prediction": baseline,
            "diffusion_loss": diffusion,
            "diffusion_loss_terms": {
                "score_matching": 1.5,
                "clean_x0_reconstruction": 0.5,
                "total": 2.0,
            },
        },
        batch,
    )

    # RMSE(ones, zeros) = 1, so 0.5 * 1 + 3 * 2 = 6.5.
    assert torch.allclose(total, torch.tensor(6.5))
    total.backward()
    assert baseline.grad is not None
    expected_baseline_grad = torch.full_like(
        baseline, 0.5 / baseline.numel()
    )
    assert torch.allclose(baseline.grad, expected_baseline_grad)
    assert diffusion.grad is not None
    assert torch.allclose(diffusion.grad, torch.tensor(3.0))

    terms = loss_fn.get_last_terms()
    assert terms["deterministic.base.rmse"] == pytest.approx(1.0)
    assert terms["diffusion.score_matching"] == pytest.approx(1.5)
    assert terms["diffusion.clean_x0_reconstruction"] == pytest.approx(0.5)
    assert terms["diffusion.total"] == pytest.approx(2.0)
    assert terms["joint.total"] == pytest.approx(6.5)
    description = loss_fn.describe()
    assert description["type"] == "joint_residual_diffusion"
    assert description["deterministic_weight"] == pytest.approx(0.5)
    assert description["diffusion_weight"] == pytest.approx(3.0)


def test_joint_residual_diffusion_loss_requires_both_forward_outputs():
    config = SimpleNamespace(
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion=SimpleNamespace(residual_diffusion=True),
        )
    )
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    batch = {"y": torch.zeros(1, 2, 2, 2)}

    with pytest.raises(TypeError, match="baseline_prediction"):
        loss_fn({"diffusion_loss": torch.tensor(1.0)}, batch)
    with pytest.raises(TypeError, match="diffusion_loss"):
        loss_fn({"baseline_prediction": batch["y"]}, batch)


# ---------------------------------------------------------------------------
# Standardized encode/decode and precipitation non-negativity
# ---------------------------------------------------------------------------
class _DecoderDouble(ClimateECCCFinetuneWrapper):
    """Minimal object exposing the buffers the encode/decode helpers need."""

    def __init__(self, sigma, mu, codes, nonneg):
        torch.nn.Module.__init__(self)
        self.output_scalers_sigma = sigma
        self.output_scalers_mu = mu
        self.register_buffer("predictand_scaling_method_codes", codes)
        self.register_buffer("predictand_nonneg_enabled_mask", nonneg)

    def _resolve_output_scalers(self, x, scaler_offset=None):
        return self.output_scalers_mu.to(x.dtype), self.output_scalers_sigma.to(x.dtype)

    def _decode_outputs(self, raw, scaler_offset=None, wet_logits=None):
        del wet_logits
        return self._decode_targets_std(raw, scaler_offset=scaler_offset), raw


def _make_decoder(codes, nonneg, sigma, mu):
    return _DecoderDouble(
        sigma=torch.tensor(sigma).view(1, -1, 1, 1),
        mu=torch.tensor(mu).view(1, -1, 1, 1),
        codes=torch.tensor(codes, dtype=torch.int64),
        nonneg=torch.tensor(nonneg, dtype=torch.bool),
    )


def test_encode_decode_roundtrip_zscore_and_divide_only():
    # channel 0 = pr (divide_only, code 1), channel 1 = tasmax (zscore, code 0)
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    y = torch.stack(
        [torch.full((1, 4, 4), 3.5), torch.full((1, 4, 4), 12.0)], dim=1
    )  # [B=1, C=2, H, W]
    std = dec._encode_targets_std(y)
    back = dec._decode_targets_std(std)
    assert torch.allclose(back, y, atol=1e-5)


def test_encode_decode_roundtrip_log1p():
    dec = _make_decoder(codes=[2], nonneg=[True], sigma=[1.5], mu=[0.3])
    y = torch.full((1, 1, 4, 4), 4.0)
    std = dec._encode_targets_std(y)
    back = dec._decode_targets_std(std)
    assert torch.allclose(back, y, atol=1e-5)


def test_decode_clamps_precipitation_non_negativity():
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    # negative standardized values -> pr would decode negative and must be clamped
    std = torch.stack(
        [torch.full((1, 4, 4), -1.0), torch.full((1, 4, 4), -1.0)], dim=1
    )
    decoded = dec._decode_targets_std(std)
    assert float(decoded[:, 0, ...].min()) >= 0.0  # pr clamped
    assert torch.allclose(decoded[:, 1, ...], torch.full((1, 4, 4), 3.0))  # tasmax untouched


# ---------------------------------------------------------------------------
# Shared _diffusion_forward path (the exact glue both CORDEX models call).
#
# Training/validation: the trainer calls ``model(batch)`` -> scalar loss.
# Inference: the blending helper calls
# ``model(batch, return_pre_inverse=True, return_raw_output=True)`` -> 3-tuple.
# ---------------------------------------------------------------------------
def _diffusion_decoder_double():
    dec = _make_decoder(codes=[1, 0], nonneg=[True, False], sigma=[10.0, 2.0], mu=[0.0, 5.0])
    dec.diffusion_head = DiffusionHead(
        cond_channels=4, output_channels=2, head_config=_small_head_config()
    )
    dec._last_precip_hurdle_aux = "stale"  # must be reset to None by the forward
    return dec


def test_diffusion_forward_training_branch_returns_scalar_loss():
    torch.manual_seed(0)
    dec = _diffusion_decoder_double()
    cond = torch.randn(2, 4, 6, 6)
    batch = {"y": torch.rand(2, 2, 12, 12)}  # physical pr/tasmax targets
    out = dec._diffusion_forward(cond, batch)  # no flags -> training loss
    assert out.ndim == 0 and torch.isfinite(out)
    assert dec._last_precip_hurdle_aux is None  # diffusion mode never emits hurdle aux


def test_diffusion_forward_inference_branch_returns_decoded_tuple():
    torch.manual_seed(0)
    dec = _diffusion_decoder_double()
    cond = torch.randn(2, 4, 5, 5)
    batch = {"y": torch.rand(2, 2, 16, 16)}
    with torch.no_grad():
        result = dec._diffusion_forward(
            cond, batch, return_pre_inverse=True, return_raw_output=True
        )
    assert isinstance(result, tuple) and len(result) == 3
    x_out, x_pre_inverse, raw_out = result
    # matches the target grid the NetCDF writer expects: [B, 2, H_target, W_target]
    assert x_out.shape == (2, 2, 16, 16)
    assert x_pre_inverse.shape[-2:] == (16, 16)
    assert raw_out.shape[-2:] == (16, 16)
    assert torch.isfinite(x_out).all()
    # precipitation channel remains non-negative after decoding
    assert float(x_out[:, 0, ...].min()) >= 0.0


def test_residual_inference_publishes_direct_baseline_without_subtraction():
    dec = _make_decoder(
        codes=[1, 0],
        nonneg=[True, False],
        sigma=[10.0, 2.0],
        mu=[0.0, 5.0],
    )
    dec.output_conv_block = torch.nn.Conv2d(4, 2, kernel_size=1)
    with torch.no_grad():
        dec.output_conv_block.weight.zero_()
        dec.output_conv_block.bias.copy_(torch.tensor([0.2, 1.5]))
    dec.diffusion_head = DiffusionHead(
        cond_channels=4,
        output_channels=2,
        head_config=_small_head_config(
            residual_diffusion=True,
            residual_application_scale=0.0,
        ),
    )
    cond = torch.randn(2, 4, 5, 5)
    batch = {"y": torch.rand(2, 2, 16, 16)}

    with torch.no_grad():
        physical, standardized, raw = dec._diffusion_forward(
            cond,
            batch,
            return_pre_inverse=True,
            return_raw_output=True,
        )
    direct_baseline = dec.get_last_diffusion_baseline()

    assert direct_baseline is not None
    baseline_physical, baseline_standardized = direct_baseline
    assert torch.equal(standardized, baseline_standardized)
    assert torch.equal(physical, baseline_physical)
    assert torch.count_nonzero(raw) == 0
    assert baseline_standardized.shape == (2, 2, 16, 16)

