"""Tests for the Prithvi-native paired-state temporal pathway.

Focused on the properties that could silently be wrong and that the existing
suite does not already cover:

* legacy behaviour when the pathway is disabled;
* the baseline-preserving patch-embedding expansion, and that a zero weight is
  not a zero gradient;
* correct paired dates and correct native time metadata;
* no future information in either the main or the auxiliary branch;
* masking with no reconstruction shortcut;
* chunked versus full-sequence inference with equivalent history;
* both real case configurations.

Deliberately not re-testing the engineering properties the bottleneck backends
already prove (state isolation, TBPTT, param-group classification) -- those are
backend-independent and covered by ``test_temporal_model.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from tests.temporal_fixtures import (
    TINY_H,
    TINY_N_DYNAMIC,
    TINY_W,
    build_tiny_model,
    make_sequence_batch,
    temporal_config,
    tiny_config_dict,
)

from granitewxc.temporal.config import TemporalConfigError, parse_temporal_config
from granitewxc.temporal.model import (
    TemporalSequenceModel,
    attach_native_pair_adapter,
    attach_temporal_adapter,
)
from granitewxc.temporal.native_pair import (
    HOURS_PER_DAY,
    MissingHistoryError,
    NativePairAdapter,
    NativeTimeConditioning,
    apply_input_mask,
    build_pair_input,
    compute_pretext_losses,
    expand_patch_embed_weight,
    mask_units_to_pixels,
    pair_time_scalars,
    sample_mask_units,
)

REPO = Path(__file__).resolve().parent.parent
CORDEX = REPO / "examples/CORDEX_ML"
NARR = REPO / "examples/NARR_PRISM"

SA_NATIVE_PAIR = CORDEX / "SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair.yaml"
SA_NATIVE_PAIR_PRETEXT = (
    CORDEX / "SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair_pretext.yaml"
)


def _pair_config(tmp_path: Path, **native_pair):
    """A valid native-pair TemporalConfig on the tiny geometry."""
    np_block = {"history_offsets": [1]}
    np_block.update(native_pair)
    return temporal_config(
        backend="native_pair",
        context_length=5,
        warmup_length=max(2, max(np_block["history_offsets"]) + 1),
        output_length=3,
        native_pair=np_block,
    )


def _build_pair_model(tmp_path: Path, cfg, *, use_static: bool = True, seed: int = 0):
    """Tiny model with ``n_input_timestamps`` matching the config, plus adapter."""
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.utils.config import ExperimentConfig

    raw = tiny_config_dict(tmp_path, use_static=use_static)
    raw["data"]["n_input_timestamps"] = cfg.native_pair.n_input_timestamps
    exp = ExperimentConfig.from_dict(raw)
    torch.manual_seed(seed)
    model = get_finetune_model_UNET(exp)
    model.eval()
    attach_native_pair_adapter(model, cfg)
    return model, exp


# ---------------------------------------------------------------------------
# 1. legacy behaviour when the pathway is disabled
# ---------------------------------------------------------------------------
def test_token_hook_is_identity_without_an_adapter(tmp_path):
    """No adapter attached: the token hook returns the very same object."""
    model, _ = build_tiny_model(tmp_path)
    tokens = torch.randn(2, 4, 4, model.embed_dim_backbone)
    assert model._apply_temporal_tokens(tokens) is tokens


def test_token_hook_is_identity_for_the_recurrent_backend(tmp_path):
    """The bottleneck backends have no ``apply_tokens``; the hook must no-op.

    This is what guarantees the ConvGRU/Mamba results already recorded in
    ``docs/temporal_model_results.md`` remain reproducible after this branch.
    """
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(backend="recurrent")
    attach_temporal_adapter(model, cfg, time_feature_dim=5)
    model._temporal_ctx = {"state": None, "time_features": None, "interval_ratio": None}
    tokens = torch.randn(2, 4, 4, model.embed_dim_backbone)
    assert model._apply_temporal_tokens(tokens) is tokens


def test_native_pair_leaves_the_bottleneck_untouched(tmp_path):
    """``injects_at_bottleneck`` False means the U-Net bottleneck is the legacy one."""
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    assert model.temporal_adapter.injects_at_bottleneck is False
    model._temporal_ctx = {"state": None}
    out = torch.randn(2, model.conv_after_backbone.out_channels, 4, 4)
    assert model._apply_temporal_latent(out) is out


# ---------------------------------------------------------------------------
# 2. checkpoint compatibility and baseline-preserving initialization
# ---------------------------------------------------------------------------
def test_expand_patch_embed_puts_pretrained_weights_on_the_current_date():
    w = torch.randn(8, 3, 2, 2)
    e = expand_patch_embed_weight(w, n_timestamps=2)
    assert e.shape == (8, 6, 2, 2)
    # Oldest first: history occupies the leading block and starts at zero.
    assert torch.equal(e[:, 3:], w)
    assert torch.count_nonzero(e[:, :3]) == 0


def test_expand_patch_embed_is_identity_for_one_timestamp():
    w = torch.randn(4, 5, 3, 3)
    assert expand_patch_embed_weight(w, n_timestamps=1) is w


def test_expanded_embedding_reproduces_the_frame_independent_features():
    """Baseline preservation, stated at the precision it actually holds.

    Convolution reduction order can change with channel count, dtype, kernel,
    and device. The tolerances below verify numerical agreement on this host;
    they do not claim universal bit-exactness.
    """
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        torch.manual_seed(0)
        w = torch.randn(16, 4, 2, 2, dtype=dtype)
        b = torch.randn(16, dtype=dtype)
        x_t = torch.randn(1, 4, 16, 16, dtype=dtype)
        x_h = torch.randn(1, 4, 16, 16, dtype=dtype) * 3.0 + 1.0
        e = expand_patch_embed_weight(w, n_timestamps=2)
        ref = torch.nn.functional.conv2d(x_t, w, b)
        got = torch.nn.functional.conv2d(torch.cat([x_h, x_t], dim=1), e, b)
        torch.testing.assert_close(ref, got, atol=tolerance, rtol=tolerance)


def test_zero_history_weights_still_receive_gradient():
    """A zero *weight* is not a zero *gradient*: dL/dW = dL/dout * x_history."""
    w = expand_patch_embed_weight(torch.randn(8, 3, 2, 2), n_timestamps=2)
    w = w.clone().requires_grad_(True)
    x = torch.cat([torch.randn(1, 3, 8, 8) + 2.0, torch.randn(1, 3, 8, 8)], dim=1)
    torch.nn.functional.conv2d(x, w).square().mean().backward()
    assert w.grad[:, :3].abs().sum() > 0, "history half must train from step 1"


def test_adapter_state_dict_lives_under_the_temporal_prefix(tmp_path):
    """Migration, freezing and lr_temporal all key off this prefix."""
    cfg = _pair_config(tmp_path, time_conditioning=True)
    model, _ = _build_pair_model(tmp_path, cfg)
    names = [n for n, _ in model.named_parameters() if n.startswith("temporal_adapter.")]
    assert names, "the adapter must contribute parameters under temporal_adapter.*"
    assert all(
        n.startswith("temporal_adapter.") or "temporal_adapter" not in n
        for n, _ in model.named_parameters()
    )


def test_attach_rejects_a_timestamp_mismatch(tmp_path):
    """``n_input_timestamps`` and ``history_offsets`` are not independent knobs."""
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.utils.config import ExperimentConfig

    cfg = _pair_config(tmp_path)  # needs 2 timestamps
    raw = tiny_config_dict(tmp_path)
    raw["data"]["n_input_timestamps"] = 1  # deliberately wrong
    model = get_finetune_model_UNET(ExperimentConfig.from_dict(raw))
    with pytest.raises(ValueError, match="n_input_timestamps"):
        attach_native_pair_adapter(model, cfg)


# ---------------------------------------------------------------------------
# 3. correct paired dates and time metadata
# ---------------------------------------------------------------------------
def test_build_pair_input_is_oldest_first():
    x = torch.arange(2 * 5 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 5, 3, 4, 4)
    got = build_pair_input(x, 3, history_offsets=[1])
    assert got.shape == (2, 6, 4, 4)
    assert torch.equal(got[:, :3], x[:, 2]), "leading block must be the older date"
    assert torch.equal(got[:, 3:], x[:, 3]), "trailing block must be date t"


def test_build_pair_input_multi_offset_ordering():
    x = torch.randn(1, 6, 2, 4, 4)
    got = build_pair_input(x, 4, history_offsets=[1, 2])
    assert torch.equal(got[:, 0:2], x[:, 2])  # t-2 oldest
    assert torch.equal(got[:, 2:4], x[:, 3])  # t-1
    assert torch.equal(got[:, 4:6], x[:, 4])  # t


def test_duplicate_current_control_uses_no_history():
    x = torch.randn(1, 5, 2, 4, 4)
    got = build_pair_input(x, 3, history_offsets=[1], history_mode="duplicate_current")
    assert torch.equal(got[:, 0:2], x[:, 3])
    assert torch.equal(got[:, 2:4], x[:, 3])


def test_missing_history_raises_rather_than_clamping():
    """Training must never silently cold-start a supervised frame."""
    x = torch.randn(1, 5, 2, 4, 4)
    with pytest.raises(MissingHistoryError):
        build_pair_input(x, 0, history_offsets=[1])


def test_history_present_in_run_but_absent_from_batch_is_a_distinct_error():
    """A chunking bug must not be mistaken for a legitimate run start."""
    x = torch.randn(1, 5, 2, 4, 4)
    # position_offset 10 => frame 0 is absolute 10, so absolute 9 exists.
    with pytest.raises(MissingHistoryError, match="not supplied in this"):
        build_pair_input(x, 0, history_offsets=[1], position_offset=10, allow_cold_start=True)


def test_cold_start_repeats_the_current_date_only_at_a_run_start():
    x = torch.randn(1, 5, 2, 4, 4)
    got = build_pair_input(x, 0, history_offsets=[1], position_offset=0, allow_cold_start=True)
    assert torch.equal(got[:, 0:2], x[:, 0])
    assert torch.equal(got[:, 2:4], x[:, 0])


def test_native_time_scalars_are_measured_not_assumed():
    """A step across a removed 29 February must be charged as two days."""
    ratio = torch.ones(1, 5)
    it, lt = pair_time_scalars(
        ratio, 3, history_offsets=[1], cadence_days=1.0, batch_size=1, device=torch.device("cpu")
    )
    assert torch.allclose(it, torch.tensor([HOURS_PER_DAY]))
    assert torch.allclose(lt, torch.zeros(1)), "downscaling lead time must be zero"

    ratio2 = torch.ones(1, 5)
    ratio2[:, 3] = 2.0  # the step into frame 3 spanned two days
    it2, _ = pair_time_scalars(
        ratio2, 3, history_offsets=[1], cadence_days=1.0, batch_size=1, device=torch.device("cpu")
    )
    assert torch.allclose(it2, torch.tensor([2 * HOURS_PER_DAY]))


def test_transition_pass_declares_a_positive_lead():
    _, lt = pair_time_scalars(
        None,
        3,
        history_offsets=[1],
        cadence_days=1.0,
        batch_size=2,
        device=torch.device("cpu"),
        lead_steps=1,
    )
    assert torch.allclose(lt, torch.full((2,), HOURS_PER_DAY))


def test_native_time_conditioning_matches_the_upstream_form():
    """cat(cos(it), cos(lt), sin(it), sin(lt)) in four equal blocks."""
    mod = NativeTimeConditioning(16, gate_init=1.0)
    it = torch.tensor([24.0])
    lt = torch.tensor([0.0])
    enc = mod.encoding(it, lt)
    assert enc.shape == (1, 1, 1, 16)
    q = 4
    expected_cos_it = torch.cos(mod.input_time_embedding(it.reshape(-1, 1, 1, 1)))
    assert torch.allclose(enc[..., :q], expected_cos_it)
    expected_sin_lt = torch.sin(mod.lead_time_embedding(lt.reshape(-1, 1, 1, 1)))
    assert torch.allclose(enc[..., 3 * q :], expected_sin_lt)


def test_time_conditioning_requires_real_metadata(tmp_path):
    """It must never silently fall back to a fabricated interval."""
    cfg = _pair_config(tmp_path, time_conditioning=True)
    adapter = NativePairAdapter(
        cfg=cfg,
        embed_dim_backbone=16,
        post_backbone_channels=8,
        predictor_channels_per_timestamp=4,
    )
    with pytest.raises(RuntimeError, match="native_input_time_hours"):
        adapter.apply_tokens(torch.randn(1, 2, 2, 16), {})


def test_zero_gate_is_exact_identity_and_default_gate_is_not():
    mod0 = NativeTimeConditioning(16, gate_init=0.0)
    tokens = torch.randn(2, 3, 3, 16)
    it, lt = torch.tensor([24.0, 24.0]), torch.zeros(2)
    assert torch.equal(mod0(tokens, it, lt), tokens)
    mod1 = NativeTimeConditioning(16, gate_init=1e-3)
    assert not torch.equal(mod1(tokens, it, lt), tokens)


# ---------------------------------------------------------------------------
# 4. no future information; history genuinely used
# ---------------------------------------------------------------------------
def test_causality_future_frames_do_not_change_earlier_outputs(tmp_path):
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    batch = make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC)
    with torch.no_grad():
        ref = runner(batch)
    perturbed = copy.deepcopy(batch)
    perturbed["x"][:, -1] += 100.0
    with torch.no_grad():
        got = runner(perturbed)
    assert torch.equal(ref.predictions[:, :-1], got.predictions[:, :-1])


def test_history_changes_the_prediction(tmp_path):
    """Sensitivity, which is necessary but not sufficient for skill."""
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    # Give the history channels a real weight; at init they are exactly zero.
    with torch.no_grad():
        model.embedding.proj.weight[:, :TINY_N_DYNAMIC].normal_(0.0, 0.05)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    batch = make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC)
    emit = runner.emitted_indices(5)
    with torch.no_grad():
        ref = runner(batch)
    perturbed = copy.deepcopy(batch)
    perturbed["x"][:, emit[0] - 1] += 50.0  # the history of the first emitted frame
    with torch.no_grad():
        got = runner(perturbed)
    assert not torch.equal(ref.predictions[:, 0], got.predictions[:, 0])


def test_duplicate_current_control_ignores_history(tmp_path):
    """The capacity control must be provably blind to the earlier date."""
    cfg = _pair_config(tmp_path, history_mode="duplicate_current")
    model, _ = _build_pair_model(tmp_path, cfg)
    with torch.no_grad():
        model.embedding.proj.weight[:, :TINY_N_DYNAMIC].normal_(0.0, 0.05)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    batch = make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC)
    emit = runner.emitted_indices(5)
    with torch.no_grad():
        ref = runner(batch)
    perturbed = copy.deepcopy(batch)
    perturbed["x"][:, emit[0] - 1] += 50.0
    with torch.no_grad():
        got = runner(perturbed)
    assert torch.equal(ref.predictions, got.predictions)


def test_capacity_control_has_identical_parameter_count(tmp_path):
    real, _ = _build_pair_model(tmp_path, _pair_config(tmp_path), seed=1)
    ctrl, _ = _build_pair_model(
        tmp_path, _pair_config(tmp_path, history_mode="duplicate_current"), seed=1
    )
    assert sum(p.numel() for p in real.parameters()) == sum(
        p.numel() for p in ctrl.parameters()
    )


def test_stateless_pathway_only_runs_the_frames_it_emits(tmp_path):
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    with torch.no_grad():
        out = runner(make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC))
    assert out.backbone_evaluations == len(out.emitted_indices) == 3
    assert out.final_state is None


def test_every_temporal_parameter_trains_except_the_documented_case(tmp_path):
    """Exactly the structurally-dead set is dead -- no more, no less.

    With a zero main lead time the lead-time embedding's *weight* cannot receive
    gradient (``dL/dW = dL/d(lt) * lead_time = 0``); its bias can. Asserting
    against the declared set catches both a new dead parameter and a stale
    exemption.
    """
    cfg = _pair_config(tmp_path, time_conditioning=True)
    model, _ = _build_pair_model(tmp_path, cfg)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter)
    runner.train()
    out = runner(make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC))
    out.predictions.square().mean().backward()
    dead = sorted(
        n[len("temporal_adapter.") :]
        for n, p in model.named_parameters()
        if n.startswith("temporal_adapter.")
        and (p.grad is None or torch.count_nonzero(p.grad) == 0)
    )
    assert dead == sorted(model.temporal_adapter.structurally_dead_parameters())


def test_the_transition_objective_revives_the_lead_time_weight(tmp_path):
    """A positive auxiliary lead makes the lead-time embedding trainable."""
    cfg = _pair_config(
        tmp_path,
        time_conditioning=True,
        pretext={"transition": {"enabled": True, "weight": 1.0, "lead_steps": 1}},
    )
    model, _ = _build_pair_model(tmp_path, cfg)
    assert model.temporal_adapter.structurally_dead_parameters() == []
    model.train()
    terms, _ = compute_pretext_losses(
        model,
        model.temporal_adapter,
        make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC),
        4,
        cfg,
    )
    terms.total.backward()
    grad = dict(model.named_parameters())[
        "temporal_adapter.time_conditioning.lead_time_embedding.weight"
    ].grad
    assert grad is not None and torch.count_nonzero(grad) > 0


# ---------------------------------------------------------------------------
# 5. masking without a reconstruction shortcut
# ---------------------------------------------------------------------------
def test_mask_unit_count_is_fixed_per_sample():
    mask = sample_mask_units(4, (4, 4), 0.5, device=torch.device("cpu"))
    assert mask.shape == (4, 4, 4)
    assert torch.all(mask.flatten(1).sum(dim=1) == 8)


def test_mask_ratio_zero_masks_nothing():
    assert torch.count_nonzero(sample_mask_units(2, (4, 4), 0.0)) == 0


def test_pixel_mask_covers_whole_mask_units():
    unit = torch.zeros(1, 2, 2, dtype=torch.bool)
    unit[0, 0, 0] = True
    px = mask_units_to_pixels(unit, block=(8, 8), height=16, width=16)
    assert px.shape == (1, 1, 16, 16)
    assert bool(px[0, 0, :8, :8].all())
    assert not bool(px[0, 0, 8:, :].any())


def test_mask_is_applied_to_every_timestamp():
    """The core leakage guard: history channels must be masked too."""
    x = torch.randn(1, 8, 8, 8)  # 2 timestamps x 4 channels
    fill = torch.zeros(1, 4, 1, 1)
    px = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    px[..., :4, :4] = True
    out = apply_input_mask(x, px, fill=fill, n_timestamps=2)
    assert torch.count_nonzero(out[:, :, :4, :4]) == 0, "all 8 channels must be masked"
    assert torch.equal(out[:, :, 4:, :], x[:, :, 4:, :])


def test_masked_content_cannot_influence_the_masked_forward():
    """Changing the truth inside the masked region must not change the input."""
    fill = torch.zeros(1, 4, 1, 1)
    px = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    px[..., :4, :4] = True
    a = torch.randn(1, 8, 8, 8)
    b = a.clone()
    b[:, :, :4, :4] += 1000.0  # only inside the mask
    assert torch.equal(
        apply_input_mask(a, px, fill=fill, n_timestamps=2),
        apply_input_mask(b, px, fill=fill, n_timestamps=2),
    )


def test_masked_field_loss_scores_only_masked_pixels():
    from granitewxc.temporal.native_pair import masked_field_loss

    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[..., :2, :] = 5.0  # error only in the unmasked half
    px = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    px[..., 2:, :] = True
    assert masked_field_loss(pred, target, px).item() == pytest.approx(0.0)


def test_pretext_losses_reach_the_shared_trunk(tmp_path):
    """Auxiliary gradients reach shared adaptation through the frozen backbone."""
    cfg = _pair_config(
        tmp_path,
        pretext={
            "masked_reconstruction": {"enabled": True, "weight": 1.0, "mask_ratio": 0.5},
            "transition": {"enabled": True, "weight": 1.0, "lead_steps": 1},
        },
    )
    model, _ = _build_pair_model(tmp_path, cfg)
    from granitewxc.temporal.model import apply_freeze_policy
    apply_freeze_policy(model, cfg, epoch=0)
    model.train()
    batch = make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC)
    terms, n_forward = compute_pretext_losses(
        model, model.temporal_adapter, batch, 4, cfg
    )
    assert set(terms) == {"pretext_masked_reconstruction", "pretext_transition"}
    assert n_forward == 2
    model.zero_grad()
    terms.total.backward()
    for shared in ("temporal_adapter.history_projection.weight", "conv_after_backbone.weight",
                   "temporal_adapter.time_conditioning.lead_time_embedding.weight"):
        grad = dict(model.named_parameters())[shared].grad
        assert grad is not None and torch.count_nonzero(grad) > 0, f"{shared} got no gradient"
    for name, param in model.named_parameters():
        if name.startswith(("backbone.", "embedding.", "conv_before_backbone.")):
            assert not param.requires_grad and param.grad is None, name
    for head in (model.temporal_adapter.recon_head, model.temporal_adapter.transition_head):
        assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in head.parameters())


def test_pretext_is_absent_when_disabled(tmp_path):
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    terms, n_forward = compute_pretext_losses(model, model.temporal_adapter,
                                              make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC),
                                              4, cfg)
    assert dict(terms) == {} and n_forward == 0 and terms.total is None


def test_transition_pass_does_not_see_its_target(tmp_path):
    """The verification state must not be an input to the forecasting pass.

    Perturbing ``x[t]`` alone must leave the transition *input* untouched; only
    its target changes. Checked structurally on the tensor the pass is given.
    """
    x = torch.randn(1, 6, 2, 4, 4)
    lead = 1
    t = 4
    pair = build_pair_input(x, t - lead, history_offsets=[1])
    perturbed = x.clone()
    perturbed[:, t] += 1000.0
    pair2 = build_pair_input(perturbed, t - lead, history_offsets=[1])
    assert torch.equal(pair, pair2), "x[t] must not enter the transition pass"


def test_transition_pretext_prediction_is_independent_of_the_target(tmp_path):
    """Numerical counterpart of the structural check above."""
    cfg = _pair_config(
        tmp_path,
        pretext={"transition": {"enabled": True, "weight": 1.0, "lead_steps": 1}},
    )
    model, _ = _build_pair_model(tmp_path, cfg)
    model.eval()
    batch = make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC)
    pred_a = {}
    with torch.no_grad():
        a, _ = compute_pretext_losses(model, model.temporal_adapter, batch, 4, cfg,
                                     prediction_sink=pred_a)
    poisoned = copy.deepcopy(batch)
    poisoned["x"][:, 4] += 1000.0
    pred_b = {}
    with torch.no_grad():
        b, _ = compute_pretext_losses(model, model.temporal_adapter, poisoned, 4, cfg,
                                     prediction_sink=pred_b)
    assert torch.equal(pred_a["pretext_transition"], pred_b["pretext_transition"])
    # Predictions are compared directly; the loss changes only via its target.
    assert not torch.equal(a["pretext_transition"], b["pretext_transition"])
    assert b["pretext_transition"] > a["pretext_transition"]


def test_pretext_forward_restores_feature_capture(tmp_path):
    cfg = _pair_config(
        tmp_path,
        pretext={"masked_reconstruction": {"enabled": True, "weight": 1.0, "mask_ratio": 0.25}},
    )
    model, _ = _build_pair_model(tmp_path, cfg)
    before = getattr(model, "_capture_features", ())
    compute_pretext_losses(
        model, model.temporal_adapter, make_sequence_batch(frames=5, n_dynamic=TINY_N_DYNAMIC), 4, cfg
    )
    assert getattr(model, "_capture_features", ()) == before
    assert model._temporal_ctx is None


# ---------------------------------------------------------------------------
# 6. chunked versus full-sequence inference with equivalent history
# ---------------------------------------------------------------------------
def test_chunked_equals_single_pass_when_history_is_supplied(tmp_path):
    """Chunking is only equivalent if the earlier *dates* travel with the chunk.

    There is no hidden state to carry here, so the analogue of carrying state is
    extending each chunk backwards by the deepest history offset. With that done,
    the agreement is bit-exact.
    """
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    with torch.no_grad():
        model.embedding.proj.weight[:, :TINY_N_DYNAMIC].normal_(0.0, 0.05)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()

    frames = 9
    batch = make_sequence_batch(batch=1, frames=frames, n_dynamic=TINY_N_DYNAMIC)
    offset = cfg.native_pair.max_offset

    def slice_batch(lo, hi):
        out = {}
        for k, v in batch.items():
            out[k] = v[:, lo:hi] if torch.is_tensor(v) and v.dim() >= 2 and k in {
                "x", "y", "time_features", "interval_ratio", "reset", "__target_valid_mask"
            } else v
        out["__position_offset"] = lo
        return out

    # Single pass over frames 1..8 (frame 0 is history for frame 1).
    single = runner(slice_batch(0, frames), emit_indices=tuple(range(offset, frames)))

    pieces = []
    for lo in range(offset, frames, 3):
        hi = min(lo + 3, frames)
        chunk = slice_batch(lo - offset, hi)
        out = runner(chunk, emit_indices=tuple(range(offset, hi - (lo - offset))))
        pieces.append(out.predictions)
    chunked = torch.cat(pieces, dim=1)

    assert chunked.shape == single.predictions.shape
    assert torch.equal(chunked, single.predictions), "chunked must equal single pass exactly"


def test_inference_rejects_a_chunk_shorter_than_the_history(tmp_path):
    from granitewxc.temporal.config import parse_temporal_config

    with pytest.raises(TemporalConfigError):
        parse_temporal_config(
            {
                "enabled": True,
                "backend": "native_pair",
                "context_length": 5,
                "warmup_length": 4,
                "output_length": 1,
                "native_pair": {"history_offsets": [5]},
                "init_from_spatial_checkpoint": "x.ckpt",
            }
        )


# ---------------------------------------------------------------------------
# 7. both real case configurations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [SA_NATIVE_PAIR, SA_NATIVE_PAIR_PRETEXT],
    ids=["sa_native_pair", "sa_native_pair_pretext"],
)
def test_sa_configs_parse_and_are_self_consistent(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = parse_temporal_config(raw["temporal"])
    assert cfg is not None and cfg.backend == "native_pair"
    assert cfg.mode == "downscaling" and cfg.lead_time_days == 0.0
    assert raw["data"]["n_input_timestamps"] == cfg.native_pair.n_input_timestamps
    assert cfg.warmup_length >= cfg.native_pair.max_offset
    assert cfg.native_pair.history_mode == "real"


def test_sa_pretext_config_enables_both_objectives():
    raw = yaml.safe_load(SA_NATIVE_PAIR_PRETEXT.read_text(encoding="utf-8"))
    p = parse_temporal_config(raw["temporal"]).native_pair.pretext
    assert p.masked_reconstruction_enabled and p.masked_reconstruction_weight > 0
    assert p.transition_enabled and p.transition_weight > 0
    # The window must be long enough for x[t-2], x[t-1] -> x[t].
    cfg = parse_temporal_config(raw["temporal"])
    assert cfg.warmup_length >= cfg.native_pair.max_offset + p.transition_lead_steps


def test_sa_plain_config_disables_pretext():
    """The architectural change must be measurable on its own."""
    raw = yaml.safe_load(SA_NATIVE_PAIR.read_text(encoding="utf-8"))
    p = parse_temporal_config(raw["temporal"]).native_pair.pretext
    assert not p.any_enabled


@pytest.mark.parametrize(
    "name",
    [
        "NARR_PRISM_subdomain_temporal_prithvi_native_pair.yaml",
        "NARR_PRISM_subdomain_temporal_prithvi_native_pair_pretext.yaml",
    ],
)
def test_narr_prism_configs_parse(name):
    path = NARR / name
    assert path.is_file(), f"Required NARR configuration is missing: {path}"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = parse_temporal_config(raw["temporal"])
    assert cfg.backend == "native_pair"
    assert cfg.mode == "downscaling" and cfg.lead_time_days == 0.0
    assert raw["data"]["n_input_timestamps"] == cfg.native_pair.n_input_timestamps
    assert cfg.warmup_length >= cfg.native_pair.max_offset
    # num_static_channels 0 is a meaningful value for this case, not a missing one.
    assert int(raw["model"]["num_static_channels"]) == 0
    assert list(raw["data"]["output_vars"]) == ["ppt", "tmax", "tmin"]


def test_sa_native_pair_differs_from_recurrent_only_where_intended():
    """Guards the comparison: the objective and the data must not drift."""
    ref = yaml.safe_load(
        (CORDEX / "SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml").read_text(
            encoding="utf-8"
        )
    )
    cur = yaml.safe_load(SA_NATIVE_PAIR.read_text(encoding="utf-8"))
    assert cur["temporal"]["losses"] == ref["temporal"]["losses"]
    assert cur["loss"] == ref["loss"]
    assert cur["model"] == ref["model"]
    assert cur["predictands"] == ref["predictands"]
    assert cur["temporal_splits"] == ref["temporal_splits"]
    assert cur["temporal"]["seed"] == ref["temporal"]["seed"]
    assert cur["temporal"]["freeze"] == ref["temporal"]["freeze"]
    assert cur["temporal"]["evaluation"] == ref["temporal"]["evaluation"]
    for key in ("context_length", "warmup_length", "output_length", "sequence_stride"):
        assert cur["temporal"][key] == ref["temporal"][key], key
    ref_data = {k: v for k, v in ref["data"].items() if k != "n_input_timestamps"}
    cur_data = {k: v for k, v in cur["data"].items() if k != "n_input_timestamps"}
    assert cur_data == ref_data


@pytest.mark.parametrize("use_static", [True, False])
def test_initialization_matches_independently_loaded_one_timestamp_model(tmp_path, use_static):
    from granitewxc.temporal.checkpoint import initialize_from_spatial_checkpoint
    from tests.temporal_fixtures import frame_by_frame_reference

    original, exp = build_tiny_model(tmp_path, use_static=use_static, seed=17)
    checkpoint = tmp_path / "original_phase1.pt"
    torch.save({"model": original.state_dict()}, checkpoint)
    independent, _ = build_tiny_model(tmp_path, use_static=use_static, seed=99)
    independent.load_state_dict(torch.load(checkpoint, weights_only=True)["model"], strict=True)
    cfg = _pair_config(tmp_path, time_conditioning_gate_init=0.0)
    paired, _ = _build_pair_model(tmp_path, cfg, use_static=use_static, seed=100)
    initialize_from_spatial_checkpoint(
        paired, checkpoint,
        n_input_timestamps=cfg.native_pair.n_input_timestamps,
    )
    batch = make_sequence_batch(use_static=use_static)
    runner = TemporalSequenceModel(paired, cfg, adapter=paired.temporal_adapter).eval()
    with torch.no_grad():
        expected = frame_by_frame_reference(independent.eval(), batch)[:, runner.emitted_indices(5)]
        actual = runner(batch).predictions
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-5)
    assert torch.count_nonzero(paired.temporal_adapter.history_projection.weight) == 0
    assert torch.count_nonzero(paired.embedding.proj.weight[:, :TINY_N_DYNAMIC]) == 0


def test_temporal_inference_needs_geometry_but_no_verification_values(tmp_path):
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path, cfg)
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    batch = make_sequence_batch()
    without_targets = {k: v for k, v in batch.items() if k not in {"y", "__target_valid_mask"}}
    without_targets["__output_shape"] = tuple(batch["y"].shape[-2:])
    poisoned = dict(batch, y=torch.full_like(batch["y"], float("nan")))
    with torch.no_grad():
        expected = runner(batch).predictions
        target_free = runner(without_targets)
        assert torch.equal(expected, runner(poisoned).predictions)
    assert target_free.target_frames is None
    assert torch.equal(expected, target_free.predictions)


def test_reconstruction_prediction_is_independent_of_hidden_target_values(tmp_path):
    cfg = _pair_config(tmp_path, pretext={"masked_reconstruction": {
        "enabled": True, "weight": 0.3, "mask_ratio": 0.5}})
    model, _ = _build_pair_model(tmp_path, cfg)
    batch = make_sequence_batch()
    units = sample_mask_units(2, (4, 4), 0.5, generator=torch.Generator().manual_seed(19))
    mask = mask_units_to_pixels(units, block=(8, 8), height=32, width=32)
    poisoned = copy.deepcopy(batch)
    poisoned["x"][:, 4] += mask * 1000
    before, after = {}, {}
    with torch.no_grad():
        a, _ = compute_pretext_losses(model, model.temporal_adapter, batch, 4, cfg,
            generator=torch.Generator().manual_seed(19), prediction_sink=before)
        b, _ = compute_pretext_losses(model, model.temporal_adapter, poisoned, 4, cfg,
            generator=torch.Generator().manual_seed(19), prediction_sink=after)
    assert torch.equal(before["pretext_masked_reconstruction"], after["pretext_masked_reconstruction"])
    assert b.total > a.total


def test_narr_auxiliary_targets_are_weather_only_and_observation_masked(tmp_path):
    from granitewxc.temporal.native_pair import masked_field_loss
    raw = yaml.safe_load((NARR / "NARR_PRISM_subdomain_temporal_prithvi_native_pair_pretext.yaml").read_text())
    names = raw["data"]["input_vars"]
    assert len(names) == 32
    cfg = _pair_config(tmp_path, pretext={"transition": {"enabled": True, "weight": 1.0}})
    adapter = NativePairAdapter(cfg=cfg, embed_dim_backbone=16,
        post_backbone_channels=8, predictor_channels_per_timestamp=32,
        predictor_channel_names=names)
    assert adapter.atmospheric_channel_indices == tuple(range(15))
    assert adapter.atmospheric_mask_indices == tuple(range(16, 31))
    assert adapter.transition_head.body[-1].out_channels == 15
    x = torch.ones(1, 32, 4, 4)
    x[:, 16, :2] = 0
    target, valid = adapter.atmospheric_target(x, torch.zeros(1, 32, 1, 1),
                                              torch.ones(1, 32, 1, 1), 0.0)
    prediction = target.clone()
    prediction[:, 0, :2] = 1000
    assert masked_field_loss(prediction, target, torch.ones_like(valid), valid) == 0
    prediction[:, 0, 2:] += 2
    assert masked_field_loss(prediction, target, torch.ones_like(valid), valid) > 0
    changed = x.clone()
    changed[:, 15] = 5000  # elevation is an input, never an auxiliary target
    changed[:, 31] = 0  # elevation validity is not atmospheric validity
    changed_target, changed_valid = adapter.atmospheric_target(changed,
        torch.zeros(1, 32, 1, 1), torch.ones(1, 32, 1, 1), 0.0)
    assert torch.equal(target, changed_target) and torch.equal(valid, changed_valid)


def test_auxiliary_loss_excludes_nonfinite_targets_with_finite_gradients():
    from granitewxc.temporal.native_pair import masked_field_loss
    prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 0, 0, 0] = float("nan")
    loss = masked_field_loss(prediction, target, torch.ones_like(target, dtype=torch.bool))
    assert loss.item() == 1
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0, 0] == 0
