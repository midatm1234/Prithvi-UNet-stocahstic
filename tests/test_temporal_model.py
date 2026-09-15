"""Engineering checks for the temporal extension.

These are the checks listed in section 9 of the work plan, each written so that
it fails for the specific reason it exists rather than merely exercising code:

* legacy behaviour is preserved bit-for-bit with temporal modelling disabled,
* the adapter is near-identity at init but every temporal parameter trains,
* output ``t`` cannot depend on inputs after ``t``,
* identical current frame + different history => different output (which a
  date-embedding-only model cannot do),
* hidden state does not leak between unrelated sequences,
* full-sequence and chunked inference agree exactly.
"""

from __future__ import annotations

import copy

import pytest
import torch

from tests.temporal_fixtures import (
    TINY_H,
    TINY_W,
    build_tiny_model,
    frame_by_frame_reference,
    make_sequence_batch,
    temporal_config,
)

from granitewxc.temporal.model import (
    TemporalSequenceModel,
    apply_freeze_policy,
    attach_temporal_adapter,
    build_param_groups,
    classify_parameter,
    detach_temporal_adapter,
    temporal_parameter_names,
)

BACKENDS = [
    pytest.param({"backend": "recurrent"}, id="convgru"),
    pytest.param(
        {"backend": "recurrent", "recurrent": {"cell": "convlstm"}}, id="convlstm"
    ),
    pytest.param(
        {
            "backend": "mamba",
            "mamba": {"implementation": "reference", "n_layers": 2, "headdim": 8, "d_state": 8},
        },
        id="mamba",
    ),
]


def _runner(model, cfg, *, seed: int = 1):
    torch.manual_seed(seed)
    adapter = attach_temporal_adapter(model, cfg, time_feature_dim=5)
    return TemporalSequenceModel(model, cfg, adapter=adapter), adapter


# ---------------------------------------------------------------------------
# legacy preservation
# ---------------------------------------------------------------------------
def test_legacy_path_untouched_without_adapter(tmp_path):
    """With no adapter attached the hook must be a pure identity."""
    model, _ = build_tiny_model(tmp_path)
    assert model.temporal_adapter is None
    probe = torch.randn(2, 32, 4, 4)
    assert model._apply_temporal_latent(probe) is probe  # same object, no arithmetic

    batch = make_sequence_batch(frames=3, height=TINY_H, width=TINY_W)
    first = frame_by_frame_reference(model, batch)
    second = frame_by_frame_reference(model, batch)
    assert torch.equal(first, second)


def test_detach_restores_legacy(tmp_path):
    model, _ = build_tiny_model(tmp_path)
    batch = make_sequence_batch(frames=3)
    before = frame_by_frame_reference(model, batch)

    cfg = temporal_config()
    _runner(model, cfg)
    detach_temporal_adapter(model)

    after = frame_by_frame_reference(model, batch)
    assert torch.equal(before, after)
    assert model.temporal_adapter is None


@pytest.mark.parametrize("overrides", BACKENDS)
def test_zero_gate_is_bitwise_identical_to_frame_independent(tmp_path, overrides):
    """``adapter_init_gate: 0`` must reproduce the spatial model exactly.

    This is the strongest available statement that the temporal branch does not
    change existing behaviour: the temporal modules run, consume the time
    features, and advance their state, yet the emitted field is bit-for-bit the
    frame-independent prediction.
    """
    model, _ = build_tiny_model(tmp_path)
    batch = make_sequence_batch(frames=5)
    reference = frame_by_frame_reference(model, batch)

    cfg = temporal_config(latent={"adapter_init_gate": 0.0}, **overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()
    with torch.no_grad():
        out = runner(batch).predictions

    assert out.shape == reference.shape
    assert torch.equal(out, reference), (
        f"max |diff| = {(out - reference).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("overrides", BACKENDS)
def test_default_gate_is_near_identity_but_not_identity(tmp_path, overrides):
    model, _ = build_tiny_model(tmp_path)
    batch = make_sequence_batch(frames=5)
    reference = frame_by_frame_reference(model, batch)

    cfg = temporal_config(**overrides)
    assert cfg.latent.adapter_init_gate == pytest.approx(1e-3)
    runner, _ = _runner(model, cfg)
    runner.eval()
    with torch.no_grad():
        out = runner(batch).predictions

    assert not torch.equal(out, reference)
    scale = reference.abs().mean()
    deviation = (out - reference).abs().mean()
    # Near-identity: the pretrained prediction is preserved to well under a
    # percent, so a fine-tune starts from the spatial model rather than noise.
    assert deviation / scale < 5e-2


# ---------------------------------------------------------------------------
# trainability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", BACKENDS)
def test_every_temporal_parameter_receives_gradient(tmp_path, overrides):
    """A near-identity init must still let the temporal module learn.

    An exactly-zero gate would make this fail for every parameter, which is why
    the default gate is small-but-nonzero.
    """
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(**overrides)
    runner, _ = _runner(model, cfg)
    runner.train()

    out = runner(make_sequence_batch(frames=4)).predictions
    out.pow(2).mean().backward()

    params = dict(model.named_parameters())
    names = temporal_parameter_names(model)
    assert names, "no temporal parameters were registered"
    dead = [
        n
        for n in names
        if params[n].grad is None or float(params[n].grad.abs().sum()) == 0.0
    ]
    assert not dead, f"temporal parameters with zero gradient: {dead}"


@pytest.mark.parametrize("overrides", BACKENDS)
def test_temporal_parameters_change_under_optimization(tmp_path, overrides):
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(**overrides)
    runner, _ = _runner(model, cfg)
    runner.train()

    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.startswith("temporal_adapter.")}
    opt = torch.optim.Adam(
        [p for n, p in model.named_parameters() if n.startswith("temporal_adapter.")], lr=1e-2
    )
    batch = make_sequence_batch(frames=4)
    for _ in range(3):
        opt.zero_grad()
        out = runner(batch)
        loss = (out.predictions - out.target_frames).pow(2).mean()
        loss.backward()
        opt.step()

    after = dict(model.named_parameters())
    unchanged = [n for n, v in before.items() if torch.equal(v, after[n].detach())]
    assert not unchanged, f"temporal parameters did not move: {unchanged}"


def test_freeze_policy_and_param_groups(tmp_path):
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(
        freeze={
            "backbone": True,
            "encoder": True,
            "decoder": False,
            "unfreeze_schedule": [{"epoch": 2, "modules": ["backbone"], "lr_scale": 0.1}],
        }
    )
    _runner(model, cfg)

    trainable = apply_freeze_policy(model, cfg, epoch=0)
    assert trainable["temporal"] is True
    assert trainable["backbone"] is False
    assert trainable["encoder"] is False
    assert trainable["decoder"] is True
    assert all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if classify_parameter(n) == "backbone"
    )

    groups = build_param_groups(model, cfg)
    names = {g["name"] for g in groups}
    assert "backbone" not in names  # frozen params must not be handed to the optimizer
    assert "temporal" in names
    assert dict((g["name"], g["lr"]) for g in groups)["temporal"] == pytest.approx(
        cfg.freeze.lr_temporal
    )

    trainable = apply_freeze_policy(model, cfg, epoch=2)
    assert trainable["backbone"] is True
    groups = build_param_groups(model, cfg)
    assert "backbone" in {g["name"] for g in groups}


# ---------------------------------------------------------------------------
# causality and genuine use of history
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", BACKENDS)
def test_causality_future_inputs_do_not_affect_earlier_outputs(tmp_path, overrides):
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(**overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()

    batch = make_sequence_batch(frames=5)
    with torch.no_grad():
        base = runner(batch).predictions

    perturbed = copy.deepcopy(batch)
    torch.manual_seed(99)
    perturbed["x"][:, -1] = torch.randn_like(perturbed["x"][:, -1])
    perturbed["time_features"][:, -1] = torch.randn_like(perturbed["time_features"][:, -1])
    with torch.no_grad():
        after = runner(perturbed).predictions

    assert torch.equal(base[:, :-1], after[:, :-1]), "future input leaked into an earlier frame"
    assert not torch.equal(base[:, -1], after[:, -1])


@pytest.mark.parametrize("overrides", BACKENDS)
def test_same_current_frame_different_history_gives_different_output(tmp_path, overrides):
    """The controlled sequence test.

    Two sequences share the final frame *and* its time features, and differ only
    in their history. A model that merely consumed date embeddings would produce
    identical output; a model with real temporal memory cannot.

    The gate is raised above its tiny default here so the effect is measurable on
    an untrained adapter -- the point is that the pathway carries information,
    not that an untrained model is accurate.
    """
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(latent={"adapter_init_gate": 1.0}, **overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()

    batch_a = make_sequence_batch(frames=5, seed=11)
    batch_b = copy.deepcopy(batch_a)
    torch.manual_seed(23)
    # Replace only the history; keep the final frame and ALL time features equal.
    batch_b["x"][:, :-1] = torch.randn_like(batch_b["x"][:, :-1])
    assert torch.equal(batch_a["x"][:, -1], batch_b["x"][:, -1])
    assert torch.equal(batch_a["time_features"], batch_b["time_features"])

    with torch.no_grad():
        out_a = runner(batch_a).predictions
        out_b = runner(batch_b).predictions

    final_diff = (out_a[:, -1] - out_b[:, -1]).abs().max().item()
    field_scale = out_a[:, -1].abs().mean().item()
    assert final_diff > 1e-6 * max(field_scale, 1e-6), (
        "history had no effect on the final frame: the temporal pathway is inert"
    )


@pytest.mark.parametrize("overrides", BACKENDS)
def test_date_features_alone_do_not_explain_history_sensitivity(tmp_path, overrides):
    """Separate date conditioning from temporal memory.

    Holding history fixed and changing only the time features gives one effect;
    holding time features fixed and changing history gives another. Both must be
    non-zero, which shows the two pathways are distinct.
    """
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(latent={"adapter_init_gate": 1.0}, **overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()

    base = make_sequence_batch(frames=5, seed=31)
    with torch.no_grad():
        ref = runner(base).predictions[:, -1]

    only_time = copy.deepcopy(base)
    torch.manual_seed(5)
    only_time["time_features"] = torch.randn_like(only_time["time_features"])
    with torch.no_grad():
        out_time = runner(only_time).predictions[:, -1]

    only_hist = copy.deepcopy(base)
    torch.manual_seed(6)
    only_hist["x"][:, :-1] = torch.randn_like(only_hist["x"][:, :-1])
    with torch.no_grad():
        out_hist = runner(only_hist).predictions[:, -1]

    d_time = (out_time - ref).abs().mean().item()
    d_hist = (out_hist - ref).abs().mean().item()
    assert d_time > 0.0, "time features are inert"
    assert d_hist > 0.0, "history is inert -- this would be a date-embedding-only model"


# ---------------------------------------------------------------------------
# state isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", BACKENDS)
def test_state_does_not_leak_between_batch_elements(tmp_path, overrides):
    """Sample 0's history must not influence sample 1's output.

    Two batch elements are independent sequences. Perturbing element 0's history
    must leave element 1 bit-for-bit unchanged.
    """
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(latent={"adapter_init_gate": 1.0}, **overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()

    batch = make_sequence_batch(batch=2, frames=4, seed=41)
    with torch.no_grad():
        base = runner(batch).predictions

    perturbed = copy.deepcopy(batch)
    torch.manual_seed(77)
    perturbed["x"][0] = torch.randn_like(perturbed["x"][0])
    with torch.no_grad():
        after = runner(perturbed).predictions

    assert torch.equal(base[1], after[1]), "state leaked across batch elements"
    assert not torch.equal(base[0], after[0])


@pytest.mark.parametrize("overrides", BACKENDS)
def test_reset_mask_clears_state_per_sample(tmp_path, overrides):
    """A reset flag on one sample must zero only that sample's memory."""
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(latent={"adapter_init_gate": 1.0}, **overrides)
    runner, adapter = _runner(model, cfg)
    runner.eval()

    batch = make_sequence_batch(batch=2, frames=4, seed=53)
    # Reset sample 0 midway; sample 1 keeps its history.
    reset_mid = batch["reset"].clone()
    reset_mid[0, 2] = True
    batch_reset = {**copy.deepcopy(batch), "reset": reset_mid}

    with torch.no_grad():
        plain = runner(batch).predictions
        with_reset = runner(batch_reset).predictions

    assert torch.equal(plain[1], with_reset[1]), "reset on sample 0 disturbed sample 1"
    assert not torch.equal(plain[0], with_reset[0]), "reset had no effect on sample 0"

    # And a reset must equal starting a fresh sequence from that frame.
    tail = {
        "x": batch["x"][:1, 2:],
        "y": batch["y"][:1, 2:],
        "time_features": batch["time_features"][:1, 2:],
        "interval_ratio": batch["interval_ratio"][:1, 2:],
        "reset": torch.ones(1, 2, dtype=torch.bool),
        "static_x": batch["static_x"][:1],
        "static_y": batch["static_y"][:1],
    }
    tail["reset"][:, 1] = False
    with torch.no_grad():
        fresh = runner(tail).predictions
    assert torch.allclose(with_reset[:1, 2:], fresh, atol=0), (
        "resetting state mid-sequence differs from starting a fresh sequence"
    )


# ---------------------------------------------------------------------------
# chunked inference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", BACKENDS)
def test_chunked_inference_matches_single_pass(tmp_path, overrides):
    """Carrying state across chunk boundaries must be exact, not approximate."""
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(context_length=8, output_length=8,
                          latent={"adapter_init_gate": 1.0}, **overrides)
    runner, _ = _runner(model, cfg)
    runner.eval()

    total = 8
    batch = make_sequence_batch(batch=1, frames=total, seed=61)
    with torch.no_grad():
        single = runner(batch, emit_indices=tuple(range(total))).predictions

    def slice_chunk(lo: int, hi: int) -> dict[str, torch.Tensor]:
        out = {
            "x": batch["x"][:, lo:hi],
            "y": batch["y"][:, lo:hi],
            "time_features": batch["time_features"][:, lo:hi],
            "interval_ratio": batch["interval_ratio"][:, lo:hi],
            "reset": batch["reset"][:, lo:hi].clone(),
        }
        # Only the very first chunk starts a fresh sequence.
        out["reset"][:] = False
        if lo == 0:
            out["reset"][:, 0] = True
        for key in ("static_x", "static_y"):
            out[key] = batch[key]
        return out

    chunks = [slice_chunk(0, 3), slice_chunk(3, 6), slice_chunk(6, 8)]
    results = runner.run_chunked(chunks, chunk_warmup=0, carry_state=True)
    chunked = torch.cat([r.predictions for r in results], dim=1)

    assert chunked.shape == single.shape
    assert torch.equal(chunked, single), (
        f"chunked != single pass, max |diff| = {(chunked - single).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# warm-up / emission bookkeeping
# ---------------------------------------------------------------------------
def test_warmup_frames_are_not_emitted(tmp_path):
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(context_length=7, output_length=3, warmup_length=4)
    runner, _ = _runner(model, cfg)
    runner.eval()

    batch = make_sequence_batch(frames=7)
    with torch.no_grad():
        out = runner(batch)
    assert out.emitted_indices == (4, 5, 6)
    assert out.predictions.shape[1] == 3
    assert out.target_frames.shape[1] == 3
