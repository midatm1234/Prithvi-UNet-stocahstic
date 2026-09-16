"""Temporal adapter and sequence runner for Prithvi-UNet.

The task is *causal, same-day, sequence-conditioned downscaling*: for output
date ``t`` the model uses the coarse predictors at ``t``, the predictor history
before ``t``, static fields, temporal metadata, and learned temporal state. It
never uses predictors after ``t``, never shifts the target, and never consumes
observed high-resolution targets.

Concretely, per frame::

    z_t          = spatial_encoder(predictors_t, static)      # existing, frozen-able
    h_t, state_t = temporal_backend(z_t, state_{t-1}, tau_t)  # new
    y_hat_t      = spatial_decoder(z_t + g * proj(h_t), skips_t)   # existing decoder

where ``z_t`` is the U-Net bottleneck latent, ``tau_t`` the temporal metadata,
and ``g`` a learnable gate initialized near zero (see
:class:`TemporalLatentAdapter`).

Why the bottleneck
------------------
For the SA case the bottleneck is ``[B, 1024, 16, 16]`` while the post-backbone
feature map is ``[B, 1024, 128, 128]`` -- 64x larger. Carrying seven frames of
recurrent state at the bottleneck costs ~29 MB in fp32 at batch 4; doing it at
full resolution would cost ~1.9 GB before autograd. The bottleneck also still
has genuine spatial extent (16x16 latent cells over the domain, each ~8 target
pixels), so a 3x3 convolutional kernel moves information a meaningful physical
distance per day. Multi-scale injection is deliberately *not* enabled by
default; it is a larger change with no measured justification yet.

Why an adapter rather than a new backbone
-----------------------------------------
The pretrained spatial weights are the asset. A near-identity adapter means the
temporal model *starts* as the frame-independent model and can only depart from
it by learning, so a failed temporal run degrades to the existing baseline
rather than to noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from granitewxc.temporal.backends import (
    ConvGRUBackend,
    TemporalBackend,
    TemporalMambaBackend,
    build_backend,
)
from granitewxc.temporal.config import TemporalConfig

__all__ = [
    "TemporalLatentAdapter",
    "TemporalSequenceModel",
    "attach_temporal_adapter",
    "detach_temporal_adapter",
    "temporal_parameter_names",
]


class TemporalLatentAdapter(nn.Module):
    """Near-identity temporal adapter around a :class:`TemporalBackend`.

    Structure::

        z  --> in_proj  --> backend.step(., state, tau) --> out_proj --> * gate --> delta

    ``in_proj`` maps the bottleneck's ``embed_dim`` channels down to
    ``hidden_channels`` so the recurrent state is cheap; ``out_proj`` maps back.

    Initialization
    --------------
    ``gate`` is a **learnable per-channel** vector initialized to
    ``adapter_init_gate`` (default ``1e-3``). This matters:

    * With ``adapter_init_gate = 0`` the adapter is an *exact* identity and the
      model reproduces the pretrained spatial prediction bit-for-bit -- useful
      for the legacy-parity test -- but the gradient with respect to every
      parameter upstream of the gate is exactly zero, so the temporal module
      could never start learning. That is the classic zero-init trap.
    * With a small non-zero gate the initial perturbation is ~0.1% of the
      bottleneck magnitude (negligible for the prediction) while *every*
      temporal parameter receives a non-zero gradient from the first step.

    Both behaviours are available and both are tested; the default is the small
    non-zero gate because it is the one that can actually train.
    """

    def __init__(
        self,
        latent_channels: int,
        *,
        cfg: TemporalConfig,
        time_dim: int,
    ):
        super().__init__()
        hidden = int(cfg.latent.hidden_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_channels = hidden
        self.cfg_backend = cfg.backend
        self.init_gate = float(cfg.latent.adapter_init_gate)

        groups = math.gcd(int(cfg.latent.groups), self.latent_channels) or 1
        self.pre_norm = (
            nn.GroupNorm(num_groups=max(1, groups), num_channels=self.latent_channels)
            if cfg.latent.norm == "group"
            else nn.Identity()
        )
        self.in_proj = nn.Conv2d(self.latent_channels, hidden, kernel_size=1)
        self.backend: TemporalBackend = build_backend(cfg, hidden, time_dim)
        self.out_proj = nn.Conv2d(hidden, self.latent_channels, kernel_size=1)
        self.gate = nn.Parameter(torch.full((1, self.latent_channels, 1, 1), self.init_gate))

        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        # Keep the pre-gate signal O(1) so the gate value alone sets the
        # perturbation magnitude and is interpretable.
        nn.init.normal_(self.out_proj.weight, std=1.0 / math.sqrt(hidden))

        self.learned_init = cfg.state.init == "learned"
        if self.learned_init:
            self.init_state_seed = nn.Parameter(torch.zeros(1, hidden, 1, 1))

    # -- state ------------------------------------------------------------
    def init_state(self, batch_size: int, height: int, width: int, *, device, dtype):
        state = self.backend.init_state(batch_size, height, width, device=device, dtype=dtype)
        return state

    def zero_state(self, batch_size: int, height: int, width: int, *, device, dtype):
        return self.backend.init_state(batch_size, height, width, device=device, dtype=dtype)

    def detach_state(self, state):
        if isinstance(self.backend, TemporalMambaBackend):
            return TemporalMambaBackend.detach_state(state)
        return TemporalBackend.detach_state(state)

    def reset_state(self, state, reset_mask: torch.Tensor, *, batch_size, height, width, device, dtype):
        if reset_mask is None or not bool(reset_mask.any()):
            return state
        zeros = self.zero_state(batch_size, height, width, device=device, dtype=dtype)
        if isinstance(self.backend, TemporalMambaBackend):
            return TemporalMambaBackend.reset_state_where(state, zeros, reset_mask)
        return TemporalBackend.reset_state_where(state, zeros, reset_mask)

    # -- step -------------------------------------------------------------
    def forward(
        self,
        latent: torch.Tensor,
        state: Any,
        time_features: torch.Tensor | None,
        interval_ratio: torch.Tensor | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        b, _, h, w = latent.shape
        if state is None:
            state = self.init_state(b, h, w, device=latent.device, dtype=latent.dtype)
            if self.learned_init:
                state = self._seed_state(state)
        elif reset_mask is not None:
            state = self.reset_state(
                state,
                reset_mask,
                batch_size=b,
                height=h,
                width=w,
                device=latent.device,
                dtype=latent.dtype,
            )
        x = self.in_proj(self.pre_norm(latent))
        hidden, new_state = self.backend.step(x, state, time_features, interval_ratio)
        delta = self.out_proj(hidden) * self.gate
        return delta, new_state

    def _seed_state(self, state):
        """Add the learned initial-state seed to a freshly zeroed ConvGRU state."""
        if not isinstance(self.backend, ConvGRUBackend):
            return state
        seeded = []
        for layer_state in state:
            if isinstance(layer_state, tuple):
                h, c = layer_state
                seeded.append((h + self.init_state_seed.to(h.dtype), c))
            else:
                seeded.append(layer_state + self.init_state_seed.to(layer_state.dtype))
        return seeded


# ---------------------------------------------------------------------------
# attach / detach
# ---------------------------------------------------------------------------
def _resolve_bottleneck_channels(model: nn.Module) -> int:
    """Channel count of the tensor reaching the temporal hook."""
    # ``_apply_temporal_latent`` runs on ``conv_after_backbone``'s output.
    conv = getattr(model, "conv_after_backbone", None)
    if conv is None:
        raise AttributeError(
            "Model has no conv_after_backbone; temporal attachment expects a "
            "ClimateDownscaleFinetuneUNETModel."
        )
    return int(conv.out_channels)


def attach_temporal_adapter(
    model: nn.Module,
    cfg: TemporalConfig,
    *,
    time_feature_dim: int,
) -> TemporalLatentAdapter:
    """Attach a :class:`TemporalLatentAdapter` to a spatial Prithvi-UNet model.

    The adapter becomes a submodule named ``temporal_adapter``, so it appears in
    ``state_dict()`` under the ``temporal_adapter.*`` prefix -- which is exactly
    the set of keys that checkpoint migration expects to be *missing* when
    loading a purely spatial checkpoint.
    """
    if getattr(model, "temporal_adapter", None) is not None:
        raise RuntimeError("A temporal adapter is already attached to this model.")
    channels = _resolve_bottleneck_channels(model)
    adapter = TemporalLatentAdapter(channels, cfg=cfg, time_dim=time_feature_dim)
    model.temporal_adapter = adapter
    model._temporal_ctx = None
    return adapter


def attach_native_pair_adapter(
    model: nn.Module,
    cfg: TemporalConfig,
) -> "NativePairAdapter":
    """Attach the paired-state adapter for ``backend: native_pair``.

    Uses the same ``temporal_adapter`` attribute name as the bottleneck adapter,
    which is what makes checkpoint migration, the freeze policy and the
    ``lr_temporal`` parameter group apply without any change: all three key off
    the ``temporal_adapter.`` state-dict prefix, not off the class.

    Unlike :func:`attach_temporal_adapter`, this adapter does not act on the
    bottleneck at all -- it conditions the token stream before the transformer,
    and the paired input is assembled by the sequence runner. It also validates
    that ``data.n_input_timestamps`` agrees with ``history_offsets``, because a
    mismatch there is silent: the patch embedding would simply read a differently
    sized channel block and the model would train on garbage.
    """
    from granitewxc.temporal.native_pair import NativePairAdapter

    if getattr(model, "temporal_adapter", None) is not None:
        raise RuntimeError("A temporal adapter is already attached to this model.")
    if cfg.backend != "native_pair":
        raise ValueError(
            f"attach_native_pair_adapter requires backend 'native_pair', got {cfg.backend!r}."
        )

    expected_ts = cfg.native_pair.n_input_timestamps
    actual_ts = int(getattr(model, "n_input_timestamps", 1))
    if actual_ts != expected_ts:
        raise ValueError(
            f"data.n_input_timestamps is {actual_ts} but temporal.native_pair.history_offsets="
            f"{list(cfg.native_pair.history_offsets)} needs {expected_ts} "
            f"(the history dates plus date t). Set data.n_input_timestamps: {expected_ts}. "
            "These are not independent knobs: the patch embedding's input channel count is "
            "what carries the time axis, exactly as upstream Prithvi-WxC does."
        )

    embedding = getattr(model, "embedding", None)
    proj = getattr(embedding, "proj", None)
    if proj is None:
        raise AttributeError(
            "Model has no embedding.proj; native-pair attachment expects the PatchEmbed "
            "input head of a ClimateDownscaleFinetuneUNETModel."
        )
    total_in = int(proj.in_channels)
    if total_in % expected_ts != 0:
        raise ValueError(
            f"embedding.proj has {total_in} input channels, which is not divisible by "
            f"{expected_ts} timestamps."
        )
    per_timestamp = total_in // expected_ts

    adapter = NativePairAdapter(
        cfg=cfg,
        embed_dim_backbone=int(getattr(model, "embed_dim_backbone")),
        post_backbone_channels=_resolve_bottleneck_channels(model),
        predictor_channels_per_timestamp=per_timestamp,
        predictor_channel_names=getattr(model, "predictor_channel_names", None),
    )
    adapter.attach_history_projection(proj)
    model.temporal_adapter = adapter
    model._temporal_ctx = None
    return adapter


def detach_temporal_adapter(model: nn.Module) -> None:
    """Remove the adapter, restoring the exact legacy computation path."""
    if getattr(model, "temporal_adapter", None) is None:
        return
    remove = getattr(model.temporal_adapter, "remove_history_projection", None)
    if remove is not None:
        remove()
    model.temporal_adapter = None
    model._temporal_ctx = None


def temporal_parameter_names(model: nn.Module) -> list[str]:
    """Names of parameters introduced by the temporal extension."""
    return [n for n, _ in model.named_parameters() if n.startswith("temporal_adapter.")]


# ---------------------------------------------------------------------------
# sequence runner
# ---------------------------------------------------------------------------
@dataclass
class SequenceOutput:
    """Result of running one sequence window.

    ``predictions`` are in *physical* units (the base model's ``_decode_outputs``
    has already inverted normalization), stacked on a new time axis:
    ``[B, T_out, C_target, H_fine, W_fine]``.

    ``normalized`` holds the corresponding pre-inverse (normalized) outputs when
    requested, which is what residual refinement and the normalized-space losses
    consume.
    """

    predictions: torch.Tensor
    normalized: torch.Tensor | None
    target_frames: torch.Tensor | None
    valid_mask: torch.Tensor | None
    emitted_indices: tuple[int, ...]
    final_state: Any
    #: ``interval_ratio`` restricted to the emitted frames. Carried here rather
    #: than re-sliced by each caller: the full-window ratio has ``context_length``
    #: entries while the emitted tensors have ``output_length``, and pairing the
    #: two by accident silently mis-scales the tendency loss (or, as it did once,
    #: raises a shape error).
    interval_ratio: torch.Tensor | None = None
    #: Number of base-model (and therefore backbone) forward passes this call
    #: performed. Recorded because variants sharing an optimizer-step budget do
    #: not necessarily share a compute budget: the recurrent backends must run
    #: every warm-up frame to build state, while a finite-history pathway runs
    #: only the frames it emits.
    backbone_evaluations: int = 0


class TemporalSequenceModel(nn.Module):
    """Drive a frame-wise Prithvi-UNet over a sequence, carrying temporal state.

    The loop is explicit and causal: frame ``t`` is fed to the base model with
    the adapter context holding ``state_{t-1}``; the adapter advances the state
    inside the base model's bottleneck and the runner reads it back. There is no
    mechanism by which frame ``t`` can see frame ``t+1``.

    Tensor layouts
    --------------
    Input batch (produced by
    :class:`granitewxc.temporal.sequence_dataset.TemporalSequenceDataset`)::

        x               [B, T, C_pred, H, W]      predictors, coarse fields already
                                                  co-registered to the fine grid
        y               [B, T, C_target, H, W]    targets, same dates as x
        static_x        [B, C_static, H, W]       time-invariant
        static_y        [B, C_static, H, W]
        time_features   [B, T, F]                 calendar metadata
        interval_ratio  [B, T]                    dt_actual / cadence
        reset           [B, T]  (bool)            True where state must reset
        valid_mask      [B, T, C_target, H, W]    finite-target mask

    ``C_pred`` here is the *native* per-frame predictor channel count, i.e.
    ``n_input_timestamps * n_vars * n_levels`` exactly as the frame model already
    expects. The outer sequence axis ``T`` is a **separate** axis from the
    backbone's native ``n_input_timestamps`` (which is 1 in every configuration
    in this repository). The two are never merged, and history is never faked by
    duplicating a snapshot into the native axis.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cfg: TemporalConfig,
        *,
        adapter: TemporalLatentAdapter | None = None,
    ):
        super().__init__()
        self.base = base_model
        self.cfg = cfg
        # Cold starts (repeating date t into the history slot when the run has no
        # earlier date) are an *inference* policy. During training a supervised
        # frame without real history means the window geometry is wrong, so the
        # default is to raise; ``run_sequence_inference`` opts in explicitly.
        self._allow_cold_start = False
        self.adapter = adapter if adapter is not None else getattr(base_model, "temporal_adapter", None)
        if self.adapter is None:
            raise ValueError(
                "TemporalSequenceModel requires an attached temporal adapter; call "
                "attach_temporal_adapter() first."
            )

    # -- helpers ----------------------------------------------------------
    @property
    def is_native_pair(self) -> bool:
        """True when temporal information enters before the transformer."""
        return self.cfg.backend == "native_pair"

    def _frame_x(self, batch: dict[str, torch.Tensor], t: int) -> torch.Tensor:
        """Predictor tensor for frame ``t``: one date, or a channel-stacked pair.

        For ``backend: native_pair`` this stacks the configured history dates and
        date ``t`` along the channel axis, oldest first, which is the native
        ``[B, time x parameter, H, W]`` layout the patch embedding expects. The
        two dates therefore reach the transformer together instead of being
        encoded separately and combined afterwards.
        """
        if not self.is_native_pair:
            return batch["x"][:, t]
        from granitewxc.temporal.native_pair import build_pair_input

        # ``__position_offset`` is the absolute index of frame 0 of this batch
        # within its contiguous run. Inference supplies it; training windows do
        # not, and there the default of 0 is what makes a too-short window raise
        # instead of quietly cold-starting a supervised frame.
        position_offset = int(batch.get("__position_offset", 0) or 0)
        return build_pair_input(
            batch["x"],
            t,
            history_offsets=self.cfg.native_pair.history_offsets,
            history_mode=self.cfg.native_pair.history_mode,
            position_offset=position_offset,
            allow_cold_start=bool(self._allow_cold_start),
        )

    def _frame(self, batch: dict[str, torch.Tensor], t: int) -> dict[str, torch.Tensor]:
        """Slice frame ``t`` into the dict the frame model expects."""
        frame: dict[str, torch.Tensor] = {"x": self._frame_x(batch, t)}
        # The decoder needs geometry, never verification values.
        if "__output_shape" in batch:
            frame["__output_shape"] = batch["__output_shape"]
        elif "y" in batch:
            frame["__output_shape"] = tuple(batch["y"].shape[-2:])
        for key in ("static_x", "static_y"):
            if key in batch:
                frame[key] = batch[key]
        if "__target_valid_mask" in batch:
            frame["__target_valid_mask"] = batch["__target_valid_mask"][:, t]
        for key in ("__output_crop", "__input_scaler_offset", "__output_scaler_offset", "__scaler_offset"):
            if key in batch:
                frame[key] = batch[key]
        return frame

    def emitted_indices(self, total: int | None = None) -> tuple[int, ...]:
        """Frame indices that are supervised / written.

        Warm-up frames at the start of a window advance the state but are not
        emitted; the remaining ``output_length`` frames at the *end* of the
        window are. Taking the tail rather than the head is what makes each
        emitted frame benefit from the maximum available history.
        """
        n = int(total if total is not None else self.cfg.context_length)
        out_len = min(int(self.cfg.output_length), n - int(self.cfg.warmup_length))
        start = n - out_len
        return tuple(range(start, n))

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        initial_state: Any = None,
        return_normalized: bool = False,
        emit_indices: Sequence[int] | None = None,
        capture_latent: bool = False,
    ) -> SequenceOutput:
        x = batch["x"]
        if x.dim() != 5:
            raise ValueError(
                f"TemporalSequenceModel expects x with shape [B, T, C, H, W]; got {tuple(x.shape)}."
            )
        b, t_total = x.shape[0], x.shape[1]
        emit = tuple(emit_indices) if emit_indices is not None else self.emitted_indices(t_total)
        emit_set = set(emit)

        time_features = batch.get("time_features")
        interval_ratio = batch.get("interval_ratio")
        reset = batch.get("reset")

        tbptt = int(self.cfg.state.tbptt_chunk or 0)
        detach_between = bool(self.cfg.state.detach_between_chunks)

        state = initial_state
        preds: list[torch.Tensor] = []
        normed: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []

        # A finite-history pathway has no hidden state to build, so a frame that
        # is not emitted has no effect whatsoever and running it would be pure
        # waste. Skipping those frames is what makes this pathway cost 5 backbone
        # evaluations per SA window instead of 7, at identical emitted dates --
        # recorded explicitly because equal optimizer steps do not imply equal
        # compute.
        visit = tuple(emit) if self.is_native_pair else tuple(range(t_total))
        backbone_evaluations = 0

        for t in visit:
            if self.cfg.state.reset_every_frame:
                # Memory-disabled ablation: drop the carried state entirely, so the
                # backend sees only this frame plus its calendar features. The
                # parameters, optimizer and data order are unchanged, which is what
                # makes this a clean control for "date conditioning alone".
                state = None
                step_reset = None
            else:
                step_reset = None if reset is None else reset[:, t].to(torch.bool)
            ctx: dict[str, Any] = {
                "state": state,
                "time_features": None if time_features is None else time_features[:, t],
                "interval_ratio": None if interval_ratio is None else interval_ratio[:, t],
                "reset_mask": step_reset,
                "capture_latent": bool(capture_latent),
            }
            if self.is_native_pair:
                from granitewxc.temporal.native_pair import pair_time_scalars

                input_time, lead_time = pair_time_scalars(
                    interval_ratio,
                    t,
                    history_offsets=self.cfg.native_pair.history_offsets,
                    cadence_days=self.cfg.cadence_days,
                    batch_size=b,
                    device=x.device,
                    dtype=x.dtype,
                    lead_steps=0,  # zero predictor-to-target lead: same-day downscaling
                    position_offset=int(batch.get("__position_offset", 0) or 0),
                    allow_cold_start=bool(self._allow_cold_start),
                    cold_start_time_mode=self.cfg.native_pair.cold_start_time_mode,
                )
                ctx["native_input_time_hours"] = input_time
                ctx["native_lead_time_hours"] = lead_time
            self.base._temporal_ctx = ctx
            backbone_evaluations += 1
            frame = self._frame(batch, t)
            try:
                if return_normalized or capture_latent:
                    y_hat, y_norm = self.base(frame, return_pre_inverse=True)
                else:
                    y_hat = self.base(frame)
                    y_norm = None
            finally:
                # Always clear, so a later *frame-wise* call cannot accidentally
                # inherit a stale sequence context.
                captured = self.base._temporal_ctx
                self.base._temporal_ctx = None
            state = captured["state"]
            if capture_latent and captured.get("latents"):
                latents.extend(captured["latents"])

            if t in emit_set:
                preds.append(y_hat)
                if y_norm is not None:
                    normed.append(y_norm)

            # Truncated BPTT: cut the graph every ``tbptt`` frames so memory is
            # bounded for long windows while the *values* carried forward are
            # unchanged.
            if tbptt and detach_between and ((t + 1) % tbptt == 0) and (t + 1) < t_total:
                state = self.adapter.detach_state(state)

        predictions = torch.stack(preds, dim=1) if preds else x.new_zeros((b, 0))
        normalized = torch.stack(normed, dim=1) if normed else None
        targets = batch["y"][:, emit] if "y" in batch else None
        mask = (
            batch["__target_valid_mask"][:, emit]
            if "__target_valid_mask" in batch
            else None
        )
        emitted_ratio = None
        if interval_ratio is not None:
            emitted_ratio = interval_ratio[:, emit]
        return SequenceOutput(
            predictions=predictions,
            normalized=normalized,
            target_frames=targets,
            valid_mask=mask,
            emitted_indices=emit,
            final_state=state,
            interval_ratio=emitted_ratio,
            backbone_evaluations=backbone_evaluations,
        )

    # -- inference over a long continuous run -----------------------------
    @torch.no_grad()
    def run_chunked(
        self,
        chunks: Iterable[dict[str, torch.Tensor]],
        *,
        chunk_warmup: int,
        carry_state: bool = True,
        return_normalized: bool = False,
    ) -> list[SequenceOutput]:
        """Process a long continuous sequence in memory-bounded chunks.

        Chunk ``k > 0`` receives ``chunk_warmup`` frames that were already
        emitted by chunk ``k - 1``. Those frames are re-run only to rebuild
        state when ``carry_state`` is False; when ``carry_state`` is True the
        state is passed directly and the warm-up frames are skipped for
        emission, which makes the chunked result *identical* to the single-pass
        result rather than merely close. The equality is asserted by
        :func:`tests.test_temporal_chunked_inference`.
        """
        results: list[SequenceOutput] = []
        state = None
        for index, chunk in enumerate(chunks):
            total = chunk["x"].shape[1]
            warm = 0 if (index == 0 or carry_state) else min(chunk_warmup, max(total - 1, 0))
            emit = tuple(range(warm, total))
            out = self.forward(
                chunk,
                initial_state=state if carry_state else None,
                return_normalized=return_normalized,
                emit_indices=emit,
            )
            state = self.adapter.detach_state(out.final_state) if carry_state else None
            results.append(out)
        return results


# ---------------------------------------------------------------------------
# parameter grouping for staged fine-tuning
# ---------------------------------------------------------------------------
#: Prefix groups used by freezing and per-group learning rates.
PARAM_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "temporal": ("temporal_adapter.",),
    "backbone": ("backbone.",),
    "encoder": ("embedding.", "embedding_static.", "conv_before_backbone.", "downsampling_layers."),
    "decoder": (
        "conv_after_backbone.",
        "upsample_layers.",
        "output_conv_block.",
        "precip_wet_head.",
        "to_logits.",
    ),
}


def classify_parameter(name: str) -> str:
    """Classify learned weights separately from immutable physical scalers."""
    # Parameters for checkpoint compatibility, never learned unit conversions.
    if name.rsplit(".", 1)[-1] in {
        "input_scalers_mu", "input_scalers_sigma",
        "output_scalers_mu", "output_scalers_sigma",
        "static_input_scalers_mu", "static_input_scalers_sigma",
        "static_output_scalers_mu", "static_output_scalers_sigma",
    }:
        return "normalization"
    for group, prefixes in PARAM_GROUP_PREFIXES.items():
        if any(name.startswith(p) for p in prefixes):
            return group
    return "other"


def build_param_groups(
    model: nn.Module,
    cfg: TemporalConfig,
    *,
    lr_scale: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Optimizer parameter groups with per-group learning rates.

    Only parameters with ``requires_grad`` are included, so the freezing policy
    and the optimizer never disagree about what is being trained.
    """
    scale = lr_scale or {}
    lrs = {
        "temporal": cfg.freeze.lr_temporal,
        "decoder": cfg.freeze.lr_decoder,
        "encoder": cfg.freeze.lr_encoder,
        "backbone": cfg.freeze.lr_backbone,
        "other": cfg.freeze.lr_decoder,
    }
    buckets: dict[str, list[nn.Parameter]] = {k: [] for k in lrs}
    for name, param in model.named_parameters():
        if not param.requires_grad or classify_parameter(name) == "normalization":
            continue
        buckets[classify_parameter(name)].append(param)
    groups: list[dict[str, Any]] = []
    for group, params in buckets.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "param_names": [name for name, param in model.named_parameters()
                                if param.requires_grad and classify_parameter(name) == group],
                "lr": float(lrs[group]) * float(scale.get(group, 1.0)),
                "name": group,
            }
        )
    return groups


def apply_freeze_policy(model: nn.Module, cfg: TemporalConfig, epoch: int) -> dict[str, bool]:
    """Apply the staged freeze/unfreeze schedule for ``epoch``.

    Returns a mapping ``group -> trainable``, which the trainer logs so a run's
    record shows what was actually being optimized at each epoch rather than
    what the schedule intended.
    """
    frozen = {
        "backbone": bool(cfg.freeze.backbone),
        "encoder": bool(cfg.freeze.encoder),
        "decoder": bool(cfg.freeze.decoder),
        "temporal": bool(cfg.freeze.temporal),
        "other": False,
    }
    for stage in cfg.freeze.unfreeze_schedule:
        if epoch >= stage.epoch:
            for module in stage.modules:
                if module == "all":
                    for key in frozen:
                        frozen[key] = False
                else:
                    frozen[module] = False
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        param.requires_grad = group != "normalization" and not frozen[group]
    return {**{k: (not v) for k, v in frozen.items()}, "normalization": False}
