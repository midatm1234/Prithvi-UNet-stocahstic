"""Paired-state temporal pathway that reuses the backbone's native time axis.

Motivation
----------
The frame-independent Prithvi-UNet embeds one date, pushes it through the
transformer, and decodes it. The first temporal extension on this branch added a
ConvGRU / Mamba module *after* the transformer, at the U-Net bottleneck, and
neither backend met the pre-registered acceptance criteria
(``docs/temporal_model_results.md``).

This module tests a different hypothesis: that letting temporally separated
atmospheric states interact **inside** the shared transformer computation --
using the mechanism the architecture already has for exactly that -- is more
useful than bolting a randomly initialized memory module onto the end of it.

What the native mechanism actually is
-------------------------------------
Upstream ``PrithviWxC`` takes ``x`` of shape ``[batch, time, parameter, lat,
lon]`` and, before anything learned happens, folds the time axis into conv
channels::

    # PrithviWxC/model.py:1364-1369
    x_rescaled = x_rescaled.flatten(1, 2)   # [B, time x parameter, lat, lon]
    x_embedded = self.patch_embedding(x_rescaled)

``input_size_time`` (2 in every released checkpoint) has exactly one functional
use upstream: it sets the patch embedding's input channel count
(``channels=in_channels * input_size_time``, ``PrithviWxC/model.py:1015-1019``).
There is no temporal attention, no temporal position embedding and no recurrence
anywhere in the model. The paired-time input structure is channel stacking:
the patch projection linearly combines both dates before the shared transformer
processes those combined atmospheric features.

This repository's downstream model already replicates that mechanism verbatim
(``granitewxc/models/cordex_finetune_model.py:971``)::

    x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W)

It was simply never used: every shipped config sets ``n_input_timestamps: 1``.
This module supplies the second timestamp with a *real earlier date*.

What is honestly reused, and what is not
----------------------------------------
Reused, exactly:

* the native paired-time input structure (channel stacking before patch
  embedding), and therefore the transformer's own self-attention as the place
  where the two dates interact;
* the native *form* and *injection point* of time conditioning -- two
  ``nn.Linear(1, embed_dim//4)`` maps combined as
  ``cat(cos(it), cos(lt), sin(it), sin(lt))`` and added once to the token stream
  before the encoder (``PrithviWxC/model.py:1233-1253, 1386-1388``);
* every spatial weight of the existing Phase-1 checkpoint, and the decoder and
  skip pathways unchanged.

Foundation-weight provenance
----------------------------
The current ECCC loading configuration uses a 2560-dimensional checkpoint with
a 1024-dimensional downstream model, so it cannot directly initialize this
transformer. That shape mismatch does not establish how an older Phase-1
checkpoint was initialized: its earlier foundation-weight provenance remains
unresolved without training records. The spatial weights here are inherited
from the downscaling checkpoint. Native time-conditioning and auxiliary heads
are newly initialized. Similar time-weight histograms alone do not establish
whether a checkpoint's time maps learned useful information.

* *The pretraining cadence.* Pretraining drew ``input_time`` from
  ``[-3, -6, -9, -12]`` hours on 3-hourly instantaneous MERRA-2 states. This
  archive is daily and its predictors are daily **means**
  (``cell_methods = 'time: mean'`` on all 15). A 24-hour interval is passed
  because 24 hours is the truth; it is outside the exercised support and is
  recorded as such rather than relabelled to match the checkpoint.

How much history reaches each output
------------------------------------
Exactly ``len(history_offsets)`` earlier dates, and with the shipped
``history_offsets: [1]`` that is **one**: the prediction for date ``t`` is a
function of the coarse predictors at ``t-1`` and ``t`` only. This is a
finite-history conditional model, not an unbounded recurrent memory, and it is
not an autoregressive rollout -- no predicted output is ever fed back as input.

Under ``pretext.transition``, date ``t-2`` additionally influences the *weights*
through the auxiliary loss, but never the prediction for ``t``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from granitewxc.temporal.config import TemporalConfig


#: Hours in one day. The native time scalars are in hours (verified against
#: ``PrithviWxC.dataloaders.merra2.SampleSpec``, which computes
#: ``(inputs[1] - inputs[0]).total_seconds() / 3600``).
HOURS_PER_DAY = 24.0


class NativeTimeConditioningError(RuntimeError):
    """Raised when native time conditioning is asked for something impossible."""


# ---------------------------------------------------------------------------
# native-form time conditioning
# ---------------------------------------------------------------------------
class NativeTimeConditioning(nn.Module):
    """Additive token-level time conditioning in Prithvi-WxC's own form.

    Reproduces ``PrithviWxC.time_encoding`` exactly -- two ``Linear(1, D//4)``
    maps of the two time scalars, combined as
    ``cat(cos(it), cos(lt), sin(it), sin(lt))`` along the embedding axis -- and
    adds the result to the token stream once, before the encoder, which is where
    upstream consumes it (``tokens = x_embedded + static_embedded +
    time_encoding``).

    Two deliberate differences from upstream, both required and both recorded:

    * ``embed_dim`` here is the *downstream* backbone width (1024), not 2560, so
      the layers are newly initialized. See the module docstring for why the
      referenced pretrained weights are not shape-compatible.

    A structural consequence worth stating rather than discovering later: for the
    downscaling task ``lead_time`` is identically ``0``, so
    ``dL/d(lead_time_embedding.weight) = dL/d(lt) * lead_time = 0`` -- that one
    weight matrix cannot train, and only its **bias** adapts. This is a property
    of a zero-lead task, not a defect, and it is reported by
    :meth:`NativePairAdapter.structurally_dead_parameters` instead of being hidden.
    The auxiliary transition pass runs with ``lead_time = +Delta`` and does train
    it, which is one concrete way the pretraining-aligned objective exercises more
    of the native mechanism than the downscaling task alone can.
    * The contribution is scaled by a learnable per-channel ``gate`` initialized
      to ``time_conditioning_gate_init``. Upstream has no gate because it trains
      from scratch; here the point is to depart from a *working* frame-independent
      model rather than perturb it. With the default ``1e-3`` every parameter of
      this module receives a non-zero gradient from step 1, which an exactly-zero
      gate would not (``dL/dW = 0`` when the downstream factor is 0) -- the same
      zero-init trap already documented for the bottleneck adapter's gate.
    """

    def __init__(self, embed_dim: int, *, gate_init: float = 1.0e-3) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise NativeTimeConditioningError(
                f"embed_dim must be divisible by 4 for the native cos/sin time encoding "
                f"layout (four blocks of embed_dim//4), got {embed_dim}."
            )
        self.embed_dim = int(embed_dim)
        quarter = self.embed_dim // 4
        # Same shapes and the same bias=True as upstream.
        self.input_time_embedding = nn.Linear(1, quarter, bias=True)
        self.lead_time_embedding = nn.Linear(1, quarter, bias=True)
        self.gate = nn.Parameter(torch.full((1, 1, 1, self.embed_dim), float(gate_init)))
        self.gate_init = float(gate_init)

    def encoding(self, input_time: torch.Tensor, lead_time: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 1, 1, embed_dim]``, broadcastable over the token axes."""
        it = self.input_time_embedding(input_time.reshape(-1, 1, 1, 1))
        lt = self.lead_time_embedding(lead_time.reshape(-1, 1, 1, 1))
        return torch.cat((torch.cos(it), torch.cos(lt), torch.sin(it), torch.sin(lt)), dim=3)

    def forward(
        self,
        tokens: torch.Tensor,
        input_time: torch.Tensor,
        lead_time: torch.Tensor,
    ) -> torch.Tensor:
        """Add gated native time conditioning to ``[B, global, local, embed]`` tokens."""
        if tokens.shape[-1] != self.embed_dim:
            raise NativeTimeConditioningError(
                f"token embedding width {tokens.shape[-1]} does not match the configured "
                f"embed_dim {self.embed_dim}."
            )
        enc = self.encoding(input_time.to(tokens.dtype), lead_time.to(tokens.dtype))
        return tokens + self.gate.to(tokens.dtype) * enc


# ---------------------------------------------------------------------------
# pretext heads
# ---------------------------------------------------------------------------
class PredictorFieldHead(nn.Module):
    """Predict a normalized predictor field from post-backbone features.

    Used by both pretraining-aligned auxiliary objectives. Each objective owns
    its own head -- infilling a masked field and extrapolating a later field are
    different maps -- but both heads sit on top of the **shared** encoder,
    transformer and post-backbone convolution, which is what makes the auxiliary
    gradients reach the components the downscaling path also uses. A head trained
    in isolation would be no evidence of an improved representation.

    The final convolution is initialized **small but non-zero** (std ``1e-3``),
    not zero. Zero would keep the auxiliary prediction at a clean constant, but it
    would also make ``dL/dfeatures = 0`` on the first update, so the shared trunk
    -- the thing these objectives exist to train -- would receive no auxiliary
    gradient at all until the head had moved off zero. That is the same zero-init
    trap that applies to the residual gates, in the one place where it would have
    silently defeated the purpose of the term.
    ``test_pretext_losses_reach_the_shared_trunk`` asserts the trunk really does
    get gradient on step 1.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        nn.init.normal_(self.body[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------
class NativePairAdapter(nn.Module):
    """Owns everything the native-pair pathway adds to the spatial model.

    Deliberately registered on the base model as ``temporal_adapter`` even though
    it is not a bottleneck adapter, because that single name is the hinge the
    rest of this package already keys off:

    * ``granitewxc/temporal/checkpoint.py`` allows exactly the
      ``temporal_adapter.`` prefix to be missing when migrating a Phase-1
      checkpoint;
    * ``PARAM_GROUP_PREFIXES`` maps that prefix to the ``temporal`` group, so the
      freeze policy and ``lr_temporal`` apply without modification.

    ``injects_at_bottleneck = False`` tells the base model's bottleneck hook to
    return its input untouched, so this pathway does **not** also perturb the
    U-Net bottleneck. All of its influence enters before or within the
    transformer.
    """

    #: Read by ``ClimateDownscaleFinetuneUNETModel._apply_temporal_latent``.
    injects_at_bottleneck: bool = False

    def __init__(
        self,
        *,
        cfg: TemporalConfig,
        embed_dim_backbone: int,
        post_backbone_channels: int,
        predictor_channels_per_timestamp: int,
        predictor_channel_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.cfg_backend = cfg.backend
        self.pair_cfg = cfg.native_pair
        self.cadence_days = float(cfg.cadence_days)
        self.predictor_channels_per_timestamp = int(predictor_channels_per_timestamp)
        self.history_projection: nn.Conv2d | None = None
        self._history_hook = None
        names = tuple(predictor_channel_names or ())
        if names and len(names) != self.predictor_channels_per_timestamp:
            raise ValueError("Predictor channel names must match the per-timestamp channel count.")
        # Elevation and validity indicators remain inputs, but are not weather
        # reconstruction targets. NARR masks use 1 for observed/valid cells.
        excluded = {"elev", "elevation", "orog", "orography", "landmask", "lsm"}
        self.atmospheric_channel_indices = tuple(
            i for i in range(self.predictor_channels_per_timestamp)
            if not names or (not names[i].startswith("mask_") and names[i].lower() not in excluded)
        )
        if not self.atmospheric_channel_indices:
            raise ValueError("The auxiliary atmospheric channel selection is empty.")
        self.atmospheric_mask_indices = tuple(
            names.index("mask_" + names[i]) if names and "mask_" + names[i] in names else None
            for i in self.atmospheric_channel_indices
        )

        self.time_conditioning: NativeTimeConditioning | None = None
        if self.pair_cfg.time_conditioning:
            self.time_conditioning = NativeTimeConditioning(
                embed_dim_backbone,
                gate_init=self.pair_cfg.time_conditioning_gate_init,
            )

        pretext = self.pair_cfg.pretext
        self.recon_head: PredictorFieldHead | None = None
        if pretext.masked_reconstruction_enabled:
            self.recon_head = PredictorFieldHead(
                post_backbone_channels,
                len(self.atmospheric_channel_indices),
                hidden=pretext.head_hidden,
            )
        self.transition_head: PredictorFieldHead | None = None
        if pretext.transition_enabled:
            self.transition_head = PredictorFieldHead(
                post_backbone_channels,
                len(self.atmospheric_channel_indices),
                hidden=pretext.head_hidden,
            )

    def attach_history_projection(self, projection: nn.Conv2d) -> None:
        """Register a zero-initialized history contribution at the temporal LR.

        The inherited widened projection remains in the spatial encoder group.
        With encoder freezing, its original current and zero historical slices
        are excluded from the optimizer, including AdamW decay and momentum.
        """
        if projection.groups != 1:
            raise ValueError("Native-pair history projection requires an ungrouped patch embedding.")
        n_history = projection.in_channels - self.predictor_channels_per_timestamp
        self.history_projection = nn.Conv2d(
            n_history, projection.out_channels, projection.kernel_size,
            stride=projection.stride, padding=projection.padding,
            dilation=projection.dilation, groups=1, bias=False,
            padding_mode=projection.padding_mode,
            device=projection.weight.device, dtype=projection.weight.dtype,
        )
        nn.init.zeros_(self.history_projection.weight)
        self._history_hook = projection.register_forward_hook(self._add_history_projection)

    def _add_history_projection(self, module, inputs, output):
        history = inputs[0][:, :-self.predictor_channels_per_timestamp]
        return output + self.history_projection(history)

    def remove_history_projection(self) -> None:
        if self._history_hook is not None:
            self._history_hook.remove()
            self._history_hook = None

    def atmospheric_target(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float):
        indices = list(self.atmospheric_channel_indices)
        target = normalize_predictors(x, mu, sigma, eps=eps)[:, indices]
        valid = torch.isfinite(target)
        for channel, mask_index in enumerate(self.atmospheric_mask_indices):
            if mask_index is not None:
                valid[:, channel] &= torch.isfinite(x[:, mask_index]) & (x[:, mask_index] > 0.5)
        return target, valid

    # -- state interface (this pathway is stateless) -----------------------
    # ``TemporalSequenceModel`` calls these generically. A finite-history model
    # carries no hidden state, so they are total no-ops rather than raising:
    # a caller that carries "state" across chunks must get correct behaviour,
    # and for this pathway correct behaviour is "there is nothing to carry".
    def init_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def zero_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def detach_state(self, state: Any = None) -> None:
        return None

    def reset_state(self, state: Any = None, *args: Any, **kwargs: Any) -> None:
        return None

    def structurally_dead_parameters(self) -> list[str]:
        """Parameter names that cannot receive gradient, with the reason implied.

        Returned relative to the adapter (prefix ``temporal_adapter.`` for
        state-dict names). Currently one case: with a zero main lead time, the
        lead-time embedding's weight has ``dL/dW = dL/d(lt) * lead_time = 0``. It
        becomes trainable as soon as the auxiliary transition objective is enabled,
        because that pass supplies a positive lead.

        Exposed so the gradient-flow check can assert "exactly the documented set
        is dead" rather than the weaker "nothing is dead", which would either fail
        spuriously or have to be relaxed into meaninglessness.
        """
        dead: list[str] = []
        if self.time_conditioning is not None and not self.pair_cfg.pretext.transition_enabled:
            dead.append("time_conditioning.lead_time_embedding.weight")
        return dead

    def apply_tokens(self, tokens: torch.Tensor, ctx: Mapping[str, Any]) -> torch.Tensor:
        """Hook invoked on the token tensor immediately before the transformer."""
        if self.time_conditioning is None:
            return tokens
        input_time = ctx.get("native_input_time_hours")
        lead_time = ctx.get("native_lead_time_hours")
        if input_time is None or lead_time is None:
            raise NativeTimeConditioningError(
                "native time conditioning is enabled but the sequence context carries no "
                "native_input_time_hours / native_lead_time_hours. The pathway must not "
                "silently fall back to a fabricated interval."
            )
        return self.time_conditioning(tokens, input_time, lead_time)


# ---------------------------------------------------------------------------
# input construction
# ---------------------------------------------------------------------------
class MissingHistoryError(IndexError):
    """Raised when a needed history frame exists in the run but not in the batch.

    Distinct from a genuine run start. This one always indicates a bug in
    windowing or chunking, and must never be papered over by clamping the index:
    doing so would silently substitute the wrong date and make chunked inference
    disagree with a single pass.
    """


def build_pair_input(
    x_window: torch.Tensor,
    t: int,
    *,
    history_offsets: Sequence[int],
    history_mode: str = "real",
    position_offset: int = 0,
    allow_cold_start: bool = False,
) -> torch.Tensor:
    """Stack the history dates and date ``t`` along the channel axis.

    ``x_window`` is ``[B, T, C, H, W]``; the result is ``[B, n_ts * C, H, W]``
    ordered **oldest first**, matching the native
    ``[batch, time, parameter, lat, lon] -> [batch, time x parameter, ...]``
    flatten order, so channel block ``k`` is timestamp ``k``.

    ``position_offset`` is the absolute index of ``x_window[:, 0]`` within its
    contiguous run, which is what lets the two genuinely different failure modes
    be told apart:

    * ``absolute index < 0`` -- the date does not exist, because ``t`` is at the
      very start of a run. This is the **cold start**, and with
      ``allow_cold_start`` it repeats date ``t`` into the history slot: the
      honest "no history available" input, and the exact analogue of the
      recurrent backends' zero initial state. It keeps the emitted date set
      identical to the frame-independent baseline, which a matched comparison
      requires.
    * ``absolute index >= 0`` but the frame is not in this batch -- the date
      exists and was simply not supplied. That is a chunking bug and raises
      :class:`MissingHistoryError`.

    ``history_mode='duplicate_current'`` is the capacity control: it substitutes
    date ``t`` into every history slot at every position. The tensor shape, the
    parameter count and the time metadata are unchanged, but the history carries
    no information the frame-independent model did not already have.
    """
    if x_window.dim() != 5:
        raise ValueError(
            f"build_pair_input expects x with shape [B, T, C, H, W], got {tuple(x_window.shape)}."
        )
    offsets = sorted(int(o) for o in history_offsets)
    if not offsets:
        return x_window[:, t]
    oldest_first = list(reversed(offsets))  # largest offset first == oldest first
    parts: list[torch.Tensor] = []
    for offset in oldest_first:
        if history_mode == "duplicate_current":
            parts.append(x_window[:, t])
            continue
        local_index = t - offset
        absolute_index = int(position_offset) + t - offset
        if absolute_index < 0:
            if not allow_cold_start:
                raise MissingHistoryError(
                    f"frame {t} (absolute {int(position_offset) + t}) needs history at offset "
                    f"{offset}, which precedes the start of the run. Increase "
                    "temporal.warmup_length so every supervised frame has real history; "
                    "cold starts are permitted only at inference."
                )
            parts.append(x_window[:, t])
            continue
        if local_index < 0:
            raise MissingHistoryError(
                f"frame {t} needs history at offset {offset} (absolute index "
                f"{absolute_index}), which exists in the run but was not supplied in this "
                f"batch of {x_window.shape[1]} frames. The caller must extend the chunk "
                f"backwards by at least {max(offsets)} frame(s); clamping the index would "
                "substitute the wrong date and break chunk/single-pass agreement."
            )
        parts.append(x_window[:, local_index])
    parts.append(x_window[:, t])
    return torch.cat(parts, dim=1)


def pair_time_scalars(
    interval_ratio: torch.Tensor | None,
    t: int,
    *,
    history_offsets: Sequence[int],
    cadence_days: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    lead_steps: int = 0,
    position_offset: int = 0,
    allow_cold_start: bool = False,
    cold_start_time_mode: str = "legacy_nominal",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_time_hours, lead_time_hours)`` for one frame.

    ``input_time`` is the span between the oldest and the newest input state, in
    hours, and is **nonnegative** -- the convention of the tensor the native model
    actually receives (``SampleSpec.input_time = (inputs[1] - inputs[0]) / 3600``
    with ``inputs[1]`` the later state), which is the opposite sign to the
    ``input_times`` *constructor* argument in the upstream dataloader.

    Measured from ``interval_ratio`` rather than assumed, so a step across a
    removed 29 February is charged as the two days it really is. Windows are cut
    at every discontinuity, so within a window the sum is exactly the cadence,
    but the arithmetic does not rely on that.

    ``lead_time`` is 0 for the downscaling task and positive only for the
    auxiliary transition pass.

    Only explicit inference cold starts may lack selected historical dates.
    ``legacy_nominal`` preserves old checkpoints: if any selected date precedes
    the true run start, use the full configured span as a *synthetic* cold-start
    condition. ``selected_timestamps`` instead measures the oldest available
    selected history date; missing slots duplicate current and contribute zero
    elapsed time. Neither mode permits losing history inside an observed run.
    The duplicate-current capacity control retains the same metadata as real
    history by design; it does not redefine the selected-date schedule here.
    """
    if cold_start_time_mode not in {"legacy_nominal", "selected_timestamps"}:
        raise ValueError(f"Unknown cold_start_time_mode: {cold_start_time_mode!r}")
    offsets = sorted(int(o) for o in history_offsets)
    span_steps = max(offsets) if offsets else 0
    absolute_position = int(position_offset) + int(t)
    cold_start = allow_cold_start and span_steps > absolute_position
    synthetic_cold_time = cold_start and cold_start_time_mode == "legacy_nominal"
    if cold_start and cold_start_time_mode == "selected_timestamps":
        span_steps = max((offset for offset in offsets if offset <= absolute_position), default=0)
    if interval_ratio is not None and span_steps > 0 and not synthetic_cold_time:
        start = t - span_steps + 1
        if start < 0:
            raise IndexError(
                f"cannot measure the input interval for frame {t} with span {span_steps}: "
                f"index {start} is outside the window."
            )
        # interval_ratio[:, k] is dt into frame k, in units of the cadence.
        span_days = interval_ratio[:, start : t + 1].sum(dim=1) * cadence_days
        input_time = span_days.to(device=device, dtype=dtype) * HOURS_PER_DAY
    else:
        input_time = torch.full(
            (batch_size,), span_steps * cadence_days * HOURS_PER_DAY, device=device, dtype=dtype
        )
    lead_time = torch.full(
        (batch_size,), float(lead_steps) * cadence_days * HOURS_PER_DAY, device=device, dtype=dtype
    )
    return input_time, lead_time


# ---------------------------------------------------------------------------
# checkpoint adaptation
# ---------------------------------------------------------------------------
#: The single tensor whose shape depends on ``n_input_timestamps``.
PATCH_EMBED_WEIGHT_KEY = "embedding.proj.weight"


def expand_patch_embed_weight(
    weight: torch.Tensor,
    *,
    n_timestamps: int,
) -> torch.Tensor:
    """Widen a single-timestamp patch-embedding weight to ``n_timestamps``.

    ``weight`` is ``[out, C, kh, kw]``; the result is ``[out, n_timestamps * C,
    kh, kw]`` with the **last** channel block -- timestamp ``t``, because the
    native flatten order is oldest-first -- carrying the pretrained weights and
    every history block set to **zero**.

    This is the whole input adaptation, and it has two properties worth stating
    because they are what make the comparison fair:

    * *Baseline-preserving.* At initialization the convolution computes
      ``0 * x_history + W_pretrained * x_t``, i.e. exactly the frame-independent
      model's shallow features. The paired model therefore departs from the
      established spatial skill rather than from a perturbation of it. It is
      equal to the frame-independent result up to float32 summation order only
      (measured ``3e-7`` relative; exactly ``0.0`` in float64), because widening
      a convolution's input channel count changes the order in which cuDNN
      accumulates the products. It is *not* bit-exact, and is not claimed to be
      -- unlike the ``adapter_init_gate: 0`` path, which is.
    * *Immediately trainable.* A zero **weight** is not a zero **gradient**:
      ``dL/dW_history = dL/dout * x_history``, which is non-zero from step 1. The
      zero-init trap that applies to a multiplicative gate does not apply here.
    """
    if weight.dim() != 4:
        raise ValueError(
            f"{PATCH_EMBED_WEIGHT_KEY} must be 4-D [out, C, kh, kw], got {tuple(weight.shape)}."
        )
    n_timestamps = int(n_timestamps)
    if n_timestamps < 1:
        raise ValueError(f"n_timestamps must be >= 1, got {n_timestamps}.")
    if n_timestamps == 1:
        return weight
    out_c, in_c, kh, kw = weight.shape
    expanded = weight.new_zeros((out_c, in_c * n_timestamps, kh, kw))
    expanded[:, in_c * (n_timestamps - 1) :] = weight
    return expanded


def adapt_state_dict_for_pairs(
    state: Mapping[str, torch.Tensor],
    *,
    n_timestamps: int,
    model_weight_shape: tuple[int, ...] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return ``(adapted_state, adapted_keys)`` for a paired-input model.

    Only ``embedding.proj.weight`` is touched, and only when the checkpoint
    carries the single-timestamp shape while the model expects the wider one.
    The adapted keys are returned so the caller can report them as *explicitly
    adapted* rather than silently loaded -- a widened tensor is not the same
    thing as a tensor that matched.
    """
    adapted = dict(state)
    changed: list[str] = []
    weight = adapted.get(PATCH_EMBED_WEIGHT_KEY)
    if weight is None or not torch.is_tensor(weight):
        return adapted, changed
    if model_weight_shape is not None and tuple(weight.shape) == tuple(model_weight_shape):
        return adapted, changed
    expanded = expand_patch_embed_weight(weight, n_timestamps=n_timestamps)
    if tuple(expanded.shape) == tuple(weight.shape):
        return adapted, changed
    if model_weight_shape is not None and tuple(expanded.shape) != tuple(model_weight_shape):
        raise ValueError(
            f"{PATCH_EMBED_WEIGHT_KEY}: expanding the checkpoint tensor "
            f"{tuple(weight.shape)} to {n_timestamps} timestamps gives "
            f"{tuple(expanded.shape)}, but the model expects {tuple(model_weight_shape)}. "
            "Check that data.n_input_timestamps matches "
            "temporal.native_pair.history_offsets."
        )
    adapted[PATCH_EMBED_WEIGHT_KEY] = expanded
    changed.append(PATCH_EMBED_WEIGHT_KEY)
    return adapted, changed


# ---------------------------------------------------------------------------
# masking for the reconstruction pretext task
# ---------------------------------------------------------------------------
def sample_mask_units(
    batch_size: int,
    grid: tuple[int, int],
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample a ``[B, gh, gw]`` boolean mask-unit mask; ``True`` means masked.

    A fixed *count* per sample (``round(ratio * n_units)``), not an independent
    Bernoulli draw per unit, which is what upstream does -- it masks a fixed
    number of global mask units so every sample presents the encoder the same
    amount of context.
    """
    gh, gw = int(grid[0]), int(grid[1])
    n_units = gh * gw
    n_masked = int(round(float(mask_ratio) * n_units))
    n_masked = max(0, min(n_units, n_masked))
    mask = torch.zeros((batch_size, n_units), dtype=torch.bool, device=device)
    if n_masked == 0:
        return mask.view(batch_size, gh, gw)
    scores = torch.rand((batch_size, n_units), generator=generator, device=device)
    idx = scores.argsort(dim=1)[:, :n_masked]
    mask.scatter_(1, idx, True)
    return mask.view(batch_size, gh, gw)


def mask_units_to_pixels(
    mask: torch.Tensor,
    *,
    block: tuple[int, int],
    height: int,
    width: int,
) -> torch.Tensor:
    """Upsample a ``[B, gh, gw]`` unit mask to a ``[B, 1, H, W]`` pixel mask."""
    expanded = mask[:, None].to(torch.float32)
    expanded = F.interpolate(expanded, scale_factor=(int(block[0]), int(block[1])), mode="nearest")
    if expanded.shape[-2] != height or expanded.shape[-1] != width:
        expanded = expanded[..., :height, :width]
    return expanded > 0.5


def apply_input_mask(
    x: torch.Tensor,
    pixel_mask: torch.Tensor,
    *,
    fill: torch.Tensor,
    n_timestamps: int,
) -> torch.Tensor:
    """Replace masked pixels with ``fill`` in **every** input timestamp.

    ``x`` is ``[B, n_timestamps * C, H, W]`` and ``fill`` is the per-parameter
    input mean, broadcast to ``[1, C, 1, 1]``, so a masked cell normalizes to
    exactly zero -- the same value a dropped token contributes upstream.

    Masking every timestamp is what makes the task honest. Two leakage paths
    close only if this is done:

    * the *history* channels hold the same variables one day earlier, and daily
      fields are strongly autocorrelated, so masking only date ``t`` would leave
      a near-copy of the answer in the input;
    * the U-Net skip pyramid is built from ``self.embedding(x)`` on this same
      tensor, so a mask applied only to the transformer's input would leave the
      masked region fully visible to the decoder through the skips.

    Upstream has the identical property for the identical reason: it drops whole
    mask units of *tokens*, and a token already contains both timestamps because
    the time axis was folded into channels before embedding.
    """
    if fill.dim() != 4:
        raise ValueError(f"fill must be [1, C, 1, 1], got {tuple(fill.shape)}.")
    per_ts = fill.shape[1]
    if x.shape[1] != per_ts * int(n_timestamps):
        raise ValueError(
            f"x has {x.shape[1]} channels but fill implies {per_ts} per timestamp with "
            f"{n_timestamps} timestamps ({per_ts * int(n_timestamps)})."
        )
    fill_full = fill.repeat(1, int(n_timestamps), 1, 1).to(device=x.device, dtype=x.dtype)
    return torch.where(pixel_mask.bool(), fill_full, x)


def masked_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error over valid masked pixels only.

    Scoring the *unmasked* pixels too would let the term be driven down by
    copying visible input, which is not the quantity of interest.
    """
    valid = pixel_mask.bool().expand_as(target) & torch.isfinite(target)
    if valid_mask is not None:
        valid = valid & valid_mask.bool().expand_as(target)
    error = (prediction.masked_select(valid) - target.masked_select(valid)).abs()
    return error.sum() / valid.sum().clamp_min(1)


class PretextTerms(dict):
    """Auxiliary loss terms, keyed by name. A plain dict so it logs directly."""

    @property
    def total(self) -> torch.Tensor | None:
        values = [v for v in self.values() if torch.is_tensor(v)]
        if not values:
            return None
        out = values[0]
        for v in values[1:]:
            out = out + v
        return out


def compute_pretext_losses(
    base: Any,
    adapter: NativePairAdapter,
    batch: Mapping[str, Any],
    t: int,
    cfg: TemporalConfig,
    *,
    generator: torch.Generator | None = None,
    prediction_sink: dict[str, torch.Tensor] | None = None,
) -> tuple[PretextTerms, int]:
    """Run the pretraining-aligned auxiliary passes for one frame.

    Returns ``(weighted_terms, n_base_forwards)``. Both objectives share the
    encoder, the transformer and the post-backbone convolution with the
    downscaling path, so their gradients reach the representation the main task
    uses -- which is the whole point. Only the small output heads are private.

    Applied to a **single** frame per window (the caller passes the last emitted
    one) rather than to all of them, because each objective costs a full extra
    forward pass. For the SA geometry, plain native inference uses five backbone
    calls per training window and the two enabled objectives raise this to seven.
    Recurrent backends also use seven calls, but equal call counts do not imply
    equal FLOPs, retained activations, memory, or runtime. Report these counts
    alongside measured resource use; they do not establish matched compute.
    """
    pretext = cfg.native_pair.pretext
    terms = PretextTerms()
    n_forward = 0
    if not pretext.any_enabled:
        return terms, n_forward

    x_window = batch["x"]
    n_ts = cfg.native_pair.n_input_timestamps
    per_ts = adapter.predictor_channels_per_timestamp
    device = x_window.device
    dtype = x_window.dtype
    b = x_window.shape[0]
    height, width = x_window.shape[-2], x_window.shape[-1]

    mu, sigma = base._resolve_input_scalers(
        x_window[:, t],
        scaler_offset=batch.get("__input_scaler_offset", batch.get("__scaler_offset")),
    )
    mu, sigma = mu.to(device=device, dtype=dtype), sigma.to(device=device, dtype=dtype)
    eps = float(base.input_scalers_epsilon)
    if mu.shape[1] != per_ts:
        raise ValueError(
            f"input_scalers_mu has {mu.shape[1]} channels but the model expects {per_ts} "
            "predictor channels per timestamp. The input scalers are per-parameter and are "
            "broadcast over the native time axis; they must not be tiled per timestamp."
        )

    def _static(frame: dict[str, Any]) -> dict[str, Any]:
        for key in ("static_x", "static_y"):
            if key in batch:
                frame[key] = batch[key]
        for key in ("__output_crop", "__input_scaler_offset", "__output_scaler_offset"):
            if key in batch:
                frame[key] = batch[key]
        return frame

    def _run(x_in: torch.Tensor, lead_steps: int, source_t: int):
        """One shared-trunk forward, returning the post-backbone feature map."""
        input_time, lead_time = pair_time_scalars(
            batch.get("interval_ratio"),
            source_t,
            history_offsets=cfg.native_pair.history_offsets,
            cadence_days=cfg.cadence_days,
            batch_size=b,
            device=device,
            dtype=dtype,
            lead_steps=lead_steps,
        )
        previous = base._temporal_ctx
        previous_capture = getattr(base, "_capture_features", ())
        base.set_feature_capture(("prithvi",))
        base._temporal_ctx = {
            "state": None,
            "time_features": None,
            "interval_ratio": None,
            "reset_mask": None,
            "capture_latent": False,
            "native_input_time_hours": input_time,
            "native_lead_time_hours": lead_time,
        }
        try:
            frame = {"x": x_in}
            if "__output_shape" in batch:
                frame["__output_shape"] = batch["__output_shape"]
            elif "y" in batch:
                frame["__output_shape"] = tuple(batch["y"].shape[-2:])
            base(_static(frame))
            features = base.get_last_phase1_features().get("prithvi")
        finally:
            base._temporal_ctx = previous
            base.set_feature_capture(previous_capture or None)
        if features is None:
            raise RuntimeError(
                "the post-backbone feature map was not captured; the auxiliary heads have "
                "nothing to attach to."
            )
        return features

    # -- A. masked atmospheric reconstruction ------------------------------
    if pretext.masked_reconstruction_enabled and adapter.recon_head is not None:
        block = tuple(int(v) for v in base.mask_unit_size_px_backbone)
        grid = (height // block[0], width // block[1])
        unit_mask = sample_mask_units(
            b, grid, pretext.mask_ratio, generator=generator, device=device
        )
        pixel_mask = mask_units_to_pixels(unit_mask, block=block, height=height, width=width)

        x_pair = build_pair_input(
            x_window,
            t,
            history_offsets=cfg.native_pair.history_offsets,
            history_mode=cfg.native_pair.history_mode,
            position_offset=int(batch.get("__position_offset", 0) or 0),
            allow_cold_start=False,
        )
        # Masked in EVERY timestamp, so the answer is absent from the history
        # channels and from the U-Net skips as well as from the transformer input.
        x_masked = apply_input_mask(x_pair, pixel_mask, fill=mu, n_timestamps=n_ts)
        features = _run(x_masked, lead_steps=0, source_t=t)
        n_forward += 1
        prediction = adapter.recon_head(features)
        target, valid = adapter.atmospheric_target(x_window[:, t], mu, sigma, eps)
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if prediction_sink is not None:
            prediction_sink["pretext_masked_reconstruction"] = prediction
        terms["pretext_masked_reconstruction"] = (
            pretext.masked_reconstruction_weight
            * masked_field_loss(prediction, target, pixel_mask, valid)
        )

    # -- B. atmospheric transition prediction ------------------------------
    if pretext.transition_enabled and adapter.transition_head is not None:
        lead = int(pretext.transition_lead_steps)
        source_t = t - lead
        # Inputs end at ``source_t`` and the verification state is at ``t``, so
        # every input date is strictly earlier than the target. The later state is
        # never handed to this pass -- that is the property
        # ``test_transition_pass_does_not_see_its_target`` asserts numerically.
        x_pair = build_pair_input(
            x_window,
            source_t,
            history_offsets=cfg.native_pair.history_offsets,
            history_mode=cfg.native_pair.history_mode,
            position_offset=int(batch.get("__position_offset", 0) or 0),
            allow_cold_start=False,
        )
        features = _run(x_pair, lead_steps=lead, source_t=source_t)
        n_forward += 1
        prediction = adapter.transition_head(features)
        target, valid = adapter.atmospheric_target(x_window[:, t], mu, sigma, eps)
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if prediction_sink is not None:
            prediction_sink["pretext_transition"] = prediction
        terms["pretext_transition"] = pretext.transition_weight * masked_field_loss(
            prediction, target, torch.ones_like(valid), valid
        )

    return terms, n_forward


def normalize_predictors(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Normalize a single-timestamp predictor field with the model's own scalers.

    The auxiliary targets live in this space, not in physical units, so the two
    auxiliary losses are dimensionless and commensurate across the 15 predictor
    channels (whose physical units span ``m s-1``, ``K``, ``m`` and
    dimensionless specific humidity). Reconstructing physical values directly
    would let geopotential height dominate the term by five orders of magnitude.
    """
    return (x - mu) / (sigma + eps)
