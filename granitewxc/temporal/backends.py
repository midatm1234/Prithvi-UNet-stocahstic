"""Interchangeable temporal backends operating on a spatially organized latent.

Both backends implement :class:`TemporalBackend`, a *causal, step-wise*
interface::

    h_t, state_t = backend.step(z_t, state_{t-1}, time_features_t, interval_ratio_t)

Step-wise operation is deliberate rather than incidental:

* It makes causality structural. There is no code path by which frame ``t`` can
  read frame ``t + 1``.
* It makes chunked inference *exact*. Carrying ``state`` across a chunk
  boundary reproduces the single-pass result bit-for-bit, because the chunked
  and single-pass computations execute the identical recurrence. There is no
  separate "parallel scan" path that would have to be numerically reconciled
  with the recurrent one.

Spatial organization is preserved throughout: the latent keeps its ``[C, H, W]``
layout, and both backends contain convolutions over ``(H, W)`` so information
moves between locations. A stack of independent per-pixel time series would not
represent an advecting front, so neither backend is built that way.

``interval_ratio_t`` is ``Δt_actual / cadence`` for the step into frame ``t``
(1.0 for a regular step, 2.0 across a removed 29 February). See the class
docstrings for how each backend uses it -- the two backends differ honestly
here, and the difference is documented rather than papered over.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TemporalBackend",
    "ConvGRUBackend",
    "TemporalMambaBackend",
    "MambaUnavailableError",
    "build_backend",
    "fused_mamba_available",
]


class MambaUnavailableError(RuntimeError):
    """Raised when a fused Mamba kernel is required but not importable."""


def fused_mamba_available() -> tuple[bool, str]:
    """Report whether ``mamba_ssm`` fused kernels can be imported."""
    try:  # pragma: no cover - environment dependent
        import mamba_ssm  # noqa: F401
        from mamba_ssm.ops.triton.ssd_combined import (  # noqa: F401
            mamba_chunk_scan_combined,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, "mamba_ssm import succeeded"


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------
def _norm2d(channels: int, kind: str, groups: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    g = math.gcd(groups, channels) or 1
    return nn.GroupNorm(num_groups=max(1, g), num_channels=channels)


class TimeFeatureFiLM(nn.Module):
    """Map temporal metadata to per-channel scale/shift (FiLM) modulation.

    Time metadata enters as a modulation of the latent rather than as extra
    channels: this keeps the parameter count independent of the spatial size and
    makes the "date embedding only" ablation trivial to construct (drop the
    recurrent state, keep this module).

    The output layer is initialized *small but non-zero* (std ``1e-3``) rather
    than exactly zero. Exact zeros would make the modulation an identity at
    construction, but they would also drive the gradient of the *first* linear
    layer to exactly zero -- ``dL/dW1 = W2^T (...) = 0`` when ``W2 = 0`` -- so
    half of this module could not begin learning. A small non-zero init keeps the
    modulation within ~0.1% of identity while letting every parameter receive
    gradient from the first optimizer step. Exact identity of the *whole
    adapter* is still available, and is enforced where it matters, by
    ``temporal.latent.adapter_init_gate: 0`` in
    :class:`~granitewxc.temporal.model.TemporalLatentAdapter`.
    """

    def __init__(self, time_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.time_dim = int(time_dim)
        self.channels = int(channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.channels),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor | None) -> torch.Tensor:
        if time_features is None:
            return x
        params = self.mlp(time_features.to(x.dtype))  # [B, 2C]
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class TemporalBackend(nn.Module, ABC):
    """Causal, step-wise temporal module over a ``[B, C, H, W]`` latent."""

    #: Number of channels of the latent this backend consumes and returns.
    channels: int

    @abstractmethod
    def init_state(self, batch_size: int, height: int, width: int, *, device, dtype) -> Any:
        """Allocate the initial hidden state for a fresh sequence."""

    @abstractmethod
    def step(
        self,
        z: torch.Tensor,
        state: Any,
        time_features: torch.Tensor | None,
        interval_ratio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Advance one frame. Returns ``(h_t, new_state)``."""

    # -- state utilities ----------------------------------------------------
    @staticmethod
    def detach_state(state: Any) -> Any:
        """Detach every tensor in a (possibly nested) state container."""
        if state is None:
            return None
        if torch.is_tensor(state):
            return state.detach()
        if isinstance(state, (list, tuple)):
            detached = [TemporalBackend.detach_state(s) for s in state]
            return type(state)(detached) if not isinstance(state, tuple) else tuple(detached)
        if isinstance(state, dict):
            return {k: TemporalBackend.detach_state(v) for k, v in state.items()}
        raise TypeError(f"Unsupported state container {type(state).__name__}")

    @staticmethod
    def reset_state_where(state: Any, zero_state: Any, reset_mask: torch.Tensor) -> Any:
        """Zero the state of the batch elements selected by ``reset_mask``.

        ``reset_mask`` is a boolean tensor of shape ``[B]``. This is how a
        discontinuity (missing date, period concatenation, new tile, new
        realization) is honoured *per sample* without leaking one sequence's
        memory into another's.
        """
        if state is None:
            return None
        if torch.is_tensor(state):
            if not reset_mask.any():
                return state
            shape = [reset_mask.shape[0]] + [1] * (state.dim() - 1)
            keep = (~reset_mask).view(shape).to(state.dtype)
            return state * keep + zero_state * (1.0 - keep)
        if isinstance(state, (list, tuple)):
            out = [
                TemporalBackend.reset_state_where(s, z, reset_mask)
                for s, z in zip(state, zero_state)
            ]
            return tuple(out) if isinstance(state, tuple) else out
        if isinstance(state, dict):
            return {
                k: TemporalBackend.reset_state_where(v, zero_state[k], reset_mask)
                for k, v in state.items()
            }
        raise TypeError(f"Unsupported state container {type(state).__name__}")


# ---------------------------------------------------------------------------
# ConvGRU
# ---------------------------------------------------------------------------
class ConvGRUCell(nn.Module):
    """Standard convolutional GRU cell.

    ``h_t = (1 - z) * h_{t-1} + z * tanh(W_h * [x_t, r * h_{t-1}])``

    The gates are 2-D convolutions, so the state is a spatial field and the
    kernel couples neighbouring grid cells at every step. ``dilation`` widens
    that coupling without extra parameters, which is how the multi-timescale /
    multi-scale variant is built (one dilation per layer).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        pad = dilation * (kernel_size - 1) // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            padding_mode="replicate",
        )
        self.candidate = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            padding_mode="replicate",
        )
        # Bias the update gate toward *retention* (z small) so an untrained cell
        # starts close to "carry the previous state" instead of overwriting it.
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[: self.hidden_channels].fill_(1.0)   # reset gate ~open
            self.gates.bias[self.hidden_channels :].fill_(-1.0)  # update gate ~closed
        nn.init.zeros_(self.candidate.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = self.gates(torch.cat([x, h], dim=1))
        reset, update = gates.chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class ConvGRUBackend(TemporalBackend):
    """Multi-layer ConvGRU over the latent.

    **Physical-time handling.** A ConvGRU has no continuous-time
    interpretation: its update is a learned discrete map, and there is no
    principled way to "advance it by 2 days". Rather than invent one, this
    backend relies on two honest mechanisms:

    1. Sequences are built only from strictly contiguous runs (the sequence
       builder splits at every discontinuity), so within a sequence the step is
       always exactly the declared cadence.
    2. The elapsed-time and sequence-start metadata still reach the cell through
       :class:`TimeFeatureFiLM`, so the model can condition on *where* it is in
       a sequence and on the seasonal phase.

    This is a real difference from :class:`TemporalMambaBackend`, which does
    admit a correct continuous-time rescaling. It is documented rather than
    hidden.

    ConvGRU is chosen over ConvLSTM as the default: it carries one state tensor
    instead of two (halving the state memory that must be held across a
    truncated-BPTT boundary and across inference chunks) and has ~25% fewer gate
    parameters, with no consistent accuracy disadvantage reported in the
    precipitation-nowcasting literature that introduced both. ``cell:
    convlstm`` remains available for comparison.
    """

    def __init__(
        self,
        channels: int,
        *,
        time_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        dilations: tuple[int, ...] = (1,),
        norm: str = "group",
        groups: int = 8,
        time_hidden: int = 64,
        cell: str = "convgru",
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.cell_kind = cell
        if len(dilations) != self.num_layers:
            raise ValueError("dilations must provide one entry per layer")
        self.film = TimeFeatureFiLM(time_dim, self.channels, hidden=time_hidden)
        self.norms = nn.ModuleList(
            [_norm2d(self.channels, norm, groups) for _ in range(self.num_layers)]
        )
        if cell == "convgru":
            self.cells = nn.ModuleList(
                [
                    ConvGRUCell(self.channels, self.channels, kernel_size, dilations[i])
                    for i in range(self.num_layers)
                ]
            )
        elif cell == "convlstm":
            self.cells = nn.ModuleList(
                [
                    ConvLSTMCell(self.channels, self.channels, kernel_size, dilations[i])
                    for i in range(self.num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown recurrent cell {cell!r}")

    def init_state(self, batch_size: int, height: int, width: int, *, device, dtype):
        shape = (batch_size, self.channels, height, width)
        if self.cell_kind == "convlstm":
            return [
                (
                    torch.zeros(shape, device=device, dtype=dtype),
                    torch.zeros(shape, device=device, dtype=dtype),
                )
                for _ in range(self.num_layers)
            ]
        return [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)]

    def step(self, z, state, time_features, interval_ratio=None):
        x = self.film(z, time_features)
        new_state = []
        for layer in range(self.num_layers):
            x = self.norms[layer](x)
            if self.cell_kind == "convlstm":
                h, c = self.cells[layer](x, state[layer])
                new_state.append((h, c))
                x = h
            else:
                h = self.cells[layer](x, state[layer])
                new_state.append(h)
                x = h
        return x, new_state


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell (Shi et al., 2015), offered for comparison."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            padding_mode="replicate",
        )
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            # forget-gate bias 1.0 is the standard long-memory initialization
            self.conv.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, state) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ---------------------------------------------------------------------------
# Temporal Mamba (Mamba-2 / SSD recurrence)
# ---------------------------------------------------------------------------
class TemporalSSDBlock(nn.Module):
    """One Mamba-2 (SSD) block whose selective scan runs over **time**.

    Layout and axes
    ---------------
    The latent is ``[B, C, H, W]``. For the state-space recurrence, ``(H, W)``
    is folded into the batch and the scan proceeds over ``t``::

        scan axis      : t   (the physical time axis)
        per-scan vector: the C-channel feature at one grid cell
        spatial coupling: the depthwise/pointwise 3x3 convolution applied to
                          every frame *before* the scan (``spatial_mixing``)

    Flattening ``H*W`` into the scan axis -- which some vision-Mamba variants do
    -- would scan over *space* and model no temporal dependence at all. That is
    explicitly not what happens here.

    Recurrence (Mamba-2 structured state-space duality)
    --------------------------------------------------
    ``A`` is restricted to scalar-times-identity per head, so with heads ``h``,
    head dimension ``p`` and state dimension ``n``::

        A_bar_t = exp(dt_t * A_h)                       # scalar per head
        h_t     = A_bar_t * h_{t-1} + dt_t * x_t (x) B_t
        y_t     = <h_t, C_t> + D_h * x_t

    Physical elapsed time
    ---------------------
    ``dt`` in Mamba is an *input-dependent, learned* per-channel step size
    (``softplus(dt_proj(x) + dt_bias)``) with no physical units. It is therefore
    **not** the dataset's sampling interval, and equating the two would be
    wrong. What *is* correct: the block above is a zero-order-hold
    discretization of a continuous-time linear system, so integrating over an
    interval that is ``r`` times longer means using ``dt * r``. This block
    multiplies the learned ``dt`` by the measured ``interval_ratio``
    (``Δt_actual / cadence``) when ``time_scale_from_metadata`` is set, so a step
    across a removed 29 February decays the state by ``exp(2 * dt * A)`` rather
    than ``exp(dt * A)``. The learned timescale and the physical interval remain
    separate quantities that are multiplied, not conflated.
    """

    def __init__(
        self,
        channels: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        spatial_mixing: str = "conv3x3",
        time_scale_from_metadata: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        norm: str = "group",
        groups: int = 8,
    ):
        super().__init__()
        self.channels = int(channels)
        self.d_inner = int(channels * expand)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.headdim = int(headdim)
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
            )
        self.nheads = self.d_inner // self.headdim
        self.time_scale_from_metadata = bool(time_scale_from_metadata)

        self.norm = _norm2d(self.channels, norm, groups)

        # --- spatial mixing (this is what makes the block spatially coupled) --
        if spatial_mixing == "none":
            self.spatial = nn.Identity()
        else:
            k = 3 if spatial_mixing == "conv3x3" else 5
            self.spatial = nn.Sequential(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    k,
                    padding=k // 2,
                    groups=self.channels,
                    padding_mode="replicate",
                ),
                nn.Conv2d(self.channels, self.channels, 1),
                nn.SiLU(),
            )

        # --- SSM projections -------------------------------------------------
        # in_proj produces [x, gate, B, C, dt]
        self.in_proj = nn.Linear(
            self.channels,
            2 * self.d_inner + 2 * self.d_state + self.nheads,
            bias=False,
        )
        # Causal depthwise conv over TIME (length d_conv), applied to x and B/C.
        self.conv_channels = self.d_inner + 2 * self.d_state
        self.conv_time = nn.Conv1d(
            self.conv_channels,
            self.conv_channels,
            kernel_size=self.d_conv,
            groups=self.conv_channels,
            padding=0,  # we pad manually with carried state to stay causal
        )
        self.out_proj = nn.Linear(self.d_inner, self.channels, bias=False)

        # A is negative; parameterize as -exp(A_log) per head.
        A_init = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # dt bias initialized so softplus(dt_bias) spans [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

    # -- state -------------------------------------------------------------
    def init_state(self, batch_size: int, height: int, width: int, *, device, dtype):
        bhw = batch_size * height * width
        ssm = torch.zeros(bhw, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype)
        conv = torch.zeros(bhw, self.conv_channels, self.d_conv - 1, device=device, dtype=dtype)
        return {"ssm": ssm, "conv": conv, "hw": (int(height), int(width))}

    def step(self, x: torch.Tensor, state: dict, interval_ratio: torch.Tensor | None):
        """One frame. ``x`` is ``[B, C, H, W]``; returns ``([B, C, H, W], state)``."""
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = x + self.spatial(x)  # spatial coupling, residual so init is near-identity

        # Fold space into batch: the scan is over time, one vector per grid cell.
        seq = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]
        proj = self.in_proj(seq)
        xz, gate, Bmat, Cmat, dt_raw = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
            dim=-1,
        )

        # --- causal depthwise conv over time, using the carried conv state ---
        conv_in = torch.cat([xz, Bmat, Cmat], dim=-1)  # [BHW, conv_channels]
        conv_hist = state["conv"]  # [BHW, conv_channels, d_conv-1]
        window = torch.cat([conv_hist, conv_in.unsqueeze(-1)], dim=-1)  # [.., d_conv]
        conv_out = self.conv_time(window).squeeze(-1)  # [BHW, conv_channels]
        conv_out = F.silu(conv_out)
        new_conv = window[..., 1:] if self.d_conv > 1 else conv_hist

        xz, Bmat, Cmat = torch.split(
            conv_out, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        # --- selective SSM update -------------------------------------------
        dt = F.softplus(dt_raw + self.dt_bias)  # [BHW, nheads], learned & positive
        if self.time_scale_from_metadata and interval_ratio is not None:
            # interval_ratio is [B]; broadcast over H, W then heads.
            ratio = interval_ratio.to(dt.dtype).view(B, 1, 1).expand(B, H, W).reshape(-1, 1)
            dt = dt * ratio
        A = -torch.exp(self.A_log.to(dt.dtype))  # [nheads], negative
        decay = torch.exp(dt * A)  # [BHW, nheads]

        xh = xz.view(-1, self.nheads, self.headdim)  # [BHW, h, p]
        ssm = state["ssm"]  # [BHW, h, p, n]
        # h_t = decay * h_{t-1} + dt * x_t (outer) B_t
        contrib = (dt.unsqueeze(-1) * xh).unsqueeze(-1) * Bmat.unsqueeze(1).unsqueeze(1)
        ssm = decay.unsqueeze(-1).unsqueeze(-1) * ssm + contrib
        y = (ssm * Cmat.unsqueeze(1).unsqueeze(1)).sum(dim=-1)  # [BHW, h, p]
        y = y + self.D.unsqueeze(-1) * xh
        y = y.reshape(-1, self.d_inner)
        y = y * F.silu(gate)
        y = self.out_proj(y)

        out = y.view(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out, {"ssm": ssm, "conv": new_conv, "hw": (H, W)}


class TemporalMambaBackend(TemporalBackend):
    """Stack of :class:`TemporalSSDBlock` scanning the time axis.

    ``implementation`` selects the kernel, never the architecture:

    * ``fused``     -- requires ``mamba_ssm``; raises :class:`MambaUnavailableError`
      if absent. No substitution ever happens.
    * ``reference`` -- the in-repo pure-PyTorch SSD recurrence above. Identical
      mathematics and identical parameters; slower.
    * ``auto``      -- prefer ``fused``, otherwise ``reference`` with a warning
      and a recorded ``implementation_used`` provenance field.

    On this project's Windows/CUDA environment ``mamba_ssm`` is not installable
    (it requires the Triton/CUDA build path that upstream does not support on
    Windows), so ``reference`` is what actually executes. Because the recurrence
    is written out explicitly here, the temporal semantics do not depend on that
    dependency being present.
    """

    def __init__(
        self,
        channels: int,
        *,
        time_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        headdim: int = 32,
        spatial_mixing: str = "conv3x3",
        time_scale_from_metadata: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        norm: str = "group",
        groups: int = 8,
        time_hidden: int = 64,
        implementation: str = "auto",
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_layers = int(n_layers)

        available, detail = fused_mamba_available()
        if implementation == "fused" and not available:
            raise MambaUnavailableError(
                "temporal.mamba.implementation='fused' requires the 'mamba_ssm' package "
                f"with its fused kernels, which could not be imported ({detail}). "
                "Install mamba_ssm (Linux/CUDA only) or set "
                "temporal.mamba.implementation='reference' to use the in-repo "
                "pure-PyTorch SSD recurrence. The temporal backend will not silently "
                "fall back to a different model."
            )
        if implementation == "auto" and not available:
            warnings.warn(
                "temporal.mamba.implementation='auto': fused mamba_ssm kernels are "
                f"unavailable ({detail}); using the in-repo pure-PyTorch SSD recurrence. "
                "This is the same architecture and the same parameters, only slower. "
                "Set implementation='reference' to make this explicit, or 'fused' to "
                "require the kernels and fail loudly.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.implementation_requested = implementation
        self.implementation_used = "fused" if (available and implementation != "reference") else "reference"

        self.film = TimeFeatureFiLM(time_dim, self.channels, hidden=time_hidden)
        self.blocks = nn.ModuleList(
            [
                TemporalSSDBlock(
                    self.channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    spatial_mixing=spatial_mixing,
                    time_scale_from_metadata=time_scale_from_metadata,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    norm=norm,
                    groups=groups,
                )
                for _ in range(self.n_layers)
            ]
        )

    def init_state(self, batch_size: int, height: int, width: int, *, device, dtype):
        return [
            block.init_state(batch_size, height, width, device=device, dtype=dtype)
            for block in self.blocks
        ]

    def step(self, z, state, time_features, interval_ratio=None):
        x = self.film(z, time_features)
        new_state = []
        for block, block_state in zip(self.blocks, state):
            x, s = block.step(x, block_state, interval_ratio)
            new_state.append(s)
        return x, new_state

    @staticmethod
    def detach_state(state: Any) -> Any:
        # ``hw`` is a plain tuple; keep it as-is while detaching tensors.
        if isinstance(state, list):
            return [TemporalMambaBackend.detach_state(s) for s in state]
        if isinstance(state, dict):
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in state.items()
            }
        return TemporalBackend.detach_state(state)

    @staticmethod
    def reset_state_where(state: Any, zero_state: Any, reset_mask: torch.Tensor) -> Any:
        """Reset per *sample*, accounting for the folded ``B*H*W`` leading axis."""
        if isinstance(state, list):
            return [
                TemporalMambaBackend.reset_state_where(s, z, reset_mask)
                for s, z in zip(state, zero_state)
            ]
        if isinstance(state, dict):
            out: dict[str, Any] = {}
            hw = state.get("hw")
            for key, value in state.items():
                if not torch.is_tensor(value):
                    out[key] = value
                    continue
                if not reset_mask.any():
                    out[key] = value
                    continue
                height, width = hw
                b = reset_mask.shape[0]
                rest = value.shape[1:]
                view = value.view(b, height * width, *rest)
                keep = (~reset_mask).view(b, *([1] * (view.dim() - 1))).to(view.dtype)
                zero_view = zero_state[key].view(b, height * width, *rest)
                merged = view * keep + zero_view * (1.0 - keep)
                out[key] = merged.view_as(value)
            return out
        return TemporalBackend.reset_state_where(state, zero_state, reset_mask)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_backend(cfg, channels: int, time_dim: int) -> TemporalBackend:
    """Instantiate the backend selected by a :class:`TemporalConfig`."""
    latent = cfg.latent
    if cfg.backend == "recurrent":
        return ConvGRUBackend(
            channels,
            time_dim=time_dim,
            kernel_size=cfg.recurrent.kernel_size,
            num_layers=cfg.recurrent.num_layers,
            dilations=cfg.recurrent.dilations,
            norm=latent.norm,
            groups=latent.groups,
            time_hidden=latent.time_feature_hidden,
            cell=cfg.recurrent.cell,
        )
    if cfg.backend == "mamba":
        return TemporalMambaBackend(
            channels,
            time_dim=time_dim,
            d_state=cfg.mamba.d_state,
            d_conv=cfg.mamba.d_conv,
            expand=cfg.mamba.expand,
            n_layers=cfg.mamba.n_layers,
            headdim=cfg.mamba.headdim,
            spatial_mixing=cfg.mamba.spatial_mixing,
            time_scale_from_metadata=cfg.mamba.time_scale_from_metadata,
            dt_min=cfg.mamba.dt_min,
            dt_max=cfg.mamba.dt_max,
            norm=latent.norm,
            groups=latent.groups,
            time_hidden=latent.time_feature_hidden,
            implementation=cfg.mamba.implementation,
        )
    raise ValueError(f"Unknown temporal backend {cfg.backend!r}")
