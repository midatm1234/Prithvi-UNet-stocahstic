"""Shared spatial backbones for the Phase-2 residual refiners.

Two backbones are provided and both map

    (noisy/interpolated residual field, conditioning field, process time)
        -> residual-shaped output field

on the **target latitude/longitude grid** of a single sample at a single
timestamp:

* :class:`ConditionalResidualUNet` -- a small FiLM-conditioned convolutional
  UNet.  Ported from the ``CORDEX_ML_diffusion_head`` score network and the
  Aurora ``ResidualFlowUNet``, generalised to arbitrary channel counts and
  rectangular domains.
* :class:`SpatialResidualTransformer` -- a DiT-style Transformer whose
  attention runs **only over 2-D spatial tokens**.

Explicitly absent, by design
----------------------------
* temporal attention / cross-time attention
* causal attention masks
* forecast lead-time embeddings
* temporal positional encodings
* recurrent or autoregressive state

The single scalar process-time input is the diffusion timestep or the flow
interpolation coordinate; it is injected through FiLM (UNet) or adaptive layer
normalisation (Transformer) and never as a spatial or temporal token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from granitewxc.refinement.schedules import ProcessTimeEmbedding

__all__ = [
    "ConditionalResidualUNet",
    "SpatialResidualTransformer",
    "sincos_2d_positional_encoding",
    "patchify_2d",
    "unpatchify_2d",
]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Convolutional backbone
# ---------------------------------------------------------------------------


class _FiLM(nn.Module):
    """Feature-wise linear modulation of a conv activation by the process time."""

    def __init__(self, time_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_emb.to(self.proj.weight.dtype)).to(h.dtype).chunk(2, dim=-1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class _SpatialSelfAttention2d(nn.Module):
    """Multi-head self-attention over the flattened spatial grid of one sample.

    Tokens are grid cells ``(lat, lon)`` of a single sample.  Batch elements are
    never mixed and there is no time axis.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"Bottleneck attention channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = int(num_heads)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads

        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.num_heads, head_dim, h * w).transpose(-2, -1)

        out = F.scaled_dot_product_attention(_heads(q), _heads(k), _heads(v))
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.film = _FiLM(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, t_emb)
        h = self.dropout(h)
        return self.act(self.norm2(self.conv2(h)))


class ConditionalResidualUNet(nn.Module):
    """Conditional convolutional UNet predicting a residual-shaped field.

    Args:
        in_channels: channels of the noisy/interpolated residual field.
        cond_channels: channels of the spatial conditioning stack.
        out_channels: channels of the predicted field (== residual channels).
        hidden_channels: width at the finest level.
        num_levels: number of resolution levels (>= 1).
        time_embedding_dim: width of the process-time embedding.
        dropout: dropout probability inside conv blocks.
        bottleneck_attention: enable spatial self-attention at the coarsest level.
        attention_heads: heads for the bottleneck attention.
        zero_init_output: zero-initialise the output projection so the refiner is
            the identity at construction (predicted residual == 0).

    The network is fully convolutional, so rectangular and non-power-of-two
    grids work.  Feature maps are padded to an even size before each
    downsampling step and cropped back on the way up, so the output height and
    width always equal the input height and width exactly.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 64,
        num_levels: int = 3,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
        bottleneck_attention: bool = True,
        attention_heads: int = 4,
        zero_init_output: bool = True,
        time_embedding_kind: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.num_levels = int(num_levels)

        self.time_embed = ProcessTimeEmbedding(time_embedding_dim, kind=time_embedding_kind)

        widths = [hidden_channels * (2**i) for i in range(num_levels)]
        self.down_blocks = nn.ModuleList()
        prev = in_channels + cond_channels
        for width in widths:
            self.down_blocks.append(_ConvBlock(prev, width, time_embedding_dim, dropout))
            prev = width

        self.bottleneck_attn = (
            _SpatialSelfAttention2d(widths[-1], attention_heads) if bottleneck_attention else None
        )

        self.up_blocks = nn.ModuleList()
        for level in range(num_levels - 1, 0, -1):
            self.up_blocks.append(
                _ConvBlock(widths[level] + widths[level - 1], widths[level - 1], time_embedding_dim, dropout)
            )

        self.out_proj = nn.Conv2d(widths[0], out_channels, 1)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Residual field {tuple(x.shape[-2:])} and conditioning "
                f"{tuple(cond.shape[-2:])} must share the spatial grid."
            )
        t_emb = self.time_embed(t)
        h = torch.cat([x, cond], dim=1)

        skips: list[torch.Tensor] = []
        for level, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            if level < self.num_levels - 1:
                skips.append(h)
                h = F.avg_pool2d(h, kernel_size=2, ceil_mode=True)

        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)

        for block in self.up_blocks:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), t_emb)

        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Spatial Transformer backbone
# ---------------------------------------------------------------------------


def patchify_2d(x: torch.Tensor, patch_h: int, patch_w: int) -> tuple[torch.Tensor, int, int]:
    """Vectorised 2-D patchification.

    Args:
        x: ``[B, C, H, W]`` where ``H`` and ``W`` are exact multiples of the patch size.

    Returns:
        ``(tokens, grid_h, grid_w)`` where ``tokens`` is
        ``[B, grid_h * grid_w, C * patch_h * patch_w]`` in **row-major**
        ``(lat, lon)`` order, so token index ``i`` maps to grid position
        ``(i // grid_w, i % grid_w)``.
    """
    b, c, h, w = x.shape
    if h % patch_h or w % patch_w:
        raise ValueError(
            f"patchify_2d requires H and W to be multiples of the patch size; "
            f"got ({h}, {w}) with patch ({patch_h}, {patch_w})."
        )
    grid_h, grid_w = h // patch_h, w // patch_w
    tokens = (
        x.reshape(b, c, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 2, 4, 1, 3, 5)  # B, gh, gw, C, ph, pw
        .reshape(b, grid_h * grid_w, c * patch_h * patch_w)
    )
    return tokens, grid_h, grid_w


def unpatchify_2d(
    tokens: torch.Tensor, channels: int, grid_h: int, grid_w: int, patch_h: int, patch_w: int
) -> torch.Tensor:
    """Exact inverse of :func:`patchify_2d`."""
    b, n, d = tokens.shape
    if n != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} tokens, got {n}.")
    if d != channels * patch_h * patch_w:
        raise ValueError(f"Expected token width {channels * patch_h * patch_w}, got {d}.")
    return (
        tokens.reshape(b, grid_h, grid_w, channels, patch_h, patch_w)
        .permute(0, 3, 1, 4, 2, 5)  # B, C, gh, ph, gw, pw
        .reshape(b, channels, grid_h * patch_h, grid_w * patch_w)
    )


def sincos_2d_positional_encoding(
    dim: int, grid_h: int, grid_w: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Fixed 2-D sin/cos positional encoding, ``[grid_h * grid_w, dim]``.

    Half of the embedding encodes the latitude token index and half the
    longitude token index, so the encoding is a genuine function of the
    two-dimensional grid location (not of a flattened sequence position).
    """
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d_positional_encoding requires dim % 4 == 0, got {dim}")
    half = dim // 2

    def _axis(positions: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(half // 2, device=device, dtype=torch.float32)
        omega = 1.0 / (10_000 ** (omega / (half / 2)))
        out = positions.float().reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    rows = torch.arange(grid_h, device=device)
    cols = torch.arange(grid_w, device=device)
    lat_grid = rows.reshape(-1, 1).expand(grid_h, grid_w).reshape(-1)
    lon_grid = cols.reshape(1, -1).expand(grid_h, grid_w).reshape(-1)
    return torch.cat([_axis(lat_grid), _axis(lon_grid)], dim=1).to(dtype)


class _SpatialAttention(nn.Module):
    """Full (exact) multi-head self-attention across spatial tokens.

    ``optimized_attention`` selects between PyTorch's fused
    ``scaled_dot_product_attention`` and an explicit reference implementation.
    Both compute the *same* mathematical operation; no sparse, local or
    linearised approximation is ever substituted.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, implementation: str = "auto") -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.set_attention_implementation(implementation)

    def set_attention_implementation(self, implementation: str) -> None:
        impl = str(implementation).lower()
        if impl not in {"auto", "sdpa", "math"}:
            raise ValueError(f"Unsupported attention implementation {implementation!r}")
        if impl == "auto":
            impl = "sdpa" if hasattr(F, "scaled_dot_product_attention") else "math"
        if impl == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "optimized_attention='sdpa' requested but this PyTorch build has no "
                "scaled_dot_product_attention; use 'math' or 'auto'."
            )
        self.implementation = impl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.dropout if self.training else 0.0
        if self.implementation == "sdpa":
            # is_causal=False: attention is bidirectional across spatial tokens.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=False)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = scores.softmax(dim=-1)
            if p > 0:
                weights = F.dropout(weights, p=p, training=True)
            out = weights @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _DiTBlock(nn.Module):
    """Pre-norm Transformer block with adaptive-layer-norm process-time conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, implementation: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _SpatialAttention(dim, num_heads, dropout, implementation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        params = self.ada_ln(cond_emb.to(x.dtype)).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (p.unsqueeze(1) for p in params)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h)
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class SpatialResidualTransformer(nn.Module):
    """DiT-style Transformer refiner operating on 2-D spatial tokens only.

    Pipeline for each sample::

        conditioning + noisy residual  [B, C, H, W]
          -> reflect-pad to a multiple of the patch size
          -> patchify into (grid_h x grid_w) 2-D tokens
          -> linear token projection + 2-D positional encoding
          -> N x (spatial self-attention + MLP) with adaLN process-time conditioning
          -> linear projection back to patch pixels
          -> unpatchify
          -> crop to the exact original (H, W)

    Padding is applied symmetrically at the trailing edges and removed by an
    exact crop, so the returned field has the original latitude/longitude
    dimensions and token order is preserved throughout.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        *,
        patch_size: tuple[int, int] = (4, 4),
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        positional_encoding: str = "learned_2d",
        max_tokens_lat: int = 256,
        max_tokens_lon: int = 256,
        gradient_checkpointing: bool = False,
        optimized_attention: str = "auto",
        time_embedding_kind: str = "sinusoidal",
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.patch_h, self.patch_w = int(patch_size[0]), int(patch_size[1])
        if self.patch_h < 1 or self.patch_w < 1:
            raise ValueError(f"patch size must be positive, got {patch_size!r}")
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.embedding_dim = int(embedding_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.positional_encoding = str(positional_encoding).lower()
        self.max_tokens_lat = int(max_tokens_lat)
        self.max_tokens_lon = int(max_tokens_lon)

        token_in = (self.in_channels + self.cond_channels) * self.patch_h * self.patch_w
        token_out = self.out_channels * self.patch_h * self.patch_w
        self.token_proj = nn.Linear(token_in, self.embedding_dim)
        self.time_embed = ProcessTimeEmbedding(self.embedding_dim, kind=time_embedding_kind)

        if self.positional_encoding == "learned_2d":
            # Separable learned 2-D encoding: one table per axis, summed. This
            # keeps the parameter count linear in the grid extent while still
            # producing a distinct embedding for every (lat, lon) token.
            self.pos_lat = nn.Parameter(torch.zeros(self.max_tokens_lat, self.embedding_dim))
            self.pos_lon = nn.Parameter(torch.zeros(self.max_tokens_lon, self.embedding_dim))
            nn.init.trunc_normal_(self.pos_lat, std=0.02)
            nn.init.trunc_normal_(self.pos_lon, std=0.02)
        elif self.positional_encoding == "sincos_2d":
            self.pos_lat = None
            self.pos_lon = None
            self._sincos_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
        else:
            raise ValueError(f"Unsupported positional_encoding {positional_encoding!r}")

        self.blocks = nn.ModuleList(
            [
                _DiTBlock(self.embedding_dim, num_heads, mlp_ratio, dropout, optimized_attention)
                for _ in range(int(num_blocks))
            ]
        )
        self.final_norm = nn.LayerNorm(self.embedding_dim, elementwise_affine=False, eps=1e-6)
        self.final_ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(self.embedding_dim, 2 * self.embedding_dim))
        nn.init.zeros_(self.final_ada_ln[1].weight)
        nn.init.zeros_(self.final_ada_ln[1].bias)
        self.out_proj = nn.Linear(self.embedding_dim, token_out)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    # -- helpers ---------------------------------------------------------
    def set_attention_implementation(self, implementation: str) -> None:
        for block in self.blocks:
            block.attn.set_attention_implementation(implementation)

    def _positional(self, grid_h: int, grid_w: int, device, dtype) -> torch.Tensor:
        if self.positional_encoding == "learned_2d":
            if grid_h > self.max_tokens_lat or grid_w > self.max_tokens_lon:
                raise ValueError(
                    f"Token grid ({grid_h}, {grid_w}) exceeds the configured "
                    f"max_tokens_lat/max_tokens_lon ({self.max_tokens_lat}, "
                    f"{self.max_tokens_lon}). Increase them or the patch size."
                )
            lat = self.pos_lat[:grid_h].unsqueeze(1)  # [gh, 1, D]
            lon = self.pos_lon[:grid_w].unsqueeze(0)  # [1, gw, D]
            return (lat + lon).reshape(grid_h * grid_w, self.embedding_dim).to(dtype)
        key = (grid_h, grid_w, device, dtype)
        cached = self._sincos_cache.get(key)
        if cached is None:
            cached = sincos_2d_positional_encoding(
                self.embedding_dim, grid_h, grid_w, device=device, dtype=dtype
            )
            self._sincos_cache[key] = cached
        return cached

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Residual field {tuple(x.shape[-2:])} and conditioning "
                f"{tuple(cond.shape[-2:])} must share the spatial grid."
            )
        b, _, h, w = x.shape
        stacked = torch.cat([x, cond], dim=1)

        pad_h = (-h) % self.patch_h
        pad_w = (-w) % self.patch_w
        if pad_h or pad_w:
            # Replicate padding avoids inventing boundary gradients; the padded
            # rows/columns are cropped away exactly after unpatchify.
            stacked = F.pad(stacked, (0, pad_w, 0, pad_h), mode="replicate")

        tokens, grid_h, grid_w = patchify_2d(stacked, self.patch_h, self.patch_w)
        tokens = self.token_proj(tokens)
        tokens = tokens + self._positional(grid_h, grid_w, tokens.device, tokens.dtype).unsqueeze(0)

        cond_emb = self.time_embed(t).to(tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, cond_emb, use_reentrant=False)
            else:
                tokens = block(tokens, cond_emb)

        shift, scale = self.final_ada_ln(cond_emb).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = self.out_proj(tokens)

        out = unpatchify_2d(tokens, self.out_channels, grid_h, grid_w, self.patch_h, self.patch_w)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        if out.shape != (b, self.out_channels, h, w):
            raise RuntimeError(
                f"Transformer output shape {tuple(out.shape)} does not match the "
                f"expected {(b, self.out_channels, h, w)}."
            )
        return out
