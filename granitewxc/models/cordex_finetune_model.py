import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Sequence

from granitewxc.utils import distributed
from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.target_transforms import PositivePrecipLink
from granitewxc.models.finetune_model import FinetuneWrapper
from granitewxc.decoders.downscaling import ConvTransposeBlock, InterpBlock, PixelShuffleBlock


PRECIP_VAR_NAMES = {"pr", "precip", "precipitation", "ppt"}


def _canonicalize_precip_model(value: str | None) -> str:
    text = str(value or "single_head").strip().lower()
    aliases = {
        "single": "single_head",
        "single_head": "single_head",
        "default": "single_head",
        "legacy": "single_head",
        "hurdle": "hurdle",
        "bernoulli_positive": "hurdle",
        "bernoulli_plus_positive": "hurdle",
        "bernoulli_positive_amount": "hurdle",
        "bernoulli-plus-positive-amount": "hurdle",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported precip_model '{value}'. Expected one of {sorted(aliases)}."
        )
    return aliases[text]


def _reshape_output_scaler_tensor(name: str, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 1:
        return tensor.reshape(1, -1, 1, 1)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(
        f"{name} must have shape [C], [C,H,W], or [1,C,H,W], got {tuple(tensor.shape)}"
    )


def _resolve_spatial_scalers(
    name: str,
    mu_full: torch.Tensor,
    sigma_full: torch.Tensor,
    reference: torch.Tensor,
    scaler_offset: object | None = None,
    *,
    resize_mode: str = "bilinear",
    align_corners: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaler tensors that align with ``reference`` spatial dimensions.

    Scalers may be channel-only ``[1,C,1,1]`` or spatial ``[1,C,H,W]``. For
    spatial scalers and cropped batches, ``scaler_offset`` can be ``(y, x)`` or
    a batched ``[B,2]`` tensor/list so each sample receives the matching slice.

    Spatial scaler fields are coordinate-bearing data, not images. If their
    shape differs from ``reference``, an explicit, valid offset is therefore
    required. Silently resizing the entire scaler field to a crop would apply
    unrelated grid points to that crop and can imprint block-scale biases.

    ``resize_mode`` and ``align_corners`` remain in the signature for API and
    checkpoint-config compatibility, but mismatched spatial scalers are never
    resized.
    """
    del resize_mode, align_corners

    batch = int(reference.shape[0])
    h, w = reference.shape[-2:]

    if mu_full.ndim != 4 or sigma_full.ndim != 4:
        raise ValueError(
            f"{name} scalers must be 4-D [1,C,H,W], got "
            f"mu={tuple(mu_full.shape)} sigma={tuple(sigma_full.shape)}"
        )
    if mu_full.shape != sigma_full.shape:
        raise ValueError(
            f"{name} scaler mean/std shapes differ: "
            f"mu={tuple(mu_full.shape)} sigma={tuple(sigma_full.shape)}"
        )
    if mu_full.shape[0] != 1:
        raise ValueError(
            f"{name} spatial scalers must have singleton batch dimension, "
            f"got {tuple(mu_full.shape)}"
        )

    if mu_full.shape[-2:] == (1, 1) and sigma_full.shape[-2:] == (1, 1):
        return mu_full, sigma_full

    if mu_full.shape[-2:] == (h, w) and sigma_full.shape[-2:] == (h, w):
        return mu_full, sigma_full

    def _offsets() -> list[tuple[int, int]]:
        if scaler_offset is None:
            return []
        if torch.is_tensor(scaler_offset):
            offset_tensor = scaler_offset.detach().cpu()
            if offset_tensor.ndim == 1:
                if offset_tensor.numel() != 2:
                    raise ValueError(
                        f"{name} scaler_offset tensor must have 2 values, "
                        f"got shape {tuple(offset_tensor.shape)}"
                    )
                return [(int(offset_tensor[0].item()), int(offset_tensor[1].item()))]
            if offset_tensor.ndim != 2 or offset_tensor.shape[1] != 2:
                raise ValueError(
                    f"{name} scaler_offset tensor must have shape [2] or [B,2], "
                    f"got {tuple(offset_tensor.shape)}"
                )
            return [
                (int(row[0].item()), int(row[1].item()))
                for row in offset_tensor
            ]
        if isinstance(scaler_offset, (list, tuple)):
            if len(scaler_offset) == 2 and not isinstance(
                scaler_offset[0], (list, tuple, torch.Tensor)
            ):
                return [(int(scaler_offset[0]), int(scaler_offset[1]))]
            offsets: list[tuple[int, int]] = []
            for row in scaler_offset:
                if torch.is_tensor(row):
                    row = row.detach().cpu().reshape(-1).tolist()
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    raise ValueError(
                        f"{name} scaler_offset rows must contain exactly 2 values, "
                        f"got {row!r}"
                    )
                offsets.append((int(row[0]), int(row[1])))
            return offsets
        raise ValueError(
            f"Unsupported {name} scaler_offset type: {type(scaler_offset).__name__}"
        )

    offsets = _offsets()
    if not offsets:
        raise ValueError(
            f"{name} spatial scaler grid {tuple(mu_full.shape[-2:])} does not "
            f"match reference grid {(h, w)}; a scaler_offset is required"
        )
    if len(offsets) == 1 and batch > 1:
        offsets = offsets * batch
    if len(offsets) != batch:
        raise ValueError(
            f"{name} scaler_offset count ({len(offsets)}) must be 1 or match "
            f"reference batch size ({batch})"
        )

    mu_parts = []
    sigma_parts = []
    full_h, full_w = mu_full.shape[-2:]
    for sample_idx, (y0, x0) in enumerate(offsets):
        y1, x1 = y0 + h, x0 + w
        if y0 < 0 or x0 < 0 or y1 > full_h or x1 > full_w:
            if name != "input":
                raise ValueError(
                    f"{name} scaler_offset[{sample_idx}]={(y0, x0)} with crop "
                    f"{(h, w)} is outside scaler grid {(full_h, full_w)}"
                )
            source_y0, source_x0 = max(0, y0), max(0, x0)
            source_y1, source_x1 = min(full_h, y1), min(full_w, x1)
            if source_y0 >= source_y1 or source_x0 >= source_x1:
                raise ValueError(
                    f"input scaler crop {(y0, x0, y1, x1)} does not overlap "
                    f"scaler grid {(full_h, full_w)}"
                )
            padding = (
                source_x0 - x0,
                x1 - source_x1,
                source_y0 - y0,
                y1 - source_y1,
            )

            def _pad_input(values: torch.Tensor) -> torch.Tensor:
                source_h, source_w = values.shape[-2:]
                mode = "reflect"
                if (
                    padding[0] >= source_w
                    or padding[1] >= source_w
                    or padding[2] >= source_h
                    or padding[3] >= source_h
                ):
                    mode = "replicate"
                return F.pad(values, padding, mode=mode)

            mu_parts.append(
                _pad_input(mu_full[..., source_y0:source_y1, source_x0:source_x1])
            )
            sigma_parts.append(
                _pad_input(
                    sigma_full[..., source_y0:source_y1, source_x0:source_x1]
                )
            )
        else:
            mu_parts.append(mu_full[..., y0:y1, x0:x1])
            sigma_parts.append(sigma_full[..., y0:y1, x0:x1])

    return torch.cat(mu_parts, dim=0), torch.cat(sigma_parts, dim=0)


def _normalize_output_crops(
    output_crop: object | None,
    batch_size: int,
) -> list[tuple[int, int, int, int]]:
    """Normalize ``(top, left, height, width)`` crop metadata per sample."""
    if output_crop is None:
        return []

    if torch.is_tensor(output_crop):
        crop_tensor = output_crop.detach().cpu()
        if crop_tensor.ndim == 1:
            if crop_tensor.numel() != 4:
                raise ValueError(
                    "__output_crop tensor must have 4 values or shape [B,4], "
                    f"got {tuple(crop_tensor.shape)}"
                )
            rows = [crop_tensor.tolist()]
        elif crop_tensor.ndim == 2 and crop_tensor.shape[1] == 4:
            rows = crop_tensor.tolist()
        else:
            raise ValueError(
                "__output_crop tensor must have shape [4] or [B,4], "
                f"got {tuple(crop_tensor.shape)}"
            )
    elif isinstance(output_crop, (list, tuple)):
        if len(output_crop) == 4 and not isinstance(
            output_crop[0], (list, tuple, torch.Tensor)
        ):
            rows = [output_crop]
        else:
            rows = []
            for row in output_crop:
                if torch.is_tensor(row):
                    row = row.detach().cpu().reshape(-1).tolist()
                if not isinstance(row, (list, tuple)) or len(row) != 4:
                    raise ValueError(
                        "__output_crop rows must contain exactly 4 values "
                        f"(top,left,height,width), got {row!r}"
                    )
                rows.append(row)
    else:
        raise ValueError(
            f"Unsupported __output_crop type: {type(output_crop).__name__}"
        )

    crops = [tuple(int(value) for value in row) for row in rows]
    if len(crops) == 1 and batch_size > 1:
        crops = crops * batch_size
    if len(crops) != batch_size:
        raise ValueError(
            f"__output_crop count ({len(crops)}) must be 1 or match batch "
            f"size ({batch_size})"
        )
    return crops


def _prepare_normalized_outputs(
    raw_out: torch.Tensor,
    wet_logits: torch.Tensor | None,
    expected_hw: tuple[int, int] | torch.Size,
    output_crop: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Crop/resize model-space outputs before constraints and denormalization."""
    expected = (int(expected_hw[-2]), int(expected_hw[-1]))
    if expected[0] <= 0 or expected[1] <= 0:
        raise ValueError(f"Expected positive output dimensions, got {expected}")

    crops = _normalize_output_crops(output_crop, int(raw_out.shape[0]))
    if crops:
        raw_parts = []
        wet_parts = []
        crop_shapes = set()
        source_h, source_w = raw_out.shape[-2:]
        for sample_idx, (top, left, height, width) in enumerate(crops):
            bottom, right = top + height, left + width
            if (
                top < 0
                or left < 0
                or height <= 0
                or width <= 0
                or bottom > source_h
                or right > source_w
            ):
                raise ValueError(
                    f"__output_crop[{sample_idx}]={(top, left, height, width)} "
                    f"is outside raw output grid {(source_h, source_w)}"
                )
            crop_shapes.add((height, width))
            raw_parts.append(raw_out[sample_idx : sample_idx + 1, ..., top:bottom, left:right])
            if wet_logits is not None:
                wet_parts.append(
                    wet_logits[sample_idx : sample_idx + 1, ..., top:bottom, left:right]
                )
        if len(crop_shapes) != 1:
            raise ValueError(
                "All __output_crop entries in a batch must use the same height/width"
            )
        raw_out = torch.cat(raw_parts, dim=0)
        if wet_logits is not None:
            wet_logits = torch.cat(wet_parts, dim=0)

    if raw_out.shape[-2:] != expected:
        raw_out = F.interpolate(
            raw_out,
            size=expected,
            mode="bilinear",
            align_corners=False,
        )
        if wet_logits is not None:
            wet_logits = F.interpolate(
                wet_logits,
                size=expected,
                mode="bilinear",
                align_corners=False,
            )

    return raw_out, wet_logits


class ClimateECCCFinetuneWrapper(FinetuneWrapper):
    """ General purpose wrapper class to finetune using us configurable head and backbone """

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__(backbone, head)

    def load_pretrained_backbone(
            self, 
            weights_path: str,
            ignore_modules: Optional[list[str]] = None,
            sel_prefix: str = 'module.',
            freeze: bool = False,
            unused_parameters: Optional[list[str]] = None,
            return_keys: bool = False
    ):
        """  Based off of load checkpoint

        Args:
            weights_path: path to model checkpoint with only model weights
            ignore_modules: modules to ignore within the selected hierarchy.
                To ignore embedding related modules, set variable to ['patch_embedding', 'unembed']
            sel_prefix: '' selects all the modules within PrithviWxC. 'encoder.' selects encoder only.
            freeze: freezes the backbone when set.
            return_keys: If True, returns the output of load_state_dict(): missing_keys and unexpected_keys fields.

        Returns:
            If *return_keys* is True, `NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields.
        """
        if distributed.is_main_process():
            print(f"Loading pre-trained model weights {weights_path}...")

        if os.path.isfile(weights_path):
            # Always load checkpoints on CPU to avoid exhausting GPU memory when loading.
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        else:
            raise ValueError(
                f"Invalid checkpoint path: {weights_path}. Please provide a valid path to a checkpoint."
            )
        
        n_clip = len(sel_prefix)
        checkpoint = {k[n_clip:]: v for k, v in checkpoint.items() if k.startswith(sel_prefix)}
        checkpoint, ignore_layers = self.ignore_patch_embed(checkpoint, ignore_modules)
        out = self.backbone.load_state_dict(checkpoint, strict=False)  # loads pre-trained weights into backbone
        
        if unused_parameters is not None:
            self.freeze_unused_parameters(unused_parameters)
        
        if freeze:
            self.freeze_model(self.backbone, ignore_layers)  # freezes backbone layers, except ignore_layers

        if return_keys:
            return out
        else:
            return



#-----------------------------------------------------
# UNET for static covariates
#-----------------------------------------------------
class ClimateDownscaleFinetuneUNETModel(ClimateECCCFinetuneWrapper):

    def __init__(
            self,
            embedding: torch.nn.Module,
            backbone: torch.nn.Module,
            patch_size_px_backbone: tuple[int, int],
            input_scalers_mu: torch.tensor,
            input_scalers_sigma: torch.tensor,
            input_scalers_epsilon: float,
            static_input_scalers_mu: torch.Tensor,
            static_input_scalers_sigma: torch.Tensor,
            static_input_scalers_epsilon: float,
            output_scalers_mu: torch.tensor,
            output_scalers_sigma: torch.tensor,
            static_output_scalers_mu: torch.Tensor, # ----- to be used in  UNET
            static_output_scalers_sigma: torch.Tensor, # ----- to be used in  UNET
            embedding_static: Optional[torch.nn.Module] = None,
            n_bins: int = 512,
            scale = [2,2,2], # ----- to be used in  UNET (config.encoder_decoder_scale_per_stage)
            kernel_size  = [3,3,3], # encoder_decoder_kernel_size_per_stage
            config: ExperimentConfig = None
        ):
        """ Climate Downscaling Model based on pre-trained backbone. 
        Args:
            embedding: module used to embed input [C, H, W] -> [E, h, w]
            backbone: module that learns the system dynamics (optionally fully trained) [E, h, w] -> [E, h, w]
            head: module to shape output  [E, h, w] -> [O, H, W]
            n_lats_px_backbone: Total latitudes in data. In pixels.
            n_lons_px_backbone: Total longitudes in data. In pixels
            patch_size_px_backbone: Patch size for tokenization. In pixels lat/lon
            mask_unit_size_px_backbone: Size of each mask unit. In pixels lat/lon
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale input
            input_scalers_sigma:Tensor of size (in_channels,). Used to rescale input
            input_scalers_epsilon: Used to rescale input/ define a lower limit on std
            target_scalers_mu: Tensor of shape (in_channels,). Used to rescale output.
            target_scalers_sigma: Tensor of shape (in_channels,). Used to rescale output.
            n_bins: (optional) Used for cross entropy loss
            return_logits: (optional) Used to determine if we cross entropy loss
            residual: (optional) Indicates the residual mode of the model. for regression
                ['climate',  None]
            residual_connection: (optional) Use a skip/residual connection around the model backbone
            backbone_residual_mode: (optional) ``legacy_ignored`` preserves the
                historical UNET behavior in which the configured residual was
                computed but discarded. ``pre_conv_add`` feeds the shallow/deep
                sum to the post-backbone convolution.
        """

        super().__init__(backbone, None)
        self.skip_activation_devices: list[torch.device] = []

        #----------- From Config

        n_input_timestamps = config.data.n_input_timestamps
        embed_dim_backbone = config.model.embed_dim
        return_logits = config.model.__dict__.get('loss_type')=='cross_entropy'
        residual = config.model.__dict__.get('residual', None)
        residual_connection = config.model.__dict__.get('residual_connection', False)
        out_channels = n_bins
        self.backbone_use = config.backbone_use 
        self.mask_unit_size_px_backbone = config.mask_unit_size
        self.encoder_decoder_scale_per_stage = config.model.encoder_decoder_scale_per_stage
        self.use_static = bool(getattr(config.data, "use_static", getattr(config, "finetune_w_static", True)))
        self.static_embedding_scale = float(
            getattr(config.model, "static_embedding_scale", 1.0)
        )
        self.static_skip_scale = float(
            getattr(config.model, "static_skip_scale", 1.0)
        )
        self.static_dropout_p = float(
            getattr(config.model, "static_dropout_p", 0.0)
        )
        self.static_dropout_p = min(max(self.static_dropout_p, 0.0), 1.0)
        self.decoder_skip_source = str(
            getattr(config.model, "decoder_skip_source", "legacy")
        ).strip().lower()
        if self.decoder_skip_source not in {"legacy", "dynamic"}:
            raise ValueError(
                "model.decoder_skip_source must be 'legacy' or 'dynamic', "
                f"got {self.decoder_skip_source!r}"
            )
        self.backbone_residual_mode = str(
            getattr(config.model, "backbone_residual_mode", "legacy_ignored")
        ).strip().lower()
        if self.backbone_residual_mode not in {"legacy_ignored", "pre_conv_add"}:
            raise ValueError(
                "model.backbone_residual_mode must be 'legacy_ignored' or "
                f"'pre_conv_add', got {self.backbone_residual_mode!r}"
            )
        self.backbone_attention_scope = str(
            getattr(config.model, "backbone_attention_scope", "legacy_global")
        ).strip().lower()
        if self.backbone_attention_scope not in {
            "legacy_global",
            "windowed_local",
        }:
            raise ValueError(
                "model.backbone_attention_scope must be 'legacy_global' or "
                f"'windowed_local', got {self.backbone_attention_scope!r}"
            )
        self.decoder_upsampling_mode = str(
            getattr(
                config.model,
                "decoder_upsampling_mode",
                getattr(config.model, "encoder_decoder_upsampling_mode", "pixel_shuffle"),
            )
        ).lower()
        if self.decoder_upsampling_mode not in {"pixel_shuffle", "bilinear", "nearest", "conv_transpose"}:
            self.decoder_upsampling_mode = "pixel_shuffle"
        self.output_scaler_resize_mode = str(
            getattr(config.model, "output_scaler_resize_mode", "bilinear")
        ).lower()
        if self.output_scaler_resize_mode not in {"nearest", "bilinear", "bicubic"}:
            self.output_scaler_resize_mode = "bilinear"
        self.output_scaler_align_corners = bool(
            getattr(config.model, "output_scaler_align_corners", False)
        )
        #-----------

        self.n_input_timestamps = n_input_timestamps

        self.residual = residual if residual is not None else ''
        self.residual_connection = residual_connection

        self.embedding = embedding
        self.embedding_static = embedding_static
        self.downscaling_embed_dim = config.model.downscaling_embed_dim
        self.embed_dim_backbone = embed_dim_backbone


        self.conv_after_backbone = nn.Conv2d(
            embed_dim_backbone, 
            embed_dim_backbone, 
            kernel_size=3, 
            stride=1,
            padding='same',
            padding_mode='replicate'
        )
        self.backbone_gradient_checkpointing = getattr(config, "backbone_gradient_checkpointing", False)

        self.conv_before_backbone = nn.Conv2d(
            2 * self.downscaling_embed_dim, 
            embed_dim_backbone, 
            kernel_size=3, 
            stride=1,
            padding='same',
            padding_mode='replicate'
        )

        # Input shape [batch, time x parameter, lat, lon]
        self.input_scalers_mu = torch.nn.Parameter(
            _reshape_output_scaler_tensor("input_scalers_mu", input_scalers_mu),
            requires_grad=False,
        )
        self.input_scalers_sigma = torch.nn.Parameter(
            _reshape_output_scaler_tensor("input_scalers_sigma", input_scalers_sigma),
            requires_grad=False,
        )
        self.input_scalers_epsilon = input_scalers_epsilon

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_mu = nn.Parameter(
            static_input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_sigma = nn.Parameter(
            static_input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_epsilon = static_input_scalers_epsilon

        if output_scalers_mu is not None:
            self.output_scalers_mu = torch.nn.Parameter(
                _reshape_output_scaler_tensor("output_scalers_mu", output_scalers_mu),
                requires_grad=False,
            )

        self.output_scalers_sigma = torch.nn.Parameter(
            _reshape_output_scaler_tensor("output_scalers_sigma", output_scalers_sigma),
            requires_grad=False,
        )

        # ----- to be used in  UNET
        self.static_output_scalers_mu = nn.Parameter(
            static_output_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_output_scalers_sigma = nn.Parameter(
            static_output_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )

        self.scale = scale
        self.kernel_size = kernel_size    
        self.num_upsample = len(self.scale)

        # downsampling layers
        self.downsampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.downscaling_embed_dim, self.downscaling_embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    #nn.LeakyReLU()
                    nn.PReLU()
                )
                for _ in range(self.num_upsample)
            ]
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_ch = config.model.embed_dim 
        channels = config.model.encoder_decoder_conv_channels
        
        for step_idx in range(self.num_upsample):
            k_i,  s_i = self.kernel_size[step_idx], self.scale[step_idx]
            
            if step_idx == self.num_upsample-1:
                current_ch = config.model.embed_dim 
            else :
                current_ch = config.model.encoder_decoder_conv_channels + self.downscaling_embed_dim
            if self.decoder_upsampling_mode == "pixel_shuffle":
                layer = PixelShuffleBlock(
                    scale=s_i,
                    in_channels=current_ch,
                    channels=channels,
                    kernel_size=k_i,
                    stride=1,
                    activation=nn.PReLU,
                )
            elif self.decoder_upsampling_mode in {"bilinear", "nearest"}:
                layer = InterpBlock(
                    scale=s_i,
                    in_channels=current_ch,
                    channels=channels,
                    kernel_size=k_i,
                    stride=1,
                    activation=nn.PReLU,
                    mode=self.decoder_upsampling_mode,
                )
            else:
                layer = ConvTransposeBlock(
                    in_channels=current_ch,
                    channels=channels,
                    kernel_size=max(2, int(s_i)),
                    stride=int(s_i),
                    activation=nn.PReLU,
                )
            self.upsample_layers.append(layer)

        current_ch = config.model.encoder_decoder_conv_channels + self.downscaling_embed_dim
        self._decoder_output_channels = current_ch
        self.output_conv_block = nn.Sequential(nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=1, padding='same', padding_mode='replicate'),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(current_ch, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='replicate'),
                                              )
        self.precip_wet_head: nn.Module | None = None

        self.apply(self._init_weights)
        
        #----------

        self.patch_size_px = patch_size_px_backbone

        self.return_logits = return_logits
        if self.return_logits:
            self.to_logits = nn.Conv2d(
                in_channels=n_bins,
                out_channels=n_bins,
                kernel_size=1,
            )
        self._configure_predictand_decoding(config)
        if self.precip_hurdle_enabled:
            self.precip_wet_head = nn.Conv2d(
                in_channels=self._decoder_output_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding='same',
                padding_mode='replicate',
            )
            self._init_weights(self.precip_wet_head)

    def _configure_predictand_decoding(self, config: ExperimentConfig | None) -> None:
        n_outputs = int(self.output_scalers_sigma.shape[1])
        output_vars = list(getattr(getattr(config, "data", None), "output_vars", []))
        if len(output_vars) < n_outputs:
            output_vars.extend(f"var_{idx}" for idx in range(len(output_vars), n_outputs))
        else:
            output_vars = output_vars[:n_outputs]

        if config is not None:
            specs = build_predictand_specs(config, output_vars=output_vars)
        else:
            specs = []
        if not specs:
            class _Spec:
                def __init__(self, name: str):
                    self.name = name
                    self.scaling = type("Scaling", (), {"method": "zscore"})
                    self.nonnegativity = type(
                        "NonNeg", (), {"enabled": False, "method": "none"}
                    )

            specs = [_Spec(name) for name in output_vars]

        scaling_codes = []
        nonneg_enabled = []
        nonneg_codes = []
        scaling_to_code = {
            "zscore": 0,
            "divide_only": 1,
            "log1p_zscore": 2,
            "log1p_standardize": 2,
        }
        nonneg_to_code = {"none": 0, "softplus": 1, "exp": 2}
        for idx, spec in enumerate(specs):
            scaling_method = spec.scaling.method
            nonneg_method = spec.nonnegativity.method if spec.nonnegativity.enabled else "none"
            scaling_codes.append(scaling_to_code[scaling_method])
            nonneg_enabled.append(bool(spec.nonnegativity.enabled))
            nonneg_codes.append(nonneg_to_code[nonneg_method])

            if scaling_method == "divide_only" and hasattr(self, "output_scalers_mu"):
                mu_slice = self.output_scalers_mu[0, idx, ...]
                mu_abs_max = float(torch.abs(mu_slice).max().item())
                if mu_abs_max > 1e-6:
                    raise ValueError(
                        f"predictands.{spec.name}.scaling.method=divide_only requires zero target_mu, "
                        f"but loaded max(|target_mu[{idx}]|)={mu_abs_max:.6g}. "
                        "Recompute scalers or use legacy config."
                    )

        precip_model = _canonicalize_precip_model(
            getattr(config, "precip_model", getattr(config, "precip_head_type", "single_head"))
            if config is not None
            else "single_head"
        )
        precip_idx = next(
            (idx for idx, name in enumerate(output_vars) if str(name).lower() in PRECIP_VAR_NAMES),
            -1,
        )
        if precip_model == "hurdle" and precip_idx < 0:
            raise ValueError(
                "precip_model='hurdle' requires a precipitation output variable (e.g. 'pr')."
            )
        precip_hurdle_enabled = precip_model == "hurdle" and precip_idx >= 0
        if precip_hurdle_enabled:
            precip_spec = specs[precip_idx]
            if precip_spec.scaling.method != "divide_only":
                raise ValueError(
                    "precip_model='hurdle' requires divide_only precipitation scaling "
                    f"(predictands.{precip_spec.name}.scaling.method='divide_only')."
                )
            if precip_spec.scaling.scale_stat != "p95":
                raise ValueError(
                    "precip_model='hurdle' requires precipitation scaling.scale_stat='p95'."
                )
            if precip_spec.scaling.mode != "global":
                raise ValueError(
                    "precip_model='hurdle' requires precipitation normalization.mode='global'."
                )

        self.output_var_names = output_vars
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.tensor(scaling_codes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.tensor(nonneg_enabled, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_method_codes",
            torch.tensor(nonneg_codes, dtype=torch.int64),
            persistent=False,
        )
        self.precip_model = precip_model
        self.precip_channel_index = precip_idx
        self.precip_hurdle_enabled = precip_hurdle_enabled
        self.precip_wet_threshold = float(
            getattr(config, "precip_wet_threshold", 0.0) if config is not None else 0.0
        )
        self.precip_occurrence_prob_threshold = float(
            getattr(config, "precip_occurrence_prob_threshold", 0.5)
            if config is not None
            else 0.5
        )
        self.precip_occurrence_prob_threshold = min(
            max(self.precip_occurrence_prob_threshold, 0.0),
            1.0,
        )
        self._last_precip_hurdle_aux = None
        self._precip_softplus_link = PositivePrecipLink()

    def _apply_output_constraints(self, raw_out: torch.Tensor) -> torch.Tensor:
        if not bool(self.predictand_nonneg_enabled_mask.any().item()):
            return raw_out

        channels: list[torch.Tensor] = []
        for channel_idx in range(raw_out.shape[1]):
            channel = raw_out[:, channel_idx : channel_idx + 1, ...]
            if bool(self.predictand_nonneg_enabled_mask[channel_idx].item()):
                method_code = int(self.predictand_nonneg_method_codes[channel_idx].item())
                if method_code == 1:
                    channel = self._precip_softplus_link(channel)
                elif method_code == 2:
                    channel = torch.exp(channel)
                else:
                    raise ValueError(
                        f"Invalid nonnegativity method code {method_code} for channel {channel_idx}"
                    )
            channels.append(channel)
        return torch.cat(channels, dim=1)

    def _resolve_output_scalers(
        self,
        constrained: torch.Tensor,
        scaler_offset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_full = self.output_scalers_sigma.to(device=constrained.device, dtype=constrained.dtype)
        if hasattr(self, "output_scalers_mu"):
            mu_full = self.output_scalers_mu.to(device=constrained.device, dtype=constrained.dtype)
        else:
            mu_full = torch.zeros_like(sigma_full)
        return _resolve_spatial_scalers(
            "output",
            mu_full,
            sigma_full,
            constrained,
            scaler_offset=scaler_offset,
            resize_mode=self.output_scaler_resize_mode,
            align_corners=self.output_scaler_align_corners,
        )

    def _resolve_input_scalers(
        self,
        reference: torch.Tensor,
        scaler_offset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_full = self.input_scalers_mu.to(device=reference.device, dtype=reference.dtype)
        sigma_full = self.input_scalers_sigma.to(device=reference.device, dtype=reference.dtype)
        return _resolve_spatial_scalers(
            "input",
            mu_full,
            sigma_full,
            reference,
            scaler_offset=scaler_offset,
            resize_mode=self.output_scaler_resize_mode,
            align_corners=self.output_scaler_align_corners,
        )

    def _decode_outputs(
        self,
        raw_out: torch.Tensor,
        scaler_offset: object | None = None,
        wet_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        constrained = self._apply_output_constraints(raw_out)
        mu, sigma = self._resolve_output_scalers(constrained, scaler_offset=scaler_offset)
        decoded = constrained * sigma + mu

        method_codes = self.predictand_scaling_method_codes.to(device=constrained.device)
        divide_mask = method_codes == 1
        if bool(divide_mask.any().item()):
            divide_indices = torch.nonzero(divide_mask, as_tuple=False).flatten().tolist()
            divide_values = constrained[:, divide_mask, ...] * sigma[:, divide_mask, ...]
            decoded_channels = list(decoded.split(1, dim=1))
            for local_idx, channel_idx in enumerate(divide_indices):
                decoded_channels[channel_idx] = divide_values[:, local_idx : local_idx + 1, ...]
            decoded = torch.cat(decoded_channels, dim=1)

        log1p_mask = method_codes == 2
        if bool(log1p_mask.any().item()):
            log_decoded = torch.expm1(
                constrained[:, log1p_mask, ...] * sigma[:, log1p_mask, ...]
                + mu[:, log1p_mask, ...]
            )
            tiny_negative_tol = 1e-7
            log_decoded = torch.where(
                (log_decoded < 0.0) & (log_decoded > -tiny_negative_tol),
                torch.zeros_like(log_decoded),
                log_decoded,
            )
            log_indices = torch.nonzero(log1p_mask, as_tuple=False).flatten().tolist()
            decoded_channels = list(decoded.split(1, dim=1))
            for local_idx, channel_idx in enumerate(log_indices):
                decoded_channels[channel_idx] = log_decoded[:, local_idx : local_idx + 1, ...]
            decoded = torch.cat(decoded_channels, dim=1)

        self._last_precip_hurdle_aux = None
        if self.precip_hurdle_enabled:
            if wet_logits is None:
                raise ValueError("precip_model='hurdle' requires wet_logits during decoding.")
            pr_idx = int(self.precip_channel_index)
            q95 = torch.clamp(sigma[:, pr_idx : pr_idx + 1, ...], min=1e-12)
            amount_raw = raw_out[:, pr_idx : pr_idx + 1, ...]
            amount_pred_norm = self._precip_softplus_link(amount_raw)
            amount_pred = amount_pred_norm * q95
            p_wet = torch.sigmoid(wet_logits)
            final_pr = torch.where(
                p_wet >= self.precip_occurrence_prob_threshold,
                amount_pred,
                torch.zeros_like(amount_pred),
            )
            decoded_channels = list(decoded.split(1, dim=1))
            decoded_channels[pr_idx] = final_pr
            decoded = torch.cat(decoded_channels, dim=1)

            constrained_channels = list(constrained.split(1, dim=1))
            constrained_channels[pr_idx] = amount_pred_norm
            constrained = torch.cat(constrained_channels, dim=1)
            self._last_precip_hurdle_aux = {
                "wet_logits": wet_logits,
                "p_wet": p_wet,
                "amount_pred_norm": amount_pred_norm,
                "amount_pred": amount_pred,
                "q95": q95,
            }

        return decoded, constrained


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _expand_static(self, tensor: torch.Tensor, batch: dict[str, torch.tensor], H: int, W: int) -> torch.Tensor:
        return tensor.to(device=batch["x"].device, dtype=batch["x"].dtype).expand(
            batch["x"].shape[0], -1, H, W
        )

    def _resolve_static(self, batch: dict[str, torch.tensor], H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_static and "static_x" in batch and "static_y" in batch:
            return batch["static_x"], batch["static_y"]
        static_x = self._expand_static(self.static_input_scalers_mu, batch, H, W)
        static_y = self._expand_static(self.static_output_scalers_mu, batch, H, W)
        return static_x, static_y


    #@profile
    def forward(
        self,
        batch: dict[str, torch.tensor],
        return_pre_inverse: bool = False,
        return_raw_output: bool = False,
    ):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and optional 'static_x'/'static_y'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        # v6 hurdle training keeps aux tensors for the precipitation loss. Clear any
        # stale reference from the previous step before allocating new activations,
        # otherwise the previous autograd graph can remain alive and raise peak VRAM.
        self._last_precip_hurdle_aux = None
        batch.pop("__precip_hurdle_aux", None)

        B, _, H, W = batch['x'].shape
        # Scale inputs
        x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W) # [batch, time x parameter, lat, lon] -> [batch, time, parameter, lat, lon]
        legacy_scaler_offset = batch.get("__scaler_offset")
        input_scaler_offset = batch.get(
            "__input_scaler_offset", legacy_scaler_offset
        )
        input_mu, input_sigma = self._resolve_input_scalers(
            batch["x"],
            scaler_offset=input_scaler_offset,
        )
        x_scale = (x_sep_time - input_mu.unsqueeze(1)) / (
                input_sigma.unsqueeze(1) + self.input_scalers_epsilon)
        x = x_scale.view(B, -1, H, W) # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        use_static = self.use_static and self.embedding_static is not None

        if use_static:
            static_x, static_y = self._resolve_static(batch, H, W)
            x_static = (static_x - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )
            if self.training and self.static_dropout_p > 0.0:
                x_static = F.dropout2d(x_static, p=self.static_dropout_p, training=True)

            if self.residual == 'climate':
                # Scale climatology
                climate = (batch['climate_x'] - input_mu) / (
                    input_sigma + self.input_scalers_epsilon
                )

                # concat with static in channels dimension
                x_static = torch.cat([x_static, climate], dim=1)

            # ----- to be used in  UNET
            # Embedding and dowsampling of static HRDPS covariates
            y_static = (static_y - self.static_output_scalers_mu) / (
                self.static_output_scalers_sigma + self.static_input_scalers_epsilon) # self.static_input_scalers_epsilon is a constant small number
            if self.training and self.static_dropout_p > 0.0:
                y_static = F.dropout2d(y_static, p=self.static_dropout_p, training=True)

            if self.decoder_skip_source == "legacy":
                # Preserve the original static-only U-Net skip pathway exactly.
                skip_seed = self.embedding_static(y_static) * self.static_skip_scale
                x_embedded = None
            else:
                # Dynamic predictors are already co-registered to the fine target
                # grid in the PRISM workflows. Keep those fine-scale activations
                # available to every decoder stage instead of replacing them with
                # zero skips.
                x_embedded = self.embedding(x)
                skip_seed = x_embedded

            copy_activations = {0: skip_seed}
            primary_device = x.device
            for step_idx in range(self.num_upsample):
                current_activation = self._ensure_on_device(
                    copy_activations[step_idx], primary_device
                )
                copy_activations[step_idx] = current_activation
                copy_activations[step_idx + 1] = self.downsampling_layers[step_idx](
                    current_activation
                )
                copy_activations[step_idx] = self._maybe_offload_skip(
                    copy_activations[step_idx], step_idx, primary_device
                )

            if x_embedded is None:
                x_embedded = self.embedding(x)
            static_embedded = self.embedding_static(x_static) * self.static_embedding_scale
            x_shallow_feats = x_embedded + static_embedded

            # ----- to be used in  UNET
            deepest_skip = self._ensure_on_device(copy_activations[self.num_upsample], x_shallow_feats.device)
            if deepest_skip.shape[-2:] != x_shallow_feats.shape[-2:]:
                deepest_skip = F.interpolate(
                    deepest_skip,
                    size=x_shallow_feats.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            x_embedded = self.embedding(x)
            x_shallow_feats = x_embedded
            primary_device = x_shallow_feats.device
            copy_activations = {}
            if self.decoder_skip_source == "dynamic":
                copy_activations[0] = x_embedded
                for step_idx in range(self.num_upsample):
                    current_activation = self._ensure_on_device(
                        copy_activations[step_idx], primary_device
                    )
                    copy_activations[step_idx] = current_activation
                    copy_activations[step_idx + 1] = self.downsampling_layers[
                        step_idx
                    ](current_activation)
                    copy_activations[step_idx] = self._maybe_offload_skip(
                        copy_activations[step_idx], step_idx, primary_device
                    )
            else:
                # Exact legacy behavior: no-static models supplied all-zero
                # decoder skips and did not execute the learned downsamplers.
                current_skip = torch.zeros(
                    (
                        B,
                        self.downscaling_embed_dim,
                        x_shallow_feats.shape[-2],
                        x_shallow_feats.shape[-1],
                    ),
                    device=primary_device,
                    dtype=x_shallow_feats.dtype,
                )
                for step_idx in range(self.num_upsample):
                    copy_activations[step_idx] = self._maybe_offload_skip(
                        current_skip, step_idx, primary_device
                    )
                    current_skip = F.max_pool2d(current_skip, kernel_size=2)
                copy_activations[self.num_upsample] = self._maybe_offload_skip(
                    current_skip, self.num_upsample, primary_device
                )

            deepest_skip = self._ensure_on_device(
                copy_activations[self.num_upsample], x_shallow_feats.device
            )
            if deepest_skip.shape[-2:] != x_shallow_feats.shape[-2:]:
                deepest_skip = F.interpolate(
                    deepest_skip,
                    size=x_shallow_feats.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        x_shallow_feats = torch.cat([x_shallow_feats, deepest_skip], dim=1)
        x_shallow_feats = self.conv_before_backbone(x_shallow_feats)

        # calculate shapes to use in backbone (use current spatial resolution)
        n_lats_px_backbone = int(H)
        n_lons_px_backbone = int(W)
        
        assert n_lats_px_backbone % self.mask_unit_size_px_backbone[0] == 0
        assert n_lons_px_backbone % self.mask_unit_size_px_backbone[1] == 0
        assert self.mask_unit_size_px_backbone[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px_backbone[1] % self.patch_size_px[1] == 0

        local_shape_mu = (
            int(self.mask_unit_size_px_backbone[0] // self.patch_size_px[0]),
            int(self.mask_unit_size_px_backbone[1] // self.patch_size_px[1]),
        )
        global_shape_mu = (
            int(n_lats_px_backbone // self.mask_unit_size_px_backbone[0]),
            int(n_lons_px_backbone // self.mask_unit_size_px_backbone[1]),
        )

        # backbone
        if self.backbone_use:
            x_tokens = (
                x_shallow_feats.reshape(
                    B,
                    self.embed_dim_backbone,
                    global_shape_mu[0],
                    local_shape_mu[0],
                    global_shape_mu[1],
                    local_shape_mu[1],
                )
                .permute(0, 2, 4, 3, 5, 1)
                .flatten(3, 4)
                .flatten(1, 2)
            )  # [batch, embed, lat//patch_size, lon//patch_size] -> [batch, global seq, local seq, embed]

            if self.backbone_gradient_checkpointing and self.training:
                backbone_fn = (
                    self.backbone
                    if self.backbone_attention_scope == "legacy_global"
                    else self._windowed_local_backbone
                )
                x_deep_feats = checkpoint(backbone_fn, x_tokens, use_reentrant=False)
            else:
                if self.backbone_attention_scope == "legacy_global":
                    x_deep_feats = self.backbone(x_tokens)
                else:
                    x_deep_feats = self._windowed_local_backbone(x_tokens)
    
            x_deep_feats = x_deep_feats.reshape(
                B,
                global_shape_mu[0],
                global_shape_mu[1],
                local_shape_mu[0],
                local_shape_mu[1],
                -1
            ).permute(0, 5, 1, 3, 2, 4)
            
            x_deep_feats = x_deep_feats.flatten(4, 5).flatten(2, 3)

        else:
            x_deep_feats = x_shallow_feats

        # ``legacy_ignored`` deliberately preserves the historical checkpoint
        # behavior: the shallow/deep sum used to be computed and then discarded
        # before ``conv_after_backbone``. Corrected runs opt into the contracted
        # ``pre_conv_add`` behavior explicitly.
        if (
            self.residual_connection
            and self.backbone_residual_mode == "pre_conv_add"
        ):
            x = x_deep_feats + x_shallow_feats
        else:
            x = x_deep_feats

        # convolution after backbone
        x_deep_feats = self.conv_after_backbone(x)

        # Upscaling
        out = x_deep_feats
        bottleneck_skip = self._ensure_on_device(copy_activations[self.num_upsample], out.device)
        if out.shape[-2:] != bottleneck_skip.shape[-2:]:
            out = F.interpolate(
                out,
                size=bottleneck_skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        for step_idx in reversed(range(self.num_upsample)):
            skip = self._ensure_on_device(copy_activations[step_idx], out.device)
            upsampled = self.upsample_layers[step_idx](out)
            if skip.shape[-2:] != upsampled.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=upsampled.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            out = torch.cat((upsampled, skip), dim=1)

        x = self.output_conv_block(out)
        wet_logits = None
        if self.precip_hurdle_enabled:
            if self.precip_wet_head is None:
                raise ValueError("precip_model='hurdle' requested but precip_wet_head is not initialized.")
            wet_logits = self.precip_wet_head(out)

        raw_out = x
        expected_hw = batch["y"].shape[-2:]
        raw_out, wet_logits = _prepare_normalized_outputs(
            raw_out,
            wet_logits,
            expected_hw,
            output_crop=batch.get("__output_crop"),
        )
        output_scaler_offset = batch.get(
            "__output_scaler_offset", legacy_scaler_offset
        )
        x_out, x_pre_inverse = self._decode_outputs(
            raw_out,
            scaler_offset=output_scaler_offset,
            wet_logits=wet_logits,
        )

        if self._last_precip_hurdle_aux is not None:
            batch["__precip_hurdle_aux"] = self._last_precip_hurdle_aux
        else:
            batch.pop("__precip_hurdle_aux", None)

        if return_pre_inverse and return_raw_output:
            return x_out, x_pre_inverse, raw_out
        if return_pre_inverse:
            return x_out, x_pre_inverse
        if return_raw_output:
            return x_out, raw_out
        return x_out

    def get_last_precip_hurdle_aux(self) -> dict[str, torch.Tensor] | None:
        return self._last_precip_hurdle_aux

    def clear_last_precip_hurdle_aux(self) -> None:
        self._last_precip_hurdle_aux = None

    def _windowed_local_backbone(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Run every pretrained transformer inside fixed local mask units.

        The upstream Prithvi block alternates local attention with attention
        across *all* mask units in the current tensor. That makes a pixel depend
        on which other tiles happened to share its inference crop, so no finite
        halo can make tiled and full-frame inference agree. This mode retains
        the pretrained transformer parameters but applies each block along the
        bounded local-token axis. Mask units are anchored by globally aligned
        tile origins, giving a finite receptive field suitable for halo/core
        inference.
        """
        lgl_block = getattr(self.backbone, "lgl_block", None)
        transformers = getattr(lgl_block, "transformers", None)
        evaluators = getattr(lgl_block, "evaluator", None)
        if transformers is None:
            raise TypeError(
                "windowed_local attention requires a Prithvi backbone with "
                "lgl_block.transformers"
            )
        if evaluators is None:
            evaluators = [lambda module, value: module(value)] * len(transformers)
        for evaluator, transformer in zip(evaluators, transformers):
            x_tokens = evaluator(transformer, (x_tokens, None))
        return x_tokens

    # ----------------------
    # Utility helpers
    # ----------------------
    def set_skip_activation_devices(self, devices: Optional[Sequence[torch.device | str]]):
        """
        Configure optional devices where skip activations can be cached.
        Passing an empty sequence disables offloading.
        """
        if not devices:
            self.skip_activation_devices = []
            return

        formatted_devices: list[torch.device] = []
        for dev in devices:
            formatted_devices.append(torch.device(dev))
        self.skip_activation_devices = formatted_devices

    def _maybe_offload_skip(
        self, tensor: torch.Tensor, skip_idx: int, primary_device: torch.device
    ) -> torch.Tensor:
        if not self.skip_activation_devices:
            return tensor

        target_device = self.skip_activation_devices[skip_idx % len(self.skip_activation_devices)]
        if target_device == primary_device:
            return tensor

        return tensor.to(target_device, non_blocking=True)

    @staticmethod
    def _ensure_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.device == device:
            return tensor

        return tensor.to(device, non_blocking=True)


class ClimateDownscaleFinetuneModel(ClimateECCCFinetuneWrapper):

    def __init__(
            self,
            embedding: torch.nn.Module,
            upscale: torch.nn.Module,
            backbone: torch.nn.Module,
            head: torch.nn.Module,
            embed_dim_backbone: int,
            encoder_decoder_scale_per_stage: list[int],
            patch_size_px_backbone: tuple[int, int],
            mask_unit_size_px_backbone: tuple[int, int],
            input_scalers_mu: torch.tensor,
            input_scalers_sigma: torch.tensor,
            input_scalers_epsilon: float,
            static_input_scalers_mu: torch.Tensor,
            static_input_scalers_sigma: torch.Tensor,
            static_input_scalers_epsilon: float,
            output_scalers_mu: torch.tensor,
            output_scalers_sigma: torch.tensor,
            n_input_timestamps: int = 1,
            embedding_static: Optional[torch.nn.Module] = None,
            n_bins: int = 512,
            return_logits: bool = False,
            residual: str = None,
            residual_connection: bool = False,
            backbone_use = True, 
            config: ExperimentConfig | None = None,
        ):
        """ Climate Downscaling Model based on pre-trained backbone. 
        Args:
            embedding: module used to embed input [C, H, W] -> [E, h, w]
            backbone: module that learns the system dynamics (optionally fully trained) [E, h, w] -> [E, h, w]
            head: module to shape output  [E, h, w] -> [O, H, W]
            n_lats_px_backbone: Total latitudes in data. In pixels.
            n_lons_px_backbone: Total longitudes in data. In pixels
            patch_size_px_backbone: Patch size for tokenization. In pixels lat/lon
            mask_unit_size_px_backbone: Size of each mask unit. In pixels lat/lon
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale input
            input_scalers_sigma:Tensor of size (in_channels,). Used to rescale input
            input_scalers_epsilon: Used to rescale input/ define a lower limit on std
            target_scalers_mu: Tensor of shape (in_channels,). Used to rescale output.
            target_scalers_sigma: Tensor of shape (in_channels,). Used to rescale output.
            n_bins: (optional) Used for cross entropy loss
            return_logits: (optional) Used to determine if we cross entropy loss
            residual: (optional) Indicates the residual mode of the model. for regression
                ['climate',  None]
            residual_connection: (optional) Use a skip/residual connection around the model backbone
        """

        super().__init__(backbone, head)

        self.n_input_timestamps = n_input_timestamps

        self.residual = residual if residual is not None else ''
        self.residual_connection = residual_connection

        self.embedding = embedding
        self.embedding_static = embedding_static
        self.encoder_decoder_scale_per_stage = encoder_decoder_scale_per_stage
        self.mask_unit_size_px_backbone = mask_unit_size_px_backbone
        self.embed_dim_backbone = embed_dim_backbone

        self.upscale = upscale

        self.backbone_use = backbone_use
        if config is not None:
            self.use_static = bool(
                getattr(config.data, "use_static", getattr(config, "finetune_w_static", True))
            )
            self.static_embedding_scale = float(
                getattr(config.model, "static_embedding_scale", 1.0)
            )
            self.static_dropout_p = float(
                getattr(config.model, "static_dropout_p", 0.0)
            )
            self.output_scaler_resize_mode = str(
                getattr(config.model, "output_scaler_resize_mode", "bilinear")
            ).lower()
            if self.output_scaler_resize_mode not in {"nearest", "bilinear", "bicubic"}:
                self.output_scaler_resize_mode = "bilinear"
            self.output_scaler_align_corners = bool(
                getattr(config.model, "output_scaler_align_corners", False)
            )
        else:
            self.use_static = True
            self.static_embedding_scale = 1.0
            self.static_dropout_p = 0.0
            self.output_scaler_resize_mode = "bilinear"
            self.output_scaler_align_corners = False
        self.static_dropout_p = min(max(self.static_dropout_p, 0.0), 1.0)

        self.conv_after_backbone = nn.Conv2d(
            embed_dim_backbone, 
            embed_dim_backbone, 
            kernel_size=3, 
            stride=1,
            padding='same',
            padding_mode='replicate'
        )

        # Input shape [batch, time x parameter, lat, lon]
        self.input_scalers_mu = torch.nn.Parameter(
            _reshape_output_scaler_tensor("input_scalers_mu", input_scalers_mu),
            requires_grad=False,
        )
        self.input_scalers_sigma = torch.nn.Parameter(
            _reshape_output_scaler_tensor("input_scalers_sigma", input_scalers_sigma),
            requires_grad=False,
        )
        self.input_scalers_epsilon = input_scalers_epsilon

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_mu = nn.Parameter(
            static_input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_sigma = nn.Parameter(
            static_input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_epsilon = static_input_scalers_epsilon

        if output_scalers_mu is not None:
            self.output_scalers_mu = torch.nn.Parameter(
                _reshape_output_scaler_tensor("output_scalers_mu", output_scalers_mu),
                requires_grad=False,
            )
        self.output_scalers_sigma = torch.nn.Parameter(
            _reshape_output_scaler_tensor("output_scalers_sigma", output_scalers_sigma),
            requires_grad=False,
        )

        self.patch_size_px = patch_size_px_backbone

        self.return_logits = return_logits
        if self.return_logits:
            self.to_logits = nn.Conv2d(
                in_channels=n_bins,
                out_channels=n_bins,
                kernel_size=1,
            )
        self._configure_predictand_decoding(config)

    def _configure_predictand_decoding(self, config: ExperimentConfig | None) -> None:
        n_outputs = int(self.output_scalers_sigma.shape[1])
        output_vars = list(getattr(getattr(config, "data", None), "output_vars", []))
        if len(output_vars) < n_outputs:
            output_vars.extend(f"var_{idx}" for idx in range(len(output_vars), n_outputs))
        else:
            output_vars = output_vars[:n_outputs]

        if config is not None:
            specs = build_predictand_specs(config, output_vars=output_vars)
        else:
            specs = []
        if not specs:
            class _Spec:
                def __init__(self, name: str):
                    self.name = name
                    self.scaling = type("Scaling", (), {"method": "zscore"})
                    self.nonnegativity = type(
                        "NonNeg", (), {"enabled": False, "method": "none"}
                    )

            specs = [_Spec(name) for name in output_vars]

        scaling_codes = []
        nonneg_enabled = []
        nonneg_codes = []
        scaling_to_code = {
            "zscore": 0,
            "divide_only": 1,
            "log1p_zscore": 2,
            "log1p_standardize": 2,
        }
        nonneg_to_code = {"none": 0, "softplus": 1, "exp": 2}
        for idx, spec in enumerate(specs):
            scaling_method = spec.scaling.method
            nonneg_method = spec.nonnegativity.method if spec.nonnegativity.enabled else "none"
            scaling_codes.append(scaling_to_code[scaling_method])
            nonneg_enabled.append(bool(spec.nonnegativity.enabled))
            nonneg_codes.append(nonneg_to_code[nonneg_method])

            if scaling_method == "divide_only" and hasattr(self, "output_scalers_mu"):
                mu_slice = self.output_scalers_mu[0, idx, ...]
                mu_abs_max = float(torch.abs(mu_slice).max().item())
                if mu_abs_max > 1e-6:
                    raise ValueError(
                        f"predictands.{spec.name}.scaling.method=divide_only requires zero target_mu, "
                        f"but loaded max(|target_mu[{idx}]|)={mu_abs_max:.6g}. "
                        "Recompute scalers or use legacy config."
                    )

        self.output_var_names = output_vars
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.tensor(scaling_codes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.tensor(nonneg_enabled, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_method_codes",
            torch.tensor(nonneg_codes, dtype=torch.int64),
            persistent=False,
        )
        self._precip_softplus_link = PositivePrecipLink()

    def _apply_output_constraints(self, raw_out: torch.Tensor) -> torch.Tensor:
        if not bool(self.predictand_nonneg_enabled_mask.any().item()):
            return raw_out

        constrained = raw_out.clone()
        indices = torch.nonzero(self.predictand_nonneg_enabled_mask, as_tuple=False).flatten()
        for channel_idx in indices.tolist():
            method_code = int(self.predictand_nonneg_method_codes[channel_idx].item())
            if method_code == 1:
                constrained[:, channel_idx, ...] = self._precip_softplus_link(
                    constrained[:, channel_idx, ...]
                )
            elif method_code == 2:
                constrained[:, channel_idx, ...] = torch.exp(
                    constrained[:, channel_idx, ...]
                )
            else:
                raise ValueError(
                    f"Invalid nonnegativity method code {method_code} for channel {channel_idx}"
                )
        return constrained

    def _resolve_output_scalers(
        self,
        constrained: torch.Tensor,
        scaler_offset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_full = self.output_scalers_sigma.to(device=constrained.device, dtype=constrained.dtype)
        if hasattr(self, "output_scalers_mu"):
            mu_full = self.output_scalers_mu.to(device=constrained.device, dtype=constrained.dtype)
        else:
            mu_full = torch.zeros_like(sigma_full)
        return _resolve_spatial_scalers(
            "output",
            mu_full,
            sigma_full,
            constrained,
            scaler_offset=scaler_offset,
            resize_mode=self.output_scaler_resize_mode,
            align_corners=self.output_scaler_align_corners,
        )

    def _resolve_input_scalers(
        self,
        reference: torch.Tensor,
        scaler_offset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_full = self.input_scalers_mu.to(device=reference.device, dtype=reference.dtype)
        sigma_full = self.input_scalers_sigma.to(device=reference.device, dtype=reference.dtype)
        return _resolve_spatial_scalers(
            "input",
            mu_full,
            sigma_full,
            reference,
            scaler_offset=scaler_offset,
            resize_mode=self.output_scaler_resize_mode,
            align_corners=self.output_scaler_align_corners,
        )

    def _decode_outputs(
        self,
        raw_out: torch.Tensor,
        scaler_offset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        constrained = self._apply_output_constraints(raw_out)
        mu, sigma = self._resolve_output_scalers(constrained, scaler_offset=scaler_offset)

        decoded = constrained * sigma + mu

        method_codes = self.predictand_scaling_method_codes.to(device=constrained.device)
        divide_mask = method_codes == 1
        if bool(divide_mask.any().item()):
            decoded[:, divide_mask, ...] = constrained[:, divide_mask, ...] * sigma[:, divide_mask, ...]

        log1p_mask = method_codes == 2
        if bool(log1p_mask.any().item()):
            log_decoded = torch.expm1(
                constrained[:, log1p_mask, ...] * sigma[:, log1p_mask, ...]
                + mu[:, log1p_mask, ...]
            )
            tiny_negative_tol = 1e-7
            log_decoded = torch.where(
                (log_decoded < 0.0) & (log_decoded > -tiny_negative_tol),
                torch.zeros_like(log_decoded),
                log_decoded,
            )
            decoded[:, log1p_mask, ...] = log_decoded

        return decoded, constrained

    def _expand_static(self, tensor: torch.Tensor, batch: dict[str, torch.tensor], H: int, W: int) -> torch.Tensor:
        return tensor.to(device=batch["x"].device, dtype=batch["x"].dtype).expand(
            batch["x"].shape[0], -1, H, W
        )

    def _resolve_static(self, batch: dict[str, torch.tensor], H: int, W: int) -> torch.Tensor:
        if self.use_static and "static_x" in batch:
            return batch["static_x"]
        return self._expand_static(self.static_input_scalers_mu, batch, H, W)

    def swap_masking(self) -> None:
        return  

    #@profile
    def forward(
        self,
        batch: dict[str, torch.tensor],
        return_pre_inverse: bool = False,
        return_raw_output: bool = False,
    ):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and optional 'static_x'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """

        B, _, H, W = batch['x'].shape
        # Scale inputs
        x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W) # [batch, time x parameter, lat, lon] -> [batch, time, parameter, lat, lon]
        legacy_scaler_offset = batch.get("__scaler_offset")
        input_scaler_offset = batch.get(
            "__input_scaler_offset", legacy_scaler_offset
        )
        input_mu, input_sigma = self._resolve_input_scalers(
            batch["x"],
            scaler_offset=input_scaler_offset,
        )
        x_scale = (x_sep_time - input_mu.unsqueeze(1)) / (
                input_sigma.unsqueeze(1) + self.input_scalers_epsilon)
        x = x_scale.view(B, -1, H, W) # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]

        use_static = self.use_static and self.embedding_static is not None
        if use_static:
            static_x = self._resolve_static(batch, H, W)
            x_static = (static_x - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )
            if self.training and self.static_dropout_p > 0.0:
                x_static = F.dropout2d(x_static, p=self.static_dropout_p, training=True)

            if self.residual == 'climate':
                # Scale climatology
                climate = (batch['climate_x'] - input_mu) / (
                    input_sigma + self.input_scalers_epsilon
                )

                # concat with static in channels dimension
                x_static = torch.cat([x_static, climate], dim=1)

            x_embedded = self.embedding(x) # [batch, time x parameter, lat, lon] -> [batch, emb, lat*scale[0], lon*scale[0]]
            static_embedded = self.embedding_static(x_static) * self.static_embedding_scale
            x_shallow_feats = x_embedded + static_embedded
        else:
            x_shallow_feats = self.embedding(x)

        x_upscale = self.upscale(x_shallow_feats)

        # calculate shapes to use in backbone (use current spatial resolution)
        H, W = batch['x'].shape[2:]
        n_lats_px_backbone = int(H)
        n_lons_px_backbone = int(W)

        assert n_lats_px_backbone % self.mask_unit_size_px_backbone[0] == 0
        assert n_lons_px_backbone % self.mask_unit_size_px_backbone[1] == 0
        assert self.mask_unit_size_px_backbone[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px_backbone[1] % self.patch_size_px[1] == 0

        self.local_shape_mu = (
            int(self.mask_unit_size_px_backbone[0] // self.patch_size_px[0]),
            int(self.mask_unit_size_px_backbone[1] // self.patch_size_px[1]),
        )
        self.global_shape_mu = (
            int(n_lats_px_backbone // self.mask_unit_size_px_backbone[0]),
            int(n_lons_px_backbone // self.mask_unit_size_px_backbone[1]),
        )

        if self.backbone_use:
            x_tokens = (
                x_upscale.reshape(
                    B,
                    self.embed_dim_backbone,
                    self.global_shape_mu[0],
                    self.local_shape_mu[0],
                    self.global_shape_mu[1],
                    self.local_shape_mu[1],
                )
                .permute(0, 2, 4, 3, 5, 1)
                .flatten(3, 4)
                .flatten(1, 2)
            )  # [batch, embed, lat//patch_size, lon//patch_size] -> [batch, global seq, local seq, embed]

            #print("x_tokens", x_tokens.shape)
    
            x_deep_feats = self.backbone(x_tokens)  # [batch, global seq, local seq, embed]
    
            x_deep_feats = x_deep_feats.reshape(
                B,
                self.global_shape_mu[0],
                self.global_shape_mu[1],
                self.local_shape_mu[0],
                self.local_shape_mu[1],
                -1
            ).permute(0, 5, 1, 3, 2, 4)
            
            x_deep_feats = x_deep_feats.flatten(4, 5).flatten(2, 3)

        else:
            x_deep_feats = x_upscale
            

        x_deep_feats = self.conv_after_backbone(x_deep_feats)

        if self.residual_connection:
            x = x_deep_feats + x_upscale
        else:
            x = x_deep_feats

        x = self.head(x)  # [batch, out_channels, lat*scale[0]*scale[1], lon*scale[0]*scale[1]]
        raw_out = x
        raw_out, _ = _prepare_normalized_outputs(
            raw_out,
            None,
            batch["y"].shape[-2:],
            output_crop=batch.get("__output_crop"),
        )
        output_scaler_offset = batch.get(
            "__output_scaler_offset", legacy_scaler_offset
        )

        if self.return_logits:
            x_out = self.to_logits(raw_out)
            x_pre_inverse = x_out
        elif self.residual == 'climate':
            _, output_sigma = self._resolve_output_scalers(
                raw_out,
                scaler_offset=output_scaler_offset,
            )
            x_out = output_sigma * raw_out + batch['climate_y']
            x_pre_inverse = raw_out
        else:
            x_out, x_pre_inverse = self._decode_outputs(
                raw_out,
                scaler_offset=output_scaler_offset,
            )

        if return_pre_inverse and return_raw_output:
            return x_out, x_pre_inverse, raw_out
        if return_pre_inverse:
            return x_out, x_pre_inverse
        if return_raw_output:
            return x_out, raw_out
        return x_out
