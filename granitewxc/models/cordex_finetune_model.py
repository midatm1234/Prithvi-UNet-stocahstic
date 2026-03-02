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
            input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_sigma = torch.nn.Parameter(
            input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
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
                output_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
            )

        self.output_scalers_sigma = torch.nn.Parameter(
            output_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
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
            # in_channels // (scale_factor ** 2)
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=current_ch, out_channels=channels * s_i ** 2,
                          kernel_size=k_i, stride=1, padding='same', padding_mode='replicate'),
                nn.PixelShuffle(s_i),
                nn.PReLU()
            ))

        current_ch = config.model.encoder_decoder_conv_channels + self.downscaling_embed_dim
        self.output_conv_block = nn.Sequential(nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=1, padding='same', padding_mode='replicate'),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(current_ch, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='replicate'),
                                              )

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
        scaling_to_code = {"zscore": 0, "divide_only": 1, "log1p_zscore": 2}
        nonneg_to_code = {"none": 0, "softplus": 1, "exp": 2}
        for idx, spec in enumerate(specs):
            scaling_method = spec.scaling.method
            nonneg_method = spec.nonnegativity.method if spec.nonnegativity.enabled else "none"
            scaling_codes.append(scaling_to_code[scaling_method])
            nonneg_enabled.append(bool(spec.nonnegativity.enabled))
            nonneg_codes.append(nonneg_to_code[nonneg_method])

            if scaling_method == "divide_only" and hasattr(self, "output_scalers_mu"):
                mu_val = float(self.output_scalers_mu[0, idx, 0, 0].item())
                if abs(mu_val) > 1e-6:
                    raise ValueError(
                        f"predictands.{spec.name}.scaling.method=divide_only requires zero target_mu, "
                        f"but loaded target_mu[{idx}]={mu_val:.6g}. Recompute scalers or use legacy config."
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

    def _decode_outputs(self, raw_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        constrained = self._apply_output_constraints(raw_out)
        sigma = self.output_scalers_sigma.to(device=constrained.device, dtype=constrained.dtype)
        if hasattr(self, "output_scalers_mu"):
            mu = self.output_scalers_mu.to(device=constrained.device, dtype=constrained.dtype)
        else:
            mu = torch.zeros_like(sigma)
        decoded = constrained * sigma + mu

        method_codes = self.predictand_scaling_method_codes.to(device=constrained.device)
        divide_mask = method_codes == 1
        if bool(divide_mask.any().item()):
            decoded[:, divide_mask, ...] = constrained[:, divide_mask, ...] * sigma[:, divide_mask, ...]

        log1p_mask = method_codes == 2
        if bool(log1p_mask.any().item()):
            decoded[:, log1p_mask, ...] = torch.expm1(
                constrained[:, log1p_mask, ...] * sigma[:, log1p_mask, ...]
                + mu[:, log1p_mask, ...]
            )

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

        B, _, H, W = batch['x'].shape
        # Scale inputs
        x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W) # [batch, time x parameter, lat, lon] -> [batch, time, parameter, lat, lon]
        x_scale = (x_sep_time - self.input_scalers_mu.view(1, 1, -1, 1, 1)) / ( 
                self.input_scalers_sigma.view(1, 1, -1, 1, 1) + self.input_scalers_epsilon)
        x = x_scale.view(B, -1, H, W) # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        use_static = self.use_static and self.embedding_static is not None

        if use_static:
            static_x, static_y = self._resolve_static(batch, H, W)
            x_static = (static_x - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

            if self.residual == 'climate':
                # Scale climatology
                climate = (batch['climate_x'] - self.input_scalers_mu) / (
                    self.input_scalers_sigma + self.input_scalers_epsilon
                )

                # concat with static in channels dimension
                x_static = torch.cat([x_static, climate], dim=1)

            # ----- to be used in  UNET
            # Embedding and dowsampling of static HRDPS covariates
            y_static = (static_y - self.static_output_scalers_mu) / (
                self.static_output_scalers_sigma + self.static_input_scalers_epsilon) # self.static_input_scalers_epsilon is a constant small number

            # Dowsampling step
            copy_activations = {}
            copy_activations[0] = self.embedding_static(y_static)
            primary_device = x.device

            for step_idx in range(self.num_upsample):
                current_activation = self._ensure_on_device(copy_activations[step_idx], primary_device)
                copy_activations[step_idx] = current_activation
                copy_activations[step_idx + 1] = self.downsampling_layers[step_idx](current_activation)
                copy_activations[step_idx] = self._maybe_offload_skip(copy_activations[step_idx], step_idx, primary_device)

            x_embedded = self.embedding(x) # [batch, time x parameter, lat, lon] -> [batch, emb, lat*scale[0], lon*scale[0]]
            static_embedded = self.embedding_static(x_static)
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
            x_shallow_feats = self.embedding(x)
            deepest_skip = torch.zeros_like(x_shallow_feats)
            primary_device = x_shallow_feats.device
            copy_activations = {}
            current_skip = torch.zeros(
                (B, self.downscaling_embed_dim, x_shallow_feats.shape[-2], x_shallow_feats.shape[-1]),
                device=primary_device,
                dtype=x_shallow_feats.dtype,
            )
            for step_idx in range(self.num_upsample):
                copy_activations[step_idx] = self._maybe_offload_skip(current_skip, step_idx, primary_device)
                if step_idx < self.num_upsample - 1:
                    current_skip = F.max_pool2d(current_skip, kernel_size=2)

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
                x_deep_feats = checkpoint(self.backbone, x_tokens, use_reentrant=False)
            else:
                x_deep_feats = self.backbone(x_tokens)  # [batch, global seq, local seq, embed]
    
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

        # residual connection
        if self.residual_connection:
            x = x_deep_feats + x_shallow_feats
        else:
            x = x_deep_feats

        # convolution after backbone
        x_deep_feats = self.conv_after_backbone(x_deep_feats)

        # Upscaling
        out = x_deep_feats
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

        raw_out = x
        x_out, x_pre_inverse = self._decode_outputs(raw_out)
        expected_hw = batch["y"].shape[-2:]
        if x_out.shape[-2:] != expected_hw:
            x_out = F.interpolate(
                x_out,
                size=expected_hw,
                mode="bilinear",
                align_corners=False,
            )
            x_pre_inverse = F.interpolate(
                x_pre_inverse,
                size=expected_hw,
                mode="bilinear",
                align_corners=False,
            )
            raw_out = F.interpolate(
                raw_out,
                size=expected_hw,
                mode="bilinear",
                align_corners=False,
            )

        if return_pre_inverse and return_raw_output:
            return x_out, x_pre_inverse, raw_out
        if return_pre_inverse:
            return x_out, x_pre_inverse
        if return_raw_output:
            return x_out, raw_out
        return x_out

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
        else:
            self.use_static = True

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
            input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_sigma = torch.nn.Parameter(
            input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
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
                output_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
            )
        self.output_scalers_sigma = torch.nn.Parameter(
            output_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
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
        scaling_to_code = {"zscore": 0, "divide_only": 1, "log1p_zscore": 2}
        nonneg_to_code = {"none": 0, "softplus": 1, "exp": 2}
        for idx, spec in enumerate(specs):
            scaling_method = spec.scaling.method
            nonneg_method = spec.nonnegativity.method if spec.nonnegativity.enabled else "none"
            scaling_codes.append(scaling_to_code[scaling_method])
            nonneg_enabled.append(bool(spec.nonnegativity.enabled))
            nonneg_codes.append(nonneg_to_code[nonneg_method])

            if scaling_method == "divide_only" and hasattr(self, "output_scalers_mu"):
                mu_val = float(self.output_scalers_mu[0, idx, 0, 0].item())
                if abs(mu_val) > 1e-6:
                    raise ValueError(
                        f"predictands.{spec.name}.scaling.method=divide_only requires zero target_mu, "
                        f"but loaded target_mu[{idx}]={mu_val:.6g}. Recompute scalers or use legacy config."
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

    def _decode_outputs(self, raw_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        constrained = self._apply_output_constraints(raw_out)
        sigma = self.output_scalers_sigma.to(device=constrained.device, dtype=constrained.dtype)
        if hasattr(self, "output_scalers_mu"):
            mu = self.output_scalers_mu.to(device=constrained.device, dtype=constrained.dtype)
        else:
            mu = torch.zeros_like(sigma)

        decoded = constrained * sigma + mu

        method_codes = self.predictand_scaling_method_codes.to(device=constrained.device)
        divide_mask = method_codes == 1
        if bool(divide_mask.any().item()):
            decoded[:, divide_mask, ...] = constrained[:, divide_mask, ...] * sigma[:, divide_mask, ...]

        log1p_mask = method_codes == 2
        if bool(log1p_mask.any().item()):
            decoded[:, log1p_mask, ...] = torch.expm1(
                constrained[:, log1p_mask, ...] * sigma[:, log1p_mask, ...]
                + mu[:, log1p_mask, ...]
            )

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
        x_scale = (x_sep_time - self.input_scalers_mu.view(1, 1, -1, 1, 1)) / ( 
                self.input_scalers_sigma.view(1, 1, -1, 1, 1) + self.input_scalers_epsilon)
        x = x_scale.view(B, -1, H, W) # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]

        use_static = self.use_static and self.embedding_static is not None
        if use_static:
            static_x = self._resolve_static(batch, H, W)
            x_static = (static_x - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

            if self.residual == 'climate':
                # Scale climatology
                climate = (batch['climate_x'] - self.input_scalers_mu) / (
                    self.input_scalers_sigma + self.input_scalers_epsilon
                )

                # concat with static in channels dimension
                x_static = torch.cat([x_static, climate], dim=1)

            x_embedded = self.embedding(x) # [batch, time x parameter, lat, lon] -> [batch, emb, lat*scale[0], lon*scale[0]]
            static_embedded = self.embedding_static(x_static)
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

        if self.return_logits:
            x_out = self.to_logits(x)
            x_pre_inverse = x_out
        elif self.residual == 'climate':
            x_out = self.output_scalers_sigma * x + batch['climate_y']
            x_pre_inverse = x
        else:
            x_out, x_pre_inverse = self._decode_outputs(raw_out)

        if return_pre_inverse and return_raw_output:
            return x_out, x_pre_inverse, raw_out
        if return_pre_inverse:
            return x_out, x_pre_inverse
        if return_raw_output:
            return x_out, raw_out
        return x_out
