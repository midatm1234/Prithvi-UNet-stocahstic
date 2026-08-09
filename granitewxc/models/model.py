import torch
import numpy as np

from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.distributed import is_main_process
from granitewxc.decoders.downscaling import ConvEncoderDecoder
from granitewxc.models.finetune_model import PatchEmbed
from granitewxc.models.cordex_finetune_model import ClimateDownscaleFinetuneUNETModel, ClimateDownscaleFinetuneModel, resolve_head_type
from granitewxc.utils.predictands import build_predictand_specs
from PrithviWxC.model import PrithviWxCEncoderDecoder



def _as_int_list(value, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, (int, float)):
        return [max(1, int(value))]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                if not item:
                    continue
                out.append(max(1, int(item[0])))
            else:
                out.append(max(1, int(item)))
        return out or list(default)
    return list(default)


def _resolve_scaler_device(config: ExperimentConfig) -> torch.device:
    requested = getattr(config, "scalers_device", None)
    if requested is None:
        requested = getattr(getattr(config, "model", object()), "scalers_device", None)

    # Default to CPU to avoid early CUDA allocations during model construction.
    if requested is None:
        return torch.device("cpu")

    try:
        device = torch.device(str(requested))
    except (RuntimeError, TypeError, ValueError):
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def get_scalers(config: ExperimentConfig):
    """
    calls assemble scalers func. 
    """    

    # Keep scalers on CPU by default; they move with model.to(device) later.
    device = _resolve_scaler_device(config)

    if config.data.type == 'eccc':
        input_mu = torch.load(config.model.input_mu, map_location=device, weights_only=False)
        input_sigma = torch.load(config.model.input_sigma, map_location=device, weights_only=False)
        input_static_mu = torch.load(config.model.input_static_mu, map_location=device, weights_only=False)
        input_static_sigma = torch.load(config.model.input_static_sigma, map_location=device, weights_only=False)
        target_mu = torch.load(config.model.target_mu, map_location=device, weights_only=False)
        target_sigma = torch.load(config.model.target_sigma, map_location=device, weights_only=False)
        target_static_mu = torch.load(config.model.target_static_mu, map_location=device, weights_only=False)
        target_static_sigma = torch.load(config.model.target_static_sigma, map_location=device, weights_only=False)
    elif config.data.type == 'cordex':
        specs = build_predictand_specs(config, output_vars=list(config.data.output_vars))

        def load_array(path: str) -> torch.Tensor:
            array = np.load(path)
            tensor = torch.from_numpy(array)
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            if tensor.device != device:
                tensor = tensor.to(device)
            return tensor

        input_mu_full = load_array(config.model.input_mu)
        input_sigma_full = load_array(config.model.input_sigma)
        target_mu = load_array(config.model.target_mu)
        target_sigma = load_array(config.model.target_sigma)
        if target_mu.numel() < len(specs) or target_sigma.numel() < len(specs):
            raise ValueError(
                "Loaded target scalers have fewer channels than config.data.output_vars. "
                "Check scaler files and output_vars ordering."
            )

        for idx, spec in enumerate(specs):
            if spec.scaling.method == "divide_only":
                mu_slice = target_mu[idx]
                if mu_slice.ndim == 0:
                    mu_abs_max = float(torch.abs(mu_slice).item())
                else:
                    mu_abs_max = float(torch.abs(mu_slice).max().item())
                if mu_abs_max > 1e-6:
                    raise ValueError(
                        f"predictands.{spec.name}.scaling.method='divide_only' requires "
                        f"target_mu[{idx}] == 0, got max(|mu|)={mu_abs_max:.6g}. "
                        "Recompute scalers."
                    )

        static_channels = int(getattr(config.model, "num_static_channels", 1))
        # number of dynamic predictor channels (time * vars * levels)
        n_dynamic = (
            len(config.data.input_vars)
            * len(config.data.input_levels)
            * max(1, int(getattr(config.data, "n_input_timestamps", 1)))
        )

        if static_channels <= 0:
            input_mu = input_mu_full[:n_dynamic]
            input_sigma = input_sigma_full[:n_dynamic]
            input_static_mu = torch.zeros(0, device=device)
            input_static_sigma = torch.ones(0, device=device)
            target_static_mu = torch.zeros(0, device=device)
            target_static_sigma = torch.ones(0, device=device)
        elif input_mu_full.shape[0] == n_dynamic + static_channels:
            input_mu = input_mu_full[:n_dynamic]
            input_sigma = input_sigma_full[:n_dynamic]
            input_static_mu = input_mu_full[-static_channels:]
            input_static_sigma = input_sigma_full[-static_channels:]
            target_static_mu = input_static_mu.clone()
            target_static_sigma = input_static_sigma.clone()
        else:
            input_mu = input_mu_full
            input_sigma = input_sigma_full
            input_static_mu = torch.zeros(static_channels, device=device)
            input_static_sigma = torch.ones(static_channels, device=device)
            target_static_mu = torch.zeros(static_channels, device=device)
            target_static_sigma = torch.ones(static_channels, device=device)
    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')

    return dict(
        input_mu=input_mu,
        input_sigma=input_sigma,
        input_static_mu=input_static_mu,
        input_static_sigma=input_static_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
        target_static_mu = target_static_mu,
        target_static_sigma = target_static_sigma
    )


def get_eccc_embedding_module(config: ExperimentConfig):
    '''
    n_parameters = n_surface_vars + n_vertical*level
    in ECCC we have: 
          n_parameters   =  3 (surface) + 6(other) + 6 (vertical)*5(press) +4(vertical)*3(level1) + 2(vertical)*3 (level_2) = 57
    '''

    n_parameters = len(config.data.input_surface_vars) + len(config.data.other) + len(
        config.data.vertical_pres_vars)*len(config.data.input_level_pres) + len(
        config.data.vertical_level1_vars)*len(config.data.input_level1)+ len(
        config.data.vertical_level2_vars)*len(config.data.input_level2)
    
    patch_embedding = PatchEmbed(
        patch_size=config.model.downscaling_patch_size,
        channels=n_parameters * config.data.n_input_timestamps,
        embed_dim=config.model.downscaling_embed_dim,
    )

    use_static = bool(getattr(config.data, "use_static", getattr(config, "finetune_w_static", True)))
    static_surface_vars = getattr(config.data, "input_static_surface_vars", [])
    static_channels = int(getattr(config.model, "num_static_channels", 1))
    data_type = getattr(config.data, "type", None)
    if data_type == "cordex":
        # CORDEX static predictors are already appended in the dataset; avoid double-counting.
        static_surface_vars = []
    if not use_static:
        n_static_parameters = 0
    else:
        n_static_parameters = static_channels + len(static_surface_vars)
        if config.model.residual == 'climate':
            n_static_parameters += n_parameters

    patch_embedding_static = None
    if n_static_parameters > 0:
        patch_embedding_static = PatchEmbed(
            patch_size=config.model.downscaling_patch_size,
            channels=n_static_parameters,
            embed_dim=config.model.downscaling_embed_dim,
        ) 
        
    return patch_embedding, patch_embedding_static

    
#------------------------------
def get_finetune_model_UNET(config: ExperimentConfig) -> torch.nn.Module:
    """
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
    Returns:
        The configured model.
    """

    if is_main_process():
        print("Creating the model.")

    #########################################################
    # 0. Setup parameters/scalers
    #########################################################
    # set default kernel size
    if 'encoder_decoder_kernel_size_per_stage' not in config.model.__dict__:      
        config.model.encoder_decoder_kernel_size_per_stage = [[3]*len(inner) for inner in config.model.encoder_decoder_scale_per_stage]

    n_output_parameters = len(config.data.output_vars)
    if config.model.__dict__.get('loss_type', 'patch_rmse_loss')=='cross_entropy':
        if config.model.__dict__.get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
            n_output_parameters = config.model.__dict__.get('cross_entropy_n_bins', 512)
        else:
            n_output_parameters = len(np.load(config.model.cross_entropy_bin_boundaries_file)) + 1

    scalers = get_scalers(config)
    
    #########################################################
    # 1. Patch Embedding/Shallow Feature Extraction
    #########################################################
    if config.data.type in ('eccc', 'cordex'):  # eccc + cordex share embedding setup
        embedding, embedding_static = get_eccc_embedding_module(config)
    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')

    #########################################################
    # 3. FM/Deep Feature Extraction 
    #########################################################
    backbone = PrithviWxCEncoderDecoder(
        embed_dim=config.model.embed_dim,
        n_blocks=config.model.n_blocks_encoder,
        mlp_multiplier=config.model.mlp_multiplier,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout_rate,
        drop_path=config.model.drop_path,
    )


    #########################################################
    # 5. Putting it all together
    #########################################################
    unet_scales = _as_int_list(
        getattr(config.model, "unet_upsample_scales", None),
        [2, 2, 2],
    )
    unet_kernels = _as_int_list(
        getattr(config.model, "unet_decoder_kernel_size", None),
        [3] * len(unet_scales),
    )
    if len(unet_kernels) < len(unet_scales):
        unet_kernels.extend([unet_kernels[-1]] * (len(unet_scales) - len(unet_kernels)))
    elif len(unet_kernels) > len(unet_scales):
        unet_kernels = unet_kernels[: len(unet_scales)]

    model = ClimateDownscaleFinetuneUNETModel(
        embedding=embedding,
        embedding_static=embedding_static,
        backbone=backbone,
        input_scalers_mu=scalers['input_mu'],
        input_scalers_sigma=scalers['input_sigma'],
        input_scalers_epsilon=1e-6,
        static_input_scalers_mu=scalers['input_static_mu'],
        static_input_scalers_sigma=scalers['input_static_sigma'],
        static_input_scalers_epsilon=1e-6,
        static_output_scalers_mu = scalers['target_static_mu'], 
        static_output_scalers_sigma = scalers['target_static_sigma'],
        output_scalers_mu=scalers['target_mu'],
        output_scalers_sigma=scalers['target_sigma'],
        patch_size_px_backbone=(1, 1),
        n_bins=n_output_parameters, #n_bins: int = 512,
        scale=unet_scales,
        kernel_size=unet_kernels,
        config = config
    )

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--> model has {total_params:,.0f} params.")
        
    return model


def get_finetune_model(config: ExperimentConfig) -> torch.nn.Module:
    """
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
    Returns:
        The configured model.
    """

    if is_main_process():
        print("Creating the model.")

    #########################################################
    # 0. Setup parameters/scalers
    #########################################################
    # set default kernel size
    if 'encoder_decoder_kernel_size_per_stage' not in config.model.__dict__:      
        config.model.encoder_decoder_kernel_size_per_stage = [[3]*len(inner) for inner in config.model.encoder_decoder_scale_per_stage]

    n_output_parameters = len(config.data.output_vars)
    if config.model.__dict__.get('loss_type', 'patch_rmse_loss')=='cross_entropy':
        if config.model.__dict__.get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
            n_output_parameters = config.model.__dict__.get('cross_entropy_n_bins', 512)
        else:
            n_output_parameters = len(np.load(config.model.cross_entropy_bin_boundaries_file)) + 1

    scalers = get_scalers(config)

    
    #########################################################
    # 1. Patch Embedding/Shallow Feature Extraction
    #########################################################
    if config.data.type in ('eccc', 'cordex'):  # eccc + cordex share embedding setup
        embedding, embedding_static = get_eccc_embedding_module(config)
    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')

    #########################################################
    # 2. Upscale before FM 
    # Keep token resolution similar to trained backbone
    #########################################################
    upscale = ConvEncoderDecoder(
        in_channels=config.model.downscaling_embed_dim,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=config.model.embed_dim,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[0],
        scale=config.model.encoder_decoder_scale_per_stage[0],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    ) 
    
    
    #########################################################
    # 3. FM/Deep Feature Extraction 
    #########################################################
    backbone = PrithviWxCEncoderDecoder(
        embed_dim=config.model.embed_dim,
        n_blocks=config.model.n_blocks_encoder,
        mlp_multiplier=config.model.mlp_multiplier,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout_rate,
        drop_path=config.model.drop_path,
    )

    #########################################################
    # 4. Upscale after FM 
    #########################################################
    if resolve_head_type(config) == "diffusion" and not _residual_diffusion_enabled(config):
        # Full-field diffusion does not need a deterministic decoder. Residual
        # diffusion does: it jointly supervises this exact head and defines
        # target - stop_gradient(baseline) from its prediction.
        head = None
    elif config.model.encoder_decoder_type == 'conv':
        head = ConvEncoderDecoder(
                in_channels=config.model.embed_dim,
                channels=config.model.encoder_decoder_conv_channels,
                out_channels=n_output_parameters,
                kernel_size=config.model.encoder_decoder_kernel_size_per_stage[1],
                scale=config.model.encoder_decoder_scale_per_stage[1],
                upsampling_mode=config.model.encoder_decoder_upsampling_mode,
        )
    else:
        raise NotImplementedError(f"Head type {config.model.encoder_decoder_type} not implemented.")

    #########################################################
    # 5. Putting it all together
    #########################################################
    model = ClimateDownscaleFinetuneModel(
        embedding=embedding,
        embedding_static=embedding_static,
        upscale=upscale,
        backbone=backbone,
        head=head,
        input_scalers_mu=scalers['input_mu'],
        input_scalers_sigma=scalers['input_sigma'],
        input_scalers_epsilon=1e-6,
        static_input_scalers_mu=scalers['input_static_mu'],
        static_input_scalers_sigma=scalers['input_static_sigma'],
        static_input_scalers_epsilon=1e-6,
        output_scalers_mu=scalers['target_mu'],
        output_scalers_sigma=scalers['target_sigma'],
        n_input_timestamps=config.data.n_input_timestamps,
        embed_dim_backbone=config.model.embed_dim,
        encoder_decoder_scale_per_stage=config.model.encoder_decoder_scale_per_stage,
        patch_size_px_backbone=(1, 1),
        mask_unit_size_px_backbone=config.mask_unit_size,
        n_bins=n_output_parameters,
        return_logits=config.model.__dict__.get('loss_type')=='cross_entropy',
        residual=config.model.__dict__.get('residual', None),
        residual_connection=config.model.__dict__.get('residual_connection', False),
        backbone_use = config.backbone_use,
        config=config,
    )

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--> model has {total_params:,.0f} params.")

    return model


def _residual_diffusion_enabled(config: ExperimentConfig) -> bool:
    model_cfg = getattr(config, "model", None)
    diffusion_cfg = getattr(model_cfg, "diffusion", None)
    if isinstance(diffusion_cfg, dict):
        return bool(diffusion_cfg.get("residual_diffusion", False))
    return bool(getattr(diffusion_cfg, "residual_diffusion", False))
