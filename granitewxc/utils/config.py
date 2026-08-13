import os
from argparse import Namespace
from typing import Optional

import yaml


# Canonical names written by ``examples/CORDEX_ML/compute_scalars_cordex.py``.
# Keeping this small contract here lets training, inference, and the validation
# utility resolve the same case-local files without duplicating literals.
SCALAR_FILENAMES = {
    "inputs_mean": "inputs_mean.npy",
    "inputs_std": "inputs_std.npy",
    "targets_mean": "targets_mean.npy",
    "targets_std": "targets_std.npy",
}

CASE_SUBDIRS = ("scalars", "preproc", "checkpoints", "inference", "logs")


class MissingCaseNameError(ValueError):
    """Raised when a configuration omits its required ``case_name``."""


def _clean_case_name(value: object) -> str:
    return "" if value is None else str(value).strip()


class DataConfig:
    def __init__(
        self,
        surface_vars = [],
        static_surface_vars = [],
        vertical_vars = [],
        levels = [],
        time_range_train = [],
        time_range_valid = [],
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        self.surface_vars = surface_vars
        self.static_surface_vars = static_surface_vars
        self.vertical_vars = vertical_vars
        self.levels = levels
        self.time_range_train = time_range_train
        self.time_range_valid = time_range_valid
        if not hasattr(self, "input_static_surface_vars"):
            self.input_static_surface_vars = []

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return DataConfig(**args.__dict__)


class ModelConfig:
    def __init__(
        self,
        num_static_channels: Optional[int] = None,
        embed_dim: Optional[int] = None,
        token_size: Optional[tuple[int, int]] = None,
        n_blocks_encoder: Optional[int] = None,
        n_blocks_decoder: Optional[int] = None,
        mlp_multiplier: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        residual: Optional[bool] = False,
        train_loss: Optional[str] = None,
        val_loss: Optional[str] = None,
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        self.num_static_channels = num_static_channels
        self.embed_dim = embed_dim
        self.token_size = token_size
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.train_loss = train_loss
        self.val_loss = val_loss

        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return ModelConfig(**args.__dict__)

    @property
    def encoder_d_ff(self):
        return int(self.enc_embed_size * self.mlp_ratio)

    @property
    def decoder_d_ff(self):
        return int(self.dec_embed_size * self.mlp_ratio)

    def __str__(self):
        return (
            f"Input channels: {self.num_input_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )

    def __repr__(self):
        return (
            f"Input channels: {self.num_input_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )


class ExperimentConfig:
    def __init__(
        self,
        job_id: str = "",
        data_config: DataConfig = None,
        model_config: ModelConfig = None,
        case_name: Optional[str] = None,
        num_epochs: int = 1,
        limit_steps_train: int = 1,
        limit_steps_valid: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        min_lr: float = 1e-5,
        dl_num_workers: int = 1,
        dl_prefetch_size: int = 0,
        path_experiment: str = "./",
        warm_up_steps: int = 0,
        mask_unit_size: Optional[tuple[int]] = None,
        mask_ratio_inputs: Optional[float] = None,
        mask_ratio_targets: Optional[float] = None,
        **kwargs,
    ):
        # additional experiment parameters used in downstream tasks
        self.__dict__.update(kwargs)

        self.job_id = job_id
        self.case_name = _clean_case_name(case_name)
        self.data = data_config
        self.model = model_config
        self.num_epochs = num_epochs
        self.limit_steps_train = limit_steps_train
        self.limit_steps_valid = limit_steps_valid
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warm_up_steps = warm_up_steps
        self.dl_num_workers = dl_num_workers
        self.dl_prefetch_size = dl_prefetch_size
        self.mask_unit_size = mask_unit_size
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.path_experiment = path_experiment

    @property
    def path_checkpoint(self) -> str:
        if self.path_experiment == '':
            return os.path.join(self.path_weights, 'train', 'checkpoint.pt')
        else:
            return os.path.join(
                os.path.dirname(self.path_experiment), 'weights', 'train', 'checkpoint.pt'
            )

    @property
    def path_weights(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path(), "weights")

    @property
    def path_wandb(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path())

    def require_case_name(self) -> str:
        """Return the configured case name or fail with an actionable error."""

        name = _clean_case_name(getattr(self, "case_name", None))
        if not name:
            raise MissingCaseNameError(
                "`case_name` is not defined for this config. Add `case_name: "
                "<name>` so generated outputs can be organized under a "
                "case-specific folder."
            )
        return name

    @property
    def case_dir(self) -> str:
        """Root output directory ``<path_experiment>/<case_name>``."""

        case_name = self.require_case_name()
        base = os.path.normpath(self.path_experiment or ".")
        # Notebook callers sometimes already pass the case directory itself.
        if os.path.basename(base) == case_name:
            return base
        return os.path.join(base, case_name)

    @property
    def path_scalars(self) -> str:
        return os.path.join(self.case_dir, "scalars")

    @property
    def path_preproc(self) -> str:
        return os.path.join(self.case_dir, "preproc")

    @property
    def path_checkpoints(self) -> str:
        return os.path.join(self.case_dir, "checkpoints")

    @property
    def path_inference(self) -> str:
        return os.path.join(self.case_dir, "inference")

    @property
    def path_logs(self) -> str:
        return os.path.join(self.case_dir, "logs")

    def scalar_path(self, key: str) -> str:
        try:
            filename = SCALAR_FILENAMES[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown scalar {key!r}; expected one of "
                f"{sorted(SCALAR_FILENAMES)}."
            ) from exc
        return os.path.join(self.path_scalars, filename)

    def to_dict(self):
        d = self.__dict__.copy()
        d["model"] = self.model.to_dict()
        d["data"] = self.data.to_dict()

        return d

    @staticmethod
    def from_argparse(args: Namespace):
        return ExperimentConfig(
            data_config=DataConfig.from_argparse(args),
            model_config=ModelConfig.from_argparse(args),
            **args.__dict__,
        )

    @staticmethod
    def from_dict(params: dict):
        return ExperimentConfig(
            data_config=DataConfig(**params['data']),
            model_config=ModelConfig(**params['model']),
            **params,
        )

    def make_folder_name(self) -> str:
        param_folder = "v1"
        return param_folder

    def make_suffix_path(self) -> str:
        return os.path.join(self.make_folder_name(), self.job_id)

    def __str__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.num_epochs}, "
            f"Truncate train: {self.limit_steps_train}, "
            f"Truncate valid: {self.limit_steps_valid}, "
            f"Batch size: {self.batch_size}, "
            f"LR: {self.learning_rate}, "
            f"DL workers: {self.dl_num_workers}"
        )

    def __repr__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.num_epochs}, "
            f"Truncate train: {self.limit_steps_train}, "
            f"Truncate valid: {self.limit_steps_valid}, "
            f"Batch size: {self.batch_size}, "
            f"LR: {self.learning_rate}, "
            f"DL workers: {self.dl_num_workers}"
        )


def apply_case_output_paths(config: ExperimentConfig) -> ExperimentConfig:
    """Apply the legacy CORDEX case-local artifact layout additively.

    Explicit run-directory attributes continue to win.  Scalar paths are
    derived only when ``derive_output_paths: true`` is explicitly selected.
    Existing CORDEX configurations retain their configured scaler paths.
    """

    config.require_case_name()
    data_type = str(getattr(config.data, "type", "") or "").strip().lower()
    if data_type != "cordex":
        # PRISM workflows authenticate and resolve their own scalar artifacts.
        # Never repoint those paths through the legacy CORDEX case layout.
        return config
    if not bool(getattr(config, "derive_output_paths", False)):
        return config
    if config.model is not None:
        config.model.input_mu = config.scalar_path("inputs_mean")
        config.model.input_sigma = config.scalar_path("inputs_std")
        config.model.target_mu = config.scalar_path("targets_mean")
        config.model.target_sigma = config.scalar_path("targets_std")
    if config.data is not None:
        config.data.scalers = {
            key: config.scalar_path(key) for key in SCALAR_FILENAMES
        }

    for attr, path in (
        ("scalar_dir", config.path_scalars),
        ("preproc_dir", config.path_preproc),
        ("checkpoint_dir", config.path_checkpoints),
        ("inference_dir", config.path_inference),
        ("log_dir", config.path_logs),
    ):
        if not getattr(config, attr, None):
            setattr(config, attr, path)
    return config


def get_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping at the top level of {config_path}")
    if not _clean_case_name(cfg.get("case_name")):
        raise MissingCaseNameError(
            f"`case_name` must be defined in {config_path!r}. Add "
            "`case_name: <name>` so generated artifacts use a case-specific "
            "directory."
        )
    return apply_case_output_paths(ExperimentConfig.from_dict(cfg))
