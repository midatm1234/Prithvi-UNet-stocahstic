import os
from argparse import Namespace
from typing import List, Optional

import yaml


# Canonical file names produced by ``compute_scalars_cordex.py``. These are the
# normalization statistics that training/inference load back from the
# case-specific ``scalars`` folder.
SCALAR_FILENAMES = {
    "inputs_mean": "inputs_mean.npy",
    "inputs_std": "inputs_std.npy",
    "targets_mean": "targets_mean.npy",
    "targets_std": "targets_std.npy",
}

# Sub-folders created underneath a case directory. Every generated artifact for
# a config lives under ``<path_experiment>/<case_name>/<subdir>``.
CASE_SUBDIRS = ("scalars", "preproc", "checkpoints", "inference", "logs")


class MissingCaseNameError(ValueError):
    """Raised when a config is loaded without the mandatory ``case_name`` field."""


def _clean_case_name(value: object) -> str:
    """Return a filesystem-friendly, non-empty case name or ``""`` when unset."""

    if value is None:
        return ""
    name = str(value).strip()
    return name


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

    # ------------------------------------------------------------------
    # Case-specific output layout
    #
    # Every generated artifact for a config is organized underneath a single
    # case directory named after ``case_name``. The layout is::
    #
    #     <path_experiment>/<case_name>/
    #         scalars/       # normalization statistics (inputs/targets *.npy)
    #         preproc/       # regridded / preprocessed predictor files
    #         checkpoints/   # training checkpoints (best/last/epoch_*.ckpt)
    #         inference/     # NetCDF predictions and diagnostics
    #         logs/          # training / evaluation logs
    # ------------------------------------------------------------------
    def require_case_name(self) -> str:
        """Return the case name, raising a clear error when it is missing."""

        name = _clean_case_name(getattr(self, "case_name", None))
        if not name:
            raise MissingCaseNameError(
                "`case_name` is not defined for this config. Add `case_name: "
                "<name>` as the first line of the YAML file so all generated "
                "outputs (scalars, preprocessed files, checkpoints, inference "
                "outputs and logs) can be organized under a case-specific folder."
            )
        return name

    @property
    def case_dir(self) -> str:
        """Root output directory for this case: ``<path_experiment>/<case_name>``.

        If ``path_experiment`` has already been pointed at the case directory
        (e.g. the fine-tune notebooks set ``path_experiment`` to
        ``<runs_root>/<case_name>``) the case name is not appended twice.
        """

        case_name = self.require_case_name()
        base = os.path.normpath(self.path_experiment or ".")
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
        """Absolute path of a normalization-statistic file inside the case."""

        try:
            filename = SCALAR_FILENAMES[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown scalar '{key}'. Expected one of {sorted(SCALAR_FILENAMES)}."
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
        param_folder = f"v1"
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


def apply_case_output_paths(config: "ExperimentConfig") -> "ExperimentConfig":
    """Point every generated-output location at the case-specific folder.

    This makes ``case_name`` the single source of truth for where artifacts are
    read from / written to:

    * ``config.model.{input_mu,input_sigma,target_mu,target_sigma}`` and
      ``config.data.scalers`` are repointed at ``<case_dir>/scalars`` so that
      training/inference load the statistics produced by
      ``compute_scalars_cordex.py``.
    * ``config.scalar_dir``/``config.preproc_dir``/``config.checkpoint_dir``/
      ``config.inference_dir`` default to the matching case sub-folders unless a
      caller (e.g. a notebook wiring up ``run_utils``) has already set them.

    Set ``derive_output_paths: false`` in the YAML to opt out of repointing the
    scalar files (the case sub-folder defaults are still exposed via the
    ``config.path_*`` properties).
    """

    config.require_case_name()

    derive = getattr(config, "derive_output_paths", True)
    if derive:
        model = getattr(config, "model", None)
        if model is not None:
            model.input_mu = config.scalar_path("inputs_mean")
            model.input_sigma = config.scalar_path("inputs_std")
            model.target_mu = config.scalar_path("targets_mean")
            model.target_sigma = config.scalar_path("targets_std")

        data = getattr(config, "data", None)
        if data is not None:
            data.scalers = {
                "inputs_mean": config.scalar_path("inputs_mean"),
                "inputs_std": config.scalar_path("inputs_std"),
                "targets_mean": config.scalar_path("targets_mean"),
                "targets_std": config.scalar_path("targets_std"),
            }

    # Provide case-specific defaults for the run directories consumed by the
    # trainer / preprocessing scripts. Existing explicit values win so the
    # notebook-driven ``run_utils`` layout keeps working unchanged.
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
    cfg = yaml.safe_load(open(config_path, 'r'))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path!r} did not parse into a mapping.")
    if not _clean_case_name(cfg.get("case_name")):
        raise MissingCaseNameError(
            f"`case_name` must be defined (as the first line) in {config_path!r}. "
            "Add `case_name: <name>` so all generated outputs (scalars, "
            "preprocessed files, checkpoints, inference outputs and logs) are "
            "organized under a case-specific folder named after `case_name`."
        )
    config = ExperimentConfig.from_dict(cfg)
    return apply_case_output_paths(config)

