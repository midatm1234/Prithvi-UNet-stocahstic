from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from granitewxc.decoders.diffusion_head import DiffusionHead, DiffusionHeadConfig
from granitewxc.models.diffusion_loss import JointResidualDiffusionLoss
from granitewxc.models.diffusion_sampling import get_ddim_sampler
from granitewxc.models.diffusion_sde import VPSDE
from granitewxc.models.loss import build_loss_fn
from granitewxc.models.cordex_finetune_model import ClimateECCCFinetuneWrapper
import granitewxc.utils.trainer as trainer_module
from granitewxc.utils.trainer import validate_one_epoch


def _small_mean_head(*, application_scale: float = 0.0) -> DiffusionHead:
    cfg = DiffusionHeadConfig(
        sde="vpsde",
        num_scales=8,
        num_sampling_steps=4,
        residual_diffusion=True,
        residual_mean_enabled=True,
        residual_mean_channels=8,
        residual_application_scale=application_scale,
        residual_magnitude_guard_multiple=0.0,
        base_channels=8,
        channel_multipliers=(1,),
        num_res_blocks=1,
        time_embed_dim=16,
        projected_cond_channels=4,
        dropout=0.0,
        fourier_scale=2.0,
        zero_init_output=False,
    )
    return DiffusionHead(cond_channels=3, output_channels=2, head_config=cfg)


def test_residual_mean_is_opt_in_and_legacy_state_dict_stays_compatible():
    legacy_cfg = DiffusionHeadConfig(
        residual_diffusion=True,
        base_channels=8,
        channel_multipliers=(1,),
        num_res_blocks=1,
        time_embed_dim=16,
        projected_cond_channels=4,
    )
    first = DiffusionHead(3, 2, legacy_cfg)
    second = DiffusionHead(3, 2, legacy_cfg)
    assert first.residual_mean_model is None
    assert not any(key.startswith("residual_mean_model.") for key in first.state_dict())
    second.load_state_dict(first.state_dict(), strict=True)

    with pytest.raises(ValueError, match="requires residual_diffusion=true"):
        DiffusionHeadConfig(residual_mean_enabled=True)


def test_score_and_configured_mean_objectives_have_separate_gradients():
    torch.manual_seed(7)
    head = _small_mean_head()
    head.train()
    cond = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.randn(2, 2, 4, 4)
    baseline = torch.randn(2, 2, 4, 4, requires_grad=True)

    score_loss = head.training_loss(cond, target, baseline_std=baseline)
    score_loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in head.score_model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.residual_mean_model.parameters()
    )
    assert cond.grad is None
    assert baseline.grad is None

    head.zero_grad(set_to_none=True)
    cond.grad = None
    baseline.grad = None
    head.training_loss(cond, target, baseline_std=baseline)
    residual_mean = head.get_last_residual_mean()
    assert residual_mean is not None and residual_mean.requires_grad
    configured_mean_loss = (
        baseline.detach() + residual_mean - target
    ).square().mean()
    configured_mean_loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in head.residual_mean_model.parameters()
    )
    assert all(parameter.grad is None for parameter in head.score_model.parameters())
    assert cond.grad is None
    assert baseline.grad is None


def test_sampling_recombines_mean_and_innovation(monkeypatch):
    head = _small_mean_head(application_scale=0.0)
    head.eval()
    final = head.residual_mean_model.net[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(0.75)

    def factory(_cfg, _sde, shape, device="cpu"):
        def sampler(_score_fn, conditioning, generator=None):
            del generator
            return torch.full(
                (conditioning.shape[0], *shape),
                0.25,
                device=device,
                dtype=conditioning.dtype,
            )

        return sampler

    monkeypatch.setattr(
        "granitewxc.decoders.diffusion_head.build_sampler", factory
    )
    cond = torch.zeros(1, 3, 4, 4)
    baseline = torch.full((1, 2, 4, 4), 2.0)
    full, raw_correction = head.sample_components(
        cond,
        4,
        4,
        baseline_std=baseline,
        application_scale=0.5,
    )
    assert torch.allclose(raw_correction, torch.ones_like(raw_correction))
    assert torch.allclose(full, baseline + 0.5 * raw_correction)
    assert torch.allclose(
        head.get_last_residual_mean(),
        torch.full_like(raw_correction, 0.75),
    )


class _CorrectionDecoder(ClimateECCCFinetuneWrapper):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.output_scalers_sigma = torch.ones(1, 2, 1, 1)
        self.output_scalers_mu = torch.zeros(1, 2, 1, 1)
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.zeros(2, dtype=torch.int64),
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.zeros(2, dtype=torch.bool),
        )
        self.output_conv_block = torch.nn.Conv2d(3, 2, kernel_size=1)
        with torch.no_grad():
            self.output_conv_block.weight.zero_()
            self.output_conv_block.bias.copy_(torch.tensor([0.2, 0.4]))
        self.diffusion_head = _small_mean_head()
        final = self.diffusion_head.residual_mean_model.net[-1]
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.tensor([0.1, -0.2]))
        self._last_precip_hurdle_aux = None
        self._last_bg_aux = None

    def _resolve_output_scalers(self, value, scaler_offset=None):
        del scaler_offset
        return (
            self.output_scalers_mu.to(value),
            self.output_scalers_sigma.to(value),
        )

    def _decode_outputs(self, raw, scaler_offset=None, wet_logits=None):
        del wet_logits
        return self._decode_targets_std(raw, scaler_offset=scaler_offset), raw


def test_model_training_forward_exposes_differentiable_corrected_mean():
    decoder = _CorrectionDecoder()
    decoder.train()
    cond = torch.randn(1, 3, 4, 4, requires_grad=True)
    batch = {"y": torch.zeros(1, 2, 4, 4)}
    result = decoder._diffusion_forward(cond, batch)
    assert set(result) >= {
        "baseline_prediction",
        "corrected_mean_prediction",
        "diffusion_loss",
    }
    expected_baseline = torch.stack(
        [torch.full((1, 4, 4), 0.2), torch.full((1, 4, 4), 0.4)],
        dim=1,
    )
    expected_corrected = torch.stack(
        [torch.full((1, 4, 4), 0.3), torch.full((1, 4, 4), 0.2)],
        dim=1,
    )
    assert torch.allclose(result["baseline_prediction"], expected_baseline)
    assert torch.allclose(
        result["corrected_mean_prediction"], expected_corrected
    )

    result["corrected_mean_prediction"].sum().backward()
    final = decoder.diffusion_head.residual_mean_model.net[-1]
    assert final.bias.grad is not None
    assert decoder.output_conv_block.weight.grad is None
    assert cond.grad is None


class _RecordingConfiguredLoss:
    def __init__(self):
        self.calls: list[torch.Tensor] = []
        self._last_terms: dict[str, float] = {}

    def __call__(self, prediction, batch):
        self.calls.append(prediction)
        value = (prediction - batch["y"]).square().mean()
        self._last_terms = {"arbitrary.yaml.term": float(value.detach())}
        return value

    def get_last_terms(self):
        return dict(self._last_terms)


def test_joint_loss_reuses_same_configured_loss_and_separates_gradients():
    configured = _RecordingConfiguredLoss()
    joint = JointResidualDiffusionLoss(
        configured,
        deterministic_weight=0.5,
        corrected_mean_weight=2.0,
        diffusion_weight=3.0,
    )
    baseline = torch.ones(1, 1, 2, 2, requires_grad=True)
    correction = torch.full((1, 1, 2, 2), -0.5, requires_grad=True)
    diffusion = torch.tensor(2.0, requires_grad=True)
    batch = {"y": torch.zeros_like(baseline)}
    total = joint(
        {
            "baseline_prediction": baseline,
            "corrected_mean_prediction": baseline.detach() + correction,
            "diffusion_loss": diffusion,
        },
        batch,
    )
    assert total.item() == pytest.approx(7.0)
    total.backward()
    assert len(configured.calls) == 2
    assert torch.allclose(baseline.grad, torch.full_like(baseline, 0.25))
    assert torch.allclose(correction.grad, torch.full_like(correction, 0.5))
    assert diffusion.grad.item() == pytest.approx(3.0)
    terms = joint.get_last_terms()
    assert terms["baseline.arbitrary.yaml.term"] == pytest.approx(1.0)
    assert terms["corrected_mean.arbitrary.yaml.term"] == pytest.approx(0.25)


def test_improvement_hinge_cannot_make_baseline_worse():
    configured = _RecordingConfiguredLoss()
    joint = JointResidualDiffusionLoss(
        configured,
        deterministic_weight=0.0,
        corrected_mean_weight=0.0,
        diffusion_weight=0.0,
        improvement_penalty_weight=1.0,
        minimum_relative_improvement=0.1,
    )
    baseline = torch.full((1, 1, 1, 1), 2.0, requires_grad=True)
    corrected = torch.full((1, 1, 1, 1), 2.0, requires_grad=True)
    diffusion = torch.tensor(0.0, requires_grad=True)
    loss = joint(
        {
            "baseline_prediction": baseline,
            "corrected_mean_prediction": corrected,
            "diffusion_loss": diffusion,
        },
        {"y": torch.zeros_like(baseline)},
    )
    assert loss.item() == pytest.approx(0.4)
    loss.backward()
    assert baseline.grad is not None
    assert torch.count_nonzero(baseline.grad).item() == 0
    assert corrected.grad is not None
    assert corrected.grad.item() > 0.0


def _builder_config(*, precip_model: str = "single_head"):
    return SimpleNamespace(
        model=SimpleNamespace(
            head_type="diffusion",
            diffusion={"residual_diffusion": True},
        ),
        precip_model=precip_model,
        mask_unit_size=(2, 2),
        loss={
            "base": "rmse",
            "deterministic_weight": 1.0,
            "diffusion": {
                "weight": 0.25,
                "corrected_mean_weight": 1.0,
                "improvement_penalty_weight": 0.5,
                "minimum_relative_improvement": 0.02,
            },
            "predictands": {
                "tasmax": {
                    "distribution_loss": {
                        "enabled": True,
                        "method": "moment",
                        "weight": 0.1,
                    }
                }
            },
            "spatial_gradient": {
                "enabled": True,
                "weight": 0.2,
                "predictands": {"pr": 0.5, "tasmax": 1.0},
                "precipitation_wet_only": False,
            },
        },
    )


def test_builder_applies_yaml_composite_to_baseline_and_corrected_mean():
    loss = build_loss_fn(_builder_config(), ["pr", "tasmax"])
    torch.manual_seed(3)
    target = torch.randn(2, 2, 4, 4)
    baseline = torch.zeros_like(target)
    corrected = torch.full_like(target, 0.1)
    loss(
        {
            "baseline_prediction": baseline,
            "corrected_mean_prediction": corrected,
            "diffusion_loss": torch.tensor(0.5),
        },
        {"y": target},
    )
    terms = loss.get_last_terms()
    for prefix in ("baseline", "corrected_mean"):
        assert f"{prefix}.base.rmse" in terms
        assert f"{prefix}.spatial_gradient" in terms
        assert f"{prefix}.tasmax.distribution.moment" in terms
    description = loss.describe()
    assert description["corrected_mean_weight"] == pytest.approx(1.0)
    assert description["minimum_relative_improvement"] == pytest.approx(0.02)


def test_builder_rejects_auxiliary_precip_head_for_corrected_field():
    with pytest.raises(ValueError, match="requires precip_model='single_head'"):
        build_loss_fn(_builder_config(precip_model="bernoulli_gamma"), ["pr", "tasmax"])


class _OneBatchDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "x": torch.zeros(1, 2, 2),
            "y": torch.full((1, 2, 2), 2.0),
        }


class _ValidationLoss:
    minimum_relative_improvement = 0.0

    def __call__(self, prediction, batch):
        del prediction
        return torch.zeros((), device=batch["y"].device)

    def evaluate_configured_prediction(self, prediction, batch):
        value = (prediction - batch["y"]).square().mean()
        return value, {"unlisted.custom": float(value.detach())}


class _ValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.sample_calls = 0
        self._baseline = None

    def forward(
        self,
        batch,
        return_pre_inverse=False,
        diffusion_generator=None,
        diffusion_application_scale=None,
        **kwargs,
    ):
        del diffusion_generator, kwargs
        if not return_pre_inverse:
            return self.anchor * 0.0
        assert diffusion_application_scale == 1.0
        value = 1.0 if self.sample_calls % 2 == 0 else 3.0
        self.sample_calls += 1
        sample = torch.full_like(batch["y"], value)
        baseline = torch.zeros_like(batch["y"])
        self._baseline = (baseline, baseline)
        return sample, sample

    def get_last_diffusion_baseline(self):
        return self._baseline


def test_validation_compares_actual_ensemble_mean_with_configured_loss():
    config = {
        "ensemble_size": 2,
        "base_seed": 11,
        "application_scale": 1.0,
        "minimum_relative_improvement": 0.0,
        "require_each_configured_term_non_degradation": True,
        "term_tolerance": 0.0,
        "limit_steps": 1,
        "sampling_method": "ddim",
        "num_sampling_steps": 4,
        "eta": 0.0,
    }
    _, metrics = validate_one_epoch(
        model=_ValidationModel(),
        local_rank=0,
        validation_loader=DataLoader(_OneBatchDataset(), batch_size=1),
        loss_func=_ValidationLoss(),
        epoch=0,
        gpu=False,
        limit_steps=1,
        correction_validation=config,
    )
    assert metrics["val.correction.baseline.total"] == pytest.approx(4.0)
    assert metrics["val.correction.ensemble_mean.total"] == pytest.approx(0.0)
    assert metrics[
        "val.correction.baseline.term.unlisted.custom"
    ] == pytest.approx(4.0)
    assert metrics[
        "val.correction.ensemble_mean.term.unlisted.custom"
    ] == pytest.approx(0.0)
    assert metrics["val.correction.qualified"] == 1.0


class _TrainingLossStub:
    minimum_relative_improvement = 0.02

    def evaluate_configured_prediction(self, prediction, batch):
        del batch
        return prediction.mean(), {}


def test_best_checkpoint_requires_qualified_ensemble(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = SimpleNamespace(
        validation_enabled=True,
        validation={
            "residual_correction": {
                "enabled": True,
                "ensemble_size": 2,
                "application_scale": 1.0,
            }
        },
        model=SimpleNamespace(
            diffusion={
                "sampling_method": "ddim",
                "num_sampling_steps": 4,
                "eta": 0.0,
            }
        ),
        checkpoint_dir=str(tmp_path),
        path_experiment=str(tmp_path),
        auto_resume_if_checkpoint_exists=False,
        resume_training=False,
        resume_from_last_checkpoint=False,
        resume_checkpoint_path=None,
        num_epochs=2,
        limit_steps_train=1,
        limit_steps_valid=1,
        gradient_accumulation_steps=1,
        save_epoch_checkpoints=False,
    )
    monkeypatch.setattr(
        trainer_module,
        "train_one_epoch",
        lambda **kwargs: (torch.tensor(1.0), {}),
    )
    validation_calls = {"count": 0}

    def fake_validation(**kwargs):
        del kwargs
        call = validation_calls["count"]
        validation_calls["count"] += 1
        qualified = float(call == 1)
        ensemble_loss = 0.9 if call == 0 else 0.8
        return torch.tensor(0.5 - 0.1 * call), {
            "val.correction.qualified": qualified,
            "val.correction.baseline.total": 1.0,
            "val.correction.ensemble_mean.total": ensemble_loss,
            "val.correction.relative_improvement": 1.0 - ensemble_loss,
        }

    monkeypatch.setattr(trainer_module, "validate_one_epoch", fake_validation)
    checkpoint_calls = []
    monkeypatch.setattr(
        trainer_module,
        "save_checkpoint",
        lambda **kwargs: checkpoint_calls.append(kwargs),
    )
    trainer_module.train_model(
        config,
        model,
        train_dl=[{}],
        val_dl=[{}],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        local_rank=0,
        use_gpu=False,
        save_every=1,
        loss_func=_TrainingLossStub(),
    )
    epoch_calls = [call for call in checkpoint_calls if call["epoch"] >= 0]
    assert [call["is_best"] for call in epoch_calls] == [False, True]
    assert epoch_calls[0]["correction_report"]["qualified"] is False
    assert epoch_calls[1]["correction_report"]["qualified"] is True
    assert epoch_calls[1]["best_correction_loss"] == pytest.approx(0.8)


def test_ddim_explicit_generator_is_reproducible_and_rng_isolated():
    sde = VPSDE(beta_min=0.1, beta_max=2.0, N=5)
    sampler = get_ddim_sampler(
        sde,
        shape=(1, 2, 2),
        eta=0.0,
        denoise=True,
        eps=1e-3,
        device="cpu",
    )
    cond = torch.zeros(1, 1, 2, 2)

    def score_fn(x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)

    global_state = torch.random.get_rng_state().clone()
    first = sampler(
        score_fn, cond, generator=torch.Generator().manual_seed(123)
    )
    assert torch.equal(global_state, torch.random.get_rng_state())
    second = sampler(
        score_fn, cond, generator=torch.Generator().manual_seed(123)
    )
    third = sampler(
        score_fn, cond, generator=torch.Generator().manual_seed(124)
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
