"""Trainer regression tests: actual updates, immutable units, and exact resume."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from granitewxc.temporal import training
from granitewxc.temporal.checkpoint import MigrationReport, TemporalCheckpointError
from granitewxc.temporal.model import apply_freeze_policy, build_param_groups
from tests.temporal_fixtures import temporal_config, make_sequence_batch, build_tiny_model, tiny_config_dict


class ScalarDataset(Dataset):
    def __init__(self, n=7):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        return {"x": torch.full((1, 1, 1, 1), float(index + 1)),
                "y": torch.ones(1, 1, 1, 1), "time_features": torch.zeros(1, 5)}
    def describe(self):
        return {"fixture": "scalar", "n": self.n}
    def set_epoch(self, epoch):
        self.epoch = epoch


class ScalarRunner(nn.Module):
    def __init__(self, noisy=False):
        super().__init__()
        self.base = nn.Module()
        self.base.temporal_adapter = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.base.temporal_adapter.weight, 0.2)
        self.base.output_scalers_sigma = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=False)
        self.noisy = noisy
    def forward(self, batch):
        prediction = self.base.temporal_adapter(batch["x"])
        if self.noisy:
            prediction = prediction + torch.rand_like(prediction) * 0.2
        return SimpleNamespace(predictions=prediction, target_frames=batch["y"], valid_mask=None,
                               emitted_indices=(0,), interval_ratio=torch.ones(prediction.shape[:2]),
                               backbone_evaluations=1)


def scalar_setup(monkeypatch, n=7, noisy=False, resume=None):
    runners = []
    monkeypatch.setattr("granitewxc.temporal.checkpoint.contract_from_model",
                        lambda *a: SimpleNamespace(to_dict=lambda: {"fixture": "scalar"}))
    def loaders(config, cfg, **kwargs):
        ds = ScalarDataset(n)
        return {"train": DataLoader(ds, batch_size=1,
                    sampler=training.EpochSequenceSampler(ds, cfg.seed),
                    generator=torch.Generator().manual_seed(cfg.seed)),
                "validation": DataLoader(ds, batch_size=1,
                    generator=torch.Generator().manual_seed(cfg.seed + 1))}
    def model(config, cfg, **kwargs):
        runner = ScalarRunner(noisy)
        payload = torch.load(cfg.resume_from_temporal_checkpoint, weights_only=False) if cfg.resume_from_temporal_checkpoint else {}
        if payload:
            runner.base.load_state_dict(payload["model"])
        runners.append(runner)
        return runner, MigrationReport(source="synthetic", kind="spatial_init"), payload
    monkeypatch.setattr(training, "build_sequence_dataloaders", loaders)
    monkeypatch.setattr(training, "build_temporal_model", model)
    monkeypatch.setattr(training, "build_loss_fn", lambda *a: lambda prediction, batch: ((prediction-batch["y"])**2).mean())
    config = SimpleNamespace(data=SimpleNamespace(output_vars=["pr"]), batch_size=1,
                             num_epochs=1, gradient_accumulation_steps=4, max_grad_norm=0)
    cfg = temporal_config(losses={"per_frame_weight": 1.0})
    return config, cfg, runners


def test_budget_counts_updates_and_partial_accumulation(monkeypatch, tmp_path):
    config, cfg, runners = scalar_setup(monkeypatch, n=7)
    result = training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path,
                max_steps_per_epoch=2, verbose=False)
    assert result["global_step"] == result["trained_temporal_steps"] == 2
    assert result["counters"]["microbatches"] == result["counters"]["backward_passes"] == 7
    assert result["counters"]["accumulation_cycles"] == 2
    # Independently reproduce two optimizer updates using actual mean batch losses.
    reference = ScalarRunner()
    optimizer = torch.optim.AdamW(reference.base.temporal_adapter.parameters(), lr=cfg.freeze.lr_temporal, weight_decay=0)
    order = list(training.EpochSequenceSampler(ScalarDataset(7), cfg.seed))
    for indices in (order[:4], order[4:]):
        optimizer.zero_grad()
        losses = [((reference.base.temporal_adapter(torch.tensor([[float(i+1)]])) - 1)**2).mean() for i in indices]
        torch.stack(losses).mean().backward()
        optimizer.step()
    torch.testing.assert_close(runners[0].base.temporal_adapter.weight, reference.base.temporal_adapter.weight, rtol=0, atol=1e-8)
    assert torch.equal(runners[0].base.output_scalers_sigma, torch.ones(1, 1, 1, 1))


def test_one_update_smoke_uses_four_microbatches(monkeypatch, tmp_path):
    config, cfg, _ = scalar_setup(monkeypatch)
    result = training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path,
                max_steps_per_epoch=1, verbose=False)
    assert result["global_step"] == 1
    assert result["counters"]["microbatches"] == 4
    with pytest.raises(ValueError, match="at least one"):
        training.train_temporal_model(config, cfg, output_dir=tmp_path, max_steps_per_epoch=0)


def test_epoch_resume_matches_uninterrupted_rng_order_and_optimizer(monkeypatch, tmp_path):
    config, cfg, runners = scalar_setup(monkeypatch, noisy=True)
    continuous = training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path/"continuous",
                max_steps_per_epoch=1, num_epochs=2, verbose=False)
    expected = copy.deepcopy(runners[-1].base.state_dict())
    training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path/"split",
                max_steps_per_epoch=1, num_epochs=1, verbose=False)
    resumed_cfg = replace(cfg, init_from_spatial_checkpoint=None, resume_from_temporal_checkpoint=str(tmp_path/"split"/"last.ckpt"))
    resumed = training.train_temporal_model(config, resumed_cfg, device="cpu", output_dir=tmp_path/"resumed",
                max_steps_per_epoch=1, num_epochs=2, verbose=False)
    assert resumed["resume_status"] == "exact_epoch_boundary"
    for name, value in expected.items():
        assert torch.equal(value, runners[-1].base.state_dict()[name]), name
    for key in ("global_step", "microbatches", "backward_passes", "target_samples", "accumulation_cycles"):
        assert continuous["counters"][key] == resumed["counters"][key]
    assert resumed["best_val"] == continuous["best_val"]


def test_mid_epoch_resume_matches_uninterrupted(monkeypatch, tmp_path):
    config, cfg, runners = scalar_setup(monkeypatch, noisy=True, n=11)
    config.gradient_accumulation_steps = 2
    training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path/"continuous",
                max_steps_per_epoch=4, verbose=False, checkpoint_interval_updates=1)
    expected = copy.deepcopy(runners[-1].base.state_dict())
    def stop(model, optimizer, state):
        if state.global_step == 3:
            raise RuntimeError("simulated interrupt")
    with pytest.raises(RuntimeError, match="simulated"):
        training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path/"split",
                max_steps_per_epoch=4, verbose=False, checkpoint_interval_updates=1,
                optimizer_step_callback=stop)
    saved = torch.load(tmp_path/"split"/"last.ckpt", weights_only=False)
    assert saved["global_step"] == 2
    resumed_cfg = replace(cfg, init_from_spatial_checkpoint=None, resume_from_temporal_checkpoint=str(tmp_path/"split"/"last.ckpt"))
    result = training.train_temporal_model(config, resumed_cfg, device="cpu", output_dir=tmp_path/"resumed",
                max_steps_per_epoch=4, verbose=False)
    assert result["resume_status"] == "exact_update_boundary"
    assert result["global_step"] == 4
    assert result["counters"]["microbatches"] == 8
    for name, value in expected.items():
        assert torch.equal(value, runners[-1].base.state_dict()[name]), name


def test_nonfinite_gradient_does_not_count_as_update(monkeypatch, tmp_path):
    config, cfg, _ = scalar_setup(monkeypatch, n=8)
    def before(model, optimizer, state):
        calls = [0]
        def gradient(grad):
            calls[0] += 1
            return torch.full_like(grad, float("inf")) if calls[0] <= 4 else grad
        model.temporal_adapter.weight.register_hook(gradient)
    result = training.train_temporal_model(config, cfg, device="cpu", output_dir=tmp_path,
                max_steps_per_epoch=1, verbose=False, before_training_callback=before)
    assert result["global_step"] == 1
    assert result["counters"]["accumulation_cycles"] == 2
    assert result["counters"]["skipped_nonfinite_updates"] == 1
    assert result["counters"]["skipped_amp_overflow_updates"] == 0


def test_immutable_scalers_survive_all_unfreeze(tmp_path):
    model, _ = build_tiny_model(tmp_path)
    cfg = temporal_config(freeze={"unfreeze_schedule": [{"epoch": 1, "modules": ["all"]}]})
    expected = {name: p.detach().clone() for name, p in model.named_parameters() if "scalers_" in name}
    assert len(expected) == 8
    for epoch in (0, 1, 2):
        apply_freeze_policy(model, cfg, epoch)
        groups = build_param_groups(model, cfg)
        optimized = {id(p) for group in groups for p in group["params"]}
        for name, parameter in model.named_parameters():
            if name in expected:
                assert not parameter.requires_grad
                assert id(parameter) not in optimized
                assert torch.equal(expected[name], parameter)


@pytest.mark.parametrize("pretext", [False, True])
def test_native_pair_actually_learns_through_real_trainer(monkeypatch, tmp_path, pretext):
    from granitewxc.utils.config import ExperimentConfig
    from granitewxc.temporal.model import TemporalSequenceModel
    spatial, _ = build_tiny_model(tmp_path)
    spatial_path = tmp_path / "spatial.ckpt"
    torch.save({"model": spatial.state_dict()}, spatial_path)
    raw = tiny_config_dict(tmp_path)
    raw["data"]["n_input_timestamps"] = 2
    raw["gradient_accumulation_steps"] = 4
    exp = ExperimentConfig.from_dict(raw)
    cfg = temporal_config(backend="native_pair", context_length=5, output_length=3,
             warmup_length=2, init_from_spatial_checkpoint=str(spatial_path),
             native_pair={"history_offsets": [1], "pretext": {
                 "masked_reconstruction": {"enabled": pretext, "weight": 0.1, "mask_ratio": 0.5},
                 "transition": {"enabled": pretext, "weight": 0.1}}})
    probe = make_sequence_batch(batch=1, frames=5)
    class SequenceFixture(ScalarDataset):
        def __getitem__(self, index):
            return {key: value[0].clone() for key, value in probe.items()}
    def loaders(config, cfg, **kwargs):
        ds = SequenceFixture(1)
        return {"train": DataLoader(ds, batch_size=1, sampler=training.EpochSequenceSampler(ds,cfg.seed)),
                "validation": DataLoader(ds, batch_size=1)}
    monkeypatch.setattr(training, "build_sequence_dataloaders", loaders)
    before_values, captured = {}, {}
    def before(model, optimizer, state):
        before_values.update({name:p.detach().clone() for name,p in model.named_parameters()})
        history = model.temporal_adapter.history_projection.weight
        groups = [g for g in optimizer.param_groups if any(p is history for p in g["params"])]
        assert history.requires_grad and len(groups) == 1
        assert groups[0]["name"] == "temporal" and groups[0]["lr"] == cfg.freeze.lr_temporal
    def after(model, optimizer, state):
        captured["model"] = model
        history = model.temporal_adapter.history_projection.weight
        assert torch.count_nonzero(history.grad) > 0
        assert not torch.equal(history, before_values["temporal_adapter.history_projection.weight"])
        for name, parameter in model.named_parameters():
            if name.startswith(("embedding.", "backbone.", "conv_before_backbone.")) or "scalers_" in name:
                assert not parameter.requires_grad
                assert torch.equal(parameter, before_values[name]), name
    result = training.train_temporal_model(exp, cfg, device="cpu", output_dir=tmp_path/"train",
                 max_steps_per_epoch=1, verbose=False,
                 before_training_callback=before, optimizer_step_callback=after)
    assert result["global_step"] == result["trained_temporal_steps"] == 1
    assert result["counters"]["microbatches"] == 1  # correctly flush 1/4 cycle
    runner = TemporalSequenceModel(captured["model"], cfg, adapter=captured["model"].temporal_adapter).eval()
    altered = {key: value.clone() for key, value in probe.items()}
    altered["x"][:, 3] += 20.0  # history only for final scored output t=4
    with torch.no_grad():
        original = runner(probe).predictions[:, -1]
        changed = runner(altered).predictions[:, -1]
    assert not torch.equal(original, changed)
    if pretext:
        metrics = result["history"][0]["train"]
        assert metrics["pretext_masked_reconstruction"] > 0
        assert metrics["pretext_transition"] > 0



def test_staged_groups_preserve_existing_adam_moments():
    learned = nn.Parameter(torch.tensor([1.0]))
    frozen = nn.Parameter(torch.tensor([2.0]))
    optimizer = training._update_optimizer_groups(None, [{"name": "temporal", "params": [learned], "lr": 0.01}])
    learned.square().sum().backward()
    optimizer.step()
    moments = {key: value.clone() if torch.is_tensor(value) else value for key,value in optimizer.state[learned].items()}
    updated = training._update_optimizer_groups(optimizer, [
        {"name": "temporal", "params": [learned], "lr": 0.01},
        {"name": "backbone", "params": [frozen], "lr": 0.0001}])
    for key,value in moments.items():
        assert torch.equal(updated.state[learned][key], value)
    assert frozen not in updated.state


def test_native_checkpoint_contract_validates_layout_and_legacy_notes(tmp_path):
    from tests.test_temporal_native_pair import _pair_config, _build_pair_model
    from granitewxc.temporal.checkpoint import save_temporal_checkpoint, resume_temporal_checkpoint
    cfg = _pair_config(tmp_path)
    model, _ = _build_pair_model(tmp_path,cfg)
    path = tmp_path/"model.ckpt"
    save_temporal_checkpoint(path, model=model, cfg=cfg, output_vars=["pr","tasmax"],
        time_feature_names=[],epoch=0,global_step=1,trained_temporal_steps=1)
    report, payload = resume_temporal_checkpoint(model,path,cfg)
    assert report.ok
    payload["temporal"].pop("native_pair_contract")
    torch.save(payload,path)
    report,_ = resume_temporal_checkpoint(model,path,cfg)
    assert report.ok and report.contract_notes
    payload["temporal"]["native_pair_contract"] = {"history_projection":"incompatible"}
    torch.save(payload,path)
    with pytest.raises(TemporalCheckpointError,match="native_pair_contract"):
        resume_temporal_checkpoint(model,path,cfg)



def test_resume_at_update_budget_finishes_validation_without_an_extra_step(monkeypatch,tmp_path):
    """An interruption after the last periodic save must not repeat an update."""
    config,cfg,runners=scalar_setup(monkeypatch,noisy=True,n=11)
    config.gradient_accumulation_steps=2
    continuous=training.train_temporal_model(config,cfg,device="cpu",output_dir=tmp_path/"continuous",
        max_steps_per_epoch=4,max_val_steps=2,verbose=False,checkpoint_interval_updates=1)
    expected=copy.deepcopy(runners[-1].base.state_dict())
    evaluate=training.evaluate_split
    def interrupted_validation(*args,**kwargs):
        raise RuntimeError("interrupted before validation")
    monkeypatch.setattr(training,"evaluate_split",interrupted_validation)
    with pytest.raises(RuntimeError,match="interrupted before validation"):
        training.train_temporal_model(config,cfg,device="cpu",output_dir=tmp_path/"interrupted",
            max_steps_per_epoch=4,max_val_steps=2,verbose=False,checkpoint_interval_updates=1)
    path=tmp_path/"interrupted/last.ckpt"
    saved=torch.load(path,weights_only=False)
    assert saved["global_step"]==4
    assert saved["training_state"]["epoch_complete"] is False
    assert saved["training_state"]["epoch_progress"]["optimizer_updates"]==4
    monkeypatch.setattr(training,"evaluate_split",evaluate)
    resumed_cfg=replace(cfg,init_from_spatial_checkpoint=None,resume_from_temporal_checkpoint=str(path))
    def forbidden_update(*args):
        pytest.fail("Resuming at the update budget must perform validation only")
    resumed=training.train_temporal_model(config,resumed_cfg,device="cpu",output_dir=tmp_path/"resumed",
        max_steps_per_epoch=4,max_val_steps=2,verbose=False,optimizer_step_callback=forbidden_update)
    assert resumed["global_step"]==resumed["trained_temporal_steps"]==4
    assert resumed["counters"]["microbatches"]==resumed["counters"]["backward_passes"]==8
    assert resumed["best_val"]==continuous["best_val"]
    for name,value in expected.items():
        assert torch.equal(value,runners[-1].base.state_dict()[name]),name
    completed=torch.load(tmp_path/"resumed/last.ckpt",weights_only=False)
    assert completed["training_state"]["epoch_complete"] is True
    assert (tmp_path/"resumed/best.ckpt").exists()
