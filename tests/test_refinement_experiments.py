"""Versioned mean/remainder/native-auxiliary/support experiment regressions."""
import copy
from dataclasses import replace

import pytest
import torch

from granitewxc.refinement.base import build_refiner
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.mean_correction import SignedResidualMeanCorrector, FrozenMeanComposition
from granitewxc.refinement.experiments import (ExperimentalRefiner, TotalPrecipitationSupport,
    experiment_recipes, select_channels, RainfallRegimes, gradient_diagnostics)
from test_refinement_models import REFINERS, small_config

PROVENANCE = {"phase1_fingerprint": "frozen-phase1", "conditioning_fingerprint": "inference-inputs-only", "training_selection_fingerprint": "fitting-dates"}
TARGET = {"version": "direct_signed_physical_residual_v1", "normalizer_fingerprint": "fitting-residual-statistics"}


def fields():
    torch.manual_seed(7)
    cond = torch.randn(3, 4, 9, 11)
    base = torch.zeros(3, 2, 9, 11)
    base[:, 0] = 2.
    truth = base + torch.stack((.4*cond[:, 0]-.3, .7*cond[:, 1]+.5), dim=1)
    valid = torch.ones_like(truth, dtype=torch.bool)
    return cond, base, truth, valid


def mean_model():
    cond, base, truth, valid = fields()
    model = SignedResidualMeanCorrector(4, ("pr", "tasmax"), hidden_channels=8, depth=1)
    model.fit_scaling(truth-base, valid, split="train", provenance=PROVENANCE)
    return model


def test_mean_signed_mse_learning_and_inference_inputs_only():
    cond, base, truth, valid = fields()
    model = mean_model()
    assert torch.equal(model(cond), torch.zeros_like(base))
    initial = float(model.training_loss(cond, base, truth, valid)["loss"].detach())
    opt = torch.optim.Adam(model.parameters(), lr=.025)
    for _ in range(120):
        opt.zero_grad()
        loss = model.training_loss(cond, base, truth, valid)["loss"]
        loss.backward()
        opt.step()
    assert float(loss.detach()) < .02 * initial
    corrections = model(cond)
    assert (corrections < 0).any() and (corrections > 0).any()
    assert torch.equal(base, fields()[1])
    cond.requires_grad_(True)
    model(cond).sum().backward()
    assert cond.grad is None


def test_mean_checkpoint_strict_statistics_provenance_and_corruption():
    model = mean_model()
    checkpoint = model.checkpoint()
    loaded = SignedResidualMeanCorrector.from_checkpoint(checkpoint, expected_provenance=PROVENANCE)
    assert loaded.fingerprint() == model.fingerprint()
    with pytest.raises(ValueError, match="provenance"):
        SignedResidualMeanCorrector.from_checkpoint(checkpoint, expected_provenance={**PROVENANCE, "phase1_fingerprint": "different"})
    bad = copy.deepcopy(checkpoint)
    bad["state_dict"]["scaling.scale"] *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        SignedResidualMeanCorrector.from_checkpoint(bad)
    with pytest.raises(ValueError, match="split='train'"):
        model.fit_scaling(torch.ones(1, 2, 2, 2), split="validation", provenance=PROVENANCE)
    with pytest.raises(RuntimeError, match="already fitted"):
        model.fit_scaling(torch.ones(1, 2, 2, 2), split="train", provenance=PROVENANCE)


@pytest.mark.parametrize("head", REFINERS)
def test_frozen_mean_composition_fresh_stats_exact_reconstruction_no_ensemble_centering(head):
    cond, base, truth, valid = fields()
    mean = mean_model()
    with torch.no_grad():
        mean.net[-1].bias.copy_(torch.tensor([1.5, -.75]))
    composition = FrozenMeanComposition(mean)
    composition.train()
    assert not mean.training and not any(p.requires_grad for p in mean.parameters())
    composition.fit_remainder_statistics([{"baseline": base, "truth": truth, "conditioning": cond, "valid": valid}], split="train", provenance=PROVENANCE)
    assert not torch.equal(composition.normalizer.mean, mean.scaling.mean)
    prepared = composition.prepare(base, cond, truth, valid)
    cfg = resolve_refinement_config(small_config(head))
    refiner = ExperimentalRefiner(cfg, experiment_recipes(cfg, ("pr", "tasmax"))["native"], cond_channels=4)
    members = prepared["remainder_normalized"][:, None].repeat(1, 3, 1, 1, 1)
    reconstructed = composition.reconstruct(base, cond, members, refiner)
    assert torch.equal(reconstructed["baseline_physical"], base)
    assert torch.allclose(reconstructed["raw_members_physical"], truth[:, None].expand_as(members), atol=1e-6)
    shifted = composition.reconstruct(base, cond, members+1., refiner)
    assert not torch.allclose(shifted["raw_ensemble_mean_physical"], reconstructed["raw_ensemble_mean_physical"])
    assert composition.semantic_contract()["mean_fingerprint"] == mean.fingerprint()
    with torch.no_grad():
        mean.net[-1].bias.add_(1.)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        composition.semantic_contract()


@pytest.mark.parametrize("head", REFINERS)
def test_native_process_exact_target_loss_sampler_and_nonzero_updates(head):
    cond, _, truth, valid = fields()
    cfg = resolve_refinement_config(small_config(head))
    exp = ExperimentalRefiner(cfg, experiment_recipes(cfg, ("pr", "tasmax"))["native"], cond_channels=4)
    out = exp.training_loss(truth, cond, valid, generator=torch.Generator().manual_seed(3))
    native = exp.refiner.training_loss(truth, cond, valid, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(out["loss"], native["process_loss"], atol=1e-7)
    assert torch.equal(out["prediction"], native["prediction"])
    before = [p.clone() for p in exp.refiner.net.parameters()]
    opt = torch.optim.Adam([p for p in exp.parameters() if p.requires_grad], lr=.001)
    out["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in exp.parameters())
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, exp.refiner.net.parameters()))
    if cfg.uses_transformer:
        assert torch.equal(exp.refiner.correction_gate, torch.ones_like(exp.refiner.correction_gate))
        assert not exp.refiner.correction_gate.requires_grad
    a = exp.sample(cond, generator=torch.Generator().manual_seed(9))
    b = exp.refiner.sample(cond, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)


@pytest.mark.parametrize("head", REFINERS)
def test_full_recipe_matches_original_scalar_and_every_auxiliary_is_individual(head):
    cond, _, target, valid = fields()
    cfg = resolve_refinement_config(small_config(head))
    cfg = replace(cfg, flow_matching=replace(cfg.flow_matching, mean_path_loss_weight=.25))
    recipes = experiment_recipes(cfg, ("pr", "tasmax"))
    full = ExperimentalRefiner(cfg, recipes["existing_full_recipe"], cond_channels=4)
    original = build_refiner(cfg, residual_channels=2, cond_channels=4)
    original.load_state_dict(full.refiner.state_dict(), strict=True)
    expected = original.training_loss(target, cond, valid, generator=torch.Generator().manual_seed(99))["loss"]
    observed = full.training_loss(target, cond, valid, generator=torch.Generator().manual_seed(99))["loss"]
    assert torch.allclose(expected, observed, atol=1e-6)
    for name in ("reconstruction_loss", "multiscale_loss", "gradient_loss", "mean_bias_loss"):
        recipe = recipes[name]
        active = [k for k, v in recipe.to_dict().items() if k.endswith("_loss") and k != "process_loss" and v > 0]
        assert active == [name]
    if cfg.uses_transformer:
        closed = ExperimentalRefiner(cfg, recipes["native_production_gate"], cond_channels=4)
        gate_loss = closed.training_loss(target, cond, valid)["loss"]
        gate_loss.backward()
        assert closed.refiner.correction_gate.grad is None or torch.count_nonzero(closed.refiner.correction_gate.grad) == 0
        assert torch.count_nonzero(closed.apply_correction_gate(torch.ones_like(target))) == 0
        calibrated = ExperimentalRefiner(cfg, recipes["gate_calibration_loss"], cond_channels=4)
        # A zero clean-output initialization legitimately gives zero gate gradient
        # on update one. Native process learning must establish a nonzero output.
        optimizer = torch.optim.Adam(calibrated.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            calibrated.training_loss(target, cond, valid)["loss"].backward()
            optimizer.step()
        assert calibrated.refiner.correction_gate.grad is not None
        assert calibrated.refiner.correction_gate.grad.abs().sum() > 0


@pytest.mark.parametrize("head", REFINERS)
def test_experiment_checkpoint_contract_and_selected_single_variables(head):
    cfg = resolve_refinement_config(small_config(head))
    recipes = experiment_recipes(cfg, ("pr", "tasmax"))
    for variable in ("pr", "tasmax"):
        model = ExperimentalRefiner(cfg, recipes[variable+"_only"], cond_channels=4)
        cond, _, target, valid = fields()
        loss = model.training_loss(select_channels(target, ("pr", "tasmax"), (variable,)), cond,
                                   select_channels(valid, ("pr", "tasmax"), (variable,)))["loss"]
        assert torch.isfinite(loss)
        cp = model.checkpoint(target_contract=TARGET, provenance=PROVENANCE)
        loaded = ExperimentalRefiner.from_checkpoint(cp, expected_target_contract=TARGET, expected_provenance=PROVENANCE)
        assert loaded.recipe == model.recipe
        with pytest.raises(ValueError, match="mismatch"):
            ExperimentalRefiner.from_checkpoint(cp, expected_target_contract={**TARGET, "normalizer_fingerprint": "old-y-b"}, expected_provenance=PROVENANCE)
        bad = copy.deepcopy(cp)
        bad["contract"]["recipe"]["gate_policy"] = "production_trainable"
        with pytest.raises(ValueError, match="fingerprint"):
            ExperimentalRefiner.from_checkpoint(bad, expected_target_contract=TARGET, expected_provenance=PROVENANCE)


def test_total_precipitation_support_roundtrip_zero_trace_signed_other_channels():
    y = torch.tensor([[[[0., .0001, .01, 100.]], [[270., 275., 260., 285.]], [[1., 2., 3., 4.]]]])
    b = torch.tensor([[[[8., 9., 10., 11.]], [[271., 274., 272., 280.]], [[4., 3., 2., 1.]]]])
    mask = torch.ones_like(y, dtype=torch.bool)
    support = TotalPrecipitationSupport(("pr", "tasmax", "other"))
    support.fit(y, b, mask, split="train", provenance=PROVENANCE)
    z = support.target(y, b, mask)
    out = support.reconstruct(z[:, None], b, gate_policy="fixed_unit_fresh")
    assert torch.allclose(out["members_physical"][:, 0], y, atol=1e-6)
    assert out["members_physical"][0, 0, 0, 0, 1] > 0
    latent = support.normalizer.denormalize(z)
    latent[:, 0] = -2
    negative = support.normalizer.normalize(latent)
    supported = support.reconstruct(negative[:, None], b, gate_policy="fixed_unit_fresh")
    assert supported["latent_censor_mask"][:, :, 0].all()
    assert torch.count_nonzero(supported["members_physical"][:, :, 0]) == 0
    assert (supported["correction_physical"][:, :, 0] < 0).all()
    with pytest.raises(ValueError, match="incompatible"):
        support.reconstruct(z[:, None], b, gate_policy="production_trainable")
    bad = y.clone()
    bad[:, 0] = -1
    with pytest.raises(ValueError, match="Negative observed"):
        support.encode_physical(bad, b, mask)
    with pytest.raises(ValueError, match="split='train'"):
        TotalPrecipitationSupport(("pr", "tasmax", "other")).fit(y, b, mask, split="test", provenance=PROVENANCE)


@pytest.mark.parametrize("head", REFINERS)
def test_gradient_attribution_rain_regimes_partition_and_fixed_balance(head):
    cond, _, truth, valid = fields()
    truth[:, 0] = truth[:, 0].abs()
    truth[:, 0, :2] = 0
    regimes = RainfallRegimes.fit(truth[:, 0], valid[:, 0], split="train", fitting_fingerprint="fitting")
    cfg = resolve_refinement_config(small_config(head))
    model = ExperimentalRefiner(cfg, experiment_recipes(cfg, ("pr", "tasmax"))["fixed_pr_weight_2"], cond_channels=4)
    result = model.training_loss(truth, cond, valid)
    assert torch.allclose(result["loss"], (2*result["per_variable_total"][0]+result["per_variable_total"][1])/3)
    report = gradient_diagnostics(result, model, truth_physical=truth, valid=valid, regimes=regimes)
    assert report["joint_gradient_l2"] > 0
    contributions = report["rainfall_regimes"]["contributions"]
    assert sum(v["count"] for v in contributions.values()) == int(valid[:, 0].sum())
    assert sum(v["process_mse_contribution"] for v in contributions.values()) == pytest.approx(float(result["per_variable_terms"]["process_loss"][0]), rel=1e-6)
    result["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_frozen_mean_and_support_checkpoint_roundtrips_reject_swapped_statistics():
    cond, base, truth, valid = fields()
    composition = FrozenMeanComposition(mean_model())
    composition.fit_remainder_statistics([{"baseline": base, "truth": truth, "conditioning": cond, "valid": valid}], split="train", provenance=PROVENANCE)
    restored = FrozenMeanComposition.from_checkpoint(composition.checkpoint(), expected_provenance=PROVENANCE)
    assert restored.semantic_contract() == composition.semantic_contract()
    assert not restored.mean_corrector.training
    bad = copy.deepcopy(composition.checkpoint())
    bad["normalizer_state"]["mean"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenMeanComposition.from_checkpoint(bad, expected_provenance=PROVENANCE)
    truth[:, 0] = truth[:, 0].abs()
    support = TotalPrecipitationSupport(("pr", "tasmax"))
    support.fit(truth, base, valid, split="train", provenance=PROVENANCE)
    restored_support = TotalPrecipitationSupport.from_checkpoint(support.checkpoint(), expected_provenance=PROVENANCE)
    assert restored_support.semantic_contract() == support.semantic_contract()
    bad = copy.deepcopy(support.checkpoint())
    bad["state_dict"]["normalizer.scale"] *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        TotalPrecipitationSupport.from_checkpoint(bad, expected_provenance=PROVENANCE)


@pytest.mark.parametrize("head", REFINERS)
def test_every_auxiliary_rainfall_regime_attribution_sums_to_executed_loss(head):
    from granitewxc.refinement.experiments import rainfall_loss_partition
    cond, _, truth, valid = fields()
    truth[:, 0] = truth[:, 0].abs()
    truth[:, 0, :2] = 0
    valid[0, 0, 0, :] = False
    valid[1, 0, 3:5, 4] = False
    cfg = resolve_refinement_config(small_config(head))
    cfg = replace(cfg, flow_matching=replace(cfg.flow_matching, mean_path_loss_weight=.25))
    model = ExperimentalRefiner(cfg, experiment_recipes(cfg, ("pr", "tasmax"))["existing_full_recipe"], cond_channels=4)
    result = model.training_loss(truth, cond, valid)
    regimes = RainfallRegimes.fit(truth[:, 0], valid[:, 0], split="train", fitting_fingerprint="fit")
    parts = rainfall_loss_partition(result, model, truth[:, 0], valid[:, 0], regimes)
    for key, values in result["per_variable_terms"].items():
        assert torch.allclose(sum(part[key] for part in parts.values()), values[0], atol=1e-6), (head, key)
    total = sum(getattr(model.recipe, key)*sum(part[key] for part in parts.values()) for key in result["per_variable_terms"])
    assert torch.allclose(total, result["per_variable_total"][0], atol=1e-6)


@pytest.mark.parametrize("head", REFINERS)
def test_support_alternative_uses_joint_native_sampler_on_odd_domain(head):
    cond, base, truth, valid = fields()
    truth[:, 0] = truth[:, 0].abs()
    truth[:, 0, :2] = 0
    support = TotalPrecipitationSupport(("pr", "tasmax"))
    support.fit(truth, base, valid, split="train", provenance=PROVENANCE)
    cfg = resolve_refinement_config(small_config(head))
    model = ExperimentalRefiner(cfg, experiment_recipes(cfg, ("pr", "tasmax"))["native"], cond_channels=4)
    target = support.target(truth, base, valid)
    loss = model.training_loss(target, cond, valid)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    samples = model.sample(cond, generator=torch.Generator().manual_seed(8))[:, None]
    out = support.reconstruct(samples, base, gate_policy=model.recipe.gate_policy)
    assert torch.isfinite(out["members_physical"]).all()
    assert (out["members_physical"][:, :, 0] >= 0).all()
    assert torch.allclose(out["members_physical"][:, :, 1], base[:, None, 1] + out["unconstrained_latent"][:, :, 1])