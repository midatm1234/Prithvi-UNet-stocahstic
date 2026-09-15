"""Whole-year cross-fitting, fresh statistics and native process integration."""
import copy

import pytest
import torch

from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.crossfit import (CrossFitMeanComposition,
    CROSSFIT_REMAINDER_VERSION, make_crossfit_registration)
from granitewxc.refinement.experiments import ExperimentalRefiner, experiment_recipes
from granitewxc.refinement.mean_correction import SignedResidualMeanCorrector, FrozenMeanComposition
from test_refinement_models import REFINERS, small_config

PROVENANCE = {"phase1_fingerprint": "unchanged-phase1", "conditioning_fingerprint": "physical-inference-inputs",
              "training_selection_fingerprint": "all-fitting-date-ids"}
YEARS = [1961, 1962, 1963, 1964, 2080, 2081, 2082, 2083]
DATES = [f"{year}-01-01T12:00:00" for year in YEARS]


def registration_kwargs():
    return {"registration_id": "written-before-fitting-means",
            "fitting_years": YEARS,
            "heldout_years": [[1961, 1962, 2080, 2081], [1963, 1964, 2082, 2083]],
            "mean_training_years": [[1963, 2082], [1961, 2080]],
            "mean_selection_years": [[1964, 2083], [1962, 2081]],
            "mean_provenance": [{**PROVENANCE, "training_selection_fingerprint": f"fit-fold-{i}"} for i in range(2)],
            "era_years": {"historical": YEARS[:4], "future": YEARS[4:]},
            "selection_rule": "minimum internal-selection scaled MSE; last epoch tie break"}


def fixture():
    torch.manual_seed(11)
    kwargs = registration_kwargs()
    plan = make_crossfit_registration(**kwargs)
    checkpoints = []
    # Three arbitrary configured variables; no two-channel special case.
    for i, correction in enumerate(([2., -1., .5], [-4., 3., -1.5])):
        mean = SignedResidualMeanCorrector(4, ("pr", "tasmax", "tasmin"), hidden_channels=4, depth=1)
        scale_field = torch.tensor([-1., 1.]).view(2, 1, 1, 1).expand(2, 3, 3, 5)
        mean.fit_scaling(scale_field, split="train", provenance=kwargs["mean_provenance"][i])
        with torch.no_grad():
            # Exact signed physical constants distinguish which mean was used.
            mean.net[-1].bias.copy_(torch.tensor(correction) / mean.scaling.scale.float().reshape(-1))
        checkpoints.append(mean.checkpoint())
    comp = CrossFitMeanComposition(checkpoints, preregistration=plan)
    base = torch.zeros(8, 3, 9, 11)
    base[:, 0] = 2.
    base[:, 1:] = 290.
    cond = torch.randn(8, 4, 9, 11)
    ramp = torch.linspace(-2., 2., 99).view(1, 1, 9, 11)
    truth = base + ramp * torch.tensor([1., -.5, .2]).view(1, 3, 1, 1)
    truth[:, 0, 0, 0] = 0.  # A valid dry day is not missing.
    valid = torch.ones_like(truth, dtype=torch.bool)
    valid[0, 0, 0, 1] = False
    truth[0, 0, 0, 1] = float("nan")
    return comp, base, cond, truth, valid, checkpoints, plan


def fitted():
    comp, base, cond, truth, valid, checkpoints, plan = fixture()
    comp.fit_remainder_statistics([
        {"baseline": base[:3], "conditioning": cond[:3], "truth": truth[:3], "valid": valid[:3], "sample_ids": DATES[:3]},
        {"baseline": base[3:], "conditioning": cond[3:], "truth": truth[3:], "valid": valid[3:], "sample_ids": DATES[3:]},
    ], split="train", provenance=PROVENANCE)
    return comp, base, cond, truth, valid, checkpoints, plan


def test_oof_dispatch_order_and_no_truth_conditioning_frozen_phase1():
    comp, base, cond, _, _, checkpoints, _ = fixture()
    base.requires_grad_(True)
    cond.requires_grad_(True)
    comp.train()
    assert all(not mean.training and not any(p.requires_grad for p in mean.parameters()) for mean in comp.means)
    assert not any(p.requires_grad for p in comp.parameters())
    order = [7, 1, 5, 3, 0, 6, 2, 4]
    ordered = comp.training_mean(base, cond, DATES)
    shuffled = comp.training_mean(base[order], cond[order], [DATES[i] for i in order])
    assert torch.equal(ordered[order], shuffled)
    for i, year in enumerate(YEARS):
        expected = torch.tensor([2., -1., .5] if year in {1961, 1962, 2080, 2081} else [-4., 3., -1.5])
        assert torch.allclose(ordered[i] - base[i], expected[:, None, None], atol=1e-6)
    inferred = comp.inference_mean(base, cond)
    assert torch.allclose(inferred - base, torch.tensor([-1., 1., -.5])[None, :, None, None], atol=1e-6)
    assert not ordered.requires_grad and not inferred.requires_grad
    assert base.grad is None and cond.grad is None
    # The stored models were independently reconstructed, not aliased to a caller's state.
    checkpoints[0]["state_dict"]["net.2.bias"].add_(999.)
    assert torch.equal(ordered, comp.training_mean(base, cond, DATES))


@pytest.mark.parametrize("change,match", [
    (lambda k: k["heldout_years"][0].append(1963), "partition"),
    (lambda k: k["mean_training_years"][0].append(1961), "complement"),
    (lambda k: k["mean_selection_years"][0].append(1963), "disjoint"),
    (lambda k: k["mean_training_years"][0].remove(2082), "complement"),
    (lambda k: k["era_years"]["future"].append(1961), "Era year"),
    (lambda k: k["mean_provenance"][1].update(phase1_fingerprint="other"), "phase1"),
    (lambda k: k["mean_provenance"][1].update(training_selection_fingerprint="fit-fold-0"), "distinct"),
])
def test_preregistration_rejects_year_leakage_and_incompatible_sources(change, match):
    kwargs = registration_kwargs()
    change(kwargs)
    with pytest.raises(ValueError, match=match):
        make_crossfit_registration(**kwargs)


def test_registration_requires_all_eras_in_each_fit_and_selection():
    kwargs = registration_kwargs()
    kwargs["mean_training_years"][0] = [1963, 1964]
    kwargs["mean_selection_years"][0] = [2082, 2083]
    with pytest.raises(ValueError, match="each declared era"):
        make_crossfit_registration(**kwargs)


def test_fresh_oof_normalization_matches_independent_masked_values_and_rejects_reuse():
    comp, base, cond, truth, valid, _, _ = fitted()
    expected = truth - comp.training_mean(base, cond, DATES)
    for i in range(3):
        values = expected[:, i][valid[:, i]].double()
        assert int(comp.normalizer.count.reshape(-1)[i]) == values.numel()
        assert torch.allclose(comp.normalizer.mean.reshape(-1)[i], values.mean(), atol=1e-10)
        assert torch.allclose(comp.normalizer.scale.reshape(-1)[i], torch.sqrt(values.var()+comp.normalizer.epsilon), atol=1e-10)
    # Constant corrections with opposite signs make OOF and average-mean target scales different.
    average_remainder = truth - comp.inference_mean(base, cond)
    assert not torch.isclose(comp.normalizer.scale.reshape(-1)[0], average_remainder[:, 0][valid[:, 0]].double().std(), atol=.1)
    with pytest.raises(RuntimeError, match="already fitted"):
        comp.fit_remainder_statistics([], split="train", provenance=PROVENANCE)
    before = copy.deepcopy(comp.normalizer.state_dict())
    evaluation = comp.prepare_evaluation(base, cond, truth, valid, split="validation")
    assert torch.equal(evaluation["mean_physical"], comp.inference_mean(base, cond))
    assert all(torch.equal(before[k], v) for k, v in comp.normalizer.state_dict().items())
    with pytest.raises(ValueError, match="fitting needs OOF"):
        comp.prepare_evaluation(base, cond, truth, valid, split="train")


def test_statistics_stream_failure_is_atomic_and_never_accepts_heldout_or_duplicate_dates():
    comp, base, cond, truth, valid, _, _ = fixture()
    batch = {"baseline": base, "conditioning": cond, "truth": truth, "valid": valid, "sample_ids": DATES}
    for bad_dates, match in ((DATES[:7]+["2041-01-01"], "outside"), (DATES[:7]+[DATES[0]], "Repeated")):
        with pytest.raises(ValueError, match=match):
            comp.fit_remainder_statistics([{**batch, "sample_ids": bad_dates}], split="train", provenance=PROVENANCE)
        assert not comp.normalizer.is_fitted and comp.normalizer.count.sum() == 0
    subset = {k: v[:4] for k, v in batch.items()}
    with pytest.raises(ValueError, match="every preregistered fitting year"):
        comp.fit_remainder_statistics([subset], split="train", provenance=PROVENANCE)
    with pytest.raises(ValueError, match="split='train'"):
        comp.fit_remainder_statistics([batch], split="validation", provenance=PROVENANCE)
    with pytest.raises(ValueError, match="phase1"):
        comp.fit_remainder_statistics([batch], split="train", provenance={**PROVENANCE, "phase1_fingerprint": "changed"})
    with pytest.raises(ValueError, match="ISO calendar"):
        comp.training_mean(base, cond, list(range(8)))
    assert not comp.normalizer.is_fitted and comp.normalizer.count.sum() == 0
    comp.fit_remainder_statistics([batch], split="train", provenance=PROVENANCE)
    assert comp.normalizer.is_fitted


@pytest.mark.parametrize("head", REFINERS)
def test_all_four_fresh_remainder_training_and_signed_odd_domain_reconstruction(head):
    comp, base, cond, truth, valid, _, _ = fitted()
    prepared = comp.prepare(base, cond, truth, valid, dateids=DATES)
    cfg = resolve_refinement_config(small_config(head))
    variables = ("pr", "tasmax", "tasmin")
    model = ExperimentalRefiner(cfg, experiment_recipes(cfg, variables)["native"], cond_channels=4)
    initial = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    result = model.training_loss(prepared["remainder_normalized"], cond, prepared["valid"],
                                zero_residual=prepared["zero_anchor"], generator=torch.Generator().manual_seed(33))
    result["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
    members = prepared["remainder_normalized"][:, None].repeat(1, 3, 1, 1, 1)
    roundtrip = comp.reconstruct(base, cond, members, model, dateids=DATES)
    assert torch.allclose(roundtrip["raw_members_physical"][:, 0][valid], truth[valid], atol=2e-5)
    assert roundtrip["ungated_remainder_physical"].min() < 0 < roundtrip["ungated_remainder_physical"].max()
    inferred = comp.reconstruct(base, cond, members, model)
    expected_difference = comp.inference_mean(base, cond)-comp.training_mean(base, cond, DATES)
    assert torch.allclose(inferred["raw_members_physical"]-roundtrip["raw_members_physical"], expected_difference[:, None], atol=2e-5)
    perturbed = comp.reconstruct(base, cond, members+1., model)
    assert not torch.allclose(inferred["raw_ensemble_mean_physical"], perturbed["raw_ensemble_mean_physical"])
    # Native process sample + composition inference, no observation argument.
    sample = model.sample(cond[:1], generator=torch.Generator().manual_seed(91))
    prediction = comp.reconstruct(base[:1], cond[:1], sample[:, None], model)
    assert prediction["raw_members_physical"].shape == (1, 1, 3, 9, 11)
    assert torch.isfinite(prediction["raw_members_physical"]).all()
    ckpt = model.checkpoint(target_contract=comp.semantic_contract(), provenance=PROVENANCE)
    restored = ExperimentalRefiner.from_checkpoint(ckpt, expected_target_contract=comp.semantic_contract(), expected_provenance=PROVENANCE)
    assert torch.equal(sample, restored.sample(cond[:1], generator=torch.Generator().manual_seed(91)))


def test_checkpoint_roundtrip_binds_both_means_folds_stats_and_provenance(tmp_path):
    comp, base, cond, _, _, _, plan = fitted()
    path = tmp_path / "crossfit.pt"
    torch.save(comp.checkpoint(), path)
    checkpoint = torch.load(path, weights_only=False)
    restored = CrossFitMeanComposition.from_checkpoint(checkpoint, expected_provenance=PROVENANCE)
    assert restored.mean_fingerprint == comp.mean_fingerprint
    assert restored.semantic_contract() == comp.semantic_contract()
    assert torch.equal(restored.training_mean(base, cond, DATES), comp.training_mean(base, cond, DATES))
    assert torch.equal(restored.inference_mean(base, cond), comp.inference_mean(base, cond))
    assert checkpoint["kind"] == CROSSFIT_REMAINDER_VERSION
    corruptions = [
        lambda c: c["normalizer_state"]["mean"].add_(1.),
        lambda c: c["mean_checkpoints"][0]["state_dict"]["net.2.bias"].add_(1.),
        lambda c: c["contract"]["preregistration"]["heldout_years"][0].append(1970),
        lambda c: c["contract"].update(mean_fingerprints=["a", "b"]),
    ]
    for corrupt in corruptions:
        bad = copy.deepcopy(checkpoint)
        corrupt(bad)
        with pytest.raises(ValueError):
            CrossFitMeanComposition.from_checkpoint(bad, expected_provenance=PROVENANCE)
    other_plan = copy.deepcopy(registration_kwargs())
    other_plan["registration_id"] = "different-preregistration"
    with pytest.raises(ValueError, match="preregistration mismatch"):
        CrossFitMeanComposition.from_checkpoint(checkpoint, expected_provenance=PROVENANCE,
                                              expected_preregistration=make_crossfit_registration(**other_plan))
    with pytest.raises(ValueError, match="provenance"):
        CrossFitMeanComposition.from_checkpoint(checkpoint, expected_provenance={**PROVENANCE, "conditioning_fingerprint": "other"})
    with pytest.raises(ValueError, match="Single-mean"):
        CrossFitMeanComposition.from_checkpoint({"kind": "frozen_mean_signed_physical_remainder_v1"}, expected_provenance=PROVENANCE)
    with pytest.raises(ValueError, match="Direct-residual"):
        FrozenMeanComposition.from_checkpoint(checkpoint, expected_provenance=PROVENANCE)


def test_changed_frozen_mean_or_training_mode_is_rejected():
    comp, base, cond, _, _, _, _ = fitted()
    comp.means[0].train()
    with pytest.raises(RuntimeError, match="evaluation mode"):
        comp.inference_mean(base, cond)
    comp.train()
    with torch.no_grad():
        comp.means[1].net[-1].bias.add_(.1)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        comp.checkpoint()
