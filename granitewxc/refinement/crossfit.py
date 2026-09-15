"""Frozen, preregistered year-fold means for fresh stochastic remainders.

Fitting targets use the one mean whose fitting AND checkpoint-selection years
exclude that sample's year. Inference uses the arithmetic average of both means.
The latter is a different finite-data estimator; this API does not claim that
cross-fitting alone gives calibrated remainders or preserves ensemble spread.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .config import ResidualNormalizationConfig, config_fingerprint
from .mean_correction import SignedResidualMeanCorrector, checked_provenance, tensor_fingerprint, training_only
from .normalization import ResidualNormalizer

CROSSFIT_REGISTRATION_VERSION = "two_mean_year_folds_preregistration_v1"
CROSSFIT_REMAINDER_VERSION = "out_of_fold_mean_signed_physical_remainder_v1"


def _years(values: Sequence[int], label: str, *, allow_empty: bool = False) -> list[int]:
    if isinstance(values, (str, bytes)) or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(f"{label} must contain integer calendar years.")
    result = sorted(values)
    if (not result and not allow_empty) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique calendar years.")
    return result


def make_crossfit_registration(*, registration_id: str, fitting_years: Sequence[int],
                              heldout_years: Sequence[Sequence[int]],
                              mean_training_years: Sequence[Sequence[int]],
                              mean_selection_years: Sequence[Sequence[int]],
                              mean_provenance: Sequence[Mapping[str, Any]],
                              era_years: Mapping[str, Sequence[int]],
                              selection_rule: str) -> dict[str, Any]:
    """Build a hash-bound plan before fitting either mean or viewing its outputs.

    Each held-out fold partitions the entire fitting selection. A mean's scaling,
    optimization and checkpoint selection must be confined to the opposite fold;
    the two latter year sets partition it. Explicit eras avoid hard-coded climate
    dates and ensure each fitting, selection and held-out set covers every era.
    The caller must persist this plan before training: a hash cannot prove timing
    or that supplied provenance truthfully describes an external training run.
    """
    if not registration_id or not selection_rule:
        raise ValueError("A separate preregistration identifier and checkpoint-selection rule are required.")
    if any(len(items) != 2 for items in (heldout_years, mean_training_years, mean_selection_years, mean_provenance)):
        raise ValueError("Cross-fitting requires exactly two independently fitted/selected means.")
    fitting = _years(fitting_years, "fitting_years")
    heldout = [_years(v, f"heldout_years[{i}]") for i, v in enumerate(heldout_years)]
    train = [_years(v, f"mean_training_years[{i}]") for i, v in enumerate(mean_training_years)]
    select = [_years(v, f"mean_selection_years[{i}]") for i, v in enumerate(mean_selection_years)]
    eras = {str(k): _years(v, f"era_years[{k}]") for k, v in era_years.items()}
    if len(eras) < 2 or any(not name for name in eras):
        raise ValueError("Declare at least two named eras (historical/future for CORDEX).")
    flat_eras = [year for values in eras.values() for year in values]
    if sorted(flat_eras) != fitting:
        raise ValueError("Era year sets must partition the fitting years without duplication.")
    if sorted(heldout[0] + heldout[1]) != fitting:
        raise ValueError("Held-out year folds must partition the fitting years without duplication.")
    provenance = [checked_provenance(p) for p in mean_provenance]
    for key in ("phase1_fingerprint", "conditioning_fingerprint"):
        if provenance[0][key] != provenance[1][key]:
            raise ValueError(f"Both means must share the same {key}.")
    if provenance[0]["training_selection_fingerprint"] == provenance[1]["training_selection_fingerprint"]:
        raise ValueError("The independently fitted means need distinct fitting-selection fingerprints.")
    for i in range(2):
        if set(train[i]) & set(select[i]):
            raise ValueError("Mean fitting and checkpoint-selection years must be disjoint.")
        if sorted(train[i] + select[i]) != sorted(set(fitting) - set(heldout[i])):
            raise ValueError("Each mean's fitting/selection years must partition the complement of its held-out years.")
        for label, years in (("held-out", heldout[i]), ("training", train[i]), ("selection", select[i])):
            if any(not (set(years) & set(era)) for era in eras.values()):
                raise ValueError(f"Every mean {label} selection must span each declared era.")
    result = {"version": CROSSFIT_REGISTRATION_VERSION, "registration_id": str(registration_id),
              "fitting_years": fitting, "heldout_years": heldout, "mean_training_years": train,
              "mean_selection_years": select, "mean_provenance": provenance,
              "era_years": eras, "selection_rule": str(selection_rule)}
    result["registration_fingerprint"] = config_fingerprint(result)
    return result


def _checked_registration(plan: Mapping[str, Any]) -> dict[str, Any]:
    values = copy.deepcopy(dict(plan))
    if values.pop("version", None) != CROSSFIT_REGISTRATION_VERSION:
        raise ValueError("Incompatible or missing cross-fit preregistration version.")
    expected = values.pop("registration_fingerprint", None)
    try:
        checked = make_crossfit_registration(**values)
    except TypeError as exc:
        raise ValueError("Cross-fit preregistration fields are incompatible.") from exc
    if checked["registration_fingerprint"] != expected:
        raise ValueError("Cross-fit preregistration fingerprint mismatch.")
    return checked


def _date_year(value: Any) -> int:
    # Calendar labels, including cftime non-Gregorian dates, are used only for
    # their year. Integer dataset positions and arbitrary IDs are not dates.
    match = re.match(r"^(\d{4})-\d{2}-\d{2}(?:$|[T ])", str(value))
    if match is None:
        raise ValueError("Cross-fit sample IDs must be ISO calendar date labels, not dataset positions.")
    return int(match.group(1))


class CrossFitMeanComposition(nn.Module):
    """Two frozen mean checkpoints and a new, fitting-only OOF normalizer.

    This class cannot reinterpret a single-mean or direct-residual checkpoint.
    Native stochastic weights must be freshly trained against its semantic
    contract. Physical nonnegativity is an external, separately audited step.
    """

    def __init__(self, mean_checkpoints: Sequence[Mapping[str, Any]], *,
                 preregistration: Mapping[str, Any]) -> None:
        super().__init__()
        self.preregistration = _checked_registration(preregistration)
        if len(mean_checkpoints) != 2:
            raise ValueError("Exactly two selected mean checkpoints are required.")
        means = [SignedResidualMeanCorrector.from_checkpoint(c, expected_provenance=p)
                 for c, p in zip(mean_checkpoints, self.preregistration["mean_provenance"])]
        if means[0].variables != means[1].variables or means[0].cond_channels != means[1].cond_channels:
            raise ValueError("Mean checkpoint variable ordering/conditioning channels must match.")
        self.means = nn.ModuleList([mean.requires_grad_(False).eval() for mean in means])
        self.variables = means[0].variables
        self.mean_fingerprints = [mean.fingerprint() for mean in means]
        self.mean_fingerprint = config_fingerprint({"preregistration": self.preregistration,
                                                    "mean_fingerprints": self.mean_fingerprints})
        self.normalizer = ResidualNormalizer(len(self.variables), ResidualNormalizationConfig())
        self.remainder_provenance: dict[str, Any] | None = None
        self.statistics_sample_fingerprint: str | None = None
        self.statistics_sample_count = 0
        self._year_to_mean = {year: i for i, years in enumerate(self.preregistration["heldout_years"]) for year in years}

    def train(self, mode: bool = True):
        super().train(mode)
        for mean in self.means:
            mean.eval()
        return self

    def _check_frozen(self, *, fingerprint: bool = False) -> None:
        if any(mean.training or any(p.requires_grad for p in mean.parameters()) for mean in self.means):
            raise RuntimeError("Cross-fit means must remain frozen and in evaluation mode.")
        if fingerprint and [mean.fingerprint() for mean in self.means] != self.mean_fingerprints:
            raise RuntimeError("Frozen cross-fit mean fingerprint changed; cached targets/statistics are invalid.")
        if fingerprint:
            _checked_registration(self.preregistration)

    def _check_inputs(self, baseline: torch.Tensor, conditioning: torch.Tensor) -> None:
        self._check_frozen()
        if baseline.ndim != 4 or baseline.shape[1] != len(self.variables):
            raise ValueError("Baseline must be BCHW with the registered variable ordering.")
        if conditioning.ndim != 4 or conditioning.shape[0] != baseline.shape[0] or conditioning.shape[-2:] != baseline.shape[-2:]:
            raise ValueError("Baseline and conditioning sample/domain alignment mismatch.")

    def _fold_indices(self, dateids: Sequence[Any], count: int) -> list[int]:
        if isinstance(dateids, (str, bytes)) or len(dateids) != count:
            raise ValueError("Provide one calendar date ID per fitting sample.")
        years = [_date_year(value) for value in dateids]
        if any(year not in self._year_to_mean for year in years):
            raise ValueError("A sample year is outside the preregistered fitting years; validation/test cannot enter OOF targets.")
        return [self._year_to_mean[year] for year in years]

    @torch.no_grad()
    def training_mean(self, baseline: torch.Tensor, conditioning: torch.Tensor,
                      dateids: Sequence[Any]) -> torch.Tensor:
        """Physical Phase 1 plus the mean that never used this sample's year."""
        self._check_inputs(baseline, conditioning)
        folds = self._fold_indices(dateids, baseline.shape[0])
        result = torch.empty_like(baseline, dtype=torch.float32)
        for i, model in enumerate(self.means):
            selected = [j for j, fold in enumerate(folds) if fold == i]
            if selected:
                result[selected] = baseline[selected].detach().float() + model(conditioning[selected])
        return result

    @torch.no_grad()
    def inference_mean(self, baseline: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Physical Phase 1 plus the arithmetic average of the two frozen means."""
        self._check_inputs(baseline, conditioning)
        corrections = [model(conditioning) for model in self.means]
        return baseline.detach().float() + .5 * (corrections[0] + corrections[1])

    @torch.no_grad()
    def fit_remainder_statistics(self, batches: Iterable[Mapping[str, Any]], *,
                                 split: str, provenance: Mapping[str, Any]) -> None:
        training_only(split)
        if self.normalizer.is_fitted:
            raise RuntimeError("Cross-fit remainder statistics already fitted; construct a fresh composition.")
        self._check_frozen(fingerprint=True)
        provenance = checked_provenance(provenance)
        for key in ("phase1_fingerprint", "conditioning_fingerprint"):
            if provenance[key] != self.preregistration["mean_provenance"][0][key]:
                raise ValueError(f"Remainder and both means must share {key}.")
        # Stage in a fresh instance so invalid/incomplete streams never leave
        # partially fitted statistics or silently accumulate on a retry.
        staged = ResidualNormalizer(len(self.variables), ResidualNormalizationConfig()).to(self.normalizer.mean.device)
        seen: set[str] = set()
        for batch in batches:
            dateids = batch["sample_ids"]
            mean = self.training_mean(batch["baseline"], batch["conditioning"], dateids)
            names = [str(v) for v in dateids]
            if len(set(names)) != len(names) or seen.intersection(names):
                raise ValueError("Repeated fitting sample IDs would duplicate normalization contributions.")
            seen.update(names)
            truth = batch["truth"].detach().float()
            mask = self._valid(truth, mean, batch.get("valid"))
            staged.update(truth - mean, mask)
        if {_date_year(v) for v in seen} != set(self.preregistration["fitting_years"]):
            raise ValueError("Remainder statistics must cover every preregistered fitting year.")
        staged.finalize()
        self.normalizer.load_state_dict(staged.state_dict(), strict=True)
        self.remainder_provenance = provenance
        self.statistics_sample_fingerprint = config_fingerprint(sorted(seen))
        self.statistics_sample_count = len(seen)

    @staticmethod
    def _valid(truth: torch.Tensor, mean: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if truth.shape != mean.shape:
            raise ValueError("Truth and physical mean must have identical sample/channel/domain shapes.")
        mask = torch.isfinite(truth) & torch.isfinite(mean)
        if valid is not None:
            if valid.shape != truth.shape:
                raise ValueError("Valid mask must match all truth dimensions.")
            mask &= valid.to(device=truth.device, dtype=torch.bool)
        return mask

    @torch.no_grad()
    def prepare(self, baseline: torch.Tensor, conditioning: torch.Tensor, truth: torch.Tensor,
                valid: torch.Tensor | None = None, *, dateids: Sequence[Any]) -> dict[str, torch.Tensor]:
        mean = self.training_mean(baseline, conditioning, dateids)
        return self._prepared(baseline, mean, truth, valid)

    @torch.no_grad()
    def prepare_evaluation(self, baseline: torch.Tensor, conditioning: torch.Tensor, truth: torch.Tensor,
                           valid: torch.Tensor | None = None, *, split: str) -> dict[str, torch.Tensor]:
        """Evaluation targets around the average inference mean, without fitting.

        Observations are used only after mean prediction to form evaluation
        targets, never as conditioning. Callers still enforce their registered
        validation/test access protocol.
        """
        if split not in {"validation", "test"}:
            raise ValueError("Evaluation target preparation needs split='validation' or 'test'; fitting needs OOF dateids.")
        mean = self.inference_mean(baseline, conditioning)
        return self._prepared(baseline, mean, truth, valid)

    def _prepared(self, baseline: torch.Tensor, mean: torch.Tensor, truth: torch.Tensor,
                  valid: torch.Tensor | None) -> dict[str, torch.Tensor]:
        mask = self._valid(truth, mean, valid)
        remainder = truth.detach().float() - mean
        return {"baseline_physical": baseline.detach(), "mean_physical": mean,
                "remainder_physical": remainder,
                "remainder_normalized": self.normalizer.normalize(remainder, mask),
                "zero_anchor": self.normalizer.normalize(torch.zeros_like(remainder)), "valid": mask}

    def reconstruct(self, baseline: torch.Tensor, conditioning: torch.Tensor,
                    normalized_members: torch.Tensor, refiner: nn.Module, *,
                    dateids: Sequence[Any] | None = None) -> dict[str, torch.Tensor]:
        """Restore each physical member; dateids explicitly selects OOF diagnostics.

        The default is inference composition. Supplying fitting dateids selects
        the OOF mean used to form targets, useful for exact training round trips.
        No ensemble centering, residual clipping or precipitation postprocessing
        occurs here. A refiner's explicit gate is applied to physical remainders.
        """
        if normalized_members.ndim != 5:
            raise ValueError("Members must be [B,M,C,H,W].")
        b, m, c, h, w = normalized_members.shape
        if m < 1 or baseline.shape != (b, c, h, w):
            raise ValueError("Member and Phase-1 sample/channel/domain alignment mismatch.")
        mean = self.inference_mean(baseline, conditioning) if dateids is None else self.training_mean(baseline, conditioning, dateids)
        remainder = self.normalizer.denormalize(normalized_members.reshape(b*m, c, h, w)).reshape(b, m, c, h, w)
        gated = refiner.apply_correction_gate(remainder)
        raw = mean[:, None] + gated
        return {"baseline_physical": baseline.detach(), "mean_physical": mean,
                "ungated_remainder_physical": remainder, "gated_remainder_physical": gated,
                "raw_members_physical": raw, "raw_ensemble_mean_physical": raw.mean(dim=1)}

    def semantic_contract(self) -> dict[str, Any]:
        self.normalizer._assert_ready()
        self._check_frozen(fingerprint=True)
        if self.remainder_provenance is None or self.statistics_sample_count <= 0 or not self.statistics_sample_fingerprint:
            raise RuntimeError("Cross-fit remainder fitting provenance/sample identity is missing.")
        return {"version": CROSSFIT_REMAINDER_VERSION, "variables": list(self.variables),
                "preregistration": copy.deepcopy(self.preregistration),
                "mean_fingerprints": list(self.mean_fingerprints),
                "mean_fingerprint": self.mean_fingerprint,
                "training_target": "truth_physical - (exact_phase1 + heldout_year_mean_correction)",
                "inference_mean": "exact_phase1 + arithmetic_average_of_two_frozen_mean_corrections",
                "finite_data_caveat": "individual_OOF_mean_and_average_inference_mean_have_different_estimation_errors",
                "normalizer_fingerprint": tensor_fingerprint(self.normalizer), "normalizer": self.normalizer.metadata(),
                "statistics_sample_fingerprint": self.statistics_sample_fingerprint,
                "statistics_sample_count": self.statistics_sample_count,
                "provenance": copy.deepcopy(self.remainder_provenance), "requires_fresh_stochastic_weights": True}

    def checkpoint(self) -> dict[str, Any]:
        return {"kind": CROSSFIT_REMAINDER_VERSION, "contract": self.semantic_contract(),
                "mean_checkpoints": [mean.checkpoint() for mean in self.means],
                "normalizer_state": {k: v.detach().cpu().clone() for k, v in self.normalizer.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, expected_provenance: Mapping[str, Any],
                        expected_preregistration: Mapping[str, Any] | None = None) -> "CrossFitMeanComposition":
        if checkpoint.get("kind") != CROSSFIT_REMAINDER_VERSION:
            raise ValueError("Single-mean/direct-residual checkpoints cannot be loaded as cross-fit remainders.")
        contract = checkpoint["contract"]
        provenance = checked_provenance(expected_provenance)
        plan = _checked_registration(contract["preregistration"] if expected_preregistration is None else expected_preregistration)
        if contract.get("provenance") != provenance or contract.get("preregistration") != plan:
            raise ValueError("Cross-fit remainder provenance/preregistration mismatch.")
        model = cls(checkpoint["mean_checkpoints"], preregistration=plan)
        model.normalizer.load_state_dict(checkpoint["normalizer_state"], strict=True)
        model.remainder_provenance = provenance
        model.statistics_sample_fingerprint = contract.get("statistics_sample_fingerprint")
        model.statistics_sample_count = contract.get("statistics_sample_count", 0)
        if model.semantic_contract() != contract:
            raise ValueError("Cross-fit mean/statistics semantic fingerprint mismatch.")
        return model
