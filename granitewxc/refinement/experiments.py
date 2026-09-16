"""Fresh, versioned experimental controls around the four native processes.

Production configurations and checkpoint loaders are intentionally unchanged.
These controls need their own recipe and target/provenance contract; they cannot
be loaded as old direct-residual weights or silently bypass production gates.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .base import build_refiner, masked_loss, residual_reconstruction_terms
from .config import RefinementConfig, ResidualNormalizationConfig, config_fingerprint, resolve_refinement_config
from .mean_correction import checked_provenance, tensor_fingerprint, training_only
from .normalization import ResidualNormalizer

EXPERIMENT_VERSION = "native_process_individual_auxiliary_v1"
SUPPORT_VERSION = "joint_censored_sqrt_total_precipitation_v1"
_TERMS = ("process_loss", "reconstruction_loss", "gate_calibration_loss", "multiscale_loss", "gradient_loss", "mean_bias_loss", "mean_path_loss")


@dataclass(frozen=True)
class ExperimentRecipe:
    name: str
    variables: tuple[str, ...]
    variable_weights: tuple[float, ...]
    gate_policy: str = "fixed_unit_fresh"
    process_loss: float = 1.0
    reconstruction_loss: float = 0.0
    gate_calibration_loss: float = 0.0
    multiscale_loss: float = 0.0
    gradient_loss: float = 0.0
    mean_bias_loss: float = 0.0
    mean_path_loss: float = 0.0
    version: str = EXPERIMENT_VERSION

    def __post_init__(self):
        if self.version != EXPERIMENT_VERSION or self.gate_policy not in {"fixed_unit_fresh", "production_trainable"}:
            raise ValueError("Unknown experimental semantic version or gate policy.")
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("Recipe variables must be unique and ordered.")
        w = torch.tensor(self.variable_weights)
        if w.numel() != len(self.variables) or not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
            raise ValueError("Recipe variable weights must match variables and be positive finite constants.")
        coefficients = torch.tensor([getattr(self, key) for key in _TERMS])
        if not bool(torch.isfinite(coefficients).all()) or bool((coefficients < 0).any()) or self.process_loss != 1.0:
            raise ValueError("Native process coefficient remains exactly one; auxiliary weights must be finite and nonnegative.")
        if self.gate_calibration_loss and self.gate_policy != "production_trainable":
            raise ValueError("Gate-calibration requires the explicit trainable production-gate experiment.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]):
        values = dict(mapping)
        values["variables"] = tuple(values["variables"])
        values["variable_weights"] = tuple(values["variable_weights"])
        return cls(**values)


def experiment_recipes(config: RefinementConfig, variables: Sequence[str]) -> dict[str, ExperimentRecipe]:
    """Matched individual changes, never an uncontrolled Cartesian product.

    Native fixed-unit and native production-gate Transformers are separate
    controls. Gate-calibration contrasts against the latter. The full existing
    recipe uses its original production gate and configured coefficients.
    Fixed precipitation weight 2 is registered a priori, not evaluation-fitted.
    """
    names = tuple(variables)
    native = ExperimentRecipe("native", names, (1.,) * len(names))
    result = {"native": native, "native_production_gate": replace(native, name="native_production_gate", gate_policy="production_trainable")}
    for key, weight in (("reconstruction_loss", config.reconstruction_loss_weight), ("multiscale_loss", config.multiscale_loss_weight), ("gradient_loss", config.gradient_loss_weight), ("mean_bias_loss", config.mean_bias_loss_weight)):
        result[key] = replace(native, name=key, **{key: weight})
    if config.uses_transformer:
        result["gate_calibration_loss"] = replace(result["native_production_gate"], name="gate_calibration_loss", gate_calibration_loss=config.reconstruction_loss_weight)
    if config.is_flow_matching:
        result["zero_source_025"] = replace(native, name="zero_source_025", mean_path_loss=.25)
    full = replace(native, name="existing_full_recipe", gate_policy="production_trainable",
                   reconstruction_loss=config.reconstruction_loss_weight,
                   gate_calibration_loss=config.reconstruction_loss_weight if config.uses_transformer else 0.,
                   multiscale_loss=config.multiscale_loss_weight,
                   gradient_loss=config.gradient_loss_weight,
                   mean_bias_loss=config.mean_bias_loss_weight,
                   mean_path_loss=config.flow_matching.mean_path_loss_weight if config.is_flow_matching else 0.)
    result[full.name] = full
    for variable in names:
        result[variable + "_only"] = replace(native, name=variable + "_only", variables=(variable,), variable_weights=(1.,))
    if "pr" in names and len(names) > 1:
        result["fixed_pr_weight_2"] = replace(native, name="fixed_pr_weight_2", variable_weights=tuple(2. if v == "pr" else 1. for v in names))
    return result


def select_channels(values: torch.Tensor, source_variables: Sequence[str], selected_variables: Sequence[str]) -> torch.Tensor:
    if values.shape[-3] != len(source_variables) or len(set(source_variables)) != len(source_variables):
        raise ValueError("Source variable metadata does not match channel dimension -3.")
    indices = [tuple(source_variables).index(v) for v in selected_variables]
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("Selected variables must be unique and nonempty.")
    return values.index_select(-3, torch.tensor(indices, device=values.device))


class ExperimentalRefiner(nn.Module):
    """Keep each algorithm's target/sampler, expose separate auxiliary weights."""
    def __init__(self, config: RefinementConfig, recipe: ExperimentRecipe, *, cond_channels: int):
        super().__init__()
        if not config.is_active or config.loss != "mse":
            raise ValueError("Experimental native controls require an active refiner and native process MSE.")
        if recipe.mean_path_loss and not config.is_flow_matching:
            raise ValueError("Zero-source flow auxiliary cannot be attached to diffusion.")
        self.config, self.recipe, self.cond_channels = config, recipe, int(cond_channels)
        # Core evaluation supplies exact native process/path and clean estimate.
        # Recompute only the reductions so each auxiliary has its own coefficient.
        internal = replace(config, checkpoint=None, reconstruction_loss_weight=0., multiscale_loss_weight=0., gradient_loss_weight=0., mean_bias_loss_weight=0.,
                           flow_matching=replace(config.flow_matching, mean_path_loss_weight=0.))
        self.refiner = build_refiner(internal, residual_channels=len(recipe.variables), cond_channels=cond_channels)
        if hasattr(self.refiner, "correction_gate") and recipe.gate_policy == "fixed_unit_fresh":
            with torch.no_grad():
                self.refiner.correction_gate.fill_(1.)
            self.refiner.correction_gate.requires_grad_(False)
        self.residual_channels = len(recipe.variables)

    def apply_correction_gate(self, residual: torch.Tensor) -> torch.Tensor:
        return self.refiner.apply_correction_gate(residual)

    def sample(self, conditioning: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.refiner.sample(conditioning, **kwargs)

    def deterministic_residual(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.refiner.deterministic_residual(conditioning)

    def training_loss(self, residual_target: torch.Tensor, conditioning: torch.Tensor,
                      valid_mask: torch.Tensor | None = None, generator: torch.Generator | None = None,
                      zero_residual: torch.Tensor | None = None) -> dict[str, Any]:
        if residual_target.shape[1] != self.residual_channels:
            raise ValueError("Select recipe target channels explicitly before experimental training.")
        if valid_mask is not None and (not bool(torch.isfinite(valid_mask).all()) or bool(((valid_mask != 0) & (valid_mask != 1)).any())):
            raise ValueError("Experimental valid masks must be Boolean/0-1; variable weights belong in the recipe.")
        valid = torch.ones_like(residual_target, dtype=torch.bool) if valid_mask is None else torch.broadcast_to(valid_mask.to(device=residual_target.device, dtype=torch.bool), residual_target.shape)
        valid = valid & torch.isfinite(residual_target)
        residual_target = torch.where(valid, residual_target, torch.zeros_like(residual_target))
        native = self.refiner.training_loss(residual_target, conditioning, valid, generator=generator, zero_residual=zero_residual)
        process_target = native["process_target"] if self.config.is_diffusion else native["target_velocity"]
        clean = native["clean_residual_prediction"]
        zero = torch.zeros_like(clean) if zero_residual is None else zero_residual.to(clean)
        zero_source_prediction = None
        if self.recipe.mean_path_loss > 0:
            t = native["flow_time"]
            state = t.reshape(-1, 1, 1, 1) * residual_target
            zero_source_prediction = self.refiner.predict_process(state, conditioning, t)
        terms: dict[str, list[torch.Tensor]] = {key: [] for key in _TERMS}
        for index in range(self.residual_channels):
            sl = slice(index, index+1)
            gate = None
            if hasattr(self.refiner, "correction_gate"):
                gate = lambda value, index=index: (value if index in self.config.unit_correction_gate_channels else value * self.refiner.correction_gate[:, index:index+1].to(value))
            component = residual_reconstruction_terms(clean[:, sl], residual_target[:, sl], valid[:, sl], zero[:, sl], gate, excluded_channels=(0,) if index in self.config.native_only_channels else ())
            terms["process_loss"].append(masked_loss(native["prediction"][:, sl], process_target[:, sl], valid[:, sl], "mse"))
            for key in _TERMS[1:-1]:
                terms[key].append(component[key])
            terms["mean_path_loss"].append(clean[:, sl].sum() * 0 if zero_source_prediction is None or index in self.config.native_only_channels else masked_loss(zero_source_prediction[:, sl], residual_target[:, sl], valid[:, sl], "huber"))
        per_variable = {key: torch.stack(values) for key, values in terms.items()}
        per_variable_total = sum(getattr(self.recipe, key) * value for key, value in per_variable.items())
        weights = torch.tensor(self.recipe.variable_weights, device=clean.device, dtype=torch.float32) * valid.any(dim=(0, 2, 3))
        denominator = weights.sum().clamp(min=torch.finfo(weights.dtype).tiny)
        result = dict(native)
        result.update({key: (value * weights).sum() / denominator for key, value in per_variable.items()})
        result.update(loss=(per_variable_total * weights).sum() / denominator,
                      per_variable_terms=per_variable, per_variable_total=per_variable_total,
                      variable_reduction_weights=weights / denominator, process_target=process_target,
                      zero_source_prediction=zero_source_prediction, residual_target=residual_target, zero_anchor=zero, valid_mask=valid)
        return result

    def semantic_contract(self, target_contract: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
        if not target_contract.get("version") or not target_contract.get("normalizer_fingerprint"):
            raise ValueError("Experimental checkpoints require a versioned target and fitted normalizer fingerprint.")
        return {"version": EXPERIMENT_VERSION, "config": self.config.to_dict(), "recipe": self.recipe.to_dict(),
                "cond_channels": self.cond_channels, "target_contract": copy.deepcopy(dict(target_contract)),
                "provenance": checked_provenance(provenance)}

    def checkpoint(self, *, target_contract: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
        contract = self.semantic_contract(target_contract, provenance)
        return {"kind": EXPERIMENT_VERSION, "contract": contract,
                "fingerprint": config_fingerprint({"contract": contract, "state": tensor_fingerprint(self)}),
                "state_dict": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, expected_target_contract: Mapping[str, Any],
                        expected_provenance: Mapping[str, Any]) -> "ExperimentalRefiner":
        if checkpoint.get("kind") != EXPERIMENT_VERSION:
            raise ValueError("Production/refinement checkpoints cannot be reinterpreted as fresh experimental weights.")
        contract = checkpoint["contract"]
        if contract["target_contract"] != dict(expected_target_contract) or contract["provenance"] != checked_provenance(expected_provenance):
            raise ValueError("Experiment target/statistics/mean/Phase-1/conditioning provenance mismatch.")
        config = resolve_refinement_config({"refinement": contract["config"]})
        model = cls(config, ExperimentRecipe.from_mapping(contract["recipe"]), cond_channels=contract["cond_channels"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        expected = config_fingerprint({"contract": model.semantic_contract(expected_target_contract, expected_provenance), "state": tensor_fingerprint(model)})
        if expected != checkpoint.get("fingerprint"):
            raise ValueError("Experimental checkpoint semantic/state fingerprint mismatch.")
        return model


class TotalPrecipitationSupport(nn.Module):
    """Different formulation: joint censored square-root TOTAL rainfall latent.

    For declared precipitation channels encode a=sqrt(y / amount_unit), then
    training-standardize a. Decode y=amount_unit*max(a,0)^2. Zeros remain exact;
    positive trace rain is not censored during fitting/evaluation. The learned
    joint latent diffusion/flow supplies spatially dependent occurrence through
    a<=0, with no independent pixelwise Bernoulli draw. Its zero-atom probability
    is model-dependent and must be validated; this is not an occurrence likelihood
    or a claim to solve precipitation. Other variables retain signed physical
    residuals. Fresh model/statistics and fixed-unit gate are mandatory.
    """
    def __init__(self, variables: Sequence[str], precipitation_variables: Sequence[str] = ("pr",), *, amount_unit: float = 1.):
        super().__init__()
        self.variables, self.precipitation_variables = tuple(variables), tuple(precipitation_variables)
        if not self.variables or len(set(self.variables)) != len(self.variables) or not set(self.precipitation_variables) <= set(self.variables):
            raise ValueError("Support variables must be ordered, unique and include declared precipitation.")
        if not torch.isfinite(torch.tensor(amount_unit)) or amount_unit <= 0:
            raise ValueError("amount_unit must be positive and finite in the data precipitation units.")
        self.amount_unit = float(amount_unit)
        self.normalizer = ResidualNormalizer(len(self.variables), ResidualNormalizationConfig())
        self.provenance: dict[str, Any] | None = None

    def encode_physical(self, truth: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if truth.shape != baseline.shape or truth.shape[1] != len(self.variables):
            raise ValueError("Support target/baseline channels and shapes must match.")
        valid = torch.isfinite(truth) & torch.isfinite(baseline) if valid is None else valid.bool() & torch.isfinite(truth) & torch.isfinite(baseline)
        parts = []
        for i, variable in enumerate(self.variables):
            y = torch.where(valid[:, i:i+1], truth[:, i:i+1].float(), 0.)
            if variable in self.precipitation_variables:
                if bool(((y < 0) & valid[:, i:i+1]).any()):
                    raise ValueError("Negative observed total precipitation cannot enter square-root total modeling.")
                part = torch.sqrt(y / self.amount_unit)
            else:
                part = y - torch.where(valid[:, i:i+1], baseline[:, i:i+1].float(), 0.)
            parts.append(part)
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def fit(self, truth: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor | None = None, *, split: str, provenance: Mapping[str, Any]) -> None:
        training_only(split)
        if self.normalizer.is_fitted:
            raise RuntimeError("Support statistics already fitted; do not reuse residual statistics.")
        self.provenance = checked_provenance(provenance)
        mask = torch.isfinite(truth) & torch.isfinite(baseline)
        if valid is not None:
            mask &= valid.bool()
        self.normalizer.update(self.encode_physical(truth, baseline, mask), mask)
        self.normalizer.finalize()

    def target(self, truth: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        mask = torch.isfinite(truth) & torch.isfinite(baseline)
        if valid is not None:
            mask &= valid.bool()
        return self.normalizer.normalize(self.encode_physical(truth, baseline, mask), mask)

    def reconstruct(self, normalized_members: torch.Tensor, baseline: torch.Tensor, *, gate_policy: str) -> dict[str, torch.Tensor]:
        if gate_policy != "fixed_unit_fresh":
            raise ValueError("A physical-residual gate is incompatible with total-field latent reconstruction.")
        if normalized_members.ndim != 5:
            raise ValueError("Support members must be [B,M,C,H,W].")
        b, m, c, h, w = normalized_members.shape
        if baseline.shape != (b, c, h, w):
            raise ValueError("Support member/Phase-1 alignment mismatch.")
        latent = self.normalizer.denormalize(normalized_members.reshape(b*m, c, h, w)).reshape(b, m, c, h, w)
        parts, censored = [], torch.zeros_like(latent, dtype=torch.bool)
        for i, variable in enumerate(self.variables):
            amount = latent[:, :, i:i+1]
            if variable in self.precipitation_variables:
                censored[:, :, i:i+1] = amount <= 0
                part = self.amount_unit * amount.clamp(min=0).square()
            else:
                part = baseline[:, None, i:i+1] + amount
            parts.append(part)
        members = torch.cat(parts, dim=2)
        return {"unconstrained_latent": latent, "latent_censor_mask": censored,
                "members_physical": members, "ensemble_mean_physical": members.mean(dim=1),
                "correction_physical": members - baseline[:, None]}

    def semantic_contract(self) -> dict[str, Any]:
        self.normalizer._assert_ready()
        return {"version": SUPPORT_VERSION, "variables": list(self.variables), "precipitation_variables": list(self.precipitation_variables),
                "amount_unit": self.amount_unit, "target": "sqrt_TOTAL_precipitation_and_signed_other_residuals",
                "normalizer_fingerprint": tensor_fingerprint(self.normalizer), "normalizer": self.normalizer.metadata(),
                "provenance": self.provenance, "required_gate_policy": "fixed_unit_fresh", "wet_day_threshold_unchanged": .01}


    def checkpoint(self) -> dict[str, Any]:
        return {"kind": SUPPORT_VERSION, "contract": self.semantic_contract(),
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, expected_provenance: Mapping[str, Any]) -> "TotalPrecipitationSupport":
        if checkpoint.get("kind") != SUPPORT_VERSION:
            raise ValueError("Signed-residual transforms/statistics cannot be used for total support modeling.")
        contract = checkpoint["contract"]
        if contract.get("provenance") != checked_provenance(expected_provenance):
            raise ValueError("Support fitting provenance mismatch.")
        model = cls(contract["variables"], contract["precipitation_variables"], amount_unit=contract["amount_unit"])
        model.provenance = checked_provenance(expected_provenance)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if model.semantic_contract() != contract:
            raise ValueError("Support target/statistics semantic fingerprint mismatch.")
        return model


@dataclass(frozen=True)
class RainfallRegimes:
    wet_threshold: float
    light_upper: float
    heavy_lower: float
    fitting_fingerprint: str

    @classmethod
    def fit(cls, precipitation: torch.Tensor, valid: torch.Tensor, *, split: str,
            fitting_fingerprint: str, wet_threshold: float = .01):
        training_only(split)
        if not fitting_fingerprint or not torch.isfinite(torch.tensor(wet_threshold)) or wet_threshold < 0:
            raise ValueError("A fitting fingerprint and nonnegative configured wet threshold are required.")
        wet = precipitation[valid.bool() & torch.isfinite(precipitation) & (precipitation > wet_threshold)].float()
        if wet.numel() < 2:
            raise ValueError("Need at least two fitting wet cells for regime quantiles.")
        q50, q95 = torch.quantile(wet, torch.tensor([.5, .95], device=wet.device)).tolist()
        return cls(float(wet_threshold), float(q50), float(q95), fitting_fingerprint)

    def masks(self, precipitation: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        ok = valid.bool() & torch.isfinite(precipitation)
        return {"dry_or_trace": ok & (precipitation <= self.wet_threshold),
                "light": ok & (precipitation > self.wet_threshold) & (precipitation <= self.light_upper),
                "moderate": ok & (precipitation > self.light_upper) & (precipitation <= self.heavy_lower),
                "heavy": ok & (precipitation > self.heavy_lower)}


def rainfall_loss_partition(result: Mapping[str, Any], model: ExperimentalRefiner,
                            precipitation: torch.Tensor, valid: torch.Tensor,
                            regimes: RainfallRegimes) -> dict[str, dict[str, torch.Tensor]]:
    """Additive attribution of every active loss to observed rainfall regimes.

    Cell terms use the original valid-cell denominator. Each pooled block, valid
    adjacent pair, or spatial-mean error is split in proportion to the observed
    regime membership of its valid constituent cells. This is an attribution
    convention (including cross-regime pairs), not a changed training objective
    or a claim that a neighborhood loss is a rainfall likelihood.
    """
    i = model.recipe.variables.index("pr")
    p = result["clean_residual_prediction"][:, i:i+1].float()
    y = result["residual_target"][:, i:i+1].float()
    ok = valid.bool().unsqueeze(1) & torch.isfinite(precipitation).unsqueeze(1)
    masks = {k: v.unsqueeze(1) for k, v in regimes.masks(precipitation, valid).items()}
    p, y = torch.where(ok, p, 0.), torch.where(ok, y, 0.)
    zero = p.sum()*0.
    partition = {name: {key: zero for key in _TERMS} for name in masks}
    def huber(a, b):
        return torch.nn.functional.huber_loss(a, b, delta=1., reduction="none")
    errors = {"process_loss": (result["prediction"][:, i:i+1].float()-result["process_target"][:, i:i+1].float()).square(),
              "reconstruction_loss": huber(p, y)}
    if hasattr(model.refiner, "correction_gate"):
        anchor = result["zero_anchor"][:, i:i+1]
        effective = anchor + (p-anchor).detach() * model.refiner.correction_gate[:, i:i+1]
        errors["gate_calibration_loss"] = huber(effective, y)
    if result["zero_source_prediction"] is not None:
        errors["mean_path_loss"] = huber(result["zero_source_prediction"][:, i:i+1].float(), y)
    count = ok.sum().clamp(min=1)
    for name, mask in masks.items():
        for key, error in errors.items():
            partition[name][key] = torch.where(mask, error, 0.).sum()/count
    # The same complete 2x2 and 4x4 blocks as the executed auxiliary objective.
    factors = [f for f in (2, 4) if min(p.shape[-2:]) >= f]
    for factor in factors:
        def pool(v):
            return torch.nn.functional.avg_pool2d(v.float(), factor, factor)*factor**2
        counts = pool(ok)
        good = counts > 0
        error = huber(pool(p)/counts.clamp(min=1), pool(y)/counts.clamp(min=1))
        for name, mask in masks.items():
            fraction = pool(mask)/counts.clamp(min=1)
            contribution = (error*fraction*good).sum()/good.sum().clamp(min=1)
            partition[name]["multiscale_loss"] = partition[name]["multiscale_loss"] + contribution/len(factors)
    directions = [d for d in (-1, -2) if p.shape[d] > 1]
    for dim in directions:
        left = [slice(None)]*4
        right = [slice(None)]*4
        left[dim], right[dim] = slice(1, None), slice(None, -1)
        left, right = tuple(left), tuple(right)
        good = ok[left] & ok[right]
        error = huber(p[left]-p[right], y[left]-y[right])
        for name, mask in masks.items():
            fraction = .5*(mask[left].float()+mask[right].float())
            contribution = (error*fraction*good).sum()/good.sum().clamp(min=1)
            partition[name]["gradient_loss"] = partition[name]["gradient_loss"] + contribution/len(directions)
    counts = ok.sum(dim=(-2, -1))
    good = counts > 0
    error = huber(p.sum(dim=(-2, -1))/counts.clamp(min=1), y.sum(dim=(-2, -1))/counts.clamp(min=1))
    for name, mask in masks.items():
        fraction = mask.sum(dim=(-2, -1))/counts.clamp(min=1)
        partition[name]["mean_bias_loss"] = (error*fraction*good).sum()/good.sum().clamp(min=1)
    return partition


def gradient_diagnostics(result: Mapping[str, Any], model: ExperimentalRefiner, *,
                         truth_physical: torch.Tensor | None = None, valid: torch.Tensor | None = None,
                         regimes: RainfallRegimes | None = None,
                         normalizer: ResidualNormalizer | None = None) -> dict[str, Any]:
    """Read-only autograd attribution, retaining the graph for the real update.

    Rain regimes partition cellwise terms with the all-valid-channel denominator.
    Cross-cell terms use the explicit proportional-footprint attribution in
    rainfall_loss_partition; their allocations are not separate objectives.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    shared_ids = {id(p) for p in model.refiner.net.parameters()}
    def gradients(loss):
        if not loss.requires_grad:
            return [torch.zeros_like(p) for p in params]
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if v is None else v for p, v in zip(params, g)]
    def norm(g, shared=False):
        return float(torch.sqrt(sum(v.double().square().sum() for p, v in zip(params, g) if not shared or id(p) in shared_ids)))
    rows, variable_gradients = {}, []
    for i, variable in enumerate(model.recipe.variables):
        total = result["per_variable_total"][i]
        g = gradients(total)
        variable_gradients.append(g)
        rows[variable] = {"total_loss_before_variable_weight": float(total.detach()), "gradient_l2": norm(g),
                          "shared_backbone_gradient_l2": norm(g, True), "terms": {}}
        for key in _TERMS:
            value = result["per_variable_terms"][key][i]
            weighted = getattr(model.recipe, key) * value
            rows[variable]["terms"][key] = {"unweighted_loss": float(value.detach()), "coefficient": getattr(model.recipe, key), "weighted_gradient_l2": norm(gradients(weighted))}
    alignment = {}
    for i in range(len(variable_gradients)):
        for j in range(i+1, len(variable_gradients)):
            a, b = variable_gradients[i], variable_gradients[j]
            dot = sum((x.double()*y.double()).sum() for p, x, y in zip(params, a, b) if id(p) in shared_ids)
            denominator = norm(a, True) * norm(b, True)
            alignment[f"{model.recipe.variables[i]}:{model.recipe.variables[j]}"] = None if denominator == 0 else float(dot) / denominator
    report = {"variables": rows, "shared_gradient_cosine": alignment, "joint_gradient_l2": norm(gradients(result["loss"]))}
    if hasattr(model.refiner, "correction_gate"):
        gate = model.refiner.correction_gate
        report["gate_values"] = gate.detach().cpu().reshape(-1).tolist()
        report["gate_trainable"] = gate.requires_grad
        report["gate_gradient"] = None if not gate.requires_grad else next(v.detach().cpu().reshape(-1).tolist() for p, v in zip(params, gradients(result["loss"])) if p is gate)
    if regimes is not None:
        if truth_physical is None or valid is None or "pr" not in model.recipe.variables:
            raise ValueError("Rainfall attribution requires aligned pr observations and masks.")
        i = model.recipe.variables.index("pr")
        masks = regimes.masks(truth_physical[:, i], valid[:, i])
        error = (result["prediction"][:, i].float() - result["process_target"][:, i].float()).square()
        denominator = sum(int(mask.sum()) for mask in masks.values())
        contributions = {}
        for name, mask in masks.items():
            contribution = torch.where(mask, error, 0.).sum() / max(denominator, 1)
            contributions[name] = {"count": int(mask.sum()), "valid_pr_count_denominator": denominator,
                                   "process_mse_contribution": float(contribution.detach()),
                                   "process_gradient_l2": norm(gradients(contribution))}
        partitions = rainfall_loss_partition(result, model, truth_physical[:, i], valid[:, i], regimes)
        for name, term_values in partitions.items():
            contributions[name]["all_terms"] = {
                key: {"unweighted_contribution": float(value.detach()),
                      "coefficient": getattr(model.recipe, key),
                      "weighted_gradient_l2": norm(gradients(getattr(model.recipe, key)*value))}
                for key, value in term_values.items()}
        report["rainfall_regimes"] = {"thresholds": asdict(regimes), "contributions": contributions,
             "neighborhood_allocation": "proportion_of_valid_constituent_cells_in_regime"}
    return report