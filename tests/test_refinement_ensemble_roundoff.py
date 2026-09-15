"""Physical ensemble statistics must not undo precipitation constraints."""

import warnings

import pytest
import torch

from granitewxc.refinement import build_two_phase_model
from refinement_fixtures import TinyPhase1, make_batch


@pytest.mark.parametrize("strategy", ["memberwise", "mean_preserving"])
@pytest.mark.parametrize("ensemble_size", [1, 3, 10])
def test_ensemble_statistics_preserve_dry_cells_and_zero_corrections(
    monkeypatch, strategy, ensemble_size
):
    batch = make_batch(batch_size=1, height=8, width=8)
    config = {
        "refinement": {
            "type": "flow_matching_unet",
            "nonnegative_ensemble_strategy": strategy,
            "residual_normalization": {"method": "identity"},
            "unet": {"hidden_channels": 8, "num_levels": 2},
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = build_two_phase_model(TinyPhase1(), config)
    model.initialize_from_batch(batch)
    model.eval()
    base = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(4)) * 32
    normalized = model.target_space.encode(base)
    monkeypatch.setattr(model, "run_phase1", lambda batch: (base, normalized, {}))

    # Dry precipitation, negative temperatures, and an unchanged channel.
    corrections = torch.zeros_like(base)
    corrections[:, 0] = -2 * base[:, 0]
    corrections[:, 1] = -base[:, 1] - 2
    monkeypatch.setattr(
        model.refiner,
        "sample",
        lambda cond: corrections.expand(cond.shape[0], -1, -1, -1).clone(),
    )
    mask = torch.ones_like(base, dtype=torch.bool)
    mask[..., 0, 0] = False
    batch["__prediction_mask"] = mask
    output = model.predict(batch, ensemble_size=ensemble_size, seed=9)

    for values in (output.refined, output.ensemble_mean):
        assert torch.isnan(values[~mask]).all()
        precipitation = values[:, 0][mask[:, 0]]
        assert torch.equal(precipitation, torch.zeros_like(precipitation))
        assert (values[:, 1][mask[:, 1]] < 0).all()
        assert torch.equal(values[:, 2][mask[:, 2]], base[:, 2][mask[:, 2]])
    torch.testing.assert_close(
        output.ensemble_mean, output.members.mean(dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        output.residual_physical, output.refined - base, equal_nan=True
    )
    if ensemble_size > 1:
        assert torch.isnan(output.ensemble_spread[~mask]).all()
        assert torch.count_nonzero(output.ensemble_spread[mask]) == 0
