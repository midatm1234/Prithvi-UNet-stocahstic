from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "CORDEX_ML"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from utils.diffusion_diagnostics import (  # noqa: E402
    compute_crps,
    residual_distribution_report,
    transformation_stage_report,
)


def test_empirical_crps_includes_zero_self_pairs() -> None:
    truth = np.zeros((1, 1, 1), dtype=np.float64)
    ensemble = np.array([[[[-1.0]], [[1.0]]]], dtype=np.float64)

    # E|X-y| - 0.5 E|X-X'| = 1 - 0.5*(1) = 0.5 for
    # the empirical two-member distribution {-1,+1}.
    assert compute_crps(ensemble, truth)["crps_mean"] == pytest.approx(0.5)


def test_transformation_report_derives_physical_residual_by_subtraction() -> None:
    baseline = np.array([[[1.0, 2.0]]])
    truth = np.array([[[1.5, 1.25]]])
    final = np.array([[[1.4, 1.5]]])
    report = transformation_stage_report(
        baseline_physical=baseline,
        truth_physical=truth,
        true_residual_normalized=np.array([[[0.1, -0.2]]]),
        generated_residual_normalized=np.array([[[0.08, -0.1]]]),
        final_physical=final,
    )

    assert report["physical_true_residual"]["mean"] == pytest.approx(-0.125)
    assert report["denormalized_predicted_residual"]["mean"] == pytest.approx(-0.05)


def test_residual_distribution_report_rejects_space_or_shape_mismatch() -> None:
    values = np.zeros((2, 4, 4))
    with pytest.raises(ValueError, match="must share a shape"):
        residual_distribution_report(values, values[:, :-1], values)
