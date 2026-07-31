import numpy as np
import pytest

from granitewxc.utils.prism_tiling import (
    TilePlan,
    WeightedTileStitcher,
    blend_window,
    boundary_gradient_ratio,
    extract_halo_context,
    overlap_crossovers,
    overlap_disagreement,
)


@pytest.mark.parametrize("shape", [(71, 83), (64, 64), (17, 101)])
def test_patch_extract_and_weighted_reconstruction(shape):
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    field = np.stack([np.ones(shape), 3.0 * yy - 0.25 * xx])
    plan = TilePlan.build(shape, (32, 35), overlap=(9, 11), halo=(4, 5))
    window = blend_window(plan.core_shape, plan.overlap)
    stitcher = WeightedTileStitcher(2, shape)

    for origin in plan.positions:
        context = extract_halo_context(field, origin, plan.core_shape, plan.halo)
        hy, hx = plan.halo
        core = context[
            ..., hy : hy + plan.core_shape[0], hx : hx + plan.core_shape[1]
        ]
        stitcher.add(core, origin, window)

    reconstructed = stitcher.finalize()
    np.testing.assert_allclose(reconstructed, field, rtol=0.0, atol=1.0e-12)
    assert np.all(stitcher.weight > 0.0)


def test_halo_tiling_matches_full_local_operator():
    torch = pytest.importorskip("torch")
    torch_f = pytest.importorskip("torch.nn.functional")
    generator = torch.Generator().manual_seed(7)
    field = torch.randn((1, 1, 53, 61), generator=generator)
    kernel = torch.tensor(
        [[[[0.1, -0.2, 0.05], [0.3, 0.5, -0.1], [0.0, 0.2, 0.15]]]],
        dtype=field.dtype,
    )
    full = torch_f.conv2d(torch_f.pad(field, (1, 1, 1, 1), mode="reflect"), kernel)

    plan = TilePlan.build((53, 61), (23, 25), overlap=(7, 8), halo=1)
    stitcher = WeightedTileStitcher(1, plan.domain_shape)
    window = blend_window(plan.core_shape, plan.overlap)
    for origin in plan.positions:
        context = extract_halo_context(field[0], origin, plan.core_shape, plan.halo)
        predicted_core = torch_f.conv2d(context.unsqueeze(0), kernel)[0]
        stitcher.add(predicted_core.numpy(), origin, window)

    tiled = stitcher.finalize()
    np.testing.assert_allclose(tiled, full[0].numpy(), rtol=0.0, atol=2.0e-6)


def test_overlap_crossovers_and_disagreement_are_measured_on_shared_pixels():
    origins = [0, 16, 32]
    assert overlap_crossovers(origins, tile_size=24) == [19, 35]

    first = np.zeros((1, 12, 12), dtype=np.float32)
    second = np.ones((1, 12, 12), dtype=np.float32)
    metrics = overlap_disagreement(first, (0, 0), second, (0, 8))
    assert metrics["count"] == 48
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)


def test_boundary_gradient_ratio_targets_requested_band():
    field = np.zeros((30, 30), dtype=np.float64)
    field[:, 15:] = 10.0
    ratio = boundary_gradient_ratio(field, [15], axis=1, half_width=1)
    assert ratio > 5.0


def test_stitcher_rejects_incomplete_coverage():
    stitcher = WeightedTileStitcher(1, (8, 8))
    stitcher.add(np.ones((1, 4, 4)), (0, 0), np.ones((4, 4)))
    with pytest.raises(ValueError, match="uncovered"):
        stitcher.finalize()


def test_windowed_model_tile_plan_requires_global_phase_alignment():
    TilePlan.build((128, 128), (64, 64), overlap=(16, 16), halo=(32, 32)).assert_globally_aligned(16)
    with pytest.raises(ValueError, match="model mask/decoder phase"):
        TilePlan.build((128, 128), (64, 64), overlap=(8, 8), halo=(32, 32)).assert_globally_aligned(16)
