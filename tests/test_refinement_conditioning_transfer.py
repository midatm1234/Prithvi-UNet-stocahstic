import copy

import pytest
import torch

from granitewxc.refinement import build_refiner
from granitewxc.refinement.conditioning_transfer import fold_conditioning_affine
from test_refinement_boundary_preconditioning import HEADS, configuration


@pytest.mark.parametrize('head', HEADS)
def test_affine_transfer_preserves_native_outputs_on_odd_grid(head):
    torch.manual_seed(902)
    model = build_refiner(configuration(head), residual_channels=2, cond_channels=3).eval()
    for parameter in model.net.parameters():
        torch.nn.init.normal_(parameter, std=0.08)
    transferred = copy.deepcopy(model)
    mean = torch.tensor([1.5, -2., 10.])
    scale = torch.tensor([0.05, 3., 5.])
    fold_conditioning_affine(transferred, mean, scale)
    normalized = torch.randn(2, 3, 7, 9)
    raw = normalized * scale[None, :, None, None] + mean[None, :, None, None]
    state = torch.randn(2, 2, 7, 9)
    time = torch.tensor([0.1, 0.8]) if head.startswith('flow') else torch.tensor([1, 8])
    with torch.no_grad():
        expected = model.predict_process(state, raw, time)
        actual = transferred.predict_process(state, normalized, time)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_affine_transfer_rejects_zero_padding():
    model = build_refiner(configuration('flow_matching_unet'), residual_channels=2, cond_channels=3)
    model.net.down_blocks[0].conv1.padding_mode = 'zeros'
    with pytest.raises(ValueError, match='replication padding'):
        fold_conditioning_affine(model, torch.zeros(3), torch.ones(3))
