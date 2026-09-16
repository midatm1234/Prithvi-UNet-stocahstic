"""Precision configuration must affect convolutions as well as GEMMs."""
import pytest
import torch

from granitewxc.refinement.config import resolve_performance_config
from granitewxc.refinement.precision import configure_refinement_precision


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_tf32_policy_overrides_both_backend_defaults(enabled):
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = not enabled
        torch.backends.cudnn.allow_tf32 = not enabled
        policy = resolve_performance_config({'performance': {'precision': {'allow_tf32': enabled}}})
        actual = configure_refinement_precision(policy)
        assert actual == {'matmul_allow_tf32': enabled, 'cudnn_allow_tf32': enabled}
        assert torch.backends.cuda.matmul.allow_tf32 == enabled
        assert torch.backends.cudnn.allow_tf32 == enabled
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
