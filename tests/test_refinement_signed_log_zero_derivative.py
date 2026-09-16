"""Signed log is differentiable at zero despite the sign/abs implementation."""
import pytest
import torch
from granitewxc.refinement import ResidualNormalizer
from granitewxc.refinement.config import ResidualNormalizationConfig


@pytest.mark.parametrize('scale',[.1,1.,10.])
def test_signed_log_exact_zero_has_analytic_forward_and_inverse_derivatives(scale):
    normalizer=ResidualNormalizer(2,ResidualNormalizationConfig(signed_log_nonnegative_channels=True,signed_log_scale=scale),nonnegative_mask=torch.tensor([True,False]))
    values=torch.zeros(2,2,7,9,dtype=torch.float64,requires_grad=True)
    forward_grad=torch.autograd.grad(normalizer._forward_transform(values).sum(),values)[0]
    inverse_grad=torch.autograd.grad(normalizer._inverse_transform(values).sum(),values)[0]
    torch.testing.assert_close(forward_grad[:,0],torch.full_like(values[:,0],1/scale))
    torch.testing.assert_close(inverse_grad[:,0],torch.full_like(values[:,0],scale))
    assert torch.equal(forward_grad[:,1],torch.ones_like(values[:,1]))
    assert torch.equal(inverse_grad[:,1],torch.ones_like(values[:,1]))


@pytest.mark.parametrize('scale',[.1,1.,10.])
def test_signed_log_forward_values_and_tmax_remain_bitwise_compatible(scale):
    normalizer=ResidualNormalizer(2,ResidualNormalizationConfig(signed_log_nonnegative_channels=True,signed_log_scale=scale),nonnegative_mask=torch.tensor([True,False]))
    values=torch.linspace(-6.,6.,2*2*7*9).reshape(2,2,7,9)
    values[:,0,0,0]=0.;values[:,0,-1,-1]=-0.
    old_forward=values.sign()*torch.log1p(values.abs()/scale)
    old_inverse=values.sign()*torch.expm1(values.abs())*scale
    assert torch.equal(normalizer._forward_transform(values)[:,0],old_forward[:,0])
    assert torch.equal(normalizer._inverse_transform(values)[:,0],old_inverse[:,0])
    assert torch.equal(normalizer._forward_transform(values)[:,1],values[:,1])
    assert torch.equal(normalizer._inverse_transform(values)[:,1],values[:,1])
