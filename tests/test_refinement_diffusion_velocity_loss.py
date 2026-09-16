"""Precipitation time weighting, physical parameterization, and Tmax isolation."""
from dataclasses import replace

import pytest
import torch

from granitewxc.refinement import build_refiner,resolve_refinement_config
from granitewxc.refinement.base import masked_loss
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.config import ConfigValidationError,DiffusionConfig
from granitewxc.refinement.process_losses import clean_prediction_process_loss
from granitewxc.refinement.schedules import DiffusionSchedule


def test_clean_error_weighting_equals_native_velocity_error():
    schedule=DiffusionSchedule()
    t=torch.tensor([0,50,500,999])
    target=torch.randn(4,2,5,7)
    noise=torch.randn_like(target)
    noisy=schedule.add_noise(target,noise,t)
    alpha=schedule.sqrt_alphas_cumprod[t,None,None,None]
    sigma=schedule.sqrt_one_minus_alphas_cumprod[t,None,None,None]
    true_velocity=schedule.velocity_target(target,noise,t)
    velocity_error=torch.randn_like(target)*.2
    predicted_velocity=true_velocity+velocity_error
    predicted_clean=alpha*noisy-sigma*predicted_velocity
    actual=clean_prediction_process_loss(predicted_clean,target,None,'mse',sigma,(0,))
    expected=(velocity_error[:,0].square().mean()+(predicted_clean[:,1]-target[:,1]).square().mean())/2
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)


def test_tmax_value_gradient_and_masked_denominator_remain_unchanged():
    target=torch.randn(2,2,5,7)
    target[0,0,0,0]=float('nan')
    valid=torch.ones_like(target,dtype=torch.bool);valid[0,0,0,0]=False
    prediction=torch.randn_like(target,requires_grad=True)
    old=masked_loss(prediction,target,valid,'mse')
    old_grad=torch.autograd.grad(old,prediction)[0]
    sigma=torch.tensor([.05,.8])
    new=clean_prediction_process_loss(prediction,target,valid,'mse',sigma,(0,))
    new_grad=torch.autograd.grad(new,prediction)[0]
    torch.testing.assert_close(new_grad[:,1],old_grad[:,1],atol=0,rtol=0)
    torch.testing.assert_close(new_grad[:,0],old_grad[:,0]/sigma[:,None,None].square())
    assert new_grad[0,0,0,0]==0 and torch.isfinite(new)


@pytest.mark.parametrize('head',['diffusion_unet','diffusion_transformer'])
def test_both_heads_keep_native_clean_target_and_predictions(head):
    cfg=resolve_refinement_config({'refinement':{'type':head,'process_preconditioning_channels':[0],
        'unet':{'hidden_channels':8,'num_levels':2,'time_embedding_dim':16,'bottleneck_attention':False},
        'transformer':{'embedding_dim':24,'num_heads':4,'num_blocks':1,'patch_size':[2,2]},
        'diffusion':{'training_timesteps':10,'inference_steps':2}}})
    old=build_refiner(cfg,residual_channels=2,cond_channels=3)
    updated=replace(cfg,diffusion=replace(cfg.diffusion,velocity_loss_channels=(0,)))
    new=build_refiner(updated,residual_channels=2,cond_channels=3)
    new.load_state_dict(old.state_dict())
    target=torch.randn(2,2,7,9);condition=torch.randn(2,3,7,9)
    before=old.training_loss(target,condition,generator=torch.Generator().manual_seed(99))
    after=new.training_loss(target,condition,generator=torch.Generator().manual_seed(99))
    torch.testing.assert_close(after['prediction'],before['prediction'],atol=0,rtol=0)
    torch.testing.assert_close(after['process_target'],target,atol=0,rtol=0)
    after['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in new.parameters())
    assert _build_scientific_contract(cfg)!=_build_scientific_contract(updated)


@pytest.mark.parametrize('channels',[[0,0],[-1],[True],[1.5],'0'])
def test_invalid_selected_channel_configuration_fails(channels):
    with pytest.raises(ConfigValidationError,match='velocity_loss_channels'):
        DiffusionConfig.from_mapping({'velocity_loss_channels':channels})


def test_velocity_weighting_is_only_for_clean_prediction():
    with pytest.raises(ConfigValidationError,match='sample prediction'):
        DiffusionConfig.from_mapping({'prediction_type':'epsilon','velocity_loss_channels':[0]})


@pytest.mark.parametrize('sigma',[torch.tensor([0.]),torch.tensor([float('nan')])])
def test_invalid_noise_scale_fails(sigma):
    with pytest.raises(ValueError,match='positive noise scales'):
        clean_prediction_process_loss(torch.ones(1,2,3,5),torch.zeros(1,2,3,5),None,'mse',sigma,(0,))


def test_default_loss_is_bitwise_original_and_contract_omits_empty_option():
    prediction=torch.randn(2,2,7,9);target=torch.randn_like(prediction)
    expected=masked_loss(prediction,target,None,'mse')
    actual=clean_prediction_process_loss(prediction,target,None,'mse',torch.tensor([.1,.9]),())
    assert torch.equal(actual,expected)
    cfg=resolve_refinement_config({'refinement':{'type':'diffusion_unet'}})
    assert 'velocity_loss_channels' not in __import__('json').dumps(_build_scientific_contract(cfg))
