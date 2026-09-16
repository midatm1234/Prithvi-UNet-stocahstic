"""Loss-only boundary emphasis with valid denominators and unchanged Tmax."""
from dataclasses import replace
import pytest
import torch
from granitewxc.refinement import build_refiner, resolve_refinement_config
from granitewxc.refinement.base import masked_loss
from granitewxc.refinement.config import ConfigValidationError
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.process_losses import boundary_balanced_process_weights


def test_weighted_loss_matches_half_global_half_valid_band_means():
    target=torch.zeros(2,2,35,37)
    prediction=torch.arange(target.numel(),dtype=torch.float32).reshape_as(target)/1000
    valid=torch.ones_like(target);valid[:,0,0,:]=0;valid[0,0,12:15,17]=0
    prediction[valid==0]=float('nan')
    weights=boundary_balanced_process_weights(target,valid,(0,))
    y,x=torch.meshgrid(torch.arange(35),torch.arange(37),indexing='ij')
    distance=torch.minimum(torch.minimum(y,34-y),torch.minimum(x,36-x))
    bands=[distance==n for n in range(4)]+[(distance>=4)&(distance<8),(distance>=8)&(distance<16),distance>=16]
    errors=prediction[:,0].square()
    band_means=[errors[(valid[:,0]>0)&band].mean() for band in bands if ((valid[:,0]>0)&band).any()]
    pr=.5*errors[valid[:,0]>0].mean()+.5*torch.stack(band_means).mean()
    expected=(pr+prediction[:,1].square().mean())/2
    torch.testing.assert_close(masked_loss(prediction,target,weights),expected)
    torch.testing.assert_close(weights.sum((0,2,3)),valid.sum((0,2,3)))
    assert torch.equal(weights[:,1],valid[:,1])
    assert torch.all(weights[valid==0]==0)


@pytest.mark.parametrize('shape',[(1,2,1,1),(2,2,3,5),(2,2,7,9),(1,2,128,128)])
def test_empty_bands_missing_channels_and_zeros_are_safe(shape):
    prediction=torch.ones(shape,requires_grad=True)
    target=torch.zeros(shape)
    valid=torch.ones(shape,dtype=torch.bool)
    valid[:,0]=False
    weights=boundary_balanced_process_weights(prediction,valid,(0,))
    loss=masked_loss(prediction,target,weights)
    assert torch.isfinite(weights).all() and torch.equal(loss,torch.tensor(1.))
    loss.backward();assert torch.isfinite(prediction.grad).all()
    full=boundary_balanced_process_weights(prediction,None,(0,))
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full[:,0].sum(),torch.tensor(float(prediction[:,0].numel())))


def test_tmax_gradient_is_bitwise_original_with_unequal_masks():
    prediction=torch.randn(2,2,35,37,requires_grad=True);target=torch.randn_like(prediction)
    valid=torch.ones_like(prediction,dtype=torch.bool);valid[0,0,:3]=False
    before=torch.autograd.grad(masked_loss(prediction,target,valid),prediction)[0]
    weights=boundary_balanced_process_weights(prediction,valid,(0,))
    after=torch.autograd.grad(masked_loss(prediction,target,weights),prediction)[0]
    assert torch.equal(before[:,1],after[:,1])
    assert torch.all(after[~valid]==0)


@pytest.mark.parametrize('head',['flow_matching_unet','flow_matching_transformer','diffusion_unet','diffusion_transformer'])
def test_all_four_heads_keep_predictions_targets_and_sampling_identical(head):
    cfg=resolve_refinement_config({'refinement':{'type':head,
        'unet':{'hidden_channels':8,'num_levels':2,'time_embedding_dim':16,'bottleneck_attention':False},
        'transformer':{'embedding_dim':24,'num_heads':4,'num_blocks':1,'patch_size':[2,2]},
        'diffusion':{'training_timesteps':10,'inference_steps':2},'flow_matching':{'integration_steps':2}}})
    updated=replace(cfg,process_boundary_balance_channels=(0,))
    old=build_refiner(cfg,residual_channels=2,cond_channels=3)
    new=build_refiner(updated,residual_channels=2,cond_channels=3);new.load_state_dict(old.state_dict())
    target=torch.randn(2,2,7,9);cond=torch.randn(2,3,7,9)
    before=old.training_loss(target,cond,generator=torch.Generator().manual_seed(19))
    after=new.training_loss(target,cond,generator=torch.Generator().manual_seed(19))
    assert torch.equal(before['prediction'],after['prediction'])
    key='target_velocity' if cfg.is_flow_matching else 'process_target'
    assert torch.equal(before[key],after[key])
    for key in ('reconstruction_loss','multiscale_loss','gradient_loss','mean_bias_loss'):
        assert torch.equal(before[key],after[key])
    after['loss'].backward()
    with torch.inference_mode():
        a=old.sample(cond,generator=torch.Generator().manual_seed(5))
        b=new.sample(cond,generator=torch.Generator().manual_seed(5))
    assert torch.equal(a,b)
    assert 'process_boundary_balance' not in _build_scientific_contract(cfg)
    assert _build_scientific_contract(updated)['process_boundary_balance']['channels']==[0]


@pytest.mark.parametrize('channels',[[0,0],[-1],[True],[1.5],'0'])
def test_invalid_config_fails(channels):
    with pytest.raises(ConfigValidationError,match='process_boundary_balance_channels'):
        resolve_refinement_config({'refinement':{'type':'flow_matching_unet','process_boundary_balance_channels':channels}})


def test_disabled_weights_are_exact_original_object():
    reference=torch.randn(2,2,5,7);mask=torch.ones_like(reference,dtype=torch.bool)
    assert boundary_balanced_process_weights(reference,mask,()) is mask
    assert boundary_balanced_process_weights(reference,None,()) is None
