"""Noise-source endpoint, valid-domain bands and all-head training regressions."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from granitewxc.refinement import build_refiner, resolve_refinement_config
from granitewxc.refinement.boundary import boundary_distance, boundary_regions, physical_metrics
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.preconditioning import diffusion_coefficients, predict_process


HEADS = ('flow_matching_unet','flow_matching_transformer','diffusion_unet','diffusion_transformer')


def configuration(head):
    return resolve_refinement_config({'refinement': {
        'type': head, 'process_preconditioning_channels':[0],
        'unet': {'hidden_channels':8,'num_levels':2,'time_embedding_dim':16,'bottleneck_attention':False},
        'transformer': {'embedding_dim':24,'num_heads':4,'num_blocks':1,
                        'patch_size':[2,2],'spatial_alignment':'coordinates'},
        **({'flow_matching':{'integration_steps':2}} if head.startswith('flow') else
           {'diffusion':{'training_timesteps':10,'inference_steps':2}})}})


@pytest.mark.parametrize('head',HEADS)
def test_all_heads_train_sample_odd_domains_and_keep_native_target(head):
    cfg=configuration(head)
    model=build_refiner(cfg,residual_channels=2,cond_channels=3)
    cond=torch.randn(2,3,7,9)
    target=torch.randn(2,2,7,9)
    valid=torch.ones_like(target,dtype=torch.bool);valid[0,0,0,0]=False
    target[0,0,0,0]=0
    result=model.training_loss(target,cond,valid,generator=torch.Generator().manual_seed(12))
    assert torch.isfinite(result['loss'])
    process_target=result['target_velocity'] if cfg.is_flow_matching else result['process_target']
    from granitewxc.refinement.base import masked_loss
    torch.testing.assert_close(result['process_loss'],masked_loss(result['prediction'],process_target,valid,'mse'))
    result['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.net.parameters())
    model.eval()
    sample=model.sample(cond,generator=torch.Generator().manual_seed(13))
    assert sample.shape==target.shape and torch.isfinite(sample).all()


@pytest.mark.parametrize('head',HEADS[:2])
@pytest.mark.parametrize('shape',[(7,9),(16,16)])
def test_flow_source_response_is_exact_on_edges_corners_and_interior(head,shape):
    model=build_refiner(configuration(head),residual_channels=2,cond_channels=3).eval()
    # Nonzero arbitrary weights ensure this is architectural, not zero-init.
    for p in model.net.parameters():
        torch.nn.init.normal_(p, std=.1)
    source=torch.randn(2,2,*shape);perturb=torch.randn_like(source);cond=torch.randn(2,3,*shape)
    t=torch.zeros(2)
    p=model.predict_process(source,cond,t)
    q=model.predict_process(source+perturb,cond,t)
    torch.testing.assert_close((q-p)[:,0],-(1-model.sigma_min)*perturb[:,0],atol=1e-6,rtol=1e-6)
    # Untouched variable uses exactly the historical prediction mapping.
    torch.testing.assert_close(p[:,1],model.net(source,cond,model._embed_time(t))[:,1],atol=0,rtol=0)


@pytest.mark.parametrize('kind',['epsilon','sample','velocity'])
def test_diffusion_noise_endpoint_has_no_spatially_learned_identity(kind):
    class Net(torch.nn.Module):
        def forward(self,x,c,t):return torch.nn.functional.avg_pool2d(x,3,1,1)+c[:,:2]
    source=torch.randn(2,2,7,9);delta=torch.randn_like(source);cond=torch.randn(2,3,7,9)
    coefficients=diffusion_coefficients(torch.zeros(2),torch.ones(2),kind,source)
    p=predict_process(Net(),source,cond,torch.zeros(2),coefficients,(0,))
    q=predict_process(Net(),source+delta,cond,torch.zeros(2),coefficients,(0,))
    expected=delta[:,0] if kind=='epsilon' else torch.zeros_like(delta[:,0])
    torch.testing.assert_close((q-p)[:,0],expected,atol=1e-6,rtol=1e-6)


def test_new_process_semantics_change_checkpoint_contract_only_when_enabled():
    cfg=configuration('flow_matching_unet')
    old=_build_scientific_contract(replace(cfg,process_preconditioning_channels=()))
    new=_build_scientific_contract(cfg)
    assert 'process_preconditioning' not in old
    assert new.pop('process_preconditioning')=={'version':'gaussian_source_endpoint_v1','channels':[0]}
    assert new==old


def test_boundary_distance_retains_valid_zeros_holes_and_geographic_orientation():
    valid=np.ones((40,44),bool);valid[20,20]=False
    distance=boundary_distance(valid)
    assert distance[0,0]==0 and distance[1,1]==1 and distance[20,20]==-1
    assert distance[19,19]==0 and distance[17,17]==2
    d,regions=boundary_regions(valid,latitude=np.linspace(-35,-20,40),longitude=np.linspace(20,35,44))
    assert regions['north_1'][-1].all() and not regions['north_1'][0].any()
    truth=np.zeros((3,40,44));prediction=truth.copy();prediction[:,0]=2
    result=physical_metrics(prediction,truth,np.broadcast_to(valid,truth.shape),region=regions['outer_1'],precipitation=True)
    assert result['count']==3*regions['outer_1'].sum()
    assert result['bias']>0 and result['wet_frequency_bias']>0
    prediction[:,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite prediction'):
        physical_metrics(prediction,truth,region=regions['outer_1'])

@pytest.mark.parametrize('head', ('flow_matching_transformer','diffusion_transformer'))
def test_unit_precipitation_gate_preserves_signed_corrections_and_temperature(head):
    cfg=replace(configuration(head), unit_correction_gate_channels=(0,))
    model=build_refiner(cfg,residual_channels=2,cond_channels=3)
    with torch.no_grad(): model.correction_gate.fill_(0.3)
    residual=torch.randn(3,4,2,7,9)
    actual=model.apply_correction_gate(residual)
    torch.testing.assert_close(actual[:,:,0],residual[:,:,0],rtol=0,atol=0)
    torch.testing.assert_close(actual[:,:,1],residual[:,:,1]*.3)
    assert _build_scientific_contract(cfg)['unit_correction_gate_channels']==[0]


def test_signed_log_gate_does_not_commute_with_physical_reconstruction():
    # The prior gate calibration multiplied a transformed residual, whereas
    # inference multiplied its physical inverse. A unit precipitation gate
    # removes this discrepancy without modifying signed corrections.
    physical=torch.tensor([-20.,-1.,0.,1.,20.])
    transformed=physical.sign()*physical.abs().log1p()
    inverse=lambda x:x.sign()*x.abs().expm1()
    assert not torch.allclose(inverse(.3*transformed),.3*physical)
    torch.testing.assert_close(inverse(transformed),physical)

@pytest.mark.parametrize('head',('flow_matching_unet','flow_matching_transformer'))
def test_precipitation_auxiliary_ablation_keeps_original_temperature_coefficient(head):
    cfg=configuration(head)
    cfg=replace(cfg,flow_matching=replace(cfg.flow_matching,mean_path_loss_weight=.25,mean_path_channel_weights=(0.,1.)))
    model=build_refiner(cfg,residual_channels=2,cond_channels=3)
    target=torch.randn(2,2,7,9);cond=torch.randn(2,3,7,9)
    result=model.training_loss(target,cond)
    from granitewxc.refinement.base import masked_loss
    expected=.5*masked_loss(result['zero_source_velocity_prediction'][:,1:2],target[:,1:2],None,'huber')
    torch.testing.assert_close(result['mean_path_loss'],expected)
    result['mean_path_loss'].backward()
    projection=model.net.out_proj
    assert torch.count_nonzero(projection.weight.grad[0])==0
    assert projection.weight.grad[1].abs().sum()>0

@pytest.mark.parametrize('scale',[0.1,1.,10.])
def test_signed_sqrt_round_trip_edges_signed_dry_values_and_temperature(scale):
    from granitewxc.refinement import ResidualNormalizer
    from granitewxc.refinement.config import ResidualNormalizationConfig
    cfg=ResidualNormalizationConfig(signed_sqrt_nonnegative_channels=True,signed_sqrt_scale=scale)
    model=ResidualNormalizer(2,cfg,nonnegative_mask=torch.tensor([True,False]))
    field=torch.linspace(-1000.,1000.,2*2*7*9).reshape(2,2,7,9)
    field[:,0,0,0]=0.;field[:,0,-1,-1]=-1e-8
    valid=torch.ones_like(field,dtype=torch.bool);valid[0,0,3,3]=False
    model.update(field,valid);model.finalize()
    restored=model.denormalize(model.normalize(field,valid))
    torch.testing.assert_close(restored[valid],field[valid],rtol=2e-6,atol=2e-4)
    torch.testing.assert_close(model._forward_transform(field)[:,1],field[:,1],rtol=0,atol=0)
    torch.testing.assert_close(model._inverse_transform(field)[:,1],field[:,1],rtol=0,atol=0)
    zero=torch.zeros(1,2,1,1,requires_grad=True)
    grad=torch.autograd.grad(model._forward_transform(zero).sum(),zero)[0]
    torch.testing.assert_close(grad.flatten(),torch.tensor([1/(2*scale),1.]))
    inverse_grad=torch.autograd.grad(model._inverse_transform(zero).sum(),zero)[0]
    torch.testing.assert_close(inverse_grad.flatten(),torch.tensor([2*scale,1.]))
    assert model.metadata()['signed_sqrt_mask']==[True,False]


def test_signed_sqrt_is_exclusive_and_part_of_checkpoint_contract():
    from granitewxc.refinement.config import ConfigValidationError
    with pytest.raises(ConfigValidationError,match='mutually exclusive'):
        resolve_refinement_config({'refinement':{'type':'flow_matching_unet','residual_normalization':{'signed_log_nonnegative_channels':True,'signed_sqrt_nonnegative_channels':True}}})
    cfg=configuration('flow_matching_unet')
    sqrt_cfg=replace(cfg,residual_normalization=replace(cfg.residual_normalization,signed_sqrt_nonnegative_channels=True))
    assert _build_scientific_contract(cfg)!=_build_scientific_contract(sqrt_cfg)
