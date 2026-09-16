"""Analytic distribution and parameterization tests for noise-angle Heun."""
from dataclasses import replace

import pytest
import torch

from granitewxc.refinement import build_refiner, resolve_refinement_config
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.config import ConfigValidationError, DiffusionConfig
from granitewxc.refinement.diffusion_integrators import integrate_noise_angle
from granitewxc.refinement.schedules import DiffusionSchedule


def test_stationary_gaussian_keeps_pointwise_noise_without_edge_variance_loss():
    schedule=DiffusionSchedule()
    initial=torch.randn(3,2,7,9)
    for steps in (5,20,100):
        times=schedule.inference_timesteps(steps,initial.device)
        actual=integrate_noise_angle(initial,times,schedule,
            lambda state,t:schedule.sqrt_alphas_cumprod[t]*state)
        expected=schedule.sqrt_alphas_cumprod[times[-1]]*initial
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)


def test_conditional_gaussian_converges_to_analytic_transport_on_odd_grid():
    schedule=DiffusionSchedule().double()
    initial=torch.randn(2,2,7,9,dtype=torch.float64)
    mean=torch.linspace(-2,2,63,dtype=torch.float64).reshape(1,1,7,9)
    variance=torch.tensor([.3,2.],dtype=torch.float64).reshape(1,2,1,1)
    alpha,sigma=schedule.sqrt_alphas_cumprod,schedule.sqrt_one_minus_alphas_cumprod
    def clean(state,t):
        total=alpha[t]**2*variance+sigma[t]**2
        return mean+alpha[t]*variance/total*(state-alpha[t]*mean)
    start_sd=(alpha[-1]**2*variance+sigma[-1]**2).sqrt()
    end_sd=(alpha[0]**2*variance+sigma[0]**2).sqrt()
    endpoint=alpha[0]*mean+end_sd*(initial-alpha[-1]*mean)/start_sd
    exact=clean(endpoint,0)
    errors=[]
    for steps in (16,64):
        actual=integrate_noise_angle(initial,schedule.inference_timesteps(steps,initial.device),schedule,clean)
        errors.append(float((actual-exact).square().mean().sqrt()))
    assert errors[1]<errors[0]/5
    assert errors[1]<.002


@pytest.mark.parametrize('head',['diffusion_unet','diffusion_transformer'])
@pytest.mark.parametrize('prediction_type',['sample','epsilon','velocity'])
def test_both_backbones_use_same_heun_path_for_native_parameterizations(head,prediction_type):
    cfg=resolve_refinement_config({'refinement':{'type':head,
        'unet':{'hidden_channels':8,'num_levels':2,'time_embedding_dim':16,'bottleneck_attention':False},
        'transformer':{'embedding_dim':24,'num_heads':4,'num_blocks':1,'patch_size':[2,2]},
        'diffusion':{'solver':'heun','prediction_type':prediction_type,'inference_steps':20}}})
    model=build_refiner(cfg,residual_channels=2,cond_channels=3)
    def predict(state,cond,t):
        if prediction_type=='velocity':return torch.zeros_like(state)
        coefficient=model.schedule.sqrt_alphas_cumprod if prediction_type=='sample' else model.schedule.sqrt_one_minus_alphas_cumprod
        return coefficient[t].reshape(-1,1,1,1)*state
    model.predict_process=predict
    initial=torch.randn(2,2,7,9)
    snapshots=[]
    actual=model.sample(torch.zeros(2,3,7,9),initial_state=initial,
        trajectory_callback=lambda stage,i,t,x:snapshots.append((stage,x)))
    expected=model.schedule.sqrt_alphas_cumprod[0]*initial
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
    assert len(snapshots)==21
    assert snapshots[-1][0]=='final_normalized_residual'


@pytest.mark.parametrize('options',[{'eta':.1},{'clip_sample':True}])
def test_heun_rejects_incompatible_stochastic_or_clipped_config(options):
    with pytest.raises(ConfigValidationError,match='Heun requires'):
        DiffusionConfig.from_mapping({'solver':'heun',**options})


def test_default_ddim_contract_stays_compatible_and_heun_is_explicit():
    cfg=resolve_refinement_config({'refinement':{'type':'diffusion_unet'}})
    legacy=_build_scientific_contract(cfg)
    heun=_build_scientific_contract(replace(cfg,diffusion=replace(cfg.diffusion,solver='heun')))
    assert legacy!=heun
    assert '"solver": "ddim"' not in __import__('json').dumps(legacy)


def test_invalid_or_nonfinite_integration_inputs_fail():
    schedule=DiffusionSchedule()
    x=torch.zeros(1,2,3,5)
    with pytest.raises(ValueError,match='strictly decrease'):
        integrate_noise_angle(x,torch.tensor([2,2]),schedule,lambda state,t:state)
    with pytest.raises(ValueError,match='finite'):
        integrate_noise_angle(x,torch.tensor([2,0]),schedule,lambda state,t:state+float('nan'))
