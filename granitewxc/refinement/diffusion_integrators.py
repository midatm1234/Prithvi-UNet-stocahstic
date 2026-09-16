"""Deterministic integration in the variance-preserving noise angle."""
from __future__ import annotations

import torch


def integrate_noise_angle(initial_state, timesteps, schedule, predict_clean,
                          trajectory_callback=None):
    """Second-order Heun integration of dx/dtheta=(alpha*x-x0)/sigma.

    Here alpha=cos(theta) and sigma=sin(theta), using the actual discrete
    training schedule. The network is evaluated only at trained timestep
    indices. The last step uses the usual clean prediction at the smallest
    timestep, without evaluating a network at an invented negative index.
    No spatial filtering, residual clipping, or stochastic innovation occurs.
    """
    if timesteps.ndim != 1 or not timesteps.numel():
        raise ValueError('A nonempty one-dimensional timestep grid is required')
    if bool((timesteps < 0).any()) or bool((timesteps >= schedule.num_train_timesteps).any()):
        raise ValueError('Noise-angle timesteps are outside the training schedule')
    if bool((timesteps[1:] >= timesteps[:-1]).any()):
        raise ValueError('Noise-angle timesteps must strictly decrease')
    x=initial_state.float() if initial_state.dtype in (torch.float16,torch.bfloat16) else initial_state.clone()
    alpha=schedule.sqrt_alphas_cumprod.to(x)
    sigma=schedule.sqrt_one_minus_alphas_cumprod.to(x)
    theta=torch.atan2(sigma,alpha)

    def velocity(state,t):
        clean=predict_clean(state,t).to(state)
        if clean.shape != state.shape or not bool(torch.isfinite(clean).all()):
            raise ValueError('Clean prediction must be finite and match the state shape')
        return (alpha[t]*state-clean)/sigma[t],clean

    if trajectory_callback is not None:
        trajectory_callback('initial',0,timesteps[0].detach().clone(),x.detach().clone())
    for index,t in enumerate(timesteps):
        first,clean=velocity(x,t)
        terminal=index==len(timesteps)-1
        if terminal:
            x=clean
            state_time=torch.tensor(-1,device=x.device)
        else:
            previous=timesteps[index+1]
            delta=theta[previous]-theta[t]
            trial=x+delta*first
            second,_=velocity(trial,previous)
            x=x+0.5*delta*(first+second)
            state_time=previous
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError('Noise-angle integration produced a nonfinite state')
        if trajectory_callback is not None:
            stage='final_normalized_residual' if terminal else 'intermediate'
            trajectory_callback(stage,index+1,state_time.detach().clone(),x.detach().clone())
    return x
