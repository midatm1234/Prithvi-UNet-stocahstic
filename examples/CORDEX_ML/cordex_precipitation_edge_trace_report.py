"""Plot saved computational stages for one actual ensemble member.

The model transforms the signed physical residual. There is no separate
transformed-precipitation baseline or reconstructed transformed-precipitation
field in this contract; those quantities must not be invented for a diagram.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def write_trajectory_diagnostics(root,output,report,stages,trajectory):
    """Pooled member statistics; inverse Jacobians before physical constraints."""
    valid=stages['valid'][0].astype(bool)
    yy,xx=np.indices(valid.shape[-2:])
    distance=np.minimum.reduce([yy,xx,valid.shape[-2]-1-yy,valid.shape[-1]-1-xx])
    regions={'edge':distance==0,'interior':distance>=16}
    recorded_times=root/'trajectory_times.json'
    times=json.loads(recorded_times.read_text()) if recorded_times.exists() else None
    statistics={'scope':'first date; moments pool actual members and valid cells; raw solver-state units',
                'times_recorded':times is not None,'trajectory':{},'native_predictions':{},'inverse_jacobian':{}}
    def moments(a,channel):
        result={}
        for name,region in regions.items():
            values=a[:,channel,valid[channel]&region].astype(np.float64)
            if values.size:
                result[name]={'mean':float(values.mean()),'std':float(values.std()),'rms':float(np.sqrt(np.square(values).mean()))}
        return result
    for key,state in trajectory.items():
        statistics['trajectory'][key]={'time':times.get(key) if times else None,
            'variables':{v:moments(state,c) for c,v in enumerate(('pr','tasmax'))}}
    with np.load(root/'network_outputs.npz') as saved:
        for key in saved.files:
            if not key.startswith('process_prediction_'):continue
            suffix=key.removeprefix('process_prediction_')
            statistics['native_predictions'][key]={'time':float(saved['process_time_'+suffix].reshape(-1)[0]),
                'variables':{v:moments(saved[key],c) for c,v in enumerate(('pr','tasmax'))}}
    with np.load(root/'normalization_fields.npz') as saved:
        normalizer={k:saved[k][0] for k in ('mean','scale','count')}
    u=stages['denormalized_transformed_residual'][0].astype(np.float64)
    jacobian=np.broadcast_to(normalizer['scale'],u.shape).astype(np.float64).copy()
    meta=report['normalizer']
    if meta.get('signed_log_nonnegative_channels',False):
        jacobian[:,0]*=float(meta['signed_log_scale'])*np.exp(np.abs(u[:,0]))
    elif meta.get('signed_sqrt_nonnegative_channels',False):
        jacobian[:,0]*=2*float(meta['signed_sqrt_scale'])*(1+np.abs(u[:,0]))
    maps={'inverse_jacobian_mean':jacobian.mean(0),'inverse_jacobian_max':jacobian.max(0),**normalizer}
    for c,var in enumerate(('pr','tasmax')):
        statistics['inverse_jacobian'][var]=moments(jacobian,c)
        fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        entries=list(statistics['trajectory'].items())
        horizontal=[row['time'] for _,row in entries] if times else list(range(len(entries)))
        for region in regions:
            for ax,metric in zip(axes[:2],('mean','std')):
                ax.plot(horizontal,[row['variables'][var].get(region,{}).get(metric,np.nan) for _,row in entries],label=region)
                ax.set_ylabel('Solver-state '+metric+' (normalized units)')
                ax.set_xlabel('Native process time' if times else 'Saved state index (time not recorded)')
            native=list(statistics['native_predictions'].values())
            axes[2].plot([row['time'] for row in native],[row['variables'][var].get(region,{}).get('rms',np.nan) for row in native],label=region)
        axes[2].set_xlabel('Native process time');axes[2].set_ylabel('Native prediction RMS')
        for ax in axes:ax.grid(alpha=.25);ax.legend()
        if report['head'].startswith('diffusion'):
            for ax in axes:
                if times or ax is axes[2]:ax.invert_xaxis()
        fig.suptitle(report['head']+': '+var+', '+str(stages['dates'][0]))
        fig.savefig(output/(var+'_trajectory_moments.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(23,5),constrained_layout=True)
        for ax,(key,array) in zip(axes,maps.items()):
            values=np.where(valid[c],array[c],np.nan)
            artist=ax.pcolormesh(stages['longitude'],stages['latitude'],values,shading='auto',cmap='viridis')
            ax.set_title(key.replace('_',' '));fig.colorbar(artist,ax=ax,shrink=.7)
        fig.suptitle(var+': normalization fields and physical inverse sensitivity per normalized unit')
        fig.savefig(output/(var+'_inverse_sensitivity.png'),dpi=150);plt.close(fig)
    np.savez_compressed(output/'inverse_sensitivity_maps.npz',**maps,latitude=stages['latitude'],longitude=stages['longitude'])
    (output/'trajectory_statistics.json').write_text(json.dumps(statistics,indent=2,allow_nan=False),newline='\n')


def main(args):
    root=Path(args.trace)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((root/'report.json').read_text())
    with np.load(root/'stages.npz',allow_pickle=True) as data:
        stages={key:data[key] for key in data.files}
    with np.load(root/'trajectory.npz') as data:
        trajectory={key:data[key] for key in data.files}
    with np.load(root/'network_outputs.npz') as data:
        native=data['process_prediction_000']
    initial=next(value for key,value in trajectory.items() if key.startswith('initial_'))
    date=str(stages['dates'][0])
    member=args.member
    if not 0<=member<stages['physical_members'].shape[1]:
        raise ValueError('Requested member is absent from the saved trace')
    longitude,latitude=stages['longitude'],stages['latitude']
    metadata={'trace':str(root.resolve()),'date':date,'member':member,
              'scope':'individual member; first traced date; no averaging before inverse',
              'head':report['head'],'limits':{}}
    for channel,variable in enumerate(('pr','tasmax')):
        def stage(key):
            array=stages[key][0]
            return array[member,channel] if array.ndim==4 else array[channel]
        panels=[
            ('Physical target',stage('target_physical'),'physical'),
            ('Phase-1 physical prediction',stage('phase1_physical'),'physical'),
            ('Signed physical target residual',stage('target_physical_residual'),'physical'),
            ('Transformed target residual',stage('target_transformed_residual'),'transformed'),
            ('Normalized target residual',stage('target_normalized_residual'),'normalized'),
            ('Initial stochastic state',initial[member,channel],'normalized'),
            ('Native process prediction at source',native[member,channel],'native'),
            ('Final normalized residual',stage('final_normalized_residual'),'normalized'),
            ('De-normalized transformed residual',stage('denormalized_transformed_residual'),'transformed'),
            ('Inverse-transformed physical residual',stage('physical_residual'),'physical'),
            ('Reconstructed physical member',stage('unbounded_members'),'physical'),
            ('After existing nonnegative constraint',stage('physical_members'),'physical'),
        ]
        valid=stages['valid'][0,channel].astype(bool)
        fig,axes=plt.subplots(3,4,figsize=(19,13),constrained_layout=True)
        metadata['limits'][variable]={}
        for ax,(label,field,space) in zip(axes.flat,panels):
            shown=np.where(valid,field,np.nan)
            if not np.isfinite(shown[valid]).all():
                raise ValueError(f'Nonfinite valid values in {label}')
            low,high=float(np.nanmin(shown)),float(np.nanmax(shown))
            # Each distinct computational space has its own explicit full range.
            artist=ax.pcolormesh(longitude,latitude,shown,shading='auto',cmap='viridis',vmin=low,vmax=high)
            units=('mm/day' if variable=='pr' else 'K') if space=='physical' else space+' units'
            fig.colorbar(artist,ax=ax,label=units,shrink=.75)
            ax.set_title(label,fontsize=10)
            ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
            metadata['limits'][variable][label]=[low,high]
        fig.suptitle(f"{report['head']}: {variable}, {date}, member {member}")
        fig.savefig(output/f'{variable}_computational_stages.png',dpi=160)
        plt.close(fig)
    write_trajectory_diagnostics(root,output,report,stages,trajectory)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--member',type=int,default=0)
    main(parser.parse_args())
