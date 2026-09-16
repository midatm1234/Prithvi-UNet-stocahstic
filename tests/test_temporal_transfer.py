"""Focused scientific contracts for the separate pretrained-transfer branch."""
import copy
import pytest
import torch
from torch import nn

from granitewxc.models.model import audited_pretrained_load, extract_pretrained_state
from granitewxc.temporal.transfer import (
    FrozenLocalGlobalPair, RegionalPretrainedBranch, RegionalTransferModel,
    restore_transfer_adapters, save_transfer_adapters, active_representation_contract, SCHEMA,
)


def make_branch(mask_indices=None, *, initialization="random_control", seed=31, heads=2):
    # Tiny seeded fixtures exercise mode semantics; no toy weights claim real pretraining.
    torch.manual_seed(seed)
    pair = FrozenLocalGlobalPair(width=8, heads=heads, multiplier=2)
    return RegionalPretrainedBranch(pair, initialization=initialization,
        indices=[0, 1], mask_indices=mask_indices,
        input_mu=torch.zeros(1, 2, 1, 1), input_sigma=torch.ones(1, 2, 1, 1),
        latent_channels=3, width=8, token_grid=(4, 4), local_grid=(2, 2))


def test_official_checkpoint_layout_is_recognized():
    state = {'encoder.weight': torch.ones(1)}
    assert extract_pretrained_state({'model_state': state, 'optimizer_state': {}}) == state


def test_coverage_distinguishes_learned_parameters_buffers_and_fixed_scalers():
    model = nn.Module()
    model.backbone = nn.Linear(2, 3)
    model.register_buffer('buffer', torch.ones(4))
    model.input_scalers_mu = nn.Parameter(torch.zeros(2), requires_grad=False)
    report = audited_pretrained_load(model, model.state_dict(), require_backbone=True)
    assert report['backbone_parameters']['loaded_numel'] == 9
    assert report['learned_parameters']['total_numel'] == 9
    assert report['buffers']['loaded_numel'] == 4
    assert report['fixed_normalization_parameters']['loaded_numel'] == 0


def test_incompatible_claim_fails_before_any_parameter_mutation():
    model = nn.Module()
    model.backbone = nn.Linear(2, 3)
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='zero compatible backbone'):
        audited_pretrained_load(model, {'backbone.weight': torch.zeros(3, 4)}, require_backbone=True)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_strict_selected_component_rejects_partial_loading():
    model = nn.Module()
    model.backbone = nn.Linear(2, 3)
    with pytest.raises(ValueError, match='Incomplete pretrained component'):
        audited_pretrained_load(model, {'backbone.weight': model.backbone.weight},
                                require_backbone=True, strict=True)


def test_frozen_pretrained_attention_transmits_gradients_and_updates_input_adapter():
    branch = make_branch()
    old_pair = copy.deepcopy(branch.pair.state_dict())
    old_input = branch.input_projection.weight.detach().clone()
    old_mu = branch.input_scalers_mu.clone()
    states = torch.randn(2, 2, 2, 8, 8)
    target = torch.randn(2, 3, 4, 4)
    opt = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=1e-3)
    loss = (branch(states, torch.tensor([24., 24.])) - target).square().mean()
    loss.backward()
    assert branch.input_projection.weight.grad.abs().max() > 0
    assert all(p.grad is None for p in branch.pair.parameters())
    opt.step()
    assert not torch.equal(old_input, branch.input_projection.weight)
    assert all(torch.equal(v, old_pair[k]) for k, v in branch.pair.state_dict().items())
    assert torch.equal(old_mu, branch.input_scalers_mu)
    changed_history = states.clone()
    changed_history[:, 0] += 2
    assert not torch.equal(branch(states, torch.tensor([24.,24.])), branch(changed_history, torch.tensor([24.,24.])))


def test_invalid_atmospheric_values_are_not_used_as_weather():
    branch = make_branch(mask_indices=[2,3])
    states = torch.randn(1,2,4,8,8)
    states[:,:,2:] = 1
    states[:,:,2:,:,0:4] = 0
    perturbed = states.clone()
    perturbed[:,:,:2,:,0:4] = 10000
    assert torch.equal(branch(states, torch.tensor([24.])), branch(perturbed, torch.tensor([24.])))


class DummyPhase1(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_adapter = None
        self._temporal_ctx = None
        self.spatial = nn.Conv2d(2,3,1)
    def forward(self, batch):
        z = self.spatial(batch['x'])
        if self._temporal_ctx is not None:
            delta,_ = self.temporal_adapter(z, self._temporal_ctx['state'])
            z = z + delta
        return z


def test_prediction_ignores_verification_targets_and_preserves_independent_phase1():
    base = DummyPhase1()
    reference = copy.deepcopy(base)
    model = RegionalTransferModel(base, make_branch(), output_shape=(8,8), output_channels=3)
    batch = {'x':torch.randn(1,2,2,8,8), 'y':torch.randn(1,2,3,8,8),
             '__timestamps':[['2001-01-01T00:00:00','2001-01-02T00:00:00']]}
    a=model(batch)
    b=model(dict(batch,y=batch['y']*1000))
    assert torch.equal(a,b)
    assert torch.equal(model(batch,use_branch=False), reference({'x':batch['x'][:,-1]}))
    assert not torch.equal(a,model(batch,history_mode='duplicate_current'))


def test_missing_or_discontinuous_history_is_rejected():
    model=RegionalTransferModel(DummyPhase1(),make_branch(),output_shape=(8,8),output_channels=3)
    batch={'x':torch.randn(1,2,2,8,8)}
    with pytest.raises(ValueError,match='timestamps'): model(batch)
    batch['__timestamps']=[['2001-01-01T00:00:00','2001-01-03T00:00:00']]
    with pytest.raises(ValueError,match='discontinuity'): model(batch)


@pytest.mark.parametrize('path',[
    'examples/CORDEX_ML/sa_regional_pretrained_transfer_v2.yaml',
    'examples/NARR_PRISM/narr_regional_pretrained_transfer_v2.yaml'])
def test_required_transfer_configuration_exists_and_pins_source(path):
    from pathlib import Path
    import yaml
    config=yaml.safe_load(Path(path).read_text())
    assert len(config['foundation']['revision'])==40
    assert len(config['foundation']['sha256'])==64
    assert config['semantics']['native_merra2_input'] is False
    assert config['semantics']['downscaling_lead_hours']==0

@pytest.mark.parametrize("initialization", ["pretrained", "random_control"])
def test_adapter_checkpoint_restores_optimizer_and_identical_predictions(tmp_path, initialization):
    branch=make_branch(initialization=initialization)
    opt=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=1e-3)
    states=torch.randn(1,2,2,8,8)
    hours=torch.tensor([24.])
    branch(states,hours).square().sum().backward()
    opt.step()
    saved=branch(states,hours).detach()
    path=tmp_path/'adapters.pt'
    save_transfer_adapters(branch,path,foundation_sha256='foundation',phase1_digest='phase1',
        config={'initialization':initialization,'atmospheric_indices':[0,1],
                'token_grid':[4,4],'local_grid':[2,2]},optimizer=opt,optimizer_updates=1)
    payload=torch.load(path,weights_only=False)
    assert payload['active_representation']==active_representation_contract(branch)
    restored=make_branch(initialization=initialization)
    opt2=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=2e-3)
    restore_transfer_adapters(restored,path,foundation_sha256='foundation',phase1_digest='phase1',optimizer=opt2)
    assert torch.equal(saved,restored(states,hours))
    assert opt2.state_dict()['param_groups']==opt.state_dict()['param_groups']
    assert all(torch.equal(opt.state_dict()['state'][i]['exp_avg'],v['exp_avg'])
               for i,v in opt2.state_dict()['state'].items())
    with pytest.raises(ValueError,match='source differs'):
        restore_transfer_adapters(restored,path,foundation_sha256='wrong',phase1_digest='phase1')
    restored.indices=(1,0)
    with pytest.raises(ValueError,match='semantic/geometry contract mismatch'):
        restore_transfer_adapters(restored,path,foundation_sha256='foundation',phase1_digest='phase1')
    restored.indices=(0,1)
    restored.input_scalers_mu.add_(1)
    with pytest.raises(ValueError,match='normalization mismatch'):
        restore_transfer_adapters(restored,path,foundation_sha256='foundation',phase1_digest='phase1')


def test_pretrained_native_time_scalar_matches_upstream_samplespec_and_zero_main_lead():
    pair=FrozenLocalGlobalPair(width=8,heads=2,multiplier=2)
    observed={}
    h1=pair.input_time_embedding.register_forward_pre_hook(lambda module,args: observed.update(input=args[0].detach().clone()))
    h2=pair.lead_time_embedding.register_forward_pre_hook(lambda module,args: observed.update(lead=args[0].detach().clone()))
    import pandas as pd
    from PrithviWxC.dataloaders.merra2 import SampleSpec
    spec=SampleSpec((pd.Timestamp("2001-01-01"),pd.Timestamp("2001-01-02")),
                    lead_time=0,target=pd.Timestamp("2001-01-02"))
    pair(torch.zeros(1,4,4,8),torch.tensor([spec.input_time]))
    h1.remove();h2.remove()
    assert observed['input'].item()==spec.input_time==24
    assert observed['lead'].item()==0


def test_transfer_preserves_tiled_halo_and_scaler_offsets_without_target_inputs():
    from granitewxc.temporal.transfer import FRAME_METADATA
    class GeometryPhase1(DummyPhase1):
        def forward(self,batch):
            self.last_batch=batch
            return super().forward(batch)
    base=GeometryPhase1()
    model=RegionalTransferModel(base,make_branch(),output_shape=(6,6),output_channels=3)
    batch={'x':torch.randn(1,2,2,8,8),'y':torch.randn(1,2,3,6,6),
           '__timestamps':[['2001-01-01T00:00:00','2001-01-02T00:00:00']],
           '__output_crop':torch.tensor([[1,1,6,6]]),
           '__scaler_offset':torch.tensor([[12,18]]),
           '__input_scaler_offset':torch.tensor([[11,17]]),
           '__output_scaler_offset':torch.tensor([[12,18]])}
    model(batch)
    for key in FRAME_METADATA:
        if key in batch: assert torch.equal(base.last_batch[key],batch[key])
    assert base.last_batch['y'].shape==(1,3,6,6)
    assert torch.count_nonzero(base.last_batch['y'])==0


def save_fixture_checkpoint(branch, path):
    optimizer=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=1e-3)
    save_transfer_adapters(branch,path,foundation_sha256='same_asset',phase1_digest='same_phase1',
        config={'initialization':branch.initialization,'atmospheric_indices':[0,1],
                'token_grid':[4,4],'local_grid':[2,2]},optimizer=optimizer,optimizer_updates=0)


@pytest.mark.parametrize('source_mode,target_mode,target_seed,target_heads,message',[
    ('random_control','pretrained',31,2,'initialization differs'),
    ('pretrained','random_control',99,2,'initialization differs'),
    ('pretrained','pretrained',99,2,'state or computation differs'),
    ('random_control','random_control',99,2,'state or computation differs'),
    ('pretrained','pretrained',31,4,'state or computation differs'),
])
def test_restore_rejects_different_active_representation_before_mutation(
        tmp_path, source_mode, target_mode, target_seed, target_heads, message):
    source=make_branch(initialization=source_mode)
    with torch.no_grad():
        source.input_projection.weight.add_(0.5)
    path=tmp_path/'adapters.pt'
    save_fixture_checkpoint(source,path)
    target=make_branch(initialization=target_mode,seed=target_seed,heads=target_heads)
    original=copy.deepcopy(target.state_dict())
    optimizer=torch.optim.AdamW([p for p in target.parameters() if p.requires_grad],lr=2e-3)
    optimizer_before=copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError,match=message):
        restore_transfer_adapters(target,path,foundation_sha256='same_asset',
                                  phase1_digest='same_phase1',optimizer=optimizer)
    assert all(torch.equal(v,original[k]) for k,v in target.state_dict().items())
    assert optimizer.state_dict()==optimizer_before


@pytest.mark.parametrize('schema,message',[
    ('granitewxc.regional_pretrained_transfer.v1','Legacy transfer checkpoint lacks verified'),
    ('granitewxc.regional_pretrained_transfer.v2','Legacy transfer checkpoint lacks verified'),
    (SCHEMA,'lacks active representation identity'),
])
def test_unverifiable_legacy_or_missing_identity_checkpoint_fails_clearly(tmp_path,schema,message):
    branch=make_branch()
    path=tmp_path/'adapters.pt'
    save_fixture_checkpoint(branch,path)
    payload=torch.load(path,weights_only=False)
    payload['schema']=schema
    del payload['active_representation']
    payload['branch_adapters']['gate'].add_(1)
    torch.save(payload,path)
    original=copy.deepcopy(branch.state_dict())
    with pytest.raises(ValueError,match=message):
        restore_transfer_adapters(branch,path,foundation_sha256='same_asset',phase1_digest='same_phase1')
    assert all(torch.equal(v,original[k]) for k,v in branch.state_dict().items())


def test_save_and_restore_reject_inconsistent_initialization_metadata(tmp_path):
    branch=make_branch()
    path=tmp_path/'adapters.pt'
    optimizer=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=1e-3)
    with pytest.raises(ValueError,match='config and active representation initialization differ'):
        save_transfer_adapters(branch,path,config={'initialization':'pretrained'},
            foundation_sha256='same_asset',phase1_digest='same_phase1',
            optimizer=optimizer,optimizer_updates=0)
    assert not path.exists()
    save_fixture_checkpoint(branch,path)
    payload=torch.load(path,weights_only=False)
    payload['config']['initialization']='pretrained'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='config and representation initialization differ'):
        restore_transfer_adapters(branch,path,foundation_sha256='same_asset',phase1_digest='same_phase1')


@pytest.mark.parametrize('path',[
    'examples/CORDEX_ML/sa_regional_pretrained_transfer_v2.yaml',
    'examples/NARR_PRISM/narr_regional_pretrained_transfer_v2.yaml'])
def test_one_case_config_supports_all_controls_without_changing_source_or_data(path):
    from granitewxc.temporal.transfer import resolve_transfer_config
    original=resolve_transfer_config(path)
    assert original['initialization']=='pretrained'
    assert original['history_mode']=='observed'
    for initialization in ('pretrained','random_control'):
        for history_mode in ('observed','duplicate_current'):
            resolved=resolve_transfer_config(path,initialization=initialization,history_mode=history_mode)
            assert resolved['initialization']==initialization
            assert resolved['history_mode']==history_mode
            assert {k:v for k,v in resolved.items() if k not in ('initialization','history_mode')} == {
                k:v for k,v in original.items() if k not in ('initialization','history_mode')}
    assert resolve_transfer_config(path)==original


def test_omitted_overrides_preserve_yaml_control_values_and_save_effective_config(tmp_path):
    from granitewxc.temporal.transfer import resolve_transfer_config
    yaml_path=tmp_path/'case.yaml'
    yaml_path.write_text('initialization: random_control\nhistory_mode: duplicate_current\n'
        'atmospheric_indices: [0, 1]\ntoken_grid: [4, 4]\nlocal_grid: [2, 2]\n',encoding='utf-8')
    default_config=resolve_transfer_config(yaml_path)
    assert default_config['initialization']=='random_control'
    assert default_config['history_mode']=='duplicate_current'
    effective=resolve_transfer_config(yaml_path,history_mode='observed')
    branch=make_branch(initialization=effective['initialization'])
    optimizer=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=1e-3)
    path=tmp_path/'adapters.pt'
    save_transfer_adapters(branch,path,config=effective,foundation_sha256='same_asset',
        phase1_digest='same_phase1',optimizer=optimizer,optimizer_updates=0)
    payload=torch.load(path,weights_only=False)
    assert payload['config']==effective
    assert payload['config']['history_mode']=='observed'
    assert payload['active_representation']['initialization']=='random_control'


@pytest.mark.parametrize('options,initialization,history_mode',[
    ([],None,None),
    (['--initialization','random_control','--history-mode','duplicate_current'],
     'random_control','duplicate_current'),
])
def test_cli_forwards_explicit_controls_and_defers_omitted_controls_to_yaml(
        monkeypatch,options,initialization,history_mode):
    import granitewxc.temporal.transfer as transfer
    calls=[]
    monkeypatch.setattr(transfer,'run_update_audit',lambda *args,**kwargs: calls.append((args,kwargs)))
    transfer.main(['--config','case.yaml','--output','out',*options])
    assert calls==[(('case.yaml','out','cpu',1),
                   {'initialization':initialization,'history_mode':history_mode})]


@pytest.mark.parametrize('key,value',[
    ('initialization','unverified'),('history_mode','unknown'),
])
def test_invalid_control_fails_before_output_creation_or_model_loading(tmp_path,key,value):
    from granitewxc.temporal.transfer import run_update_audit
    yaml_path=tmp_path/'case.yaml'
    yaml_path.write_text(f'{key}: {value}\n',encoding='utf-8')
    output=tmp_path/'out'
    with pytest.raises(ValueError,match=f'{key} must be one of'):
        run_update_audit(yaml_path,output)
    assert not output.exists()
