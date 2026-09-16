"""Full-domain shape handling, including the last seven real CORDEX cells."""
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from granitewxc.refinement import build_refiner,resolve_refinement_config
from granitewxc.refinement.backbones import _resize_from_centers,SpatialResidualTransformer
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.spatial_lattice import endpoint_patch_embed,interpolate_position_table


@pytest.mark.parametrize('shape',[(128,128),(127,125),(129,130),(1,1),(1,17),(9,1),(7,9)])
@pytest.mark.parametrize('pattern',['constant','east_west','north_south','two_dimensional'])
def test_entire_valid_domain_constant_and_ramp_round_trip(shape,pattern):
    h,w=shape
    y=torch.linspace(-1,1,h)[:,None].expand(h,w)
    x=torch.linspace(-1,1,w)[None,:].expand(h,w)
    field={'constant':torch.ones_like(x),'east_west':x,'north_south':y,'two_dimensional':.3*x+.7*y}[pattern][None,None]
    conv=torch.nn.Conv2d(1,1,15,stride=8,padding=7,padding_mode='replicate',bias=False)
    with torch.no_grad():conv.weight.zero_();conv.weight[0,0,7,7]=1
    projected,ys,xs=endpoint_patch_embed(conv,field)
    assert ys[0]==0 and ys[-1]==h-1 and xs[0]==0 and xs[-1]==w-1
    assert (ys[1:]>ys[:-1]).all() and (xs[1:]>xs[:-1]).all()
    restored=_resize_from_centers(projected,ys,xs,(h,w))
    # No excluded perimeter rows or columns; tests isolate shape handling.
    torch.testing.assert_close(restored,field,atol=2e-6,rtol=2e-6)
    if shape==(128,128):assert projected.shape[-2:]==(17,17)


def test_original_coordinate_lattice_has_demonstrable_trailing_flat_band():
    x=torch.arange(128,dtype=torch.float32)
    token_values=x[::8][None,None,None,:].expand(1,1,16,16)
    restored=_resize_from_centers(token_values,torch.arange(16)*8,torch.arange(16)*8,(128,128))
    torch.testing.assert_close(restored[0,0,64,-8:],torch.full((8,),120.))
    assert abs(float(restored[0,0,64,-1])-127.)>6.9


def test_added_tokens_equal_dense_projection_at_real_centers_and_receive_gradients():
    field=torch.randn(2,3,17,19,requires_grad=True)
    projection=torch.nn.Conv2d(3,5,7,stride=4,padding=3,padding_mode='replicate')
    grid,ys,xs=endpoint_patch_embed(projection,field)
    dense=F.conv2d(F.pad(field,(3,3,3,3),mode='replicate'),projection.weight,projection.bias)
    expected=dense.index_select(-2,ys.long()).index_select(-1,xs.long())
    torch.testing.assert_close(grid,expected,atol=2e-6,rtol=2e-6)
    grid[..., -1,-1].sum().backward()
    assert field.grad[..., -1,-1].abs().sum()>0
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize('encoding',['learned_2d','sincos_2d'])
def test_regular_center_position_encoding_is_preserved(encoding):
    model=SpatialResidualTransformer(2,3,2,patch_size=(8,8),embedding_dim=24,num_heads=4,num_blocks=1,positional_encoding=encoding,spatial_alignment='endpoints')
    positions=torch.arange(16,dtype=torch.float32)*8
    old=model._positional(16,16,positions.device,torch.float32)
    new=model._positional_at_centers(positions,positions,torch.float32)
    torch.testing.assert_close(new,old,atol=0,rtol=0)


def test_learned_endpoint_position_uses_fractional_physical_coordinate():
    table=torch.arange(20,dtype=torch.float32)[:,None].expand(20,8).clone()
    actual=interpolate_position_table(table,torch.tensor([0.,15.,127/8]))
    torch.testing.assert_close(actual[:,0],torch.tensor([0.,15.,15.875]))
    with pytest.raises(ValueError,match='max_tokens'):
        interpolate_position_table(table,torch.tensor([20.]))


@pytest.mark.parametrize('head',['flow_matching_transformer','diffusion_transformer'])
def test_both_transformers_use_only_real_endpoint_tokens_and_return_original_shape(head):
    cfg=resolve_refinement_config({'refinement':{'type':head,'process_preconditioning_channels':[0],
        'transformer':{'embedding_dim':24,'num_heads':4,'num_blocks':1,'patch_size':[8,8],
                       'spatial_alignment':'endpoints','zero_init_output':False}}})
    model=build_refiner(cfg,residual_channels=2,cond_channels=3)
    seen=[]
    hook=model.net.blocks[0].attn.register_forward_pre_hook(lambda _,args:seen.append(args[0].shape[1]))
    state=torch.randn(1,2,128,128,requires_grad=True);cond=torch.randn(1,3,128,128)
    output=model.net(state,cond,torch.tensor([.5]))
    assert output.shape==state.shape and seen==[17*17]
    output.square().mean().backward();assert torch.isfinite(state.grad).all()
    hook.remove()
    previous=replace(cfg,transformer=replace(cfg.transformer,spatial_alignment='coordinates'))
    assert _build_scientific_contract(previous)!=_build_scientific_contract(cfg)
    restored=resolve_refinement_config({'refinement':cfg.to_dict()})
    assert restored.transformer.spatial_alignment=='endpoints'
