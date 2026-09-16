"""Real-cell endpoint coverage for overlapping spatial token projections."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def endpoint_patch_embed(projection, features):
    """Keep regular tokens and append missing last-row/column physical centers.

    A stride-eight projection on 128 cells has centers 0,8,...,120. Adding
    center 127 makes coordinate interpolation cover the whole physical domain.
    The extra tokens use the same convolution and nonperiodic replication as
    regular tokens. Every attention token is centered on an actual valid grid
    location; no divisibility padding or exterior token is introduced.
    """
    if projection.padding_mode != 'replicate' or tuple(projection.dilation)!=(1,1):
        raise ValueError('Endpoint projection requires undilated replication padding')
    kh,kw=projection.kernel_size
    py,px=projection.padding
    if (kh,kw)!=(2*py+1,2*px+1):
        raise ValueError('Endpoint projection requires odd centered kernels')
    sy,sx=projection.stride
    h,w=features.shape[-2:]
    ys=torch.arange(0,h,sy,device=features.device,dtype=torch.float32)
    xs=torch.arange(0,w,sx,device=features.device,dtype=torch.float32)
    add_y=(h-1)%sy != 0
    add_x=(w-1)%sx != 0
    grid=projection(features)
    if not add_y and not add_x:
        return grid,ys,xs
    padded=F.pad(features,(px,px,py,py),mode='replicate')
    def convolve(strip,stride):
        return F.conv2d(strip,projection.weight,projection.bias,stride=stride,groups=projection.groups)
    if add_x:
        right=convolve(padded[..., :, -kw:],(sy,1))
        grid=torch.cat((grid,right),dim=-1)
        xs=torch.cat((xs,xs.new_tensor([w-1])))
    if add_y:
        bottom=convolve(padded[..., -kh:, :],(1,sx))
        if add_x:
            corner=convolve(padded[..., -kh:, -kw:],(1,1))
            bottom=torch.cat((bottom,corner),dim=-1)
        grid=torch.cat((grid,bottom),dim=-2)
        ys=torch.cat((ys,ys.new_tensor([h-1])))
    return grid,ys,xs


def interpolate_position_table(table, positions, *, validate=True):
    """Learned axis embeddings at actual (possibly fractional) token positions."""
    lower=positions.floor().long()
    upper=positions.ceil().long()
    if validate and (bool((lower<0).any()) or bool((upper>=len(table)).any())):
        raise ValueError('Endpoint token coordinates exceed the learned position table; increase max_tokens')
    fraction=(positions-lower).to(table.dtype).unsqueeze(-1)
    return table[lower]+fraction*(table[upper]-table[lower])
