"""Tiled residual model. Spatial pixel coordinates refer to cell centers."""
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU(),nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU())
    def forward(self,x): return self.net(x)

class ResidualUNet(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        if width%4: raise ValueError('width must be divisible by 4')
        self.enc=Block(4*channels,width); self.down=Block(width,width*2); self.bottleneck=Block(width*2,width*2)
        self.dec=Block(width*3,width); self.coarse=nn.Conv2d(width,1,1)
        nn.init.zeros_(self.coarse.weight); nn.init.zeros_(self.coarse.bias)
        self.fine=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,3,padding=1))
        nn.init.zeros_(self.fine[-1].weight); nn.init.zeros_(self.fine[-1].bias)
    def forward(self, conditions, baseline_mm, reference_mm=None):
        if conditions.ndim!=5 or conditions.shape[1]!=4 or conditions.shape[-2:]!=(32,32): raise ValueError('expected [B,4,C,32,32]')
        if baseline_mm.shape!=(conditions.shape[0],1,32,32): raise ValueError('baseline shape')
        x=conditions.flatten(1,2); e=self.enc(x); d=self.down(F.avg_pool2d(e,2)); d=self.bottleneck(d)
        h=self.dec(torch.cat((e,F.interpolate(d,size=e.shape[-2:],mode='bilinear',align_corners=False)),1))
        coarse=self.coarse(h); fine=self.fine(h)
        fine=fine-F.avg_pool2d(fine,4,4).repeat_interleave(4,-2).repeat_interleave(4,-1)
        bg=F.interpolate(baseline_mm,scale_factor=4,mode='bilinear',align_corners=False)
        reference_mm=torch.zeros_like(baseline_mm) if reference_mm is None else reference_mm
        reference=F.interpolate(reference_mm,scale_factor=4,mode='bilinear',align_corners=False)
        field=bg+reference+F.interpolate(coarse,scale_factor=4,mode='nearest')+fine
        return {'field_mm':field,'baseline_mm':bg,'coarse_mm':coarse,'fine_mm':fine,'reference_mm':reference}

class PairCritic(nn.Module):
    """Sparse actual-observation pairs: residual a,b and their separation metadata."""
    def __init__(self, metadata_channels=2, width=32):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2+metadata_channels,width),nn.LeakyReLU(.2),nn.Linear(width,width),nn.LeakyReLU(.2),nn.Linear(width,1))
    def forward(self, residual_pair, metadata): return self.net(torch.cat((residual_pair,metadata),-1)).squeeze(-1)

def sample_points(field, xy):
    """xy in normalized grid_sample coordinates, shape [B,N,2], x then y."""
    if not torch.isfinite(xy).all() or (xy.abs()>1).any(): raise ValueError('invalid point coordinates')
    return F.grid_sample(field,xy.unsqueeze(2),align_corners=False,mode='bilinear',padding_mode='border')[:,0,:,0]
