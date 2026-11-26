"""US 100-to-25 km adapter for the European U-Net and pair critics."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

S=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(S.parent/'europe_dense_zwd_20261104_v1/03_code'))
from model.network import ResidualUNet, PairCritic, sample_points
from training.core import critic_loss

class USResidualUNet(ResidualUNet):
    def forward(self,conditions,baseline_mm,reference_mm=None):
        if conditions.ndim!=5 or conditions.shape[1]!=4 or conditions.shape[-2:]!=(6,6):
            raise ValueError('expected [B,4,C,6,6]')
        if baseline_mm.shape!=(conditions.shape[0],1,6,6): raise ValueError('baseline shape')
        e=self.enc(conditions.flatten(1,2))
        d=self.bottleneck(self.down(F.avg_pool2d(e,2)))
        hidden=self.dec(torch.cat((e,F.interpolate(d,size=e.shape[-2:],mode='bilinear',align_corners=False)),1))
        coarse=self.coarse(hidden); fine=self.fine(hidden)
        fine=fine-F.avg_pool2d(fine,4,4).repeat_interleave(4,-2).repeat_interleave(4,-1)
        baseline=F.interpolate(baseline_mm,scale_factor=4,mode='bilinear',align_corners=False)
        field=baseline+F.interpolate(coarse,scale_factor=4,mode='nearest')+fine
        return dict(field_mm=field,baseline_mm=baseline,coarse_mm=coarse,fine_mm=fine)

def valid_indices(years,allowed):
    # Each sample is (t,t-1), each with lags0,6,12,18; all must share a fold.
    mask=np.isin(years,allowed)
    result=[]
    for t in range(19,len(years)):
        if mask[t] and mask[t-19] and years[t]==years[t-19]: result.append(t)
    return np.asarray(result,dtype=int)
