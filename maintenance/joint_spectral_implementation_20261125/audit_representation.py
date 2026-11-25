"""Isolate fixed-support encoding loss using real cached OOF residual labels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.spectral_operators import AnchorField
from tools.terrain_diffusion.joint_experiment import load_events
from tools.terrain_diffusion.data import sha,write_json


def farthest_points(xy,n):
    chosen=[int(np.linalg.norm(xy-xy.mean(0),axis=1).argmin())]
    d=np.full(len(xy),np.inf)
    for _ in range(n-1):
        d=np.minimum(d,np.linalg.norm(xy-xy[chosen[-1]],axis=1));chosen.append(int(d.argmax()))
    return xy[chosen]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    train=load_events(args.run/'oof','train');inner=load_events(args.run/'oof','inner')
    geometry=dict(np.load(ENGINE/'results/20261125_temporal_continuous_data/geometry.npz'))
    saved=dict(np.load(args.run/'representation/anchor.npz'))
    bias,scale=saved['bias'],saved['scale'];xy=train['query_xy']
    u=(train['residual']-bias)/scale;ui=(inner['residual']-bias)/scale
    # Physical groups, rather than parity of row index, define this masked check.
    group_ids=np.unique(geometry['groups']);hide=np.isin(geometry['groups'],group_ids[::2])
    masked=u.copy();masked[:,:,hide]=np.nan
    good=np.isfinite(masked).any(-1).all(-1);masked=masked[good]
    support={'regular256':saved['xy'],'farthest256':farthest_points(xy,256),'station410':xy}
    results=[]
    with threadpool_limits(limits=2):
        for name,anchors in support.items():
            field=AnchorField(anchors);H,out=field.readout(xy,True)
            for ridge,smooth in ((.01,.01),(.0001,.0001),(.000001,0.)):
                a=field.encode(u,H,ridge,smooth);av=field.encode(ui,H,ridge,smooth)
                masked_a=field.encode(masked,H,ridge,smooth)
                partial=(masked_a@H.T-u[good])*scale;partial[:,:,~hide]=np.nan
                metrics=dict(support=name,anchors=len(anchors),ridge=ridge,smooth=smooth,
                    train_reconstruction_mm=float(np.sqrt(np.nanmean(((a@H.T-u)*scale)**2))),
                    inner2019_reconstruction_mm=float(np.sqrt(np.nanmean(((av@H.T-ui)*scale)**2))),
                    spatial_group_masked_mm=float(np.sqrt(np.nanmean(partial**2))),
                    spatial_group_masked_count=int(np.isfinite(partial).sum()),outside_hull=int(out.sum()))
                results.append(metrics);print(json.dumps(metrics),flush=True)
    write_json(args.run/'representation_support_audit.json',dict(results=results,
        training='fixed saved OOF residuals; encoder diagnostic only, no network retraining',
        mask='same physical groups withheld from label reconstruction',
        label_scope='2016Jan8-12 training fragment /2019Jan8 internal fragment',
        source_anchor_sha256=sha(args.run/'representation/anchor.npz'),
        code_sha256=sha(__file__),selection='post-pilot diagnostic; cannot retrospectively replace preregistered A0/A1/A2 representation'))


if __name__=='__main__':main()
