"""Exact frozen geometry readout; target vertices can be pruned without changing inputs."""
import numpy as np
from retrain_common import ROOT,PROJECT,OPERATORS
from local_context_features import LocalFeatureBuilder,VertexQuery
from local_coverage_data import compact67
from local_innovation_components import aggregate_innovation
from gnss_field.artifacts import read_json,sha256,immutable_json
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.data.context import cycles


def ridge_field(features,row,scale):
    x=np.asarray(features,'f8');center=np.asarray(row['center']);std=np.asarray(row['std']);beta=np.asarray(row['beta'])
    if x.shape[-1]!=len(center) or len(beta)!=len(center)+1 or (std<=0).any() or not np.isfinite(x).all():raise ValueError('Invalid frozen ridge design')
    return scale*((x-center)/std@beta[:-1]+beta[-1])


class MomentOperator:
    def __init__(self,xy,height,axy,aheight):
        delta=np.asarray(axy)[None]-np.asarray(xy)[:,None];distance=np.linalg.norm(delta,axis=-1);dh=np.asarray(aheight)[None]-np.asarray(height)[:,None]
        self.kernels=[];self.matrices=[]
        for radius in (50000.,100000.,200000.):
            for vertical in (None,750.):
                k=np.exp(-.5*(distance/radius)**2)*(distance<=3*radius)
                if vertical:k*=np.exp(-.5*(dh/vertical)**2)
                k=k.astype('f4');self.kernels.append(k)
                self.matrices.append([k*m.astype('f4') for m in (np.ones_like(distance),delta[:,:,0]/radius,delta[:,:,1]/radius,dh/1000)])

    def __call__(self,residual,valid):
        valid=np.asarray(valid,'f4');residual=np.where(valid,np.asarray(residual,'f4'),0.);columns=[]
        for kernel,matrices in zip(self.kernels,self.matrices):
            den=np.maximum(valid@kernel.T,1e-30)
            columns.extend([(residual@matrix.T)/den for matrix in matrices])
        return np.stack(columns,-1).astype('f4')


class GeometryBase:
    def __init__(self,store,nodes=None):
        self.store=store;self.aids=store.ids('A');grid=store.grid;allnodes=np.arange(np.prod(store.shape))
        aq,_=safe_query_geometry(store,self.aids);self.qa=VertexQuery(store.shape,aq,np.zeros(len(aq)),allnodes)
        requested=allnodes if nodes is None else np.asarray(nodes)
        self.nodes=np.unique(np.r_[requested,self.qa.indices.numpy().ravel()]);self.alookup=np.searchsorted(self.nodes,self.qa.indices.numpy())
        xy=np.c_[grid['x'].ravel(),grid['y'].ravel()];height=store.stations.height.to_numpy()[self.aids]
        self.maker=LocalFeatureBuilder(xy,store.xy[self.aids],height,grid['height'],grid['region'],self.nodes,store.stats,aq,store.query_valid[self.aids])
        self.moments=MomentOperator(xy[self.nodes],grid['height'].ravel()[self.nodes],store.xy[self.aids],height)
        self.selection=read_json(OPERATORS/'selection.lock.json')
        self.settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')
        self.scale=self.settings['grid_scale'];self.weight=self.settings['baseline_weight']
        assert abs(self.scale-store.stats['z0_std'])<1e-8
        self.rows=[self.selection['coefficients'][str(f)]['directional'] for f in (0,1)]

    def predict(self,frames,m0,hour):
        f=compact67(self.maker,frames,m0,cycles([hour])[0]);base=np.asarray(m0).ravel()[self.nodes]+self.weight*(f[:,59]*self.scale-np.asarray(m0).ravel()[self.nodes])
        base_a=(base[self.alookup]*self.qa.weights.numpy()).sum(1)
        valid=np.isfinite(frames[0]).all(1)&self.store.query_valid[self.aids]
        residual=np.where(valid,frames[0,:,0]-base_a,0.)
        j=np.stack([aggregate_innovation(frames[0,:,0],base_a,k,valid) for k in self.maker.kernels],-1).astype('f4')
        features=np.concatenate([f,j/self.scale,self.moments(residual,valid)/self.scale],-1)
        fields=np.array([(base+ridge_field(features,row,self.scale)).astype('f4') for row in self.rows])
        if not np.isfinite(fields).all():raise ValueError('Nonfinite geometry base field')
        return fields,base


def lock_mean():
    from grid_pair_data import local_runtime
    r=local_runtime();immutable_json(ROOT/'mean.lock.json',dict(M0=sha256(r['m0']),selection=sha256(OPERATORS/'selection.lock.json'),
        mixture=sha256(r['parameters']),coefficients='both frozen B-only directional readouts; no C fold selection',
        readout_schema='67+2innovation+24moments',generator_condition_schema='67+2absoluteXY',code=sha256(__file__),D2020=False))
