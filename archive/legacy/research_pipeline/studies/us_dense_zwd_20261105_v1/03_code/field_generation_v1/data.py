"""Train-only statistics and immutable field/input geometry."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree,Delaunay,QhullError
from field_generation_v1.prepare import DATA,START,STOP
from residual_edm_v1.data import write_json,sha256,stratified_hours,canonical_hash

class FieldData:
    def __init__(self,device='cpu'):
        self.device=device
        self.roles=pd.read_csv(DATA/'station_roles.csv')
        self.g=np.load(DATA/'geometry.npz')
        self.data_contract=canonical_hash({p:sha256(DATA/p) for p in ['station_roles.csv','geometry.npz','audit.json']})
        self.shape=self.g['height'].shape
        self.xy=self.roles[['x','y']].to_numpy()
        self.qxy=(self.xy-self.g['origin'])/25000-.5
        self.height=self.roles.height.to_numpy()
        self.inputs=np.flatnonzero(self.roles.role=='input')
        self.train=np.flatnonzero(self.roles.role=='train')
        self.dev=np.flatnonzero(self.roles.role=='development')
        self.retro=np.flatnonzero(self.roles.role=='retrospective')
        self.hours=np.arange(START,STOP)
        self.dates=pd.to_datetime(self.hours,unit='h')
        self.train_stop=int(np.searchsorted(self.hours,int(np.datetime64('2019-01-01','h').astype('i8'))))
        cache=DATA/'observations.npy'
        cache_meta=DATA/'cache_contract.json'
        if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())['data_contract']!=self.data_contract:
            raise ValueError('Observation cache data contract changed; rebuild explicitly')
        if not cache.exists():
            values=np.lib.format.open_memmap(DATA/'observations.partial.npy',mode='w+',dtype='f4',shape=(STOP-START,len(self.roles),5))
            values[:]=np.nan
            for j,row in self.roles.iterrows():
                archive=np.load(DATA/'stations'/f'{row.station}.npz')
                values[archive['indices'],j]=archive['values']
                values[:,j,0]*=np.exp(self.height[j]/3000)
            values.flush();del values
            (DATA/'observations.partial.npy').replace(cache)
        if not cache_meta.exists():write_json(cache_meta,dict(data_contract=self.data_contract))
        self.values=np.load(cache,mmap_mode='r')
        self.input_values=np.array(self.values[:,self.inputs])
        self.available=np.isfinite(self.input_values).all(-1)
        stats_path=DATA/'statistics.json'
        if not stats_path.exists():
            v=self.input_values[:self.train_stop].astype('f8')
            target=np.asarray(self.values[:self.train_stop,self.train,0],dtype='f8')
            stats=dict(mean=np.nanmean(v,axis=(0,1)).tolist(),std=np.maximum(np.nanstd(v,axis=(0,1)),1e-6).tolist(),
                z0_std=float(np.nanstd(target)),surface_std=float(np.nanstd(target*np.exp(-self.height[self.train]/3000))),
                input_ids=self.inputs.tolist(),train_ids=self.train.tolist(),training_stop=self.train_stop,
                roles_sha256=sha256(DATA/'station_roles.csv'),geometry_sha256=sha256(DATA/'geometry.npz'))
            write_json(stats_path,stats)
        self.stats=json.loads(stats_path.read_text())
        if self.stats['roles_sha256']!=sha256(DATA/'station_roles.csv'): raise ValueError('Role manifest changed')
        if self.stats['geometry_sha256']!=sha256(DATA/'geometry.npz'): raise ValueError('Geometry changed')
        self.contract=canonical_hash(dict(data=self.data_contract,statistics=sha256(stats_path),
            config=sha256(DATA.parents[2]/'04_models/field_generation_v1/config.json'),
            code={p.name:sha256(p) for p in sorted(__import__('pathlib').Path(__file__).parent.glob('*.py'))}))
        self.mean=np.array(self.stats['mean'],dtype='f4');self.std=np.array(self.stats['std'],dtype='f4')
        self.origins=self.g['origins']
        # A global coarse lattice shares origins with every patch.
        cy,cx=np.meshgrid(np.arange(self.shape[0]//4),np.arange(self.shape[1]//4),indexing='ij')
        self.coarse_xy=np.c_[self.g['origin'][0]+(cx.ravel()+.5)*100000,self.g['origin'][1]+(cy.ravel()+.5)*100000]
        self.grid_xy=np.c_[self.g['x'].ravel(),self.g['y'].ravel()]
        self.label_near=cKDTree(self.xy[self.train]).query(self.grid_xy)[0].reshape(self.shape)<=100000
        self.static_support=self.g['region']&self.label_near
        self.query_label_near=cKDTree(self.xy[self.train]).query(self.xy)[0]<=100000
        self.patch_inputs=[]
        for y,x in self.origins:
            lower=self.g['origin']+np.array([x,y])*25000
            self.patch_inputs.append(np.flatnonzero(((self.xy[self.inputs]>=lower-150000)&(self.xy[self.inputs]<=lower+750000)).all(1)))
        self.qualified={}
        self._qualification=None

    @lru_cache(maxsize=128)
    def geometry_for(self,pattern_bytes,kind):
        valid=np.frombuffer(pattern_bytes,dtype='?')
        xy=self.xy[self.inputs[valid]]
        query={'grid':self.grid_xy,'coarse':self.coarse_xy,'stations':self.xy}[kind]
        if len(xy)<3:return np.zeros((len(query),4),int),np.zeros((len(query),4)),np.zeros(len(query),bool),np.full(len(query),1000000.),np.zeros(len(query))
        tree=cKDTree(xy);dist,idx=tree.query(query,k=min(4,len(xy)))
        count=np.array(tree.query_ball_point(query,150000,return_length=True))
        try:hull=Delaunay(xy).find_simplex(query)>=0
        except QhullError:hull=np.zeros(len(query),bool)
        support=(dist[:,0]<=100000)&(count>=3)&hull
        w=np.where(dist<=150000,1/np.maximum(dist,1)**2,0)
        w/=np.maximum(w.sum(1,keepdims=True),1e-30)
        return np.flatnonzero(valid)[idx],w,support,dist[:,0],count

    def support(self,t):
        _,_,mask,near,count=self.geometry_for(self.available[t].tobytes(),'grid')
        return mask.reshape(self.shape)&self.static_support,near.reshape(self.shape),count.reshape(self.shape)

    @lru_cache(maxsize=64)
    def coarse(self,t):
        ids,w,mask,near,count=self.geometry_for(self.available[t].tobytes(),'coarse')
        values=np.nan_to_num(self.input_values[t])
        raw=(values[ids]*w[...,None]).sum(1)
        raw[~mask]=0
        return raw.T.reshape(5,self.shape[0]//4,self.shape[1]//4),mask.reshape(self.shape[0]//4,self.shape[1]//4)

    def qualified_hours(self,station,year):
        key=(int(station),int(year))
        if key in self.qualified:return self.qualified[key]
        self.prepare_qualification()
        valid=self._qualification[:,station]&(self.dates.year==year)&np.isfinite(self.values[:,station,0])
        result=np.flatnonzero(valid);result=result[result>=18];self.qualified[key]=result
        return result

    def prepare_qualification(self):
        if self._qualification is not None:return
        path=DATA/'input_qualification.npy'
        if path.exists():self._qualification=np.load(path,mmap_mode='r');return
        patterns,inverse=np.unique(self.available,axis=0,return_inverse=True)
        result=np.zeros((len(patterns),len(self.roles)),bool)
        x,y=np.floor(self.qxy).astype(int).T
        for k,pattern in enumerate(patterns):
            _,_,mask,_,_=self.geometry_for(pattern.tobytes(),'grid');m=mask.reshape(self.shape)&self.static_support
            result[k]=m[y,x]&m[y+1,x]&m[y,x+1]&m[y+1,x+1]
            if k%100==0:print('QUALIFICATION',k,len(patterns),flush=True)
        self._qualification=result[inverse]
        temp=path.with_suffix('.partial.npy');np.save(temp,self._qualification);temp.replace(path)

    def patch_ids(self,station):
        x,y=np.floor(self.qxy[station]).astype(int)
        # Include the full one-cell halo for the five-point curvature stencil.
        return np.flatnonzero((self.origins[:,0]<=y+2)&(self.origins[:,0]+23>=y-1)&(self.origins[:,1]<=x+2)&(self.origins[:,1]+23>=x-1))

    def batch(self,t,patch_ids,mu=None):
        stations=[];padding=[];coarse=[];maps=[];baseline=[];positions=[]
        cyc=np.array([np.sin(2*np.pi*(self.dates[t].dayofyear-1+self.dates[t].hour/24)/365.25),np.cos(2*np.pi*(self.dates[t].dayofyear-1+self.dates[t].hour/24)/365.25),np.sin(2*np.pi*self.dates[t].hour/24),np.cos(2*np.pi*self.dates[t].hour/24)],dtype='f4')
        support,near,count=self.support(t)
        maxn=max(1,max(len(self.patch_inputs[p]) for p in patch_ids))
        for p in patch_ids:
            y,x=self.origins[p];center=self.g['origin']+np.array([x+12,y+12])*25000
            ids=self.patch_inputs[p];valid=self.available[t,ids]
            feat=np.zeros((maxn,17),dtype='f4');pad=np.ones(maxn,bool)
            pos=np.c_[(self.xy[self.inputs[ids]]-center)/400000,self.height[self.inputs[ids]]/3000]
            feat[:len(ids),:5]=np.nan_to_num((self.input_values[t,ids]-self.mean)/self.std)
            feat[:len(ids),5:10]=valid[:,None];feat[:len(ids),10:13]=pos;feat[:len(ids),13:]=cyc
            pad[:len(ids)]=~valid;feat[pad]=0
            stations.append(feat);padding.append(pad)
            terrain=self.g['height'][y:y+24,x:x+24]
            xx=(self.g['x'][y:y+24,x:x+24]-center[0])/400000
            yy=(self.g['y'][y:y+24,x:x+24]-center[1])/400000
            maps.append(np.stack([terrain/3000,xx,yy,support[y:y+24,x:x+24],np.minimum(near[y:y+24,x:x+24]/400000,2.5),count[y:y+24,x:x+24]/16]))
            frames=[]
            for offset in (0,6,12,18):
                values,ok=self.coarse(t-offset);raw=values[:,y//4:y//4+6,x//4:x//4+6];mask=ok[y//4:y//4+6,x//4:x//4+6]
                frame=np.zeros((15,6,6),dtype='f4');frame[:5]=(raw-self.mean[:,None,None])/self.std[:,None,None];frame[:5,~mask]=0
                frame[5]=mask;frame[6]=terrain.reshape(6,4,6,4).mean((1,3))/3000;frame[7]=xx.reshape(6,4,6,4).mean((1,3));frame[8]=yy.reshape(6,4,6,4).mean((1,3))
                date=self.dates[t-offset];annual=2*np.pi*(date.dayofyear-1+date.hour/24)/365.25;daily=2*np.pi*date.hour/24
                frame[9:13]=np.array([np.sin(annual),np.cos(annual),np.sin(daily),np.cos(daily)])[:,None,None]
                frame[13]=maps[-1][4].reshape(6,4,6,4).mean((1,3));frame[14]=maps[-1][5].reshape(6,4,6,4).mean((1,3));frames.append(frame)
                if offset==0:baseline.append(raw[0:1])
            coarse.append(np.stack(frames))
        tensor=lambda a:torch.as_tensor(np.asarray(a),device=self.device,dtype=torch.float32)
        result=dict(stations=tensor(stations),padding=torch.as_tensor(np.array(padding),device=self.device),coarse=tensor(coarse),maps=tensor(maps),baseline=tensor(baseline))
        result['baseline_condition']=torch.nn.functional.interpolate(result['baseline'],size=(24,24),mode='bilinear',align_corners=False)/self.stats['z0_std']
        result['mu_condition']=result['baseline_condition'] if mu is None else torch.stack([mu[:,y:y+24,x:x+24] for y,x in self.origins[patch_ids]])/self.stats['z0_std']
        result['domain']=torch.ones_like(result['baseline_condition'])
        return result
