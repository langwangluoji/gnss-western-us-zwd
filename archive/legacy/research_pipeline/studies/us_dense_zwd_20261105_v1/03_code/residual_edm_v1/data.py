"""Frozen observation contracts, streamed data, and immutable mean caches."""
from pathlib import Path
import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

STUDY = Path(__file__).resolve().parents[2]
PROCESSED = STUDY/'02_data/processed'
PAIRED = PROCESSED/'expanded_2000_2021/paired_fields.h5'
SOURCE = PROCESSED/'mae_station_archive/station_observations_2000_2021.h5'
DEFAULT_ROOT = STUDY/'04_models/residual_edm_v1'
SCHEMA = 1


class MemoryReadCache:
    """Read-only input arrays; datasets not explicitly selected remain lazy."""
    def __init__(self, handle, keys):
        self.handle=handle
        self.arrays={key:handle[key][:] for key in keys}
        for value in self.arrays.values(): value.setflags(write=False)

    def __getitem__(self,key):
        return self.arrays[key] if key in self.arrays else self.handle[key]

    @property
    def attrs(self): return self.handle.attrs

    def close(self): self.handle.close()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def validate_manifest(manifest,indices=None):
    payload={key:value for key,value in manifest.items() if key!='fingerprint'}
    if canonical_hash(payload)!=manifest.get('fingerprint'): raise ValueError('Manifest fingerprint mismatch')
    for path,digest in manifest['sources'].items():
        if sha256(path)!=digest: raise ValueError(f'Data source changed: {path}')
    if indices is not None:
        digest=hashlib.sha256(b''.join(k.encode()+np.ascontiguousarray(indices[k]).tobytes() for k in sorted(indices.files))).hexdigest()
        if digest!=manifest['indices_sha256']: raise ValueError('Sample indices/domain changed')


def stage_root(stage, root=None):
    if stage not in ('stage1','stage2'): raise ValueError('stage must be stage1 or stage2')
    return Path(root) if root is not None else DEFAULT_ROOT/stage


def read_rows(dataset, indices):
    """h5py requires increasing unique indices; restore caller order afterward."""
    ids=np.asarray(indices,dtype=np.int64).reshape(-1)
    if len(ids)==0: return np.empty((0,*dataset.shape[1:]),dtype=dataset.dtype)
    unique,inverse=np.unique(ids,return_inverse=True)
    return dataset[unique][inverse]


def eligible_indices(years, hours, usable, anchor):
    ids=np.arange(19,len(years),dtype=np.int64)
    return ids[(years[ids]==years[ids-19]) & usable[ids] & (hours[ids-1]>=anchor)]


def split_eligible(years,hours,usable,anchor):
    # The historical usable mask depends on teacher labels; it is training-only.
    all_ids=eligible_indices(years,hours,np.ones(len(years),bool),anchor)
    training=all_ids[(years[all_ids]<=2018)&usable[all_ids]]
    return all_ids,training


def stratified_hours(hours, indices):
    ids=np.asarray(indices,dtype=np.int64)
    dates=pd.to_datetime(hours[ids],unit='h')
    result=[]
    for month in range(1,13):
        for hour in range(24):
            group=ids[(dates.month==month)&(dates.hour==hour)]
            if len(group):
                result.extend(group[np.unique([len(group)//3,2*len(group)//3])].tolist())
    return np.asarray(sorted(result),dtype=np.int64)


def station_features(values, mean, std, positions, calendar):
    available=np.isfinite(values).all(-1)
    normalized=(values-np.asarray(mean))/np.asarray(std)
    normalized[~available]=0
    features=np.concatenate([normalized,np.repeat(available[...,None],5,-1).astype('f4'),
        np.broadcast_to(positions[None],(*available.shape,3)),
        np.broadcast_to(calendar[:,None],(*available.shape,4))],axis=-1).astype('f4')
    features[~available]=0
    return features,~available


def residual_scale(square_sum,count):
    if count<=0: raise ValueError('No residual labels in training domain')
    scale=2*np.sqrt(square_sum/count)
    if not np.isfinite(scale) or scale<1e-6: raise ValueError('Invalid residual RMS; inspect labels and units')
    return float(scale)


def checkpoint_path(stage,root):
    if stage=='stage1':
        manifest=json.loads((STUDY/'04_models/component_ablation_v1/best_model.json').read_text(encoding='utf-8'))
        path=Path(manifest['checkpoint']['path'])
        if sha256(path)!=manifest['checkpoint']['sha256']: raise ValueError('Frozen checkpoint hash mismatch')
        return path
    path=root/'early_mean.pt'
    if not path.exists(): raise FileNotFoundError('stage2 requires early_mean.pt trained on 2000-2014')
    return path


def audit(stage='stage1',output_root=None):
    root=stage_root(stage,output_root); root.mkdir(parents=True,exist_ok=True)
    if stage=='stage2' and not (root/'early_mean.pt').exists():
        raise FileNotFoundError('Prepare early_mean.pt first; early_mean training indices are in stage1/indices.npz')
    with h5py.File(PROCESSED/'expanded_mae_cutoff2018_features.h5','r') as f: anchor=int(f['hour_id'][0])
    with h5py.File(PAIRED,'r') as f,h5py.File(SOURCE,'r') as source:
        years=f['year'][:]; hours=f['hour_id'][:]
        np.testing.assert_array_equal(hours,source['hour_id'][:])
        np.testing.assert_array_equal(f['station_sparse'][:],source['sparse_input'][:])
        np.testing.assert_allclose(f['station_xy'][:]*300,source['xy_km'][:],atol=1e-6)
        np.testing.assert_allclose(f['station_height_factor'][:],np.exp(-source['height_m'][:]/3000),rtol=1e-10)
        usable=np.load(PROCESSED/'expanded_2000_2021/usable_pair_hours.npy')
        all_ids,training=split_eligible(years,hours,usable,anchor)
        domain=np.zeros((24,24),bool)
        for start in range(0,len(training),2048): domain|=read_rows(f['target_mask'],training[start:start+2048])[:,0].any(0)
        complete=[]
        for start in range(0,len(training),2048):
            ix=training[start:start+2048]
            complete.extend(ix[read_rows(f['target_mask'],ix)[:,0][:,domain].all(1)].tolist())
        complete=np.asarray(complete,dtype=np.int64)
        expected={'eligible':len(training),'complete':len(complete),'domain':int(domain.sum())}
        if expected!={'eligible':118149,'complete':65369,'domain':226}: raise ValueError(f'Dataset contract drift: {expected}')
        train=complete if stage=='stage1' else complete[years[complete]>=2015]
        dev=all_ids[years[all_ids]==2019]; retro=all_ids[years[all_ids]>=2020]
        # Qualification uses only input context, not validation label availability.
        visible=[]
        for start in range(0,len(dev),1024):
            ix=dev[start:start+1024]
            visible.extend(ix[(read_rows(f['conditions'],ix)[:,5]>.5).any((1,2))].tolist())
        screen=stratified_hours(hours,np.asarray(visible,dtype=np.int64))
        dates=pd.to_datetime(hours[training],unit='h')
        selected=np.isin(training,train)
        coverage=[]
        # Wet regime audit only reads construction-station training labels.
        wet=np.zeros(len(training),bool)
        factor=f['station_height_factor'][:]
        for start in range(0,len(training),1024):
            z=read_rows(f['station_h0'],training[start:start+1024])*factor[None]
            wet[start:start+len(z)]=np.any(z>=100,axis=1)
        for year in sorted(np.unique(years[training]).tolist()):
            for month in range(1,13):
                mask=(dates.year==year)&(dates.month==month)
                coverage.append(dict(year=year,month=month,eligible=int(mask.sum()),selected=int((mask&selected).sum()),
                    eligible_wet=int((mask&wet).sum()),selected_wet=int((mask&wet&selected).sum())))
        names=[x.decode() for x in source['station'][:]]
        sparse=source['sparse_input'][:]
        arrays=dict(train=train,eligible=all_ids,dev=dev,screen=screen,retro=retro,domain=domain,
                    early_mean=training[years[training]<=2014],full_train=training)
    index_hash=hashlib.sha256(b''.join(k.encode()+np.ascontiguousarray(v).tobytes() for k,v in sorted(arrays.items()))).hexdigest()
    checkpoint=checkpoint_path(stage,root) if stage=='stage1' or (root/'early_mean.pt').exists() else None
    manifest=dict(schema=SCHEMA,stage=stage,time_basis='GPAC nominal product hours; UTC/GPST unspecified',
        sources={str(PAIRED):sha256(PAIRED),str(SOURCE):sha256(SOURCE),
                 str(PROCESSED/'expanded_2000_2021/usable_pair_hours.npy'):sha256(PROCESSED/'expanded_2000_2021/usable_pair_hours.npy')},
        checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        indices_sha256=index_hash,station_names=names,sparse_station_names=np.asarray(names)[sparse].tolist(),
        counts={k:int(len(v)) for k,v in arrays.items() if k!='domain'},domain_cells=int(domain.sum()),
        statistics_fit_years=[2000,2018] if stage=='stage1' else [2000,2014],
        residual_train_years=[2000,2018] if stage=='stage1' else [2015,2018],coverage=coverage,
        wet_rule='any construction station surface ZWD >=100 mm; training years only',
        mask_policy='all five station values finite; no fill; static D distinct from context and teacher masks')
    manifest['fingerprint']=canonical_hash(manifest)
    path=root/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=manifest: raise ValueError('Audit manifest changed: use a new output directory')
    else:
        np.savez_compressed(root/'indices.npz',**arrays)
        write_json(path,manifest)
    return manifest


class FrozenMean:
    def __init__(self,stage='stage1',root=None,device='cpu'):
        from architecture_search import SearchGenerator
        self.stage=stage; self.root=stage_root(stage,root); self.device=torch.device(device)
        self.path=checkpoint_path(stage,self.root); self.sha=sha256(self.path)
        saved=torch.load(self.path,map_location='cpu',weights_only=False)
        if stage=='stage2':
            if saved.get('train_years')!=[2000,2014]: raise ValueError('early mean provenance must state train_years=[2000,2014]')
        self.mean=saved['mean'].float().reshape(1,1,15,1,1)
        self.std=saved['std'].float().reshape(1,1,15,1,1)
        self.paired=h5py.File(PAIRED,'r'); self.source=h5py.File(SOURCE,'r')
        self.hours=self.paired['hour_id'][:]
        np.testing.assert_array_equal(self.hours,self.source['hour_id'][:])
        self.sparse=np.flatnonzero(self.source['sparse_input'][:])
        self.positions=np.c_[self.source['xy_km'][:][self.sparse]/400,self.source['height_m'][:][self.sparse]/3000].astype('f4')
        height=torch.as_tensor(-3000*np.log(self.paired['fine_height_factor'][:]),dtype=torch.float32)
        self.model=SearchGenerator('query_film',False,height).to(self.device)
        self.model.load_state_dict(saved['model'],strict=True)
        self.model.eval().requires_grad_(False)

    def inputs(self,indices):
        ids=np.asarray(indices,dtype=np.int64)
        raw=torch.as_tensor(read_rows(self.paired['conditions'],(ids[:,None]-np.array([0,6,12,18])).ravel()).reshape(-1,4,15,6,6),device=self.device)
        mean=self.mean.to(self.device); std=self.std.to(self.device)
        coarse=(raw-mean)/std
        coarse[:,:,:5]=torch.where(raw[:,:,5:6]>.5,coarse[:,:,:5],0)
        values=read_rows(self.source['values'],ids)[:,self.sparse]
        current=raw[:,0]
        calendar=current[:,9:13,0,0].cpu().numpy()
        stations,padding=station_features(values,self.mean.reshape(-1).numpy()[:5],self.std.reshape(-1).numpy()[:5],self.positions,calendar)
        tokens=np.zeros((len(ids),len(self.sparse),67),dtype='f4')
        tokens[...,:5]=stations[...,:5];tokens[...,5]=1;tokens[...,64:]=self.positions
        tokens[padding]=0
        legacy_padding=padding.copy(); legacy_padding[legacy_padding.all(1),0]=False
        queries=torch.stack([current[:,7]*.75,current[:,8]*.75,current[:,6]/3000,
            current[:,9],current[:,10],current[:,11],current[:,12]],dim=-1).reshape(-1,36,7)
        return dict(coarse=coarse,baseline=current[:,0:1]/20,raw=raw,stations=torch.as_tensor(stations,device=self.device),
            padding=torch.as_tensor(padding,device=self.device),tokens=torch.as_tensor(tokens,device=self.device),
            legacy_padding=torch.as_tensor(legacy_padding,device=self.device),queries=queries)

    @torch.inference_mode()
    def predict(self,batch):
        values=self.inputs(batch['indices']) if 'indices' in batch else batch
        return 20*self.model(values['coarse'],values['baseline'],values['tokens'],values['legacy_padding'],values['queries'])['field_mm']

    def close(self):
        self.paired.close(); self.source.close()


def cache_mean(stage='stage1',root=None,device='cuda'):
    root=stage_root(stage,root)
    if not (root/'manifest.json').exists(): audit(stage,root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    validate_manifest(manifest)
    adapter=FrozenMean(stage,root,device)
    if manifest['checkpoint_sha256']!=adapter.sha: raise ValueError('Manifest checkpoint mismatch; audit after preparing early mean')
    arrays=np.load(root/'indices.npz'); ids=arrays['eligible']; train=arrays['train']; domain=arrays['domain']
    contract=dict(schema=SCHEMA,manifest=manifest['fingerprint'],checkpoint=adapter.sha,
                  mean=adapter.mean.reshape(-1).tolist(),std=adapter.std.reshape(-1).tolist())
    destination=root/'mean_cache.h5'; metadata_path=root/'cache.json'
    if destination.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        with h5py.File(destination,'r') as f:
            if not f.attrs.get('complete') or json.loads(f.attrs['contract'])!=contract or metadata['contract']!=contract:
                raise ValueError('Mean cache contract mismatch')
        adapter.close(); return metadata
    temporary=destination.with_suffix('.partial.h5')
    if temporary.exists():
        raise FileExistsError(f'Unfinished cache exists; inspect/archive before rebuilding: {temporary}')
    square_sum=0.;count=0
    try:
        with h5py.File(temporary,'w') as cache:
            cache.attrs['contract']=json.dumps(contract,sort_keys=True)
            cache.create_dataset('hour_id',data=adapter.hours)
            cache.create_dataset('cached_indices',data=ids)
            dataset=cache.create_dataset('mu_z0_mm',(len(adapter.hours),1,24,24),dtype='f4',fillvalue=np.nan,chunks=(128,1,24,24),compression='lzf')
            for start in range(0,len(ids),128):
                ix=ids[start:start+128]
                prediction=adapter.predict({'indices':ix}).cpu().numpy()
                if not np.isfinite(prediction).all(): raise ValueError('Nonfinite frozen mean output')
                dataset[ix]=prediction
                selected=np.isin(ix,train)
                if selected.any():
                    labels=read_rows(adapter.paired['target_h0'],ix[selected]).astype('f8')
                    error=(labels-prediction[selected].astype('f8'))[:,0][:,domain]
                    square_sum+=float(np.square(error).sum());count+=error.size
                if start%8192==0: print(f'CACHE {stage} {start}/{len(ids)}',flush=True)
            scale=residual_scale(square_sum,count)
            cache.attrs.update(complete=True,residual_scale_mm=scale)
        metadata=dict(contract=contract,residual_scale_mm=scale,residual_square_sum=square_sum,
                      residual_count=count,cached_hours=len(ids),normalization='2*training residual RMS; no centering')
        temporary.replace(destination); write_json(metadata_path,metadata)
    finally: adapter.close()
    return metadata


class ExperimentData:
    def __init__(self,stage='stage1',root=None,device='cpu'):
        self.stage=stage;self.root=stage_root(stage,root);self.device=torch.device(device)
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.indices=np.load(self.root/'indices.npz')
        validate_manifest(self.manifest,self.indices)
        self.train_indices=self.indices['train'];self.dev_indices=self.indices['dev'];self.screen_indices=self.indices['screen']
        self.all_indices=self.indices['eligible']
        self.domain=torch.as_tensor(self.indices['domain'][None,None],device=self.device)
        self.adapter=FrozenMean(stage,self.root,device)
        self.hours=self.adapter.hours
        self.years=self.adapter.paired['year'][:]
        self.cache=h5py.File(self.root/'mean_cache.h5','r')
        contract=json.loads(self.cache.attrs['contract'])
        if not self.cache.attrs.get('complete') or contract['manifest']!=self.manifest['fingerprint'] or contract['checkpoint']!=self.adapter.sha:
            raise ValueError('Mean cache is stale or incomplete')
        self.scale=float(self.cache.attrs['residual_scale_mm'])
        self.mean=self.adapter.mean;self.std=self.adapter.std
        self._evaluations={}
        self._teacher_complete=None

    def preload_inputs(self):
        """Cache inputs only, preserving every dtype/value and lazy label access."""
        self.adapter.paired=MemoryReadCache(self.adapter.paired,['conditions','fine_height_factor','year'])
        self.adapter.source=MemoryReadCache(self.adapter.source,['values'])
        self.cache=MemoryReadCache(self.cache,['mu_z0_mm'])
        return sum(value.nbytes for cache in (self.adapter.paired,self.adapter.source,self.cache)
                   for value in cache.arrays.values())

    @property
    def teacher_complete(self):
        """Post-prediction diagnostic only; never used to qualify inference inputs."""
        if self._teacher_complete is None:
            domain=self.indices['domain']
            complete=np.empty(len(self.hours),dtype=bool)
            for start in range(0,len(complete),2048):
                mask=self.adapter.paired['target_mask'][start:start+2048,0]
                complete[start:start+len(mask)]=mask[:,domain].all(1)
            self._teacher_complete=complete
        return self._teacher_complete

    def batch(self,indices,targets=True):
        ids=np.asarray(indices,dtype=np.int64)
        values=self.adapter.inputs(ids);raw=values['raw'][:,0]
        mu=torch.as_tensor(read_rows(self.cache['mu_z0_mm'],ids),device=self.device)
        if not torch.isfinite(mu).all(): raise ValueError('Requested hour is not in frozen mean cache')
        support=F.interpolate(raw[:,5:6],size=(24,24),mode='bilinear',align_corners=False)>=1-1e-6
        baseline=F.interpolate(raw[:,0:1],size=(24,24),mode='bilinear',align_corners=False)
        maps=F.interpolate(torch.stack([raw[:,6]/3000,raw[:,7]*.75,raw[:,8]*.75,
                    raw[:,5],raw[:,13]/1000,raw[:,14]/39],dim=1),size=(24,24),mode='bilinear',align_corners=False)
        # Use actual fine terrain and cell coordinates, rather than interpolated coarse centres.
        maps[:,0]=torch.as_tensor(-np.log(self.adapter.paired['fine_height_factor'][:]),dtype=maps.dtype,device=self.device)
        axis=torch.arange(24,device=self.device,dtype=maps.dtype)*25-287.5
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');maps[:,1]=xx/400;maps[:,2]=yy/400;maps[:,3]=support[:,0]
        result=dict(coarse=values['coarse'],stations=values['stations'],padding=values['padding'],maps=maps,
            mu_condition=(mu-float(self.mean.flatten()[0]))/float(self.std.flatten()[0]),
            baseline_condition=(baseline-float(self.mean.flatten()[0]))/float(self.std.flatten()[0]),
            domain=self.domain.expand(len(ids),-1,-1,-1),mu_mm=mu,context_support=support,
            hour_id=torch.as_tensor(self.hours[ids],device=self.device),indices=ids)
        if targets:
            labels=torch.as_tensor(read_rows(self.adapter.paired['target_h0'],ids),device=self.device)
            mask=torch.as_tensor(read_rows(self.adapter.paired['target_mask'],ids),device=self.device)
            if not mask.masked_select(result['domain']).all(): raise ValueError('Training targets incomplete on D')
            result['residual']=torch.where(result['domain'],(labels-mu)/self.scale,0)
        return result

    def evaluation(self,role='validation',years=(2019,)):
        if role not in ('validation','independent_test'): raise ValueError('Unknown evaluation role')
        key=(role,tuple(years))
        if key not in self._evaluations:
            from expanded_architecture_search import truth_for_years
            held,truth=truth_for_years(self.hours,self.adapter.paired['year'][:],list(years),role)
            self._evaluations[key]=dict(truth=truth,xy=held[['x_km','y_km']].to_numpy(dtype='f4')/300,
                factor=np.exp(-held.orthometric_height_m.to_numpy(dtype='f4')/3000),names=held.station.tolist())
        return self._evaluations[key]

    def close(self):
        self.cache.close();self.adapter.close()
