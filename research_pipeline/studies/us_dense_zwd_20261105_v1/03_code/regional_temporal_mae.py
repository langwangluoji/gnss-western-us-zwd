"""Train a small temporal masked autoencoder entirely from regional GNSS data."""
from pathlib import Path
import json
import hashlib
import time
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from mae_gan_cvae import FusionGenerator, train_arm
from moment_superresolution import MomentFeatures, SpatialAdapter, inputs, validation_truth, evaluate
from train_us_gan import load

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY/'04_models/regional_temporal_mae_v1'
CACHE = STUDY/'02_data/processed/regional_mae_512h_6h_features.h5'
SOURCE = STUDY/'02_data/processed/mae_station_archive/station_observations_2000_2021.h5'


def file_sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


class TemporalMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(8,64)
        self.position = nn.Parameter(torch.randn(1,64,64)*.02)
        layer = nn.TransformerEncoderLayer(64,4,128,dropout=0,activation='gelu',batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(64,4,128,dropout=0,activation='gelu',batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,1)
        self.mask_token = nn.Parameter(torch.zeros(1,1,64))
        self.readout = nn.Linear(64,8)

    def normalized_patches(self,windows,visible):
        patches = windows.reshape(-1,64,8)
        clean = torch.where(visible[:,:,None],patches,0)
        count = visible.sum(1).clamp_min(1)*8
        mean = clean.sum((1,2))/count
        centered = torch.where(visible[:,:,None],patches-mean[:,None,None],0)
        scale = (centered.square().sum((1,2))/count).sqrt().clamp_min(1.)
        values = torch.nan_to_num((patches-mean[:,None,None])/scale[:,None,None])
        return values,mean,scale

    def masked_reconstruction(self,windows,selected):
        valid = torch.isfinite(windows.reshape(-1,64,8)).all(2)
        visible = torch.zeros_like(valid).scatter(1,selected,True)
        assert torch.all(visible <= valid)
        patches,mean,scale = self.normalized_patches(windows,visible)
        tokens = self.embedding(patches)+self.position
        memory = self.encoder(tokens.gather(1,selected[:,:,None].expand(-1,-1,64)))
        query = (self.mask_token+self.position).expand(len(windows),-1,-1)
        prediction = self.readout(self.decoder(query,memory))
        target_mask = valid & ~visible
        return prediction,patches,target_mask,scale

    def representations(self,windows):
        valid = torch.isfinite(windows.reshape(-1,64,8)).all(2)
        if not valid.any(1).all():
            raise ValueError('A history must contain at least one complete patch')
        patches,_,_ = self.normalized_patches(windows,valid)
        tokens = self.embedding(patches.masked_fill(~valid[:,:,None],0))+self.position
        features = self.encoder(tokens,src_key_padding_mask=~valid)
        return (features*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]


def pretrain(output_dir=OUT,last_year=2010):
    checkpoint = output_dir/'regional_mae.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['years']==list(range(2000,last_year+1)) and state['epochs']==20
        assert state['external_weights_used'] is False
        return state
    with h5py.File(SOURCE,'r') as handle:
        years = handle['year'][:]
        stop = int(np.flatnonzero(years==last_year+1)[0])
        values = handle['values'][:stop,:,0]
        stations = handle['station'][:]
    # Four-day anchor spacing limits redundant overlapping long windows.
    anchors = np.arange(511,len(values),96)
    ends = []
    station_indices = []
    for anchor in anchors:
        complete = np.isfinite(values[anchor-511:anchor+1].T.reshape(-1,64,8)).all(2)
        selected = np.flatnonzero(complete.sum(1)>=32)
        ends.extend([int(anchor)]*len(selected))
        station_indices.extend(selected.tolist())
    ends = np.array(ends)
    station_indices = np.array(station_indices)
    torch.manual_seed(142)
    rng = np.random.default_rng(142)
    model = TemporalMAE().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    records = []
    started = time.time()
    for epoch in range(20):
        model.train()
        order = rng.permutation(len(ends))
        sse = physical_sse = count = 0
        for offset in range(0,len(order),256):
            ids = order[offset:offset+256]
            windows = values[ends[ids,None]-np.arange(511,-1,-1)[None,:],station_indices[ids,None]]
            valid = np.isfinite(windows.reshape(-1,64,8)).all(2)
            scores = rng.random(valid.shape)
            scores[~valid] = 2
            selected = np.sort(np.argsort(scores,axis=1)[:,:16],axis=1)
            windows = torch.tensor(windows,device='cuda')
            prediction,target,mask,scale = model.masked_reconstruction(windows,torch.tensor(selected,device='cuda'))
            error = (prediction-target)[mask]
            loss = error.square().mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            sse += float(error.detach().square().sum())
            physical_sse += float(((prediction.detach()-target)*scale[:,None,None]).square()[mask].sum())
            count += error.numel()
        record = dict(epoch=epoch+1,normalized_rmse=np.sqrt(sse/count),h0_rmse_mm=np.sqrt(physical_sse/count),
                      targets=count,windows=len(ends),seconds=time.time()-started)
        records.append(record)
        pd.DataFrame(records).to_csv(output_dir/'mae_pretraining.csv',index=False)
        print('PRETRAIN',json.dumps(record),flush=True)
    state = dict(model={key:value.cpu() for key,value in model.state_dict().items()},
                 years=list(range(2000,last_year+1)),epochs=20,seed=142,external_weights_used=False,
                 reference_stations=[value.decode() for value in stations],history_hours=512,patch_hours=8,
                 visible_patches=16,training_windows=len(ends),parameters=sum(p.numel() for p in model.parameters()),
                 seconds=time.time()-started)
    torch.save(state,checkpoint)
    return state


def prepare_cache(state,output_dir=OUT,cache_path=CACHE,start_year=2008,end_year=2011):
    encoder_sha = file_sha256(output_dir/'regional_mae.pt')
    source_sha = file_sha256(SOURCE)
    if cache_path.exists():
        with h5py.File(cache_path,'r') as handle:
            assert handle.attrs['complete'] and handle.attrs['encoder']=='regional_temporal_mae_v1'
            assert handle.attrs['encoder_sha256']==encoder_sha
            assert handle.attrs['source_sha256']==source_sha
        return
    model = TemporalMAE().cuda().eval()
    model.load_state_dict(state['model'])
    model.requires_grad_(False)
    with h5py.File(SOURCE,'r') as handle:
        sparse = np.flatnonzero(handle['sparse_input'][:])
        years = handle['year'][:]
        hours = handle['hour_id'][:]
        start = int(np.flatnonzero(years==start_year)[0])
        later = np.flatnonzero(years==end_year+1)
        stop = int(later[0]) if len(later) else len(years)
        read_start=max(start-511,0)
        values = handle['values'][read_start:stop,:,0][:,sparse]
        positions = np.c_[handle['xy_km'][:][sparse]/400,handle['height_m'][:][sparse]/3000]
        stations = handle['station'][:][sparse]
    first_anchor=((max(start,511)+5)//6)*6
    anchors = np.arange(first_anchor,stop,6)
    temporary = cache_path.with_suffix('.partial.h5')
    started = time.time()
    with h5py.File(temporary,'w') as handle,torch.inference_mode():
        handle.create_dataset('hour_id',data=hours[anchors])
        handle.create_dataset('positions',data=positions.astype('f4'))
        handle.create_dataset('station',data=stations)
        features = handle.create_dataset('features',(len(anchors),39,64),dtype='f2',compression='lzf')
        available = handle.create_dataset('available',(len(anchors),39),dtype='?')
        for offset in range(0,len(anchors),16):
            times = anchors[offset:offset+16]-read_start-511
            windows = values[times[:,None]+np.arange(512)[None,:]].transpose(0,2,1).reshape(-1,512)
            valid = np.isfinite(windows.reshape(-1,64,8)).all(2).sum(1)>=32
            result = np.zeros((len(windows),64),dtype='f4')
            ids = np.flatnonzero(valid)
            result[ids] = model.representations(torch.tensor(windows[ids],device='cuda')).cpu().numpy()
            assert np.isfinite(result).all()
            features[offset:offset+len(times)] = result.reshape(-1,39,64)
            available[offset:offset+len(times)] = valid.reshape(-1,39)
        handle.attrs.update(complete=True,encoder='regional_temporal_mae_v1',seconds=time.time()-started,
                            encoder_sha256=encoder_sha,source_sha256=source_sha)
    temporary.replace(cache_path)


class RegionalGenerator(FusionGenerator):
    def __init__(self,use_latent=True,residual_latent=False):
        super().__init__(use_latent,residual_latent)
        self.mae = SpatialAdapter(feature_dim=64)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    state = pretrain()
    prepare_cache(state)
    encoder_sha = file_sha256(OUT/'regional_mae.pt')
    cache_sha = file_sha256(CACHE)
    provenance_path = OUT/'feature_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else {}
    provenance.update(encoder_sha256=encoder_sha,source_sha256=file_sha256(SOURCE),cache_sha256=cache_sha)
    provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    archive = MomentFeatures(CACHE)
    data,train,val,years,hours = load(train_years=(2008,2009,2010))
    for seed in [42,17,101]:
        torch.manual_seed(seed)
        initial = {key:value.clone() for key,value in RegionalGenerator().state_dict().items()}
        for name,gan,latent in [('MAE',False,False),('MAE_GAN',True,False),('MAE_GAN_CVAE',True,True)]:
            run = f'{name}_seed{seed}'
            path = OUT/f'{run}.pt'
            if path.exists():
                saved = torch.load(path,map_location='cpu',weights_only=False)
                assert saved['seed']==seed and saved['epochs']==20 and saved['encoder']=='regional_temporal_mae_v1'
                assert saved['GAN']==gan and saved['CVAE']==latent and saved['random_hour_subset']
                assert saved['encoder_sha256']==encoder_sha and saved['cache_sha256']==cache_sha
                assert saved['supervised_years']==[2008,2009,2010]
                torch.testing.assert_close(saved['mean'],data['mean'])
                torch.testing.assert_close(saved['std'],data['std'])
                continue
            train_arm(run,False,gan,latent,archive,data,train,hours,initial,None,'cuda',
                      model_factory=RegionalGenerator,input_builder=inputs,epochs=20,seed=seed,
                      output_dir=OUT,supervised_years=(2008,2009,2010),random_hour_subset=True)
            saved = torch.load(path,map_location='cpu',weights_only=False)
            saved.update(MAE=True,encoder='regional_temporal_mae_v1',external_weights_used=False,
                         encoder_frozen=True,pretrain_years=list(range(2000,2011)),history_hours=512,refresh_hours=6,
                         encoder_sha256=encoder_sha,cache_sha256=cache_sha)
            torch.save(saved,path)
    held,truth = validation_truth(hours,years)
    results = []
    for seed in [42,17,101]:
        for name,latent in [('MAE',False),('MAE_GAN',False),('MAE_GAN_CVAE',True)]:
            run = f'{name}_seed{seed}'
            saved = torch.load(OUT/f'{run}.pt',map_location='cpu',weights_only=False)
            model = RegionalGenerator(latent).cuda()
            model.load_state_dict(saved['model'])
            metrics = evaluate(model,archive,data,val,hours,held,truth,run,output_dir=OUT)
            metrics.update(model=name,seed=seed)
            results.append(metrics)
            pd.DataFrame(results).to_csv(OUT/'comparison_2011.csv',index=False)
            print('VALIDATION',json.dumps(metrics),flush=True)
    table = pd.DataFrame(results)
    assert len(table)==9 and table.station_records.nunique()==table.grid_points.nunique()==1
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',external_weights_used=False,
        pretrain_years=list(range(2000,2011)),train_years=[2008,2009,2010],validation_year=2011,
        downstream_seeds=[42,17,101],pretrain_seed=142,epochs=20,independent_test_read=False,
        parameters=state['parameters'],pretrain_seconds=state['seconds']),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
