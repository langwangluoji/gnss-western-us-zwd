"""Frozen MOMENT with matched deterministic, GAN and GAN-CVAE decoders."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from mae_gan_cvae import FusionGenerator, train_arm
from train_us_gan import load, batch
from us_gan import sample_points

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / '04_models/moment_superresolution_v1'
CACHE = STUDY / '02_data/processed/moment_512h_6h_features.h5'
REVISION = '411e288267f82cce86296dbe4d6c8bc533cc162f'


def prepare_cache():
    if CACHE.exists():
        with h5py.File(CACHE, 'r') as handle:
            assert handle.attrs['complete'] and handle.attrs['revision'] == REVISION
        return
    os.environ['HF_HOME'] = str(STUDY / '04_models/moment_small_candidate/hf_cache')
    sys.path.insert(0, str(STUDY / '07_operations/moment_vendor'))
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        str(STUDY / '04_models/moment_small_candidate/checkpoint'),
        model_kwargs={'task_name': 'embedding'}, local_files_only=True,
    )
    model.init()
    model.requires_grad_(False).eval().cuda()
    with h5py.File(STUDY / '02_data/processed/mae_station_archive/station_observations_2000_2021.h5', 'r') as handle:
        sparse = np.flatnonzero(handle['sparse_input'][:])
        all_hours = handle['hour_id'][:]
        years = handle['year'][:]
        start = np.flatnonzero(years == 2008)[0]
        stop = np.flatnonzero(years == 2012)[0]
        values = handle['values'][start-511:stop, :, 0][:, sparse]
        positions = np.c_[handle['xy_km'][:][sparse]/400, handle['height_m'][:][sparse]/3000]
        stations = handle['station'][:][sparse]
    anchors = np.arange(start, stop, 6)
    temporary = CACHE.with_suffix('.partial.h5')
    started = time.time()
    with h5py.File(temporary, 'w') as handle, torch.inference_mode():
        handle.create_dataset('hour_id', data=all_hours[anchors])
        handle.create_dataset('positions', data=positions.astype('f4'))
        handle.create_dataset('station', data=stations)
        features = handle.create_dataset('features', (len(anchors),39,512), dtype='f2', compression='lzf')
        available = handle.create_dataset('available', (len(anchors),39), dtype='?')
        for offset in range(0, len(anchors), 8):
            times = anchors[offset:offset+8]-start
            windows = values[times[:,None]+np.arange(512)[None,:]].transpose(0,2,1).reshape(-1,512)
            mask = np.isfinite(windows)
            valid = mask.reshape(-1,64,8).all(2).sum(1) >= 32
            result = np.zeros((len(windows),512), dtype='f4')
            selected = np.flatnonzero(valid)
            for begin in range(0,len(selected),64):
                ids = selected[begin:begin+64]
                x = torch.tensor(np.nan_to_num(windows[ids]),device='cuda')[:,None]
                observed = torch.tensor(mask[ids].astype('i8'),device='cuda')
                feature = model(x_enc=x,input_mask=observed).embeddings
                assert torch.isfinite(feature).all()
                result[ids] = feature.cpu().numpy()
            features[offset:offset+len(times)] = result.reshape(-1,39,512)
            available[offset:offset+len(times)] = valid.reshape(-1,39)
            if offset % 200 == 0:
                handle.flush()
                print('CACHE',offset,len(anchors),'seconds',round(time.time()-started,1),flush=True)
        handle.attrs.update(complete=True,revision=REVISION,history_hours=512,refresh_hours=6,
                            minimum_complete_patches=32,seconds=time.time()-started)
    temporary.replace(CACHE)


class MomentFeatures:
    def __init__(self,cache_path=CACHE):
        with h5py.File(cache_path,'r') as handle:
            assert handle.attrs['complete']
            self.hours = handle['hour_id'][:]
            self.features = handle['features'][:].astype('f4')
            self.available = handle['available'][:]
            self.positions = handle['positions'][:]
        # Coarse inputs retain absolute levels, the frozen model uses per-window normalization.
        self.mean = np.zeros(1,dtype='f4')
        self.std = np.ones(1,dtype='f4')


def inputs(archive,data,indices,hours,device):
    selected = np.searchsorted(archive.hours,hours[indices],side='right')-1
    assert selected.min() >= 0
    assert np.all(archive.hours[selected] <= hours[indices])
    assert np.all(hours[indices]-archive.hours[selected] < 6)
    positions = np.broadcast_to(archive.positions[None],(len(indices),39,3))
    features = np.concatenate([archive.features[selected],positions],axis=-1)
    padding = ~archive.available[selected]
    # A zero feature fallback is used when no station has enough history.
    features[padding] = 0
    empty = padding.all(1)
    padding[empty,0] = False
    raw = data['conditions'][indices]
    query = torch.stack([raw[:,7]*.75,raw[:,8]*.75,raw[:,6]/3000,
                         raw[:,9],raw[:,10],raw[:,11],raw[:,12]],dim=-1).reshape(-1,36,7)
    return torch.tensor(features,device=device),torch.tensor(padding,device=device),query.to(device)


class SpatialAdapter(nn.Module):
    def __init__(self,feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,48))
        self.position = nn.Linear(3,48)
        self.query = nn.Linear(7,48)
        self.attention = nn.MultiheadAttention(48,4,dropout=0,batch_first=True)
        self.norm = nn.LayerNorm(48)

    def representations(self,tokens,padding,queries):
        tokens = tokens.masked_fill(padding[:,:,None],0)
        encoded = self.embedding(tokens[:,:,:self.feature_dim])+self.position(tokens[:,:,self.feature_dim:])
        query = self.query(queries)
        result,_ = self.attention(query,encoded,encoded,key_padding_mask=padding,need_weights=False)
        return self.norm(query+result)


class MomentGenerator(FusionGenerator):
    def __init__(self,use_latent=True,residual_latent=False):
        super().__init__(use_latent,residual_latent)
        self.mae = SpatialAdapter()


def validation_truth(hours,years):
    roles = pd.read_csv(STUDY/'02_data/processed/reference_25km_v1/station_roles.csv')
    held = roles[roles.in_main_2008_2010 & roles.role.eq('validation') & roles.in_core].copy()
    truth = np.full((len(hours),len(held)),np.nan,dtype='f4')
    first = int(hours[np.flatnonzero(years==2011)[0]])
    for j,row in enumerate(held.itertuples()):
        frame = pd.read_parquet(STUDY/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',
                                columns=['hour_id','zwd_mm','joint_valid'])
        frame = frame[frame.joint_valid & frame.hour_id.between(first,int(hours[-1]))]
        truth[frame.hour_id.to_numpy()-hours[0],j] = frame.zwd_mm.to_numpy()
    return held,truth


def evaluate(model,archive,data,indices,hours,held,truth,name,output_dir=OUT):
    model.eval()
    xy = torch.tensor(held[['x_km','y_km']].to_numpy()/300,dtype=torch.float32,device='cuda')[None]
    sf = torch.tensor(np.exp(-held.orthometric_height_m.to_numpy()/3000),dtype=torch.float32,device='cuda')[None]
    factor = data['fine_height_factor'].float().cuda()[None,None]
    errors = []
    baseline_errors = []
    masks = []
    grid_sse = baseline_grid_sse = 0.
    count = 0
    with torch.inference_mode():
        for offset in range(0,len(indices),64):
            times = indices[offset:offset+64]
            x,base,mask = batch(data,times,'cuda')
            tokens,padding,queries = inputs(archive,data,times,hours,'cuda')
            result = model(x,base,tokens,padding,queries)
            prediction = result['field_mm']
            label = data['target_h0'][times].cuda()/20
            grid_sse += float(((prediction-label)*factor*20).square()[mask].sum())
            baseline_grid_sse += float(((result['baseline_mm']-label)*factor*20).square()[mask].sum())
            count += int(mask.sum())
            query = xy.expand(len(times),-1,-1)
            target = torch.tensor(truth[times],device='cuda')
            valid = torch.isfinite(target)&(sample_points(mask.float(),query)>=1-1e-6)
            errors.append((sample_points(prediction,query)*sf*20-target).cpu().numpy())
            baseline_errors.append((sample_points(result['baseline_mm'],query)*sf*20-target).cpu().numpy())
            masks.append(valid.cpu().numpy())
    error = np.concatenate(errors)
    baseline = np.concatenate(baseline_errors)
    valid = np.concatenate(masks)
    np.savez_compressed(output_dir/f'{name}_station_errors.npz',error=error,baseline=baseline,valid=valid,
                        hour_id=hours[indices],station=held.station.to_numpy(dtype='U'))
    station_rows = []
    for j,station in enumerate(held.station):
        e = error[valid[:,j],j]
        if len(e):
            station_rows.append(dict(station=station,n=len(e),rmse_mm=float(np.sqrt(np.mean(e**2))),
                                     mae_mm=float(np.mean(abs(e))),bias_mm=float(e.mean())))
    pd.DataFrame(station_rows).to_csv(output_dir/f'{name}_station_metrics.csv',index=False)
    e = error[valid]
    b = baseline[valid]
    return dict(arm=name,grid_rmse_mm=np.sqrt(grid_sse/count),grid_points=count,
                station_rmse_mm=float(np.sqrt(np.mean(e**2))),station_mae_mm=float(abs(e).mean()),
                station_bias_mm=float(e.mean()),station_p95_mm=float(np.quantile(abs(e),.95)),
                station_macro_rmse_mm=float(np.mean([r['rmse_mm'] for r in station_rows])),
                station_records=len(e),stations=len(station_rows),
                baseline_station_rmse_mm=float(np.sqrt(np.mean(b**2))),
                baseline_grid_rmse_mm=np.sqrt(baseline_grid_sse/count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    prepare_cache()
    if args.cache_only:
        return
    archive = MomentFeatures()
    data,train,val,years,hours = load(train_years=(2008,2009,2010))
    results = []
    # Fixed training schedules finish before validation observations are read.
    for seed in [42,17,101]:
        torch.manual_seed(seed)
        initial = {key:value.clone() for key,value in MomentGenerator().state_dict().items()}
        for name,gan,latent in [('MAE',False,False),('MAE_GAN',True,False),('MAE_GAN_CVAE',True,True)]:
            run = f'{name}_seed{seed}'
            if (OUT/f'{run}.pt').exists():
                saved = torch.load(OUT/f'{run}.pt',map_location='cpu',weights_only=False)
                assert saved['seed']==seed and saved['epochs']==20
                assert saved['GAN']==gan and saved['CVAE']==latent
                assert saved['supervised_years']==[2008,2009,2010]
                assert saved['pretrained_revision']==REVISION and saved['encoder_frozen']
                assert saved['history_hours']==512 and saved['refresh_hours']==6
                assert saved['random_hour_subset']
                torch.testing.assert_close(saved['mean'],data['mean'])
                torch.testing.assert_close(saved['std'],data['std'])
                continue
            train_arm(run,False,gan,latent,archive,data,train,hours,initial,None,'cuda',
                      model_factory=MomentGenerator,input_builder=inputs,epochs=20,seed=seed,
                      output_dir=OUT,supervised_years=(2008,2009,2010),random_hour_subset=True)
            saved = torch.load(OUT/f'{run}.pt',map_location='cpu',weights_only=False)
            saved.update(MAE=True,regional_mae_pretrained=False,pretrained_encoder='AutonLab/MOMENT-1-small',
                         pretrained_revision=REVISION,encoder_frozen=True,
                         external_pretraining_cutoff='unverified',history_hours=512,refresh_hours=6,
                         feature_cache=str(CACHE))
            torch.save(saved,OUT/f'{run}.pt')
    held,truth = validation_truth(hours,years)
    for seed in [42,17,101]:
        for name,latent in [('MAE',False),('MAE_GAN',False),('MAE_GAN_CVAE',True)]:
            run = f'{name}_seed{seed}'
            saved = torch.load(OUT/f'{run}.pt',map_location='cpu',weights_only=False)
            model = MomentGenerator(latent).cuda()
            model.load_state_dict(saved['model'])
            metrics = evaluate(model,archive,data,val,hours,held,truth,run)
            metrics.update(model=name,seed=seed)
            results.append(metrics)
            print('VALIDATION',json.dumps(metrics),flush=True)
            pd.DataFrame(results).to_csv(OUT/'comparison_2011.csv',index=False)
    table = pd.DataFrame(results)
    assert table.station_records.nunique()==table.grid_points.nunique()==1
    table.groupby('model').agg({key:['mean','std'] for key in ['station_rmse_mm','station_macro_rmse_mm',
                         'station_mae_mm','station_bias_mm','grid_rmse_mm']}).to_csv(OUT/'three_model_summary.csv')
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',seeds=[42,17,101],epochs=20,
        train_years=[2008,2009,2010],validation_year=2011,independent_test_read=False,
        pretrained_revision=REVISION,external_pretraining_cutoff='unverified',
        checkpoint_selection='fixed final epoch',encoder='frozen MOMENT, Z0 history only',
        history_hours=512,feature_refresh_hours=6),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
