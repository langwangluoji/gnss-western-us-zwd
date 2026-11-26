"""Visible-station masked autoencoder with coordinate/time queries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[1]
ARCHIVE = STUDY/'02_data/processed/mae_station_archive'

class StationArchive:
    def __init__(self):
        with h5py.File(ARCHIVE/'station_observations_2000_2021.h5','r') as handle:
            all_years = handle['year'][:]
            stop = int(np.flatnonzero(all_years==2011)[0])
            self.values = handle['values'][:stop]
            self.hours = handle['hour_id'][:stop]
            self.years = all_years[:stop]
            self.positions = np.c_[handle['xy_km'][:]/400,handle['height_m'][:]/3000].astype('f4')
            self.height_factor = np.exp(-handle['height_m'][:]/3000).astype('f4')
            self.sparse = np.flatnonzero(handle['sparse_input'][:])
        self.available = np.isfinite(self.values).all(axis=2)
        count = np.zeros(5)
        total = np.zeros(5)
        squared = np.zeros(5)
        for year in range(2000,2010):
            values = self.values[self.years==year]
            valid = np.isfinite(values)
            count += valid.sum((0,1))
            clean = np.where(valid,values,0).astype(float)
            total += clean.sum((0,1))
            squared += (clean**2).sum((0,1))
        self.mean = (total/count).astype('f4')
        self.std = np.maximum(np.sqrt(squared/count-(total/count)**2),1e-6).astype('f4')
        self.normalized = np.nan_to_num((self.values-self.mean)/self.std,nan=0)
        dates = pd.to_datetime(self.hours,unit='h')
        annual = 2*np.pi*(dates.dayofyear.to_numpy()-1+dates.hour.to_numpy()/24)/365.25
        daily = 2*np.pi*dates.hour.to_numpy()/24
        self.calendar = np.stack([np.sin(annual),np.cos(annual),np.sin(daily),np.cos(daily)],axis=1).astype('f4')
        candidates = np.arange(18,len(self.hours),12)
        self.pretrain_indices = candidates[(self.years[candidates]<=2009)&(self.years[candidates]==self.years[candidates-18])&(self.available[candidates].sum(1)>=12)]

    def inputs(self,indices,visible,device):
        indices = np.asarray(indices)
        visible = np.asarray(visible)
        safe = np.maximum(visible,0)
        times = indices[:,None]-np.array([0,6,12,18])[None,:]
        observed = self.normalized[times[:,:,None],safe[:,None,:]]
        valid = self.available[times[:,:,None],safe[:,None,:]]&(visible[:,None,:]>=0)
        positions = np.broadcast_to(self.positions[safe][:,None],observed.shape[:-1]+(3,))
        calendar = np.broadcast_to(self.calendar[times][:,:,None],observed.shape[:-1]+(4,))
        lag = np.broadcast_to(np.array([0,6,12,18],dtype='f4')[None,:,None,None]/18,observed.shape[:-1]+(1,))
        tokens = np.concatenate([observed,positions,calendar,lag],axis=-1)
        tokens[~valid] = 0
        return torch.from_numpy(tokens.reshape(len(indices),-1,13)).to(device),torch.from_numpy(~valid.reshape(len(indices),-1)).to(device)

    def queries(self,indices,positions,device):
        position_array = np.broadcast_to(positions[None],(len(indices),len(positions),3))
        calendar = np.broadcast_to(self.calendar[indices,None],(len(indices),len(positions),4))
        return torch.from_numpy(np.concatenate([position_array,calendar],axis=-1).astype('f4')).to(device)

    def masked_batch(self,indices,rng,device):
        visible = np.full((len(indices),39),-1,dtype=int)
        targets = self.available[indices].copy()
        for row,index in enumerate(indices):
            available = np.flatnonzero(self.available[index])
            if rng.random()<.5:
                selected = self.sparse[self.available[index,self.sparse]]
            else:
                selected = rng.choice(available,min(39,max(4,len(available)//2)),replace=False)
            if len(selected)<4 or len(selected)>=len(available):
                selected = rng.choice(available,min(39,max(4,len(available)//2)),replace=False)
            selected = np.sort(selected)
            visible[row,:len(selected)] = selected
            targets[row,selected] = False
        tokens,padding = self.inputs(indices,visible,device)
        queries = self.queries(indices,self.positions,device)
        truth = torch.from_numpy(self.normalized[indices]).to(device)
        return tokens,padding,queries,truth,torch.from_numpy(targets).to(device)

class StationMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(13,48)
        layer = nn.TransformerEncoderLayer(48,4,96,dropout=0,activation='gelu',batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query_embedding = nn.Linear(7,48)
        self.cross_attention = nn.MultiheadAttention(48,4,dropout=0,batch_first=True)
        self.query_norm = nn.LayerNorm(48)
        self.query_mlp = nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
        self.readout = nn.Linear(48,5)

    def representations(self,tokens,padding,queries):
        if padding.all(dim=1).any():
            raise ValueError('A station window must contain at least one observed token')
        clean = tokens.masked_fill(padding[:,:,None],0)
        encoded = self.encoder(self.embedding(clean),src_key_padding_mask=padding)
        query = self.query_embedding(queries)
        context,_ = self.cross_attention(query,encoded,encoded,key_padding_mask=padding,need_weights=False)
        context = self.query_norm(context+query)
        return context+self.query_mlp(context)

    def forward(self,tokens,padding,queries):
        return self.readout(self.representations(tokens,padding,queries))

def pretrain(archive,out,device):
    out.mkdir(parents=True,exist_ok=True)
    path = out/'mae_pretrained.pt'
    if path.exists():
        state = torch.load(path,map_location='cpu',weights_only=False)
        np.testing.assert_allclose(state['mean'],archive.mean)
        np.testing.assert_allclose(state['std'],archive.std)
        return state['model']
    torch.manual_seed(142)
    model = StationMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng = np.random.default_rng(142)
    factor = torch.from_numpy(archive.height_factor).to(device)[None]
    scale = float(archive.std[0])/20
    records = []
    for epoch in range(8):
        order = rng.permutation(archive.pretrain_indices)
        squared = 0.
        target_count = 0
        for offset in range(0,len(order),32):
            indices = order[offset:offset+32]
            tokens,padding,queries,truth,mask = archive.masked_batch(indices,rng,device)
            prediction = model(tokens,padding,queries)
            error = (prediction[:,:,0]-truth[:,:,0])*factor*scale
            loss = error.square()[mask].mean()+.1*(prediction[:,:,1:]-truth[:,:,1:]).square()[mask].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MAE loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            squared += float(error.detach().square()[mask].sum())*400
            target_count += int(mask.sum())
        record = dict(epoch=epoch+1,surface_reconstruction_rmse_mm=float(np.sqrt(squared/target_count)),masked_station_targets=target_count)
        records.append(record)
        print('MAE',json.dumps(record),flush=True)
    pd.DataFrame(records).to_csv(out/'mae_pretraining.csv',index=False)
    state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    torch.save(dict(model=state,mean=archive.mean,std=archive.std,pretrain_years=list(range(2000,2010)),epochs=8),path)
    return state

if __name__=='__main__':
    torch.set_num_threads(2)
    archive = StationArchive()
    pretrain(archive,STUDY/'04_models/mae_gan_cvae_v1','cuda' if torch.cuda.is_available() else 'cpu')
